"""
Hybrid Retrieval combining keyword search and semantic search.
Uses reciprocal rank fusion to combine results with comprehensive error handling.
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from threading import RLock
from typing import List, Dict, Set, Tuple, Any, Optional

from .vector_store import get_vector_store
from .qa_solver import _tokenize, _score_pramana, get_external_corpus_version
from .config import get_config
from .logging_setup import logger_retrieval
from .pramana_registry import ALL_PRAMANAS as _ALL_PRAMANAS
_HYBRID_CACHE: "OrderedDict[Tuple[str, Tuple[str, ...], int, str], Tuple[float, List[Dict[str, Any]]]]" = OrderedDict()
_HYBRID_CACHE_LOCK = RLock()

_KB_INDEX_LOCK = RLock()
_KB_INDEX_SIGNATURE: Optional[Tuple[int, int, str]] = None
_KB_INDEX: List[Dict[str, Any]] = []

# Query expansion: maps common English/Sanskrit terms to synonyms so paraphrased
# questions match the same knowledge-base chunks as canonical phrasing.
_QUERY_SYNONYMS: Dict[str, List[str]] = {
    "smoke": ["dhuma", "vapour", "vapor"],
    "fire": ["agni", "flames", "burning"],
    "perception": ["pratyaksha", "pratyaksa", "direct", "sense", "sensory"],
    "inference": ["anumana", "hetu", "vyapti", "reason", "inferential"],
    "testimony": ["shabda", "sabda", "verbal", "authority", "scriptural"],
    "comparison": ["upamana", "analogy", "similar", "similarity"],
    "postulation": ["arthapatti", "presumption", "assume", "presuppose"],
    "absence": ["anupalabdhi", "abhava", "non-perception", "non perception"],
    "knowledge": ["prama", "cognition", "valid", "epistemology"],
    "fallacy": ["hetvabhasa", "viruddha", "savyabhicara", "invalid reason"],
    "syllogism": ["pancavayava", "avayava", "five member"],
    "hill": ["parvata", "mountain"],
    "river": ["nadi", "stream", "flood"],
    "cow": ["dhenu", "gavaya"],
    "debate": ["vada", "jalpa", "vitanda", "discussion"],
    "doubt": ["samsaya", "uncertainty"],
    "error": ["viparyaya", "illusion", "mistake"],
}


def _normalize_question(question: str) -> str:
    return " ".join((question or "").strip().lower().split())


def _normalize_pramana_types(pramana_types: Optional[List[str]]) -> Tuple[str, ...]:
    if pramana_types is None:
        return _ALL_PRAMANAS

    normalized: List[str] = []
    seen: Set[str] = set()
    for item in pramana_types:
        if not isinstance(item, str):
            continue
        value = item.strip().lower()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)

    return tuple(normalized) if normalized else _ALL_PRAMANAS


def _cache_get(cache_key: Tuple[str, Tuple[str, ...], int, str]) -> Optional[List[Dict[str, Any]]]:
    config = get_config()
    ttl_seconds = max(1, int(config.retrieval.cache_ttl_seconds))

    with _HYBRID_CACHE_LOCK:
        payload = _HYBRID_CACHE.get(cache_key)
        if payload is None:
            return None
        created_at, cached = payload
        if (time.monotonic() - created_at) > ttl_seconds:
            del _HYBRID_CACHE[cache_key]
            return None
        _HYBRID_CACHE.move_to_end(cache_key)
        # Return copies to keep cached values immutable from callers.
        return [row.copy() for row in cached]


def _cache_set(cache_key: Tuple[str, Tuple[str, ...], int, str], value: List[Dict[str, Any]]) -> None:
    config = get_config()
    max_entries = max(8, int(config.retrieval.cache_max_entries))

    with _HYBRID_CACHE_LOCK:
        _HYBRID_CACHE[cache_key] = (time.monotonic(), [row.copy() for row in value])
        _HYBRID_CACHE.move_to_end(cache_key)
        if len(_HYBRID_CACHE) > max_entries:
            _HYBRID_CACHE.popitem(last=False)


def clear_hybrid_retrieval_cache() -> None:
    """Clear in-memory hybrid retrieval caches (useful for tests/benchmarks)."""
    global _KB_INDEX_SIGNATURE
    with _HYBRID_CACHE_LOCK:
        _HYBRID_CACHE.clear()
    with _KB_INDEX_LOCK:
        _KB_INDEX.clear()
        _KB_INDEX_SIGNATURE = None


def _get_knowledge_index() -> List[Dict[str, Any]]:
    from .qa_solver import _load_external_knowledge_base, KNOWLEDGE_BASE

    corpus_version = get_external_corpus_version()
    all_chunks = KNOWLEDGE_BASE + _load_external_knowledge_base()
    signature = (len(all_chunks), hash(tuple(chunk.get("id", "") for chunk in all_chunks)), corpus_version)

    global _KB_INDEX_SIGNATURE
    with _KB_INDEX_LOCK:
        if _KB_INDEX_SIGNATURE == signature and _KB_INDEX:
            return _KB_INDEX

        indexed: List[Dict[str, Any]] = []
        for chunk in all_chunks:
            text = chunk.get("text", "")
            indexed.append(
                {
                    "chunk": chunk,
                    "token_set": set(_tokenize(text)),
                    "tag_set": set(chunk.get("tags", [])),
                    "supports_set": set(chunk.get("supports", [])),
                }
            )

        _KB_INDEX.clear()
        _KB_INDEX.extend(indexed)
        _KB_INDEX_SIGNATURE = signature
        logger_retrieval.debug(f"Knowledge index rebuilt with {len(indexed)} chunks")
        return _KB_INDEX


def reciprocal_rank_fusion(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: float = 60.0,
    semantic_weight: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Combine keyword and semantic results using weighted reciprocal rank fusion.

    Weighted RRF formula:
        score = (1 - alpha) / (k + keyword_rank) + alpha / (k + semantic_rank)

    where alpha = semantic_weight (0.0 = keyword-only, 1.0 = semantic-only).

    Args:
        keyword_results: Results from keyword search
        semantic_results: Results from semantic search
        k: RRF constant (higher k dampens rank differences)
        semantic_weight: Weight for semantic results (0.0–1.0); keyword gets 1 - semantic_weight

    Returns:
        Fused and ranked results

    Raises:
        ValueError: If inputs are invalid
    """
    try:
        if not isinstance(keyword_results, list) or not isinstance(semantic_results, list):
            raise ValueError("Results must be lists")

        alpha = max(0.0, min(1.0, semantic_weight))
        kw_weight = 1.0 - alpha

        id_to_result: Dict[str, Dict[str, Any]] = {}
        id_to_fused: Dict[str, float] = {}
        id_to_sources: Dict[str, set] = {}

        # Process keyword results
        for rank, result in enumerate(keyword_results, 1):
            result_id = result.get("id")
            if not result_id:
                logger_retrieval.warning(f"Skipping keyword result without ID: {result}")
                continue
            if result_id not in id_to_result:
                id_to_result[result_id] = result.copy()
                id_to_fused[result_id] = 0.0
                id_to_sources[result_id] = set()
            id_to_fused[result_id] += kw_weight / (k + rank)
            id_to_sources[result_id].add("keyword")

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            result_id = result.get("id")
            if not result_id:
                logger_retrieval.warning(f"Skipping semantic result without ID: {result}")
                continue
            if result_id not in id_to_result:
                id_to_result[result_id] = result.copy()
                id_to_fused[result_id] = 0.0
                id_to_sources[result_id] = set()
            id_to_fused[result_id] += alpha / (k + rank)
            id_to_sources[result_id].add("semantic")

        # Calculate fused scores and rank
        fused_results = []
        for result_id, result in id_to_result.items():
            result["fused_score"] = round(id_to_fused[result_id], 4)
            srcs = id_to_sources[result_id]
            result["sources"] = "keyword+semantic" if len(srcs) > 1 else next(iter(srcs))
            fused_results.append(result)

        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        logger_retrieval.debug(
            "Weighted RRF fused %d unique results (k=%.0f, semantic_weight=%.2f)",
            len(fused_results), k, alpha,
        )
        return fused_results
    except Exception as e:
        logger_retrieval.error(f"✗ RRF fusion failed: {e}", exc_info=True)
        raise RuntimeError(f"RRF fusion failed: {e}")


def hybrid_search(
    question: str,
    pramana_types: Optional[List[str]] = None,
    k: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining keyword and semantic retrieval with error handling.
    
    Args:
        question: Query question
        pramana_types: Filter by pramana types (e.g., ['perception', 'inference'])
        k: Number of top results to return
        
    Returns:
        Ranked list of chunks with metadata
        
    Raises:
        ValueError: If question is empty or k is invalid
        RuntimeError: If search fails
    """
    try:
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be > 0")
        
        normalized_pramanas = _normalize_pramana_types(pramana_types)
        cache_key = (_normalize_question(question), normalized_pramanas, int(k), get_external_corpus_version())
        cached = _cache_get(cache_key)
        if cached is not None:
            logger_retrieval.debug("Hybrid search cache hit")
            return cached

        logger_retrieval.debug(f"Hybrid search: question='{question[:50]}...', k={k}, pramanas={normalized_pramanas}")

        # ===== SEMANTIC SEARCH (FAISS + E5) =====
        vector_store = get_vector_store()
        semantic_results = vector_store.search(question, k=k * 2)
        logger_retrieval.debug(f"Semantic results: {len(semantic_results)}")

        # ===== KEYWORD SEARCH WITH QUERY EXPANSION =====
        keyword_results = []
        question_tokens = _tokenize(question)
        # Expand tokens with Nyaya/Sanskrit synonyms so paraphrased questions
        # match the same chunks as their canonical equivalents.
        expanded: set = set(question_tokens)
        for tok in question_tokens:
            for syn in _QUERY_SYNONYMS.get(tok, []):
                expanded.update(_tokenize(syn))
        q_set = expanded
        pramana_set = set(normalized_pramanas)

        indexed_chunks = _get_knowledge_index()
        logger_retrieval.debug(f"Total indexed knowledge chunks: {len(indexed_chunks)}")

        for row in indexed_chunks:
            chunk = row["chunk"]
            # Skip if pramana doesn't match
            if not row["supports_set"].intersection(pramana_set):
                continue

            # Score based on token overlap
            overlap = len(q_set.intersection(row["token_set"]))

            if overlap > 0:
                # Boost for pramana match
                tag_boost = 1.0
                for tag in row["tag_set"]:
                    if tag in q_set:
                        tag_boost += 0.5
                
                score = overlap + tag_boost
                keyword_results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "tags": chunk.get("tags", []),
                    "supports": chunk.get("supports", []),
                    "keyword_score": round(score, 3),
                })

        # Sort keyword results by score
        keyword_results.sort(key=lambda x: x["keyword_score"], reverse=True)
        keyword_results = keyword_results[:k * 2]
        logger_retrieval.debug(f"Keyword results: {len(keyword_results)}")

        # ===== FUSION (Weighted Reciprocal Rank Fusion) =====
        cfg = get_config().retrieval
        fused = reciprocal_rank_fusion(
            keyword_results,
            semantic_results,
            k=cfg.reciprocal_rank_k,
            semantic_weight=cfg.semantic_weight,
        )
        logger_retrieval.debug(f"Fused results: {len(fused)}")

        # ===== FILTER BY PRAMANA =====
        # Keep results that either (a) explicitly support at least one requested
        # pramana, OR (b) have no 'supports' metadata at all — semantic-only results
        # from FAISS lack this field and must not be silently dropped.
        filtered_results = []
        for r in fused:
            supports = r.get("supports")
            if supports is None or not supports or set(supports).intersection(pramana_set):
                filtered_results.append(r)

        if not filtered_results and fused:
            logger_retrieval.warning(
                "Post-fusion pramana filter removed all %d results for pramanas=%s; "
                "returning top-%d fused results unfiltered.",
                len(fused), sorted(pramana_set), k,
            )
            filtered_results = fused

        final_results = filtered_results[:k]
        _cache_set(cache_key, final_results)
        logger_retrieval.info(f"✓ Hybrid search complete. Results: {len(final_results)}")
        
        return final_results
    except Exception as e:
        logger_retrieval.error(f"✗ Hybrid search failed: {e}", exc_info=True)
        raise RuntimeError(f"Hybrid search failed: {e}")


def filter_by_pramana(
    chunks: List[Dict[str, Any]],
    pramana_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Filter chunks by pramana support type with error handling.
    
    Args:
        chunks: List of chunks to filter
        pramana_types: Pramana types to include
        
    Returns:
        Filtered chunks
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list")
        if not isinstance(pramana_types, list):
            raise ValueError("Pramana types must be a list")
        
        filtered = [
            c for c in chunks
            if any(p in pramana_types for p in c.get("supports", []))
        ]
        
        logger_retrieval.debug(f"Filtered {len(chunks)} chunks to {len(filtered)} by pramana")
        return filtered
    except Exception as e:
        logger_retrieval.error(f"✗ Pramana filtering failed: {e}", exc_info=True)
        raise


def rerank_by_target_pramana(
    chunks: List[Dict[str, Any]],
    target_pramana: str
) -> List[Dict[str, Any]]:
    """
    Rerank chunks, boosting those that support the target pramana.
    
    Args:
        chunks: List of chunks with scores
        target_pramana: Pramana type to prioritize
        
    Returns:
        Reranked chunks
        
    Raises:
        ValueError: If inputs are invalid
    """
    try:
        if not isinstance(chunks, list):
            raise ValueError("Chunks must be a list")
        if not isinstance(target_pramana, str):
            raise ValueError("Target pramana must be a string")
        
        reranked = []
        for chunk in chunks:
            result = chunk.copy()
            if target_pramana in chunk.get("supports", []):
                result["fused_score"] = result.get("fused_score", 0) + 0.3
            reranked.append(result)

        reranked.sort(key=lambda x: x.get("fused_score", 0), reverse=True)
        logger_retrieval.debug(f"Reranked {len(reranked)} chunks by target pramana: {target_pramana}")
        return reranked
    except Exception as e:
        logger_retrieval.error(f"✗ Pramana reranking failed: {e}", exc_info=True)
        raise
