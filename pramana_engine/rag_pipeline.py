"""
RAG Pipeline Orchestrator.
Coordinates hybrid retrieval, LLM reasoning, and Pramana verification.
Includes comprehensive error handling and persistence management.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional
import json

from .hybrid_retrieval import hybrid_search, rerank_by_target_pramana
from .llm_integration import get_llm_engine
from .vector_store import get_vector_store, initialize_vector_store
from .qa_solver import _load_external_knowledge_base, KNOWLEDGE_BASE, _run_symbolic_verifier, get_external_corpus_version
from .config import get_config
from .logging_setup import logger_pipeline


_ANSWER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there", "they",
    "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with", "you",
}

_HETVABHASA_ALIASES: Dict[str, Tuple[str, ...]] = {
    "savyabhicara": ("savyabhicara", "anaikantika", "vyabhicari"),
    "viruddha": ("viruddha",),
    "satpratipaksha": ("satpratipaksha", "satpratipaksa"),
    "asiddha": ("asiddha", "svarupasiddha", "ashrayasiddha", "asrayasiddha"),
    "badhita": ("badhita", "kalatyayapadista", "kalatyayapadisht"),
}

_HETVABHASA_CORE_LABELS = tuple(_HETVABHASA_ALIASES.keys())

_NYAYA_DEBATE_FAULT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "chala": ("chala", "quibble", "equivocation"),
    "jati": ("jati", "futile rejoinder", "sophistical refutation"),
    "nigrahasthana": ("nigrahasthana", "point of defeat", "point of refutation"),
    "vitanda": ("vitanda", "cavil", "destructive debate"),
}

_NYAYA_DEBATE_FAULT_CORE_LABELS = tuple(_NYAYA_DEBATE_FAULT_ALIASES.keys())

_NYAYA_DEBATE_MODE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "vada": ("vada", "honest debate", "truth-seeking debate", "legitimate debate"),
    "jalpa": ("jalpa", "debate to win", "victory debate", "winning debate", "competitive debate"),
    "vitanda": ("vitanda", "cavil", "destructive refutation", "pure refutation"),
}

_NYAYA_DEBATE_MODE_CORE_LABELS = tuple(_NYAYA_DEBATE_MODE_ALIASES.keys())


class RAGPipeline:
    """Full RAG pipeline with production error handling and persistence."""

    def __init__(self):
        """Initialize RAG pipeline with error handling."""
        try:
            logger_pipeline.info("Initializing RAG pipeline...")
            self.llm_engine = get_llm_engine()
            self.vector_store = None
            self._initialized = False
            self._retrieval_cache: "OrderedDict[Tuple[str, Tuple[str, ...], int, str], Tuple[float, List[Dict[str, Any]]]]" = OrderedDict()
            config = get_config()
            configured_max = int(config.retrieval.cache_max_entries)
            configured_ttl = int(config.retrieval.cache_ttl_seconds)
            self._retrieval_cache_max = max(8, configured_max)
            self._retrieval_cache_ttl_seconds = max(1, configured_ttl)
            if configured_max < 8:
                logger_pipeline.warning(f"cache_max_entries={configured_max} is below minimum 8; using 8.")
            if configured_ttl < 1:
                logger_pipeline.warning(f"cache_ttl_seconds={configured_ttl} is below minimum 1; using 1.")
            logger_pipeline.info("✓ RAG pipeline created (not yet initialized)")
        except Exception as e:
            logger_pipeline.error(f"✗ Failed to create RAG pipeline: {e}", exc_info=True)
            raise RuntimeError(f"RAG pipeline creation failed: {e}")

    @staticmethod
    def _normalize_query_key(
        question: str,
        pramana_types: Optional[List[str]],
        k: int,
        corpus_version: str,
    ) -> Tuple[str, Tuple[str, ...], int, str]:
        normalized_question = " ".join((question or "").strip().lower().split())
        if pramana_types is None:
            normalized_pramanas = ("perception", "inference", "testimony", "comparison", "postulation")
        else:
            seen = set()
            values: List[str] = []
            for item in pramana_types:
                if not isinstance(item, str):
                    continue
                value = item.strip().lower()
                if value and value not in seen:
                    values.append(value)
                    seen.add(value)
            normalized_pramanas = tuple(values) if values else ("perception", "inference", "testimony", "comparison", "postulation")
        return normalized_question, normalized_pramanas, int(k), corpus_version

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        folded = unicodedata.normalize("NFKD", text)
        ascii_text = folded.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", ascii_text).strip().lower()

    @classmethod
    def _content_tokens(cls, text: str) -> set[str]:
        normalized = cls._normalize_text(text)
        tokens = re.findall(r"[a-z]{4,}", normalized)
        return {tok for tok in tokens if tok not in _ANSWER_STOPWORDS}

    @classmethod
    def _is_supported_by_evidence(cls, answer_text: str, chunks: List[Dict[str, Any]]) -> bool:
        answer_tokens = cls._content_tokens(answer_text)
        # Allow single-token answers like "Perception", "Yes", "Anumana"
        if len(answer_tokens) == 0:
            return False
        evidence_tokens: set[str] = set()
        for chunk in chunks[:3]:
            evidence_tokens.update(cls._content_tokens(chunk.get("text", "")))
        overlap = answer_tokens & evidence_tokens
        # Single-token answer: only needs 1 match; multi-token: needs 2
        required_overlap = 1 if len(answer_tokens) == 1 else 2
        return len(overlap) >= required_overlap

    @classmethod
    def _looks_like_fallacy_question(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        return (
            "fallacy" in normalized
            or "hetvabhasa" in normalized
            or any(alias in normalized for aliases in _HETVABHASA_ALIASES.values() for alias in aliases)
            or "bad inference" in normalized
            or "invalid inference" in normalized
            or "wrong reasoning" in normalized
            or "false reasoning" in normalized
        )

    @classmethod
    def _looks_like_debate_fault_question(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        if any(alias in normalized for aliases in _NYAYA_DEBATE_FAULT_ALIASES.values() for alias in aliases):
            return True

        # Check if this is actually a debate-mode question first (including clues)
        if cls._looks_like_debate_mode_question(question):
            return False

        broad_markers = ("debate", "defect", "point of defeat", "refutation", "quibble")
        has_broad_marker = any(marker in normalized for marker in broad_markers)
        return "nyaya" in normalized and has_broad_marker

    @classmethod
    def _looks_like_debate_mode_question(cls, question: str) -> bool:
        normalized = cls._normalize_text(question)
        # Check for explicit debate-mode aliases
        if any(alias in normalized for aliases in _NYAYA_DEBATE_MODE_ALIASES.values() for alias in aliases):
            return True
        
        # Check for debate-mode clue markers
        mode_clues = {
            "truth-seeking", "legitimate debate", "honest debate", "seeking truth",
            "to win", "victory", "winning", "competitive", "argument to win",
            "refutation only", "pure refutation", "only attacks", "no thesis"
        }
        if any(clue in normalized for clue in mode_clues):
            return True
        
        return False

    @classmethod
    def _select_debate_fault_label(cls, chunks: List[Dict[str, Any]]) -> Optional[str]:
        scores: Dict[str, float] = {label: 0.0 for label in _NYAYA_DEBATE_FAULT_ALIASES}
        for rank, chunk in enumerate(chunks[:5]):
            text = cls._normalize_text(chunk.get("text", ""))
            if not text:
                continue
            rank_weight = 1.0 / (rank + 1)
            for label, aliases in _NYAYA_DEBATE_FAULT_ALIASES.items():
                for alias in aliases:
                    if alias in text:
                        scores[label] += rank_weight
        top_label, top_score = max(scores.items(), key=lambda item: item[1])
        return top_label if top_score > 0 else None

    @classmethod
    def _select_debate_mode_label(cls, chunks: List[Dict[str, Any]]) -> Optional[str]:
        scores: Dict[str, float] = {label: 0.0 for label in _NYAYA_DEBATE_MODE_ALIASES}
        for rank, chunk in enumerate(chunks[:5]):
            text = cls._normalize_text(chunk.get("text", ""))
            if not text:
                continue
            rank_weight = 1.0 / (rank + 1)
            for label, aliases in _NYAYA_DEBATE_MODE_ALIASES.items():
                for alias in aliases:
                    if alias in text:
                        scores[label] += rank_weight
        top_label, top_score = max(scores.items(), key=lambda item: item[1])
        return top_label if top_score > 0 else None

    @classmethod
    def _select_hetvabhasa_label(cls, chunks: List[Dict[str, Any]]) -> Optional[str]:
        scores: Dict[str, float] = {label: 0.0 for label in _HETVABHASA_ALIASES}
        for rank, chunk in enumerate(chunks[:5]):
            text = cls._normalize_text(chunk.get("text", ""))
            if not text:
                continue
            rank_weight = 1.0 / (rank + 1)
            for label, aliases in _HETVABHASA_ALIASES.items():
                for alias in aliases:
                    if alias in text:
                        scores[label] += rank_weight

        top_label, top_score = max(scores.items(), key=lambda item: item[1])
        return top_label if top_score > 0 else None

    @staticmethod
    def _format_fallacy_label(label: str) -> str:
        display = {
            "savyabhicara": "Savyabhicara",
            "viruddha": "Viruddha",
            "satpratipaksha": "Satpratipaksha",
            "asiddha": "Asiddha",
            "badhita": "Badhita",
        }
        return display.get(label, label.title())

    @classmethod
    def _heuristic_answer_from_chunks(
        cls,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Optional[str]:
        normalized_question = cls._normalize_text(question)
        is_fallacy_question = cls._looks_like_fallacy_question(question)
        is_debate_fault_question = cls._looks_like_debate_fault_question(question)
        is_debate_mode_question = cls._looks_like_debate_mode_question(question)

        if not is_fallacy_question and not is_debate_fault_question and not is_debate_mode_question:
            return None

        if is_debate_mode_question and not is_fallacy_question and not is_debate_fault_question:
            mode_label = cls._select_debate_mode_label(retrieved_chunks)
            cue_label = cls._infer_debate_mode_from_question_clues(normalized_question)
            final_label = mode_label or cue_label
            if final_label:
                return (
                    f"The Nyaya debate type here is {cls._format_debate_mode_label(final_label)}, "
                    f"characterized by its approach to argumentation and refutation."
                )
            return (
                "Nyaya debate modes include Vada (truth-seeking), Jalpa (winning-oriented), and Vitanda "
                "(refutation-only). Please clarify the debate context for precise classification."
            )

        if is_debate_fault_question and not is_fallacy_question:
            debate_label = cls._select_debate_fault_label(retrieved_chunks)
            cue_label = cls._infer_debate_fault_from_question_clues(normalized_question)
            final_label = debate_label or cue_label
            if final_label:
                return (
                    f"A likely Nyaya debate fault is {cls._format_debate_fault_label(final_label)} based on the available evidence."
                )
            return (
                "Nyaya debate faults here are likely in the chala-jati-nigrahasthana-vitanda family; "
                "please share the exact argument exchange for precise labeling."
            )

        asks_catalog = cls._asks_for_fallacy_catalog(normalized_question)
        has_inference_statement = cls._has_explicit_inference_statement(normalized_question)

        label = cls._select_hetvabhasa_label(retrieved_chunks)
        cue_label = cls._infer_fallacy_from_question_clues(normalized_question)
        if asks_catalog:
            return (
                "Nyaya classifies Hetvabhasa into five major types: Savyabhicara, Viruddha, "
                "Satpratipaksha, Asiddha, and Badhita."
            )

        if label and has_inference_statement:
            return (
                f"Based on the retrieved Nyaya references, the relevant Hetvabhasa is "
                f"{cls._format_fallacy_label(label)}. This is a likely classification because the supporting "
                f"evidence repeatedly associates similar inferential patterns with {cls._format_fallacy_label(label)}."
            )

        if cue_label and has_inference_statement:
            return (
                f"A likely Hetvabhasa is {cls._format_fallacy_label(cue_label)} based on the inference wording. "
                f"The phrasing suggests the hallmark pattern of {cls._format_fallacy_label(cue_label)} in Nyaya analysis."
            )

        if label:
            return (
                f"A likely Hetvabhasa is {cls._format_fallacy_label(label)}, but a precise classification requires the "
                "exact inference (paksha-hetu-sadhya form)."
            )

        return (
            "Nyaya classifies Hetvabhasa into five major types: Savyabhicara, Viruddha, "
            "Satpratipaksha, Asiddha, and Badhita. Please share the exact inference to identify the specific one."
        )

    @classmethod
    def _synthesize_answer_from_chunks(
        cls,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """Build a readable answer by combining the top retrieved chunks."""
        if not chunks:
            return "No relevant information found in the knowledge base."
        parts = []
        for chunk in chunks[:3]:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            sentences = re.split(r"(?<=[.!?])\s+", text)
            snippet = " ".join(sentences[:3]).strip()
            if snippet:
                parts.append(snippet)
        if parts:
            return " ".join(parts)
        return chunks[0].get("text", "No relevant information found.")

    @staticmethod
    def _infer_fallacy_from_question_clues(normalized_question: str) -> Optional[str]:
        clues = {
            "savyabhicara": (
                "inconsistent", "inconsistency", "deviates", "deviation", "variable", "both presence",
                "both absent", "anaikantika", "irregular",
            ),
            "viruddha": (
                "contradictory", "opposite", "negates", "opposed", "contrary",
            ),
            "satpratipaksha": (
                "counter reason", "opposing reason", "equally strong opposite", "counter inference",
            ),
            "asiddha": (
                "unestablished", "unproved", "no locus", "no subject", "not established",
            ),
            "badhita": (
                "contradicted by perception", "contradicted by fact", "defeated by perception", "badhita",
            ),
        }
        for label, markers in clues.items():
            if any(marker in normalized_question for marker in markers):
                return label
        return None

    @staticmethod
    def _infer_debate_fault_from_question_clues(normalized_question: str) -> Optional[str]:
        clues = {
            "chala": (
                "equivocation", "twisting words", "intentional misinterpretation", "quibble",
                "twists the opponent", "twist the opponent", "twists words", "twist words",
            ),
            "jati": ("futile rejoinder", "specious objection", "sophistical refutation"),
            "nigrahasthana": ("point of defeat", "self-contradiction", "conceded and denied"),
            "vitanda": ("only attacks opponent", "no own thesis", "pure refutation"),
        }
        for label, markers in clues.items():
            if any(marker in normalized_question for marker in markers):
                return label
        return None

    @staticmethod
    def _infer_debate_mode_from_question_clues(normalized_question: str) -> Optional[str]:
        clues = {
            "vada": ("truth-seeking", "legitimate debate", "honest debate", "seeking truth"),
            "jalpa": ("to win", "victory", "winning", "competitive", "argument to win"),
            "vitanda": ("refutation only", "pure refutation", "only attacks", "no thesis"),
        }
        for label, markers in clues.items():
            if any(marker in normalized_question for marker in markers):
                return label
        return None

    @staticmethod
    def _format_debate_fault_label(label: str) -> str:
        display = {
            "chala": "Chala",
            "jati": "Jati",
            "nigrahasthana": "Nigrahasthana",
            "vitanda": "Vitanda",
        }
        return display.get(label, label.title())

    @staticmethod
    def _format_debate_mode_label(label: str) -> str:
        display = {
            "vada": "Vada",
            "jalpa": "Jalpa",
            "vitanda": "Vitanda",
        }
        return display.get(label, label.title())

    @staticmethod
    def _asks_for_fallacy_catalog(normalized_question: str) -> bool:
        catalog_markers = ("category", "categories", "types", "kinds", "list", "name")
        asks_about_five = "five" in normalized_question or "5" in normalized_question
        return asks_about_five and any(marker in normalized_question for marker in catalog_markers)

    @staticmethod
    def _has_explicit_inference_statement(normalized_question: str) -> bool:
        inference_markers = (
            " because ", " therefore ", " hence ", " thus ", " so ", " infer ", " inference ",
            " hetu ", " sadhya ", " paksha ", " vyapti ",
        )
        return any(marker in f" {normalized_question} " for marker in inference_markers)

    @classmethod
    def _llm_answer_has_fallacy_labels(cls, answer_text: str) -> bool:
        normalized = cls._normalize_text(answer_text)
        label_hits = 0
        for label in _HETVABHASA_CORE_LABELS:
            aliases = _HETVABHASA_ALIASES.get(label, ())
            if any(alias in normalized for alias in aliases):
                label_hits += 1
        return label_hits >= 2

    @classmethod
    def _llm_answer_has_debate_fault_labels(cls, answer_text: str) -> bool:
        normalized = cls._normalize_text(answer_text)
        label_hits = 0
        for label in _NYAYA_DEBATE_FAULT_CORE_LABELS:
            aliases = _NYAYA_DEBATE_FAULT_ALIASES.get(label, ())
            if any(alias in normalized for alias in aliases):
                label_hits += 1
        return label_hits >= 1

    @classmethod
    def _llm_answer_has_debate_mode_labels(cls, answer_text: str) -> bool:
        normalized = cls._normalize_text(answer_text)
        label_hits = 0
        for label in _NYAYA_DEBATE_MODE_CORE_LABELS:
            aliases = _NYAYA_DEBATE_MODE_ALIASES.get(label, ())
            if any(alias in normalized for alias in aliases):
                label_hits += 1
        return label_hits >= 1

    @classmethod
    def _llm_mentions_expected_debate_fault(cls, answer_text: str, expected_label: str) -> bool:
        normalized = cls._normalize_text(answer_text)
        aliases = _NYAYA_DEBATE_FAULT_ALIASES.get(expected_label, ())
        for alias in aliases:
            if " " in alias:
                if alias in normalized:
                    return True
            else:
                if re.search(rf"\b{re.escape(alias)}\b", normalized):
                    return True
        return False

    @classmethod
    def _llm_mentions_expected_debate_mode(cls, answer_text: str, expected_label: str) -> bool:
        normalized = cls._normalize_text(answer_text)
        aliases = _NYAYA_DEBATE_MODE_ALIASES.get(expected_label, ())
        for alias in aliases:
            if " " in alias:
                if alias in normalized:
                    return True
            else:
                if re.search(rf"\b{re.escape(alias)}\b", normalized):
                    return True
        return False

    def _get_cached_retrieval(
        self,
        question: str,
        pramana_types: Optional[List[str]],
        k: int,
    ) -> List[Dict[str, Any]]:
        corpus_version = get_external_corpus_version()
        key = self._normalize_query_key(question, pramana_types, k, corpus_version)
        payload = self._retrieval_cache.get(key)
        if payload is not None:
            created_at, cached = payload
            if (time.monotonic() - created_at) <= self._retrieval_cache_ttl_seconds:
                self._retrieval_cache.move_to_end(key)
                logger_pipeline.debug("Retrieval cache hit")
                return [chunk.copy() for chunk in cached]
            del self._retrieval_cache[key]

        results = hybrid_search(question=question, pramana_types=pramana_types, k=k)
        self._retrieval_cache[key] = (time.monotonic(), [chunk.copy() for chunk in results])
        self._retrieval_cache.move_to_end(key)
        if len(self._retrieval_cache) > self._retrieval_cache_max:
            self._retrieval_cache.popitem(last=False)
        return results

    def clear_runtime_caches(self) -> None:
        """Clear runtime caches to force cold-path profiling and debugging."""
        self._retrieval_cache.clear()

    def initialize(self) -> bool:
        """
        Initialize vector store with knowledge base and load from cache if available.
        Must be called once before first use.
        
        Returns:
            True if initialization successful
            
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            logger_pipeline.info("Initializing vector store...")
            
            # Try to load from cache first
            from .vector_store import FAISSVectorStore
            self.vector_store = FAISSVectorStore(load_cache=True)
            
            # If cache was empty, add chunks
            if self.vector_store.size() == 0:
                logger_pipeline.info("Cache empty. Loading knowledge base chunks...")
                chunks = KNOWLEDGE_BASE + _load_external_knowledge_base()
                logger_pipeline.info(f"Loading {len(chunks)} chunks into vector store...")
                added = self.vector_store.add_chunks(chunks)
                logger_pipeline.info(f"✓ Loaded {added} chunks")
            else:
                logger_pipeline.info(f"✓ Loaded {self.vector_store.size()} chunks from cache")
            
            self._initialized = True
            logger_pipeline.info("✓ Vector store initialization complete")
            return True
        except Exception as e:
            logger_pipeline.error(f"✗ RAG pipeline initialization failed: {e}", exc_info=True)
            return False

    def answer_question(
        self,
        question: str,
        pramana_types: Optional[List[str]] = None,
        use_llm: bool = True,
        use_reasoning_chain: bool = False,
    ) -> Dict[str, Any]:
        """
        Answer a question using full RAG pipeline with error handling.
        
        Args:
            question: Question to answer
            pramana_types: Pramana types to filter by
            use_llm: Whether to use LLM for reasoning (vs keyword rules)
            use_reasoning_chain: Whether to use chain-of-thought prompting
            
        Returns:
            Dict with answer, citations, confidence, and reasoning
            
        Raises:
            ValueError: If question is invalid
            RuntimeError: If pipeline execution fails
        """
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Question must be a non-empty string")
            
            logger_pipeline.info(f"Processing question: {question[:80]}...")
            
            if not self._initialized:
                logger_pipeline.info("Pipeline not initialized, initializing now...")
                self.initialize()

            if pramana_types is None:
                pramana_types = ["perception", "inference", "testimony", "comparison", "postulation"]

            # ===== STEP 1: HYBRID RETRIEVAL =====
            logger_pipeline.debug("STEP 1: Hybrid retrieval...")
            retrieved_chunks = self._get_cached_retrieval(
                question=question,
                pramana_types=pramana_types,
                k=10
            )

            if not retrieved_chunks:
                logger_pipeline.warning("No relevant chunks found in retrieval")
                return {
                    "question": question,
                    "answer": "No relevant information found in knowledge base.",
                    "confidence": 0.0,
                    "status": "invalid",
                    "rag_chunks": [],
                    "trace": {"stage": "retrieval", "result": "no_matches"},
                }

            logger_pipeline.debug(f"Retrieved {len(retrieved_chunks)} chunks")

            # ===== STEP 2: LLM REASONING =====
            llm_answer = None
            reasoning_trace = None

            if use_llm and self.llm_engine:
                try:
                    logger_pipeline.debug("STEP 2: LLM reasoning...")
                    # Detect MCQ questions (contain "A. " or "A) " option patterns)
                    import re as _re_mcq
                    _is_mcq = bool(_re_mcq.search(r'\b[A-D][.)]\s', question))
                    if _is_mcq:
                        mcq_result = self.llm_engine.answer_mcq(question, retrieved_chunks)
                        llm_answer = mcq_result.get("reason") or mcq_result.get("raw", "")
                        reasoning_trace = {"reasoning": mcq_result.get("raw", ""), "mcq_key": mcq_result.get("answer_key")}
                    elif use_reasoning_chain:
                        llm_result = self.llm_engine.generate_with_reasoning_chain(
                            question, retrieved_chunks
                        )
                        llm_answer = llm_result["answer"]
                        reasoning_trace = llm_result
                    else:
                        llm_answer, reasoning = self.llm_engine.generate_answer(
                            question, retrieved_chunks
                        )
                        reasoning_trace = {"reasoning": reasoning}
                    logger_pipeline.info(f"✓ LLM generated answer ({len(llm_answer or '')} chars)")
                except Exception as e:
                    logger_pipeline.error(f"LLM generation failed: {e}", exc_info=True)
                    # Fall back to retrieval answer
                    llm_answer = None

            # ===== STEP 3: ANSWER EXTRACTION =====
            logger_pipeline.debug("STEP 3: Answer extraction...")
            heuristic_answer = self._heuristic_answer_from_chunks(question, retrieved_chunks)
            llm_supported = bool(llm_answer) and self._is_supported_by_evidence(llm_answer, retrieved_chunks)
            is_fallacy_question = self._looks_like_fallacy_question(question)
            is_debate_fault_question = self._looks_like_debate_fault_question(question)
            is_debate_mode_question = self._looks_like_debate_mode_question(question)
            debate_cue_label = self._infer_debate_fault_from_question_clues(self._normalize_text(question))
            mode_cue_label = self._infer_debate_mode_from_question_clues(self._normalize_text(question))

            if llm_supported and is_fallacy_question and not self._llm_answer_has_fallacy_labels(llm_answer or ""):
                llm_supported = False
            if llm_supported and is_debate_fault_question and not self._llm_answer_has_debate_fault_labels(llm_answer or ""):
                llm_supported = False
            if llm_supported and is_debate_fault_question and debate_cue_label:
                if not self._llm_mentions_expected_debate_fault(llm_answer or "", debate_cue_label):
                    llm_supported = False
            if llm_supported and is_debate_mode_question and not self._llm_answer_has_debate_mode_labels(llm_answer or ""):
                llm_supported = False
            if llm_supported and is_debate_mode_question and mode_cue_label:
                if not self._llm_mentions_expected_debate_mode(llm_answer or "", mode_cue_label):
                    llm_supported = False

            if llm_supported:
                answer_text = llm_answer
                answer_source = "llm_reasoning"
            elif heuristic_answer:
                answer_text = heuristic_answer
                answer_source = "symbolic_fallback"
            else:
                answer_text = self._synthesize_answer_from_chunks(retrieved_chunks)
                answer_source = "retrieval_synthesis"

            # ===== STEP 4: PRAMANA VERIFICATION =====
            logger_pipeline.debug("STEP 4: Pramana verification...")
            raw_citations = [
                {
                    "id": chunk["id"],
                    "source": chunk.get("source", "unknown"),
                    "score": chunk.get("fused_score", chunk.get("score", 0)),
                    "excerpt": chunk["text"][:200],
                }
                for chunk in retrieved_chunks[:5]
            ]
            # Normalize RRF scores to [0,1] so verifier thresholds work correctly
            max_score = max((c["score"] for c in raw_citations), default=1.0) or 1.0
            citations = [{**c, "score": c["score"] / max_score} for c in raw_citations]

            try:
                verifier = _run_symbolic_verifier(
                    question=question,
                    answer_text=answer_text,
                    citations=citations,
                    rule_match={"score": 0.85} if answer_source == "llm_reasoning" else None,
                    used_rule_bank=False,
                    option_scores=[{"score": 0.85}, {"score": 0.45}],
                )
            except Exception as e:
                logger_pipeline.warning(f"Verifier failed: {e}, using default scores")
                verifier = {
                    "confidence_decomposition": {"final_confidence": 0.75},
                    "belief_revision": {"final_status": "justified"},
                    "constraints": [],
                    "violated_constraints": [],
                }

            # ===== BUILD RESPONSE =====
            response = {
                "question": question,
                "answer": answer_text,
                "answer_source": answer_source,
                "confidence": verifier.get("confidence_decomposition", {}).get("final_confidence", 0.75),
                "epistemic_status": verifier.get("belief_revision", {}).get("final_status", "justified"),
                "rag_chunks": self._format_rag_chunks(retrieved_chunks),
                "verifier": {
                    "constraints": verifier.get("constraints", []),
                    "violated": verifier.get("violated_constraints", []),
                    "final_status": verifier.get("belief_revision", {}).get("final_status", "justified"),
                },
                "reasoning": reasoning_trace,
            }

            if use_reasoning_chain and reasoning_trace:
                response["cot_reasoning"] = {
                    "understanding": reasoning_trace.get("understanding", ""),
                    "evidence": reasoning_trace.get("relevant_evidence", ""),
                    "steps": reasoning_trace.get("reasoning", ""),
                    "llm_confidence": reasoning_trace.get("confidence", 0.5),
                }

            logger_pipeline.info(f"✓ Question answered (confidence={response['confidence']:.2f}, status={response['epistemic_status']})")
            return response
            
        except Exception as e:
            logger_pipeline.error(f"✗ Pipeline execution failed: {e}", exc_info=True)
            raise RuntimeError(f"Question answering failed: {e}")

    def _format_rag_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format chunks for API response."""
        formatted = []
        for chunk in chunks[:5]:
            formatted.append({
                "id": chunk["id"],
                "source": chunk.get("source", "unknown"),
                "text": chunk["text"],
                "score": round(chunk.get("fused_score", chunk.get("score", 0)), 4),
                "tags": chunk.get("tags", []),
                "supports": chunk.get("supports", []),
                "sources": chunk.get("sources", "unknown"),
            })
        return formatted

    def search_only(
        self,
        question: str,
        pramana_types: Optional[List[str]] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform only retrieval without LLM reasoning with error handling.
        Useful for debugging/analysis.
        
        Args:
            question: Query
            pramana_types: Pramana filter
            k: Number of results
            
        Returns:
            Retrieved chunks
            
        Raises:
            ValueError: If question is invalid
            RuntimeError: If search fails
        """
        try:
            if not question or not isinstance(question, str):
                raise ValueError("Question must be a non-empty string")
            
            logger_pipeline.debug(f"Search-only mode: {question[:80]}...")
            
            if not self._initialized:
                self.initialize()

            if pramana_types is None:
                pramana_types = ["perception", "inference", "testimony", "comparison", "postulation"]

            results = self._get_cached_retrieval(
                question=question,
                pramana_types=pramana_types,
                k=k
            )
            
            logger_pipeline.info(f"✓ Search returned {len(results)} results")
            return results
        except Exception as e:
            logger_pipeline.error(f"✗ Search-only failed: {e}", exc_info=True)
            raise RuntimeError(f"Search failed: {e}")

    def explain_answer(
        self,
        question: str,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Return detailed explanation of why a specific answer was chosen.
        Shows the decision path through the fallback hierarchy.
        
        Args:
            question: User's query
            use_llm: Whether to include LLM in reasoning
            
        Returns:
            Explanation with decision rationale, alternative paths, and confidence factors
        """
        try:
            if not self._initialized:
                self.initialize()

            normalized_question = self._normalize_text(question)
            is_fallacy_question = self._looks_like_fallacy_question(question)
            is_debate_fault_question = self._looks_like_debate_fault_question(question)
            is_debate_mode_question = self._looks_like_debate_mode_question(question)

            explanation = {
                "question": question,
                "question_type": {
                    "is_fallacy_question": is_fallacy_question,
                    "is_debate_fault_question": is_debate_fault_question,
                    "is_debate_mode_question": is_debate_mode_question,
                },
                "fallback_path": [],
                "evidence_found": False,
                "clue_inferred": False,
                "llm_attempted": False,
            }

            # Retrieve evidence
            retrieved_chunks = self._get_cached_retrieval(question, None, 10)
            explanation["retrieval_count"] = len(retrieved_chunks)

            # Trace fallback path based on question type
            if is_debate_mode_question:
                mode_label = self._select_debate_mode_label(retrieved_chunks)
                cue_label = self._infer_debate_mode_from_question_clues(normalized_question)
                explanation["fallback_path"].append("STEP 1: Evidence-based label selection")
                if mode_label:
                    explanation["fallback_path"].append(f"  → Found '{mode_label}' in evidence chunks")
                    explanation["evidence_found"] = True
                else:
                    explanation["fallback_path"].append("  → No explicit debate mode labels in evidence")
                explanation["fallback_path"].append("STEP 2: Question clue inference")
                if cue_label:
                    explanation["fallback_path"].append(f"  → Inferred '{cue_label}' from question markers")
                    explanation["clue_inferred"] = True
                else:
                    explanation["fallback_path"].append("  → No clear mode markers in question")
                final_label = mode_label or cue_label
                explanation["selected_label"] = final_label
                explanation["reason"] = f"Question detected as debate-mode type. Final label: {final_label or 'generic catalog'}"

            elif is_debate_fault_question:
                debate_label = self._select_debate_fault_label(retrieved_chunks)
                cue_label = self._infer_debate_fault_from_question_clues(normalized_question)
                explanation["fallback_path"].append("STEP 1: Evidence-based label selection")
                if debate_label:
                    explanation["fallback_path"].append(f"  → Found '{debate_label}' in evidence chunks")
                    explanation["evidence_found"] = True
                else:
                    explanation["fallback_path"].append("  → No explicit debate-fault labels in evidence")
                explanation["fallback_path"].append("STEP 2: Question clue inference")
                if cue_label:
                    explanation["fallback_path"].append(f"  → Inferred '{cue_label}' from question description")
                    explanation["clue_inferred"] = True
                else:
                    explanation["fallback_path"].append("  → No clear fault markers in question")
                final_label = debate_label or cue_label
                explanation["selected_label"] = final_label
                explanation["reason"] = f"Question detected as debate-fault type. Final label: {final_label or 'generic fallback'}"

            elif is_fallacy_question:
                label = self._select_hetvabhasa_label(retrieved_chunks)
                cue_label = self._infer_fallacy_from_question_clues(normalized_question)
                explanation["fallback_path"].append("STEP 1: Evidence-based label selection")
                if label:
                    explanation["fallback_path"].append(f"  → Found '{label}' in evidence chunks")
                    explanation["evidence_found"] = True
                else:
                    explanation["fallback_path"].append("  → No Hetvabhasa labels in evidence")
                explanation["fallback_path"].append("STEP 2: Question clue inference")
                if cue_label:
                    explanation["fallback_path"].append(f"  → Inferred '{cue_label}' from fallacy wording")
                    explanation["clue_inferred"] = True
                else:
                    explanation["fallback_path"].append("  → No clear fallacy markers in question")
                final_label = label or cue_label
                explanation["selected_label"] = final_label
                explanation["reason"] = f"Question detected as Hetvabhasa type. Final label: {final_label or 'generic catalog'}"
            else:
                explanation["reason"] = "Not a recognized Nyaya question type. Standard retrieval applies."

            return explanation
        except Exception as e:
            logger_pipeline.error(f"Explanation generation failed: {e}", exc_info=True)
            return {"error": str(e), "question": question}

    def answer_batch(
        self,
        questions: List[str],
        use_llm: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Answer multiple questions efficiently with shared cache.
        
        Args:
            questions: List of query strings
            use_llm: Whether to use LLM
            
        Returns:
            List of answer results
        """
        if not self._initialized:
            self.initialize()

        results = []
        for i, question in enumerate(questions):
            try:
                logger_pipeline.debug(f"Batch item {i+1}/{len(questions)}: {question[:60]}...")
                result = self.answer_question(question, use_llm=use_llm)
                results.append({**result, "batch_index": i})
            except Exception as e:
                logger_pipeline.warning(f"Batch item {i+1} failed: {e}")
                results.append({
                    "batch_index": i,
                    "question": question,
                    "error": str(e),
                    "answer": None,
                })

        logger_pipeline.info(f"Batch processing complete: {len(results)} results")
        return results


# Singleton instance
_pipeline_instance: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create global RAG pipeline instance with error handling."""
    global _pipeline_instance
    try:
        if _pipeline_instance is None:
            logger_pipeline.info("Creating RAG pipeline singleton...")
            _pipeline_instance = RAGPipeline()
        return _pipeline_instance
    except Exception as e:
        logger_pipeline.error(f"✗ Failed to get RAG pipeline: {e}", exc_info=True)
        raise


def rag_answer_question(
    question: str,
    pramana_types: Optional[List[str]] = None,
    use_llm: bool = True,
    use_reasoning_chain: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to answer a question using RAG pipeline with error handling.
    
    Args:
        question: Question to answer
        pramana_types: Pramana types to filter
        use_llm: Use LLM reasoning
        use_reasoning_chain: Use chain-of-thought
        
    Returns:
        Answer with citations and confidence
        
    Raises:
        ValueError: If question is invalid
        RuntimeError: If answering fails
    """
    try:
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")
        
        pipeline = get_rag_pipeline()
        if not pipeline._initialized:
            logger_pipeline.info("Initializing RAG pipeline...")
            pipeline.initialize()
        
        result = pipeline.answer_question(
            question=question,
            pramana_types=pramana_types,
            use_llm=use_llm,
            use_reasoning_chain=use_reasoning_chain,
        )
        
        return result
    except Exception as e:
        logger_pipeline.error(f"✗ RAG answer failed: {e}", exc_info=True)
        raise
