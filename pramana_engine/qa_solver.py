from __future__ import annotations

import json
import re
import threading
from hashlib import blake2b
from pathlib import Path
from typing import Any, Dict, List, Tuple


KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "NS-1.1.3",
        "source": "Nyaya Sutra 1.1.3",
        "text": "Pratyaksha, Anumana, Upamana, and Shabda are the four core pramanas.",
        "tags": ["pramana", "classification", "nyaya", "four-pramanas"],
        "supports": ["perception", "inference", "comparison", "testimony"],
    },
    {
        "id": "NS-2.1.10",
        "source": "Nyaya Sutra 2.1.10",
        "text": "When perception directly contradicts inferential or testimonial claims, perception prevails.",
        "tags": ["pratyaksha", "priority", "badhita", "conflict"],
        "supports": ["perception"],
    },
    {
        "id": "VB-1.1.1",
        "source": "Vatsyayana Bhashya 1.1.1",
        "text": "Direct perceptual cognition is immediate and foundational, because it rests on sense-object contact.",
        "tags": ["sense-object", "contact", "perception", "immediate"],
        "supports": ["perception"],
    },
    {
        "id": "NS-1.1.5",
        "source": "Nyaya Sutra 1.1.5",
        "text": "Anumana depends on established vyapti and inferential signs (hetu).",
        "tags": ["anumana", "vyapti", "hetu", "inference"],
        "supports": ["inference"],
    },
    {
        "id": "UV-2.2.1",
        "source": "Uddyotakara Varttika 2.2.1",
        "text": "Shabda gains force from apta-vacana, testimony of credible knowers.",
        "tags": ["shabda", "testimony", "authority", "apta"],
        "supports": ["testimony"],
    },
    {
        "id": "CONF-2026-OVERVIEW",
        "source": "Unriddling Inference Conference Brief 2026",
        "text": "ISS Delhi, in collaboration with Brhat, organizes Unriddling Inference to connect Pramana theory with modern logic and artificial intelligence, aiming at transparent and explainable AI.",
        "tags": ["unriddling", "inference", "conference", "iss", "brhat", "ai", "explainable"],
        "supports": ["testimony", "inference"],
    },
    {
        "id": "CONF-2026-METACOGNITION",
        "source": "Unriddling Inference Conference Brief 2026",
        "text": "The initiative emphasizes honest AI agents with meta-cognition: systems should track information sources, justify inference steps, and recognize when they may be wrong.",
        "tags": ["meta-cognition", "honest-ai", "traceability", "justification", "error-awareness"],
        "supports": ["inference", "testimony"],
    },
    {
        "id": "CONF-2026-TRACKS",
        "source": "Unriddling Inference Conference Brief 2026",
        "text": "The program has two tracks: a conference on April 12, 2026 and a 6-to-8-week hackathon where students build truth-seeking inference engines.",
        "tags": ["april-12-2026", "hackathon", "truth-seeking", "students", "tracks"],
        "supports": ["testimony", "comparison"],
    },
    {
        "id": "CONF-2026-GOAL",
        "source": "Unriddling Inference Conference Brief 2026",
        "text": "Unlike black-box statistical guessing, the project goal is AI that can admit ignorance, revise beliefs under new evidence, and preserve logical integrity.",
        "tags": ["black-box", "ignorance", "belief-revision", "logical-integrity", "truth-seeking"],
        "supports": ["inference", "postulation"],
    },
]

PRAMANA_ALIAS: Dict[str, Tuple[str, ...]] = {
    "perception": ("pratyaksha", "pratyaksa", "perception", "sense-object", "sense object", "direct"),
    "inference": ("anumana", "inference", "inferential", "hetu", "vyapti"),
    "testimony": ("shabda", "sabda", "testimony", "verbal", "scriptural"),
    "comparison": ("upamana", "comparison", "analogy"),
    "postulation": ("arthapatti", "postulation", "presumption"),
    "aprama": ("aprama", "non-valid cognition", "invalid cognition", "error", "illusion", "bhrama"),
}

DISPLAY_NAME: Dict[str, str] = {
    "perception": "Pratyaksha",
    "inference": "Anumana",
    "testimony": "Shabda",
    "comparison": "Upamana",
    "postulation": "Arthapatti",
    "aprama": "Aprama",
}


CONCEPT_TRIGGERS: Dict[str, Tuple[str, ...]] = {
    "aprama": ("illusion", "illusory", "bhrama", "error", "false cognition"),
}

RULE_BANK_PATH = Path(__file__).with_name("question_rule_bank.json")
EXTERNAL_CORPUS_DIR = Path(__file__).with_name("data") / "external_json_output"
_RULE_BANK_CACHE: List[Dict[str, Any]] | None = None
_EXTERNAL_KB_CACHE: List[Dict[str, Any]] | None = None
_EXTERNAL_KB_VERSION: str | None = None
_GRAPH_CACHE: Dict[str, Any] | None = None
_CACHE_LOCK = threading.Lock()  # Guards all _*_CACHE globals against concurrent writes

# Conference ID prefix — using a tuple so adding future years is one-line
_CONFERENCE_ID_PREFIXES: tuple[str, ...] = ("CONF-2026", "CONF-2027", "CONF-2028")

DEFAULT_RULE_CONFIDENCE = 0.85  # Named constant: prior confidence for rules without explicit value

DEFAULT_SUPPORTS = ["perception", "inference", "comparison", "testimony", "postulation"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "won", "into", "under", "about",
    "can", "may", "might", "does", "did", "do", "if", "than", "then", "this", "those",
    "these", "their", "its", "our", "your", "his", "her", "them", "they", "we", "you",
}

CONFERENCE_QUERY_MARKERS: Tuple[str, ...] = (
    "unriddling",
    "conference",
    "meta-cognition",
    "metacognition",
    "iss delhi",
    "brhat",
    "truth-seeking",
)

DOMAIN_QUERY_MARKERS: Tuple[str, ...] = (
    "nyaya",
    "pramana",
    "pratyaksha",
    "anumana",
    "shabda",
    "upamana",
    "arthapatti",
    "aprama",
    "bhrama",
    "vyapti",
    "hetu",
    "paksha",
    "sadhya",
    "nigamana",
    "conference",
    "unriddling",
    "meta-cognition",
    "metacognition",
    "iss delhi",
    "brhat",
)


def _norm(text: str) -> str:
    return (
        text.lower()
        .replace("ā", "a")
        .replace("ī", "i")
        .replace("ū", "u")
        .replace("ṣ", "s")
        .replace("ś", "s")
        .replace("ñ", "n")
        .replace("ṛ", "r")
        .replace("ṇ", "n")
        .replace("ṁ", "m")
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_]+", _norm(text))


def _content_tokens(tokens: List[str]) -> set[str]:
    return {
        t for t in tokens
        if t not in STOPWORDS and len(t) > 2 and not t.isdigit()
    }


def _is_conference_query(question_norm: str) -> bool:
    return any(marker in question_norm for marker in CONFERENCE_QUERY_MARKERS)


def _has_domain_anchor(question_norm: str) -> bool:
    return any(marker in question_norm for marker in DOMAIN_QUERY_MARKERS)


def _is_conference_passage(passage: Dict[str, Any]) -> bool:
    pid = str(passage.get("id", ""))
    if any(pid.startswith(prefix) for prefix in _CONFERENCE_ID_PREFIXES):
        return True

    source = _norm(str(passage.get("source", "")))
    tags = " ".join(str(t) for t in passage.get("tags", []))
    text = _norm(str(passage.get("text", "")))
    combined = f"{source} {_norm(tags)} {text}"
    return any(marker in combined for marker in CONFERENCE_QUERY_MARKERS)


def _load_rule_bank() -> List[Dict[str, Any]]:
    global _RULE_BANK_CACHE
    with _CACHE_LOCK:
        if _RULE_BANK_CACHE is not None:
            return _RULE_BANK_CACHE
        if not RULE_BANK_PATH.exists():
            _RULE_BANK_CACHE = []
            return _RULE_BANK_CACHE

        raw = json.loads(RULE_BANK_PATH.read_text(encoding="utf-8"))
        normalized_rules = []
        for rule in raw:
            keywords = [str(k) for k in rule.get("keywords", []) if str(k).strip()]
            normalized_rules.append(
                {
                    "id": str(rule.get("id", "")),
                    "keywords": keywords,
                    "answer_option": str(rule.get("answer_option", "")).upper() or None,
                    "answer_text": str(rule.get("answer_text", "")).strip(),
                    "explanation": str(rule.get("explanation", "")).strip(),
                    "pramana": str(rule.get("pramana", "")).strip(),
                    "confidence": float(rule.get("confidence", DEFAULT_RULE_CONFIDENCE)),
                }
            )

        _RULE_BANK_CACHE = normalized_rules
        return _RULE_BANK_CACHE


def _split_into_chunks(text: str, max_chars: int = 420) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: List[str] = []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    current = ""
    for part in parts:
        if not part:
            continue
        tentative = f"{current} {part}".strip()
        if len(tentative) <= max_chars:
            current = tentative
            continue

        if current:
            chunks.append(current)
        if len(part) <= max_chars:
            current = part
            continue

        # Hard-wrap oversized spans when sentence segmentation is unavailable.
        for idx in range(0, len(part), max_chars):
            chunks.append(part[idx : idx + max_chars].strip())
        current = ""

    if current:
        chunks.append(current)
    return [c for c in chunks if c]


def _extract_text_candidates(value: Any, output: List[str], depth: int = 0, max_depth: int = 6) -> None:
    if depth > max_depth:
        return
    if isinstance(value, str):
        cleaned = value.strip()
        if len(cleaned) >= 30:
            output.append(cleaned)
        return
    if isinstance(value, list):
        for item in value:
            _extract_text_candidates(item, output, depth + 1, max_depth)
        return
    if isinstance(value, dict):
        preferred_keys = [
            "text",
            "content",
            "body",
            "abstract",
            "summary",
            "paragraph",
            "paragraphs",
            "chunks",
            "sentences",
            "ocr_text",
        ]
        for key in preferred_keys:
            if key in value:
                _extract_text_candidates(value[key], output, depth + 1, max_depth)
        for item in value.values():
            _extract_text_candidates(item, output, depth + 1, max_depth)


def _supports_for_text(text: str) -> List[str]:
    normalized = _norm(text)
    detected: List[str] = []
    for canonical, aliases in PRAMANA_ALIAS.items():
        if canonical == "aprama":
            continue
        if any(alias in normalized for alias in aliases):
            detected.append(canonical)
    return detected or list(DEFAULT_SUPPORTS)


def _tags_for_source(source_name: str, text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", _norm(source_name))
    tags = [t for t in tokens if len(t) > 2][:8]
    tags.extend(_tokenize(text)[:8])
    deduped: List[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped[:12]


def _load_external_knowledge_base() -> List[Dict[str, Any]]:
    global _EXTERNAL_KB_CACHE, _EXTERNAL_KB_VERSION, _GRAPH_CACHE
    version = get_external_corpus_version()
    # Fast path: check outside lock (version comparison is atomic enough for reads)
    if _EXTERNAL_KB_CACHE is not None and _EXTERNAL_KB_VERSION == version:
        return _EXTERNAL_KB_CACHE
    with _CACHE_LOCK:
        # Re-check inside lock to avoid double-build under concurrent requests
        if _EXTERNAL_KB_CACHE is not None and _EXTERNAL_KB_VERSION == version:
            return _EXTERNAL_KB_CACHE
        if not EXTERNAL_CORPUS_DIR.exists():
            _EXTERNAL_KB_CACHE = []
            _EXTERNAL_KB_VERSION = version
            return _EXTERNAL_KB_CACHE

        passages: List[Dict[str, Any]] = []
        max_total_chunks = 1200

        for json_path in sorted(EXTERNAL_CORPUS_DIR.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            candidates: List[str] = []
            _extract_text_candidates(payload, candidates)
            if not candidates:
                continue

            chunk_counter = 0
            for candidate in candidates:
                for chunk in _split_into_chunks(candidate):
                    supports = _supports_for_text(chunk)
                    passages.append(
                        {
                            "id": f"EXT:{json_path.stem}:{chunk_counter}",
                            "source": json_path.name,
                            "text": chunk,
                            "tags": _tags_for_source(json_path.name, chunk),
                            "supports": supports,
                        }
                    )
                    chunk_counter += 1
                    if len(passages) >= max_total_chunks:
                        _EXTERNAL_KB_CACHE = passages
                        _EXTERNAL_KB_VERSION = version
                        _GRAPH_CACHE = None
                        return _EXTERNAL_KB_CACHE

        _EXTERNAL_KB_CACHE = passages
        _EXTERNAL_KB_VERSION = version
        _GRAPH_CACHE = None
        return _EXTERNAL_KB_CACHE


def get_external_corpus_version() -> str:
    """Return a deterministic version hash for external corpus files.

    This is inexpensive (metadata only) and is used by retrieval caches to
    invalidate stale entries when corpus files are added/removed/modified.
    """
    if not EXTERNAL_CORPUS_DIR.exists():
        return "missing"

    digest = blake2b(digest_size=16)
    files = sorted(EXTERNAL_CORPUS_DIR.glob("*.json"))
    digest.update(str(len(files)).encode("utf-8"))

    for path in files:
        try:
            stat = path.stat()
            digest.update(path.name.encode("utf-8", errors="ignore"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
            digest.update(str(stat.st_size).encode("ascii"))
        except OSError:
            continue

    return digest.hexdigest()


def _knowledge_base() -> List[Dict[str, Any]]:
    return KNOWLEDGE_BASE + _load_external_knowledge_base()


def _graph_entities(text: str) -> List[str]:
    n = _norm(text)
    entities: List[str] = []
    for canonical, aliases in PRAMANA_ALIAS.items():
        if canonical == "aprama":
            continue
        if any(alias in n for alias in aliases):
            entities.append(DISPLAY_NAME[canonical])

    extra_terms = ["prama", "aprama", "vyapti", "hetu", "paksha", "sadhya", "nigamana", "anupalabdhi", "abhava"]
    for term in extra_terms:
        if term in n:
            entities.append(term)

    deduped: List[str] = []
    for entity in entities:
        if entity not in deduped:
            deduped.append(entity)
    return deduped


def _build_knowledge_graph() -> Dict[str, Any]:
    global _GRAPH_CACHE
    if _GRAPH_CACHE is not None:
        return _GRAPH_CACHE

    nodes: set[str] = set()
    edges: List[Dict[str, Any]] = []
    for passage in _knowledge_base():
        ents = _graph_entities(passage.get("text", ""))
        for ent in ents:
            nodes.add(ent)
        for idx, source in enumerate(ents):
            for target in ents[idx + 1 :]:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": "co_occurs",
                        "support_id": passage.get("id", "unknown"),
                    }
                )

    _GRAPH_CACHE = {"nodes": sorted(nodes), "edges": edges[:5000]}
    return _GRAPH_CACHE


def _graph_alignment_score(question: str, answer_text: str) -> float:
    graph = _build_knowledge_graph()
    q_entities = set(_graph_entities(question))
    a_entities = set(_graph_entities(answer_text))
    if not q_entities or not a_entities:
        return 0.0

    connected = 0
    for edge in graph.get("edges", []):
        src = edge.get("source")
        dst = edge.get("target")
        if (src in q_entities and dst in a_entities) or (dst in q_entities and src in a_entities):
            connected += 1

    return min(1.0, connected / 4.0)


def _question_polarity(question: str) -> str:
    n = _norm(question)
    negative_markers = [" not ", " except", "reject", "does not", "isnt", "isn't", "not a"]
    if any(marker in f" {n} " for marker in negative_markers):
        return "negative"
    return "positive"


def _evidence_polarity_for_answer(excerpt: str, answer_text: str) -> str:
    n_excerpt = _norm(excerpt)
    n_answer = _norm(answer_text)
    if not n_answer or n_answer not in n_excerpt:
        return "neutral"

    negative_markers = [" not ", " no ", "reject", "invalid", "false", "does not", "isnt", "isn't"]
    if any(marker in f" {n_excerpt} " for marker in negative_markers):
        return "negative"
    return "positive"


def _source_authority(citations: List[Dict[str, Any]]) -> float:
    if not citations:
        return 0.5

    vals: List[float] = []
    for c in citations:
        cid = str(c.get("id", ""))
        src = str(c.get("source", "")).lower()
        if cid.startswith("RULE:"):
            vals.append(0.95)
        elif cid.startswith("EXT:"):
            vals.append(0.72)
        elif "nyaya sutra" in src or "bhashya" in src or "varttika" in src:
            vals.append(0.9)
        else:
            vals.append(0.8)
    return round(sum(vals) / max(1, len(vals)), 4)


def _retrieval_support(citations: List[Dict[str, Any]]) -> float:
    if not citations:
        return 0.0
    avg_score = sum(float(c.get("score", 0.0)) for c in citations) / len(citations)
    return round(min(1.0, avg_score / 8.0), 4)


def _build_propositions(question: str, answer_text: str, citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    propositions: List[Dict[str, Any]] = []
    question_entities = _graph_entities(question)
    target = answer_text.strip() or "candidate_answer"

    for c in citations:
        excerpt = str(c.get("excerpt", "")).strip()
        evidence_polarity = _evidence_polarity_for_answer(excerpt, target)
        relation = "supports" if evidence_polarity in {"positive", "neutral"} else "contradicts"
        propositions.append(
            {
                "subject": question_entities[0] if question_entities else "question",
                "predicate": relation,
                "object": target,
                "evidence_id": c.get("id", "unknown"),
                "evidence_excerpt": excerpt,
            }
        )
    return propositions


def _run_symbolic_verifier(
    question: str,
    answer_text: str,
    citations: List[Dict[str, Any]],
    rule_match: Dict[str, Any] | None,
    used_rule_bank: bool,
    option_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    question_polarity = _question_polarity(question)
    evidence_polarities = [_evidence_polarity_for_answer(str(c.get("excerpt", "")), answer_text) for c in citations]

    mismatches = 0
    for pol in evidence_polarities:
        if pol == "neutral":
            continue
        if pol != question_polarity:
            mismatches += 1

    ambiguity = 0.0
    if len(option_scores) >= 2:
        top = float(option_scores[0].get("score", 0.0))
        runner = float(option_scores[1].get("score", 0.0))
        if top > 0:
            margin = (top - runner) / top
            ambiguity = max(0.0, min(1.0, 1.0 - margin))

    contradiction_strength = min(1.0, (mismatches / max(1, len(citations))) + 0.35 * ambiguity)
    contradiction_penalty = round(-contradiction_strength, 4)

    retrieval = _retrieval_support(citations)
    graph_alignment = _graph_alignment_score(question, answer_text)
    base_rule = 1.0 if used_rule_bank else (0.8 if rule_match else 0.65)
    rule_consistency = round(min(1.0, base_rule + 0.2 * graph_alignment), 4)
    source_authority = _source_authority(citations)

    raw = retrieval + rule_consistency + contradiction_penalty + source_authority
    final_confidence = round(max(0.0, min(1.0, (raw + 1.0) / 4.0)), 4)

    constraints = {
        "non_empty_answer": bool(answer_text.strip()),
        "has_retrieval_support": retrieval >= 0.15,
        "rule_satisfiable": rule_consistency >= 0.6,
        "no_strong_contradiction": contradiction_strength < 0.45,
    }
    violated = [k for k, v in constraints.items() if not v]
    verifier_pass = len(violated) == 0

    preliminary_status = "valid" if verifier_pass else "unjustified"
    belief_revision_applied = False
    final_status = preliminary_status
    if contradiction_strength >= 0.45 and preliminary_status == "valid":
        final_status = "suspended"
        belief_revision_applied = True
    elif contradiction_strength >= 0.45 and preliminary_status != "valid":
        final_status = "suspended"

    return {
        "verifier_pass": verifier_pass,
        "violated_constraints": violated,
        "constraints": constraints,
        "propositions": _build_propositions(question, answer_text, citations),
        "confidence_decomposition": {
            "retrieval_support": retrieval,
            "rule_consistency": rule_consistency,
            "contradiction_penalty": contradiction_penalty,
            "source_authority": source_authority,
            "graph_alignment": round(graph_alignment, 4),
            "final_confidence": final_confidence,
            "formula": "retrieval_support + rule_consistency + contradiction_penalty + source_authority",
        },
        "belief_revision": {
            "applied": belief_revision_applied,
            "reason": "conflicting evidence with accepted belief" if belief_revision_applied else "none",
            "preliminary_status": preliminary_status,
            "final_status": final_status,
        },
    }


def _match_rule(question: str) -> Dict[str, Any] | None:
    question_norm = _norm(question)
    best_rule: Dict[str, Any] | None = None
    best_score = 0.0

    for rule in _load_rule_bank():
        keywords = rule.get("keywords", [])
        score = 0.0
        for keyword in keywords:
            normalized_keyword = _norm(keyword)
            if not normalized_keyword or normalized_keyword not in question_norm:
                continue
            token_weight = max(1, len(normalized_keyword.split()))
            char_weight = min(len(normalized_keyword) / 12.0, 2.0)
            score += token_weight + char_weight

        if score > best_score:
            best_score = score
            best_rule = rule

    if best_rule is None or best_score <= 0:
        return None

    matched_keywords = [k for k in best_rule.get("keywords", []) if _norm(k) in question_norm]
    out = dict(best_rule)
    out["match_score"] = best_score
    out["matched_keywords"] = matched_keywords
    return out


def _option_for_rule(rule_match: Dict[str, Any], options: List[Dict[str, str]]) -> Dict[str, str] | None:
    answer_text_norm = _norm(rule_match.get("answer_text", ""))
    if answer_text_norm:
        def _phrase_match(text: str, phrase: str) -> bool:
            return re.search(rf"\b{re.escape(phrase)}\b", text) is not None

        by_text = next(
            (
                opt
                for opt in options
                if _phrase_match(_norm(opt["text"]), answer_text_norm)
                or _phrase_match(answer_text_norm, _norm(opt["text"]))
            ),
            None,
        )
        if by_text is not None:
            return by_text

    answer_key = rule_match.get("answer_option")
    if answer_key:
        by_key = next((opt for opt in options if opt["key"] == answer_key), None)
        if by_key is not None:
            return by_key

    return None


def _display_from_pramana(pramana_text: str) -> str:
    canonical = _detect_pramana_from_text(pramana_text)
    if canonical and canonical in DISPLAY_NAME:
        return DISPLAY_NAME[canonical]
    return pramana_text.strip() or "Anumana"


def _parse_options(question: str) -> List[Dict[str, str]]:
    option_pattern = re.compile(r"\b([A-D])\.\s*(.+?)(?=\s+\b[A-D]\.\s*|$)", re.IGNORECASE)
    options = []
    for m in option_pattern.finditer(question):
        options.append({"key": m.group(1).upper(), "text": m.group(2).strip()})
    return options


def _detect_pramana_from_text(text: str) -> str | None:
    n = _norm(text)
    for canonical, aliases in PRAMANA_ALIAS.items():
        for alias in aliases:
            if alias in n:
                return canonical
    return None


def _passage_retrieval_score(question: str, question_tokens: List[str], passage: Dict[str, Any], target_pramana: str) -> float:
    passage_tokens = _tokenize(passage["text"] + " " + " ".join(passage["tags"]))
    q_set = _content_tokens(question_tokens)
    p_set = _content_tokens(passage_tokens)
    overlap = len(q_set.intersection(p_set))
    overlap_ratio = overlap / max(1, len(q_set))

    # Require lexical grounding before applying pramana support bonus.
    support_bonus = 1.8 if (target_pramana in passage["supports"] and overlap > 0) else 0.0

    conference_boost = 0.0
    question_norm = _norm(question)
    if _is_conference_query(question_norm):
        if _is_conference_passage(passage):
            conference_boost = 3.0
        elif overlap == 0:
            conference_boost = -0.75

    return (1.7 * overlap) + (2.0 * overlap_ratio) + support_bonus + conference_boost


def _score_pramana(question: str, pramana: str) -> Tuple[float, List[Dict[str, Any]]]:
    question_norm = _norm(question)
    question_tokens = _tokenize(question)

    alias_hits = 0.0
    for alias in PRAMANA_ALIAS[pramana]:
        if alias in question_norm:
            alias_hits += 1.5

    # Give perception a mild default authority bonus for verification-oriented questions.
    verification_bonus = 0.8 if pramana == "perception" and any(k in question_norm for k in ["verify", "ultimately", "final", "certain"]) else 0.0

    retrieved = []
    for passage in _knowledge_base():
        score = _passage_retrieval_score(question, question_tokens, passage, pramana)
        if score > 0.05:
            retrieved.append(
                {
                    "id": passage["id"],
                    "source": passage["source"],
                    "score": round(score, 3),
                    "excerpt": passage["text"],
                }
            )

    retrieved.sort(key=lambda x: x["score"], reverse=True)
    top_retrieved = retrieved[:3]
    retrieval_score = sum(r["score"] for r in top_retrieved) / max(len(top_retrieved), 1)

    total = alias_hits + verification_bonus + retrieval_score
    return round(total, 4), top_retrieved


def _citation_query_coverage(question: str, citations: List[Dict[str, Any]]) -> float:
    question_terms = _content_tokens(_tokenize(question))
    if not question_terms or not citations:
        return 0.0

    cited_text = " ".join(str(c.get("excerpt", "")) for c in citations)
    cited_terms = _content_tokens(_tokenize(cited_text))
    overlap = len(question_terms.intersection(cited_terms))
    return overlap / max(1, len(question_terms))


def solve_question(question: str) -> Dict[str, Any]:
    q = question.strip()
    if not q:
        raise ValueError("Question must not be empty.")

    options = _parse_options(q)
    rule_match = _match_rule(q)

    if options:
        if rule_match is not None:
            forced = _option_for_rule(rule_match, options)
            if forced is not None:
                answer_pramana = _display_from_pramana(rule_match.get("pramana", ""))
                option_scores = [
                    {
                        "key": opt["key"],
                        "text": opt["text"],
                        "canonical_pramana": _detect_pramana_from_text(opt["text"]) or "inference",
                        "display_pramana": DISPLAY_NAME.get(_detect_pramana_from_text(opt["text"]) or "inference", "Anumana"),
                        "score": 1.0 if opt["key"] == forced["key"] else 0.1,
                        "citations": [{"id": f"RULE:{rule_match['id']}", "source": "question_rule_bank", "score": rule_match.get("match_score", 1), "excerpt": rule_match.get("explanation", "")}],
                    }
                    for opt in options
                ]
                verifier = _run_symbolic_verifier(
                    question=q,
                    answer_text=forced["text"],
                    citations=option_scores[0].get("citations", []),
                    rule_match=rule_match,
                    used_rule_bank=True,
                    option_scores=option_scores,
                )
                return {
                    "mode": "mcq",
                    "question": q,
                    "answer_key": forced["key"],
                    "answer_text": forced["text"],
                    "answer_pramana": answer_pramana,
                    "confidence": verifier["confidence_decomposition"]["final_confidence"],
                    "epistemic_status": verifier["belief_revision"]["final_status"],
                    "option_scores": option_scores,
                    "verifier": verifier,
                    "trace": {
                        "type": "neuro_symbolic_rule_bank_match",
                        "rule_id": rule_match["id"],
                        "matched_keywords": rule_match.get("matched_keywords", []),
                        "explanation": rule_match.get("explanation", ""),
                        "forced_by": "question_rule_bank",
                        "neural_parser": {
                            "mode": "rule_and_keyword_parser",
                            "selected_answer_key": forced["key"],
                        },
                        "symbolic_verifier": {
                            "verifier_pass": verifier["verifier_pass"],
                            "violated_constraints": verifier["violated_constraints"],
                        },
                    },
                }

        question_norm = _norm(q)
        target_concepts = {
            concept
            for concept, triggers in CONCEPT_TRIGGERS.items()
            if any(trigger in question_norm for trigger in triggers)
        }

        option_rows = []
        for opt in options:
            canonical = _detect_pramana_from_text(opt["text"]) or "inference"
            score, citations = _score_pramana(q + " " + opt["text"], canonical)

            option_norm = _norm(opt["text"])
            for concept in target_concepts:
                concept_aliases = PRAMANA_ALIAS.get(concept, ())
                if any(alias in option_norm for alias in concept_aliases):
                    score += 3.0

            option_rows.append(
                {
                    "key": opt["key"],
                    "text": opt["text"],
                    "canonical_pramana": canonical,
                    "display_pramana": DISPLAY_NAME[canonical],
                    "score": score,
                    "citations": citations,
                }
            )

        option_rows.sort(key=lambda x: x["score"], reverse=True)
        winner = option_rows[0]
        verifier = _run_symbolic_verifier(
            question=q,
            answer_text=winner["text"],
            citations=winner.get("citations", []),
            rule_match=rule_match,
            used_rule_bank=False,
            option_scores=option_rows,
        )

        return {
            "mode": "mcq",
            "question": q,
            "answer_key": winner["key"],
            "answer_text": winner["text"],
            "answer_pramana": winner["display_pramana"],
            "confidence": verifier["confidence_decomposition"]["final_confidence"],
            "epistemic_status": verifier["belief_revision"]["final_status"],
            "option_scores": option_rows,
            "verifier": verifier,
            "trace": {
                "type": "neuro_symbolic_option_extraction_and_scoring",
                "parsed_options": options,
                "winning_row": winner,
                "neural_parser": {
                    "mode": "retrieval_scoring_parser",
                    "selected_answer_key": winner["key"],
                },
                "symbolic_verifier": {
                    "verifier_pass": verifier["verifier_pass"],
                    "violated_constraints": verifier["violated_constraints"],
                },
            },
        }

    rows = []
    for pramana in PRAMANA_ALIAS:
        score, citations = _score_pramana(q, pramana)
        rows.append(
            {
                "canonical_pramana": pramana,
                "display_pramana": DISPLAY_NAME[pramana],
                "score": score,
                "citations": citations,
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    winner = rows[0]

    if rule_match is not None:
        answer_pramana = _display_from_pramana(rule_match.get("pramana", ""))
        answer_text = rule_match.get("answer_text") or answer_pramana
        verifier = _run_symbolic_verifier(
            question=q,
            answer_text=answer_text,
            citations=winner.get("citations", []),
            rule_match=rule_match,
            used_rule_bank=True,
            option_scores=rows,
        )
        return {
            "mode": "classification",
            "question": q,
            "answer_key": rule_match.get("answer_option"),
            "answer_text": answer_text,
            "answer_pramana": answer_pramana,
            "confidence": verifier["confidence_decomposition"]["final_confidence"],
            "epistemic_status": verifier["belief_revision"]["final_status"],
            "option_scores": rows,
            "verifier": verifier,
            "trace": {
                "type": "neuro_symbolic_rule_bank_match",
                "rule_id": rule_match["id"],
                "matched_keywords": rule_match.get("matched_keywords", []),
                "explanation": rule_match.get("explanation", ""),
                "fallback_winner": winner,
                "neural_parser": {
                    "mode": "rule_and_keyword_parser",
                    "selected_answer_text": answer_text,
                },
                "symbolic_verifier": {
                    "verifier_pass": verifier["verifier_pass"],
                    "violated_constraints": verifier["violated_constraints"],
                },
            },
        }

    verifier = _run_symbolic_verifier(
        question=q,
        answer_text=winner["display_pramana"],
        citations=winner.get("citations", []),
        rule_match=None,
        used_rule_bank=False,
        option_scores=rows,
    )

    confidence = verifier["confidence_decomposition"]["final_confidence"]
    retrieval_support = verifier["confidence_decomposition"]["retrieval_support"]
    coverage = _citation_query_coverage(q, winner.get("citations", []))
    question_norm = _norm(q)
    conference_query = _is_conference_query(question_norm)
    has_domain_anchor = _has_domain_anchor(question_norm)
    conference_citation_found = any(
        str(c.get("id", "")).startswith("CONF-2026")
        for row in rows[:3]
        for c in row.get("citations", [])
    )
    uncertain = (
        confidence < 0.4
        or retrieval_support < 0.12
        or coverage < 0.12
        or not has_domain_anchor
        or (conference_query and not conference_citation_found)
        or verifier["belief_revision"]["final_status"] == "suspended"
    )
    answer_text = winner["display_pramana"]
    answer_pramana = winner["display_pramana"]
    final_status = verifier["belief_revision"]["final_status"]

    if uncertain:
        answer_text = "Insufficient supported evidence in current conference corpus."
        answer_pramana = "Unknown"
        final_status = "suspended"

    return {
        "mode": "classification",
        "question": q,
        "answer_key": None,
        "answer_text": answer_text,
        "answer_pramana": answer_pramana,
        "confidence": confidence,
        "epistemic_status": final_status,
        "option_scores": rows,
        "verifier": verifier,
        "trace": {
            "type": "neuro_symbolic_pramana_classification",
            "winning_row": winner,
            "abstained": uncertain,
            "neural_parser": {
                "mode": "classification_parser",
                "selected_answer_text": answer_text,
            },
            "symbolic_verifier": {
                "verifier_pass": verifier["verifier_pass"],
                "violated_constraints": verifier["violated_constraints"],
            },
        },
    }


def build_inference_mapping(question_result: Dict[str, Any]) -> Dict[str, Any]:
    answer = question_result["answer_pramana"]
    answer_norm = _norm(answer)

    if "pratyak" in answer_norm:
        pramana_types = ["Pratyaksha", "Anumana"]
        hetu = "direct cognition depends on immediate sense-object contact"
        udaharana = "seeing a pot through eye-object contact is pratyaksha"
    elif "anumana" in answer_norm:
        pramana_types = ["Anumana", "Pratyaksha"]
        hetu = "the claim depends on inferential sign and vyapti relation"
        udaharana = "smoke-fire relation on the hill and in the kitchen"
    elif "shabda" in answer_norm:
        pramana_types = ["Shabda", "Anumana"]
        hetu = "reliable apta testimony supports the proposition"
        udaharana = "trusted scriptural or expert testimony in stable tradition"
    elif "upamana" in answer_norm:
        pramana_types = ["Upamana", "Anumana"]
        hetu = "the target is known through similarity to a known case"
        udaharana = "new object recognized from an earlier comparison model"
    elif "aprama" in answer_norm:
        pramana_types = ["Anumana", "Shabda"]
        hetu = "the cognition is contradicted by stronger warrants and fails as valid knowledge"
        udaharana = "rope-snake illusion is later corrected and classified as aprama"
    else:
        pramana_types = ["Arthapatti", "Anumana"]
        hetu = "postulation is required to explain observed facts coherently"
        udaharana = "unseen cause postulated to explain otherwise inconsistent data"

    citations = []
    for row in question_result.get("option_scores", [])[:2]:
        for c in row.get("citations", [])[:2]:
            citations.append(c["id"])
    unique_citations = []
    for cid in citations:
        if cid not in unique_citations:
            unique_citations.append(cid)

    return {
        "paksha": "the asked knowledge claim",
        "sadhya": f"{answer} is the best-supported answer",
        "hetu": hetu,
        "udaharana": udaharana,
        "pramanaTypes": pramana_types,
        "combineAllPramanas": False,
        "citations": unique_citations,
    }
