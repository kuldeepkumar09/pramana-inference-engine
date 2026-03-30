from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .engine import PramanaInferenceEngine
from .models import Evidence, InferenceRequest, InferenceResult, Proposition, Rule


def _normalize_pramana(label: str) -> str:
    value = (label or "").strip().lower()
    mapping = {
        "pratyaksa": "perception",
        "pratyaksha": "perception",
        "pratyakṣa": "perception",
        "sabda": "testimony",
        "shabda": "testimony",
        "śabda": "testimony",
        "anumana": "inference",
        "anumāna": "inference",
        "upamana": "comparison",
        "upamāna": "comparison",
        "perception": "perception",
        "testimony": "testimony",
        "comparison": "comparison",
        "inference": "inference",
    }
    return mapping.get(value, "testimony")


def _normalize_mapping_payload(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    """Normalize question-solver mapping payloads into inference payload format.

    Supported shapes:
    - {"mapping": {...}, "question_result": {...}}
    - direct mapping-like dict with paksha/sadhya/hetu/pramanaTypes
    """
    if not isinstance(raw, dict):
        return None

    mapping = raw.get("mapping") if isinstance(raw.get("mapping"), dict) else raw
    if not isinstance(mapping, dict):
        return None

    paksha = str(mapping.get("paksha", "")).strip()
    sadhya = str(mapping.get("sadhya", "")).strip()
    hetu = str(mapping.get("hetu", "")).strip()
    if not (paksha and sadhya and hetu):
        return None

    pramana_types_raw = mapping.get("pramanaTypes", [])
    pramana_types: List[str] = []
    if isinstance(pramana_types_raw, list):
        for entry in pramana_types_raw:
            pramana_types.append(_normalize_pramana(str(entry)))

    if not pramana_types:
        answer_pramana = ""
        question_result = raw.get("question_result")
        if isinstance(question_result, dict):
            answer_pramana = str(question_result.get("answer_pramana", "")).strip()
        pramana_types = [_normalize_pramana(answer_pramana or "anumana")]

    required_pramanas = sorted(set(pramana_types))

    # Build a deterministic atom for the paksha+hetu observation.
    antecedent_claim = f"{hetu} @ {paksha}"

    return {
        "rules": [
            {
                "rule_id": "R1",
                "name": "Auto-adapted from question mapping",
                "pattern": "modus_ponens",
                "required_pramanas": required_pramanas,
                "min_reliability": 0.7,
                "suspension_margin": 0.05,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "proposition": {"kind": "atom", "value": antecedent_claim},
                "pramana": required_pramanas[0],
                "reliability": float(raw.get("hetuConf", 0.8)),
                "source": "question_mapping",
                "defeated": False,
                "metadata": {"adapted_from": "mapping.hetu"},
            },
            {
                "evidence_id": "E2",
                "proposition": {
                    "kind": "implies",
                    "antecedent": antecedent_claim,
                    "consequent": sadhya,
                },
                "pramana": required_pramanas[-1],
                "reliability": float(raw.get("vyaptiStr", 0.82)),
                "source": "question_mapping",
                "defeated": False,
                "metadata": {"adapted_from": "mapping.sadhya"},
            },
        ],
        "request": {
            "rule_id": "R1",
            "premise_evidence_ids": ["E1", "E2"],
            "target": {"kind": "atom", "value": sadhya},
        },
    }


def _normalize_corpus_payload(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        adapted = _normalize_mapping_payload(raw)
        if adapted is not None:
            return adapted

        # Common PDF-chunk export schema used for RAG ingestion, not infer-upload.
        if "chunks" in raw and "source_file" in raw:
            raise ValueError(
                "Detected RAG chunk payload (source_file/chunks). "
                "Use this file for RAG ingestion, not /api/infer-upload inference evaluation."
            )

        return raw

    if not isinstance(raw, list):
        return raw

    if not raw:
        raise ValueError("Input list is empty; cannot build inference payload.")

    first = raw[0]
    if not isinstance(first, dict):
        raise ValueError("Unsupported list payload shape.")

    proposition = first.get("proposition", {})
    evidence_items = first.get("evidence", [])
    if not isinstance(proposition, dict) or not isinstance(evidence_items, list) or not evidence_items:
        raise ValueError("List payload must contain proposition and non-empty evidence list.")

    target_claim = proposition.get("claim")
    lead_evidence = evidence_items[0]
    antecedent_claim = lead_evidence.get("claim")
    if not isinstance(target_claim, str) or not target_claim:
        raise ValueError("Proposition claim is missing in list payload.")
    if not isinstance(antecedent_claim, str) or not antecedent_claim:
        raise ValueError("At least one evidence claim is required in list payload.")

    premise_pramana = _normalize_pramana(str(lead_evidence.get("pramana_type", "testimony")))
    implication_pramana = _normalize_pramana(str(proposition.get("pramana_type", "testimony")))
    required = sorted({premise_pramana, implication_pramana})

    return {
        "rules": [
            {
                "rule_id": "R1",
                "name": "Auto-adapted from corpus extraction",
                "pattern": "modus_ponens",
                "required_pramanas": required,
                "min_reliability": 0.7,
                "suspension_margin": 0.05,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "proposition": {"kind": "atom", "value": antecedent_claim},
                "pramana": premise_pramana,
                "reliability": float(lead_evidence.get("confidence", 0.7)),
                "source": str(lead_evidence.get("source", "unknown")),
                "defeated": False,
                "metadata": {"adapted_from": "list_payload_evidence_0"},
            },
            {
                "evidence_id": "E2",
                "proposition": {
                    "kind": "implies",
                    "antecedent": antecedent_claim,
                    "consequent": target_claim,
                },
                "pramana": implication_pramana,
                "reliability": float(proposition.get("confidence", 0.7)),
                "source": str(proposition.get("source", "unknown")),
                "defeated": False,
                "metadata": {"adapted_from": "list_payload_proposition"},
            },
        ],
        "request": {
            "rule_id": "R1",
            "premise_evidence_ids": ["E1", "E2"],
            "target": {"kind": "atom", "value": target_claim},
        },
    }


def _normalize_corpus_item_payload(item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
    proposition = item.get("proposition", {})
    evidence_items = item.get("evidence", [])
    if not isinstance(proposition, dict) or not isinstance(evidence_items, list) or not evidence_items:
        raise ValueError(f"List payload item {item_index} must contain proposition and non-empty evidence list.")

    target_claim = proposition.get("claim")
    lead_evidence = evidence_items[0]
    antecedent_claim = lead_evidence.get("claim")
    if not isinstance(target_claim, str) or not target_claim:
        raise ValueError(f"List payload item {item_index} is missing proposition claim.")
    if not isinstance(antecedent_claim, str) or not antecedent_claim:
        raise ValueError(f"List payload item {item_index} has no usable evidence claim.")

    premise_pramana = _normalize_pramana(str(lead_evidence.get("pramana_type", "testimony")))
    implication_pramana = _normalize_pramana(str(proposition.get("pramana_type", "testimony")))
    required = sorted({premise_pramana, implication_pramana})

    return {
        "rules": [
            {
                "rule_id": "R1",
                "name": f"Auto-adapted corpus item {item_index}",
                "pattern": "modus_ponens",
                "required_pramanas": required,
                "min_reliability": 0.7,
                "suspension_margin": 0.05,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "proposition": {"kind": "atom", "value": antecedent_claim},
                "pramana": premise_pramana,
                "reliability": float(lead_evidence.get("confidence", 0.7)),
                "source": str(lead_evidence.get("source", "unknown")),
                "defeated": False,
                "metadata": {"adapted_from": f"list_payload_item_{item_index}_evidence_0"},
            },
            {
                "evidence_id": "E2",
                "proposition": {
                    "kind": "implies",
                    "antecedent": antecedent_claim,
                    "consequent": target_claim,
                },
                "pramana": implication_pramana,
                "reliability": float(proposition.get("confidence", 0.7)),
                "source": str(proposition.get("source", "unknown")),
                "defeated": False,
                "metadata": {"adapted_from": f"list_payload_item_{item_index}_proposition"},
            },
        ],
        "request": {
            "rule_id": "R1",
            "premise_evidence_ids": ["E1", "E2"],
            "target": {"kind": "atom", "value": target_claim},
        },
    }


def _validate_payload_dict(payload: Dict[str, Any]) -> None:
    required_keys = ["rules", "evidence", "request"]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(f"Payload is missing required keys: {missing}")
    if not isinstance(payload["rules"], list) or not payload["rules"]:
        raise ValueError("'rules' must be a non-empty list.")
    if not isinstance(payload["evidence"], list) or not payload["evidence"]:
        raise ValueError("'evidence' must be a non-empty list.")
    if not isinstance(payload["request"], dict):
        raise ValueError("'request' must be an object.")


def _proposition_from_dict(raw: Dict[str, Any]) -> Proposition:
    kind = raw.get("kind")
    if kind == "atom":
        value = raw.get("value")
        if not isinstance(value, str) or not value:
            raise ValueError("atom proposition requires non-empty 'value'.")
        return Proposition.atom(value)

    if kind == "implies":
        antecedent = raw.get("antecedent")
        consequent = raw.get("consequent")
        if not isinstance(antecedent, str) or not antecedent:
            raise ValueError("implies proposition requires non-empty 'antecedent'.")
        if not isinstance(consequent, str) or not consequent:
            raise ValueError("implies proposition requires non-empty 'consequent'.")
        return Proposition.implies(antecedent, consequent)

    raise ValueError("proposition 'kind' must be 'atom' or 'implies'.")


def _rule_from_dict(raw: Dict[str, Any]) -> Rule:
    return Rule(
        rule_id=raw["rule_id"],
        name=raw["name"],
        pattern=raw["pattern"],
        required_pramanas=list(raw["required_pramanas"]),
        min_reliability=float(raw.get("min_reliability", 0.7)),
        suspension_margin=float(raw.get("suspension_margin", 0.05)),
        pramana_weights=dict(raw.get("pramana_weights", {})),
        calibration_exponent=float(raw.get("calibration_exponent", 1.0)),
    )


def _evidence_from_dict(raw: Dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=raw["evidence_id"],
        proposition=_proposition_from_dict(raw["proposition"]),
        pramana=raw["pramana"],
        reliability=float(raw["reliability"]),
        source=raw["source"],
        defeated=bool(raw.get("defeated", False)),
        metadata=dict(raw.get("metadata", {})),
    )


def _request_from_dict(raw: Dict[str, Any]) -> InferenceRequest:
    return InferenceRequest(
        rule_id=raw["rule_id"],
        premise_evidence_ids=list(raw["premise_evidence_ids"]),
        target=_proposition_from_dict(raw["target"]),
    )


def infer_from_payload(payload: Any) -> InferenceResult:
    normalized = _normalize_corpus_payload(payload)
    _validate_payload_dict(normalized)
    rules = {_raw["rule_id"]: _rule_from_dict(_raw) for _raw in normalized["rules"]}
    evidence_base = {_raw["evidence_id"]: _evidence_from_dict(_raw) for _raw in normalized["evidence"]}
    request = _request_from_dict(normalized["request"])
    engine = PramanaInferenceEngine(evidence_base=evidence_base, rules=rules)
    return engine.infer(request)


def infer_from_file(file_path: str) -> InferenceResult:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return infer_from_payload(payload)


def infer_many_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        results: List[Dict[str, Any]] = []
        for idx, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            try:
                normalized_item = _normalize_corpus_item_payload(item, idx)
                result = infer_from_payload(normalized_item).to_dict()
                results.append({"index": idx, "ok": True, "result": result})
            except (ValueError, KeyError, TypeError) as exc:
                results.append({"index": idx, "ok": False, "error": str(exc)})
        return results

    # Standard dict payload still runs as a single-item batch for uniform reporting.
    result = infer_from_payload(payload).to_dict()
    return [{"index": 0, "ok": True, "result": result}]


def infer_many_from_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = infer_many_from_payload(payload)
    status_counts: Dict[str, int] = {"valid": 0, "unjustified": 0, "suspended": 0, "invalid": 0, "error": 0}
    for row in rows:
        if not row.get("ok"):
            status_counts["error"] += 1
            continue
        status = row["result"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "file": str(path),
        "total": len(rows),
        "status_counts": status_counts,
        "rows": rows,
    }
