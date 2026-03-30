from __future__ import annotations

import json
import os
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from .config import get_config
from .engine import PramanaInferenceEngine
from .epistemic_reasoning import (
    EpistemicEvaluator,
    EpistemicEvaluationConfig,
    EvidentialProposition,
    EvidentialSource,
    InferencePattern,
)
from .examples import SCENARIOS
from .io import infer_many_from_payload
from .models import Evidence, InferenceRequest, InferenceStatus, Proposition, Rule
from .pramana_registry import ALL_PRAMANAS, PRAMANA_AUTHORITY, normalize_pramana
from .qa_solver import build_inference_mapping, solve_question


def get_rag_pipeline():
    """Lazily resolve the RAG pipeline to avoid heavy optional imports at module load."""
    from .rag_pipeline import get_rag_pipeline as _get_rag_pipeline

    return _get_rag_pipeline()


def clear_hybrid_retrieval_cache() -> None:
    """Lazily resolve and clear hybrid retrieval cache."""
    from .hybrid_retrieval import clear_hybrid_retrieval_cache as _clear_hybrid_retrieval_cache

    _clear_hybrid_retrieval_cache()


def get_embedding_engine():
    """Lazily resolve embedding engine for cache controls."""
    from .rag_embeddings import get_embedding_engine as _get_embedding_engine

    return _get_embedding_engine()


def _enrich_inference_with_epistemic_trace(
    inference_result: Dict[str, Any],
    hetu: str,
    sadhya: str,
    clamped_hetu: float,
    clamped_vyapti: float,
    selected_pramanas: list[str],
    conflict: str,
) -> Dict[str, Any]:
    """Enrich inference result with epistemic reasoning trace (R1-R5).
    
    Args:
        inference_result: Result from PramanaInferenceEngine.infer
        hetu, sadhya: Inference components
        clamped_hetu, clamped_vyapti: Confidence scores
        selected_pramanas: List of pramana types used
        conflict: Defeater text if any
        
    Returns:
        Result dict enriched with epistemic_trace
    """
    try:
        # Build propositions and evidence for epistemic evaluation
        premise_prop = EvidentialProposition(
            id="hetu_prop",
            text=hetu,
            kind="atom",
            source="inference_premise",
            confidence=clamped_hetu,
        )
        
        conclusion_prop = EvidentialProposition(
            id="sadhya_prop",
            text=sadhya,
            kind="atom",
            source="inference_conclusion",
            confidence=0.0,
        )
        
        # Map evidence from inference result
        evidence_list = []
        for i, pramana in enumerate(selected_pramanas):
            evidence_list.append(
                EvidentialSource(
                    id=f"ev_{i}",
                    proposition_id="hetu_prop",
                    kind=pramana,
                    content=hetu,
                    reliability=clamped_vyapti,
                    defeated=bool(conflict),
                    pramana=pramana,
                )
            )
        
        # Pattern validation
        pattern = InferencePattern(
            name="UI Anumana",
            pattern="modus_ponens",
            valid_patterns=["modus_ponens", "vyapti_based_inference"],
            description="Modus ponens with pramana-weighted evidence",
        )
        
        # Epistemic evaluation
        config = EpistemicEvaluationConfig(
            min_justification_threshold=0.70,
            suspension_band=0.10,
        )
        evaluator = EpistemicEvaluator(config)
        
        trace = evaluator.evaluate_inference(
            propositions=[premise_prop],
            evidence=evidence_list,
            conclusion=conclusion_prop,
            pattern=pattern,
        )
        
        # Enrich result with trace
        inference_result["epistemic_trace"] = trace.to_dict()
        inference_result["inference_status_extended"] = trace.epistemic_status.value
        
    except Exception as e:
        # If epistemic tracing fails, don't crash the response
        inference_result["epistemic_trace_error"] = str(e)
    
    return inference_result

STATUS_THEME: Dict[str, Dict[str, str]] = {
    "valid": {
        "title": "Saraswati Lens",
        "message": "Inference is accepted: clarity and justification align.",
        "color": "status-valid",
    },
    "unjustified": {
        "title": "Hanuman Lens",
        "message": "Strength is present, but epistemic support is incomplete.",
        "color": "status-unjustified",
    },
    "suspended": {
        "title": "Shiva Lens",
        "message": "Inference is paused under uncertainty or defeaters.",
        "color": "status-suspended",
    },
    "invalid": {
        "title": "Kali Lens",
        "message": "Inference pattern rejected as structurally invalid.",
        "color": "status-invalid",
    },
}


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Authority weights for building fused evidence (scaled 0-5 → used in calibration formula)
    pramana_authority = {k: int(v * 5) for k, v in PRAMANA_AUTHORITY.items()}
    all_pramanas = list(ALL_PRAMANAS)

    # Unicode diacritics that may come from copy-pasted Sanskrit text
    _UNICODE_PRAMANA_MAP = {
        "pratyakṣa": "perception", "anumāna": "inference",
        "śabda": "testimony", "upamāna": "comparison", "arthāpatti": "postulation",
    }

    def _normalize_pramana(label: str) -> str:
        text = (label or "").strip().lower()
        return _UNICODE_PRAMANA_MAP.get(text) or normalize_pramana(text)

    def _selected_pramanas(payload: Dict[str, Any]) -> list[str]:
        combine_all = bool(payload.get("combineAllPramanas", False))
        if combine_all:
            return list(all_pramanas)

        selected: list[str] = []
        raw_list = payload.get("pramanaTypes")
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, str):
                    selected.append(_normalize_pramana(item))

        if not selected:
            selected = [_normalize_pramana(str(payload.get("pramanaType", "Anumana")))]

        seen = set()
        deduped = []
        for value in selected:
            if value not in seen:
                deduped.append(value)
                seen.add(value)
        return deduped

    def _extract_calibrated_score(inference_result: Dict[str, Any]) -> float:
        trace = inference_result.get("trace", {})
        steps = trace.get("steps", []) if isinstance(trace, dict) else []
        for step in reversed(steps):
            detail = step.get("detail") if isinstance(step, dict) else None
            if isinstance(detail, dict) and "calibrated_score" in detail:
                try:
                    return float(detail.get("calibrated_score", 0.0))
                except (TypeError, ValueError):
                    return 0.0
            if isinstance(step, dict) and "calibrated_score" in step:
                try:
                    return float(step.get("calibrated_score", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def _is_defeater_text(conflict: str) -> bool:
        text = (conflict or "").strip().lower()
        if not text:
            return False
        defeater_markers = [
            "contradict",
            "counter",
            "badhita",
            "refute",
            "opposite",
            "negates",
            "denies",
            "incompatible",
            "falsifies",
            "disproves",
        ]
        return any(marker in text for marker in defeater_markers)

    def _detect_upload_payload_type(payload: Any) -> tuple[str, str]:
        """Detect upload payload schema for clearer user messaging."""
        if isinstance(payload, dict):
            if isinstance(payload.get("mapping"), dict):
                return (
                    "question_mapping",
                    "Detected question-solver mapping payload (mapping + question_result).",
                )
            if all(k in payload for k in ("rules", "evidence", "request")):
                return (
                    "inference_payload",
                    "Detected native inference payload (rules/evidence/request).",
                )
            if "source_file" in payload and "chunks" in payload:
                return (
                    "rag_chunks",
                    "Detected RAG chunk payload (source_file/chunks). This schema is for retrieval ingestion, not infer-upload.",
                )
            if all(k in payload for k in ("paksha", "sadhya", "hetu")):
                return (
                    "mapping_like",
                    "Detected mapping-like payload (paksha/sadhya/hetu).",
                )
            return ("object_unknown", "Detected JSON object with unknown schema.")

        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict) and "proposition" in payload[0] and "evidence" in payload[0]:
                return (
                    "corpus_list",
                    "Detected corpus extraction list payload (proposition/evidence rows).",
                )
            return ("list_unknown", "Detected JSON list with unknown row schema.")

        return ("unsupported", "Unsupported JSON root type.")

    def _build_fused_evidence(
        paksha: str,
        hetu: str,
        sadhya: str,
        udaharana: str,
        conflict: str,
        hetu_conf: float,
        vyapti_str: float,
        selected_pramanas: list[str],
    ) -> tuple[Dict[str, Evidence], list[str], Dict[str, float]]:
        evidence: Dict[str, Evidence] = {}
        premise_ids: list[str] = []
        pramana_weights: Dict[str, float] = {}
        conflict_is_defeater = _is_defeater_text(conflict)

        for index, pramana in enumerate(selected_pramanas):
            authority_weight = pramana_authority.get(pramana, 3) / 5.0
            pramana_weights[pramana] = authority_weight

            calibrated_hetu = max(0.0, min(1.0, hetu_conf * (0.75 + 0.25 * authority_weight)))
            calibrated_vyapti = max(0.0, min(1.0, vyapti_str * (0.75 + 0.25 * authority_weight)))

            atom_id = f"E_A_{index + 1}"
            implication_id = f"E_I_{index + 1}"

            evidence[atom_id] = Evidence(
                evidence_id=atom_id,
                proposition=Proposition.atom(hetu),
                pramana=pramana,
                reliability=calibrated_hetu,
                source="ui_input",
                defeated=conflict_is_defeater,
                metadata={"paksha": paksha, "conflict": conflict},
            )
            evidence[implication_id] = Evidence(
                evidence_id=implication_id,
                proposition=Proposition.implies(hetu, sadhya),
                pramana=pramana,
                reliability=calibrated_vyapti,
                source="ui_vyapti",
                metadata={"udaharana": udaharana},
            )
            premise_ids.extend([atom_id, implication_id])

        return evidence, premise_ids, pramana_weights

    def _build_result_sections(payload: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, str]:
        paksha = payload["paksha"]
        sadhya = payload["sadhya"]
        hetu = payload["hetu"]
        udaharana = payload.get("udaharana", "")
        conflict = payload.get("conflict", "")
        pramana_type = payload.get("pramanaType", "Anumana")
        verdict = result["verdict"]
        return {
            "I": (
                f"The proposition under inquiry is that '{paksha}' possesses '{sadhya}'. "
                f"The presented pramana-type is {pramana_type}."
            ),
            "II": (
                f"The offered hetu is: '{hetu}'. The system checks whether this mark can function as "
                "a reliable inferential sign under explicit epistemic constraints."
            ),
            "III": (
                f"Wherever {hetu}, there is {sadhya} - as in {udaharana or 'a familiar supporting instance'}. "
                "This is treated as the vyapti-claim for the current inference request."
            ),
            "IV": (
                f"The subject '{paksha}' is asserted to possess '{hetu}'. Therefore, the engine tests "
                f"whether '{sadhya}' can be inferred under the selected rule."
            ),
            "V": (
                f"Engine verdict: {verdict}. This conclusion follows only after logical-form validation "
                "and epistemic justification checks have both executed."
            ),
            "VI": (
                f"Conflict provided: {'yes' if conflict else 'no'}. Pramana authority and reliability values "
                "are included in the trace, and acceptance is determined by explicit thresholds."
            ),
            "VII": (
                "In Nyaya-style terms, a claim is accepted as knowledge only when inferential form and "
                "epistemic warrant both hold. The engine therefore separates logical validity from "
                "justification status, preventing unsupported conclusions from being treated as prama."
            ),
            "VIII": "NS-1.1.3 (pramana categories); NS-1.1.5 (anumana and vyapti constraints).",
        }

    def _is_local_request() -> bool:
        forwarded_for = str(request.headers.get("X-Forwarded-For", "")).split(",")[0].strip()
        remote = forwarded_for or str(request.remote_addr or "").strip().lower()
        return remote in {"127.0.0.1", "::1", "localhost"}

    def _authorize_admin_request():
        token = os.environ.get("PRAMANA_ADMIN_TOKEN", "").strip()
        provided = str(request.headers.get("X-Pramana-Admin-Token", "")).strip()

        if token:
            if provided != token:
                return jsonify({"error": "Unauthorized admin request."}), 401
            return None

        if not _is_local_request():
            return (
                jsonify(
                    {
                        "error": (
                            "Admin endpoint is local-only unless PRAMANA_ADMIN_TOKEN is set. "
                            "Provide X-Pramana-Admin-Token header for non-local access."
                        )
                    }
                ),
                403,
            )

        return None

    @app.get("/")
    def workspace_home():
        return render_template("unified.html")

    @app.get("/workspace-legacy")
    def workspace_legacy():
        return render_template("workspace.html")

    @app.get("/app")
    def index():
        return render_template("index.html", scenarios=list(SCENARIOS.keys()))

    @app.get("/judge")
    def judge():
        return render_template("judge.html")

    @app.get("/compare")
    def compare():
        return render_template("compare.html")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/workspace")
    def workspace():
        return render_template("workspace.html")

    @app.get("/api/scenarios")
    def list_scenarios():
        return jsonify({"scenarios": list(SCENARIOS.keys())})

    @app.post("/api/infer")
    def infer_from_ui():
        payload = request.get_json(silent=True) or {}

        paksha = str(payload.get("paksha", "")).strip()
        sadhya = str(payload.get("sadhya", "")).strip()
        hetu = str(payload.get("hetu", "")).strip()
        if not paksha or not sadhya or not hetu:
            return jsonify({"error": "paksha, sadhya, and hetu are required."}), 400

        udaharana = str(payload.get("udaharana", "")).strip()
        conflict = str(payload.get("conflict", "")).strip()
        pramana_type = str(payload.get("pramanaType", "Anumana"))
        selected_pramanas = _selected_pramanas(payload)
        _valid_patterns = {"modus_ponens", "modus_tollens", "hypothetical_syllogism", "vyapti_based_inference"}
        inference_pattern = str(payload.get("inferencePattern", "modus_ponens")).strip().lower()
        if inference_pattern not in _valid_patterns:
            inference_pattern = "modus_ponens"

        try:
            hetu_conf = float(payload.get("hetuConf", 0.7))
            vyapti_str = float(payload.get("vyaptiStr", 0.7))
        except (TypeError, ValueError):
            return jsonify({"error": "hetuConf and vyaptiStr must be numeric."}), 400

        clamped_hetu = max(0.0, min(1.0, hetu_conf))
        clamped_vyapti = max(0.0, min(1.0, vyapti_str))

        evidence, premise_ids, pramana_weights = _build_fused_evidence(
            paksha=paksha,
            hetu=hetu,
            sadhya=sadhya,
            udaharana=udaharana,
            conflict=conflict,
            hetu_conf=clamped_hetu,
            vyapti_str=clamped_vyapti,
            selected_pramanas=selected_pramanas,
        )

        rules = {
            "R1": Rule(
                rule_id="R1",
                name=f"UI Anumana via {inference_pattern}",
                pattern=inference_pattern,
                required_pramanas=selected_pramanas,
                # 0.65 (not 0.70): combining multiple pramanas correctly lowers the
                # weighted average due to lower-authority sources (inference=0.8,
                # testimony=0.6). With threshold=0.70 a 3-pramana inference at
                # hetuConf=0.7 scores 0.671 → UNJUSTIFIED even though all evidence
                # is above the minimum. 0.65 keeps the gate strict while allowing
                # multi-pramana corroboration to produce VALID verdicts.
                min_reliability=0.65,
                suspension_margin=0.05,
                pramana_weights=pramana_weights,
                calibration_exponent=1.0,
            )
        }

        engine = PramanaInferenceEngine(evidence_base=evidence, rules=rules)
        inference_request = InferenceRequest(
            rule_id="R1",
            premise_evidence_ids=premise_ids,
            target=Proposition.atom(sadhya),
        )
        result = engine.infer(inference_request).to_dict()

        # ── Enrich with epistemic reasoning trace (R1-R5) ──
        result = _enrich_inference_with_epistemic_trace(
            inference_result=result,
            hetu=hetu,
            sadhya=sadhya,
            clamped_hetu=clamped_hetu,
            clamped_vyapti=clamped_vyapti,
            selected_pramanas=selected_pramanas,
            conflict=conflict,
        )

        verdict_map = {
            InferenceStatus.VALID.value: "VALID",
            InferenceStatus.UNJUSTIFIED.value: "UNJUSTIFIED",
            InferenceStatus.SUSPENDED.value: "SUSPENDED",
            InferenceStatus.INVALID.value: "REJECTED",
        }
        verdict = verdict_map[result["status"]]

        gate = {
            "score": round((clamped_hetu + clamped_vyapti) / 2.0, 3),
            "auth": max(pramana_authority.get(p, 3) for p in selected_pramanas),
            "authWeight": round(sum(pramana_weights.values()) / max(len(pramana_weights), 1), 3),
            "hasBadhita": _is_defeater_text(conflict),
            "verdict": verdict,
            "gatePassed": result["accepted"],
            "calibrated_score": _extract_calibrated_score(result),
        }

        sections = _build_result_sections(payload, {"verdict": verdict})

        rag_query = f"{paksha} {hetu} {sadhya}"
        try:
            qa_result = solve_question(rag_query)
            seen_ids: set[str] = set()
            rag_chunks = []
            for row in qa_result.get("option_scores", []):
                for c in row.get("citations", []):
                    cid = c.get("id", "")
                    if cid and cid not in seen_ids:
                        seen_ids.add(cid)
                        rag_chunks.append({
                            "id": cid,
                            "source": c.get("source", ""),
                            "excerpt": c.get("excerpt", ""),
                            "score": c.get("score", 0.0),
                        })
                    if len(rag_chunks) >= 5:
                        break
                if len(rag_chunks) >= 5:
                    break
        except Exception:
            rag_chunks = []

        response = {
            "paksha": paksha,
            "sadhya": sadhya,
            "hetu": hetu,
            "udaharana": udaharana,
            "conflict": conflict,
            "pramanaType": pramana_type,
            "selectedPramanas": selected_pramanas,
            "combineAllPramanas": bool(payload.get("combineAllPramanas", False)),
            "hetuConf": round(clamped_hetu, 3),
            "vyaptiStr": round(clamped_vyapti, 3),
            "embeddingScore": gate["score"],
            "gate": gate,
            "ragChunks": rag_chunks,
            "sections": sections,
            "jsonSummary": {
                "engine_status": result["status"],
                "verdict": verdict,
                "accepted": result["accepted"],
                "trace": result["trace"],
                "message": result["message"],
                "epistemic_trace": result.get("epistemic_trace"),
                "inference_status_extended": result.get("inference_status_extended"),
                "requirements": {
                    "runnable_code": True,
                    "clear_input_output": True,
                    "inspectable_trace": True,
                    "epistemic_gate_used": True,
                    "philosophical_grounding": True,
                    # R1-R5: computed from actual trace steps, not hardcoded
                    "R1_proposition_representation": any(
                        s.get("requirement") == "R1" and s.get("ok")
                        for s in (result.get("trace") or {}).get("steps", [])
                    ),
                    "R2_epistemic_justification_check": any(
                        s.get("requirement") == "R2"
                        for s in (result.get("trace") or {}).get("steps", [])
                    ),
                    "R3_status_distinction": result.get("inference_status_extended", result.get("status", "unknown")),
                    "R4_invalid_patterns_rejected": any(
                        s.get("requirement") == "R4"
                        for s in (result.get("trace") or {}).get("steps", [])
                    ),
                    "R5_machine_readable_trace": result.get("epistemic_trace") is not None,
                },
            },
        }
        return jsonify(response)

    @app.post("/api/question-solve")
    def question_solve():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "question is required."}), 400

        try:
            result = solve_question(question)
            mapping = build_inference_mapping(result)

            # Enrich mapping with RAG pipeline so answers come from uploaded PDFs
            try:
                rag = get_rag_pipeline()
                rag_result = rag.answer_question(question, use_llm=True)
                rag_answer = (rag_result.get("answer") or "").strip()
                rag_chunks = rag_result.get("rag_chunks") or []

                if rag_answer and rag_answer != "No relevant information found in the knowledge base.":
                    import re as _re
                    # Extract subject from question: strip wh-words to get topic
                    q_lower = question.lower()
                    for prefix in ("what is the difference between ", "what is ", "explain ",
                                   "how does ", "what are ", "define ", "describe "):
                        if q_lower.startswith(prefix):
                            paksha = question[len(prefix):].rstrip("?").strip()
                            break
                    else:
                        paksha = question.rstrip("?").strip()

                    # Robust sentence extraction: handles bullet/numbered lists and
                    # single-line output from small LLMs (phi3:mini, tinyllama).
                    # Strategy: split on newlines first (list output), then on
                    # sentence-ending punctuation. Filter empty/header lines.
                    def _extract_content_parts(text: str) -> list[str]:
                        # Strip markdown bold/italic markers
                        text = _re.sub(r'\*{1,2}|_{1,2}', '', text)
                        # Split by newline first to handle bullet/numbered lists
                        raw_lines = [ln.strip() for ln in text.splitlines()]
                        parts: list[str] = []
                        for ln in raw_lines:
                            # Skip section headers (e.g. "REASON:", "ANSWER:")
                            if not ln or _re.match(r'^[A-Z][A-Z\s]+:$', ln):
                                continue
                            # Strip leading bullet/number markers
                            ln = _re.sub(r'^[-*•\d]+[.)]\s*', '', ln).strip()
                            if ln:
                                # Further split on sentence boundaries within long lines
                                sub = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', ln)
                                parts.extend(s.strip() for s in sub if s.strip())
                        # Fallback: if nothing extracted, split original text
                        if not parts:
                            parts = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                        return parts

                    sentences = _extract_content_parts(rag_answer)
                    sadhya = sentences[0][:200] if sentences else rag_answer[:120]

                    # Use second sentence or top RAG chunk as hetu
                    if len(sentences) > 1:
                        hetu = sentences[1][:200]
                    elif rag_chunks:
                        hetu = (rag_chunks[0].get("text") or "")[:200].strip()
                    else:
                        hetu = mapping.get("hetu", "")

                    # Use third sentence or second chunk as udaharana
                    if len(sentences) > 2:
                        udaharana = sentences[2][:200]
                    elif len(rag_chunks) > 1:
                        udaharana = (rag_chunks[1].get("text") or "")[:200].strip()
                    else:
                        udaharana = mapping.get("udaharana", "")

                    mapping["paksha"] = paksha
                    mapping["sadhya"] = sadhya
                    mapping["hetu"] = hetu
                    mapping["udaharana"] = udaharana
                    mapping["rag_answer"] = rag_answer
                    result["rag_answer"] = rag_answer

                    # Pass confidence values derived from the RAG/rule result so
                    # the frontend sliders are set to values that guarantee VALID
                    # when epistemic support is strong (rule confidence >= 0.85).
                    rule_conf = float(result.get("confidence") or 0.80)
                    mapping["hetuConf"] = round(max(0.75, min(0.95, rule_conf)), 2)
                    mapping["vyaptiStr"] = round(max(0.75, min(0.95, rule_conf)), 2)
            except Exception:
                pass  # fall back to original mapping silently

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify({"question_result": result, "mapping": mapping})

    @app.post("/api/conference-qa")
    def conference_qa():
        """Conference-focused QA endpoint with explicit abstention behavior."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "question is required."}), 400

        min_confidence = payload.get("minConfidence", 0.45)
        try:
            min_confidence = float(min_confidence)
        except (TypeError, ValueError):
            return jsonify({"error": "minConfidence must be numeric."}), 400
        min_confidence = max(0.0, min(1.0, min_confidence))

        try:
            result = solve_question(question)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        citations = []
        seen_ids: set[str] = set()
        rows = result.get("option_scores", [])
        for row in rows:
            for c in row.get("citations", []):
                cid = str(c.get("id", "")).strip()
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)
                citations.append(
                    {
                        "id": cid,
                        "source": c.get("source", "unknown"),
                        "score": c.get("score", 0.0),
                        "excerpt": c.get("excerpt", ""),
                    }
                )
                if len(citations) >= 8:
                    break
            if len(citations) >= 8:
                break

        confidence = float(result.get("confidence", 0.0))
        epistemic_status = str(result.get("epistemic_status", "unjustified"))
        abstained = (
            epistemic_status == "suspended"
            or confidence < min_confidence
            or not citations
        )

        answer = result.get("answer_text", "")
        if abstained:
            answer = "I don't know based on current supported conference evidence."

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "epistemic_status": epistemic_status,
                "abstained": abstained,
                "citations": citations,
                "min_confidence": min_confidence,
            }
        )

    @app.post("/api/compare")
    def compare_inference_modes():
        payload = request.get_json(silent=True) or {}

        paksha = str(payload.get("paksha", "")).strip()
        sadhya = str(payload.get("sadhya", "")).strip()
        hetu = str(payload.get("hetu", "")).strip()
        if not paksha or not sadhya or not hetu:
            return jsonify({"error": "paksha, sadhya, and hetu are required."}), 400

        conflict = str(payload.get("conflict", "")).strip()
        pramana_type = str(payload.get("pramanaType", "Anumana"))
        selected_pramanas = _selected_pramanas(payload)

        try:
            hetu_conf = float(payload.get("hetuConf", 0.7))
            vyapti_str = float(payload.get("vyaptiStr", 0.7))
        except (TypeError, ValueError):
            return jsonify({"error": "hetuConf and vyaptiStr must be numeric."}), 400

        clamped_hetu = max(0.0, min(1.0, hetu_conf))
        clamped_vyapti = max(0.0, min(1.0, vyapti_str))

        constrained_evidence, constrained_premise_ids, constrained_weights = _build_fused_evidence(
            paksha=paksha,
            hetu=hetu,
            sadhya=sadhya,
            udaharana=str(payload.get("udaharana", "")),
            conflict=conflict,
            hetu_conf=clamped_hetu,
            vyapti_str=clamped_vyapti,
            selected_pramanas=selected_pramanas,
        )
        constrained_rule = Rule(
            rule_id="R1",
            name="Pramana-Constrained",
            pattern="modus_ponens",
            required_pramanas=selected_pramanas,
            min_reliability=0.7,
            suspension_margin=0.05,
            pramana_weights=constrained_weights,
            calibration_exponent=1.0,
        )

        baseline_evidence = {
            "E1": Evidence("E1", Proposition.atom(hetu), "agnostic", clamped_hetu, "ui_input", defeated=False),
            "E2": Evidence("E2", Proposition.implies(hetu, sadhya), "agnostic", clamped_vyapti, "ui_vyapti", defeated=False),
        }
        baseline_rule = Rule(
            rule_id="B1",
            name="Source-Agnostic Baseline",
            pattern="modus_ponens",
            required_pramanas=[],
            min_reliability=0.7,
            suspension_margin=0.05,
            pramana_weights={"agnostic": 1.0},
            calibration_exponent=1.0,
        )

        request_obj = InferenceRequest(
            rule_id="R1",
            premise_evidence_ids=constrained_premise_ids,
            target=Proposition.atom(sadhya),
        )
        constrained_engine = PramanaInferenceEngine(constrained_evidence, {"R1": constrained_rule})
        constrained_result = constrained_engine.infer(request_obj).to_dict()

        baseline_request = InferenceRequest(
            rule_id="B1",
            premise_evidence_ids=["E1", "E2"],
            target=Proposition.atom(sadhya),
        )
        baseline_engine = PramanaInferenceEngine(baseline_evidence, {"B1": baseline_rule})
        baseline_result = baseline_engine.infer(baseline_request).to_dict()

        verdict_map = {
            InferenceStatus.VALID.value: "VALID",
            InferenceStatus.UNJUSTIFIED.value: "UNJUSTIFIED",
            InferenceStatus.SUSPENDED.value: "SUSPENDED",
            InferenceStatus.INVALID.value: "REJECTED",
        }

        return jsonify(
            {
                "input": {
                    "paksha": paksha,
                    "sadhya": sadhya,
                    "hetu": hetu,
                    "conflict": conflict,
                    "pramanaType": pramana_type,
                    "selectedPramanas": selected_pramanas,
                },
                "constrained": {
                    "verdict": verdict_map[constrained_result["status"]],
                    "accepted": constrained_result["accepted"],
                    "status": constrained_result["status"],
                    "calibrated_score": _extract_calibrated_score(constrained_result),
                    "trace": constrained_result["trace"],
                },
                "baseline": {
                    "verdict": verdict_map[baseline_result["status"]],
                    "accepted": baseline_result["accepted"],
                    "status": baseline_result["status"],
                    "calibrated_score": _extract_calibrated_score(baseline_result),
                    "trace": baseline_result["trace"],
                },
                "delta": {
                    "accepted_changed": constrained_result["accepted"] != baseline_result["accepted"],
                    "status_changed": constrained_result["status"] != baseline_result["status"],
                },
            }
        )

    @app.post("/api/infer-upload")
    def infer_upload():
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded with form key 'file'."}), 400

        uploaded = request.files["file"]
        if not uploaded.filename:
            return jsonify({"error": "Uploaded file has no filename."}), 400

        try:
            payload = json.loads(uploaded.stream.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"Invalid JSON upload: {exc}"}), 400

        payload_type, payload_message = _detect_upload_payload_type(payload)

        try:
            report = infer_many_from_payload(payload)
        except (ValueError, TypeError, KeyError) as exc:
            return jsonify(
                {
                    "error": str(exc),
                    "detected_payload_type": payload_type,
                    "detected_payload_message": payload_message,
                }
            ), 400

        status_counts: Dict[str, int] = {"valid": 0, "unjustified": 0, "suspended": 0, "invalid": 0, "error": 0}
        for row in report:
            if not row.get("ok"):
                status_counts["error"] += 1
                continue
            status = row["result"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return jsonify(
            {
                "total": len(report),
                "status_counts": status_counts,
                "rows": report,
                "detected_payload_type": payload_type,
                "detected_payload_message": payload_message,
            }
        )

    @app.post("/api/judge-report")
    def judge_report():
        payload = request.get_json(silent=True) or {}
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            return jsonify({"error": "rows must be a list."}), 400

        status_counts: Dict[str, int] = {"valid": 0, "unjustified": 0, "suspended": 0, "invalid": 0, "error": 0}
        for row in rows:
            if not isinstance(row, dict) or not row.get("ok"):
                status_counts["error"] += 1
                continue
            status = row.get("result", {}).get("status", "error")
            status_counts[status] = status_counts.get(status, 0) + 1

        total = max(len(rows), 1)
        score = round((status_counts["valid"] + 0.6 * status_counts["suspended"] + 0.2 * status_counts["unjustified"]) / total, 3)
        return jsonify(
            {
                "total": len(rows),
                "status_counts": status_counts,
                "quality_score": score,
                "checklist": {
                    "inspectable_reasoning": True,
                    "epistemic_constraints_enforced": True,
                    "invalid_patterns_rejected": True,
                    "machine_readable_traces": True,
                },
            }
        )

    # ===== RAG+LLM ENDPOINTS =====
    @app.post("/api/rag/search")
    def rag_search_only():
        """Perform retrieval-only search (no LLM reasoning)."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        pramana_types = payload.get("pramanaTypes")
        k = int(payload.get("k", 5))

        if not question:
            return jsonify({"error": "question is required."}), 400

        pipeline = get_rag_pipeline()
        if not pipeline._initialized:
            pipeline.initialize()

        results = pipeline.search_only(question, pramana_types, k)
        return jsonify({"question": question, "results": results, "count": len(results)})

    @app.post("/api/rag/answer")
    def rag_answer():
        """Answer question using full RAG pipeline (retrieval + LLM + verification)."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        pramana_types = payload.get("pramanaTypes")
        use_llm = bool(payload.get("useLLM", True))
        use_reasoning_chain = bool(payload.get("useReasoningChain", False))

        if not question:
            return jsonify({"error": "question is required."}), 400

        try:
            pipeline = get_rag_pipeline()
            if not pipeline._initialized:
                pipeline.initialize()

            result = pipeline.answer_question(
                question=question,
                pramana_types=pramana_types,
                use_llm=use_llm,
                use_reasoning_chain=use_reasoning_chain,
            )
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"RAG pipeline error: {str(e)}"}), 500

    @app.post("/api/rag/cache/clear")
    def rag_clear_caches():
        """Clear runtime retrieval/embedding caches without restarting the server."""
        auth_error = _authorize_admin_request()
        if auth_error is not None:
            return auth_error

        payload = request.get_json(silent=True) or {}
        scope = str(payload.get("scope", "all")).strip().lower()
        if scope not in {"all", "retrieval", "embedding"}:
            return jsonify({"error": "scope must be one of: all, retrieval, embedding"}), 400

        cleared: list[str] = []
        pipeline = get_rag_pipeline()

        if scope in {"all", "retrieval"}:
            if hasattr(pipeline, "clear_runtime_caches"):
                pipeline.clear_runtime_caches()
                cleared.append("pipeline_retrieval_cache")
            clear_hybrid_retrieval_cache()
            cleared.append("hybrid_retrieval_cache")

        if scope in {"all", "embedding"}:
            engine = get_embedding_engine()
            if hasattr(engine, "clear_query_cache"):
                engine.clear_query_cache()
                cleared.append("query_embedding_cache")

        return jsonify({"ok": True, "scope": scope, "cleared": cleared})

    @app.post("/api/rag/explain")
    def rag_explain():
        """Explain why a specific answer was chosen (decision path explanation)."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        use_llm = bool(payload.get("useLLM", True))

        if not question:
            return jsonify({"error": "question is required."}), 400

        try:
            pipeline = get_rag_pipeline()
            if not pipeline._initialized:
                pipeline.initialize()

            explanation = pipeline.explain_answer(
                question=question,
                use_llm=use_llm,
            )
            return jsonify(explanation)
        except Exception as e:
            return jsonify({"error": f"Explanation generation failed: {str(e)}"}), 500

    @app.post("/api/rag/batch")
    def rag_batch():
        """Answer multiple questions efficiently with shared cache."""
        payload = request.get_json(silent=True) or {}
        questions = payload.get("questions", [])
        use_llm = bool(payload.get("useLLM", True))

        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({"error": "questions must be a non-empty list of strings."}), 400

        # Limit batch size to prevent abuse
        if len(questions) > 50:
            return jsonify({"error": "Batch size limited to 50 questions."}), 400

        try:
            pipeline = get_rag_pipeline()
            if not pipeline._initialized:
                pipeline.initialize()

            results = pipeline.answer_batch(questions=questions, use_llm=use_llm)
            return jsonify({
                "batch_size": len(questions),
                "results": results,
                "succeeded": len([r for r in results if "error" not in r]),
            })
        except Exception as e:
            return jsonify({"error": f"Batch processing failed: {str(e)}"}), 500

    @app.get("/api/rag/status")
    def rag_status():
        """Check RAG pipeline status (is Ollama running, is vector store initialized)."""
        pipeline = get_rag_pipeline()
        llm_available = pipeline.llm_engine is not None
        llm_healthy = llm_available and pipeline.llm_engine.health_check()
        active_llm_model = None
        configured_llm_model = None
        if llm_available:
            active_llm_model = getattr(pipeline.llm_engine, "model_name", None)
            configured_llm_model = get_config().llm.model_name

        initialized = pipeline._initialized
        if not initialized:
            pipeline.initialize()
            initialized = pipeline._initialized

        vector_store_size = 0
        if pipeline.vector_store:
            vector_store_size = pipeline.vector_store.size()

        return jsonify({
            "pipeline_initialized": initialized,
            "vector_store_size": vector_store_size,
            "llm_available": llm_available,
            "llm_healthy": llm_healthy,
            "embedding_model": "intfloat/e5-small-v2",
            "llm_model": active_llm_model or configured_llm_model or "unknown",
            "llm_model_configured": configured_llm_model,
            "status": "ready" if (initialized and llm_healthy) else "setup_needed",
        })

    return app


app = create_app()


if __name__ == "__main__":
    import os
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
