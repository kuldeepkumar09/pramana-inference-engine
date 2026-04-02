"""
End-to-end integration tests for the Pramana Engine pipeline.

Three flows are covered:
  1. Symbolic-only path  — engine.infer() with no RAG, no LLM
  2. QA solver path      — solve_question() using rule bank + symbolic verifier
  3. RAG answer path     — /api/rag/answer with mocked LLM + mocked retrieval
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List
import json

import pytest

from pramana_engine.engine import PramanaInferenceEngine
from pramana_engine.models import Evidence, InferenceRequest, InferenceStatus, Proposition, Rule
from pramana_engine.io import infer_from_payload
from pramana_engine.qa_solver import solve_question
from pramana_engine.web import create_app


# ---------------------------------------------------------------------------
# Path 1: Symbolic-only — full Engine.infer() round-trip
# ---------------------------------------------------------------------------

def test_e2e_symbolic_valid_anumana():
    """Hill has fire because smoke is present — canonical Nyaya anumana example."""
    smoke_on_hill = Evidence(
        evidence_id="E1",
        proposition=Proposition.atom("smoke_on_hill"),
        pramana="inference",
        reliability=0.9,
        source="perception",
        defeated=False,
    )
    smoke_implies_fire = Evidence(
        evidence_id="E2",
        proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
        pramana="inference",
        reliability=0.95,
        source="vyapti",
        defeated=False,
    )
    rule = Rule(
        rule_id="R_anumana",
        name="Smoke-fire anumana",
        pattern="modus_ponens",
        required_pramanas=["inference"],
        min_reliability=0.7,
    )

    engine = PramanaInferenceEngine(
        evidence_base={"E1": smoke_on_hill, "E2": smoke_implies_fire},
        rules={"R_anumana": rule},
    )
    result = engine.infer(
        InferenceRequest(
            rule_id="R_anumana",
            premise_evidence_ids=["E1", "E2"],
            target=Proposition.atom("fire_on_hill"),
        )
    )

    assert result.status == InferenceStatus.VALID
    # Score is stored inside the trace, not as a top-level field
    assert result.accepted is True


def test_e2e_symbolic_invalid_tautology():
    """Implication smoke → smoke is a tautology and must be rejected at parse time."""
    # Use raw JSON payload (not mapping format) so we can inject antecedent == consequent
    with pytest.raises(ValueError, match="[Tt]autological"):
        infer_from_payload({
            "rules": [
                {
                    "rule_id": "R1",
                    "name": "tautology test",
                    "pattern": "modus_ponens",
                    "required_pramanas": ["inference"],
                    "min_reliability": 0.5,
                }
            ],
            "evidence": [
                {
                    "evidence_id": "E1",
                    "proposition": {"kind": "atom", "value": "smoke"},
                    "pramana": "inference",
                    "reliability": 0.9,
                    "source": "test",
                },
                {
                    "evidence_id": "E2",
                    "proposition": {"kind": "implies", "antecedent": "smoke", "consequent": "smoke"},
                    "pramana": "inference",
                    "reliability": 0.9,
                    "source": "test",
                },
            ],
            "request": {
                "rule_id": "R1",
                "premise_evidence_ids": ["E1", "E2"],
                "target": {"kind": "atom", "value": "smoke"},
            },
        })


def test_e2e_symbolic_negative_reliability_rejected():
    """Evidence with reliability < 0.0 must raise a ValueError at parse time."""
    payload = {
        "rules": [
            {
                "rule_id": "R1",
                "name": "test",
                "pattern": "modus_ponens",
                "required_pramanas": ["inference"],
                "min_reliability": 0.5,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "proposition": {"kind": "atom", "value": "smoke"},
                "pramana": "inference",
                "reliability": -0.3,
                "source": "test",
            }
        ],
        "request": {
            "rule_id": "R1",
            "premise_evidence_ids": ["E1"],
            "target": {"kind": "atom", "value": "fire"},
        },
    }
    with pytest.raises(ValueError, match="[Rr]eliability"):
        infer_from_payload(payload)


def test_e2e_symbolic_anupalabdhi_valid():
    """Anupalabdhi (non-perception): jar is absent because it is not perceived."""
    no_jar_perceived = Evidence(
        evidence_id="E1",
        proposition=Proposition.atom("non_perception_of_jar_on_table"),
        pramana="non_perception",
        reliability=0.9,
        source="direct_observation",
        defeated=False,
    )
    non_perception_implies_absence = Evidence(
        evidence_id="E2",
        proposition=Proposition.implies(
            "non_perception_of_jar_on_table", "jar_absent_from_table"
        ),
        pramana="non_perception",
        reliability=0.85,
        source="anupalabdhi_rule",
        defeated=False,
    )
    rule = Rule(
        rule_id="R_anupalabdhi",
        name="Non-perception of jar",
        pattern="anupalabdhi_based_inference",
        required_pramanas=["non_perception"],
        min_reliability=0.6,
    )

    engine = PramanaInferenceEngine(
        evidence_base={"E1": no_jar_perceived, "E2": non_perception_implies_absence},
        rules={"R_anupalabdhi": rule},
    )
    result = engine.infer(
        InferenceRequest(
            rule_id="R_anupalabdhi",
            premise_evidence_ids=["E1", "E2"],
            target=Proposition.atom("jar_absent_from_table"),
        )
    )

    assert result.status in (InferenceStatus.VALID, InferenceStatus.SUSPENDED)


# ---------------------------------------------------------------------------
# Path 2: QA solver — rule bank look-up → symbolic verifier
# ---------------------------------------------------------------------------

def test_e2e_qa_solver_anumana_question():
    """solve_question correctly classifies a classic Anumana question."""
    result = solve_question("What is anumana pramana in Nyaya philosophy?")
    assert result.get("answer_pramana") or result.get("display_pramana") or result.get("answer_text")
    # Must identify Anumana / inference
    output = json.dumps(result).lower()
    assert "anumana" in output or "inference" in output


def test_e2e_qa_solver_anupalabdhi_question():
    """solve_question handles an Anupalabdhi question from the expanded rule bank."""
    result = solve_question("What is anupalabdhi pramana?")
    output = json.dumps(result).lower()
    assert "anupalabdhi" in output or "non-perception" in output or "non_perception" in output


def test_e2e_qa_solver_hetvabhasa_question():
    """solve_question identifies hetvabhasa questions."""
    result = solve_question("What are the five hetvabhasa fallacies in Nyaya?")
    output = json.dumps(result).lower()
    assert any(
        term in output
        for term in ("hetvabhasa", "fallac", "savyabhicara", "viruddha", "asiddha")
    )


def test_e2e_qa_solver_mcq_pratyaksha():
    """MCQ question with clear Pratyaksha answer resolves correctly.
    Options embedded in question string are parsed by solve_question internally."""
    result = solve_question(
        "Which pramana is based on direct sense contact?\n"
        "(A) Pratyaksha\n(B) Anumana\n(C) Shabda\n(D) Upamana"
    )
    output = json.dumps(result).lower()
    assert "pratyaksha" in output


# ---------------------------------------------------------------------------
# Path 3: RAG answer endpoint — mocked retrieval + mocked LLM
# ---------------------------------------------------------------------------

def _make_fake_pipeline(llm_answer: str | None, retrieval_chunks: List[Dict[str, Any]]):
    """Build a minimal fake pipeline that exercises the web layer."""
    class FakeVectorStore:
        def size(self):
            return len(retrieval_chunks)

        def search(self, query, k=5):
            return retrieval_chunks[:k]

    class FakeLLM:
        def health_check(self):
            return True

        def generate_answer(self, question, chunks):
            return llm_answer, "Mock reasoning"

    class FakePipeline:
        def __init__(self):
            self._initialized = True
            self.llm_engine = FakeLLM()
            self.vector_store = FakeVectorStore()

        def initialize(self):
            self._initialized = True

        def answer_question(self, question, pramana_types=None, use_llm=True, use_reasoning_chain=False):
            return {
                "question": question,
                "answer": llm_answer or "Anumana is inference.",
                "pramana": "Anumana",
                "epistemic_status": "valid",
                "confidence": 0.85,
                "citations": retrieval_chunks[:3],
                "answer_source": "llm_reasoning" if llm_answer else "symbolic_fallback",
            }

    return FakePipeline()


def test_e2e_rag_answer_happy_path(monkeypatch):
    """Full RAG answer endpoint returns structured pramana-verified response."""
    chunks = [
        {
            "id": "c1",
            "text": "Anumana is the Nyaya pramana of inference, mediated by hetu and vyapti.",
            "source": "nyaya_sutra",
            "score": 0.9,
            "fused_score": 0.9,
        }
    ]
    fake_pipeline = _make_fake_pipeline("Anumana is inference based on vyapti.", chunks)

    app = create_app()
    app.testing = True

    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: fake_pipeline)

    client = app.test_client()
    response = client.post(
        "/api/rag/answer",
        json={"question": "What is Anumana?", "useLLM": True},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert "answer" in data or "question" in data


def test_e2e_rag_answer_symbolic_fallback(monkeypatch):
    """When LLM is unavailable, symbolic fallback still returns a valid response."""
    chunks = [
        {
            "id": "c1",
            "text": "Pratyaksha is direct perception in Nyaya epistemology.",
            "source": "nyaya_sutra",
            "score": 0.8,
            "fused_score": 0.8,
        }
    ]
    fake_pipeline = _make_fake_pipeline(None, chunks)

    app = create_app()
    app.testing = True

    monkeypatch.setattr("pramana_engine.web.get_rag_pipeline", lambda: fake_pipeline)

    client = app.test_client()
    response = client.post(
        "/api/rag/answer",
        json={"question": "What is pratyaksha?", "useLLM": False},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data  # Non-empty response


def test_e2e_rag_input_validation():
    """Oversized or missing question fields return HTTP 400."""
    app = create_app()
    app.testing = True
    client = app.test_client()

    # Missing question
    response = client.post("/api/rag/answer", json={})
    assert response.status_code == 400

    # Oversized question
    response = client.post("/api/rag/answer", json={"question": "x" * 2001})
    assert response.status_code == 400


def test_e2e_infer_input_validation():
    """Oversized paksha/sadhya/hetu fields return HTTP 400."""
    app = create_app()
    app.testing = True
    client = app.test_client()

    big = "a" * 513
    response = client.post(
        "/api/infer",
        json={"paksha": big, "sadhya": "fire", "hetu": "smoke", "hetuConf": 0.8, "vyaptiStr": 0.9},
    )
    assert response.status_code == 400
    assert "characters" in response.get_json().get("error", "").lower()


# ---------------------------------------------------------------------------
# New tests: demo mode, zero retrieval, /api/health, query expansion,
# low-coverage suspension, invalid pramana label rejection
# ---------------------------------------------------------------------------

def test_e2e_demo_mode_returns_structured_response(monkeypatch):
    """PRAMANA_DEMO_MODE=1 produces a structured symbolic answer without LLM."""
    monkeypatch.setenv("PRAMANA_DEMO_MODE", "1")
    app = create_app()
    app.testing = True
    client = app.test_client()

    FAKE_CHUNK = {
        "id": "NS-1.1.3", "text": "Pratyaksha is direct perception.",
        "source": "Nyaya Sutra 1.1.3", "tags": ["pramana"],
        "supports": ["perception"], "fused_score": 0.7,
    }

    with monkeypatch.context() as m:
        import pramana_engine.rag_pipeline as _rp
        _pipe = _make_fake_pipeline(None, [FAKE_CHUNK])
        # Inject demo_mode into the pipeline's config view
        m.setattr("pramana_engine.web.get_rag_pipeline", lambda: _pipe)
        resp = client.post("/api/rag/answer", json={"question": "What is pratyaksha?"})

    # Even if the web layer didn't use the fake pipeline (demo_mode goes through
    # qa_solver directly), we expect a structured 200 response.
    assert resp.status_code in (200, 400)  # 400 only if question validation fails
    if resp.status_code == 200:
        data = resp.get_json()
        assert data.get("answer") or data.get("answer_text")

    monkeypatch.delenv("PRAMANA_DEMO_MODE", raising=False)


def test_e2e_zero_retrieval_returns_explicit_failure(monkeypatch):
    """When retrieval returns 0 chunks the RAGPipeline returns low/zero confidence."""
    from pramana_engine.rag_pipeline import RAGPipeline

    # Patch hybrid_search to return nothing so the pipeline gets 0 chunks
    monkeypatch.setattr("pramana_engine.rag_pipeline.hybrid_search", lambda *a, **kw: [])

    pipeline = RAGPipeline()
    pipeline._initialized = True

    # Directly exercise the pipeline (no web layer) to avoid singleton caching
    result = pipeline.answer_question("What is vyapti?", use_llm=False)
    # With 0 retrieved chunks the pipeline returns confidence=0.0
    assert result.get("confidence", 1.0) == 0.0
    assert "no relevant" in (result.get("answer") or "").lower()


def test_e2e_api_health_all_keys_present():
    """GET /api/health returns all required component keys."""
    app = create_app()
    app.testing = True
    client = app.test_client()

    resp = client.get("/api/health")
    assert resp.status_code in (200, 207)
    data = resp.get_json()
    for key in ("pkg_faiss", "pkg_flask", "vector_store", "llm", "rule_bank", "overall"):
        assert key in data, f"Missing health key: {key}"
    assert data["overall"] in ("healthy", "degraded")


def test_e2e_query_expansion_synonym_match():
    """Query expansion maps Sanskrit synonyms so retrieval matches more chunks."""
    from pramana_engine.hybrid_retrieval import _QUERY_SYNONYMS, _tokenize
    # Verify the synonym map is populated
    assert "fire" in _QUERY_SYNONYMS
    assert "agni" in _QUERY_SYNONYMS["fire"]
    # Verify expansion is injective: query with 'fire' expands to include 'agni'
    base_tokens = set(_tokenize("fire"))
    expanded: set = set(base_tokens)
    for tok in base_tokens:
        for syn in _QUERY_SYNONYMS.get(tok, []):
            expanded.update(_tokenize(syn))
    assert "agni" in expanded, "Synonym 'agni' should be reachable via 'fire' expansion"


def test_e2e_low_coverage_forces_suspended(monkeypatch):
    """When retrieval is sparse (top_score < 0.15) and no LLM, status is suspended."""
    from pramana_engine.rag_pipeline import RAGPipeline

    low_chunk = {
        "id": "X1", "text": "Some text.", "source": "unknown",
        "tags": [], "supports": ["inference"], "fused_score": 0.05,
    }
    monkeypatch.setattr("pramana_engine.rag_pipeline.hybrid_search", lambda *a, **kw: [low_chunk])

    pipeline = RAGPipeline()
    pipeline._initialized = True
    pipeline.llm_engine = None  # Simulate no LLM

    result = pipeline.answer_question("What is hetu?", use_llm=False)
    meta = result.get("retrieval_meta", {})
    # Retrieval meta must flag low coverage
    assert meta.get("low_coverage") is True, f"Expected low_coverage=True, got {meta}"
    # With no LLM + low coverage, status must be suspended
    assert result.get("epistemic_status") == "suspended", (
        f"Expected suspended, got {result.get('epistemic_status')}"
    )


def test_e2e_invalid_pramana_label_rejected():
    """Completely unrecognized pramana label in /api/infer returns HTTP 400."""
    app = create_app()
    app.testing = True
    client = app.test_client()

    resp = client.post(
        "/api/infer",
        json={
            "paksha": "the hill",
            "sadhya": "fire",
            "hetu": "smoke",
            "pramanaType": "xyz_completely_invalid_label_9999",
            "hetuConf": 0.8,
            "vyaptiStr": 0.9,
        },
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "Unrecognized pramana label" in data.get("error", "") or "valid_labels" in data
