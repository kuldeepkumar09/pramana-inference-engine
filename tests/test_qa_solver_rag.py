import pytest

from pramana_engine.qa_solver import (
    _load_external_knowledge_base,
    _match_rule,
    _parse_options,
    build_inference_mapping,
    solve_question,
)
from pramana_engine.web import create_app

KNOWN_STATUSES = {"valid", "unjustified", "suspended", "invalid"}


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def test_external_json_corpus_is_indexed_into_kb_chunks():
    kb = _load_external_knowledge_base()
    assert kb
    assert any(item["id"].startswith("EXT:") for item in kb)
    assert any(str(item.get("source", "")).endswith(".json") for item in kb)


def test_external_corpus_chunks_have_required_fields():
    kb = _load_external_knowledge_base()
    for item in kb[:20]:
        assert "id" in item
        assert "source" in item
        assert "text" in item
        assert "supports" in item
        assert isinstance(item["supports"], list)


# ---------------------------------------------------------------------------
# Rule bank matching
# ---------------------------------------------------------------------------


def test_match_rule_returns_dict_for_known_keyword():
    result = _match_rule("What is pratyaksha perception?")
    assert result is not None
    assert "id" in result
    assert "answer_text" in result
    assert result["match_score"] > 0


def test_match_rule_returns_none_for_unrelated_text():
    result = _match_rule("What is the capital of France?")
    assert result is None


def test_match_rule_picks_vyapti_for_anumana_question():
    result = _match_rule("Anumana is based on what invariable relation?")
    assert result is not None
    assert "vyapti" in result["answer_text"].lower() or "vyapti" in result.get("explanation", "").lower()


def test_match_rule_picks_aprama_for_illusion_question():
    result = _match_rule("What is illusion called in Nyaya? Is it aprama or prama?")
    assert result is not None
    assert result["id"] == "illusion_aprama"


# ---------------------------------------------------------------------------
# Option parsing
# ---------------------------------------------------------------------------


def test_parse_options_extracts_abcd():
    q = "Which is direct? A. Inference B. Perception C. Testimony D. Comparison"
    opts = _parse_options(q)
    assert len(opts) == 4
    keys = [o["key"] for o in opts]
    assert keys == ["A", "B", "C", "D"]


def test_parse_options_returns_empty_for_plain_question():
    opts = _parse_options("What is pratyaksha?")
    assert opts == []


# ---------------------------------------------------------------------------
# solve_question — MCQ mode (rule bank hit)
# ---------------------------------------------------------------------------


def test_solve_mcq_pratyaksha_immediate_picks_correct_option():
    question = (
        "Pratyaksha knowledge is described as A. Mediate knowledge "
        "B. Immediate knowledge C. Inferential D. Scriptural"
    )
    result = solve_question(question)
    assert result["mode"] == "mcq"
    assert result["answer_key"] == "B"
    assert "immediate" in result["answer_text"].lower()


def test_solve_mcq_returns_required_fields():
    question = (
        "What does anumana depend on? "
        "A. Sense organs B. Vyapti C. Verbal testimony D. Comparison"
    )
    result = solve_question(question)
    for field in ("mode", "question", "answer_key", "answer_text", "answer_pramana",
                  "confidence", "epistemic_status", "option_scores", "verifier", "trace"):
        assert field in result, f"Missing field: {field}"


def test_solve_mcq_confidence_in_unit_interval():
    question = "A. Perception B. Inference C. Testimony D. Comparison"
    result = solve_question(question)
    assert 0.0 <= result["confidence"] <= 1.0


def test_solve_mcq_epistemic_status_is_known_value():
    question = "Which pramana is primary in Nyaya? A. Anumana B. Pratyaksha C. Shabda D. Upamana"
    result = solve_question(question)
    assert result["epistemic_status"] in KNOWN_STATUSES


def test_solve_mcq_option_scores_all_options_present():
    question = "A. Pratyaksha B. Anumana C. Shabda D. Upamana"
    result = solve_question(question)
    keys = {row["key"] for row in result["option_scores"]}
    assert keys == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# solve_question — classification mode (no options)
# ---------------------------------------------------------------------------


def test_solve_classification_returns_structured_result():
    result = solve_question("What kind of cognition is an illusion in Nyaya?")
    assert result["mode"] == "classification"
    assert result["answer_text"]
    assert result["answer_pramana"]


def test_solve_classification_illusion_returns_aprama():
    result = solve_question("Is illusion considered aprama in Nyaya?")
    assert "aprama" in result["answer_text"].lower() or "aprama" in result["answer_pramana"].lower()


def test_solve_classification_fallback_no_rule_match():
    result = solve_question("What is the speed of light in a vacuum?")
    assert result["mode"] in ("classification", "mcq")
    assert "answer_pramana" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_solve_classification_out_of_domain_abstains_when_support_is_low():
    result = solve_question("Who won the FIFA World Cup in 2014?")
    assert result["mode"] == "classification"
    assert result["epistemic_status"] == "suspended"
    assert "Insufficient supported evidence" in result["answer_text"]


def test_conference_question_surfaces_unriddling_or_meta_cognition_evidence():
    result = solve_question("What is the goal of the Unriddling Inference conference for AI?")
    citation_ids = [c.get("id", "") for row in result.get("option_scores", []) for c in row.get("citations", [])]
    assert any(cid.startswith("CONF-2026") for cid in citation_ids)


# ---------------------------------------------------------------------------
# Verifier output structure
# ---------------------------------------------------------------------------


def test_verifier_has_required_decomposition_fields():
    result = solve_question("What is vyapti in Nyaya inference?")
    decomp = result["verifier"]["confidence_decomposition"]
    for field in ("retrieval_support", "rule_consistency", "contradiction_penalty",
                  "source_authority", "final_confidence"):
        assert field in decomp, f"Missing decomp field: {field}"


def test_verifier_belief_revision_fields_present():
    result = solve_question("What is pratyaksha?")
    br = result["verifier"]["belief_revision"]
    assert "final_status" in br
    assert br["final_status"] in KNOWN_STATUSES


# ---------------------------------------------------------------------------
# build_inference_mapping
# ---------------------------------------------------------------------------


def test_build_inference_mapping_structure():
    qa = solve_question("What is the basis of anumana? A. Vyapti B. Sense organs C. Testimony D. Comparison")
    mapping = build_inference_mapping(qa)
    for field in ("paksha", "sadhya", "hetu", "udaharana", "pramanaTypes"):
        assert field in mapping, f"Missing mapping field: {field}"
    assert isinstance(mapping["pramanaTypes"], list)
    assert len(mapping["pramanaTypes"]) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_solve_question_raises_on_empty_string():
    with pytest.raises(ValueError):
        solve_question("")


def test_solve_question_raises_on_whitespace_only():
    with pytest.raises(ValueError):
        solve_question("   ")


# ---------------------------------------------------------------------------
# Web API — /api/question-solve and ragChunks
# ---------------------------------------------------------------------------


def test_api_question_solve_returns_200():
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post("/api/question-solve", json={"question": "What is pratyaksha?"})
    assert response.status_code == 200
    data = response.get_json()
    assert "question_result" in data
    assert "mapping" in data


def test_api_question_solve_missing_question_returns_400():
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post("/api/question-solve", json={})
    assert response.status_code == 400


def test_api_infer_rag_chunks_populated():
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post("/api/infer", json={
        "paksha": "hill",
        "sadhya": "fire",
        "hetu": "smoke",
        "hetuConf": 0.85,
        "vyaptiStr": 0.9,
        "pramanaTypes": ["perception"],
    })
    assert response.status_code == 200
    data = response.get_json()
    assert "ragChunks" in data
    assert isinstance(data["ragChunks"], list)
    assert len(data["ragChunks"]) > 0, "ragChunks should be populated with citations"


def test_api_infer_rag_chunks_have_expected_fields():
    app = create_app()
    app.testing = True
    client = app.test_client()
    response = client.post("/api/infer", json={
        "paksha": "inference",
        "sadhya": "vyapti_required",
        "hetu": "anumana",
        "hetuConf": 0.8,
        "vyaptiStr": 0.8,
        "pramanaTypes": ["inference"],
    })
    assert response.status_code == 200
    data = response.get_json()
    for chunk in data.get("ragChunks", []):
        assert "id" in chunk
        assert "source" in chunk
        assert "excerpt" in chunk


def test_question_solver_surfaces_external_citation_for_external_corpus_query():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "What does the DocScanner OCR test corpus say about pramana?"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    rows = data["question_result"].get("option_scores", [])
    citation_ids = [c.get("id", "") for row in rows for c in row.get("citations", [])]
    assert any(citation_id.startswith("EXT:") for citation_id in citation_ids)
