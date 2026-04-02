from pramana_engine.web import create_app


def test_api_infer_returns_valid_payload_shape():
    app = create_app()
    app.testing = True
    client = app.test_client()

    payload = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "udaharana": "kitchen hearth",
        "conflict": "",
        "pramanaType": "Anumana",
        "hetuConf": 0.9,
        "vyaptiStr": 0.88,
        "mode": "standard",
    }

    response = client.post("/api/infer", json=payload)
    assert response.status_code == 200

    data = response.get_json()
    assert "gate" in data
    assert "jsonSummary" in data
    assert "sections" in data
    assert data["gate"]["verdict"] in {"VALID", "UNJUSTIFIED", "SUSPENDED", "REJECTED"}
    assert "calibrated_score" in data["gate"]


def test_api_infer_rejects_missing_required_fields():
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.post("/api/infer", json={"paksha": "x"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_api_compare_returns_side_by_side_payload():
    app = create_app()
    app.testing = True
    client = app.test_client()

    payload = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "conflict": "",
        "pramanaType": "Anumana",
        "hetuConf": 0.9,
        "vyaptiStr": 0.88,
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    data = response.get_json()

    assert "constrained" in data
    assert "baseline" in data
    assert "delta" in data
    assert "verdict" in data["constrained"]
    assert "verdict" in data["baseline"]


def test_api_infer_supports_fused_pramana_selection():
    app = create_app()
    app.testing = True
    client = app.test_client()

    payload = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "conflict": "",
        "pramanaType": "Anumana",
        "pramanaTypes": ["Pratyaksha", "Anumana", "Shabda", "Upamana", "Arthapatti"],
        "combineAllPramanas": True,
        "hetuConf": 0.9,
        "vyaptiStr": 0.88,
    }
    response = client.post("/api/infer", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["combineAllPramanas"] is True
    # combineAllPramanas uses ALL_PRAMANAS (6 pramanas including Anupalabdhi)
    assert len(data["selectedPramanas"]) == 6


def test_non_defeater_conflict_text_does_not_force_suspension():
    app = create_app()
    app.testing = True
    client = app.test_client()

    payload = {
        "paksha": "the distant hill",
        "sadhya": "fire is present",
        "hetu": "smoke is visible",
        "conflict": "Pratyaksha is: A. Mediate knowledge B. Immediate knowledge",
        "pramanaType": "Anumana",
        "hetuConf": 0.95,
        "vyaptiStr": 0.92,
    }
    response = client.post("/api/infer", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["gate"]["hasBadhita"] is False


def test_question_solver_mcq_returns_option_answer_and_mapping():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Sense-object contact is necessary for: A. Anumana B. Pratyaksha C. Shabda D. Arthapatti"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "mcq"
    assert data["question_result"]["answer_key"] == "B"
    assert "mapping" in data
    assert "Pratyaksha" in data["mapping"]["pramanaTypes"]


def test_question_solver_conceptual_question_returns_pramana_classification():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Which Pramana ultimately verifies all knowledge?"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "classification"
    assert data["question_result"]["answer_text"] == "Pratyaksha"


def test_question_solver_illusion_mcq_resolves_to_aprama_option_b():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Illusion is called: A. Prama B. Aprama C. Smrti D. Pratyaksha"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "mcq"
    assert data["question_result"]["answer_key"] == "B"
    assert data["question_result"]["answer_text"] == "Aprama"


def test_question_solver_rule_bank_nyaya_accepts_four_maps_to_option_c():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Nyaya accepts how many pramanas? A. 2 B. 3 C. 4 D. 6"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "mcq"
    assert data["question_result"]["answer_key"] == "C"
    assert data["question_result"]["trace"]["type"] == "neuro_symbolic_rule_bank_match"
    assert "verifier" in data["question_result"]
    assert "confidence_decomposition" in data["question_result"]["verifier"]


def test_question_solver_rule_bank_viparyaya_maps_to_error_option_b():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Viparyaya means: A. valid cognition B. error or wrong cognition C. memory D. perception"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "mcq"
    assert data["question_result"]["answer_key"] == "B"
    assert data["question_result"]["trace"]["type"] == "neuro_symbolic_rule_bank_match"


def test_question_solver_reordered_options_prefers_rule_answer_text_over_letter():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Nyaya accepts how many pramanas? A. 4 B. 2 C. 6 D. 3"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    assert data["question_result"]["mode"] == "mcq"
    assert data["question_result"]["answer_key"] == "A"
    assert data["question_result"]["answer_text"] == "4"


def test_question_solver_emits_verifier_fields_and_confidence_formula():
    app = create_app()
    app.testing = True
    client = app.test_client()

    question = "Sense-object contact is necessary for: A. Anumana B. Pratyaksha C. Shabda D. Arthapatti"
    response = client.post("/api/question-solve", json={"question": question})
    assert response.status_code == 200
    data = response.get_json()

    q = data["question_result"]
    assert "verifier" in q
    assert "epistemic_status" in q
    assert q["epistemic_status"] in {"valid", "unjustified", "suspended"}
    decomp = q["verifier"]["confidence_decomposition"]
    assert decomp["formula"] == "retrieval_support + rule_consistency + contradiction_penalty + source_authority"
    assert 0.0 <= q["confidence"] <= 1.0


def test_conference_qa_returns_answer_confidence_and_citations():
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/conference-qa",
        json={"question": "What is the goal of the Unriddling Inference conference?"},
    )
    assert response.status_code == 200
    data = response.get_json()

    assert "answer" in data
    assert "confidence" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert 0.0 <= data["confidence"] <= 1.0


def test_conference_qa_abstains_for_low_supported_query():
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/conference-qa",
        json={"question": "Who won the FIFA World Cup in 2014?", "minConfidence": 0.6},
    )
    assert response.status_code == 200
    data = response.get_json()

    assert data["abstained"] is True
    assert data["answer"].startswith("I don't know")
