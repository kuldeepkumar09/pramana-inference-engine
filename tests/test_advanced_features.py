import io
import json

from pramana_engine.engine import PramanaInferenceEngine
from pramana_engine.io import infer_many_from_payload
from pramana_engine.models import Evidence, InferenceRequest, InferenceStatus, Proposition, Rule
from pramana_engine.web import create_app


def test_modus_tollens_pattern_valid():
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.implies("rain", "wet_ground"),
            pramana="inference",
            reliability=0.9,
            source="rulebook",
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.atom("not_wet_ground"),
            pramana="perception",
            reliability=0.9,
            source="observation",
        ),
    }
    rules = {
        "R2": Rule(
            rule_id="R2",
            name="Modus Tollens",
            pattern="modus_tollens",
            required_pramanas=["inference", "perception"],
            min_reliability=0.7,
        )
    }
    engine = PramanaInferenceEngine(evidence, rules)
    result = engine.infer(
        InferenceRequest(
            rule_id="R2",
            premise_evidence_ids=["E1", "E2"],
            target=Proposition.atom("not_rain"),
        )
    )
    assert result.status == InferenceStatus.VALID


def test_hypothetical_syllogism_valid():
    evidence = {
        "E1": Evidence("E1", Proposition.implies("a", "b"), "inference", 0.8, "src1"),
        "E2": Evidence("E2", Proposition.implies("b", "c"), "testimony", 0.8, "src2"),
    }
    rules = {
        "R3": Rule(
            rule_id="R3",
            name="Hypothetical Syllogism",
            pattern="hypothetical_syllogism",
            required_pramanas=["inference", "testimony"],
            min_reliability=0.7,
        )
    }
    engine = PramanaInferenceEngine(evidence, rules)
    result = engine.infer(
        InferenceRequest(
            rule_id="R3",
            premise_evidence_ids=["E1", "E2"],
            target=Proposition.implies("a", "c"),
        )
    )
    assert result.status == InferenceStatus.VALID


def test_batch_payload_inference_counts():
    payload = [
        {
            "proposition": {
                "claim": "fire_on_hill",
                "source": "text",
                "pramana_type": "Anumana",
                "confidence": 0.9,
            },
            "evidence": [
                {
                    "claim": "smoke_on_hill",
                    "source": "obs",
                    "pramana_type": "Pratyaksha",
                    "confidence": 0.9,
                }
            ],
        },
        {
            "proposition": {
                "claim": "",
                "source": "text",
                "pramana_type": "Anumana",
                "confidence": 0.9,
            },
            "evidence": [{"claim": "x", "source": "obs", "pramana_type": "Pratyaksha", "confidence": 0.9}],
        },
    ]
    rows = infer_many_from_payload(payload)
    assert len(rows) == 2
    assert rows[0]["ok"] is True
    assert rows[1]["ok"] is False


def test_upload_and_judge_endpoints():
    app = create_app()
    app.testing = True
    client = app.test_client()

    upload_payload = [
        {
            "proposition": {
                "claim": "fire_on_hill",
                "source": "text",
                "pramana_type": "Anumana",
                "confidence": 0.9,
            },
            "evidence": [
                {
                    "claim": "smoke_on_hill",
                    "source": "obs",
                    "pramana_type": "Pratyaksha",
                    "confidence": 0.9,
                }
            ],
        }
    ]
    data = {
        "file": (io.BytesIO(json.dumps(upload_payload).encode("utf-8")), "input.json"),
    }
    upload_response = client.post("/api/infer-upload", data=data, content_type="multipart/form-data")
    assert upload_response.status_code == 200
    upload_json = upload_response.get_json()
    assert upload_json["total"] == 1

    judge_response = client.post("/api/judge-report", json={"rows": upload_json["rows"]})
    assert judge_response.status_code == 200
    judge_json = judge_response.get_json()
    assert "quality_score" in judge_json
    assert "checklist" in judge_json


def test_weighted_pramana_calibration_changes_outcome():
    evidence = {
        "E1": Evidence("E1", Proposition.atom("smoke"), "testimony", 0.75, "report"),
        "E2": Evidence("E2", Proposition.implies("smoke", "fire"), "testimony", 0.75, "manual"),
    }

    neutral_rule = Rule(
        rule_id="R1",
        name="Neutral Calibration",
        pattern="modus_ponens",
        required_pramanas=["testimony"],
        min_reliability=0.7,
        pramana_weights={"testimony": 1.0},
    )
    strict_rule = Rule(
        rule_id="R2",
        name="Strict Testimony Weight",
        pattern="modus_ponens",
        required_pramanas=["testimony"],
        min_reliability=0.7,
        pramana_weights={"testimony": 0.5},
        calibration_exponent=1.5,
    )

    neutral = PramanaInferenceEngine(evidence, {"R1": neutral_rule}).infer(
        InferenceRequest("R1", ["E1", "E2"], Proposition.atom("fire"))
    )
    strict = PramanaInferenceEngine(evidence, {"R2": strict_rule}).infer(
        InferenceRequest("R2", ["E1", "E2"], Proposition.atom("fire"))
    )

    assert neutral.status == InferenceStatus.VALID
    assert strict.status in {InferenceStatus.SUSPENDED, InferenceStatus.UNJUSTIFIED}
