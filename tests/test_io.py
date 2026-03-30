import json

from pramana_engine.io import infer_from_file, infer_many_from_payload
from pramana_engine.models import InferenceStatus


def test_infer_from_file_valid_payload(tmp_path):
    payload = {
        "rules": [
            {
                "rule_id": "R1",
                "name": "Anumana via Modus Ponens",
                "pattern": "modus_ponens",
                "required_pramanas": ["perception", "testimony"],
                "min_reliability": 0.7,
                "suspension_margin": 0.05,
            }
        ],
        "evidence": [
            {
                "evidence_id": "E1",
                "proposition": {"kind": "atom", "value": "smoke_on_hill"},
                "pramana": "perception",
                "reliability": 0.9,
                "source": "field_observation",
            },
            {
                "evidence_id": "E2",
                "proposition": {
                    "kind": "implies",
                    "antecedent": "smoke_on_hill",
                    "consequent": "fire_on_hill",
                },
                "pramana": "testimony",
                "reliability": 0.88,
                "source": "trusted_text",
            },
        ],
        "request": {
            "rule_id": "R1",
            "premise_evidence_ids": ["E1", "E2"],
            "target": {"kind": "atom", "value": "fire_on_hill"},
        },
    }

    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    result = infer_from_file(str(input_file))

    assert result.status == InferenceStatus.VALID
    assert result.accepted is True


def test_infer_from_file_mapping_payload_shape(tmp_path):
    payload = {
        "mapping": {
            "paksha": "the asked knowledge claim",
            "sadhya": "Pratyaksha is the best-supported answer",
            "hetu": "direct cognition depends on immediate sense-object contact",
            "udaharana": "seeing a pot through eye-object contact is pratyaksha",
            "pramanaTypes": ["Pratyaksha", "Anumana"],
        },
        "question_result": {
            "answer_text": "Pratyaksha",
            "answer_pramana": "Pratyaksha",
            "confidence": 0.7425,
        },
    }

    input_file = tmp_path / "mapping_input.json"
    input_file.write_text(json.dumps(payload), encoding="utf-8")

    result = infer_from_file(str(input_file))

    assert result.status in {InferenceStatus.VALID, InferenceStatus.SUSPENDED, InferenceStatus.UNJUSTIFIED}
    assert result.trace.get("request", {}).get("rule_id") == "R1"


def test_infer_many_from_payload_mapping_shape_returns_row():
    payload = {
        "mapping": {
            "paksha": "the asked knowledge claim",
            "sadhya": "Pratyaksha is the best-supported answer",
            "hetu": "direct cognition depends on immediate sense-object contact",
            "pramanaTypes": ["Pratyaksha", "Anumana"],
        },
        "question_result": {
            "answer_text": "Pratyaksha",
            "answer_pramana": "Pratyaksha",
        },
    }

    rows = infer_many_from_payload(payload)
    assert len(rows) == 1
    assert rows[0]["ok"] is True
    assert rows[0]["result"]["status"] in {"valid", "unjustified", "suspended", "invalid"}
