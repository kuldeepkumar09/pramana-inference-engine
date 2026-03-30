from __future__ import annotations

from typing import Dict, Tuple

from .engine import PramanaInferenceEngine
from .models import Evidence, InferenceRequest, Proposition, Rule


def build_default_rulebook() -> Dict[str, Rule]:
    return {
        "R1": Rule(
            rule_id="R1",
            name="Anumana via Modus Ponens",
            pattern="modus_ponens",
            required_pramanas=["perception", "testimony"],
            min_reliability=0.7,
            suspension_margin=0.05,
        )
    }


def scenario_valid() -> Tuple[PramanaInferenceEngine, InferenceRequest]:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.atom("smoke_on_hill"),
            pramana="perception",
            reliability=0.9,
            source="field_observation",
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
            pramana="testimony",
            reliability=0.88,
            source="trusted_text",
        ),
    }
    request = InferenceRequest(
        rule_id="R1",
        premise_evidence_ids=["E1", "E2"],
        target=Proposition.atom("fire_on_hill"),
    )
    return PramanaInferenceEngine(evidence, build_default_rulebook()), request


def scenario_unjustified_missing_pramana() -> Tuple[PramanaInferenceEngine, InferenceRequest]:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.atom("smoke_on_hill"),
            pramana="testimony",
            reliability=0.92,
            source="single_report",
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
            pramana="testimony",
            reliability=0.9,
            source="manual",
        ),
    }
    request = InferenceRequest(
        rule_id="R1",
        premise_evidence_ids=["E1", "E2"],
        target=Proposition.atom("fire_on_hill"),
    )
    return PramanaInferenceEngine(evidence, build_default_rulebook()), request


def scenario_suspended_near_threshold() -> Tuple[PramanaInferenceEngine, InferenceRequest]:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.atom("smoke_on_hill"),
            pramana="perception",
            reliability=0.66,
            source="distant_observation",
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
            pramana="testimony",
            reliability=0.72,
            source="local_report",
        ),
    }
    request = InferenceRequest(
        rule_id="R1",
        premise_evidence_ids=["E1", "E2"],
        target=Proposition.atom("fire_on_hill"),
    )
    return PramanaInferenceEngine(evidence, build_default_rulebook()), request


def scenario_invalid_pattern() -> Tuple[PramanaInferenceEngine, InferenceRequest]:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.atom("fire_on_hill"),
            pramana="perception",
            reliability=0.91,
            source="field_observation",
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
            pramana="testimony",
            reliability=0.9,
            source="trusted_text",
        ),
    }
    request = InferenceRequest(
        rule_id="R1",
        premise_evidence_ids=["E1", "E2"],
        target=Proposition.atom("smoke_on_hill"),
    )
    return PramanaInferenceEngine(evidence, build_default_rulebook()), request


def scenario_suspended_defeated() -> Tuple[PramanaInferenceEngine, InferenceRequest]:
    evidence = {
        "E1": Evidence(
            evidence_id="E1",
            proposition=Proposition.atom("smoke_on_hill"),
            pramana="perception",
            reliability=0.86,
            source="field_observation",
            defeated=True,
            metadata={"challenge": "fog_like_dust"},
        ),
        "E2": Evidence(
            evidence_id="E2",
            proposition=Proposition.implies("smoke_on_hill", "fire_on_hill"),
            pramana="testimony",
            reliability=0.84,
            source="trusted_text",
        ),
    }
    request = InferenceRequest(
        rule_id="R1",
        premise_evidence_ids=["E1", "E2"],
        target=Proposition.atom("fire_on_hill"),
    )
    return PramanaInferenceEngine(evidence, build_default_rulebook()), request


SCENARIOS = {
    "valid": scenario_valid,
    "unjustified": scenario_unjustified_missing_pramana,
    "suspended_threshold": scenario_suspended_near_threshold,
    "invalid": scenario_invalid_pattern,
    "suspended_defeated": scenario_suspended_defeated,
}
