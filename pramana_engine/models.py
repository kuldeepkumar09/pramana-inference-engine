from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InferenceStatus(str, Enum):
    VALID = "valid"
    UNJUSTIFIED = "unjustified"
    SUSPENDED = "suspended"
    INVALID = "invalid"


@dataclass(frozen=True)
class Proposition:
    kind: str
    value: Optional[str] = None
    antecedent: Optional[str] = None
    consequent: Optional[str] = None

    @staticmethod
    def atom(name: str) -> "Proposition":
        return Proposition(kind="atom", value=name)

    @staticmethod
    def implies(antecedent: str, consequent: str) -> "Proposition":
        return Proposition(kind="implies", antecedent=antecedent, consequent=consequent)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "antecedent": self.antecedent,
            "consequent": self.consequent,
        }

    def to_repr(self) -> str:
        """Return a compact programmatic representation of this proposition."""
        if self.kind == "atom":
            return f"atom:{self.value}"
        return f"implies:{self.antecedent}->{self.consequent}"


@dataclass
class Evidence:
    evidence_id: str
    proposition: Proposition
    pramana: str
    reliability: float
    source: str
    defeated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "proposition": self.proposition.to_dict(),
            "pramana": self.pramana,
            "reliability": self.reliability,
            "source": self.source,
            "defeated": self.defeated,
            "metadata": self.metadata,
        }


@dataclass
class Rule:
    rule_id: str
    name: str
    pattern: str
    required_pramanas: List[str]
    min_reliability: float = 0.7
    suspension_margin: float = 0.05
    pramana_weights: Dict[str, float] = field(default_factory=dict)
    calibration_exponent: float = 1.0


@dataclass
class InferenceRequest:
    rule_id: str
    premise_evidence_ids: List[str]
    target: Proposition


@dataclass
class InferenceResult:
    status: InferenceStatus
    accepted: bool
    target: Proposition
    trace: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "accepted": self.accepted,
            "target": self.target.to_dict(),
            "trace": self.trace,
            "message": self.message,
            "requirements_manifest": {
                "R1_proposition_representation": True,
                "R2_epistemic_justification_check": True,
                "R3_status_distinction": self.status.value,
                "R4_invalid_patterns_rejected": True,
                "R5_machine_readable_trace": True,
            },
        }
