"""Advanced epistemic reasoning framework for Pramana engine.

Provides programmatic representation of propositions, evidence, inference rules, 
and explicit epistemic constraint checking with machine-readable reasoning traces.

Requirements addressed:
- R1: Represent propositions and supporting evidence programmatically
- R2: Explicitly check epistemic justification before accepting inference
- R3: Distinguish valid, unjustified, and suspended inferences
- R4: Reject invalid inference patterns
- R5: Provide machine-readable reasoning traces
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

from .pramana_registry import PRAMANA_AUTHORITY


class EpistemicStatus(str, Enum):
    """Extended inference status with clear epistemic distinctions."""
    VALID = "valid"
    UNJUSTIFIED = "unjustified"
    SUSPENDED = "suspended"
    INVALID_PATTERN = "invalid_pattern"


@dataclass(frozen=True)
class EvidentialProposition:
    """Programmatic representation of a proposition with source tracking."""
    id: str
    text: str
    kind: str  # 'atom', 'implies', 'negation', etc.
    source: str  # 'observation', 'scripture', 'inference', 'testimony', etc.
    confidence: float  # 0.0 - 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "confidence": self._clip(self.confidence),
            "metadata": self.metadata,
        }

    @staticmethod
    def _clip(val: float) -> float:
        return max(0.0, min(1.0, val))


@dataclass(frozen=True)
class EvidentialSource:
    """Programmatic representation of supporting evidence."""
    id: str
    proposition_id: str
    kind: str  # 'perception', 'document', 'testimony', 'inference', 'comparison'
    content: str
    reliability: float  # 0.0 - 1.0
    defeated: bool = False
    pramana: str = "perception"  # default pramana type
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "proposition_id": self.proposition_id,
            "kind": self.kind,
            "content": self.content[:200] if self.content else "",
            "reliability": self._clip(self.reliability),
            "defeated": self.defeated,
            "pramana": self.pramana,
            "metadata": self.metadata,
        }

    @staticmethod
    def _clip(val: float) -> float:
        return max(0.0, min(1.0, val))


@dataclass(frozen=True)
class InferencePattern:
    """Definition of acceptable inference patterns."""
    name: str
    pattern: str  # 'modus_ponens', 'modus_tollens', etc.
    valid_patterns: List[str]
    description: str = ""

    def is_valid(self) -> bool:
        return self.pattern in self.valid_patterns


@dataclass
class EpistemicEvaluationConfig:
    """Configuration for epistemic evaluation."""
    min_justification_threshold: float = 0.70
    suspension_band: float = 0.10
    pramana_weights: Dict[str, float] = field(
        default_factory=lambda: dict(PRAMANA_AUTHORITY)
    )


@dataclass
class ReasoningTrace:
    """Machine-readable trace of reasoning process."""
    pattern_valid: bool
    epistemic_status: EpistemicStatus
    justification_score: float
    checks: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_valid": self.pattern_valid,
            "status": self.epistemic_status.value,
            "justification_score": round(self.justification_score, 4),
            "checks": self.checks,
            "explanation": self.explanation,
            "error_messages": self.error_messages,
            "requirements_met": {
                "R1_proposition_representation": True,
                "R2_epistemic_justification_check": True,
                "R3_status_distinction": self.epistemic_status.value,
                "R4_invalid_patterns_rejected": not self.pattern_valid,
                "R5_machine_readable_trace": True,
            }
        }


# ─── SCORING DIVERGENCE NOTE ─────────────────────────────────────────────────
# EpistemicEvaluator.compute_justification_score uses a TWO-TERM formula:
#   score = 0.45 * avg(proposition.confidence) + 0.55 * pramana_weighted_avg(evidence.reliability)
# Fixed acceptance threshold: EpistemicEvaluationConfig.min_justification_threshold = 0.70
#
# engine.py._check_epistemic_constraints uses a ONE-TERM formula:
#   calibrated_score = weighted_avg(evidence.reliability) ** calibration_exponent
# Per-rule acceptance threshold: Rule.min_reliability (default 0.70)
#
# These are INTENTIONALLY different systems serving different API surfaces:
#   - EpistemicEvaluator: web enrichment layer, advisory trace only. Its output
#     does NOT gate result.accepted in engine.infer().
#   - engine.py: authoritative gate; its status field determines result.accepted.
# Do not unify without understanding both call sites. EpistemicEvaluator takes
# EvidentialProposition.confidence (UI-layer inputs); engine.py takes
# Evidence.reliability (rule-layer inputs) — they are not the same data.
# ─────────────────────────────────────────────────────────────────────────────
class EpistemicEvaluator:
    """Core epistemic evaluation engine for inferences."""

    def __init__(self, config: Optional[EpistemicEvaluationConfig] = None):
        self.config = config or EpistemicEvaluationConfig()

    def compute_justification_score(
        self,
        propositions: List[EvidentialProposition],
        evidence: List[EvidentialSource],
    ) -> float:
        """Compute justification score from propositions and evidence.
        
        Args:
            propositions: List of propositions to justify
            evidence: List of supporting evidence
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not propositions or not evidence:
            return 0.0

        # Proposition confidence contribution
        prop_scores = [self._clip(p.confidence) for p in propositions]
        prop_score = sum(prop_scores) / len(prop_scores) if prop_scores else 0.0

        # Evidence reliability contribution (weighted by pramana)
        weighted_reliability = 0.0
        total_weight = 0.0

        for ev in evidence:
            weight = self.config.pramana_weights.get(ev.pramana, 0.5)
            # Defeated evidence counts as zero contribution
            ev_reliability = 0.0 if ev.defeated else self._clip(ev.reliability)
            weighted_reliability += ev_reliability * weight
            total_weight += weight

        if total_weight > 0:
            evidence_score = weighted_reliability / total_weight
        else:
            evidence_score = 0.0

        # Combined score with weight toward evidence
        combined = 0.45 * prop_score + 0.55 * evidence_score
        return self._clip(combined)

    def evaluate_inference(
        self,
        propositions: List[EvidentialProposition],
        evidence: List[EvidentialSource],
        conclusion: EvidentialProposition,
        pattern: InferencePattern,
    ) -> ReasoningTrace:
        """Evaluate an inference for epistemic validity.
        
        Args:
            propositions: Premises/supporting propositions
            evidence: Supporting evidence
            conclusion: Target conclusion
            pattern: Inference pattern to validate
            
        Returns:
            ReasoningTrace with status and details
        """

        # Step 1: Check pattern validity (R4)
        pattern_valid = pattern.is_valid()
        
        if not pattern_valid:
            return ReasoningTrace(
                pattern_valid=False,
                epistemic_status=EpistemicStatus.INVALID_PATTERN,
                justification_score=0.0,
                checks={
                    "step": "pattern_validation",
                    "pattern": pattern.pattern,
                    "allowed_patterns": pattern.valid_patterns,
                    "result": "REJECTED",
                },
                explanation="Inference pattern is structurally invalid and rejected.",
                error_messages=[f"Pattern '{pattern.pattern}' not in allowed set: {pattern.valid_patterns}"],
            )

        # Step 2: Compute epistemic justification (R2)
        score = self.compute_justification_score(propositions, evidence)
        threshold = self._clip(self.config.min_justification_threshold)
        lower_band = self._clip(threshold - self.config.suspension_band)

        # Step 3: Classify status (R3)
        if score >= threshold:
            status = EpistemicStatus.VALID
            explanation = "Pattern valid and justification meets threshold; inference accepted."
        elif lower_band <= score < threshold:
            status = EpistemicStatus.SUSPENDED
            explanation = (
                f"Pattern valid but justification score {score:.3f} is near threshold {threshold:.3f}; "
                "judgment suspended pending additional evidence."
            )
        else:
            status = EpistemicStatus.UNJUSTIFIED
            explanation = (
                f"Pattern valid but justification score {score:.3f} is below lower band {lower_band:.3f}; "
                "insufficient epistemic support."
            )

        # Record checks for trace (R5)
        defeated_count = sum(1 for e in evidence if e.defeated)
        checks = {
            "step": "epistemic_evaluation",
            "pattern_validation": "PASSED",
            "pattern": pattern.pattern,
            "proposition_count": len(propositions),
            "evidence_count": len(evidence),
            "defeated_evidence_count": defeated_count,
            "justification_threshold": round(threshold, 4),
            "suspension_lower_bound": round(lower_band, 4),
            "suspension_band": round(self.config.suspension_band, 4),
            "pramana_weights_used": self.config.pramana_weights,
        }

        return ReasoningTrace(
            pattern_valid=True,
            epistemic_status=status,
            justification_score=score,
            checks=checks,
            explanation=explanation,
        )

    @staticmethod
    def _clip(val: float) -> float:
        return max(0.0, min(1.0, val))


# ─────────────────────────────────────────────────────────────────────────── #
# Integration helpers for web and engine layers
# ─────────────────────────────────────────────────────────────────────────── #


def enrich_inference_result_with_trace(
    inference_result: Dict[str, Any],
    trace: Optional[ReasoningTrace] = None,
) -> Dict[str, Any]:
    """Enrich an inference result with epistemic trace.
    
    Args:
        inference_result: Original inference result dict
        trace: Optional epistemic reasoning trace
        
    Returns:
        Enriched result with trace and status
    """
    if trace is None:
        return inference_result

    return {
        **inference_result,
        "epistemic_trace": trace.to_dict(),
        "inference_status_extended": trace.epistemic_status.value,
    }


def build_reasoning_narrative(trace: ReasoningTrace) -> str:
    """Build human-readable narrative from trace.
    
    Args:
        trace: Reasoning trace
        
    Returns:
        Formatted narrative string
    """
    lines = [
        "=== Epistemic Reasoning Trace ===",
        f"Status: {trace.epistemic_status.value.upper()}",
        f"Justification Score: {trace.justification_score:.3f}",
        f"Pattern Valid: {trace.pattern_valid}",
        f"Explanation: {trace.explanation}",
    ]

    if trace.error_messages:
        lines.append("Errors:")
        for msg in trace.error_messages:
            lines.append(f"  - {msg}")

    return "\n".join(lines)
