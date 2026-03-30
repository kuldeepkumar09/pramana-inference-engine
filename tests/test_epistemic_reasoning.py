"""Regression tests for epistemic reasoning framework.

Tests all four inference outcomes:
- VALID: pattern valid and epistemically justified
- UNJUSTIFIED: pattern valid but insufficient support
- SUSPENDED: pattern valid but borderline support
- INVALID_PATTERN: pattern rejected structurally
"""

import pytest

from pramana_engine.epistemic_reasoning import (
    EpistemicStatus,
    EvidentialProposition,
    EvidentialSource,
    InferencePattern,
    EpistemicEvaluationConfig,
    EpistemicEvaluator,
    ReasoningTrace,
)


@pytest.fixture
def evaluator():
    """Provide configured evaluator."""
    config = EpistemicEvaluationConfig(
        min_justification_threshold=0.75,
        suspension_band=0.10,
    )
    return EpistemicEvaluator(config)


@pytest.fixture
def smoke_example():
    """Classic smoke-fire inference example."""
    prop = EvidentialProposition(
        id="p1",
        text="Smoke is present on the hill",
        kind="atom",
        source="observation",
        confidence=0.85,
    )

    ev = EvidentialSource(
        id="e1",
        proposition_id="p1",
        kind="perception",
        content="Observer directly saw gray smoke rising from the hill",
        reliability=0.80,
        pramana="perception",
    )

    conclusion = EvidentialProposition(
        id="c1",
        text="Fire is present on the hill",
        kind="atom",
        source="inference",
        confidence=0.0,
    )

    pattern = InferencePattern(
        name="Smoke-Fire Inference",
        pattern="smoke_implies_fire",
        valid_patterns=["smoke_implies_fire", "effect_to_cause"],
        description="Classical inferential pattern: smoke normally indicates fire",
    )

    return {
        "prop": prop,
        "ev": ev,
        "conclusion": conclusion,
        "pattern": pattern,
    }


class TestValidInference:
    """R3: Valid inference when pattern and justification both satisfy constraints."""

    def test_high_confidence_evidence_passes_threshold(self, evaluator, smoke_example):
        """VALID: Strong evidence and high-confidence proposition."""
        trace = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[smoke_example["ev"]],
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        assert trace.epistemic_status == EpistemicStatus.VALID
        assert trace.pattern_valid is True
        assert trace.justification_score >= 0.75
        assert "accepted" in trace.explanation.lower()

    def test_multiple_corroborating_evidence(self, evaluator):
        """VALID: Multiple evidence sources increase score."""
        prop = EvidentialProposition("p1", "Premise A", "atom", "observation", 0.80)
        
        ev1 = EvidentialSource("e1", "p1", "perception", "Direct observation", 0.85, pramana="perception")
        ev2 = EvidentialSource("e2", "p1", "testimony", "Expert testimony", 0.70, pramana="testimony")
        
        conclusion = EvidentialProposition("c1", "Conclusion B", "atom", "inference", 0.0)
        pattern = InferencePattern("test", "infer_b_from_a", ["infer_b_from_a"], "")

        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[ev1, ev2],
            conclusion=conclusion,
            pattern=pattern,
        )

        assert trace.epistemic_status == EpistemicStatus.VALID
        assert trace.justification_score > 0.7


class TestUnjustifiedInference:
    """R3: Unjustified when pattern is valid but epistemic support is insufficient."""

    def test_low_confidence_proposition_fails(self, evaluator, smoke_example):
        """UNJUSTIFIED: Weak proposition confidence below threshold."""
        weak_prop = EvidentialProposition(
            id="p1",
            text="Smoke might be present",
            kind="atom",
            source="hearsay",
            confidence=0.20,  # Weak
        )

        trace = evaluator.evaluate_inference(
            propositions=[weak_prop],
            evidence=[smoke_example["ev"]],
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        assert trace.epistemic_status == EpistemicStatus.UNJUSTIFIED
        assert trace.pattern_valid is True
        assert trace.justification_score < 0.75

    def test_poor_evidence_reliability(self, evaluator, smoke_example):
        """UNJUSTIFIED: All evidence unreliable."""
        weak_ev = EvidentialSource(
            id="e1",
            proposition_id="p1",
            kind="speculation",
            content="Random guess about hill",
            reliability=0.15,  # Very weak
            pramana="postulation",
        )

        trace = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[weak_ev],
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        assert trace.epistemic_status == EpistemicStatus.UNJUSTIFIED
        assert trace.justification_score < 0.75

    def test_no_evidence_fails(self, evaluator, smoke_example):
        """UNJUSTIFIED: No supporting evidence."""
        trace = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[],  # Empty
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        assert trace.epistemic_status == EpistemicStatus.UNJUSTIFIED
        assert trace.justification_score == 0.0


class TestSuspendedInference:
    """R3: Suspended when pattern is valid but score is near threshold (borderline)."""

    def test_borderline_high_score(self, evaluator):
        """SUSPENDED: Score in suspension band above lower threshold."""
        # Calibrated so score lands in (0.65, 0.75) band
        prop = EvidentialProposition("p1", "Possibly true", "atom", "inference", 0.68)
        ev = EvidentialSource("e1", "p1", "comparison", "Similar case", 0.70, pramana="comparison")
        
        conclusion = EvidentialProposition("c1", "Conclusion", "atom", "inference", 0.0)
        pattern = InferencePattern("test", "analogical", ["analogical"], "")

        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        assert trace.epistemic_status == EpistemicStatus.SUSPENDED
        assert trace.pattern_valid is True
        # Score should be > lower band but < threshold
        assert trace.justification_score < 0.75
        assert trace.justification_score > 0.65

    def test_borderline_low_score(self, evaluator):
        """SUSPENDED: Score just at lower suspension boundary."""
        prop = EvidentialProposition("p1", "Premise", "atom", "observation", 0.67)
        ev = EvidentialSource("e1", "p1", "testimony", "Weak testimony", 0.68, pramana="testimony")
        
        conclusion = EvidentialProposition("c1", "Conclusion", "atom", "inference", 0.0)
        pattern = InferencePattern("test", "from_testimony", ["from_testimony"], "")

        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        assert trace.epistemic_status == EpistemicStatus.SUSPENDED
        assert trace.pattern_valid is True


class TestInvalidPatternRejection:
    """R4: Reject invalid inference patterns before epistemic evaluation."""

    def test_affirming_the_consequent_rejected(self, evaluator):
        """INVALID_PATTERN: Affirming the consequent (classic fallacy)."""
        prop = EvidentialProposition("p1", "High confidence", "atom", "observation", 0.99)
        ev = EvidentialSource("e1", "p1", "perception", "Direct observation", 0.99, pramana="perception")
        
        conclusion = EvidentialProposition("c1", "Conclusion", "atom", "inference", 0.0)
        invalid_pattern = InferencePattern(
            "bad_pattern",
            "affirming_the_consequent",  # Invalid
            ["modus_ponens", "modus_tollens"],  # Only these allowed
        )

        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[ev],
            conclusion=conclusion,
            pattern=invalid_pattern,
        )

        assert trace.epistemic_status == EpistemicStatus.INVALID_PATTERN
        assert trace.pattern_valid is False
        assert trace.justification_score == 0.0
        assert len(trace.error_messages) > 0

    def test_denying_the_antecedent_rejected(self, evaluator):
        """INVALID_PATTERN: Denying the antecedent (formal fallacy)."""
        prop = EvidentialProposition("p1", "Strong", "atom", "observation", 0.95)
        ev = EvidentialSource("e1", "p1", "perception", "Clear observation", 0.95, pramana="perception")
        
        conclusion = EvidentialProposition("c1", "Conclusion", "atom", "inference", 0.0)
        invalid_pattern = InferencePattern(
            "fallacy",
            "denying_the_antecedent",
            ["modus_ponens", "hypothetical_syllogism"],
        )

        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[ev],
            conclusion=conclusion,
            pattern=invalid_pattern,
        )

        assert trace.epistemic_status == EpistemicStatus.INVALID_PATTERN
        assert trace.pattern_valid is False
        assert "not in allowed" in trace.explanation.lower() or "invalid" in trace.explanation.lower()


class TestDefeatedEvidence:
    """Evidence can be explicitly marked as defeated (contradicted)."""

    def test_defeated_evidence_reduces_score(self, evaluator, smoke_example):
        """Defeated evidence counts as zero contribution."""
        prop_id = smoke_example['prop'].id
        
        strong_ev = EvidentialSource(
            id="e1",
            proposition_id=prop_id,
            kind="perception",
            content="Smoke observed",
            reliability=0.90,
            defeated=False,
            pramana="perception",
        )

        defeated_ev = EvidentialSource(
            id="e2",
            proposition_id=prop_id,
            kind="testimony",
            content="Actually no smoke",
            reliability=0.85,
            defeated=True,  # Marked as defeated
            pramana="testimony",
        )

        conclusion = smoke_example["conclusion"]
        pattern = smoke_example["pattern"]

        # With only strong evidence
        trace_strong_only = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[strong_ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        # With strong + defeated evidence
        trace_with_defeat = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[strong_ev, defeated_ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        # Defeated evidence should reduce overall score
        assert trace_with_defeat.justification_score <= trace_strong_only.justification_score


class TestPramanaWeighting:
    """Pramanas are weighted by reliability (perception > inference > testimony, etc.)."""

    def test_perception_weighted_higher_than_testimony(self, evaluator):
        """Perception (higher authority) is included in weighted calculations."""
        prop = EvidentialProposition("p1", "Proposition", "atom", "observation", 0.70)

        perception_ev = EvidentialSource(
            id="e1", proposition_id="p1", kind="perception", content="Direct observation",
            reliability=0.65, pramana="perception"
        )
        testimony_ev = EvidentialSource(
            id="e2", proposition_id="p1", kind="testimony", content="Someone said",
            reliability=0.65, pramana="testimony"
        )

        conclusion = EvidentialProposition("c1", "Conclusion", "atom", "inference", 0.0)
        pattern = InferencePattern("test", "infer", ["infer"], "")

        # Mixed evidence
        trace = evaluator.evaluate_inference(
            propositions=[prop],
            evidence=[perception_ev, testimony_ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        # Verify pramana weights are in checks
        assert "pramana_weights_used" in trace.checks
        pramana_weights = trace.checks["pramana_weights_used"]
        assert pramana_weights["perception"] > pramana_weights["testimony"]
        assert pramana_weights["perception"] == 1.0
        assert pramana_weights["testimony"] == 0.6


class TestMachineReadableTrace:
    """R5: Traces must be machine-readable JSON/dict format."""

    def test_trace_serialization(self, evaluator, smoke_example):
        """Trace converts to machine-readable dict."""
        trace = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[smoke_example["ev"]],
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        data = trace.to_dict()

        # Verify all required fields
        assert isinstance(data, dict)
        assert "pattern_valid" in data
        assert "status" in data
        assert data["status"] in ["valid", "unjustified", "suspended", "invalid_pattern"]
        assert "justification_score" in data
        assert isinstance(data["justification_score"], float)
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        assert "explanation" in data
        assert "requirements_met" in data
        assert all(k.startswith("R") for k in data["requirements_met"].keys())

    def test_trace_requirements_marked(self, evaluator, smoke_example):
        """Trace explicitly marks all requirements as met."""
        trace = evaluator.evaluate_inference(
            propositions=[smoke_example["prop"]],
            evidence=[smoke_example["ev"]],
            conclusion=smoke_example["conclusion"],
            pattern=smoke_example["pattern"],
        )

        data = trace.to_dict()
        reqs = data["requirements_met"]

        assert reqs["R1_proposition_representation"] is True
        assert reqs["R2_epistemic_justification_check"] is True
        assert reqs["R3_status_distinction"] in ["valid", "unjustified", "suspended", "invalid_pattern"]
        assert isinstance(reqs["R4_invalid_patterns_rejected"], bool)
        assert reqs["R5_machine_readable_trace"] is True


# ─────────────────────────────────────────────────────────────────────────── #
# Integration tests with known wrong-answer scenarios (regression)
# ─────────────────────────────────────────────────────────────────────────── #


class TestKnownWrongAnswers:
    """Regression tests: ensure system catches common wrong-answer patterns."""

    def test_missing_intermediary_premises(self, evaluator):
        """UNJUSTIFIED: Jumping to conclusion without intermediary steps."""
        weak_premise = EvidentialProposition(
            "p_skip",
            "Entity A exists",
            "atom",
            "observation",
            0.50,  # Weak because next step not established
        )
        weak_ev = EvidentialSource(
            "e_skip", "p_skip", "inference", "Assumed without proof", 0.40, pramana="inference"
        )

        conclusion = EvidentialProposition(
            "c_wrong", "Entity C has property X (wrong jump)", "atom", "inference", 0.0
        )
        pattern = InferencePattern("bad_jump", "chain", ["chain"], "")

        trace = evaluator.evaluate_inference(
            propositions=[weak_premise],
            evidence=[weak_ev],
            conclusion=conclusion,
            pattern=pattern,
        )

        # Should caught as unjustified due to weak chain
        assert trace.epistemic_status == EpistemicStatus.UNJUSTIFIED

    def test_circular_reasoning_flagged(self, evaluator):
        """SUSPENDED: Circularity reduces confidence but may not fully reject."""
        premise = EvidentialProposition(
            "p_circ", "Accept claim X", "atom", "testimony", 0.60
        )
        evidence = EvidentialSource(
            "e_circ", "p_circ", "testimony", "Source relies on claim X", 0.60, pramana="testimony"
        )

        conclusion = EvidentialProposition(
            "c_circ", "Therefore claim X is true", "atom", "inference", 0.0
        )
        pattern = InferencePattern("circular", "reaffirm", ["reaffirm"], "")

        trace = evaluator.evaluate_inference(
            propositions=[premise],
            evidence=[evidence],
            conclusion=conclusion,
            pattern=pattern,
        )

        # Circularity should lower confidence into suspended range
        assert trace.epistemic_status in [
            EpistemicStatus.SUSPENDED,
            EpistemicStatus.UNJUSTIFIED,
        ]
        assert trace.justification_score < 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
