from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Evidence, InferenceRequest, InferenceResult, InferenceStatus, Proposition, Rule


class PramanaInferenceEngine:
    """Inference engine where epistemic constraints gate logical entailment."""

    def __init__(self, evidence_base: Dict[str, Evidence], rules: Dict[str, Rule]):
        self.evidence_base = evidence_base
        self.rules = rules

    def infer(self, request: InferenceRequest) -> InferenceResult:
        trace = {
            "request": {
                "rule_id": request.rule_id,
                "premise_evidence_ids": request.premise_evidence_ids,
                "target": request.target.to_dict(),
            },
            "steps": [],
        }

        rule = self.rules.get(request.rule_id)
        if rule is None:
            trace["steps"].append(
                {
                    "stage": "rule_lookup",
                    "ok": False,
                    "reason": f"Unknown rule: {request.rule_id}",
                }
            )
            return InferenceResult(
                status=InferenceStatus.INVALID,
                accepted=False,
                target=request.target,
                trace=trace,
                message="Inference rule is unknown.",
            )

        premises, missing = self._collect_premises(request.premise_evidence_ids)
        if missing:
            trace["steps"].append(
                {
                    "stage": "premise_lookup",
                    "ok": False,
                    "missing_evidence_ids": missing,
                }
            )
            return InferenceResult(
                status=InferenceStatus.INVALID,
                accepted=False,
                target=request.target,
                trace=trace,
                message="Inference request references missing evidence.",
            )

        # ── R1: Represent propositions and supporting evidence programmatically ──
        trace["steps"].append(
            {
                "stage": "premise_lookup",
                "requirement": "R1",
                "ok": True,
                "propositions": [
                    {**p.proposition.to_dict(), "repr": p.proposition.to_repr(), "evidence_id": p.evidence_id}
                    for p in premises
                ],
                "evidence": [p.to_dict() for p in premises],
                "target_proposition": {**request.target.to_dict(), "repr": request.target.to_repr()},
                "rule": {
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "pattern": rule.pattern,
                    "required_pramanas": rule.required_pramanas,
                    "min_reliability": rule.min_reliability,
                    "suspension_margin": rule.suspension_margin,
                },
            }
        )

        # ── R4: Reject invalid inference patterns ──
        logical_ok, logical_reason = self._check_pattern(rule.pattern, premises, request.target)
        fallacy = (
            logical_reason
            if not logical_ok and ("invalid" in logical_reason.lower() or "detected" in logical_reason.lower())
            else None
        )
        trace["steps"].append(
            {
                "stage": "logical_pattern_check",
                "requirement": "R4",
                "ok": logical_ok,
                "pattern": rule.pattern,
                "reason": logical_reason,
                "fallacy_detected": fallacy,
            }
        )
        if not logical_ok:
            trace["steps"].append(
                {
                    "stage": "status_classification",
                    "requirement": "R3",
                    "status": InferenceStatus.INVALID.value,
                    "accepted": False,
                    "classification_reason": f"Pattern rejected: {logical_reason}",
                }
            )
            return InferenceResult(
                status=InferenceStatus.INVALID,
                accepted=False,
                target=request.target,
                trace=trace,
                message="Inference pattern is invalid.",
            )

        # ── R2: Explicitly check epistemic justification before accepting ──
        epistemic_status, epistemic_detail = self._check_epistemic_constraints(rule, premises)
        trace["steps"].append(
            {
                "stage": "epistemic_justification_check",
                "requirement": "R2",
                "ok": epistemic_status == InferenceStatus.VALID,
                "status": epistemic_status.value,
                "detail": epistemic_detail,
            }
        )

        # ── R3: Distinguish valid, unjustified, and suspended inferences ──
        _status_meanings = {
            InferenceStatus.VALID.value:       "Logical form valid and epistemic constraints satisfied — accepted as prama.",
            InferenceStatus.UNJUSTIFIED.value: "Missing required pramana sources — logically derivable but epistemically unsupported.",
            InferenceStatus.SUSPENDED.value:   "Evidence defeated or calibrated score near threshold — inference deferred.",
            InferenceStatus.INVALID.value:     "Inference form structurally rejected — not a valid inferential pattern.",
        }
        if epistemic_detail.get("missing_required_pramanas"):
            classification_reason = "Required pramana(s) absent: inference is unjustified."
        elif epistemic_detail.get("defeated_evidence_ids"):
            classification_reason = "Defeated evidence found: inference is suspended."
        elif epistemic_status == InferenceStatus.SUSPENDED:
            classification_reason = (
                f"Calibrated score {epistemic_detail.get('calibrated_score')} "
                f"is between suspension threshold {epistemic_detail.get('suspension_threshold')} "
                f"and acceptance threshold {epistemic_detail.get('min_reliability')}: suspended."
            )
        elif epistemic_status == InferenceStatus.VALID:
            classification_reason = (
                f"Calibrated score {epistemic_detail.get('calibrated_score')} "
                f">= acceptance threshold {epistemic_detail.get('min_reliability')}: valid."
            )
        else:
            classification_reason = (
                f"Calibrated score {epistemic_detail.get('calibrated_score')} "
                f"below suspension threshold {epistemic_detail.get('suspension_threshold')}: unjustified."
            )

        trace["steps"].append(
            {
                "stage": "status_classification",
                "requirement": "R3",
                "status": epistemic_status.value,
                "accepted": epistemic_status == InferenceStatus.VALID,
                "classification_reason": classification_reason,
                "status_meaning": _status_meanings[epistemic_status.value],
            }
        )

        if epistemic_status == InferenceStatus.VALID:
            return InferenceResult(
                status=InferenceStatus.VALID,
                accepted=True,
                target=request.target,
                trace=trace,
                message="Inference accepted: logical and epistemic constraints satisfied.",
            )

        return InferenceResult(
            status=epistemic_status,
            accepted=False,
            target=request.target,
            trace=trace,
            message="Logical form is valid, but epistemic constraints were not fully satisfied.",
        )

    def _collect_premises(self, evidence_ids: List[str]) -> Tuple[List[Evidence], List[str]]:
        premises: List[Evidence] = []
        missing: List[str] = []
        for evidence_id in evidence_ids:
            evidence = self.evidence_base.get(evidence_id)
            if evidence is None:
                missing.append(evidence_id)
            else:
                premises.append(evidence)
        return premises, missing

    def _check_pattern(self, pattern: str, premises: List[Evidence], target: Proposition) -> Tuple[bool, str]:
        atoms = {p.proposition.value for p in premises if p.proposition.kind == "atom"}
        implications = [p.proposition for p in premises if p.proposition.kind == "implies"]

        normalized_pattern = (pattern or "").strip().lower()
        if normalized_pattern == "modus_ponens":
            return self._check_modus_ponens(atoms, implications, target)
        if normalized_pattern == "modus_tollens":
            return self._check_modus_tollens(atoms, implications, target)
        if normalized_pattern == "hypothetical_syllogism":
            return self._check_hypothetical_syllogism(implications, target)
        if normalized_pattern == "vyapti_based_inference":
            # Vyapti (invariable concomitance) follows modus ponens structure:
            # hetu is present → sadhya necessarily follows (via vyapti)
            return self._check_modus_ponens(atoms, implications, target)
        return False, f"Unsupported pattern: {pattern}"

    @staticmethod
    def _negated(symbol: str) -> str:
        symbol = symbol.strip()
        if symbol.startswith("not_"):
            return symbol[4:]
        return f"not_{symbol}"

    def _check_modus_ponens(self, atoms: set[str], implications: List[Proposition], target: Proposition) -> Tuple[bool, str]:
        for implication in implications:
            if implication.antecedent in atoms and target.kind == "atom" and target.value == implication.consequent:
                return True, "Premises satisfy modus ponens."

        # Explicitly detect affirming the consequent as invalid pattern.
        if target.kind == "atom":
            for implication in implications:
                if implication.consequent in atoms and target.value == implication.antecedent:
                    return False, "Detected affirming the consequent, which is invalid."
                if self._negated(implication.antecedent) in atoms and target.value == self._negated(implication.consequent):
                    return False, "Detected denying the antecedent, which is invalid."

        return False, "Premises do not instantiate a valid modus ponens pattern."

    def _check_modus_tollens(self, atoms: set[str], implications: List[Proposition], target: Proposition) -> Tuple[bool, str]:
        if target.kind != "atom" or target.value is None:
            return False, "Modus tollens requires an atomic target proposition."

        for implication in implications:
            not_consequent = self._negated(implication.consequent or "")
            not_antecedent = self._negated(implication.antecedent or "")
            if not_consequent in atoms and target.value == not_antecedent:
                return True, "Premises satisfy modus tollens."

            # Explicitly reject affirming antecedent as a tollens proof attempt.
            if implication.antecedent in atoms and target.value == implication.consequent:
                return False, "Detected modus ponens shape while modus_tollens was requested."

        return False, "Premises do not instantiate a valid modus tollens pattern."

    def _check_hypothetical_syllogism(self, implications: List[Proposition], target: Proposition) -> Tuple[bool, str]:
        if target.kind != "implies":
            return False, "Hypothetical syllogism requires an implication target proposition."

        for left in implications:
            for right in implications:
                if left.consequent == right.antecedent:
                    if target.antecedent == left.antecedent and target.consequent == right.consequent:
                        return True, "Premises satisfy hypothetical syllogism."

        return False, "Premises do not instantiate a valid hypothetical syllogism pattern."

    def _check_epistemic_constraints(self, rule: Rule, premises: List[Evidence]) -> Tuple[InferenceStatus, Dict[str, object]]:
        pramanas_present = {p.pramana for p in premises}
        missing_pramanas = [p for p in rule.required_pramanas if p not in pramanas_present]

        reliabilities = [p.reliability for p in premises]
        avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.0

        weighted_components = []
        weighted_numerator = 0.0
        weighted_denominator = 0.0
        total_weight = 0.0
        for p in premises:
            weight = float(rule.pramana_weights.get(p.pramana, 1.0))
            weighted_components.append(
                {
                    "evidence_id": p.evidence_id,
                    "pramana": p.pramana,
                    "weight": round(weight, 4),
                    "reliability": round(p.reliability, 4),
                }
            )
            weighted_numerator += p.reliability * weight
            weighted_denominator += weight
            total_weight += weight
        weighted_reliability = (weighted_numerator / weighted_denominator) if weighted_denominator > 0 else avg_reliability
        authority_factor = (total_weight / len(premises)) if premises else 1.0
        # Use weighted reliability as the primary calibrated base so fusion across
        # multiple pramanas is not over-penalized by low-authority auxiliary channels.
        calibrated_base = weighted_reliability
        # Exponent < 1 is lenient (boosts scores), exponent = 1 is linear (default), exponent > 1 penalizes uncertainty.
        # Default calibration_exponent = 1.0 so score passes through unchanged.
        calibrated_score = calibrated_base ** max(rule.calibration_exponent, 0.1)

        defeated_ids = [p.evidence_id for p in premises if p.defeated]

        detail = {
            "pramanas_present": sorted(pramanas_present),
            "missing_required_pramanas": missing_pramanas,
            "avg_reliability": round(avg_reliability, 4),
            "weighted_reliability": round(weighted_reliability, 4),
            "authority_factor": round(authority_factor, 4),
            "calibrated_score": round(calibrated_score, 4),
            "pramana_weight_components": weighted_components,
            "calibration_exponent": round(max(rule.calibration_exponent, 0.1), 4),
            "min_reliability": rule.min_reliability,
            "suspension_threshold": max(rule.min_reliability - rule.suspension_margin, 0.0),
            "defeated_evidence_ids": defeated_ids,
        }

        if missing_pramanas:
            return InferenceStatus.UNJUSTIFIED, detail

        if defeated_ids:
            return InferenceStatus.SUSPENDED, detail

        suspension_threshold = max(rule.min_reliability - rule.suspension_margin, 0.0)
        if calibrated_score >= rule.min_reliability:
            return InferenceStatus.VALID, detail
        if calibrated_score >= suspension_threshold:
            return InferenceStatus.SUSPENDED, detail
        return InferenceStatus.UNJUSTIFIED, detail
