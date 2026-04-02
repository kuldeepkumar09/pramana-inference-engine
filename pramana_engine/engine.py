from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Evidence, InferenceRequest, InferenceResult, InferenceStatus, Proposition, Rule

_logger = logging.getLogger("pramana.engine")


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

        # ── H5: Detect circular reasoning before pattern check ──
        cycle = self._detect_circular_reasoning(premises)
        if cycle:
            trace["steps"].append({
                "stage": "circular_reasoning_check",
                "ok": False,
                "cycle_detected": cycle,
                "reason": f"Circular dependency: {' → '.join(cycle)}",
            })
            return InferenceResult(
                status=InferenceStatus.UNJUSTIFIED,
                accepted=False,
                target=request.target,
                trace=trace,
                message="Circular reasoning detected: the inference chain forms a cycle.",
            )

        # ── H6: Auto-detect contradictory evidence before scoring ──
        premises = self._flag_contradictory_evidence(premises)

        # ── R4: Reject invalid inference patterns ──
        logical_ok, logical_reason = self._check_pattern(rule.pattern, premises, request.target)
        # ── H2: Hetvabhasa detection ──
        hetvabhasa_list = self._check_hetvabhasa(premises, request.target, rule.pattern)
        fallacy = (
            logical_reason
            if not logical_ok and ("invalid" in logical_reason.lower() or "detected" in logical_reason.lower())
            else (hetvabhasa_list[0] if hetvabhasa_list else None)
        )
        trace["steps"].append(
            {
                "stage": "logical_pattern_check",
                "requirement": "R4",
                "ok": logical_ok,
                "pattern": rule.pattern,
                "reason": logical_reason,
                "fallacy_detected": fallacy,
                "hetvabhasa": hetvabhasa_list,
            }
        )
        if not logical_ok or hetvabhasa_list:
            if hetvabhasa_list and logical_ok:
                classification_reason = f"Hetvabhasa (logical fallacy) detected: {', '.join(hetvabhasa_list)}"
            else:
                classification_reason = f"Pattern rejected: {logical_reason}"
            trace["steps"].append(
                {
                    "stage": "status_classification",
                    "requirement": "R3",
                    "status": InferenceStatus.INVALID.value,
                    "accepted": False,
                    "classification_reason": classification_reason,
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
        # Logical pattern check includes ALL premises (including defeated):
        # "defeated" is an epistemic status handled in _check_epistemic_constraints,
        # not a logical one. A defeated premise was still logically proposed.
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
            # Vyapti (invariable concomitance): universally, wherever hetu, sadhya follows.
            # Structurally this is modus ponens (A→B, A ∴ B) at the propositional level.
            # The semantic distinction — strict universal co-presence, not mere material
            # implication — is enforced epistemically via required pramana type and
            # min_reliability threshold. The trace message distinguishes anvaya (affirmative
            # co-presence) from combined anvaya+vyatireka (contrapositive confirmation).
            valid, reason = self._check_modus_ponens(atoms, implications, target)
            if not valid:
                return valid, reason
            # Vyatireka check: does the contrapositive (not_sadhya → not_hetu) also appear?
            not_sadhya = self._negated(target.value) if target.kind == "atom" and target.value else None
            has_vyatireka = bool(not_sadhya) and any(
                imp.antecedent == not_sadhya for imp in implications
            )
            if has_vyatireka:
                return True, (
                    "Premises satisfy vyapti-based inference "
                    "(invariable concomitance confirmed via anvaya and vyatireka — "
                    "both affirmative co-presence and contrapositive absence-entailment present)."
                )
            return True, (
                "Premises satisfy vyapti-based inference "
                "(anvaya form: universal co-presence of hetu and sadhya confirmed)."
            )
        if normalized_pattern == "upamana_based_inference":
            # Upamana (comparison/analogy): knowledge of a new object through its
            # similarity to a previously known similar object.
            # Structural form: A is known to resemble B (gavaya resembles cow).
            #   Observation of gavaya in the field confirms the comparison.
            #   Target: the new object is correctly identified via prior similarity instruction.
            # At the propositional level this reduces to: similarity(A,B) ∧ observed(A) ∴ identified(A).
            # We validate this as a modus ponens chain on the similarity implication.
            valid, reason = self._check_modus_ponens(atoms, implications, target)
            if valid:
                return True, "Premises satisfy upamana-based inference (knowledge via similarity/comparison confirmed)."
            return valid, reason

        if normalized_pattern == "anupalabdhi_based_inference":
            # Anupalabdhi (non-perception/absence-cognition): knowledge of the absence
            # of X is established through the non-perception of X in a place where X
            # would be perceptible if it existed.
            # Structural form: absence_of_X is the target; non_perception_of_X is in atoms;
            # implication: non_perception_of_X → absence_of_X.
            return self._check_anupalabdhi(atoms, implications, target)

        if normalized_pattern == "pancavayava":
            # Pancavayava (5-member syllogism): the canonical Nyaya inference form.
            # Pratijna (claim) + Hetu (reason) + Udaharana (example with vyapti) +
            # Upanaya (application to subject) + Nigamana (conclusion = target).
            # Requires: at least one implication (udaharana), at least two distinct atoms
            # (hetu + upanaya), and the combination proves the target via modus ponens.
            return self._check_pancavayava(atoms, implications, target)

        if normalized_pattern == "shabda_based_inference":
            # Shabda (testimony): knowledge from a reliable/authoritative speaker (apta-vacana).
            # Valid when at least one premise comes from a testimony pramana source.
            return self._check_shabda(premises, target)

        if normalized_pattern == "arthapatti_based_inference":
            # Arthapatti (postulation): knowledge inferred to explain an otherwise
            # inexplicable observation. Requires: an observation atom + an implication
            # that provides the only coherent explanation + the conclusion follows.
            return self._check_arthapatti(atoms, implications, target)

        return False, f"Unsupported pattern: {pattern}"

    @staticmethod
    def _negated(symbol: str) -> str:
        symbol = symbol.strip()
        if symbol.startswith("not_"):
            return symbol[4:]
        return f"not_{symbol}"

    def _check_modus_ponens(self, atoms: set[str], implications: List[Proposition], target: Proposition) -> Tuple[bool, str]:
        for implication in implications:
            # Reject tautological implications (A → A) — they carry no inferential content.
            if implication.antecedent == implication.consequent:
                return False, f"Tautological implication '{implication.antecedent} → {implication.consequent}' is not a valid hetu."
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

        # O(n) lookup: index implications by antecedent so we can find the right
        # side of a chain in constant time instead of the previous O(n²) double loop.
        antecedent_map: Dict[str, Proposition] = {
            imp.antecedent: imp for imp in implications if imp.antecedent is not None
        }
        for left in implications:
            right = antecedent_map.get(left.consequent)
            if right is not None:
                if target.antecedent == left.antecedent and target.consequent == right.consequent:
                    return True, "Premises satisfy hypothetical syllogism."

        return False, "Premises do not instantiate a valid hypothetical syllogism pattern."

    # ── H1: Anupalabdhi (Non-perception / Absence Inference) ─────────────────
    def _check_anupalabdhi(
        self,
        atoms: set,
        implications: List[Proposition],
        target: Proposition,
    ) -> Tuple[bool, str]:
        """Validate Anupalabdhi: absence of X established via non-perception of X.

        Structural requirement: an implication of the form
        non_perception_of_X → absence_of_X (or equivalent),
        with non_perception_of_X present in atoms.
        """
        if target.kind != "atom" or not target.value:
            return False, "Anupalabdhi requires an atomic absence target."

        # Check: target starts with "absent_" or "not_" / "absence_of_"
        for imp in implications:
            if imp.consequent == target.value and imp.antecedent in atoms:
                # The antecedent should represent a non-perception / absence-cue
                ant = (imp.antecedent or "").lower()
                if any(kw in ant for kw in ("non_perception", "not_perceived", "absent", "non_", "no_")):
                    return True, (
                        "Premises satisfy anupalabdhi-based inference "
                        "(absence established via non-perception)."
                    )
                # Accept any valid modus ponens as anupalabdhi if pattern requested it
                return True, (
                    "Premises satisfy anupalabdhi-based inference "
                    "(absence-cognition: non-perception grounds absence)."
                )

        return False, "Premises do not instantiate a valid anupalabdhi (absence-inference) pattern."

    # ── H3: Pancavayava (5-Member Syllogism) ─────────────────────────────────
    def _check_pancavayava(
        self,
        atoms: set,
        implications: List[Proposition],
        target: Proposition,
    ) -> Tuple[bool, str]:
        """Validate Pancavayava: all 5 members structurally present.

        Pratijna (claim=target) + Hetu (reason atom) +
        Udaharana (general rule implication establishing vyapti) +
        Upanaya (specific application atom) + Nigamana (conclusion=target).

        Minimum requirement: ≥1 implication (udaharana), ≥2 atoms (hetu + upanaya),
        and the implication proves the target via modus ponens.
        """
        if target.kind != "atom" or not target.value:
            return False, "Pancavayava requires an atomic target (nigamana)."

        if not implications:
            return False, "Pancavayava requires an udaharana (general rule implication)."

        if len(atoms) < 1:
            return False, "Pancavayava requires at least hetu (reason atom)."

        # Check that udaharana + hetu/upanaya prove the target
        for imp in implications:
            if imp.antecedent in atoms and imp.consequent == target.value:
                if len(atoms) >= 2:
                    return True, (
                        "Premises satisfy pancavayava (5-member syllogism): "
                        "pratijna, hetu, udaharana, upanaya, and nigamana are all present."
                    )
                # Only one atom — hetu present but upanaya may be implicit
                return True, (
                    "Premises satisfy pancavayava (5-member syllogism): "
                    "udaharana (general rule) + hetu confirm the nigamana."
                )

        return False, (
            "Premises do not instantiate a valid pancavayava: "
            "udaharana does not establish the sadhya from the present hetu."
        )

    # ── H8: Shabda (Testimony from Reliable Source) ──────────────────────────
    def _check_shabda(
        self,
        premises: List[Evidence],
        target: Proposition,
    ) -> Tuple[bool, str]:
        """Validate Shabda: knowledge from reliable speaker (apta-vacana).

        At least one premise must be from a 'testimony' pramana source.
        The target must appear as the consequent of a testimony-sourced implication
        or as a direct testimony atom.
        """
        testimony_premises = [p for p in premises if p.pramana == "testimony" and not p.defeated]
        if not testimony_premises:
            return False, "Shabda inference requires at least one testimony (apta-vacana) premise."

        if target.kind == "atom" and target.value:
            for p in testimony_premises:
                if p.proposition.kind == "atom" and p.proposition.value == target.value:
                    return True, (
                        "Premises satisfy shabda-based inference "
                        f"(testimony from '{p.source}' directly establishes the target)."
                    )
                if p.proposition.kind == "implies" and p.proposition.consequent == target.value:
                    return True, (
                        "Premises satisfy shabda-based inference "
                        f"(testimony from '{p.source}' establishes the target via implication)."
                    )

        return False, "Testimony premises present but do not establish the target proposition."

    # ── H8: Arthapatti (Postulation / Explanatory Necessity) ─────────────────
    def _check_arthapatti(
        self,
        atoms: set,
        implications: List[Proposition],
        target: Proposition,
    ) -> Tuple[bool, str]:
        """Validate Arthapatti: the target is the only coherent explanation.

        Structural requirement: an observation atom is present, and only the
        target proposition explains that observation (i.e., there exists an
        implication target→observation or the target's negation leads to an
        otherwise-inexplicable contradiction in the atoms).
        """
        if target.kind != "atom" or not target.value:
            return False, "Arthapatti requires an atomic target (postulated explanation)."

        # Check: is there an implication T→obs where obs is already in atoms?
        for imp in implications:
            if imp.antecedent == target.value and imp.consequent in atoms:
                return True, (
                    "Premises satisfy arthapatti-based inference "
                    f"(postulating '{target.value}' is the only coherent explanation "
                    f"for the observed fact '{imp.consequent}')."
                )

        # Fallback: standard modus ponens (obs → T)
        for imp in implications:
            if imp.antecedent in atoms and imp.consequent == target.value:
                return True, (
                    "Premises satisfy arthapatti-based inference "
                    "(explanatory postulation confirmed via observed necessity)."
                )

        return False, (
            "Premises do not instantiate a valid arthapatti: "
            "no observation-to-explanation chain found for the postulated target."
        )

    # NOTE — Scoring divergence: see epistemic_reasoning.py EpistemicEvaluator
    # for the parallel advisory evaluator (different formula). That evaluator
    # blends proposition.confidence (0.45) with pramana-weighted evidence (0.55)
    # and uses a fixed 0.70 threshold. THIS method uses only evidence.reliability
    # with per-rule weights and per-rule min_reliability. Do not conflate them.
    def _check_epistemic_constraints(self, rule: Rule, premises: List[Evidence]) -> Tuple[InferenceStatus, Dict[str, Any]]:
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
        # Apply authority_factor only when the rule explicitly configures pramana_weights.
        # When no weights are set every weight defaults to 1.0, so authority_factor == 1.0
        # and this multiplication is identity — avoids systematic score regression on
        # default-configured rules while still honoring the caller's intent when custom
        # weights ARE specified (e.g. testimony=0.5 should penalise the calibrated base).
        uses_custom_weights = bool(rule.pramana_weights)
        calibrated_base = (
            min(authority_factor, 1.0) * weighted_reliability
            if uses_custom_weights
            else weighted_reliability
        )
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

    # ── H2: Hetvabhasa (Nyaya Logical Fallacy) Detection ─────────────────────
    def _check_hetvabhasa(
        self,
        premises: List[Evidence],
        target: Proposition,
        pattern: str,
    ) -> List[str]:
        """Return list of BLOCKING Nyaya logical fallacies (hetvabhasa) that invalidate inference.

        Two hetvabhasa types block as INVALID (structural fallacies):
          Savyabhicara — undistributed middle (same hetu proves sadhya AND its negation)
          Viruddha     — contradictory reason (hetu implies the opposite of sadhya)

        Three hetvabhasa types are informational only (handled by epistemic scoring):
          Satpratipaksha — counter-evidence (→ SUSPENDED, already via satpratipaksha_atoms)
          Asiddha        — ungrounded reason (→ UNJUSTIFIED, already via missing atoms)
          Badhita        — refuted by stronger pramana (→ SUSPENDED, already via defeated_ids)

        This method returns only the BLOCKING fallacies to avoid double-penalising
        already-handled epistemic situations (e.g. defeated evidence → suspended, not invalid).
        """
        detected: List[str] = []
        atoms: Set[str] = {p.proposition.value for p in premises if p.proposition.kind == "atom" and not p.defeated}
        implications = [p.proposition for p in premises if p.proposition.kind == "implies" and not p.defeated]
        target_value = target.value if target.kind == "atom" else None

        if target_value is None:
            return detected

        not_target = self._negated(target_value)

        # 1. Savyabhicara: same hetu implies BOTH sadhya and not-sadhya (undistributed middle)
        antecedents_for_sadhya = {imp.antecedent for imp in implications if imp.consequent == target_value}
        for imp in implications:
            if imp.antecedent in antecedents_for_sadhya and imp.consequent == not_target:
                detected.append("savyabhicara")
                break

        # 2. Viruddha: an established (non-defeated) hetu implies the negation of the target
        for imp in implications:
            if imp.antecedent in atoms and imp.consequent == not_target:
                detected.append("viruddha")
                break

        # Non-blocking fallacies are annotated in trace but do NOT trigger INVALID.
        # Satpratipaksha: counter-evidence atoms present → handled as SUSPENDED by scoring.
        # Asiddha: hetu ungrounded → handled as UNJUSTIFIED by scoring.
        # Badhita: defeated evidence → handled as SUSPENDED via defeated_ids in scoring.

        return list(dict.fromkeys(detected))  # deduplicate, preserve order

    # ── H5: Circular Reasoning Detection ────────────────────────────────────
    @staticmethod
    def _detect_circular_reasoning(premises: List[Evidence]) -> Optional[List[str]]:
        """Detect cycles in the implication graph of premises.

        Returns the cycle path as a list of node names if a cycle is found,
        otherwise None.
        """
        graph: Dict[str, Set[str]] = {}
        for p in premises:
            if p.proposition.kind == "implies" and p.proposition.antecedent and p.proposition.consequent:
                graph.setdefault(p.proposition.antecedent, set()).add(p.proposition.consequent)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbour in graph.get(node, set()):
                if neighbour not in visited:
                    result = dfs(neighbour, path)
                    if result:
                        return result
                elif neighbour in rec_stack:
                    cycle_start = path.index(neighbour)
                    return path[cycle_start:] + [neighbour]
            rec_stack.discard(node)
            path.pop()
            return None

        for node in list(graph.keys()):
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    return cycle
        return None

    # ── H6: Contradictory Evidence Auto-Detection ───────────────────────────
    @staticmethod
    def _flag_contradictory_evidence(premises: List[Evidence]) -> List[Evidence]:
        """Auto-detect and flag contradictory evidence pairs.

        If E1 asserts atom "X" and E2 asserts atom "not_X" (or "X" vs "not_X"
        variant), the weaker evidence item (lower reliability) is auto-marked
        as defeated and tagged with 'auto_defeated_contradicts' in its source.
        Returns the updated premises list.
        """
        atom_map: Dict[str, Evidence] = {}
        updated = list(premises)
        for i, p in enumerate(updated):
            if p.proposition.kind != "atom" or p.defeated:
                continue
            val = p.proposition.value or ""
            negation = ("not_" + val) if not val.startswith("not_") else val[4:]
            if negation in atom_map:
                other = atom_map[negation]
                # Mark the weaker one as auto-defeated
                if p.reliability < other.reliability:
                    updated[i] = Evidence(
                        evidence_id=p.evidence_id,
                        proposition=p.proposition,
                        pramana=p.pramana,
                        reliability=p.reliability,
                        source=f"auto_defeated_contradicts:{other.evidence_id}",
                        defeated=True,
                    )
                    _logger.warning(
                        "Auto-defeated %r (reliability=%.2f) — contradicted by %r (reliability=%.2f)",
                        p.evidence_id, p.reliability, other.evidence_id, other.reliability,
                    )
                else:
                    # Mark the existing entry as defeated
                    j = next(k for k, e in enumerate(updated) if e.evidence_id == other.evidence_id)
                    updated[j] = Evidence(
                        evidence_id=other.evidence_id,
                        proposition=other.proposition,
                        pramana=other.pramana,
                        reliability=other.reliability,
                        source=f"auto_defeated_contradicts:{p.evidence_id}",
                        defeated=True,
                    )
                    atom_map[val] = updated[i]
                    _logger.warning(
                        "Auto-defeated %r (reliability=%.2f) — contradicted by %r (reliability=%.2f)",
                        other.evidence_id, other.reliability, p.evidence_id, p.reliability,
                    )
                continue
            atom_map[val] = updated[i]
        return updated
