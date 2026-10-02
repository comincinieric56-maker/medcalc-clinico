from __future__ import annotations

from typing import Any, Dict


FUSION_VERSION = "MEDCALC_ECG_EVIDENCE_FUSION_V3_AVB1_SPECIALIST_PR_RESCUE"

# Prospective defaults. These are deliberately declared before any new external
# validation and must not be tuned against SPH, which is now a consumed
# external baseline.
POLICY = {
    "AF_COMPATIBLE": (0.70, 3),
    "FLUTTER_OR_AT_COMPATIBLE": (0.68, 3),
    "SINUS_BRADYCARDIA_COMPATIBLE": (0.70, 2),
    "SINUS_TACHYCARDIA_COMPATIBLE": (0.70, 2),
    "RBBB_MORPHOLOGY_COMPATIBLE": (0.65, 2),
    "LBBB_MORPHOLOGY_COMPATIBLE": (0.65, 3),
    "LAFB_COMPATIBLE": (0.65, 2),
    "LPFB_COMPATIBLE": (0.65, 2),
    "FIRST_DEGREE_AV_DELAY_COMPATIBLE": (0.72, 3),
    "MOBITZ_I_WENCKEBACH_COMPATIBLE": (0.74, 2),
    "MOBITZ_II_COMPATIBLE": (0.74, 2),
    "TWO_TO_ONE_AV_BLOCK_COMPATIBLE": (0.74, 2),
    "HIGH_GRADE_AV_BLOCK_COMPATIBLE": (0.78, 2),
    "COMPLETE_AV_BLOCK_COMPATIBLE": (0.80, 3),
    "VENTRICULAR_PREEXCITATION_COMPATIBLE": (0.78, 3),
}

NON_PUBLISHABLE_CANDIDATES = {
    "SECOND_DEGREE_AV_BLOCK_CANDIDATE",
}


def fuse_candidate_evidence(
    candidate_layer: Dict[str, Any],
    domain_gates: Dict[str, Any],
) -> Dict[str, Any]:
    rows = []
    domains = domain_gates.get("domains") or {}

    for candidate in candidate_layer.get("candidates") or []:
        row = dict(candidate)
        code = str(row.get("code") or "")
        domain = str(row.get("domain") or "")
        score = float(row.get("score") or 0.0)
        sources = int(row.get("independent_evidence_n") or 0)
        specialist = bool(row.get("specialist_confirmed"))
        gate = dict(domains.get(domain) or {})
        domain_ok = bool(gate.get("eligible"))
        if code in {
            "SINUS_BRADYCARDIA_COMPATIBLE",
            "SINUS_TACHYCARDIA_COMPATIBLE",
        }:
            rhythm_gate = dict(domains.get("RHYTHM") or {})
            domain_ok = domain_ok and bool(rhythm_gate.get("eligible"))
            gate = {
                **gate,
                "paired_rhythm_gate": rhythm_gate,
            }

        global_unusable = set(
            domain_gates.get("global_unusable_targets")
            or domain_gates.get("global_remeasure_targets")
            or []
        )
        required_measurements = set(row.get("required_measurements") or [])
        unresolved_required = sorted(required_measurements & global_unusable)
        if (
            code in {
                "SINUS_BRADYCARDIA_COMPATIBLE",
                "SINUS_TACHYCARDIA_COMPATIBLE",
            }
            and unresolved_required == ["r_peaks"]
            and bool(domain_gates.get("rate_consensus_rescue_active"))
        ):
            unresolved_required = []

        boundary_failures = []
        for requirement in row.get("boundary_requirements") or []:
            required_relation = str(requirement.get("required_relation") or "")
            actual_relation = str(requirement.get("actual_relation") or "")
            if required_relation and actual_relation != required_relation:
                boundary_failures.append(dict(requirement))

        threshold, min_sources = POLICY.get(code, (0.80, 3))
        if specialist:
            # Existing specialist confirmation remains valuable but is no
            # longer allowed to be silenced by an unrelated global conflict.
            threshold = min(threshold, 0.60)
            min_sources = min(min_sources, 2)

        evidence = {str(x) for x in (row.get("evidence") or [])}
        boundary_metrics = {
            str(item.get("metric") or "")
            for item in boundary_failures
            if str(item.get("metric") or "")
        }
        strong_avb1_specialist_signature = bool(
            code == "FIRST_DEGREE_AV_DELAY_COMPATIBLE"
            and specialist
            and {
                "1_TO_1_P_QRS",
                "PR_MEDIAN_GT_200MS",
                "PR_STABLE",
            }.issubset(evidence)
        )
        only_pr_measurement_block = bool(
            (
                unresolved_required == ["pr_ms"]
                and (not boundary_metrics or boundary_metrics == {"pr_ms"})
            )
            or (
                not unresolved_required
                and bool(boundary_failures)
                and boundary_metrics == {"pr_ms"}
            )
        )
        avb1_specialist_pr_rescue = bool(
            strong_avb1_specialist_signature
            and only_pr_measurement_block
            and domain_ok
            and not gate.get("blocked_by_conflicts")
            and score >= threshold
            and sources >= min_sources
        )

        if code in NON_PUBLISHABLE_CANDIDATES:
            publishable = False
            state = "CANDIDATE_REVIEW"
            reason = "GENERIC_CANDIDATE_REQUIRES_SPECIFIC_SUBTYPE_EVIDENCE"
        elif avb1_specialist_pr_rescue:
            publishable = True
            state = "ESTABLISHED_COMPATIBLE"
            reason = "AVB1_SPECIALIST_PR_EVIDENCE_RESCUE"
        elif unresolved_required:
            publishable = False
            state = "MEASUREMENT_ABSTENTION"
            reason = "REQUIRED_MEASUREMENT_UNUSABLE"
        elif boundary_failures:
            publishable = False
            state = "MEASUREMENT_BOUNDARY_UNCERTAIN"
            reason = "REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED"
        elif not domain_ok:
            publishable = False
            state = "DOMAIN_ABSTENTION"
            reason = "DOMAIN_GATE_BLOCKED"
        elif score >= threshold and sources >= min_sources:
            publishable = True
            state = "ESTABLISHED_COMPATIBLE" if specialist else "PROBABLE_COMPATIBLE"
            reason = "MULTISOURCE_EVIDENCE_FUSION_PASS"
        else:
            publishable = False
            state = "CANDIDATE_REVIEW"
            reason = "INSUFFICIENT_FUSED_EVIDENCE"

        row.update({
            "publishable": publishable,
            "fusion_state": state,
            "fusion_reason": reason,
            "prospective_score_threshold": threshold,
            "prospective_min_independent_sources": min_sources,
            "domain_gate": gate,
            "unresolved_required_measurements": unresolved_required,
            "boundary_failures": boundary_failures,
            "specialist_pr_rescue_applied": avb1_specialist_pr_rescue,
        })
        rows.append(row)

    by_code = {str(row.get("code") or ""): row for row in rows}
    publishable = [row for row in rows if row.get("publishable")]
    review = [row for row in rows if not row.get("publishable")]

    return {
        "version": FUSION_VERSION,
        "policy": "MULTISOURCE_EVIDENCE_FUSION_WITH_DOMAIN_SPECIFIC_ABSTENTION",
        "threshold_note": "PROSPECTIVE_DEFAULTS_DECLARED_WITHOUT_SPH_TUNING",
        "findings": rows,
        "publishable_findings": publishable,
        "review_candidates": review,
        "by_code": by_code,
    }
