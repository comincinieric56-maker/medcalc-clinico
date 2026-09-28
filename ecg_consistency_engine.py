from __future__ import annotations

from typing import Any, Dict


CONSISTENCY_VERSION = "MEDCALC_ECG_CONSISTENCY_ENGINE_V2"


def evaluate_ecg_consistency(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
) -> Dict[str, Any]:
    """Find internal contradictions before any diagnostic label is published."""
    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    wct = specialists.get("wide_complex_tachycardia") or {}
    fascicular = specialists.get("fascicular_conduction") or {}
    consensus = specialists.get("measurement_consensus") or {}
    signal_integrity = specialists.get("signal_integrity") or {}
    ectopy = specialists.get("ectopy") or {}
    av = specialists.get("av_conduction") or {}
    preexcitation = specialists.get("preexcitation") or {}
    rhythm = feature_graph.get("rhythm") or {}

    conflicts: list[Dict[str, Any]] = []
    remeasure = list(consensus.get("remeasure_targets") or [])
    unmeasurable = list(consensus.get("unmeasurable_targets") or [])
    uncertain = list(consensus.get("uncertain_targets") or [])
    unusable = list(consensus.get("unusable_targets") or sorted(set(remeasure) | set(unmeasurable)))

    p_repro = bool(atrial.get("p_wave_reproducible"))
    coupling = float(atrial.get("rhythm_p_qrs_coupling_fraction") or 0.0)
    sinus = bool(atrial.get("sinus_compatible"))
    mechanism = str(atrial_mech.get("mechanism") or "")
    rr_regular = rhythm.get("regular")

    if mechanism == "AF_COMPATIBLE" and p_repro and coupling >= 0.70:
        conflicts.append({
            "code": "AF_VS_REPRODUCIBLE_P_QRS_CONFLICT",
            "severity": "BLOCKING",
            "action": "REVIEW_ATRIAL_MECHANISM",
        })

    if sinus and not p_repro:
        conflicts.append({
            "code": "SINUS_WITHOUT_REPRODUCIBLE_P_CONFLICT",
            "severity": "BLOCKING",
            "action": "SUPPRESS_SINUS_LABEL",
        })

    if (
        str(crosslead_conduction.get("classification") or "")
        == "CONDUCTION_MORPHOLOGY_CONFLICT"
    ):
        conflicts.append({
            "code": "RBBB_LBBB_MUTUAL_CONFLICT",
            "severity": "BLOCKING",
            "action": "REVIEW_QRS_MORPHOLOGY",
        })

    qrs_ms = crosslead_conduction.get("qrs_ms")
    if qrs_ms is not None and float(qrs_ms) < 120.0:
        bundle_findings = [
            row for row in crosslead_conduction.get("findings") or []
            if str(row.get("code") or "").startswith(("RBBB", "LBBB"))
        ]
        if bundle_findings:
            conflicts.append({
                "code": "COMPLETE_BBB_WITH_QRS_LT_120_CONFLICT",
                "severity": "BLOCKING",
                "action": "SUPPRESS_COMPLETE_BBB",
            })

    if (
        str(fascicular.get("classification") or "") == "LAFB_COMPATIBLE"
        and fascicular.get("axis_deg") is not None
        and not (-90.0 <= float(fascicular["axis_deg"]) <= -45.0)
    ):
        conflicts.append({
            "code": "LAFB_WITHOUT_REQUIRED_AXIS_CONFLICT",
            "severity": "BLOCKING",
            "action": "SUPPRESS_LAFB",
        })

    if bool(wct.get("wide_complex_tachycardia")):
        hr = wct.get("heart_rate_bpm")
        qrs = wct.get("qrs_ms")
        if hr is None or qrs is None or float(hr) < 100.0 or float(qrs) < 120.0:
            conflicts.append({
                "code": "WCT_ACTIVATED_OUTSIDE_GATE",
                "severity": "BLOCKING",
                "action": "SUPPRESS_WCT_CLASSIFICATION",
            })

    if "qrs_ms" in remeasure and (crosslead_conduction.get("findings") or []):
        conflicts.append({
            "code": "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT",
            "severity": "BLOCKING",
            "action": "REMEASURE_QRS_BEFORE_CONDUCTION_LABEL",
        })

    rhythm_lead = str(rhythm.get("lead") or "")
    rhythm_qa = ((signal_integrity.get("per_lead") or {}).get(rhythm_lead) or {})
    if rhythm.get("evaluable") and rhythm_lead and rhythm_qa and not bool(
        rhythm_qa.get("rhythm_eligible")
    ):
        conflicts.append({
            "code": "RHYTHM_SOURCE_FAILS_SIGNAL_INTEGRITY_GATE",
            "severity": "BLOCKING",
            "action": "SUPPRESS_RHYTHM_MECHANISM_AND_REVIEW_SIGNAL",
        })

    if mechanism == "AF_COMPATIBLE" and rr_regular is True:
        conflicts.append({
            "code": "AF_WITH_REGULAR_RR_REQUIRES_REVIEW",
            "severity": "WARNING",
            "action": "CHECK_FLUTTER_AT_OR_PACING",
        })

    av_cls = str(av.get("classification") or "")
    if av_cls == "FIRST_DEGREE_AV_DELAY_COMPATIBLE":
        pr = av.get("pr_median_ms")
        if pr is None or float(pr) <= 200.0 or not bool(av.get("one_to_one")):
            conflicts.append({
                "code": "FIRST_DEGREE_AV_DELAY_WITHOUT_PR_GT_200_OR_1_TO_1",
                "severity": "BLOCKING",
                "action": "SUPPRESS_FIRST_DEGREE_AV_DELAY",
            })

    if av_cls in {
        "MOBITZ_I_WENCKEBACH_COMPATIBLE",
        "MOBITZ_II_COMPATIBLE",
        "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
        "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
        "COMPLETE_AV_BLOCK_COMPATIBLE",
    } and int(av.get("nonconducted_p_n") or 0) < 1:
        conflicts.append({
            "code": "AV_BLOCK_WITHOUT_NONCONDUCTED_P_CONFLICT",
            "severity": "BLOCKING",
            "action": "SUPPRESS_AV_BLOCK",
        })

    if av_cls == "COMPLETE_AV_BLOCK_COMPATIBLE" and (
        not bool(av.get("atrial_sequence_regular"))
        or not bool(av.get("ventricular_sequence_regular"))
        or not bool(av.get("av_dissociation_phase"))
    ):
        conflicts.append({
            "code": "COMPLETE_AV_BLOCK_WITHOUT_AV_DISSOCIATION_SUPPORT",
            "severity": "BLOCKING",
            "action": "SUPPRESS_COMPLETE_AV_BLOCK",
        })

    if (
        str(preexcitation.get("classification") or "")
        == "VENTRICULAR_PREEXCITATION_COMPATIBLE"
        and any(
            str(row.get("code") or "").startswith(("RBBB_", "LBBB_"))
            for row in (crosslead_conduction.get("findings") or [])
        )
    ):
        conflicts.append({
            "code": "PREEXCITATION_CONFOUNDS_BUNDLE_BRANCH_PATTERN",
            "severity": "WARNING",
            "action": "DOWNGRADE_BBB_AND_REVIEW_PREEXCITATION",
        })

    if (
        mechanism == "AF_COMPATIBLE"
        and bool(ectopy.get("irregularity_may_be_ectopy_driven"))
        and not bool((atrial_mech.get("aggregate_features") or {}).get("guideline_af_pattern"))
    ):
        conflicts.append({
            "code": "AF_IRREGULARITY_MAY_BE_ECTOPY_DRIVEN",
            "severity": "WARNING",
            "action": "REVIEW_ECTOPY_BEFORE_AF_PUBLICATION",
        })

    blocking = [c for c in conflicts if c.get("severity") == "BLOCKING"]
    warnings = [c for c in conflicts if c.get("severity") == "WARNING"]

    return {
        "version": CONSISTENCY_VERSION,
        "status": "BLOCKED" if blocking else "PASS_WITH_WARNINGS" if warnings else "PASS",
        "blocking_conflict": bool(blocking),
        "conflicts": conflicts,
        "blocking_n": len(blocking),
        "warning_n": len(warnings),
        "remeasure_required": bool(remeasure),
        "remeasure_targets": remeasure,
        "measurement_unmeasurable_targets": unmeasurable,
        "measurement_uncertain_targets": uncertain,
        "measurement_unusable_targets": unusable,
        "publication_allowed": not bool(blocking),
        "policy": "NO_BLOCKING_CONTRADICTION_MAY_BE_PUBLISHED_AS_ESTABLISHED",
    }
