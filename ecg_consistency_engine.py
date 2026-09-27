from __future__ import annotations

from typing import Any, Dict


CONSISTENCY_VERSION = "MEDCALC_ECG_CONSISTENCY_ENGINE_V1"


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
    rhythm = feature_graph.get("rhythm") or {}

    conflicts: list[Dict[str, Any]] = []
    remeasure = list(consensus.get("remeasure_targets") or [])

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
        "publication_allowed": not bool(blocking),
        "policy": "NO_BLOCKING_CONTRADICTION_MAY_BE_PUBLISHED_AS_ESTABLISHED",
    }
