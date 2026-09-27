from __future__ import annotations

from typing import Any, Dict


REASONER_VERSION = "MEDCALC_ECG_SPECIALIST_REASONER_V2"


def _candidate(code: str, confidence: float, basis: list[str], layer: str) -> Dict[str, Any]:
    return {
        "code": code,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
        "basis": basis,
        "layer": layer,
    }


def _global_value(feature_graph: Dict[str, Any], key: str) -> float | None:
    try:
        value = ((feature_graph.get("global") or {}).get(key) or {}).get("value")
        return float(value) if value is not None else None
    except Exception:
        return None


def reason_ecg(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Evidence-constrained specialist ECG synthesis.

    Numeric measurements remain immutable. Rate labels are only promoted to
    sinus bradycardia/tachycardia when sinus mechanism is independently
    established; a low or high ventricular rate alone is never called sinus.
    """
    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    wct = specialists.get("wide_complex_tachycardia") or {}
    ectopy = specialists.get("ectopy") or {}
    av = specialists.get("av_conduction") or {}
    preexcitation = specialists.get("preexcitation") or {}
    rhythm = feature_graph.get("rhythm") or {}

    candidates: list[Dict[str, Any]] = []

    if bool(wct.get("wide_complex_tachycardia")):
        cls = str(wct.get("classification") or "")
        conf = float(wct.get("confidence") or 0.0)
        if cls == "VT_COMPATIBLE":
            candidates.append(_candidate(
                "VT_COMPATIBLE", conf,
                ["WCT_GATE_MET", "DIRECT_WCT_MORPHOLOGY"], "VENTRICULAR_ORIGIN"
            ))

    mechanism = str(atrial_mech.get("mechanism") or "")
    atrial_conf = float(atrial_mech.get("confidence") or 0.0)
    if mechanism in {
        "AF_COMPATIBLE",
        "FLUTTER_OR_AT_COMPATIBLE",
        "OTHER_SVT_COMPATIBLE",
    }:
        candidates.append(_candidate(
            mechanism,
            atrial_conf,
            ["NATIVE_ATRIAL_ANALYZER", "RR_P_QRS_AND_ECTOPY_CONTEXT"],
            "ATRIAL_MECHANISM",
        ))
    elif bool(atrial.get("sinus_compatible")):
        hr = _global_value(feature_graph, "heart_rate_bpm")
        base_conf = min(float(rhythm.get("confidence") or 0.0), 0.95)
        if hr is not None and hr < 60.0:
            code = "SINUS_BRADYCARDIA_COMPATIBLE"
            basis = ["SINUS_MECHANISM_ESTABLISHED", "VENTRICULAR_RATE_LT_60"]
        elif hr is not None and hr > 100.0:
            code = "SINUS_TACHYCARDIA_COMPATIBLE"
            basis = ["SINUS_MECHANISM_ESTABLISHED", "VENTRICULAR_RATE_GT_100"]
        else:
            code = "SINUS_COMPATIBLE"
            basis = ["REPRODUCIBLE_POSITIVE_P_IN_II", "P_QRS_COUPLING"]
        candidates.append(_candidate(code, base_conf, basis, "ATRIAL_MECHANISM"))

    if candidates:
        vt = [c for c in candidates if c["code"] == "VT_COMPATIBLE"]
        primary = vt[0] if vt else max(candidates, key=lambda c: c["confidence"])
    else:
        primary = _candidate(
            "RHYTHM_MECHANISM_UNDETERMINED",
            0.0,
            ["INSUFFICIENT_SPECIALIST_SEPARATION"],
            "RHYTHM",
        )

    blocking_codes = {
        str(row.get("code") or "")
        for row in (consistency.get("conflicts") or [])
        if str(row.get("severity") or "") == "BLOCKING"
    }

    conduction_findings = list(crosslead_conduction.get("findings") or [])
    conduction_blocking_codes = {
        "RBBB_LBBB_MUTUAL_CONFLICT",
        "COMPLETE_BBB_WITH_QRS_LT_120_CONFLICT",
        "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT",
    }
    if (
        "qrs_ms" in (consistency.get("remeasure_targets") or [])
        or bool(blocking_codes & conduction_blocking_codes)
    ):
        conduction_findings = []

    av_finding = None
    av_cls = str(av.get("classification") or "")
    av_blocking_codes = {
        "FIRST_DEGREE_AV_DELAY_WITHOUT_PR_GT_200_OR_1_TO_1",
        "AV_BLOCK_WITHOUT_NONCONDUCTED_P_CONFLICT",
        "COMPLETE_AV_BLOCK_WITHOUT_AV_DISSOCIATION_SUPPORT",
    }
    if (
        av.get("evaluable")
        and av_cls not in {
            "",
            "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED",
            "AV_CONDUCTION_NOT_EVALUABLE",
        }
        and not bool(blocking_codes & av_blocking_codes)
    ):
        av_finding = {
            "code": av_cls,
            "confidence": float(av.get("confidence") or 0.0),
            "basis": list(av.get("basis") or []),
        }

    preexcitation_finding = None
    if str(preexcitation.get("classification") or "") == "VENTRICULAR_PREEXCITATION_COMPATIBLE":
        preexcitation_finding = {
            "code": "VENTRICULAR_PREEXCITATION_COMPATIBLE",
            "confidence": float(preexcitation.get("confidence") or 0.0),
            "basis": ["SHORT_PR", "QRS_PROLONGATION", "MULTILEAD_DELTA_SLUR"],
        }
        conduction_findings = [
            row for row in conduction_findings
            if not str(row.get("code") or "").startswith(("RBBB_", "LBBB_"))
        ]

    ectopy_findings = []
    if int(ectopy.get("pvc_compatible_n") or 0) > 0:
        ectopy_findings.append({
            "code": "PVC_COMPATIBLE",
            "count": int(ectopy.get("pvc_compatible_n") or 0),
        })
    if int(ectopy.get("pac_or_narrow_premature_n") or 0) > 0:
        ectopy_findings.append({
            "code": "PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE",
            "count": int(ectopy.get("pac_or_narrow_premature_n") or 0),
        })

    publication_allowed = bool(consistency.get("publication_allowed"))
    if not publication_allowed:
        primary = {
            **primary,
            "publish_as_established": False,
            "suppressed_by_consistency_engine": True,
        }
    else:
        primary = {
            **primary,
            "publish_as_established": bool(primary["confidence"] >= 0.70),
            "suppressed_by_consistency_engine": False,
        }

    final_findings = []
    if primary.get("code") != "RHYTHM_MECHANISM_UNDETERMINED":
        final_findings.append({
            "domain": "RHYTHM",
            "code": primary.get("code"),
            "confidence": primary.get("confidence"),
            "publishable": bool(primary.get("publish_as_established")),
            "basis": list(primary.get("basis") or []),
        })
    for row in conduction_findings:
        final_findings.append({
            "domain": "CONDUCTION",
            "code": row.get("code"),
            "confidence": row.get("confidence"),
            "publishable": publication_allowed,
            "basis": list(row.get("basis") or []),
        })
    if av_finding:
        final_findings.append({
            "domain": "AV_CONDUCTION",
            **av_finding,
            "publishable": publication_allowed,
        })
    if preexcitation_finding:
        final_findings.append({
            "domain": "PREEXCITATION",
            **preexcitation_finding,
            "publishable": publication_allowed,
        })
    for row in ectopy_findings:
        final_findings.append({
            "domain": "ECTOPY",
            **row,
            "publishable": publication_allowed,
        })

    abstentions = []
    if consistency.get("blocking_conflict"):
        abstentions.append({
            "domain": "GLOBAL",
            "reason": "BLOCKING_CONSISTENCY_CONFLICT",
            "conflicts": sorted(blocking_codes),
        })
    if (feature_graph.get("specialist_evidence") or {}).get("measurement_consensus", {}).get("remeasure_required"):
        abstentions.append({
            "domain": "MEASUREMENT",
            "reason": "REMEASUREMENT_REQUIRED",
            "targets": list(consistency.get("remeasure_targets") or []),
        })

    return {
        "version": REASONER_VERSION,
        "primary_rhythm": primary,
        "rhythm_candidates": candidates,
        "conduction_findings": conduction_findings,
        "av_conduction_finding": av_finding,
        "preexcitation_finding": preexcitation_finding,
        "ectopy_findings": ectopy_findings,
        "consistency_status": consistency.get("status"),
        "publication_allowed": publication_allowed,
        "measurement_mutation_allowed": False,
        "diagnostic_summary": {
            "authoritative": True,
            "findings": final_findings,
            "abstentions": abstentions,
            "publication_allowed": publication_allowed,
        },
        "report_authority": "SPECIALIST_REASONER_STRUCTURED_OUTPUT",
        "llm_role": "REPORT_WORDING_ONLY_NOT_CLINICAL_ARBITRATION",
        "source": "SPECIALIST_EVIDENCE_GRAPH",
    }
