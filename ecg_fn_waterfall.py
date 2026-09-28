from __future__ import annotations

from typing import Any, Dict


FN_WATERFALL_VERSION = "MEDCALC_ECG_FN_WATERFALL_V2"

CODE_DOMAIN = {
    "AF_COMPATIBLE": "RHYTHM",
    "FLUTTER_OR_AT_COMPATIBLE": "RHYTHM",
    "SINUS_BRADYCARDIA_COMPATIBLE": "RATE",
    "SINUS_TACHYCARDIA_COMPATIBLE": "RATE",
    "RBBB_MORPHOLOGY_COMPATIBLE": "BUNDLE_BRANCH",
    "LBBB_MORPHOLOGY_COMPATIBLE": "BUNDLE_BRANCH",
    "LAFB_COMPATIBLE": "FASCICULAR",
    "FIRST_DEGREE_AV_DELAY_COMPATIBLE": "AV_CONDUCTION",
    "MOBITZ_I_WENCKEBACH_COMPATIBLE": "AV_CONDUCTION",
    "MOBITZ_II_COMPATIBLE": "AV_CONDUCTION",
    "TWO_TO_ONE_AV_BLOCK_COMPATIBLE": "AV_CONDUCTION",
    "HIGH_GRADE_AV_BLOCK_COMPATIBLE": "AV_CONDUCTION",
    "COMPLETE_AV_BLOCK_COMPATIBLE": "AV_CONDUCTION",
    "VENTRICULAR_PREEXCITATION_COMPATIBLE": "PREEXCITATION",
}


def classify_false_negative(
    expected_code: str,
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """Classify where a development/reference-positive ECG was lost.

    This function is for development/regression cohorts with labels already
    permitted for engineering. It must not be used to inspect frozen external
    cohorts record by record.
    """
    expected_code = str(expected_code)
    domain = CODE_DOMAIN.get(expected_code, "UNKNOWN")

    integrity = analysis.get("signal_integrity") or {}
    overall_quality = float(integrity.get("overall_quality") or 0.0)
    if overall_quality < 0.20:
        stage = "SIGNAL_FAILURE"
        detail = "OVERALL_SIGNAL_QUALITY_LT_0_20"
    else:
        candidate_layer = analysis.get("high_recall_candidates") or {}
        candidate = (candidate_layer.get("by_code") or {}).get(expected_code)
        if candidate is None:
            # Check whether required measurement-domain evidence was absent.
            consensus = analysis.get("measurement_consensus") or {}
            remeasure_targets = set(consensus.get("remeasure_targets") or [])
            unmeasurable_targets = set(consensus.get("unmeasurable_targets") or [])
            if remeasure_targets:
                stage = "MEASUREMENT_REMEASURE_REQUIRED"
                detail = "NO_CANDIDATE_WITH_REMEASURE_TARGETS:" + ",".join(sorted(remeasure_targets))
            elif unmeasurable_targets:
                stage = "MEASUREMENT_UNMEASURABLE"
                detail = "NO_CANDIDATE_WITH_UNMEASURABLE_TARGETS:" + ",".join(sorted(unmeasurable_targets))
            else:
                stage = "CANDIDATE_DETECTION_FAILURE"
                detail = "HIGH_RECALL_DETECTOR_DID_NOT_FIRE"
        else:
            gates = analysis.get("domain_gates") or {}
            gate = ((gates.get("domains") or {}).get(domain) or {})
            if not bool(gate.get("eligible")):
                stage = "DOMAIN_GATE_SUPPRESSION"
                detail = ";".join(
                    list(gate.get("blocked_by_conflicts") or [])
                    + list(gate.get("remeasure_targets") or [])
                    + list(gate.get("unusable_measurements") or [])
                ) or "DOMAIN_NOT_ELIGIBLE"
            else:
                fusion = analysis.get("evidence_fusion") or {}
                fused = (fusion.get("by_code") or {}).get(expected_code) or {}
                if not bool(fused.get("publishable")):
                    if str(fused.get("fusion_state") or "") == "MEASUREMENT_BOUNDARY_UNCERTAIN":
                        stage = "MEASUREMENT_BOUNDARY_UNCERTAIN"
                    elif str(fused.get("fusion_state") or "") == "MEASUREMENT_ABSTENTION":
                        stage = "MEASUREMENT_UNUSABLE_FOR_DIAGNOSIS"
                    else:
                        stage = "EVIDENCE_FUSION_FAILURE"
                    detail = str(fused.get("fusion_reason") or "FUSION_DID_NOT_PUBLISH")
                else:
                    reasoning = analysis.get("specialist_reasoning") or {}
                    findings = (
                        (reasoning.get("diagnostic_summary") or {}).get("findings") or []
                    )
                    published_codes = {
                        str(row.get("code") or "")
                        for row in findings
                        if bool(row.get("publishable"))
                    }
                    if expected_code not in published_codes:
                        stage = "REASONER_OR_REPORTING_FAILURE"
                        detail = "FUSED_FINDING_NOT_EXPOSED_BY_AUTHORITATIVE_REASONER"
                    else:
                        stage = "TRUE_POSITIVE"
                        detail = "EXPECTED_CODE_PUBLISHED"

    return {
        "version": FN_WATERFALL_VERSION,
        "expected_code": expected_code,
        "domain": domain,
        "stage": stage,
        "detail": detail,
        "external_frozen_record_debugging_allowed": False,
    }
