from __future__ import annotations

import math
from typing import Any, Dict


R27_CONSENSUS_VERSION = "MEDCALC_R27_CONSENSUS_V1"

# R27 remains frozen and probability-only. These are comparison bands used only
# to determine whether an independent model signal agrees/disagrees with an
# already established structured MEDCALC finding. They are NOT diagnostic
# thresholds and cannot create, delete, or mutate a clinical finding.
R27_STRONG_SUPPORT_BAND = 0.80
R27_LOW_SUPPORT_BAND = 0.20

TEMPORAL_MODULES = {
    "AF", "FLUTTER", "SVT", "SINUS", "SINUS_BRADY", "SINUS_TACHY", "SINUS_ARRHYTHMIA",
    "PVC", "PAC", "BIGEMINY", "TRIGEMINY", "AVB1", "AVB2", "AVB3",
}

DOMAIN_MAP = {
    "SINUS_COMPATIBLE": "SINUS",
    "SINUS_BRADYCARDIA_COMPATIBLE": "SINUS_BRADY",
    "SINUS_TACHYCARDIA_COMPATIBLE": "SINUS_TACHY",
    "AF_COMPATIBLE": "AF",
    "FLUTTER_OR_AT_COMPATIBLE": "FLUTTER",
    "OTHER_SVT_COMPATIBLE": "SVT",
    "VT_COMPATIBLE": None,
    "RBBB_MORPHOLOGY_COMPATIBLE": "RBBB_COMPLETE",
    "LBBB_MORPHOLOGY_COMPATIBLE": "LBBB",
    "LAFB_COMPATIBLE": "LAFB",
    "LPFB_COMPATIBLE": "LPFB",
    "FIRST_DEGREE_AV_DELAY_COMPATIBLE": "AVB1",
    "MOBITZ_I_WENCKEBACH_COMPATIBLE": "AVB2",
    "MOBITZ_II_COMPATIBLE": "AVB2",
    "TWO_TO_ONE_AV_BLOCK_COMPATIBLE": "AVB2",
    "HIGH_GRADE_AV_BLOCK_COMPATIBLE": "AVB2",
    "COMPLETE_AV_BLOCK_COMPATIBLE": "AVB3",
    "VENTRICULAR_PREEXCITATION_COMPATIBLE": "WPW",
    "PVC_COMPATIBLE": "PVC",
    "PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE": "PAC",
}


def _probability(modules: Dict[str, Any], key: str | None) -> float | None:
    if not key:
        return None
    item = modules.get(key) or {}
    if str(item.get("interpretability") or "") == "NOT_INTERPRETABLE_R27_TILED":
        return None
    try:
        value = float(item.get("probability"))
    except Exception:
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _structured_findings(structured_report: Dict[str, Any]) -> list[Dict[str, Any]]:
    reasoning = structured_report.get("specialist_reasoning") or {}
    summary = reasoning.get("diagnostic_summary") or {}
    rows = []
    for item in summary.get("findings") or []:
        if not bool(item.get("publishable")):
            continue
        code = str(item.get("code") or "")
        module = DOMAIN_MAP.get(code)
        if module:
            rows.append({
                "domain": str(item.get("domain") or ""),
                "medcalc_code": code,
                "r27_module": module,
                "confidence": item.get("confidence"),
            })

    # Repolarization and QT are structured/numeric domains rather than reasoner
    # labels; include them only as independent comparison targets.
    repol = structured_report.get("repolarization") or {}
    if repol.get("st_elevation_leads"):
        rows.append({
            "domain": "REPOLARIZATION",
            "medcalc_code": "ST_ELEVATION_MEASURED",
            "r27_module": "ST_ELEVATION",
            "confidence": None,
        })
    if repol.get("st_depression_leads"):
        rows.append({
            "domain": "REPOLARIZATION",
            "medcalc_code": "ST_DEPRESSION_MEASURED",
            "r27_module": "ST_DEPRESSION",
            "confidence": None,
        })

    motor = structured_report.get("measurement_summary") or {}
    try:
        qtc = float(motor.get("qtc_fridericia_ms"))
    except Exception:
        qtc = None
    if qtc is not None and math.isfinite(qtc) and qtc >= 480.0:
        rows.append({
            "domain": "QT",
            "medcalc_code": "MARKED_QTC_PROLONGATION_MEASURED",
            "r27_module": "LONG_QT",
            "confidence": None,
        })
    return rows


def compare_r27_with_medcalc(
    structured_report: Dict[str, Any] | None,
    r27_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Cross-engine agreement audit.

    R27 is never allowed to originate a diagnosis or override a MEDCALC
    measurement/finding. Its only roles are SUPPORT, DISCORDANCE_REVIEW, or
    NOT_COMPARABLE.
    """
    structured_report = structured_report or {}
    r27_payload = r27_payload or {}
    modules = r27_payload.get("modules") or {}
    adapter = r27_payload.get("input_adapter") or {}
    tiled = bool(adapter.get("r27_tiled"))

    comparisons = []
    support_n = 0
    discordance_n = 0

    for row in _structured_findings(structured_report):
        module = str(row["r27_module"])
        p = _probability(modules, module)
        if p is None:
            status = "NOT_COMPARABLE"
            reason = (
                "TEMPORAL_MODULE_SUPPRESSED_FOR_R27_TILED"
                if tiled and module in TEMPORAL_MODULES
                else "R27_MODULE_NOT_INTERPRETABLE"
            )
        elif p >= R27_STRONG_SUPPORT_BAND:
            status = "CROSS_ENGINE_SUPPORT"
            reason = "R27_HIGH_PROBABILITY_SUPPORTS_EXISTING_MEDCALC_FINDING"
            support_n += 1
        elif p <= R27_LOW_SUPPORT_BAND:
            status = "CROSS_ENGINE_DISCORDANCE_REVIEW"
            reason = "R27_LOW_PROBABILITY_DESPITE_EXISTING_MEDCALC_FINDING"
            discordance_n += 1
        else:
            status = "NEUTRAL"
            reason = "R27_INTERMEDIATE_PROBABILITY"

        comparisons.append({
            **row,
            "r27_probability": round(p, 6) if p is not None else None,
            "status": status,
            "reason": reason,
        })

    # High R27 signals without a corresponding MEDCALC finding are review flags
    # only. They do not become diagnoses.
    represented = {str(row["r27_module"]) for row in comparisons}
    review_only = []
    for module, item in modules.items():
        if module in represented:
            continue
        if str(item.get("interpretability") or "") == "NOT_INTERPRETABLE_R27_TILED":
            continue
        try:
            p = float(item.get("probability"))
        except Exception:
            continue
        if not math.isfinite(p) or p < R27_STRONG_SUPPORT_BAND:
            continue
        review_only.append({
            "r27_module": str(module),
            "r27_probability": round(p, 6),
            "status": "R27_ONLY_REVIEW_SIGNAL",
            "diagnostic_claim_allowed": False,
        })

    return {
        "version": R27_CONSENSUS_VERSION,
        "evaluable": bool(modules),
        "r27_tiled": tiled,
        "policy": (
            "R27_IS_INDEPENDENT_PROBABILITY_ONLY_QA; "
            "NEVER_OVERRIDES_NUMERIC_MEASUREMENTS_OR_SPECIALIST_REASONER"
        ),
        "comparison_bands": {
            "strong_support": R27_STRONG_SUPPORT_BAND,
            "low_support": R27_LOW_SUPPORT_BAND,
            "diagnostic_thresholds": False,
        },
        "comparisons": comparisons,
        "cross_engine_support_n": support_n,
        "discordance_review_n": discordance_n,
        "r27_only_review_signals": sorted(
            review_only,
            key=lambda x: x["r27_probability"],
            reverse=True,
        ),
        "measurement_mutation_allowed": False,
        "diagnostic_mutation_allowed": False,
    }
