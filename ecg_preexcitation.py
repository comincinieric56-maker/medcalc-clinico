from __future__ import annotations

from typing import Any, Dict


PREEXCITATION_VERSION = "MEDCALC_PREEXCITATION_V2"


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if x == x and abs(x) != float("inf") else None


def analyze_preexcitation(
    feature_graph: Dict[str, Any],
    qrs_morphology: Dict[str, Any],
) -> Dict[str, Any]:
    """Adult ventricular-preexcitation compatibility from independent evidence.

    Primary path uses global PR/QRS plus multilead delta morphology. A rescue
    path is allowed only when at least two *same leads* independently show a
    short measured PR and delta-slurred widened QRS. This does not lower a
    threshold; it replaces an unavailable global consensus with concordant
    multilead measurements.
    """
    g = feature_graph.get("global") or {}
    pr_ms = _finite((g.get("pr_ms") or {}).get("value"))
    qrs_ms = _finite((g.get("qrs_ms") or {}).get("value"))

    morphology_rows = qrs_morphology.get("per_lead") or {}
    # delta_slur_compatible is produced by the morphology engine only after
    # its own widened-QRS/slur criterion. Keep that existing contract for the
    # global path so legacy callers/tests do not need to duplicate duration.
    delta_leads = [
        str(lead) for lead, row in morphology_rows.items()
        if row.get("evaluable") and bool(row.get("delta_slur_compatible"))
    ]
    # The new rescue path is intentionally stricter: it requires an explicit
    # widened QRS duration in each rescued lead.
    rescue_delta_leads = [
        str(lead) for lead, row in morphology_rows.items()
        if row.get("evaluable")
        and bool(row.get("delta_slur_compatible"))
        and _finite(row.get("duration_ms")) is not None
        and float(row.get("duration_ms")) >= 110.0
    ]

    lead_nodes = feature_graph.get("leads") or {}
    short_pr_leads = []
    short_pr_audit: Dict[str, float] = {}
    for lead, row in lead_nodes.items():
        if not bool(row.get("evaluable")):
            continue
        pr = _finite(row.get("pr_ms"))
        confidence = _finite(row.get("confidence")) or 0.0
        # Adult PR physiology: a true short-PR rescue must remain above the
        # implausibly short/delineation-error range and have usable lead quality.
        if pr is not None and 70.0 <= pr < 120.0 and confidence >= 0.50:
            short_pr_leads.append(str(lead))
            short_pr_audit[str(lead)] = round(float(pr), 3)

    concordant_leads = sorted(set(short_pr_leads) & set(rescue_delta_leads))
    same_lead_multilead_rescue = bool(len(concordant_leads) >= 2)

    p_repro = bool((feature_graph.get("relations") or {}).get("p_reproducible"))
    distributed_support_leads = sorted(set(short_pr_leads) | set(rescue_delta_leads))
    distributed_multilead_rescue = bool(
        p_repro
        and len(short_pr_leads) >= 2
        and len(rescue_delta_leads) >= 2
        and len(concordant_leads) >= 1
        and len(distributed_support_leads) >= 3
    )
    multilead_rescue = bool(
        same_lead_multilead_rescue or distributed_multilead_rescue
    )
    global_path = bool(
        p_repro
        and pr_ms is not None and 70.0 <= pr_ms < 120.0
        and qrs_ms is not None and qrs_ms >= 110.0
        and len(delta_leads) >= 2
    )

    compatible = bool(global_path or multilead_rescue)
    evaluable = bool(
        (pr_ms is not None and qrs_ms is not None)
        or multilead_rescue
    )

    confidence = 0.35
    if global_path and multilead_rescue:
        confidence = 0.94
    elif global_path:
        confidence = 0.88
    elif multilead_rescue:
        confidence = 0.90

    return {
        "version": PREEXCITATION_VERSION,
        "evaluable": evaluable,
        "classification": (
            "VENTRICULAR_PREEXCITATION_COMPATIBLE"
            if compatible else "NO_PREEXCITATION_PATTERN_ESTABLISHED"
        ),
        "confidence": confidence,
        "criteria": {
            "pr_lt_120ms": bool(pr_ms is not None and 70.0 <= pr_ms < 120.0),
            "qrs_ge_110ms": bool(qrs_ms is not None and qrs_ms >= 110.0),
            "delta_slur_leads": sorted(delta_leads),
            "rescue_delta_wide_leads": sorted(rescue_delta_leads),
            "reproducible_p": p_repro,
            "short_pr_leads": sorted(short_pr_leads),
            "short_pr_ms_by_lead": short_pr_audit,
            "concordant_short_pr_delta_leads": concordant_leads,
            "distributed_support_leads": distributed_support_leads,
            "same_lead_multilead_short_pr_delta_rescue": same_lead_multilead_rescue,
            "distributed_multilead_short_pr_delta_rescue": distributed_multilead_rescue,
            "multilead_short_pr_delta_rescue": multilead_rescue,
            "global_pr_qrs_path": global_path,
        },
        "diagnostic_claim_allowed": False,
        "source": (
            "GLOBAL_PR_QRS_PLUS_MULTILEAD_DELTA_OR_"
            "STRICT_MULTILEAD_SHORT_PR_DELTA_SUPPORT"
        ),
    }
