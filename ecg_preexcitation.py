from __future__ import annotations

from typing import Any, Dict


PREEXCITATION_VERSION = "MEDCALC_PREEXCITATION_V1"


def analyze_preexcitation(feature_graph: Dict[str, Any], qrs_morphology: Dict[str, Any]) -> Dict[str, Any]:
    g = feature_graph.get("global") or {}
    pr = ((g.get("pr_ms") or {}).get("value"))
    qrs = ((g.get("qrs_ms") or {}).get("value"))
    try:
        pr_ms = float(pr) if pr is not None else None
        qrs_ms = float(qrs) if qrs is not None else None
    except Exception:
        pr_ms, qrs_ms = None, None

    rows = qrs_morphology.get("per_lead") or {}
    delta_leads = [
        lead for lead,row in rows.items()
        if row.get("evaluable") and bool(row.get("delta_slur_compatible"))
    ]
    p_repro = bool((feature_graph.get("relations") or {}).get("p_reproducible"))

    compatible = bool(
        p_repro
        and pr_ms is not None and pr_ms < 120.0
        and qrs_ms is not None and qrs_ms >= 110.0
        and len(delta_leads) >= 2
    )
    return {
        "version": PREEXCITATION_VERSION,
        "evaluable": pr_ms is not None and qrs_ms is not None,
        "classification": (
            "VENTRICULAR_PREEXCITATION_COMPATIBLE"
            if compatible else "NO_PREEXCITATION_PATTERN_ESTABLISHED"
        ),
        "confidence": 0.88 if compatible else 0.35,
        "criteria": {
            "pr_lt_120ms": bool(pr_ms is not None and pr_ms < 120.0),
            "qrs_ge_110ms": bool(qrs_ms is not None and qrs_ms >= 110.0),
            "delta_slur_leads": delta_leads,
            "reproducible_p": p_repro,
        },
        "diagnostic_claim_allowed": False,
        "source": "PR_QRS_PLUS_MULTILEAD_INITIAL_QRS_SLOPE",
    }
