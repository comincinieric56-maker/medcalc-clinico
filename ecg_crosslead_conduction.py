from __future__ import annotations

from typing import Any, Dict


CONDUCTION_VERSION = "MEDCALC_CROSSLEAD_CONDUCTION_V1"


def _value(graph: Dict[str, Any], key: str) -> float | None:
    try:
        value = ((graph.get("global") or {}).get(key) or {}).get("value")
        return float(value) if value is not None else None
    except Exception:
        return None


def _lead(graph: Dict[str, Any], lead: str) -> Dict[str, Any]:
    return dict((graph.get("leads") or {}).get(lead) or {})


def analyze_crosslead_conduction(feature_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative 12-lead conduction synthesis.

    This module deliberately requires duration plus cross-lead morphology.
    QRS width alone never establishes a bundle-branch block.
    """
    qrs_ms = _value(feature_graph, "qrs_ms")
    qrs_conf = float(
        ((feature_graph.get("global") or {}).get("qrs_ms") or {}).get("confidence")
        or 0.0
    )
    v1 = _lead(feature_graph, "V1")
    v6 = _lead(feature_graph, "V6")
    i = _lead(feature_graph, "I")

    wide = bool(qrs_ms is not None and qrs_ms >= 120.0 and qrs_conf >= 0.45)
    v1_pol = v1.get("qrs_polarity")
    v6_pol = v6.get("qrs_polarity")
    i_pol = i.get("qrs_polarity")

    rbbb_support = bool(
        wide
        and v1_pol == "R_DOMINANT"
        and (v6_pol == "S_DOMINANT" or i_pol == "S_DOMINANT")
    )
    lbbb_support = bool(
        wide
        and v1_pol == "S_DOMINANT"
        and v6_pol == "R_DOMINANT"
        and i_pol == "R_DOMINANT"
    )

    specialists = feature_graph.get("specialist_evidence") or {}
    fascicular = specialists.get("fascicular_conduction") or {}
    lafb_support = str(fascicular.get("classification") or "") == "LAFB_COMPATIBLE"

    findings: list[Dict[str, Any]] = []
    if rbbb_support:
        findings.append({
            "code": "RBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.90, 0.55 + 0.35 * qrs_conf), 6),
            "basis": ["QRS>=120ms", "V1_R_DOMINANT", "TERMINAL_S_SUPPORT_I_OR_V6"],
        })
    if lbbb_support:
        findings.append({
            "code": "LBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.90, 0.55 + 0.35 * qrs_conf), 6),
            "basis": ["QRS>=120ms", "V1_S_DOMINANT", "I_V6_R_DOMINANT"],
        })
    if lafb_support:
        findings.append({
            "code": "LAFB_COMPATIBLE",
            "confidence": float(fascicular.get("confidence") or 0.0),
            "basis": [
                "LEFT_AXIS",
                "POSITIVE_QRS_I_AVL",
                "INFERIOR_rS_PATTERN",
            ],
        })

    mutually_exclusive = rbbb_support and lbbb_support
    if mutually_exclusive:
        classification = "CONDUCTION_MORPHOLOGY_CONFLICT"
    elif findings:
        classification = "+".join(row["code"] for row in findings)
    else:
        classification = "NO_SPECIFIC_CONDUCTION_PATTERN_ESTABLISHED"

    return {
        "version": CONDUCTION_VERSION,
        "evaluable": qrs_ms is not None,
        "classification": classification,
        "findings": findings,
        "qrs_ms": qrs_ms,
        "qrs_confidence": qrs_conf,
        "criteria": {
            "qrs_ge_120ms": wide,
            "V1_polarity": v1_pol,
            "V6_polarity": v6_pol,
            "I_polarity": i_pol,
            "rbbb_support": rbbb_support,
            "lbbb_support": lbbb_support,
            "lafb_support": lafb_support,
        },
        "conflict": mutually_exclusive,
        "diagnostic_claim_allowed": False,
        "source": "CROSS_LEAD_DIGITAL_MORPHOLOGY",
        "rule": "BUNDLE_BRANCH_PATTERN_REQUIRES_QRS_DURATION_PLUS_CROSS_LEAD_MORPHOLOGY",
    }
