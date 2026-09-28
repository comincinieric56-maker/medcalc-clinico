from __future__ import annotations

from typing import Any, Dict


CONDUCTION_VERSION = "MEDCALC_CROSSLEAD_CONDUCTION_V2"


def _value(graph: Dict[str, Any], key: str) -> float | None:
    try:
        value = ((graph.get("global") or {}).get(key) or {}).get("value")
        return float(value) if value is not None else None
    except Exception:
        return None


def _morph(feature_graph: Dict[str, Any], lead: str) -> Dict[str, Any]:
    qrs = ((feature_graph.get("specialist_evidence") or {}).get("qrs_morphology") or {})
    return dict((qrs.get("per_lead") or {}).get(lead) or {})


def analyze_crosslead_conduction(feature_graph: Dict[str, Any]) -> Dict[str, Any]:
    """AHA/ACCF/HRS-shaped cross-lead bundle-branch morphology synthesis.

    Complete BBB requires measured QRS prolongation plus characteristic
    morphology across the expected right-precordial and lateral leads.
    """
    qrs_ms = _value(feature_graph, "qrs_ms")
    qrs_conf = float(
        ((feature_graph.get("global") or {}).get("qrs_ms") or {}).get("confidence")
        or 0.0
    )
    complete_wide = bool(qrs_ms is not None and qrs_ms >= 120.0 and qrs_conf >= 0.45)
    incomplete_range = bool(
        qrs_ms is not None and 110.0 <= qrs_ms < 120.0 and qrs_conf >= 0.45
    )

    v1 = _morph(feature_graph, "V1")
    v2 = _morph(feature_graph, "V2")
    i = _morph(feature_graph, "I")
    avl = _morph(feature_graph, "aVL")
    v5 = _morph(feature_graph, "V5")
    v6 = _morph(feature_graph, "V6")

    right_terminal_r = bool(
        any(
            row.get("evaluable")
            and (
                bool(row.get("r_prime_present"))
                or (
                    row.get("qrs_polarity") == "R_DOMINANT"
                    and float(row.get("terminal_positive_mv") or 0.0) >= 0.08
                )
            )
            for row in (v1, v2)
        )
    )

    lateral_terminal_s_leads = []
    for lead, row in (("I", i), ("V6", v6)):
        if not row.get("evaluable"):
            continue
        neg = float(row.get("terminal_negative_mv") or 0.0)
        dur = row.get("terminal_s_duration_ms")
        polarity = str(row.get("qrs_polarity") or "")
        broad_s = bool(
            neg <= -0.05
            and (
                (dur is not None and float(dur) >= 30.0)
                or (neg <= -0.10 and polarity in {"S_DOMINANT", "BIPHASIC"})
            )
        )
        if broad_s:
            lateral_terminal_s_leads.append(lead)

    rbbb_morphology = bool(right_terminal_r and lateral_terminal_s_leads)
    rbbb_complete = bool(complete_wide and rbbb_morphology)
    rbbb_incomplete = bool(incomplete_range and rbbb_morphology)

    v1_lbbb = bool(
        v1.get("evaluable")
        and v1.get("qrs_polarity") == "S_DOMINANT"
        and float(v1.get("terminal_positive_mv") or 0.0) < 0.10
        and (
            not v2.get("evaluable")
            or (
                v2.get("qrs_polarity") in {"S_DOMINANT", "BIPHASIC"}
                and float(v2.get("terminal_positive_mv") or 0.0) < 0.15
            )
        )
    )

    lateral_broad_r = []
    lateral_absent_q = []
    lateral_r_dominant = []
    for lead, row in (("I", i), ("aVL", avl), ("V5", v5), ("V6", v6)):
        if not row.get("evaluable"):
            continue
        if row.get("qrs_polarity") == "R_DOMINANT":
            lateral_r_dominant.append(lead)
            if (
                bool(row.get("notched_or_double_r"))
                or float(row.get("r_peak_time_ms") or 0.0) >= 60.0
            ):
                lateral_broad_r.append(lead)
        if not bool(row.get("initial_q_present")):
            lateral_absent_q.append(lead)

    key_lateral_r = bool(
        len(lateral_r_dominant) >= 2
        and ("I" in lateral_r_dominant or "V6" in lateral_r_dominant)
    )
    key_lateral_absent_q = bool(
        ("I" in lateral_absent_q and "V6" in lateral_absent_q)
        or len(lateral_absent_q) >= 3
    )
    delayed_or_notched_lateral = bool(
        len(lateral_broad_r) >= 2
        or (
            any(lead in lateral_broad_r for lead in ("V5", "V6"))
            and any(lead in lateral_broad_r for lead in ("I", "aVL"))
        )
    )

    lbbb_morphology = bool(
        v1_lbbb
        and key_lateral_r
        and key_lateral_absent_q
        and delayed_or_notched_lateral
    )
    lbbb_complete = bool(complete_wide and lbbb_morphology)
    lbbb_incomplete = bool(incomplete_range and lbbb_morphology)

    specialists = feature_graph.get("specialist_evidence") or {}
    fascicular = specialists.get("fascicular_conduction") or {}
    fascicular_classification = str(fascicular.get("classification") or "")
    lafb_support = fascicular_classification == "LAFB_COMPATIBLE"
    lpfb_support = fascicular_classification == "LPFB_COMPATIBLE"

    findings: list[Dict[str, Any]] = []
    if rbbb_complete:
        findings.append({
            "code": "RBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.96, 0.70 + 0.25*qrs_conf),6),
            "basis": [
                "QRS_GE_120MS",
                "RSR_PRIME_OR_TERMINAL_R_IN_V1_V2",
                "BROAD_TERMINAL_S_IN_I_OR_V6",
            ],
        })
    elif rbbb_incomplete:
        findings.append({
            "code": "INCOMPLETE_RBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.88,0.62+0.20*qrs_conf),6),
            "basis": [
                "QRS_110_TO_119MS",
                "RSR_PRIME_OR_TERMINAL_R_IN_V1_V2",
                "BROAD_TERMINAL_S_IN_I_OR_V6",
            ],
        })

    if lbbb_complete:
        findings.append({
            "code": "LBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.96, 0.70 + 0.25*qrs_conf),6),
            "basis": [
                "QRS_GE_120MS",
                "DOMINANT_NEGATIVE_QRS_V1_V2",
                "BROAD_OR_NOTCHED_LATERAL_R",
                "ABSENT_LATERAL_Q",
            ],
        })
    elif lbbb_incomplete:
        findings.append({
            "code": "INCOMPLETE_LBBB_MORPHOLOGY_COMPATIBLE",
            "confidence": round(min(0.86,0.60+0.20*qrs_conf),6),
            "basis": [
                "QRS_110_TO_119MS",
                "DOMINANT_NEGATIVE_QRS_V1_V2",
                "LATERAL_LBBB_MORPHOLOGY",
            ],
        })

    if lafb_support:
        findings.append({
            "code": "LAFB_COMPATIBLE",
            "confidence": float(fascicular.get("confidence") or 0.0),
            "basis": ["LEFT_AXIS","POSITIVE_QRS_I_AVL","INFERIOR_rS_PATTERN"],
        })

    if lpfb_support:
        findings.append({
            "code": "LPFB_COMPATIBLE",
            "confidence": float(fascicular.get("confidence") or 0.0),
            "basis": [
                "RIGHT_AXIS",
                "SUPERIOR_rS_PATTERN",
                "INFERIOR_qR_OR_R_DOMINANT_PATTERN",
                "QRS_LT_120MS",
            ],
        })

    mutually_exclusive = bool(rbbb_complete and lbbb_complete)
    classification = (
        "CONDUCTION_MORPHOLOGY_CONFLICT"
        if mutually_exclusive
        else "+".join(row["code"] for row in findings)
        if findings
        else "NO_SPECIFIC_CONDUCTION_PATTERN_ESTABLISHED"
    )

    return {
        "version": CONDUCTION_VERSION,
        "evaluable": qrs_ms is not None,
        "classification": classification,
        "findings": findings,
        "qrs_ms": qrs_ms,
        "qrs_confidence": qrs_conf,
        "criteria": {
            "qrs_ge_120ms": complete_wide,
            "qrs_110_119ms": incomplete_range,
            "rbbb_right_terminal_r": right_terminal_r,
            "rbbb_lateral_terminal_s_leads": lateral_terminal_s_leads,
            "rbbb_morphology": rbbb_morphology,
            "lbbb_v1_v2_negative": v1_lbbb,
            "lbbb_lateral_r_dominant_leads": lateral_r_dominant,
            "lbbb_lateral_broad_r_leads": lateral_broad_r,
            "lbbb_lateral_absent_q_leads": lateral_absent_q,
            "lbbb_key_lateral_r": key_lateral_r,
            "lbbb_key_lateral_absent_q": key_lateral_absent_q,
            "lbbb_delayed_or_notched_lateral": delayed_or_notched_lateral,
            "lbbb_morphology": lbbb_morphology,
            "lafb_support": lafb_support,
            "lpfb_support": lpfb_support,
        },
        "conflict": mutually_exclusive,
        "diagnostic_claim_allowed": False,
        "source": "MEDIAN_QRS_CROSS_LEAD_MORPHOLOGY",
        "rule": "COMPLETE_BBB_REQUIRES_QRS_GE_120MS_PLUS_CHARACTERISTIC_MULTILEAD_MORPHOLOGY",
    }
