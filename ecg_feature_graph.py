from __future__ import annotations

import math
from typing import Any, Dict


FEATURE_GRAPH_VERSION = "MEDCALC_ECG_FEATURE_GRAPH_V1"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _metric(per_lead: Dict[str, Dict[str, Any]], lead: str, name: str) -> Dict[str, Any]:
    return dict(((per_lead.get(lead) or {}).get("metrics") or {}).get(name) or {})


def _value(per_lead: Dict[str, Dict[str, Any]], lead: str, name: str) -> float | None:
    return _finite(_metric(per_lead, lead, name).get("value"))


def _lead_node(per_lead: Dict[str, Dict[str, Any]], lead: str) -> Dict[str, Any]:
    item = dict(per_lead.get(lead) or {})
    r = _value(per_lead, lead, "r_amp_mv")
    s = _value(per_lead, lead, "s_amp_mv")
    q = _value(per_lead, lead, "q_amp_mv")
    area = _value(per_lead, lead, "qrs_net_area_mv_ms")
    rs_ratio = _value(per_lead, lead, "rs_ratio")

    dominant = None
    if r is not None and s is not None:
        if abs(r) >= 1.15 * abs(s):
            dominant = "R_DOMINANT"
        elif abs(s) >= 1.15 * abs(r):
            dominant = "S_DOMINANT"
        else:
            dominant = "BIPHASIC"

    return {
        "evaluable": bool(item.get("evaluable")),
        "confidence": _finite(item.get("confidence")) or 0.0,
        "duration_s": _finite(item.get("duration_s")),
        "r_count": int(item.get("r_count") or 0),
        "qrs_polarity": dominant,
        "qrs_net_area_mv_ms": area,
        "r_mv": r,
        "s_mv": s,
        "q_mv": q,
        "rs_ratio": rs_ratio,
        "st_j60_mv": _value(per_lead, lead, "st_j60_mv"),
        "t_mv": _value(per_lead, lead, "t_amp_mv"),
        "p_mv": _value(per_lead, lead, "p_amp_mv"),
        "qrs_ms": _value(per_lead, lead, "qrs_ms"),
        "pr_ms": _value(per_lead, lead, "pr_ms"),
        "qt_ms": _value(per_lead, lead, "qt_ms"),
        "p_wave_reproducible": bool(
            (item.get("atrial_activity") or {}).get("p_wave_reproducible")
        ),
        "p_qrs_coupling_fraction": _finite(
            (item.get("atrial_activity") or {}).get("p_qrs_coupling_fraction")
        ),
    }


def build_ecg_feature_graph(
    *,
    per_lead: Dict[str, Dict[str, Any]],
    global_metrics: Dict[str, Dict[str, Any]],
    rhythm: Dict[str, Any],
    axis: Dict[str, Any],
    atrial_activity: Dict[str, Any],
    atrial_mechanism: Dict[str, Any],
    wide_complex_tachycardia: Dict[str, Any],
    fascicular_conduction: Dict[str, Any],
    measurement_consensus: Dict[str, Any],
    signal_integrity: Dict[str, Any],
) -> Dict[str, Any]:
    """Create one auditable representation shared by all interpretation layers."""
    global_values = {
        key: {
            "value": _finite((global_metrics.get(key) or {}).get("value")),
            "confidence": _finite((global_metrics.get(key) or {}).get("confidence")) or 0.0,
            "status": (global_metrics.get(key) or {}).get("status"),
        }
        for key in (
            "heart_rate_bpm",
            "p_duration_ms",
            "pr_ms",
            "qrs_ms",
            "qt_ms",
            "qtc_bazett_ms",
            "qtc_fridericia_ms",
        )
    }

    lead_nodes = {lead: _lead_node(per_lead, lead) for lead in LEADS}
    inferior = [lead_nodes[x]["qrs_polarity"] for x in ("II", "III", "aVF")]
    lateral = [lead_nodes[x]["qrs_polarity"] for x in ("I", "aVL")]

    relations = {
        "inferior_s_dominant_n": sum(v == "S_DOMINANT" for v in inferior),
        "lateral_r_dominant_n": sum(v == "R_DOMINANT" for v in lateral),
        "precordial_transition": [
            {"lead": lead, "rs_ratio": lead_nodes[lead]["rs_ratio"]}
            for lead in ("V1", "V2", "V3", "V4", "V5", "V6")
        ],
        "rr_regular": rhythm.get("regular"),
        "p_reproducible": bool(atrial_activity.get("p_wave_reproducible")),
        "p_qrs_coupling_fraction": _finite(
            atrial_activity.get("rhythm_p_qrs_coupling_fraction")
        ),
        "axis_deg": _finite(axis.get("degrees")),
        "wide_complex_tachycardia_active": bool(
            wide_complex_tachycardia.get("wide_complex_tachycardia")
        ),
        "measurement_recheck_required": bool(
            measurement_consensus.get("remeasure_required")
        ),
    }

    return {
        "version": FEATURE_GRAPH_VERSION,
        "source": "CALIBRATED_DIGITAL_SIGNAL_PLUS_SPECIALIST_AUDIT",
        "global": global_values,
        "rhythm": dict(rhythm),
        "axis": dict(axis),
        "leads": lead_nodes,
        "relations": relations,
        "specialist_evidence": {
            "atrial_activity": dict(atrial_activity),
            "atrial_mechanism": dict(atrial_mechanism),
            "wide_complex_tachycardia": dict(wide_complex_tachycardia),
            "fascicular_conduction": dict(fascicular_conduction),
            "measurement_consensus": dict(measurement_consensus),
            "signal_integrity": dict(signal_integrity),
        },
        "invariant": (
            "FEATURES_DESCRIBE_EVIDENCE; THEY_DO_NOT_MUTATE_CANONICAL_MEASUREMENTS"
        ),
    }
