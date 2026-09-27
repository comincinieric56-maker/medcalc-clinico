from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ecg_structured_report import build_structured_ecg_report


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _value(metric: Dict[str, Any] | None) -> float | None:
    metric = metric or {}
    value = metric.get("value")
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _confidence(metric: Dict[str, Any] | None) -> float:
    metric = metric or {}
    try:
        return float(metric.get("confidence") or 0.0)
    except Exception:
        return 0.0


def _format_metric(metric: Dict[str, Any] | None, unit: str) -> str:
    value = _value(metric)
    if value is None:
        return "NO EVALUABLE"
    conf = _confidence(metric)
    decimals = 0 if unit in {"MS", "LPM"} else 2
    return f"{value:.{decimals}f} {unit} (conf {conf:.2f})"


def build_signal_primary_structured_report(
    canonical_ecg: Dict[str, Any],
    digital_measurements: Dict[str, Any],
) -> Dict[str, Any]:
    """Adapt V2 numeric measurements to the existing MEDCALC report contract.

    The legacy structured-report builder is used only for waveform evidence
    rendering. All clinical numeric fields, rhythm regularity, axis and ST/T
    statements are then replaced by the calibrated digital-signal engine.
    """
    matrix_mv = np.asarray(canonical_ecg.get("legacy_matrix_mv"), dtype=float)
    if matrix_mv.shape != (5000, 12):
        raise ValueError(
            f"Canonical legacy adapter expected (5000,12), got {matrix_mv.shape}."
        )

    # Evidence/plotting only. The numeric interpretation below comes from V2.
    report = build_structured_ecg_report(
        matrix_mv * 1000.0,
        fs=int(canonical_ecg.get("fs") or 500),
        lead_names=LEADS,
    )

    global_m = digital_measurements.get("global") or {}
    rhythm_v2 = digital_measurements.get("rhythm") or {}
    axis_v2 = digital_measurements.get("axis") or {}
    st_by_lead = digital_measurements.get("st_by_lead") or {}
    t_by_lead = digital_measurements.get("t_by_lead") or {}

    hr = global_m.get("heart_rate_bpm") or {}
    pr = global_m.get("pr_ms") or {}
    qrs = global_m.get("qrs_ms") or {}
    qt = global_m.get("qt_ms") or {}
    qtc = global_m.get("qtc_bazett_ms") or {}
    qtc_fridericia = global_m.get("qtc_fridericia_ms") or {}

    rhythm_evaluable = bool(rhythm_v2.get("evaluable"))
    regular = bool(rhythm_v2.get("regular")) if rhythm_evaluable else False
    rhythm_label = (
        "RITMO REGULAR SEGÚN INTERVALOS RR"
        if rhythm_evaluable and regular
        else "RITMO IRREGULAR SEGÚN INTERVALOS RR"
        if rhythm_evaluable
        else "RITMO NO EVALUABLE"
    )
    rhythm_code = (
        "RR_REGULAR"
        if rhythm_evaluable and regular
        else "RR_IRREGULAR"
        if rhythm_evaluable
        else "NOT_EVALUABLE"
    )
    rhythm = {
        "evaluable": rhythm_evaluable,
        "lead": rhythm_v2.get("lead"),
        "duration_s": (
            (digital_measurements.get("leads") or {})
            .get(str(rhythm_v2.get("lead") or ""), {})
            .get("duration_s")
        ),
        "r_count": rhythm_v2.get("r_count"),
        "heart_rate_bpm": rhythm_v2.get("heart_rate_bpm"),
        "rr_ms": rhythm_v2.get("rr_ms"),
        "rr_mean_ms": rhythm_v2.get("rr_mean_ms"),
        "rr_median_ms": rhythm_v2.get("rr_median_ms"),
        "rr_sd_ms": rhythm_v2.get("rr_sd_ms"),
        "rr_cv": rhythm_v2.get("rr_cv"),
        "rr_cv_robust": rhythm_v2.get("rr_cv"),
        "rr_mad_ms": rhythm_v2.get("rr_mad_ms"),
        "rr_mad_ratio": rhythm_v2.get("rr_mad_ratio"),
        "regularity_cv_used": rhythm_v2.get("rr_cv"),
        "regular": regular,
        "confidence": rhythm_v2.get("confidence"),
        "signal_source": "CALIBRATED_DIGITAL_SIGNAL",
        "source": "CALIBRATED_DIGITAL_SIGNAL",
        # Do not infer sinus origin unless a dedicated P-wave rule supports it.
        "sinus_compatible": False,
        "reason": rhythm_v2.get("reason"),
    }

    rhythm_screen = {
        "evaluable": rhythm_evaluable,
        "code": rhythm_code,
        "label": rhythm_label,
        "source": "CALIBRATED_DIGITAL_SIGNAL_RR",
        "basis": (
            [
                f"RR CV {float(rhythm_v2.get('rr_cv')):.3f}"
                if rhythm_v2.get("rr_cv") is not None else "",
                f"RR MAD {float(rhythm_v2.get('rr_mad_ms')):.1f} ms"
                if rhythm_v2.get("rr_mad_ms") is not None else "",
                f"{int(rhythm_v2.get('r_count') or 0)} QRS detectados",
            ]
            if rhythm_evaluable else []
        ),
    }
    rhythm_screen["basis"] = [v for v in rhythm_screen["basis"] if v]

    per_lead_repol: Dict[str, Any] = {}
    st_elevation: list[str] = []
    st_depression: list[str] = []
    st_evaluable: list[str] = []
    t_evaluable: list[str] = []
    t_unexpected: list[str] = []

    for lead in LEADS:
        st = dict(st_by_lead.get(lead) or {})
        tv = dict(t_by_lead.get(lead) or {})
        st_value = _value(st)
        t_value = _value(tv)
        st_conf = _confidence(st)
        t_conf = _confidence(tv)
        st_ok = st_value is not None and st_conf >= 0.45
        t_ok = t_value is not None and t_conf >= 0.45
        if st_ok:
            st_evaluable.append(lead)
            if st_value > 0.10:
                st_elevation.append(lead)
            elif st_value < -0.10:
                st_depression.append(lead)
        if t_ok:
            t_evaluable.append(lead)
            expected_positive = lead in {"I", "II", "V3", "V4", "V5", "V6"}
            expected_negative = lead == "aVR"
            if expected_positive and t_value < -0.05:
                t_unexpected.append(lead)
            elif expected_negative and t_value > 0.05:
                t_unexpected.append(lead)
        per_lead_repol[lead] = {
            "evaluable": bool(st_ok or t_ok),
            "st_mv": st_value if st_ok else None,
            "st_confidence": st_conf,
            "st_direction": st.get("direction"),
            "st_mm": st.get("mm_at_paper_gain"),
            "t_mv": t_value if t_ok else None,
            "t_confidence": t_conf,
            "t_polarity": tv.get("polarity"),
            "source": "CALIBRATED_DIGITAL_SIGNAL",
        }

    if len(st_depression) > len(st_elevation):
        st_direction = "DEPRESSION_PREDOMINANT"
    elif len(st_elevation) > len(st_depression):
        st_direction = "ELEVATION_PREDOMINANT"
    elif st_depression or st_elevation:
        st_direction = "MIXED"
    else:
        st_direction = "ISOELECTRIC_COMPATIBLE"

    repol = {
        "per_lead": per_lead_repol,
        "st_evaluable_leads": st_evaluable,
        "st_abnormal_leads": st_elevation + st_depression,
        "st_elevation_leads": st_elevation,
        "st_depression_leads": st_depression,
        "st_direction": st_direction,
        "st_isoelectric_compatible": bool(st_evaluable and not st_elevation and not st_depression),
        "t_evaluable_leads": t_evaluable,
        "t_unexpected_polarity_leads": t_unexpected,
        "t_normal_polarity_compatible": bool(t_evaluable and not t_unexpected),
        "source": "CALIBRATED_DIGITAL_SIGNAL",
    }

    axis = {
        "evaluable": bool(axis_v2.get("evaluable")),
        "degrees": axis_v2.get("degrees"),
        "confidence": axis_v2.get("confidence"),
        "source": axis_v2.get("source"),
        "reason": axis_v2.get("reason"),
    }
    if axis["evaluable"] and axis["degrees"] is not None:
        deg = float(axis["degrees"])
        if -30 <= deg <= 90:
            axis["category"] = "EJE NORMAL"
        elif -90 <= deg < -30:
            axis["category"] = "DESVIACIÓN IZQUIERDA"
        elif 90 < deg <= 180:
            axis["category"] = "DESVIACIÓN DERECHA"
        else:
            axis["category"] = "EJE EXTREMO"
    else:
        axis["category"] = "NO EVALUABLE"

    measurement_summary = {
        "heart_rate_bpm": _value(hr),
        "heart_rate_confidence": _confidence(hr),
        "pr_ms": _value(pr),
        "pr_confidence": _confidence(pr),
        "qrs_ms": _value(qrs),
        "qrs_confidence": _confidence(qrs),
        "qt_ms": _value(qt),
        "qt_confidence": _confidence(qt),
        "qtc_bazett_ms": _value(qtc),
        "qtc_fridericia_ms": _value(qtc_fridericia),
        "qtc_confidence": _confidence(qtc),
        "qtc_fridericia_confidence": _confidence(qtc_fridericia),
        "axis_deg": axis.get("degrees"),
        "axis_confidence": axis.get("confidence"),
        "rr_cv": rhythm_v2.get("rr_cv"),
        "rr_sd_ms": rhythm_v2.get("rr_sd_ms"),
        "rr_mad_ms": rhythm_v2.get("rr_mad_ms"),
        "rr_rmssd_ms": rhythm_v2.get("rr_rmssd_ms"),
        "rr_pnn50": rhythm_v2.get("rr_pnn50"),
        "beat_n": rhythm_v2.get("r_count"),
        "st_abnormal_leads": repol["st_abnormal_leads"],
        "st_elevation_leads": st_elevation,
        "st_depression_leads": st_depression,
        "st_direction": st_direction,
        "measurement_source": "CALIBRATED_DIGITAL_SIGNAL_V2",
        "rhythm_measurement_source": "CALIBRATED_DIGITAL_SIGNAL_V2",
        "rhythm_fields_suppressed": not rhythm_evaluable,
        "confidence_by_measurement": {
            "FC": _confidence(hr),
            "PR": _confidence(pr),
            "QRS": _confidence(qrs),
            "QT": _confidence(qt),
            "QTc_Bazett": _confidence(qtc),
            "QTc_Fridericia": _confidence(qtc_fridericia),
            "EJE": float(axis.get("confidence") or 0.0),
        },
    }

    if not st_evaluable:
        st_text = "NO EVALUABLE"
    else:
        parts: list[str] = []
        if st_depression:
            parts.append("DEPRESIÓN DEL ST EN " + ", ".join(st_depression))
        if st_elevation:
            parts.append("ELEVACIÓN DEL ST EN " + ", ".join(st_elevation))
        st_text = "; ".join(parts) if parts else "SIN DESVIACIÓN ST >0.10 mV EN DERIVACIONES CONFIABLES"

    t_text = (
        "POLARIDAD ATÍPICA EN " + ", ".join(t_unexpected)
        if t_unexpected
        else "SIN INVERSIÓN T INESPERADA EN DERIVACIONES CONFIABLES"
        if t_evaluable
        else "NO EVALUABLE"
    )
    axis_text = (
        f"{axis['category']} ({float(axis['degrees']):.0f}°; conf {float(axis.get('confidence') or 0.0):.2f})"
        if axis.get("evaluable") and axis.get("degrees") is not None
        else "NO EVALUABLE"
    )
    formatted = {
        "rhythm_text": rhythm_label,
        "heart_rate_text": _format_metric(hr, "LPM"),
        "axis_text": axis_text,
        "pr_text": _format_metric(pr, "MS"),
        "qrs_text": _format_metric(qrs, "MS"),
        "qt_text": _format_metric(qt, "MS"),
        "qtc_text": _format_metric(qtc, "MS"),
        "qtc_fridericia_text": _format_metric(qtc_fridericia, "MS"),
        "st_text": st_text,
        "t_text": t_text,
    }
    formatted["conclusion"] = (
        f"{rhythm_label}. FC {formatted['heart_rate_text']}. "
        f"QRS {formatted['qrs_text']}. PR {formatted['pr_text']}. "
        f"QT/QTc Bazett/Fridericia {formatted['qt_text']} / "
        f"{formatted['qtc_text']} / {formatted['qtc_fridericia_text']}. "
        f"{st_text}. {t_text}."
    )
    formatted["idx"] = "MEDICIÓN PRIMARIA SOBRE SEÑAL DIGITAL CALIBRADA"
    formatted["text"] = "\n".join([
        f"RITMO: {formatted['rhythm_text']}.",
        f"FC: {formatted['heart_rate_text']}.",
        f"EJE: {formatted['axis_text']}.",
        f"SEGMENTO PR: {formatted['pr_text']}.",
        f"COMPLEJO QRS: {formatted['qrs_text']}.",
        f"QT/QTC: {formatted['qt_text']} / {formatted['qtc_text']} "
        f"(Bazett) / {formatted['qtc_fridericia_text']} (Fridericia).",
        f"SEGMENTO ST: {formatted['st_text']}.",
        f"ONDA T: {formatted['t_text']}.",
        f"CONCLUSIÓN: {formatted['conclusion']}",
        f"IDX: {formatted['idx']}.",
    ])

    report.update({
        "version": "ECG_STRUCTURED_REPORT_V2_SIGNAL_PRIMARY",
        "source": "CALIBRATED_DIGITAL_SIGNAL_PRIMARY",
        "diagnostic_model": False,
        "rhythm": rhythm,
        "rhythm_screen": rhythm_screen,
        "axis": axis,
        "repolarization": repol,
        "measurement_summary": measurement_summary,
        "formatted": formatted,
        "digital_measurements_v2": digital_measurements,
        "canonical_signal_contract": canonical_ecg.get("contract"),
        "calibration": canonical_ecg.get("calibration") or {},
        "measurement_priority_rule": (
            "RELIABLE_NUMERIC_DIGITAL_MEASUREMENT_OVERRIDES_IMAGE_OR_MODEL_LABEL"
        ),
    })
    return report
