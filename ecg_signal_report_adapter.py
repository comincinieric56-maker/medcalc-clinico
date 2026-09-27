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
    suffix = " - BAJA CONFIANZA" if conf < 0.45 else ""
    return f"{value:.{decimals}f} {unit} (conf {conf:.2f}){suffix}"


def _apply_wide_qrs_rhythm_hierarchy(
    rhythm_label: str,
    rhythm_code: str,
    rr_regularity_label: str,
    wct: Dict[str, Any] | None,
) -> tuple[str, str, str]:
    """Apply ventricular-origin/conduction evidence without erasing atrial rhythm.

    A supraventricular wide-QRS phenotype is a conduction modifier. It may
    decorate an established atrial mechanism (AF, flutter/AT, sinus, other SVT)
    but must never replace it. Only VT-compatible evidence may supersede the
    atrial layer because that changes the origin of the tachycardia itself.
    """
    wct = wct or {}
    wct_class = str(wct.get("classification") or "")
    wct_conf = float(wct.get("confidence") or 0.0)
    conduction_phenotype = "NOT_APPLICABLE"

    if not bool(wct.get("wide_complex_tachycardia")):
        return rhythm_label, rhythm_code, conduction_phenotype

    established_atrial = rhythm_code in {
        "AF_COMPATIBLE_RESEARCH",
        "FLUTTER_OR_AT_COMPATIBLE_RESEARCH",
        "SINUS_COMPATIBLE",
        "OTHER_SVT_COMPATIBLE_RESEARCH",
    }

    if wct_class == "VT_COMPATIBLE":
        return (
            "TAQUICARDIA DE QRS ANCHO COMPATIBLE CON TAQUICARDIA VENTRICULAR"
            f"; conf investigación {wct_conf:.2f}; "
            + rr_regularity_label,
            "VT_COMPATIBLE_RESEARCH",
            "VENTRICULAR_ORIGIN_COMPATIBLE",
        )

    if wct_class == "SVT_ABERRANCY_OR_PREEXCITATION_COMPATIBLE":
        conduction_phenotype = "WIDE_QRS_ABERRANCY_OR_PREEXCITATION_COMPATIBLE"
        if established_atrial:
            return (
                rhythm_label
                + "; QRS ANCHO COMPATIBLE CON CONDUCCIÓN ABERRANTE/PREEXCITACIÓN"
                + f" (conf investigación {wct_conf:.2f})",
                rhythm_code,
                conduction_phenotype,
            )
        return (
            "TAQUICARDIA SUPRAVENTRICULAR DE MECANISMO AURICULAR NO DEFINIDO; "
            "QRS ANCHO COMPATIBLE CON ABERRANCIA/PREEXCITACIÓN"
            f"; conf investigación {wct_conf:.2f}; "
            + rr_regularity_label,
            "SVT_WIDE_ATRIAL_MECHANISM_UNDETERMINED",
            conduction_phenotype,
        )

    conduction_phenotype = "WIDE_QRS_ORIGIN_UNDETERMINED"
    if established_atrial:
        return rhythm_label, rhythm_code, conduction_phenotype

    return (
        "TAQUICARDIA DE QRS ANCHO; MECANISMO VENTRICULAR VS SUPRAVENTRICULAR "
        "INDETERMINADO; "
        + rr_regularity_label,
        "WIDE_COMPLEX_TACHYCARDIA_UNDETERMINED",
        conduction_phenotype,
    )


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
    atrial_v2 = digital_measurements.get("atrial_activity") or {}
    atrial_mechanism = digital_measurements.get("atrial_mechanism") or {}
    wct = digital_measurements.get("wide_complex_tachycardia") or {}
    fascicular = digital_measurements.get("fascicular_conduction") or {}
    signal_integrity = digital_measurements.get("signal_integrity") or {}
    measurement_consensus = digital_measurements.get("measurement_consensus") or {}
    feature_graph = digital_measurements.get("feature_graph") or {}
    crosslead_conduction = digital_measurements.get("crosslead_conduction") or {}
    consistency = digital_measurements.get("consistency") or {}
    specialist_reasoning = digital_measurements.get("specialist_reasoning") or {}
    ectopy = digital_measurements.get("ectopy") or {}
    qrs_morphology = digital_measurements.get("qrs_morphology") or {}
    av_conduction = digital_measurements.get("av_conduction") or {}
    preexcitation = digital_measurements.get("preexcitation") or {}
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
    sinus_compatible = bool(atrial_v2.get("sinus_compatible"))
    p_reproducible = bool(atrial_v2.get("p_wave_reproducible"))
    heart_rate_value = _value(hr)

    rr_regularity_label = (
        "RR REGULARES"
        if rhythm_evaluable and regular
        else "RR IRREGULARES"
        if rhythm_evaluable
        else "REGULARIDAD RR NO EVALUABLE"
    )

    atrial_mechanism_code = str(atrial_mechanism.get("mechanism") or "")
    atrial_mechanism_conf = float(atrial_mechanism.get("confidence") or 0.0)

    if not rhythm_evaluable:
        rhythm_label = "MECANISMO DEL RITMO NO EVALUABLE"
        rhythm_code = "NOT_EVALUABLE"
    elif sinus_compatible:
        rhythm_label = (
            "RITMO SINUSAL COMPATIBLE; " + rr_regularity_label
        )
        rhythm_code = "SINUS_COMPATIBLE"
    elif atrial_mechanism_code == "AF_COMPATIBLE":
        rhythm_label = (
            "PATRÓN AURICULAR COMPATIBLE CON FIBRILACIÓN AURICULAR; "
            + rr_regularity_label
            + f"; conf investigación {atrial_mechanism_conf:.2f}"
        )
        rhythm_code = "AF_COMPATIBLE_RESEARCH"
    elif atrial_mechanism_code == "FLUTTER_OR_AT_COMPATIBLE":
        rhythm_label = (
            "PATRÓN AURICULAR ORGANIZADO COMPATIBLE CON FLUTTER/TAQUICARDIA AURICULAR; "
            + rr_regularity_label
            + f"; conf investigación {atrial_mechanism_conf:.2f}"
        )
        rhythm_code = "FLUTTER_OR_AT_COMPATIBLE_RESEARCH"
    elif atrial_mechanism_code == "OTHER_SVT_COMPATIBLE":
        rhythm_label = (
            "TAQUICARDIA SUPRAVENTRICULAR COMPATIBLE; "
            + rr_regularity_label
            + "; MECANISMO AURICULAR NO DEFINIDO"
        )
        rhythm_code = "OTHER_SVT_COMPATIBLE_RESEARCH"
    elif not p_reproducible:
        tachy = bool(heart_rate_value is not None and heart_rate_value >= 100.0)
        rhythm_label = (
            ("TAQUICARDIA; " if tachy else "")
            + rr_regularity_label
            + "; SIN ONDAS P REPRODUCIBLES; MECANISMO AURICULAR INDETERMINADO"
        )
        rhythm_code = "NO_REPRODUCIBLE_P_MECHANISM_UNDETERMINED"
    else:
        rhythm_label = (
            rr_regularity_label
            + "; ACTIVIDAD AURICULAR PRESENTE, MECANISMO NO SINUSAL/NO DETERMINADO"
        )
        rhythm_code = "ATRIAL_ACTIVITY_NON_SINUS_UNDETERMINED"

    rhythm_label, rhythm_code, conduction_phenotype = (
        _apply_wide_qrs_rhythm_hierarchy(
            rhythm_label,
            rhythm_code,
            rr_regularity_label,
            wct,
        )
    )

    rhythm_blocking_codes = {
        "AF_VS_REPRODUCIBLE_P_QRS_CONFLICT",
        "SINUS_WITHOUT_REPRODUCIBLE_P_CONFLICT",
        "WCT_ACTIVATED_OUTSIDE_GATE",
    }
    active_conflicts = list(consistency.get("conflicts") or [])
    if any(
        str(item.get("code") or "") in rhythm_blocking_codes
        and str(item.get("severity") or "") == "BLOCKING"
        for item in active_conflicts
    ):
        rhythm_label = (
            "MECANISMO DEL RITMO NO PUBLICABLE POR CONTRADICCIÓN INTERNA; "
            "REQUIERE REVISIÓN DE SEÑAL/FIDUCIALES"
        )
        rhythm_code = "CONSISTENCY_BLOCKED"
    primary_reasoned = dict(specialist_reasoning.get("primary_rhythm") or {})
    reasoned_code = str(primary_reasoned.get("code") or "")
    reasoned_conf = float(primary_reasoned.get("confidence") or 0.0)
    reasoned_labels = {
        "SINUS_COMPATIBLE": "RITMO SINUSAL COMPATIBLE",
        "SINUS_BRADYCARDIA_COMPATIBLE": "BRADICARDIA SINUSAL COMPATIBLE",
        "SINUS_TACHYCARDIA_COMPATIBLE": "TAQUICARDIA SINUSAL COMPATIBLE",
        "AF_COMPATIBLE": "PATRÓN COMPATIBLE CON FIBRILACIÓN AURICULAR",
        "FLUTTER_OR_AT_COMPATIBLE": "PATRÓN AURICULAR ORGANIZADO COMPATIBLE CON FLUTTER/TAQUICARDIA AURICULAR",
        "OTHER_SVT_COMPATIBLE": "TAQUICARDIA SUPRAVENTRICULAR COMPATIBLE",
        "VT_COMPATIBLE": "TAQUICARDIA DE QRS ANCHO COMPATIBLE CON TAQUICARDIA VENTRICULAR",
        "RHYTHM_MECHANISM_UNDETERMINED": "MECANISMO DEL RITMO INDETERMINADO",
    }
    if reasoned_code in reasoned_labels:
        rhythm_label = (
            reasoned_labels[reasoned_code]
            + ("; " + rr_regularity_label if rhythm_evaluable else "")
            + f"; conf especialista {reasoned_conf:.2f}"
        )
        rhythm_code = reasoned_code
        if (
            str(wct.get("classification") or "")
            == "SVT_ABERRANCY_OR_PREEXCITATION_COMPATIBLE"
            and reasoned_code != "VT_COMPATIBLE"
        ):
            rhythm_label += "; QRS ANCHO CON FENOTIPO DE ABERRANCIA/PREEXCITACIÓN"

    rhythm = {
        "evaluable": rhythm_evaluable,
        "lead": rhythm_v2.get("lead"),
        "duration_s": (
            (digital_measurements.get("leads") or {})
            .get(str(rhythm_v2.get("lead") or ""), {})
            .get("duration_s")
        ),
        "r_count": rhythm_v2.get("r_count"),
        "r_peaks_local": list(rhythm_v2.get("r_peaks_samples") or []),
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
        "rr_regularity_label": rr_regularity_label,
        "confidence": rhythm_v2.get("confidence"),
        "signal_source": "CALIBRATED_DIGITAL_SIGNAL",
        "source": "CALIBRATED_DIGITAL_SIGNAL",
        "sinus_compatible": sinus_compatible,
        "p_wave_reproducible": p_reproducible,
        "p_qrs_coupling_fraction": atrial_v2.get(
            "rhythm_p_qrs_coupling_fraction"
        ),
        "atrial_activity": atrial_v2,
        "atrial_mechanism_analysis": atrial_mechanism,
        "wide_complex_tachycardia_analysis": wct,
        "conduction_phenotype": conduction_phenotype,
        "mechanism_code": rhythm_code,
        "reason": (
            atrial_v2.get("reason")
            or rhythm_v2.get("reason")
        ),
    }

    rhythm_screen = {
        "evaluable": rhythm_evaluable,
        "code": rhythm_code,
        "label": rhythm_label,
        "rr_regularity": rr_regularity_label,
        "source": "CALIBRATED_DIGITAL_SIGNAL_RR_PLUS_NATIVE_ATRIAL_ANALYZER",
        "basis": (
            [
                f"RR CV {float(rhythm_v2.get('rr_cv')):.3f}"
                if rhythm_v2.get("rr_cv") is not None else "",
                f"RR MAD {float(rhythm_v2.get('rr_mad_ms')):.1f} ms"
                if rhythm_v2.get("rr_mad_ms") is not None else "",
                f"{int(rhythm_v2.get('r_count') or 0)} QRS detectados",
                (
                    "P-QRS reproducible"
                    if p_reproducible
                    else "ondas P no reproducibles"
                ),
                (
                    f"acoplamiento P-QRS "
                    f"{float(atrial_v2.get('rhythm_p_qrs_coupling_fraction') or 0.0):.2f}"
                ),
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
        st_measured = st_value is not None
        t_measured = t_value is not None
        st_ok = st_measured and st_conf >= 0.45
        t_ok = t_measured and t_conf >= 0.45
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
            "evaluable": bool(st_measured or t_measured),
            "st_mv": st_value,
            "st_confidence": st_conf,
            "st_reliable": bool(st_ok),
            "st_direction": st.get("direction"),
            "st_mm": st.get("mm_at_paper_gain"),
            "t_mv": t_value,
            "t_confidence": t_conf,
            "t_reliable": bool(t_ok),
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

    st_measured_leads = [
        lead for lead, item in per_lead_repol.items()
        if item.get("st_mv") is not None
    ]
    t_measured_leads = [
        lead for lead, item in per_lead_repol.items()
        if item.get("t_mv") is not None
    ]

    repol = {
        "per_lead": per_lead_repol,
        "st_evaluable_leads": st_evaluable,
        "st_measured_leads": st_measured_leads,
        "st_abnormal_leads": st_elevation + st_depression,
        "st_elevation_leads": st_elevation,
        "st_depression_leads": st_depression,
        "st_direction": st_direction,
        "st_isoelectric_compatible": bool(st_evaluable and not st_elevation and not st_depression),
        "t_evaluable_leads": t_evaluable,
        "t_measured_leads": t_measured_leads,
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
        "sinus_compatible": sinus_compatible,
        "p_wave_reproducible": p_reproducible,
        "p_qrs_coupling_fraction": atrial_v2.get(
            "rhythm_p_qrs_coupling_fraction"
        ),
        "rhythm_fields_suppressed": not rhythm_evaluable,
        "signal_integrity_quality": signal_integrity.get("overall_quality"),
        "measurement_consensus_quality": measurement_consensus.get(
            "overall_measurement_quality"
        ),
        "remeasure_required": bool(measurement_consensus.get("remeasure_required")),
        "remeasure_targets": list(measurement_consensus.get("remeasure_targets") or []),
        "consistency_status": consistency.get("status"),
        "reasoner_primary_code": rhythm_code,
        "reasoner_primary_confidence": reasoned_conf,
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

    if st_evaluable:
        parts: list[str] = []
        if st_depression:
            parts.append("DEPRESIÓN DEL ST EN " + ", ".join(st_depression))
        if st_elevation:
            parts.append("ELEVACIÓN DEL ST EN " + ", ".join(st_elevation))
        st_text = "; ".join(parts) if parts else "SIN DESVIACIÓN ST >0.10 mV EN DERIVACIONES CONFIABLES"
    elif st_measured_leads:
        st_text = (
            "ST MEDIDO CON BAJA CONFIANZA EN "
            + ", ".join(st_measured_leads)
            + "; SIN CLASIFICACIÓN CATEGÓRICA"
        )
    else:
        st_text = "NO EVALUABLE"

    t_text = (
        "POLARIDAD ATÍPICA EN " + ", ".join(t_unexpected)
        if t_unexpected
        else "SIN INVERSIÓN T INESPERADA EN DERIVACIONES CONFIABLES"
        if t_evaluable
        else "ONDA T MEDIDA CON BAJA CONFIANZA EN " + ", ".join(t_measured_leads)
        if t_measured_leads
        else "NO EVALUABLE"
    )
    axis_text = (
        f"{axis['category']} ({float(axis['degrees']):.0f}°; conf {float(axis.get('confidence') or 0.0):.2f})"
        if axis.get("evaluable") and axis.get("degrees") is not None
        else "NO EVALUABLE"
    )
    pr_text = _format_metric(pr, "MS")
    if (
        _value(pr) is None
        and str((pr or {}).get("reason") or "") == "P_WAVES_NOT_REPRODUCIBLE"
    ):
        pr_text = "NO EVALUABLE - ONDAS P NO REPRODUCIBLES"

    fascicular_blocked = any(
        str(item.get("code") or "") in {
            "LAFB_WITHOUT_REQUIRED_AXIS_CONFLICT",
            "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT",
        }
        and str(item.get("severity") or "") == "BLOCKING"
        for item in active_conflicts
    )
    if str(fascicular.get("classification") or "") == "LAFB_COMPATIBLE" and not fascicular_blocked:
        fascicular_text = (
            "PATRÓN COMPATIBLE CON HEMIBLOQUEO ANTEROSUPERIOR IZQUIERDO (HBAI/LAFB)"
            f" (conf {float(fascicular.get('confidence') or 0.0):.2f})"
        )
    elif str(fascicular.get("classification") or "") == "LAFB_COMPATIBLE" and fascicular_blocked:
        fascicular_text = (
            "PATRÓN FASCICULAR NO PUBLICABLE HASTA RESOLVER DISCORDANCIA DE MEDICIÓN"
        )
    elif fascicular.get("evaluable"):
        fascicular_text = "SIN PATRÓN FASCICULAR ESPECÍFICO ESTABLECIDO"
    else:
        fascicular_text = "CONDUCCIÓN FASCICULAR NO EVALUABLE"

    reasoned_conduction = list(specialist_reasoning.get("conduction_findings") or [])
    bundle_codes = [str(row.get("code") or "") for row in reasoned_conduction]
    if "RBBB_MORPHOLOGY_COMPATIBLE" in bundle_codes:
        bundle_text = "PATRÓN MULTIDERIVACIÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA DERECHA"
    elif "LBBB_MORPHOLOGY_COMPATIBLE" in bundle_codes:
        bundle_text = "PATRÓN MULTIDERIVACIÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA IZQUIERDA"
    elif "INCOMPLETE_RBBB_MORPHOLOGY_COMPATIBLE" in bundle_codes:
        bundle_text = "PATRÓN COMPATIBLE CON BLOQUEO INCOMPLETO DE RAMA DERECHA"
    elif "INCOMPLETE_LBBB_MORPHOLOGY_COMPATIBLE" in bundle_codes:
        bundle_text = "PATRÓN COMPATIBLE CON BLOQUEO INCOMPLETO DE RAMA IZQUIERDA"
    else:
        bundle_text = "SIN PATRÓN DE BLOQUEO DE RAMA ESTABLECIDO"

    av_finding = specialist_reasoning.get("av_conduction_finding") or {}
    av_code = str(av_finding.get("code") or "")
    av_labels = {
        "FIRST_DEGREE_AV_DELAY_COMPATIBLE": "RETARDO AV DE PRIMER GRADO COMPATIBLE",
        "MOBITZ_I_WENCKEBACH_COMPATIBLE": "BLOQUEO AV DE SEGUNDO GRADO MOBITZ I/WENCKEBACH COMPATIBLE",
        "MOBITZ_II_COMPATIBLE": "BLOQUEO AV DE SEGUNDO GRADO MOBITZ II COMPATIBLE",
        "TWO_TO_ONE_AV_BLOCK_COMPATIBLE": "BLOQUEO AV 2:1 COMPATIBLE",
        "HIGH_GRADE_AV_BLOCK_COMPATIBLE": "BLOQUEO AV DE ALTO GRADO COMPATIBLE",
    }
    av_text = av_labels.get(av_code, "SIN BLOQUEO AV ESPECÍFICO ESTABLECIDO")

    pre_finding = specialist_reasoning.get("preexcitation_finding") or {}
    preexc_text = (
        "PATRÓN COMPATIBLE CON PREEXCITACIÓN VENTRICULAR"
        if str(pre_finding.get("code") or "") == "VENTRICULAR_PREEXCITATION_COMPATIBLE"
        else "SIN PATRÓN DE PREEXCITACIÓN ESTABLECIDO"
    )

    ectopy_rows = list(specialist_reasoning.get("ectopy_findings") or [])
    ectopy_parts = []
    for row in ectopy_rows:
        code = str(row.get("code") or "")
        count = int(row.get("count") or 0)
        if code == "PVC_COMPATIBLE":
            ectopy_parts.append(f"{count} LATIDO(S) VENTRICULAR(ES) PREMATURO(S) COMPATIBLE(S)")
        elif code == "PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE":
            ectopy_parts.append(f"{count} LATIDO(S) SUPRAVENTRICULAR(ES) PREMATURO(S) COMPATIBLE(S)")
    ectopy_text = "; ".join(ectopy_parts) if ectopy_parts else "SIN ECTOPIA ESPECÍFICA ESTABLECIDA"

    formatted = {
        "rhythm_text": rhythm_label,
        "heart_rate_text": _format_metric(hr, "LPM"),
        "axis_text": axis_text,
        "pr_text": pr_text,
        "qrs_text": _format_metric(qrs, "MS"),
        "qt_text": _format_metric(qt, "MS"),
        "qtc_text": _format_metric(qtc, "MS"),
        "qtc_fridericia_text": _format_metric(qtc_fridericia, "MS"),
        "st_text": st_text,
        "t_text": t_text,
        "fascicular_text": fascicular_text,
        "bundle_text": bundle_text,
        "av_text": av_text,
        "preexcitation_text": preexc_text,
        "ectopy_text": ectopy_text,
    }
    formatted["conclusion"] = (
        f"{rhythm_label}. FC {formatted['heart_rate_text']}. "
        f"QRS {formatted['qrs_text']}. PR {formatted['pr_text']}. "
        f"QT/QTc Bazett/Fridericia {formatted['qt_text']} / "
        f"{formatted['qtc_text']} / {formatted['qtc_fridericia_text']}. "
        f"{st_text}. {t_text}. {fascicular_text}. "
        f"{bundle_text}. {av_text}. {preexc_text}. {ectopy_text}."
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
        f"CONDUCCIÓN FASCICULAR: {formatted['fascicular_text']}.",
        f"CONDUCCIÓN DE RAMA: {formatted['bundle_text']}.",
        f"CONDUCCIÓN AV: {formatted['av_text']}.",
        f"PREEXCITACIÓN: {formatted['preexcitation_text']}.",
        f"ECTOPIA: {formatted['ectopy_text']}.",
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
        "atrial_activity": atrial_v2,
        "atrial_mechanism": atrial_mechanism,
        "wide_complex_tachycardia": wct,
        "fascicular_conduction": fascicular,
        "signal_integrity": signal_integrity,
        "measurement_consensus": measurement_consensus,
        "feature_graph": feature_graph,
        "crosslead_conduction": crosslead_conduction,
        "consistency": consistency,
        "specialist_reasoning": specialist_reasoning,
        "ectopy": ectopy,
        "qrs_morphology": qrs_morphology,
        "av_conduction": av_conduction,
        "preexcitation": preexcitation,
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
