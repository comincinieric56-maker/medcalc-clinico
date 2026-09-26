from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import signal as sp_signal

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LIMB_LEADS = {"I","II","III","aVR","aVL","aVF"}
PRECORDIAL_LEADS = {"V1","V2","V3","V4","V5","V6"}


def _finite_spans(x: np.ndarray) -> List[Tuple[int, int]]:
    finite = np.isfinite(x)
    if not finite.any():
        return []
    d = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _longest_span(x: np.ndarray) -> Tuple[int, int] | None:
    spans = _finite_spans(x)
    if not spans:
        return None
    return max(spans, key=lambda ab: ab[1] - ab[0])


def _clean_segment(x: np.ndarray, fs: float) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if len(y) < int(fs):
        return y - np.nanmedian(y)

    y = y - np.nanmedian(y)
    try:
        sos = sp_signal.butter(
            3,
            [0.5, 40.0],
            btype="bandpass",
            fs=float(fs),
            output="sos",
        )
        y = sp_signal.sosfiltfilt(sos, y)
    except Exception:
        y = sp_signal.detrend(y)
    return y


def _nk_delineation(x_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    import neurokit2 as nk

    clean = nk.ecg_clean(x_mv, sampling_rate=fs, method="neurokit")
    _, peak_info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    r = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=int)

    waves: Dict[str, Any] = {}
    if len(r) >= 3:
        try:
            _, waves = nk.ecg_delineate(
                clean,
                rpeaks=r,
                sampling_rate=fs,
                method="dwt",
                show=False,
                show_type="all",
            )
        except Exception:
            waves = {}

    return {"clean": np.asarray(clean, dtype=float), "r": r, "waves": waves}


def _arr(waves: Dict[str, Any], key: str) -> np.ndarray:
    raw = waves.get(key, [])
    vals: List[float] = []
    for v in raw:
        try:
            z = float(v)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(z)
    return np.asarray(vals, dtype=float)


def _median_ms(delta_samples: np.ndarray, fs: int) -> float | None:
    z = np.asarray(delta_samples, dtype=float)
    z = z[np.isfinite(z) & (z > 0)]
    if z.size == 0:
        return None
    return float(np.median(z) * 1000.0 / fs)


def _nearest_preceding(values: np.ndarray, targets: np.ndarray, low: int, high: int) -> np.ndarray:
    out: List[float] = []
    if values.size == 0 or targets.size == 0:
        return np.asarray(out, dtype=float)
    for t in targets:
        candidates = values[(values < t - low) & (values > t - high)]
        if candidates.size:
            out.append(float(t - candidates[-1]))
    return np.asarray(out, dtype=float)


def _pairwise_duration(onsets: np.ndarray, offsets: np.ndarray, max_gap: int) -> np.ndarray:
    out: List[float] = []
    if onsets.size == 0 or offsets.size == 0:
        return np.asarray(out, dtype=float)
    for a in onsets:
        c = offsets[(offsets > a) & (offsets < a + max_gap)]
        if c.size:
            out.append(float(c[0] - a))
    return np.asarray(out, dtype=float)


def _rhythm_metrics(signal_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    pref = ["II", "I", "V5", "V1", "V6", "III", "aVF", "aVL", "aVR", "V2", "V3", "V4"]
    best = None
    for lead in pref:
        idx = LEADS.index(lead)
        span = _longest_span(signal_mv[:, idx])
        if span is None:
            continue
        length = span[1] - span[0]
        score = length + (3000 if lead == "II" else 0)
        if best is None or score > best[0]:
            best = (score, lead, idx, span)

    if best is None:
        return {
            "lead": None,
            "evaluable": False,
            "reason": "No existe un segmento continuo evaluable.",
        }

    _, lead, idx, (a, b) = best
    duration_s = (b - a) / fs
    if duration_s < 2.0:
        return {
            "lead": lead,
            "evaluable": False,
            "duration_s": duration_s,
            "reason": "Segmento de ritmo demasiado corto.",
        }

    x = signal_mv[a:b, idx]
    try:
        nk = _nk_delineation(x, fs)
    except Exception as exc:
        return {
            "lead": lead,
            "evaluable": False,
            "duration_s": duration_s,
            "reason": f"No fue posible detectar complejos QRS: {exc}",
        }

    r = nk["r"]
    waves = nk["waves"]
    if len(r) < 3:
        return {
            "lead": lead,
            "evaluable": False,
            "duration_s": duration_s,
            "r_count": int(len(r)),
            "reason": "Número insuficiente de complejos QRS.",
        }

    rr = np.diff(r) / fs
    med_rr = float(np.median(rr))
    hr = 60.0 / med_rr if med_rr > 0 else None
    cv = float(np.std(rr, ddof=1) / np.mean(rr)) if len(rr) >= 2 and np.mean(rr) > 0 else None
    regular = bool(cv is not None and cv <= 0.10)

    r_on = _arr(waves, "ECG_R_Onsets")
    r_off = _arr(waves, "ECG_R_Offsets")
    p_on = _arr(waves, "ECG_P_Onsets")
    p_peaks = _arr(waves, "ECG_P_Peaks")
    t_off = _arr(waves, "ECG_T_Offsets")

    pr_samples = _nearest_preceding(
        p_on,
        r_on if r_on.size else r.astype(float),
        low=int(0.06 * fs),
        high=int(0.40 * fs),
    )
    pr_ms = _median_ms(pr_samples, fs)

    qrs_samples = _pairwise_duration(
        r_on,
        r_off,
        max_gap=int(0.22 * fs),
    )
    qrs_ms = _median_ms(qrs_samples, fs)

    qt_samples = _pairwise_duration(
        r_on,
        t_off,
        max_gap=int(0.80 * fs),
    )
    qt_ms = _median_ms(qt_samples, fs)
    qtc_bazett_ms = (
        float(qt_ms / math.sqrt(med_rr))
        if qt_ms is not None and med_rr > 0
        else None
    )

    # A P wave preceding most QRS complexes is the operational criterion used
    # here for a sinus-compatible rhythm. This is descriptive, not a diagnostic
    # classifier.
    r_for_p = r_on if r_on.size else r.astype(float)
    p_before = _nearest_preceding(
        p_peaks,
        r_for_p,
        low=int(0.06 * fs),
        high=int(0.35 * fs),
    )
    p_ratio = float(len(p_before) / max(1, len(r_for_p)))

    p_positive = None
    if p_peaks.size and lead == "II":
        baseline = float(np.nanmedian(x))
        valid = [int(v) for v in p_peaks if 0 <= int(v) < len(x)]
        if valid:
            amp = np.asarray([x[v] - baseline for v in valid], dtype=float)
            p_positive = bool(np.nanmedian(amp) > 0)

    sinus_compatible = bool(
        p_ratio >= 0.75
        and (p_positive is not False)
        and (pr_ms is None or 80.0 <= pr_ms <= 240.0)
    )

    premature = 0
    if len(rr) >= 3:
        for i in range(len(rr) - 1):
            if rr[i] < 0.80 * med_rr and rr[i + 1] > 1.15 * med_rr:
                premature += 1

    return {
        "lead": lead,
        "evaluable": True,
        "duration_s": float(duration_s),
        "r_count": int(len(r)),
        "heart_rate_bpm": float(hr) if hr is not None else None,
        "rr_cv": cv,
        "regular": regular,
        "sinus_compatible": sinus_compatible,
        "p_before_qrs_ratio": p_ratio,
        "p_positive_in_ii": p_positive,
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_bazett_ms": qtc_bazett_ms,
        "premature_pattern_count": int(premature),
        "r_peaks_local": [int(v) for v in r.tolist()],
        "span_start": int(a),
        "span_end": int(b),
    }


def _lead_qrs_net(signal_mv: np.ndarray, fs: int, lead: str) -> float | None:
    idx = LEADS.index(lead)
    span = _longest_span(signal_mv[:, idx])
    if span is None or (span[1] - span[0]) < int(1.2 * fs):
        return None
    x = signal_mv[span[0]:span[1], idx]

    try:
        nk = _nk_delineation(x, fs)
        r = nk["r"]
    except Exception:
        return None

    if len(r) == 0:
        return None

    vals: List[float] = []
    for rp in r:
        q0 = max(0, int(rp - 0.055 * fs))
        q1 = min(len(x), int(rp + 0.075 * fs))
        b0 = max(0, int(rp - 0.22 * fs))
        b1 = max(b0 + 1, int(rp - 0.12 * fs))
        if q1 <= q0 or b1 <= b0:
            continue
        baseline = float(np.median(x[b0:b1]))
        vals.append(float(np.trapezoid(x[q0:q1] - baseline, dx=1.0 / fs)))

    if not vals:
        return None
    return float(np.median(vals))


def _axis_metrics(signal_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    lead_i = _lead_qrs_net(signal_mv, fs, "I")
    lead_avf = _lead_qrs_net(signal_mv, fs, "aVF")
    if lead_i is None or lead_avf is None:
        return {
            "evaluable": False,
            "degrees": None,
            "category": "NO EVALUABLE",
        }
    if abs(lead_i) < 1e-8 and abs(lead_avf) < 1e-8:
        return {
            "evaluable": False,
            "degrees": None,
            "category": "NO EVALUABLE",
        }

    deg = math.degrees(math.atan2(lead_avf, lead_i))
    if -30 <= deg <= 90:
        category = "EJE NORMAL"
    elif deg < -30 and deg >= -90:
        category = "DESVIACIÓN IZQUIERDA"
    elif deg > 90 and deg <= 180:
        category = "DESVIACIÓN DERECHA"
    else:
        category = "EJE EXTREMO"

    return {
        "evaluable": True,
        "degrees": float(deg),
        "category": category,
        "qrs_net_I": float(lead_i),
        "qrs_net_aVF": float(lead_avf),
    }


def _lead_st_t(signal_mv: np.ndarray, fs: int, lead: str) -> Dict[str, Any]:
    idx = LEADS.index(lead)
    span = _longest_span(signal_mv[:, idx])
    if span is None or (span[1] - span[0]) < int(1.2 * fs):
        return {"evaluable": False}

    x = signal_mv[span[0]:span[1], idx]
    try:
        nk = _nk_delineation(x, fs)
        r = nk["r"]
        waves = nk["waves"]
    except Exception:
        return {"evaluable": False}

    if len(r) == 0:
        return {"evaluable": False}

    r_off = _arr(waves, "ECG_R_Offsets")
    t_peaks = _arr(waves, "ECG_T_Peaks")
    st_vals: List[float] = []
    t_vals: List[float] = []

    for rp in r:
        b0 = max(0, int(rp - 0.22 * fs))
        b1 = max(b0 + 1, int(rp - 0.12 * fs))
        if b1 <= b0:
            continue
        baseline = float(np.median(x[b0:b1]))

        # Prefer delineated QRS offset, then use a conservative fixed fallback.
        off_candidates = r_off[(r_off > rp) & (r_off < rp + int(0.18 * fs))]
        j = int(off_candidates[0]) if off_candidates.size else int(rp + 0.07 * fs)
        st_idx = int(j + 0.06 * fs)
        if 0 <= st_idx < len(x):
            st_vals.append(float(x[st_idx] - baseline))

        tp = t_peaks[(t_peaks > rp + int(0.12 * fs)) & (t_peaks < rp + int(0.60 * fs))]
        if tp.size:
            ti = int(tp[0])
            if 0 <= ti < len(x):
                t_vals.append(float(x[ti] - baseline))

    return {
        "evaluable": bool(st_vals or t_vals),
        "st_mv": float(np.median(st_vals)) if st_vals else None,
        "t_mv": float(np.median(t_vals)) if t_vals else None,
    }


def _repolarization_metrics(signal_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    per_lead: Dict[str, Any] = {}
    st_abnormal: List[str] = []
    t_unexpected: List[str] = []

    for lead in LEADS:
        m = _lead_st_t(signal_mv, fs, lead)
        per_lead[lead] = m
        st = m.get("st_mv")
        if st is not None:
            # Descriptive screening threshold, not STEMI criteria.
            if abs(float(st)) > 0.10:
                st_abnormal.append(lead)

        tv = m.get("t_mv")
        if tv is not None:
            expected_positive = lead in {"I", "II", "V3", "V4", "V5", "V6"}
            expected_negative = lead == "aVR"
            if expected_positive and float(tv) < -0.05:
                t_unexpected.append(lead)
            if expected_negative and float(tv) > 0.05:
                t_unexpected.append(lead)

    eval_st = [lead for lead, m in per_lead.items() if m.get("st_mv") is not None]
    eval_t = [lead for lead, m in per_lead.items() if m.get("t_mv") is not None]

    return {
        "per_lead": per_lead,
        "st_evaluable_leads": eval_st,
        "st_abnormal_leads": st_abnormal,
        "st_isoelectric_compatible": bool(eval_st and not st_abnormal),
        "t_evaluable_leads": eval_t,
        "t_unexpected_polarity_leads": t_unexpected,
        "t_normal_polarity_compatible": bool(eval_t and not t_unexpected),
    }


def _fmt_ms(value: float | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NO EVALUABLE"
    return f"{float(value):.0f} MS"


def _format_report(
    rhythm: Dict[str, Any],
    axis: Dict[str, Any],
    repol: Dict[str, Any],
) -> Dict[str, Any]:
    if rhythm.get("evaluable"):
        if rhythm.get("sinus_compatible") and rhythm.get("regular"):
            rhythm_text = "SINUSAL Y REGULAR"
        elif rhythm.get("sinus_compatible"):
            rhythm_text = "SINUSAL, CON IRREGULARIDAD DEL RR"
        elif rhythm.get("regular"):
            rhythm_text = "REGULAR; ORIGEN SINUSAL NO DEMOSTRABLE"
        else:
            rhythm_text = "NO SINUSAL O NO REGULAR; REQUIERE REVISIÓN"

        hr = rhythm.get("heart_rate_bpm")
        fc_text = f"{float(hr):.0f} LPM" if hr is not None else "NO EVALUABLE"
    else:
        rhythm_text = "NO EVALUABLE"
        fc_text = "NO EVALUABLE"

    if axis.get("evaluable"):
        axis_text = f"{axis['category']} ({float(axis['degrees']):.0f}°)"
    else:
        axis_text = "NO EVALUABLE"

    pr = rhythm.get("pr_ms")
    if pr is None:
        pr_text = "NO EVALUABLE"
    else:
        qualifier = (
            "NORMAL" if 120 <= float(pr) <= 200
            else "PROLONGADO" if float(pr) > 200
            else "CORTO"
        )
        pr_text = f"{float(pr):.0f} MS ({qualifier})"

    qrs = rhythm.get("qrs_ms")
    if qrs is None:
        qrs_text = "NO EVALUABLE"
    else:
        qrs_text = f"{float(qrs):.0f} MS ({'NO PROLONGADO' if float(qrs) < 120 else 'PROLONGADO'})"

    if repol.get("st_evaluable_leads"):
        if repol.get("st_isoelectric_compatible"):
            st_text = "ISOELÉCTRICO EN DERIVACIONES EVALUABLES"
        else:
            st_text = "DESVIACIÓN DEL ST EN " + ", ".join(repol["st_abnormal_leads"])
    else:
        st_text = "NO EVALUABLE"

    if repol.get("t_evaluable_leads"):
        if repol.get("t_normal_polarity_compatible"):
            t_text = "SIN ALTERACIONES EVIDENTES DE POLARIDAD EN DERIVACIONES EVALUABLES"
        else:
            t_text = "POLARIDAD ATÍPICA EN " + ", ".join(repol["t_unexpected_polarity_leads"])
    else:
        t_text = "NO EVALUABLE"

    ectopy_count = int(rhythm.get("premature_pattern_count") or 0)
    if rhythm.get("evaluable") and rhythm.get("duration_s", 0) >= 5 and ectopy_count == 0:
        ectopy_text = "SIN EXTRASÍSTOLES EVIDENTES EN EL TRAZADO EVALUABLE"
    elif ectopy_count > 0:
        ectopy_text = f"PATRÓN PREMATURO COMPATIBLE CON EXTRASISTOLIA ({ectopy_count} EVENTO/S); REVISAR"
    else:
        ectopy_text = "EXTRASISTOLIA NO EVALUABLE"

    normal_axis = axis.get("category") == "EJE NORMAL"
    normal_pr = pr is not None and 120 <= float(pr) <= 200
    narrow_qrs = qrs is not None and float(qrs) < 120
    sinus_reg = bool(rhythm.get("sinus_compatible") and rhythm.get("regular"))
    st_ok = bool(repol.get("st_isoelectric_compatible"))
    t_ok = bool(repol.get("t_normal_polarity_compatible"))
    no_ectopy = ectopy_count == 0 and rhythm.get("duration_s", 0) >= 5

    conclusion_parts: List[str] = []
    if sinus_reg:
        conclusion_parts.append("RITMO SINUSAL REGULAR")
    if normal_pr:
        conclusion_parts.append("INTERVALO PR DENTRO DE RANGO")
    if narrow_qrs:
        conclusion_parts.append("COMPLEJO QRS NO PROLONGADO")
    if normal_axis:
        conclusion_parts.append("EJE ELÉCTRICO NO DESVIADO")
    if st_ok:
        conclusion_parts.append("SEGMENTO ST SIN DESVIACIONES SIGNIFICATIVAS EN DERIVACIONES EVALUABLES")
    if t_ok:
        conclusion_parts.append("ONDA T SIN ALTERACIONES EVIDENTES DE POLARIDAD EN DERIVACIONES EVALUABLES")
    if no_ectopy:
        conclusion_parts.append("SIN EXTRASÍSTOLES EVIDENTES")

    if not conclusion_parts:
        conclusion = (
            "TRAZADO PARCIALMENTE EVALUABLE; NO ES POSIBLE EMITIR UNA DESCRIPCIÓN "
            "AUTOMATIZADA COMPLETA CON LA INFORMACIÓN RECUPERADA."
        )
    else:
        conclusion = "ELECTROCARDIOGRAMA CON " + ", ".join(conclusion_parts) + "."

    if sinus_reg and normal_axis and normal_pr and narrow_qrs and st_ok and t_ok and no_ectopy:
        idx = (
            "TRAZADO COMPATIBLE CON RITMO SINUSAL REGULAR, SIN ALTERACIONES "
            "ELECTROCARDIOGRÁFICAS MAYORES EVIDENTES EN LOS SEGMENTOS EVALUABLES"
        )
    else:
        flags: List[str] = []
        if not sinus_reg:
            flags.append("RITMO")
        if not normal_axis:
            flags.append("EJE")
        if not normal_pr:
            flags.append("PR")
        if not narrow_qrs:
            flags.append("QRS")
        if not st_ok:
            flags.append("ST")
        if not t_ok:
            flags.append("T")
        if not no_ectopy:
            flags.append("ECTOPIA")
        idx = "REVISIÓN DIRIGIDA DE " + ", ".join(flags) if flags else "REVISIÓN MANUAL"

    lines = [
        f"RITMO: {rhythm_text}.",
        f"FC: {fc_text}.",
        f"EJE: {axis_text}.",
        f"SEGMENTO PR: {pr_text}.",
        f"COMPLEJO QRS: {qrs_text}.",
        f"SEGMENTO ST: {st_text}.",
        f"ONDA T: {t_text}.",
        f"{ectopy_text}.",
        f"CONCLUSIÓN: {conclusion}",
        f"IDX: {idx}.",
    ]

    return {
        "rhythm_text": rhythm_text,
        "heart_rate_text": fc_text,
        "axis_text": axis_text,
        "pr_text": pr_text,
        "qrs_text": qrs_text,
        "st_text": st_text,
        "t_text": t_text,
        "ectopy_text": ectopy_text,
        "conclusion": conclusion,
        "idx": idx,
        "text": "\n".join(lines),
    }


def build_structured_ecg_report(
    signal_uv: np.ndarray,
    *,
    fs: int = 500,
    lead_names: List[str] | None = None,
) -> Dict[str, Any]:
    x = np.asarray(signal_uv, dtype=float)
    if x.shape != (5000, 12):
        raise ValueError(f"Forma esperada (5000, 12); recibida {x.shape}.")
    if lead_names is not None and list(lead_names) != LEADS:
        raise ValueError("Orden de derivaciones inesperado.")

    signal_mv = x / 1000.0

    rhythm = _rhythm_metrics(signal_mv, fs)
    axis = _axis_metrics(signal_mv, fs)
    repol = _repolarization_metrics(signal_mv, fs)
    formatted = _format_report(rhythm, axis, repol)

    return {
        "version": "ECG_STRUCTURED_REPORT_V1",
        "source": "digitized_signal_only",
        "diagnostic_model": False,
        "sampling_rate_hz": int(fs),
        "rhythm": rhythm,
        "axis": axis,
        "repolarization": repol,
        "formatted": formatted,
        "limitations": [
            "Reporte descriptivo automatizado derivado de la señal reconstruida desde foto/PDF.",
            "No convierte probabilidades R27 en diagnósticos ni usa thresholds no validados.",
            "Los campos no demostrables se informan como NO EVALUABLE.",
        ],
    }
