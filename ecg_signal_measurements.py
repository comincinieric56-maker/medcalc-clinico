from __future__ import annotations

"""Numerical ECG measurements from calibrated digital signals.

No image pixels are interpreted clinically in this module.  Input is a
DigitalECG whose samples are already calibrated in mV at a known sampling rate.
The module exposes numeric values, fiducials, uncertainty/confidence and explicit
not-measurable reasons. Classifiers may consume these measurements later, but
must not overwrite a reliable signed numeric measurement.
"""

from typing import Any, Iterable

import math
import numpy as np
from scipy import signal as sp_signal

from ecg_digital_signal import DigitalECG, DigitalLead, LEADS


def _finite_number(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _wave_array(waves: dict[str, Any], key: str) -> np.ndarray:
    vals: list[int] = []
    for value in waves.get(key, []) or []:
        z = _finite_number(value)
        if z is not None:
            vals.append(int(round(z)))
    return np.asarray(vals, dtype=int)


def _nearest_before(values: np.ndarray, target: int, lo: int, hi: int) -> int | None:
    if values.size == 0:
        return None
    cand = values[(values <= target - lo) & (values >= target - hi)]
    return int(cand[-1]) if cand.size else None


def _nearest_after(values: np.ndarray, target: int, lo: int, hi: int) -> int | None:
    if values.size == 0:
        return None
    cand = values[(values >= target + lo) & (values <= target + hi)]
    return int(cand[0]) if cand.size else None


def _robust_summary(values: Iterable[float]) -> dict[str, Any]:
    z = np.asarray(list(values), dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return {
            "value": None,
            "median": None,
            "mean": None,
            "sd": None,
            "mad": None,
            "n": 0,
            "iqr": None,
        }
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    return {
        "value": med,
        "median": med,
        "mean": float(np.mean(z)),
        "sd": float(np.std(z, ddof=1)) if z.size >= 2 else 0.0,
        "mad": mad,
        "n": int(z.size),
        "iqr": (
            float(np.percentile(z, 75) - np.percentile(z, 25))
            if z.size >= 4 else
            float(np.max(z) - np.min(z))
        ),
    }


def _clean_for_fiducials(x: np.ndarray, fs: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if y.size < max(10, fs // 2):
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
        return sp_signal.sosfiltfilt(sos, y)
    except Exception:
        return sp_signal.detrend(y)


def _delineate_segment(x_mv: np.ndarray, fs: int) -> dict[str, Any]:
    import neurokit2 as nk

    clean = nk.ecg_clean(x_mv, sampling_rate=fs, method="neurokit")
    _, peak_info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    r = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=int)
    waves: dict[str, Any] = {}
    if len(r) >= 2:
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
    return {
        "clean": np.asarray(clean, dtype=float),
        "r": r,
        "waves": waves,
    }


def _longest_finite_span(x: np.ndarray) -> tuple[int, int] | None:
    finite = np.isfinite(np.asarray(x, dtype=float))
    if not finite.any():
        return None
    d = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if not len(starts):
        return None
    lengths = ends - starts
    i = int(np.argmax(lengths))
    return int(starts[i]), int(ends[i])


def _baseline_for_beat(
    raw: np.ndarray,
    fs: int,
    *,
    r: int,
    p_on: int | None,
    p_off: int | None,
    t_off_prev: int | None,
) -> tuple[float | None, str]:
    candidates: list[tuple[int, int, str]] = []

    # TP is the preferred isoelectric interval when it is explicitly available.
    if t_off_prev is not None and p_on is not None:
        a = int(t_off_prev + 0.04 * fs)
        b = int(p_on - 0.02 * fs)
        if b - a >= int(0.025 * fs):
            candidates.append((a, b, "TP"))

    # PR segment is the second choice.
    if p_off is not None:
        a = int(p_off + 0.02 * fs)
        b = int(r - 0.05 * fs)
        if b - a >= int(0.025 * fs):
            candidates.append((a, b, "PR"))

    # Conservative pre-QRS fallback.  This is explicitly labelled because it
    # is not guaranteed to be a true TP/PR segment.
    candidates.append((
        int(r - 0.20 * fs),
        int(r - 0.10 * fs),
        "PRE_QRS_FALLBACK",
    ))

    for a, b, source in candidates:
        a = max(0, a)
        b = min(len(raw), b)
        if b <= a:
            continue
        segment = np.asarray(raw[a:b], dtype=float)
        segment = segment[np.isfinite(segment)]
        if segment.size >= max(3, int(0.02 * fs)):
            return float(np.median(segment)), source
    return None, "NO_ISOELECTRIC_SEGMENT"


def _sample_at(raw: np.ndarray, idx: int | None) -> float | None:
    if idx is None or not 0 <= int(idx) < len(raw):
        return None
    value = _finite_number(raw[int(idx)])
    return value


def _q_wave_metrics(
    raw: np.ndarray,
    fs: int,
    *,
    qrs_on: int | None,
    r: int,
    baseline: float,
) -> tuple[float | None, float | None]:
    if qrs_on is None or not 0 <= qrs_on < r < len(raw):
        return None, None
    seg = np.asarray(raw[qrs_on:r + 1], dtype=float) - float(baseline)
    if not np.isfinite(seg).any():
        return None, None
    q_local = int(np.nanargmin(seg))
    q_amp = float(seg[q_local])
    if q_amp >= -0.005:
        return 0.0, 0.0

    threshold = min(-0.005, 0.20 * q_amp)
    mask = np.isfinite(seg) & (seg <= threshold)
    if not mask[q_local]:
        return q_amp, None
    a = q_local
    b = q_local
    while a > 0 and mask[a - 1]:
        a -= 1
    while b + 1 < len(mask) and mask[b + 1]:
        b += 1
    duration_ms = 1000.0 * (b - a + 1) / fs
    return q_amp, float(duration_ms)


def _lead_measurements(lead: DigitalLead, gain_mm_per_mv: float) -> dict[str, Any]:
    fs = int(lead.fs)
    signal_mv = np.asarray(lead.signal_mv, dtype=float)
    span = _longest_finite_span(signal_mv)
    if span is None:
        return {
            "evaluable": False,
            "reason": "NO_FINITE_SIGNAL",
            "confidence": 0.0,
        }

    a0, b0 = span
    raw = signal_mv[a0:b0]
    duration_s = float(len(raw) / fs)
    if duration_s < 1.0:
        return {
            "evaluable": False,
            "reason": "CONTIGUOUS_SIGNAL_LT_1S",
            "duration_s": duration_s,
            "confidence": float(lead.confidence * 0.3),
        }

    try:
        delineation = _delineate_segment(raw, fs)
    except Exception as exc:
        return {
            "evaluable": False,
            "reason": "FIDUCIAL_DETECTION_FAILED",
            "detail": str(exc),
            "duration_s": duration_s,
            "confidence": float(lead.confidence * 0.25),
        }

    r = np.asarray(delineation.get("r", []), dtype=int)
    waves = delineation.get("waves", {}) or {}
    if r.size < 2:
        return {
            "evaluable": False,
            "reason": "LT_2_QRS",
            "r_count": int(r.size),
            "duration_s": duration_s,
            "confidence": float(lead.confidence * 0.35),
        }

    qrs_on = _wave_array(waves, "ECG_R_Onsets")
    qrs_off = _wave_array(waves, "ECG_R_Offsets")
    p_on = _wave_array(waves, "ECG_P_Onsets")
    p_off = _wave_array(waves, "ECG_P_Offsets")
    p_peak = _wave_array(waves, "ECG_P_Peaks")
    t_peak = _wave_array(waves, "ECG_T_Peaks")
    t_off = _wave_array(waves, "ECG_T_Offsets")

    rr_ms = np.diff(r).astype(float) * 1000.0 / fs
    rr_stats = _robust_summary(rr_ms)

    qrs_values: list[float] = []
    p_durations: list[float] = []
    pr_values: list[float] = []
    qt_values: list[float] = []
    st_j: list[float] = []
    st_40: list[float] = []
    st_60: list[float] = []
    st_80: list[float] = []
    r_amp: list[float] = []
    s_amp: list[float] = []
    q_amp: list[float] = []
    q_dur: list[float] = []
    t_amp: list[float] = []
    baseline_sources: list[str] = []
    beat_records: list[dict[str, Any]] = []

    previous_t_off: int | None = None
    for rp in r:
        rp = int(rp)
        qon = _nearest_before(qrs_on, rp + int(0.02 * fs), 0, int(0.14 * fs))
        qoff = _nearest_after(qrs_off, rp - int(0.02 * fs), 0, int(0.18 * fs))
        pon = _nearest_before(p_on, rp, int(0.06 * fs), int(0.40 * fs))
        poff = _nearest_before(p_off, rp, int(0.03 * fs), int(0.30 * fs))
        ppk = _nearest_before(p_peak, rp, int(0.04 * fs), int(0.35 * fs))
        tpk = _nearest_after(t_peak, rp, int(0.10 * fs), int(0.60 * fs))
        toff = _nearest_after(t_off, rp, int(0.12 * fs), int(0.80 * fs))

        baseline, baseline_source = _baseline_for_beat(
            raw,
            fs,
            r=rp,
            p_on=pon,
            p_off=poff,
            t_off_prev=previous_t_off,
        )
        if toff is not None:
            previous_t_off = toff

        rec: dict[str, Any] = {
            "r_sample": int(rp + a0),
            "qrs_on_sample": int(qon + a0) if qon is not None else None,
            "qrs_off_sample": int(qoff + a0) if qoff is not None else None,
            "p_on_sample": int(pon + a0) if pon is not None else None,
            "p_off_sample": int(poff + a0) if poff is not None else None,
            "p_peak_sample": int(ppk + a0) if ppk is not None else None,
            "t_peak_sample": int(tpk + a0) if tpk is not None else None,
            "t_off_sample": int(toff + a0) if toff is not None else None,
            "baseline_source": baseline_source,
        }

        if qon is not None and qoff is not None and qoff > qon:
            qrs_ms = 1000.0 * (qoff - qon) / fs
            if 30.0 <= qrs_ms <= 260.0:
                qrs_values.append(float(qrs_ms))
                rec["qrs_ms"] = float(qrs_ms)

        if pon is not None and poff is not None and poff > pon:
            p_ms = 1000.0 * (poff - pon) / fs
            if 20.0 <= p_ms <= 220.0:
                p_durations.append(float(p_ms))
                rec["p_duration_ms"] = float(p_ms)

        if pon is not None and qon is not None and qon > pon:
            pr_ms = 1000.0 * (qon - pon) / fs
            if 60.0 <= pr_ms <= 450.0:
                pr_values.append(float(pr_ms))
                rec["pr_ms"] = float(pr_ms)

        if qon is not None and toff is not None and toff > qon:
            qt_ms = 1000.0 * (toff - qon) / fs
            if 120.0 <= qt_ms <= 800.0:
                qt_values.append(float(qt_ms))
                rec["qt_ms"] = float(qt_ms)

        if baseline is not None:
            baseline_sources.append(baseline_source)
            rec["baseline_mv"] = float(baseline)

            rv = _sample_at(raw, rp)
            if rv is not None:
                val = float(rv - baseline)
                r_amp.append(val)
                rec["r_amplitude_mv"] = val

            if qoff is not None and qoff > rp:
                s0 = min(len(raw), rp + max(1, int(0.008 * fs)))
                s1 = min(len(raw), qoff + 1)
                if s1 > s0:
                    seg_s = raw[s0:s1] - baseline
                    if np.isfinite(seg_s).any():
                        sval = float(np.nanmin(seg_s))
                        s_amp.append(sval)
                        rec["s_amplitude_mv"] = sval

            qv, qms = _q_wave_metrics(
                raw,
                fs,
                qrs_on=qon,
                r=rp,
                baseline=float(baseline),
            )
            if qv is not None:
                q_amp.append(float(qv))
                rec["q_amplitude_mv"] = float(qv)
            if qms is not None:
                q_dur.append(float(qms))
                rec["q_duration_ms"] = float(qms)

            if tpk is not None:
                tv = _sample_at(raw, tpk)
                if tv is not None:
                    val = float(tv - baseline)
                    t_amp.append(val)
                    rec["t_amplitude_mv"] = val

            if qoff is not None:
                offsets = {
                    "j": 0,
                    "j40": int(round(0.040 * fs)),
                    "j60": int(round(0.060 * fs)),
                    "j80": int(round(0.080 * fs)),
                }
                for key, delta in offsets.items():
                    value = _sample_at(raw, qoff + delta)
                    if value is None:
                        continue
                    st_value = float(value - baseline)
                    rec[f"st_{key}_mv"] = st_value
                    if key == "j":
                        st_j.append(st_value)
                    elif key == "j40":
                        st_40.append(st_value)
                    elif key == "j60":
                        st_60.append(st_value)
                    elif key == "j80":
                        st_80.append(st_value)

        beat_records.append(rec)

    def metric(values: list[float], name: str, min_n: int = 1) -> dict[str, Any]:
        stats = _robust_summary(values)
        n = int(stats["n"])
        completeness = min(1.0, n / max(int(r.size), 1))
        variability = 0.0
        if stats["median"] is not None and stats["mad"] is not None:
            denom = max(abs(float(stats["median"])), 0.03)
            variability = min(1.0, float(stats["mad"]) / denom)
        confidence = float(np.clip(
            lead.confidence
            * (0.45 + 0.55 * completeness)
            * (1.0 - 0.35 * variability),
            0.0,
            1.0,
        ))
        reportable = bool(n >= min_n)
        return {
            **stats,
            "name": name,
            "confidence": confidence if reportable else 0.0,
            "reportable": reportable,
            "reason": None if reportable else "INSUFFICIENT_FIDUCIALS",
        }

    qrs_metric = metric(qrs_values, "QRS_MS", min_n=1)
    p_metric = metric(p_durations, "P_DURATION_MS", min_n=1)
    pr_metric = metric(pr_values, "PR_MS", min_n=1)
    qt_metric = metric(qt_values, "QT_MS", min_n=1)
    stj_metric = metric(st_j, "ST_J_MV", min_n=1)
    st40_metric = metric(st_40, "ST_J40_MV", min_n=1)
    st60_metric = metric(st_60, "ST_J60_MV", min_n=1)
    st80_metric = metric(st_80, "ST_J80_MV", min_n=1)
    r_metric = metric(r_amp, "R_AMPLITUDE_MV", min_n=1)
    s_metric = metric(s_amp, "S_AMPLITUDE_MV", min_n=1)
    q_metric = metric(q_amp, "Q_AMPLITUDE_MV", min_n=1)
    qdur_metric = metric(q_dur, "Q_DURATION_MS", min_n=1)
    t_metric = metric(t_amp, "T_AMPLITUDE_MV", min_n=1)

    st60_value = _finite_number(st60_metric.get("value"))
    st_direction = "NOT_MEASURABLE"
    if st60_value is not None:
        if st60_value > 0.05:
            st_direction = "ELEVATION"
        elif st60_value < -0.05:
            st_direction = "DEPRESSION"
        else:
            st_direction = "ISOELECTRIC_COMPATIBLE"

    r_value = _finite_number(r_metric.get("value"))
    s_value = _finite_number(s_metric.get("value"))
    rs_ratio = None
    if r_value is not None and s_value is not None and abs(s_value) >= 0.005:
        rs_ratio = float(abs(r_value) / abs(s_value))

    t_value = _finite_number(t_metric.get("value"))
    t_polarity = (
        "POSITIVE" if t_value is not None and t_value > 0.02
        else "NEGATIVE" if t_value is not None and t_value < -0.02
        else "ISOPHASIC" if t_value is not None
        else "NOT_MEASURABLE"
    )

    q_value = _finite_number(q_metric.get("value"))
    qdur_value = _finite_number(qdur_metric.get("value"))

    return {
        "evaluable": True,
        "lead": lead.name,
        "duration_s": duration_s,
        "lead_confidence": float(lead.confidence),
        "r_count": int(r.size),
        "r_peaks_samples": [int(v + a0) for v in r.tolist()],
        "rr_ms": [round(float(v), 3) for v in rr_ms.tolist()],
        "rr": rr_stats,
        "heart_rate_bpm": (
            float(60000.0 / rr_stats["median"])
            if rr_stats["median"] not in (None, 0) else None
        ),
        "qrs_ms": qrs_metric,
        "p_duration_ms": p_metric,
        "pr_ms": pr_metric,
        "qt_ms": qt_metric,
        "st": {
            "baseline_sources": sorted(set(baseline_sources)),
            "j_mv": stj_metric,
            "j40_mv": st40_metric,
            "j60_mv": st60_metric,
            "j80_mv": st80_metric,
            "j60_mm": (
                float(st60_value * gain_mm_per_mv)
                if st60_value is not None else None
            ),
            "direction_j60": st_direction,
        },
        "amplitudes": {
            "r_mv": r_metric,
            "s_mv": s_metric,
            "rs_ratio": rs_ratio,
            "q_mv": q_metric,
            "q_duration_ms": qdur_metric,
            "t_mv": t_metric,
            "t_polarity": t_polarity,
            "q_numeric_flag": (
                bool(q_value is not None and q_value <= -0.10 and qdur_value is not None and qdur_value >= 30.0)
                if q_value is not None and qdur_value is not None
                else None
            ),
        },
        "beats": beat_records,
    }


def _weighted_interval_consensus(
    per_lead: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    values: list[float] = []
    weights: list[float] = []
    sources: list[dict[str, Any]] = []
    for lead, item in per_lead.items():
        metric = item.get(key) or {}
        value = _finite_number(metric.get("value"))
        confidence = _finite_number(metric.get("confidence"))
        if value is None or confidence is None or confidence <= 0:
            continue
        values.append(value)
        weights.append(confidence)
        sources.append({
            "lead": lead,
            "value": value,
            "confidence": confidence,
            "n": int(metric.get("n") or 0),
        })

    if not values:
        return {
            "value": None,
            "confidence": 0.0,
            "reportable": False,
            "reason": "NO_RELIABLE_LEAD_MEASUREMENTS",
            "sources": [],
        }

    order = np.argsort(np.asarray(values))
    v = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cdf = np.cumsum(w) / max(float(np.sum(w)), 1e-12)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    value = float(v[min(idx, len(v) - 1)])

    spread = float(np.median(np.abs(np.asarray(values) - value))) if len(values) >= 2 else 0.0
    support = min(1.0, len(values) / 4.0)
    agreement = float(np.exp(-spread / max(abs(value) * 0.20, 15.0)))
    confidence = float(np.clip(np.median(weights) * (0.55 + 0.45 * support) * agreement, 0.0, 1.0))

    return {
        "value": value,
        "confidence": confidence,
        "reportable": bool(confidence >= 0.35),
        "source_n": int(len(values)),
        "mad_between_leads": spread,
        "sources": sources,
        "reason": None if confidence >= 0.35 else "LOW_INTERLEAD_CONFIDENCE",
    }


def _rhythm_metrics(per_lead: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    preference = {"II": 0.15, "V1": 0.06, "I": 0.04, "V5": 0.03}
    for lead, item in per_lead.items():
        if not item.get("evaluable"):
            continue
        rr = np.asarray(item.get("rr_ms") or [], dtype=float)
        duration = float(item.get("duration_s") or 0.0)
        if rr.size < 2:
            continue
        conf = float(item.get("lead_confidence") or 0.0)
        score = conf + min(0.30, duration / 30.0) + preference.get(lead, 0.0)
        candidates.append((score, lead, item))

    if not candidates:
        return {
            "evaluable": False,
            "reason": "NO_LEAD_WITH_RR_SERIES",
            "confidence": 0.0,
        }

    _, lead, item = max(candidates, key=lambda x: x[0])
    rr = np.asarray(item.get("rr_ms") or [], dtype=float)
    mean = float(np.mean(rr))
    sd = float(np.std(rr, ddof=1)) if rr.size >= 2 else 0.0
    cv = float(sd / mean) if mean > 0 else None
    med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - med)))
    rmssd = (
        float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        if rr.size >= 2 else None
    )
    robust_scale = 1.4826 * mad
    robust_cv = float(robust_scale / med) if med > 0 else None
    hr = float(60000.0 / med) if med > 0 else None

    duration = float(item.get("duration_s") or 0.0)
    n_rr = int(rr.size)
    confidence = float(np.clip(
        float(item.get("lead_confidence") or 0.0)
        * min(1.0, max(0.45, duration / 8.0))
        * min(1.0, max(0.55, n_rr / 8.0)),
        0.0,
        1.0,
    ))
    regular = bool(cv is not None and cv <= 0.10)
    if cv is None:
        category = "NOT_MEASURABLE"
    elif cv <= 0.10:
        category = "REGULAR"
    elif cv >= 0.12:
        category = "IRREGULAR"
    else:
        category = "BORDERLINE"

    return {
        "evaluable": True,
        "lead": lead,
        "duration_s": duration,
        "r_count": int(item.get("r_count") or 0),
        "rr_ms": [round(float(v), 3) for v in rr.tolist()],
        "rr_mean_ms": mean,
        "rr_median_ms": med,
        "rr_sd_ms": sd,
        "rr_cv": cv,
        "rr_mad_ms": mad,
        "rr_robust_cv": robust_cv,
        "rr_rmssd_ms": rmssd,
        "heart_rate_bpm": hr,
        "regular": regular,
        "regularity": category,
        "confidence": confidence,
        "source": "DIGITAL_SIGNAL_RR",
    }


def _axis_from_digital(ecg: DigitalECG, per_lead: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def net_area(lead_name: str) -> float | None:
        lead = ecg.leads.get(lead_name)
        item = per_lead.get(lead_name) or {}
        if lead is None or not item.get("evaluable"):
            return None
        x = np.asarray(lead.signal_mv, dtype=float)
        beats = item.get("beats") or []
        vals: list[float] = []
        for beat in beats:
            on = beat.get("qrs_on_sample")
            off = beat.get("qrs_off_sample")
            base = _finite_number(beat.get("baseline_mv"))
            if on is None or off is None or base is None:
                continue
            on, off = int(on), int(off)
            if not (0 <= on < off <= len(x)):
                continue
            seg = x[on:off] - base
            if np.isfinite(seg).sum() >= 3:
                vals.append(float(np.trapezoid(np.nan_to_num(seg), dx=1.0 / lead.fs)))
        return float(np.median(vals)) if vals else None

    lead_i = net_area("I")
    lead_avf = net_area("aVF")
    if lead_i is None or lead_avf is None:
        return {
            "evaluable": False,
            "degrees": None,
            "confidence": 0.0,
            "reason": "I_OR_AVF_QRS_AREA_NOT_MEASURABLE",
        }

    deg = float(np.degrees(np.arctan2(lead_avf, lead_i)))
    if deg > 180:
        deg -= 360
    confidence = float(np.clip(
        min(
            ecg.leads.get("I").confidence if ecg.leads.get("I") else 0.0,
            ecg.leads.get("aVF").confidence if ecg.leads.get("aVF") else 0.0,
        ),
        0.0,
        1.0,
    ))
    return {
        "evaluable": True,
        "degrees": deg,
        "qrs_net_area_I": lead_i,
        "qrs_net_area_aVF": lead_avf,
        "confidence": confidence,
        "source": "DIGITAL_QRS_NET_AREA",
    }


def _morphology_summary(per_lead: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r_values: dict[str, float | None] = {}
    s_values: dict[str, float | None] = {}
    rs_values: dict[str, float | None] = {}
    q_values: dict[str, dict[str, Any]] = {}
    t_values: dict[str, dict[str, Any]] = {}

    for lead in LEADS:
        item = per_lead.get(lead) or {}
        amps = item.get("amplitudes") or {}
        r_values[lead] = _finite_number((amps.get("r_mv") or {}).get("value"))
        s_values[lead] = _finite_number((amps.get("s_mv") or {}).get("value"))
        rs_values[lead] = _finite_number(amps.get("rs_ratio"))
        q_values[lead] = {
            "amplitude_mv": _finite_number((amps.get("q_mv") or {}).get("value")),
            "duration_ms": _finite_number((amps.get("q_duration_ms") or {}).get("value")),
            "numeric_flag": amps.get("q_numeric_flag"),
            "confidence": _finite_number((amps.get("q_mv") or {}).get("confidence")),
        }
        t_values[lead] = {
            "amplitude_mv": _finite_number((amps.get("t_mv") or {}).get("value")),
            "polarity": amps.get("t_polarity"),
            "confidence": _finite_number((amps.get("t_mv") or {}).get("confidence")),
        }

    precordial = ["V1", "V2", "V3", "V4", "V5", "V6"]
    rv = np.asarray([
        np.nan if r_values[x] is None else float(r_values[x])
        for x in precordial
    ], dtype=float)
    finite = np.isfinite(rv)
    progression_slope = None
    progression_corr = None
    if finite.sum() >= 3:
        xs = np.arange(1, 7, dtype=float)[finite]
        ys = rv[finite]
        progression_slope = float(np.polyfit(xs, ys, 1)[0])
        progression_corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 3 else None

    transition_lead = None
    for lead in precordial:
        ratio = rs_values.get(lead)
        if ratio is not None and ratio >= 1.0:
            transition_lead = lead
            break

    sokolow = None
    if s_values.get("V1") is not None:
        rv5 = r_values.get("V5")
        rv6 = r_values.get("V6")
        if rv5 is not None or rv6 is not None:
            sokolow = abs(float(s_values["V1"])) + max(
                float(rv5 or 0.0),
                float(rv6 or 0.0),
            )

    return {
        "r_amplitude_mv": r_values,
        "s_amplitude_mv": s_values,
        "r_s_ratio": rs_values,
        "q_waves": q_values,
        "t_waves": t_values,
        "r_progression": {
            "slope_mv_per_lead": progression_slope,
            "correlation": progression_corr,
            "transition_lead": transition_lead,
        },
        "voltages": {
            "sokolow_lyon_sv1_plus_max_rv5_rv6_mv": sokolow,
        },
    }


def measure_digital_ecg(ecg: DigitalECG) -> dict[str, Any]:
    """Run the primary numerical ECG measurement engine."""
    per_lead: dict[str, dict[str, Any]] = {}
    gain = float(ecg.calibration.gain_mm_per_mv)

    for lead_name in LEADS:
        lead = ecg.leads.get(lead_name)
        if lead is None:
            per_lead[lead_name] = {
                "evaluable": False,
                "reason": "LEAD_NOT_RECOVERED",
                "confidence": 0.0,
            }
            continue
        per_lead[lead_name] = _lead_measurements(lead, gain)

    rhythm = _rhythm_metrics(per_lead)
    qrs = _weighted_interval_consensus(per_lead, "qrs_ms")
    p_duration = _weighted_interval_consensus(per_lead, "p_duration_ms")
    pr = _weighted_interval_consensus(per_lead, "pr_ms")
    qt = _weighted_interval_consensus(per_lead, "qt_ms")

    rr_s = (
        float(rhythm["rr_median_ms"]) / 1000.0
        if rhythm.get("rr_median_ms") not in (None, 0) else None
    )
    qt_value = _finite_number(qt.get("value"))
    qtc_bazett = (
        float(qt_value / math.sqrt(rr_s))
        if qt_value is not None and rr_s is not None and rr_s > 0
        else None
    )
    qtc_conf = float(min(
        float(qt.get("confidence") or 0.0),
        float(rhythm.get("confidence") or 0.0),
    )) if qtc_bazett is not None else 0.0

    axis = _axis_from_digital(ecg, per_lead)
    morphology = _morphology_summary(per_lead)

    st_per_lead: dict[str, Any] = {}
    for lead in LEADS:
        item = per_lead.get(lead) or {}
        st = item.get("st") or {}
        j60 = st.get("j60_mv") or {}
        st_per_lead[lead] = {
            "evaluable": bool(j60.get("reportable")),
            "j_mv": (st.get("j_mv") or {}).get("value"),
            "j40_mv": (st.get("j40_mv") or {}).get("value"),
            "j60_mv": j60.get("value"),
            "j80_mv": (st.get("j80_mv") or {}).get("value"),
            "j60_mm": st.get("j60_mm"),
            "direction": st.get("direction_j60") or "NOT_MEASURABLE",
            "confidence": float(j60.get("confidence") or 0.0),
            "baseline_sources": st.get("baseline_sources") or [],
            "reason": j60.get("reason"),
        }

    return {
        "version": "MEDCALC_DIGITAL_MEASUREMENT_ENGINE_V1",
        "source": "CALIBRATED_DIGITAL_ECG",
        "sampling_rate_hz": int(ecg.fs),
        "calibration": ecg.calibration.to_dict(),
        "rhythm": rhythm,
        "intervals": {
            "p_duration_ms": p_duration,
            "pr_ms": pr,
            "qrs_ms": qrs,
            "qt_ms": qt,
            "qtc_bazett_ms": {
                "value": qtc_bazett,
                "confidence": qtc_conf,
                "reportable": qtc_bazett is not None and qtc_conf >= 0.35,
                "reason": None if qtc_bazett is not None else "QT_OR_RR_NOT_MEASURABLE",
            },
        },
        "axis": axis,
        "st_by_lead": st_per_lead,
        "morphology": morphology,
        "per_lead": per_lead,
    }
