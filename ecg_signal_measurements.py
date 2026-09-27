from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MEASUREMENT_VERSION = "MEDCALC_DIGITAL_MEASUREMENTS_V2"


def _as_signal(item: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
        dtype=float,
    )


def _as_quality(item: Dict[str, Any], n: int) -> np.ndarray:
    q = np.asarray(item.get("quality_mask", []), dtype=np.uint8)
    if q.size != n:
        out = np.zeros(n, dtype=np.uint8)
        out[: min(n, q.size)] = q[: min(n, q.size)]
        q = out
    return q


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _longest_finite_span(x: np.ndarray) -> tuple[int, int] | None:
    runs = _finite_runs(np.isfinite(x))
    return max(runs, key=lambda ab: ab[1] - ab[0]) if runs else None


def _arr(waves: Dict[str, Any], key: str) -> np.ndarray:
    vals: list[int] = []
    for v in waves.get(key, []) or []:
        try:
            z = float(v)
        except Exception:
            continue
        if math.isfinite(z):
            vals.append(int(round(z)))
    return np.asarray(vals, dtype=int)


def _nearest_before(values: np.ndarray, target: int, low: int, high: int) -> int | None:
    if values.size == 0:
        return None
    c = values[(values <= target - low) & (values >= target - high)]
    return int(c[-1]) if c.size else None


def _nearest_after(values: np.ndarray, target: int, low: int, high: int) -> int | None:
    if values.size == 0:
        return None
    c = values[(values >= target + low) & (values <= target + high)]
    return int(c[0]) if c.size else None


def _window_quality(q: np.ndarray, a: int, b: int) -> float:
    a = max(0, int(a))
    b = min(len(q), int(b))
    if b <= a:
        return 0.0
    z = q[a:b]
    return float(np.mean(np.where(z == 2, 1.0, np.where(z == 1, 0.55, 0.0))))


def _metric(
    value: float | None,
    *,
    unit: str,
    confidence: float,
    reason: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if value is None or not math.isfinite(float(value)):
        out = {
            "value": None,
            "unit": unit,
            "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
            "status": "NOT_MEASURABLE",
            "reason": str(reason or "INSUFFICIENT_DIGITAL_SIGNAL"),
        }
    else:
        clipped_conf = float(max(0.0, min(1.0, confidence)))
        out = {
            "value": round(float(value), 6),
            "unit": unit,
            "confidence": round(clipped_conf, 6),
            "status": "MEASURED" if clipped_conf >= 0.45 else "MEASURED_LOW_CONFIDENCE",
            "reason": None,
        }
    if extra:
        out.update(extra)
    return out


def _robust_aggregate(values: Iterable[float]) -> tuple[float | None, float, int]:
    z = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if z.size == 0:
        return None, 0.0, 0
    med = float(np.median(z))
    if z.size == 1:
        return med, 0.55, 1
    mad = float(np.median(np.abs(z - med)))
    scale = max(abs(med), 20.0)
    consistency = float(np.clip(1.0 - (1.4826 * mad / scale), 0.25, 1.0))
    n_factor = float(np.clip(z.size / 5.0, 0.45, 1.0))
    return med, consistency * n_factor, int(z.size)


def _delineate(segment_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    import neurokit2 as nk

    x = np.asarray(segment_mv, dtype=float)
    clean = nk.ecg_clean(x, sampling_rate=int(fs), method="neurokit")
    _, peak_info = nk.ecg_peaks(clean, sampling_rate=int(fs), method="neurokit")
    r = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=int)
    waves: Dict[str, Any] = {}
    if len(r) >= 3:
        try:
            _, waves = nk.ecg_delineate(
                clean,
                rpeaks=r,
                sampling_rate=int(fs),
                method="dwt",
                show=False,
                show_type="all",
            )
        except Exception:
            waves = {}
    return {"clean": np.asarray(clean, dtype=float), "r": r, "waves": waves}


def _baseline_for_beat(
    x: np.ndarray,
    fs: int,
    *,
    qrs_on: int,
    p_on: int | None,
    p_off: int | None,
    prev_t_off: int | None,
) -> tuple[float | None, str | None, tuple[int, int] | None]:
    pad = max(1, int(round(0.010 * fs)))

    if p_off is not None and p_off + pad < qrs_on - pad:
        a, b = p_off + pad, qrs_on - pad
        if b - a >= max(3, int(round(0.020 * fs))):
            return float(np.nanmedian(x[a:b])), "PR", (a, b)

    if prev_t_off is not None and p_on is not None:
        a, b = prev_t_off + 2 * pad, p_on - 2 * pad
        if b - a >= max(3, int(round(0.040 * fs))):
            return float(np.nanmedian(x[a:b])), "TP", (a, b)

    a = max(0, qrs_on - int(round(0.20 * fs)))
    b = max(a, qrs_on - int(round(0.12 * fs)))
    if b - a >= 3:
        return float(np.nanmedian(x[a:b])), "PRE_QRS_FALLBACK", (a, b)
    return None, None, None


def _q_duration_ms(
    x: np.ndarray,
    baseline: float,
    qrs_on: int,
    r_peak: int,
    fs: int,
) -> float | None:
    if r_peak <= qrs_on + 1:
        return None
    seg = x[qrs_on:r_peak + 1] - baseline
    if not np.isfinite(seg).any() or float(np.nanmin(seg)) >= -0.02:
        return None
    negative = np.flatnonzero(np.isfinite(seg) & (seg < -0.01))
    if negative.size == 0:
        return None
    start = int(negative[0])
    after = np.flatnonzero(np.isfinite(seg[start:]) & (seg[start:] >= 0.0))
    end = start + int(after[0]) if after.size else int(negative[-1] + 1)
    if end <= start:
        return None
    return float((end - start) * 1000.0 / fs)



def _smooth_finite_signal(x: np.ndarray, fs: int, window_ms: float = 6.0) -> np.ndarray:
    """Short moving-average smoothing used only for local fiducial fallback."""
    y = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return y
    if not finite.all():
        idx = np.arange(len(y), dtype=float)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    n = max(1, int(round(float(window_ms) * float(fs) / 1000.0)))
    if n <= 1:
        return y
    kernel = np.ones(n, dtype=float) / float(n)
    return np.convolve(y, kernel, mode="same")


def _fallback_qrs_bounds(
    x: np.ndarray,
    rp: int,
    fs: int,
) -> tuple[int | None, int | None, float]:
    """Estimate QRS onset/offset from local slope energy around a known R peak.

    This is a signal-domain fallback used when DWT delineation cannot supply
    usable QRS fiducials. It never uses the source raster. The returned score is
    deliberately capped below primary DWT confidence.
    """
    y = _smooth_finite_signal(x, fs, window_ms=4.0)
    if rp <= 1 or rp >= len(y) - 2 or not np.isfinite(y[rp]):
        return None, None, 0.0

    lo = max(1, int(rp) - int(round(0.14 * fs)))
    hi = min(len(y) - 1, int(rp) + int(round(0.16 * fs)))
    if hi - lo < int(round(0.06 * fs)):
        return None, None, 0.0

    deriv = np.abs(np.gradient(y)) * float(fs)
    local = deriv[lo:hi]
    local = local[np.isfinite(local)]
    if local.size < 10:
        return None, None, 0.0

    peak_slope = float(np.nanmax(local))
    noise_slope = float(np.nanmedian(local))
    threshold = max(0.05, 2.5 * noise_slope, 0.08 * peak_slope)
    stable = max(2, int(round(0.012 * fs)))

    onset = None
    left_stop = max(lo + stable, int(rp) - int(round(0.025 * fs)))
    for idx in range(left_stop, lo + stable - 1, -1):
        z = deriv[idx - stable:idx]
        if z.size and float(np.nanmean(z)) <= threshold:
            onset = int(idx)
            break

    offset = None
    right_start = min(hi - stable - 1, int(rp) + int(round(0.025 * fs)))
    for idx in range(right_start, hi - stable):
        z = deriv[idx:idx + stable]
        if z.size and float(np.nanmean(z)) <= threshold:
            offset = int(idx)
            break

    if onset is None or offset is None or offset <= onset:
        return None, None, 0.0

    width_ms = (offset - onset) * 1000.0 / float(fs)
    if width_ms < 40.0 or width_ms > 220.0:
        return None, None, 0.0

    contrast = peak_slope / max(threshold, 1e-6)
    score = float(np.clip(0.45 + 0.08 * (contrast - 1.0), 0.45, 0.78))
    return onset, offset, score


def _fallback_t_fiducials(
    x: np.ndarray,
    *,
    qrs_off: int,
    next_r: int | None,
    baseline: float,
    fs: int,
) -> tuple[int | None, int | None, float]:
    """Conservative T peak/offset fallback from the calibrated digital signal."""
    y = _smooth_finite_signal(x, fs, window_ms=12.0)
    start = int(qrs_off) + int(round(0.045 * fs))
    end = min(
        len(y) - 1,
        int(qrs_off) + int(round(0.32 * fs)),
        (int(next_r) - int(round(0.055 * fs))) if next_r is not None else len(y) - 1,
    )
    if end - start < int(round(0.06 * fs)):
        return None, None, 0.0

    seg = y[start:end] - float(baseline)
    finite = np.isfinite(seg)
    if finite.sum() < 8:
        return None, None, 0.0

    abs_seg = np.abs(seg)
    rel_peak = int(np.nanargmax(abs_seg))
    amp = float(abs_seg[rel_peak])
    noise = float(np.nanmedian(np.abs(seg - np.nanmedian(seg))))
    if amp < max(0.025, 3.0 * max(noise, 0.003)):
        return None, None, 0.0

    t_peak = start + rel_peak
    return_thr = max(0.012, 0.15 * amp, 2.0 * max(noise, 0.003))
    stable = max(3, int(round(0.018 * fs)))
    t_off = None
    for idx in range(t_peak, end - stable):
        z = np.abs(y[idx:idx + stable] - float(baseline))
        if z.size and float(np.nanmean(z)) <= return_thr:
            t_off = int(idx)
            break
    if t_off is None or t_off <= t_peak:
        return t_peak, None, 0.42

    score = float(np.clip(0.42 + min(amp / 0.30, 1.0) * 0.18, 0.42, 0.60))
    return t_peak, t_off, score


def _analyze_lead(lead: str, item: Dict[str, Any]) -> Dict[str, Any]:
    fs = int(item.get("fs") or 500)
    x_full = _as_signal(item)
    q_full = _as_quality(item, len(x_full))
    lead_conf = float(item.get("confidence") or 0.0)

    span = _longest_finite_span(x_full)
    if span is None or span[1] - span[0] < int(1.5 * fs):
        return {
            "lead": lead,
            "evaluable": False,
            "confidence": round(lead_conf, 6),
            "reason": "CONTIGUOUS_DIGITAL_SIGNAL_LT_1_5S",
        }

    a0, b0 = span
    x = x_full[a0:b0]
    q = q_full[a0:b0]
    try:
        nk = _delineate(x, fs)
    except Exception as exc:
        return {
            "lead": lead,
            "evaluable": False,
            "confidence": round(lead_conf, 6),
            "reason": f"FIDUCIAL_DETECTION_FAILED: {exc}",
        }

    r = np.asarray(nk.get("r", []), dtype=int)
    waves = nk.get("waves", {}) or {}
    if r.size < 3:
        return {
            "lead": lead,
            "evaluable": False,
            "confidence": round(lead_conf, 6),
            "reason": "LT_3_QRS",
            "r_count": int(r.size),
        }

    p_on_all = _arr(waves, "ECG_P_Onsets")
    p_off_all = _arr(waves, "ECG_P_Offsets")
    p_peak_all = _arr(waves, "ECG_P_Peaks")
    qrs_on_all = _arr(waves, "ECG_R_Onsets")
    qrs_off_all = _arr(waves, "ECG_R_Offsets")
    t_peak_all = _arr(waves, "ECG_T_Peaks")
    t_off_all = _arr(waves, "ECG_T_Offsets")

    beats: list[Dict[str, Any]] = []
    prev_t_off: int | None = None
    for rp in r:
        rp = int(rp)
        q_on = _nearest_before(qrs_on_all, rp, 0, int(round(0.16 * fs)))
        q_off = _nearest_after(qrs_off_all, rp, 0, int(round(0.20 * fs)))
        fiducial_source = "NEUROKIT_DWT"
        fiducial_confidence = 1.0
        if q_on is None or q_off is None or q_off <= q_on:
            q_on, q_off, fiducial_confidence = _fallback_qrs_bounds(x, rp, fs)
            fiducial_source = "DIGITAL_SLOPE_FALLBACK"
        if q_on is None or q_off is None or q_off <= q_on:
            continue

        p_on = _nearest_before(p_on_all, q_on, int(round(0.03 * fs)), int(round(0.40 * fs)))
        p_off = _nearest_before(p_off_all, q_on, int(round(0.01 * fs)), int(round(0.30 * fs)))
        p_peak = _nearest_before(p_peak_all, q_on, int(round(0.03 * fs)), int(round(0.35 * fs)))
        t_peak = _nearest_after(t_peak_all, q_off, int(round(0.02 * fs)), int(round(0.55 * fs)))
        t_off = _nearest_after(t_off_all, q_off, int(round(0.08 * fs)), int(round(0.80 * fs)))

        baseline, baseline_source, baseline_window = _baseline_for_beat(
            x,
            fs,
            qrs_on=q_on,
            p_on=p_on,
            p_off=p_off,
            prev_t_off=prev_t_off,
        )
        if baseline is None or baseline_window is None:
            if t_off is not None:
                prev_t_off = t_off
            continue

        t_fiducial_confidence = 1.0 if t_peak is not None else 0.0
        if t_peak is None or t_off is None:
            next_r_candidates = r[r > rp]
            next_r = int(next_r_candidates[0]) if next_r_candidates.size else None
            fb_t_peak, fb_t_off, fb_t_conf = _fallback_t_fiducials(
                x,
                qrs_off=q_off,
                next_r=next_r,
                baseline=float(baseline),
                fs=fs,
            )
            if t_peak is None and fb_t_peak is not None:
                t_peak = fb_t_peak
            if t_off is None and fb_t_off is not None:
                t_off = fb_t_off
            if fb_t_conf > 0:
                t_fiducial_confidence = fb_t_conf

        local_a = max(0, q_on - int(round(0.35 * fs)))
        local_b = min(len(x), (t_off or q_off) + int(round(0.04 * fs)))
        beat_quality = _window_quality(q, local_a, local_b)
        if beat_quality < 0.45:
            if t_off is not None:
                prev_t_off = t_off
            continue

        qrs_ms = (q_off - q_on) * 1000.0 / fs
        p_duration_ms = (
            (p_off - p_on) * 1000.0 / fs
            if p_on is not None and p_off is not None and p_off > p_on
            else None
        )
        pr_ms = (
            (q_on - p_on) * 1000.0 / fs
            if p_on is not None and q_on > p_on
            else None
        )
        qt_ms = (
            (t_off - q_on) * 1000.0 / fs
            if t_off is not None and t_off > q_on
            else None
        )

        def amp_at(idx: int | None) -> float | None:
            if idx is None or idx < 0 or idx >= len(x) or not np.isfinite(x[idx]):
                return None
            return float(x[idx] - baseline)

        st: dict[str, float | None] = {}
        for delay_ms in (0, 40, 60, 80):
            idx = q_off + int(round(delay_ms * fs / 1000.0))
            st[str(delay_ms)] = amp_at(idx)

        q_seg = x[q_on:rp + 1] - baseline
        s_seg = x[rp:q_off + 1] - baseline
        q_amp = float(np.nanmin(q_seg)) if np.isfinite(q_seg).any() else None
        s_amp = float(np.nanmin(s_seg)) if np.isfinite(s_seg).any() else None
        r_amp = amp_at(rp)
        t_amp = amp_at(t_peak)
        p_amp = amp_at(p_peak)
        rs_ratio = (
            abs(float(r_amp)) / abs(float(s_amp))
            if r_amp is not None and s_amp is not None and abs(float(s_amp)) >= 0.02
            else None
        )

        qrs_area = None
        qrs_seg = x[q_on:q_off + 1] - baseline
        if np.isfinite(qrs_seg).sum() >= 3:
            qrs_area = float(np.trapezoid(np.nan_to_num(qrs_seg), dx=1000.0 / fs))

        baseline_confidence = (
            0.98
            if baseline_source in {"PR", "TP"}
            else 0.65
            if baseline_source == "PRE_QRS_FALLBACK"
            else 0.50
        )

        beats.append({
            "r_sample": int(rp + a0),
            "qrs_onset_sample": int(q_on + a0),
            "qrs_offset_sample": int(q_off + a0),
            "p_onset_sample": int(p_on + a0) if p_on is not None else None,
            "p_offset_sample": int(p_off + a0) if p_off is not None else None,
            "t_peak_sample": int(t_peak + a0) if t_peak is not None else None,
            "t_offset_sample": int(t_off + a0) if t_off is not None else None,
            "baseline_mv": float(baseline),
            "baseline_source": baseline_source,
            "baseline_confidence": float(baseline_confidence),
            "beat_quality": float(beat_quality),
            "fiducial_source": fiducial_source,
            "fiducial_confidence": float(fiducial_confidence),
            "t_fiducial_confidence": float(t_fiducial_confidence),
            "qrs_ms": float(qrs_ms),
            "p_duration_ms": p_duration_ms,
            "pr_ms": pr_ms,
            "qt_ms": qt_ms,
            "j_mv": st["0"],
            "st_j40_mv": st["40"],
            "st_j60_mv": st["60"],
            "st_j80_mv": st["80"],
            "r_amp_mv": r_amp,
            "s_amp_mv": s_amp,
            "rs_ratio": rs_ratio,
            "q_amp_mv": q_amp,
            "q_duration_ms": _q_duration_ms(x, baseline, q_on, rp, fs),
            "t_amp_mv": t_amp,
            "p_amp_mv": p_amp,
            "qrs_net_area_mv_ms": qrs_area,
        })
        if t_off is not None:
            prev_t_off = t_off

    if not beats:
        return {
            "lead": lead,
            "evaluable": False,
            "confidence": round(lead_conf, 6),
            "reason": "NO_HIGH_QUALITY_FIDUCIAL_BEATS",
            "r_count": int(r.size),
        }

    rr_ms = np.diff(r) * 1000.0 / fs
    rr_ms = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]
    rr_med = float(np.median(rr_ms)) if rr_ms.size else None
    rr_mean = float(np.mean(rr_ms)) if rr_ms.size else None
    rr_sd = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else 0.0 if rr_ms.size else None
    rr_cv = (
        float(rr_sd / rr_mean)
        if rr_mean is not None and rr_sd is not None and rr_mean > 0
        else None
    )
    rr_mad = (
        float(np.median(np.abs(rr_ms - np.median(rr_ms))))
        if rr_ms.size else None
    )
    rr_delta = np.diff(rr_ms)
    rr_rmssd = (
        float(np.sqrt(np.mean(rr_delta ** 2)))
        if rr_delta.size else None
    )
    rr_pnn50 = (
        float(np.mean(np.abs(rr_delta) > 50.0))
        if rr_delta.size else None
    )

    result: Dict[str, Any] = {
        "lead": lead,
        "evaluable": True,
        "fs": fs,
        "duration_s": round(float(len(x) / fs), 6),
        "confidence": round(lead_conf, 6),
        "r_count": int(r.size),
        "r_peaks_samples": [int(v + a0) for v in r.tolist()],
        "rr_ms": [round(float(v), 3) for v in rr_ms.tolist()],
        "rr_mean_ms": rr_mean,
        "rr_median_ms": rr_med,
        "rr_sd_ms": rr_sd,
        "rr_cv": rr_cv,
        "rr_mad_ms": rr_mad,
        "rr_rmssd_ms": rr_rmssd,
        "rr_pnn50": rr_pnn50,
        "heart_rate_bpm": (60000.0 / rr_med) if rr_med and rr_med > 0 else None,
        "beats_used": int(len(beats)),
        "beats": beats,
    }

    fields = [
        ("qrs_ms", "ms"),
        ("p_duration_ms", "ms"),
        ("pr_ms", "ms"),
        ("qt_ms", "ms"),
        ("j_mv", "mV"),
        ("st_j40_mv", "mV"),
        ("st_j60_mv", "mV"),
        ("st_j80_mv", "mV"),
        ("r_amp_mv", "mV"),
        ("s_amp_mv", "mV"),
        ("rs_ratio", "ratio"),
        ("q_amp_mv", "mV"),
        ("q_duration_ms", "ms"),
        ("t_amp_mv", "mV"),
        ("p_amp_mv", "mV"),
        ("qrs_net_area_mv_ms", "mV*ms"),
    ]

    metrics: Dict[str, Any] = {}
    baseline_dependent = {
        "j_mv",
        "st_j40_mv",
        "st_j60_mv",
        "st_j80_mv",
        "r_amp_mv",
        "s_amp_mv",
        "rs_ratio",
        "q_amp_mv",
        "t_amp_mv",
        "p_amp_mv",
        "qrs_net_area_mv_ms",
    }
    avg_beat_quality = float(np.mean([b["beat_quality"] for b in beats]))
    avg_baseline_confidence = float(
        np.mean([b["baseline_confidence"] for b in beats])
    )
    avg_fiducial_confidence = float(
        np.mean([b.get("fiducial_confidence", 1.0) for b in beats])
    )
    avg_t_fiducial_confidence = float(
        np.mean([b.get("t_fiducial_confidence", 1.0) for b in beats])
    )
    for field, unit in fields:
        vals = [b[field] for b in beats if b.get(field) is not None]
        value, consistency, n = _robust_aggregate(vals)
        conf = lead_conf * consistency * avg_beat_quality * avg_fiducial_confidence
        if field in {"qt_ms", "t_amp_mv"}:
            conf *= avg_t_fiducial_confidence
        if field in baseline_dependent:
            conf *= avg_baseline_confidence
        metrics[field] = _metric(
            value,
            unit=unit,
            confidence=conf,
            reason="FIDUCIAL_NOT_REPRODUCIBLE",
            extra={"beat_n": int(n)},
        )

    gain_mm_per_mv = float(
        (item.get("calibration") or {}).get("gain_mm_per_mv")
        or item.get("gain_mm_per_mv")
        or 10.0
    )
    for st_name in ("j_mv", "st_j40_mv", "st_j60_mv", "st_j80_mv"):
        st_metric = metrics[st_name]
        if st_metric["value"] is None:
            continue
        st_val = float(st_metric["value"])
        st_metric["mm_at_paper_gain"] = round(
            st_val * gain_mm_per_mv,
            6,
        )
        st_metric["paper_gain_mm_per_mv"] = gain_mm_per_mv
        st_metric["mm_confidence"] = st_metric["confidence"]
        if st_val > 0.02:
            st_metric["direction"] = "ELEVATION"
        elif st_val < -0.02:
            st_metric["direction"] = "DEPRESSION"
        else:
            st_metric["direction"] = "ISOELECTRIC_COMPATIBLE"

    t = metrics["t_amp_mv"]
    if t["value"] is not None:
        tv = float(t["value"])
        t["polarity"] = "POSITIVE" if tv > 0.02 else "NEGATIVE" if tv < -0.02 else "FLAT"
        t["inverted"] = bool(tv < -0.05)

    p = metrics["p_amp_mv"]
    if p["value"] is not None:
        pv = float(p["value"])
        p["polarity"] = "POSITIVE" if pv > 0.02 else "NEGATIVE" if pv < -0.02 else "FLAT"

    q = metrics["q_amp_mv"]
    qdur = metrics["q_duration_ms"]
    result["q_wave_candidate"] = bool(
        q.get("value") is not None
        and float(q["value"]) <= -0.10
        and qdur.get("value") is not None
        and float(qdur["value"]) >= 30.0
    )
    result["metrics"] = metrics
    return result


def _consensus_metric(
    per_lead: Dict[str, Dict[str, Any]],
    metric_name: str,
    *,
    unit: str,
    min_confidence: float = 0.45,
    min_sources: int = 2,
) -> Dict[str, Any]:
    all_candidates: list[tuple[str, float, float]] = []
    trusted: list[tuple[str, float, float]] = []

    for lead, item in per_lead.items():
        m = (item.get("metrics") or {}).get(metric_name) or {}
        v = m.get("value")
        c = float(m.get("confidence") or 0.0)
        if v is None or not math.isfinite(float(v)) or c <= 0.0:
            continue
        candidate = (lead, float(v), c)
        all_candidates.append(candidate)
        if c >= float(min_confidence):
            trusted.append(candidate)

    def aggregate(rows: list[tuple[str, float, float]]) -> tuple[float, float, float]:
        z = np.asarray([v for _, v, _ in rows], dtype=float)
        confs = np.asarray([c for _, _, c in rows], dtype=float)
        med = float(np.median(z))
        mad = float(np.median(np.abs(z - med))) if z.size >= 2 else 0.0
        consistency = float(
            np.clip(1.0 - 1.4826 * mad / max(abs(med), 20.0), 0.25, 1.0)
        )
        mean_conf = float(np.mean(confs)) if confs.size else 0.0
        return med, consistency, mean_conf

    if len(trusted) >= int(min_sources):
        med, consistency, mean_conf = aggregate(trusted)
        conf = mean_conf * consistency * float(
            np.clip(len(trusted) / 6.0, 0.55, 1.0)
        )
        return _metric(
            med,
            unit=unit,
            confidence=conf,
            extra={
                "source_leads": [lead for lead, _, _ in trusted],
                "source_n": len(trusted),
                "cross_lead_mad": round(
                    float(np.median(np.abs(
                        np.asarray([v for _, v, _ in trusted], dtype=float) - med
                    ))) if len(trusted) >= 2 else 0.0,
                    6,
                ),
                "consensus_mode": "MULTILEAD_TRUSTED",
            },
        )

    # A numeric measurement that exists should not be converted into
    # NOT_MEASURABLE merely because a second lead missed an arbitrary
    # publication threshold. Publish it with explicit reduced confidence.
    usable = [row for row in all_candidates if row[2] >= 0.20]
    if not usable:
        return _metric(
            None,
            unit=unit,
            confidence=max([c for _, _, c in all_candidates], default=0.0),
            reason="NO_USABLE_DIGITAL_MEASUREMENT",
            extra={"source_leads": [lead for lead, _, _ in all_candidates]},
        )

    if trusted:
        best = max(trusted, key=lambda row: row[2])
        return _metric(
            best[1],
            unit=unit,
            confidence=float(best[2]) * 0.85,
            extra={
                "source_leads": [best[0]],
                "source_n": 1,
                "consensus_mode": "SINGLE_TRUSTED_LEAD_LOW_CONFIDENCE",
            },
        )

    if len(usable) >= 2:
        med, consistency, mean_conf = aggregate(usable)
        conf = mean_conf * consistency * 0.75
        return _metric(
            med,
            unit=unit,
            confidence=conf,
            extra={
                "source_leads": [lead for lead, _, _ in usable],
                "source_n": len(usable),
                "consensus_mode": "MULTILEAD_LOW_CONFIDENCE",
            },
        )

    best = max(usable, key=lambda row: row[2])
    return _metric(
        best[1],
        unit=unit,
        confidence=float(best[2]) * 0.65,
        extra={
            "source_leads": [best[0]],
            "source_n": 1,
            "consensus_mode": "SINGLE_LOW_CONFIDENCE_LEAD",
        },
    )

def _select_rhythm_lead(per_lead: Dict[str, Dict[str, Any]]) -> str | None:
    ii = per_lead.get("II") or {}
    if (
        ii.get("evaluable")
        and float(ii.get("duration_s") or 0.0) >= 5.0
        and int(ii.get("r_count") or 0) >= 5
    ):
        return "II"

    candidates = []
    for lead, item in per_lead.items():
        if not item.get("evaluable"):
            continue
        if int(item.get("r_count") or 0) < 4:
            continue
        score = (
            float(item.get("duration_s") or 0.0)
            * max(float(item.get("confidence") or 0.0), 0.1)
        )
        candidates.append((score, lead))
    return max(candidates)[1] if candidates else None


def analyze_canonical_ecg(canonical_ecg: Dict[str, Any]) -> Dict[str, Any]:
    """Measure ECG intervals/morphology only from the calibrated digital signal."""
    lead_items = canonical_ecg.get("leads") or {}
    calibration = canonical_ecg.get("calibration") or {}
    per_lead = {}
    for lead in LEADS:
        source_item = dict(lead_items.get(lead) or {})
        source_item["calibration"] = calibration
        per_lead[lead] = _analyze_lead(lead, source_item)

    rhythm_lead = _select_rhythm_lead(per_lead)
    rhythm: Dict[str, Any] = {
        "evaluable": False,
        "lead": rhythm_lead,
        "reason": "NO_DIGITAL_LEAD_WITH_SUFFICIENT_RR",
    }
    if rhythm_lead is not None:
        src = per_lead[rhythm_lead]
        rr = np.asarray(src.get("rr_ms") or [], dtype=float)
        rr_mean = src.get("rr_mean_ms")
        rr_cv = src.get("rr_cv")
        rr_mad = src.get("rr_mad_ms")
        rr_median = src.get("rr_median_ms")
        mad_ratio = (
            float(rr_mad) / float(rr_median)
            if rr_mad is not None and rr_median not in (None, 0)
            else None
        )
        regular = bool(
            rr.size >= 4
            and rr_cv is not None
            and mad_ratio is not None
            and float(rr_cv) <= 0.10
            and float(mad_ratio) <= 0.08
        )
        rhythm = {
            "evaluable": True,
            "lead": rhythm_lead,
            "source": "CALIBRATED_DIGITAL_SIGNAL",
            "r_count": int(src.get("r_count") or 0),
            "heart_rate_bpm": src.get("heart_rate_bpm"),
            "rr_ms": src.get("rr_ms"),
            "rr_mean_ms": rr_mean,
            "rr_median_ms": rr_median,
            "rr_sd_ms": src.get("rr_sd_ms"),
            "rr_cv": rr_cv,
            "rr_mad_ms": rr_mad,
            "rr_rmssd_ms": src.get("rr_rmssd_ms"),
            "rr_pnn50": src.get("rr_pnn50"),
            "rr_mad_ratio": mad_ratio,
            "regular": regular,
            "confidence": src.get("confidence"),
            "regularity_rule": "RR_CV<=0.10_AND_RR_MAD_MEDIAN<=0.08",
        }

    global_metrics = {
        "heart_rate_bpm": _metric(
            rhythm.get("heart_rate_bpm") if rhythm.get("evaluable") else None,
            unit="bpm",
            confidence=float(rhythm.get("confidence") or 0.0),
            reason="RHYTHM_NOT_MEASURABLE",
        ),
        "qrs_ms": _consensus_metric(per_lead, "qrs_ms", unit="ms"),
        "p_duration_ms": _consensus_metric(per_lead, "p_duration_ms", unit="ms"),
        "pr_ms": _consensus_metric(per_lead, "pr_ms", unit="ms"),
        "qt_ms": _consensus_metric(per_lead, "qt_ms", unit="ms"),
    }

    rr_s = (
        float(rhythm.get("rr_median_ms")) / 1000.0
        if rhythm.get("rr_median_ms") not in (None, 0)
        else None
    )
    qt = global_metrics["qt_ms"].get("value")
    qtc = (
        float(qt) / math.sqrt(rr_s)
        if qt is not None and rr_s is not None and rr_s > 0
        else None
    )
    qtc_confidence = min(
        float(global_metrics["qt_ms"].get("confidence") or 0.0),
        float(rhythm.get("confidence") or 0.0),
    )
    global_metrics["qtc_bazett_ms"] = _metric(
        qtc,
        unit="ms",
        confidence=qtc_confidence,
        reason="QT_OR_RR_NOT_MEASURABLE",
    )
    qtc_fridericia = (
        float(qt) / (rr_s ** (1.0 / 3.0))
        if qt is not None and rr_s is not None and rr_s > 0
        else None
    )
    global_metrics["qtc_fridericia_ms"] = _metric(
        qtc_fridericia,
        unit="ms",
        confidence=qtc_confidence,
        reason="QT_OR_RR_NOT_MEASURABLE",
    )

    axis = {
        "evaluable": False,
        "degrees": None,
        "confidence": 0.0,
        "reason": "I_OR_AVF_QRS_AREA_NOT_MEASURABLE",
    }
    mi = ((per_lead.get("I") or {}).get("metrics") or {}).get("qrs_net_area_mv_ms") or {}
    mf = ((per_lead.get("aVF") or {}).get("metrics") or {}).get("qrs_net_area_mv_ms") or {}
    if mi.get("value") is not None and mf.get("value") is not None:
        deg = math.degrees(math.atan2(float(mf["value"]), float(mi["value"])))
        axis = {
            "evaluable": True,
            "degrees": round(float(deg), 6),
            "confidence": round(min(float(mi.get("confidence") or 0.0), float(mf.get("confidence") or 0.0)), 6),
            "source": "DIGITAL_QRS_NET_AREA_I_AVF",
        }

    st_by_lead: Dict[str, Any] = {}
    t_by_lead: Dict[str, Any] = {}
    amplitudes: Dict[str, Any] = {}
    for lead in LEADS:
        metrics = (per_lead.get(lead) or {}).get("metrics") or {}
        st = dict(metrics.get("st_j60_mv") or {})
        if st.get("value") is not None:
            gain = float((canonical_ecg.get("calibration") or {}).get("gain_mm_per_mv") or 10.0)
            st["mm_at_paper_gain"] = round(float(st["value"]) * gain, 6)
            st["paper_gain_mm_per_mv"] = gain
        st_by_lead[lead] = st
        t_by_lead[lead] = dict(metrics.get("t_amp_mv") or {})
        amplitudes[lead] = {
            "R": metrics.get("r_amp_mv"),
            "S": metrics.get("s_amp_mv"),
            "R_S_ratio": metrics.get("rs_ratio"),
            "Q": metrics.get("q_amp_mv"),
            "Q_duration": metrics.get("q_duration_ms"),
            "P": metrics.get("p_amp_mv"),
            "T": metrics.get("t_amp_mv"),
        }

    r_progression = []
    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        r = ((per_lead.get(lead) or {}).get("metrics") or {}).get("r_amp_mv") or {}
        r_progression.append({
            "lead": lead,
            "r_mv": r.get("value"),
            "confidence": r.get("confidence"),
        })

    def value(lead: str, metric_name: str) -> float | None:
        m = ((per_lead.get(lead) or {}).get("metrics") or {}).get(metric_name) or {}
        return float(m["value"]) if m.get("value") is not None else None

    s_v1 = value("V1", "s_amp_mv")
    r_v5 = value("V5", "r_amp_mv")
    r_v6 = value("V6", "r_amp_mv")
    r_avl = value("aVL", "r_amp_mv")
    s_v3 = value("V3", "s_amp_mv")
    sokolow = (
        abs(float(s_v1)) + max(float(r_v5 or 0.0), float(r_v6 or 0.0))
        if s_v1 is not None and (r_v5 is not None or r_v6 is not None)
        else None
    )
    cornell = (
        float(r_avl) + abs(float(s_v3))
        if r_avl is not None and s_v3 is not None
        else None
    )

    return {
        "version": MEASUREMENT_VERSION,
        "source": "CALIBRATED_DIGITAL_SIGNAL_ONLY",
        "fs": int(canonical_ecg.get("fs") or 500),
        "calibration": canonical_ecg.get("calibration") or {},
        "rhythm": rhythm,
        "global": global_metrics,
        "axis": axis,
        "leads": per_lead,
        "st_by_lead": st_by_lead,
        "t_by_lead": t_by_lead,
        "amplitudes_by_lead": amplitudes,
        "r_progression": r_progression,
        "voltages": {
            "sokolow_lyon_raw_mv": sokolow,
            "cornell_raw_mv": cornell,
            "note": "Valores de voltaje descriptivos; no se convierten aquí en diagnóstico.",
        },
        "measurement_priority_rule": (
            "RELIABLE_NUMERIC_DIGITAL_MEASUREMENT_OVERRIDES_IMAGE_OR_MODEL_LABEL"
        ),
    }
