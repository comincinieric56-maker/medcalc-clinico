from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np

from ecg_atrial_rhythm import analyze_native_atrial_mechanism
from ecg_wide_complex_tachycardia import analyze_wide_complex_tachycardia
from ecg_measurement_consensus import build_measurement_consensus
from ecg_measurement_failure_audit import audit_measurement_consensus
from ecg_signal_integrity import analyze_signal_integrity
from ecg_feature_graph import build_ecg_feature_graph
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_consistency_engine import evaluate_ecg_consistency
from ecg_reasoner import reason_ecg
from ecg_candidate_detectors import build_high_recall_candidates
from ecg_domain_gating import build_domain_gates
from ecg_evidence_fusion import fuse_candidate_evidence
from ecg_ectopy import analyze_ectopy
from ecg_qrs_morphology import analyze_qrs_morphology
from ecg_av_conduction import analyze_av_conduction
from ecg_preexcitation import analyze_preexcitation
from ecg_rhythm_consensus import build_rhythm_consensus, rr_irregularity_score


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
    """Estimate QRS onset/offset from digital slope activity with hysteresis.

    A QRS may be multiphasic, notched, or have a relatively slow terminal
    component. Stopping at the first quiet slope interval systematically
    truncates those complexes. This fallback therefore:
      1. identifies a high-slope core close to the known R peak;
      2. expands through lower-slope QRS activity;
      3. bridges only very short inactive gaps (<=12 ms);
      4. stops before later atrial/T-wave activity can be joined.

    It operates only on the calibrated digital signal and remains lower
    confidence than a reliable primary delineation.
    """
    y = _smooth_finite_signal(x, fs, window_ms=4.0)
    if rp <= 1 or rp >= len(y) - 2 or not np.isfinite(y[rp]):
        return None, None, 0.0

    lo = max(1, int(rp) - int(round(0.14 * fs)))
    hi = min(len(y) - 1, int(rp) + int(round(0.18 * fs)))
    if hi - lo < int(round(0.06 * fs)):
        return None, None, 0.0

    deriv = np.abs(np.gradient(y)) * float(fs)
    local = np.asarray(deriv[lo:hi], dtype=float)
    finite = local[np.isfinite(local)]
    if finite.size < 10:
        return None, None, 0.0

    peak_slope = float(np.nanmax(finite))
    noise_slope = float(np.nanpercentile(finite, 35.0))
    high_threshold = max(0.05, 2.5 * noise_slope, 0.08 * peak_slope)
    low_threshold = max(0.03, 1.8 * noise_slope, 0.05 * peak_slope)

    high_active = np.isfinite(local) & (local >= high_threshold)
    low_active = np.isfinite(local) & (local >= low_threshold)

    r_local = int(rp - lo)
    high_idx = np.flatnonzero(high_active)
    if high_idx.size == 0:
        return None, None, 0.0

    near = high_idx[
        np.abs(high_idx - r_local) <= int(round(0.060 * fs))
    ]
    if near.size:
        core = int(near[np.argmin(np.abs(near - r_local))])
    else:
        core = int(high_idx[np.argmin(np.abs(high_idx - r_local))])

    # Morphological closing only across short gaps. A 12 ms gap can occur
    # inside a notched/multiphasic QRS; substantially longer gaps should
    # terminate the complex and prevent bridging into flutter/T activity.
    max_gap = max(1, int(round(0.012 * fs)))
    active = low_active.copy()
    inactive_runs = _finite_runs(~active)
    for a, b in inactive_runs:
        gap = int(b - a)
        bounded = a > 0 and b < len(active) and active[a - 1] and active[b]
        if bounded and gap <= max_gap:
            active[a:b] = True

    if not active[core]:
        active[core] = True

    onset_local = core
    while onset_local > 0 and active[onset_local - 1]:
        onset_local -= 1

    offset_local = core
    while offset_local < len(active) - 1 and active[offset_local + 1]:
        offset_local += 1

    pad = max(0, int(round(0.002 * fs)))
    onset_local = max(0, onset_local - pad)
    offset_local = min(len(active) - 1, offset_local + pad)

    onset = int(lo + onset_local)
    offset = int(lo + offset_local)
    if offset <= onset:
        return None, None, 0.0

    width_ms = (offset - onset) * 1000.0 / float(fs)
    if width_ms < 40.0 or width_ms > 220.0:
        return None, None, 0.0

    contrast = peak_slope / max(high_threshold, 1e-6)
    score = float(
        np.clip(0.48 + 0.06 * (contrast - 1.0), 0.48, 0.80)
    )
    return onset, offset, score

def _choose_qrs_bounds(
    *,
    dwt_on: int | None,
    dwt_off: int | None,
    fb_on: int | None,
    fb_off: int | None,
    fb_confidence: float,
    fs: int,
) -> tuple[int | None, int | None, float, str]:
    """Fuse NeuroKit DWT and digital-slope QRS boundaries conservatively."""
    dwt_width_ms = (
        (dwt_off - dwt_on) * 1000.0 / float(fs)
        if dwt_on is not None and dwt_off is not None and dwt_off > dwt_on
        else None
    )
    fb_width_ms = (
        (fb_off - fb_on) * 1000.0 / float(fs)
        if fb_on is not None and fb_off is not None and fb_off > fb_on
        else None
    )

    if dwt_width_ms is None or not 40.0 <= dwt_width_ms <= 220.0:
        return (
            fb_on,
            fb_off,
            float(fb_confidence),
            "DIGITAL_HYSTERESIS_SLOPE_FALLBACK",
        )

    if fb_width_ms is not None:
        disagreement = abs(float(dwt_width_ms) - float(fb_width_ms))

        # Prefer the independent calibrated-signal boundary only when it is
        # morphologically plausible and materially narrower than DWT.  This
        # addresses DWT tails that include low-slope baseline/T activity
        # without applying a benchmark-derived fixed correction.
        fb_materially_narrower = (
            fb_width_ms >= 60.0
            and fb_width_ms <= 160.0
            and dwt_width_ms - fb_width_ms >= 16.0
            and fb_confidence >= 0.55
        )
        if fb_materially_narrower:
            return (
                fb_on,
                fb_off,
                float(fb_confidence),
                "DIGITAL_HYSTERESIS_FUSED_OVER_DWT",
            )

        if (
            disagreement >= 50.0
            and (
                dwt_width_ms >= 160.0
                or dwt_width_ms <= 90.0
                or fb_width_ms >= 110.0
            )
        ):
            return (
                fb_on,
                fb_off,
                float(fb_confidence),
                "DIGITAL_HYSTERESIS_FUSED_OVER_DWT",
            )
        return (
            dwt_on,
            dwt_off,
            1.0,
            "NEUROKIT_DWT_VERIFIED_BY_SLOPE",
        )

    return dwt_on, dwt_off, 1.0, "NEUROKIT_DWT"


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



def _choose_t_fiducials(
    *,
    dwt_peak: int | None,
    dwt_off: int | None,
    fb_peak: int | None,
    fb_off: int | None,
    fb_confidence: float,
    qrs_off: int,
    fs: int,
) -> tuple[int | None, int | None, float, str]:
    """Record DWT/baseline-return candidates; defer replacement to lead consensus."""
    if dwt_off is None or dwt_off <= qrs_off:
        return fb_peak, fb_off, float(fb_confidence), "DIGITAL_BASELINE_RETURN_FALLBACK"
    return dwt_peak, dwt_off, 1.0, "NEUROKIT_DWT_PENDING_BASELINE_CONSENSUS"


def _fallback_repetitive_p_map(
    x: np.ndarray,
    r_peaks: np.ndarray,
    fs: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Recover reproducible P waves in regular non-tachycardic rhythms.

    This fallback is deliberately disabled for tachycardia and irregular RR,
    where repetitive atrial activity or fibrillatory/flutter waves could be
    mistaken for sinus P waves. It aligns pre-R windows across beats and asks
    for a stable deflection, consistent polarity/timing and a relatively quiet
    PR segment before QRS.
    """
    r = np.asarray(r_peaks, dtype=int)
    if r.size < 4:
        return {}, {"status": "SKIPPED", "reason": "LT_4_R_PEAKS"}

    rr_ms = np.diff(r) * 1000.0 / float(fs)
    rr_ms = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]
    if rr_ms.size < 3:
        return {}, {"status": "SKIPPED", "reason": "INSUFFICIENT_RR"}

    rr_med = float(np.median(rr_ms))
    hr = 60000.0 / rr_med if rr_med > 0 else None
    rr_mean = float(np.mean(rr_ms))
    rr_sd = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else 0.0
    rr_cv = rr_sd / rr_mean if rr_mean > 0 else 1.0
    if hr is None or hr < 30.0 or hr > 110.0:
        return {}, {
            "status": "SKIPPED",
            "reason": "HR_OUTSIDE_REGULAR_P_FALLBACK_RANGE",
            "heart_rate_bpm": hr,
            "rr_cv": rr_cv,
        }
    if rr_cv > 0.12:
        return {}, {
            "status": "SKIPPED",
            "reason": "RR_IRREGULAR_FOR_P_FALLBACK",
            "heart_rate_bpm": hr,
            "rr_cv": rr_cv,
        }

    pre = int(round(0.35 * fs))
    tail = int(round(0.05 * fs))
    seg_len = pre - tail
    segments: list[np.ndarray] = []
    segment_r: list[int] = []
    for rp in r:
        a = int(rp) - pre
        b = int(rp) - tail
        if a < 0 or b > len(x) or b <= a:
            continue
        seg = np.asarray(x[a:b], dtype=float)
        if seg.size != seg_len or not np.isfinite(seg).all():
            continue
        edge = max(3, int(round(0.03 * fs)))
        baseline = float(np.median(np.r_[seg[:edge], seg[-edge:]]))
        segments.append(seg - baseline)
        segment_r.append(int(rp))

    if len(segments) < 4:
        return {}, {
            "status": "SKIPPED",
            "reason": "LT_4_COMPLETE_PRE_R_WINDOWS",
            "heart_rate_bpm": hr,
            "rr_cv": rr_cv,
        }

    stack = np.vstack(segments)
    template = np.median(stack, axis=0)
    smooth_n = max(1, int(round(0.010 * fs)))
    smooth = np.convolve(
        template,
        np.ones(smooth_n, dtype=float) / float(smooth_n),
        mode="same",
    )

    edge = max(3, int(round(0.04 * fs)))
    baseline_pool = np.r_[smooth[:edge], smooth[-edge:]]
    baseline = float(np.median(baseline_pool))
    noise = float(
        1.4826 * np.median(np.abs(baseline_pool - baseline))
    )

    # Search 260-80 ms before R. This encompasses ordinary sinus P timing
    # while excluding the immediate QRS foot and most preceding T-wave energy.
    search_a = int(round((0.35 - 0.26) * fs))
    search_b = int(round((0.35 - 0.08) * fs))
    if search_b <= search_a + 3:
        return {}, {"status": "SKIPPED", "reason": "INVALID_P_SEARCH_WINDOW"}

    region = smooth[search_a:search_b] - baseline
    peak_rel = int(np.argmax(np.abs(region)))
    peak_i = search_a + peak_rel
    amp = float(smooth[peak_i] - baseline)
    min_amp = max(0.015, 5.0 * max(noise, 0.001))
    if abs(amp) < min_amp:
        return {}, {
            "status": "SKIPPED",
            "reason": "NO_REPRODUCIBLE_PRE_R_DEFLECTION",
            "template_peak_mv": amp,
            "noise_mv": noise,
        }

    threshold = max(0.006, 0.15 * abs(amp))
    p_on_i = peak_i
    while p_on_i > search_a and abs(smooth[p_on_i] - baseline) > threshold:
        p_on_i -= 1
    p_off_i = peak_i
    while p_off_i < search_b - 1 and abs(smooth[p_off_i] - baseline) > threshold:
        p_off_i += 1

    p_duration_ms = (p_off_i - p_on_i) * 1000.0 / float(fs)
    if not 30.0 <= p_duration_ms <= 160.0:
        return {}, {
            "status": "SKIPPED",
            "reason": "P_TEMPLATE_DURATION_IMPLAUSIBLE",
            "p_duration_ms": p_duration_ms,
        }

    pad = max(2, int(round(0.012 * fs)))
    ta = max(search_a, p_on_i - pad)
    tb = min(search_b, p_off_i + pad)
    ref = template[ta:tb] - float(np.mean(template[ta:tb]))
    ref_norm = float(np.linalg.norm(ref))
    if ref.size < 5 or ref_norm <= 1e-9:
        return {}, {"status": "SKIPPED", "reason": "P_TEMPLATE_ZERO_NORM"}
    ref = ref / ref_norm

    candidates: dict[int, dict[str, Any]] = {}
    correlations: list[float] = []
    quiet_values: list[float] = []
    polarity = 1.0 if amp >= 0 else -1.0

    for rp, seg in zip(segment_r, segments):
        z = seg[ta:tb] - float(np.mean(seg[ta:tb]))
        z_norm = float(np.linalg.norm(z))
        if z.size != ref.size or z_norm <= 1e-9:
            continue
        corr = float(np.dot(z / z_norm, ref))
        beat_amp = float(seg[peak_i])
        if corr < 0.65 or beat_amp * polarity <= max(0.008, 2.0 * noise):
            continue

        global_a = int(rp) - pre
        p_on = global_a + p_on_i
        p_peak = global_a + peak_i
        p_off = global_a + p_off_i

        # Require a relatively quiet segment after P and before the expected
        # QRS foot. Continuous flutter-like activity should fail this gate.
        quiet_a = p_off + max(1, int(round(0.010 * fs)))
        quiet_b = int(rp) - int(round(0.060 * fs))
        if quiet_b <= quiet_a + 2:
            continue
        quiet_seg = np.asarray(x[quiet_a:quiet_b], dtype=float)
        if not np.isfinite(quiet_seg).all():
            continue
        quiet_baseline = float(np.median(quiet_seg))
        quiet_mad = float(np.median(np.abs(quiet_seg - quiet_baseline)))
        quiet_values.append(quiet_mad)
        if quiet_mad > max(0.020, 0.55 * abs(amp)):
            continue

        confidence = float(
            np.clip(
                0.55
                + 0.20 * (corr - 0.65) / 0.35
                + 0.15 * min(abs(beat_amp) / 0.10, 1.0),
                0.55,
                0.90,
            )
        )
        candidates[int(rp)] = {
            "p_on": int(p_on),
            "p_peak": int(p_peak),
            "p_off": int(p_off),
            "confidence": confidence,
            "correlation": corr,
            "amplitude_mv": beat_amp,
        }
        correlations.append(corr)

    required = max(3, int(math.ceil(0.60 * len(segment_r))))
    if len(candidates) < required:
        return {}, {
            "status": "SKIPPED",
            "reason": "P_TEMPLATE_NOT_REPRODUCIBLE_ACROSS_BEATS",
            "candidate_n": len(candidates),
            "required_n": required,
            "heart_rate_bpm": hr,
            "rr_cv": rr_cv,
            "template_peak_mv": amp,
        }

    return candidates, {
        "status": "APPLIED",
        "source": "REGULAR_RHYTHM_PRE_R_TEMPLATE",
        "candidate_n": len(candidates),
        "window_n": len(segment_r),
        "heart_rate_bpm": hr,
        "rr_cv": rr_cv,
        "template_peak_mv": amp,
        "template_p_duration_ms": p_duration_ms,
        "template_peak_before_r_ms": (pre - peak_i) * 1000.0 / float(fs),
        "median_correlation": (
            float(np.median(correlations)) if correlations else None
        ),
        "median_quiet_pr_mad_mv": (
            float(np.median(quiet_values)) if quiet_values else None
        ),
    }


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
    p_fallback_map, p_fallback_summary = _fallback_repetitive_p_map(
        x,
        r,
        fs,
    )
    qrs_on_all = _arr(waves, "ECG_R_Onsets")
    qrs_off_all = _arr(waves, "ECG_R_Offsets")
    t_peak_all = _arr(waves, "ECG_T_Peaks")
    t_off_all = _arr(waves, "ECG_T_Offsets")

    # Cross-beat corroboration for QRS boundaries. The existing per-beat
    # selector remains unchanged. This additional path is enabled only when
    # the calibrated-signal hysteresis boundary is repeatedly narrower than
    # DWT across the same lead with low dispersion and plausible widths.
    qrs_candidate_map: dict[int, dict[str, Any]] = {}
    qrs_narrowing_deltas_ms: list[float] = []
    qrs_comparable_n = 0
    qrs_narrower_n = 0
    for rp0 in r:
        rp0 = int(rp0)
        dwt_on0 = _nearest_before(
            qrs_on_all, rp0, 0, int(round(0.16 * fs))
        )
        dwt_off0 = _nearest_after(
            qrs_off_all, rp0, 0, int(round(0.20 * fs))
        )
        fb_on0, fb_off0, fb_conf0 = _fallback_qrs_bounds(x, rp0, fs)
        dwt_width0 = (
            (dwt_off0 - dwt_on0) * 1000.0 / float(fs)
            if dwt_on0 is not None and dwt_off0 is not None and dwt_off0 > dwt_on0
            else None
        )
        fb_width0 = (
            (fb_off0 - fb_on0) * 1000.0 / float(fs)
            if fb_on0 is not None and fb_off0 is not None and fb_off0 > fb_on0
            else None
        )
        qrs_candidate_map[rp0] = {
            "dwt_on": dwt_on0,
            "dwt_off": dwt_off0,
            "fb_on": fb_on0,
            "fb_off": fb_off0,
            "fb_confidence": float(fb_conf0),
            "dwt_width_ms": dwt_width0,
            "fb_width_ms": fb_width0,
        }
        if (
            dwt_width0 is not None
            and fb_width0 is not None
            and 40.0 <= dwt_width0 <= 220.0
            and 60.0 <= fb_width0 <= 160.0
            and float(fb_conf0) >= 0.55
        ):
            qrs_comparable_n += 1
            delta0 = float(dwt_width0 - fb_width0)
            if delta0 > 0.0:
                qrs_narrower_n += 1
                qrs_narrowing_deltas_ms.append(delta0)

    qrs_delta_median = (
        float(np.median(qrs_narrowing_deltas_ms))
        if qrs_narrowing_deltas_ms else None
    )
    qrs_delta_mad = (
        float(
            np.median(
                np.abs(
                    np.asarray(qrs_narrowing_deltas_ms, dtype=float)
                    - float(qrs_delta_median)
                )
            )
        )
        if qrs_delta_median is not None else None
    )
    qrs_dwt_widths = [
        float(item["dwt_width_ms"])
        for item in qrs_candidate_map.values()
        if item.get("dwt_width_ms") is not None
    ]
    qrs_dwt_width_median = (
        float(np.median(qrs_dwt_widths)) if qrs_dwt_widths else None
    )
    qrs_lead_consensus = (
        qrs_comparable_n >= max(3, int(np.ceil(0.60 * len(r))))
        and qrs_narrower_n >= int(np.ceil(0.80 * qrs_comparable_n))
        and qrs_delta_median is not None
        and qrs_dwt_width_median is not None
        and qrs_delta_median / max(qrs_dwt_width_median, 1.0) >= 0.08
        and qrs_delta_mad is not None
        and qrs_delta_mad <= max(6.0, 0.50 * qrs_delta_median)
    )

    beats: list[Dict[str, Any]] = []
    prev_t_off: int | None = None
    for rp in r:
        rp = int(rp)
        qrs_candidate = qrs_candidate_map.get(rp) or {}
        dwt_on = qrs_candidate.get("dwt_on")
        dwt_off = qrs_candidate.get("dwt_off")
        fb_on = qrs_candidate.get("fb_on")
        fb_off = qrs_candidate.get("fb_off")
        fb_conf = float(qrs_candidate.get("fb_confidence") or 0.0)

        q_on, q_off, fiducial_confidence, fiducial_source = (
            _choose_qrs_bounds(
                dwt_on=dwt_on,
                dwt_off=dwt_off,
                fb_on=fb_on,
                fb_off=fb_off,
                fb_confidence=fb_conf,
                fs=fs,
            )
        )

        # Do not relax the existing per-beat fusion gate. If it retained DWT,
        # allow the independent digital boundary only when the entire lead
        # supplies a repeatable, directionally consistent corroboration.
        if (
            qrs_lead_consensus
            and fiducial_source == "NEUROKIT_DWT_VERIFIED_BY_SLOPE"
            and fb_on is not None
            and fb_off is not None
            and qrs_candidate.get("fb_width_ms") is not None
            and qrs_candidate.get("dwt_width_ms") is not None
            and float(qrs_candidate["fb_width_ms"]) < float(qrs_candidate["dwt_width_ms"])
            and 60.0 <= float(qrs_candidate["fb_width_ms"]) <= 160.0
            and fb_conf >= 0.55
        ):
            q_on = int(fb_on)
            q_off = int(fb_off)
            fiducial_confidence = float(fb_conf)
            fiducial_source = "DIGITAL_HYSTERESIS_LEAD_CONSENSUS_OVER_DWT"

        if q_on is None or q_off is None or q_off <= q_on:
            continue

        p_on = _nearest_before(p_on_all, q_on, int(round(0.03 * fs)), int(round(0.40 * fs)))
        p_off = _nearest_before(p_off_all, q_on, int(round(0.01 * fs)), int(round(0.30 * fs)))
        p_peak = _nearest_before(p_peak_all, q_on, int(round(0.03 * fs)), int(round(0.35 * fs)))
        p_fiducial_source = "NEUROKIT_DWT" if p_peak is not None else None
        p_fiducial_confidence = 1.0 if p_peak is not None else 0.0
        p_fb = p_fallback_map.get(int(rp))
        if p_fb is not None:
            p_on = int(p_fb["p_on"])
            p_peak = int(p_fb["p_peak"])
            p_off = int(p_fb["p_off"])
            p_fiducial_source = "REGULAR_RHYTHM_PRE_R_TEMPLATE"
            p_fiducial_confidence = float(p_fb.get("confidence") or 0.0)
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

        next_r_candidates = r[r > rp]
        next_r = int(next_r_candidates[0]) if next_r_candidates.size else None
        fb_t_peak, fb_t_off, fb_t_conf = _fallback_t_fiducials(
            x,
            qrs_off=q_off,
            next_r=next_r,
            baseline=float(baseline),
            fs=fs,
        )
        t_peak, t_off, t_fiducial_confidence, t_fiducial_source = (
            _choose_t_fiducials(
                dwt_peak=t_peak,
                dwt_off=t_off,
                fb_peak=fb_t_peak,
                fb_off=fb_t_off,
                fb_confidence=fb_t_conf,
                qrs_off=q_off,
                fs=fs,
            )
        )

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
            "p_fiducial_source": p_fiducial_source,
            "p_fiducial_confidence": float(p_fiducial_confidence),
            "t_fiducial_confidence": float(t_fiducial_confidence),
            "t_fiducial_source": t_fiducial_source,
            "_t_fb_peak_local": int(fb_t_peak) if fb_t_peak is not None else None,
            "_t_fb_off_local": int(fb_t_off) if fb_t_off is not None else None,
            "_t_fb_confidence": float(fb_t_conf),
            "_t_q_on_local": int(q_on),
            "_t_q_off_local": int(q_off),
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

    # Lead-level T-end consensus: a baseline-return candidate may replace DWT
    # only when the same earlier displacement repeats across high-quality beats.
    t_deltas_ms = []
    for beat in beats:
        fb_off = beat.get("_t_fb_off_local")
        dwt_off_abs = beat.get("t_offset_sample")
        if fb_off is None or dwt_off_abs is None:
            continue
        dwt_off = int(dwt_off_abs) - int(a0)
        q_off = int(beat.get("_t_q_off_local"))
        fb_conf = float(beat.get("_t_fb_confidence") or 0.0)
        fb_qrs_to_off_ms = (int(fb_off) - q_off) * 1000.0 / float(fs)
        delta_ms = (dwt_off - int(fb_off)) * 1000.0 / float(fs)
        if fb_conf >= 0.48 and 80.0 <= fb_qrs_to_off_ms <= 320.0 and delta_ms >= 20.0:
            t_deltas_ms.append(float(delta_ms))

    t_consensus_delta = float(np.median(t_deltas_ms)) if len(t_deltas_ms) >= 3 else None
    t_consensus_mad = (
        float(np.median(np.abs(np.asarray(t_deltas_ms) - t_consensus_delta)))
        if t_consensus_delta is not None else None
    )
    t_consistent = (
        t_consensus_delta is not None
        and t_consensus_mad is not None
        and t_consensus_mad <= 12.0
        and len(t_deltas_ms) >= max(3, int(np.ceil(0.60 * len(beats))))
    )

    for beat in beats:
        fb_off = beat.pop("_t_fb_off_local", None)
        fb_peak = beat.pop("_t_fb_peak_local", None)
        fb_conf = float(beat.pop("_t_fb_confidence", 0.0) or 0.0)
        q_on = int(beat.pop("_t_q_on_local"))
        q_off = int(beat.pop("_t_q_off_local"))
        if t_consistent and fb_off is not None:
            dwt_off = int(beat["t_offset_sample"]) - int(a0) if beat.get("t_offset_sample") is not None else None
            fb_qrs_to_off_ms = (int(fb_off) - q_off) * 1000.0 / float(fs)
            delta_ms = ((dwt_off - int(fb_off)) * 1000.0 / float(fs)) if dwt_off is not None else None
            agrees = (
                dwt_off is not None
                and fb_conf >= 0.48
                and 80.0 <= fb_qrs_to_off_ms <= 320.0
                and delta_ms is not None
                and abs(delta_ms - float(t_consensus_delta)) <= 18.0
            )
            if agrees:
                beat["t_offset_sample"] = int(fb_off + a0)
                if fb_peak is not None:
                    beat["t_peak_sample"] = int(fb_peak + a0)
                beat["t_fiducial_confidence"] = float(fb_conf)
                beat["t_fiducial_source"] = "DIGITAL_BASELINE_RETURN_LEAD_CONSENSUS"
                beat["qt_ms"] = float((int(fb_off) - q_on) * 1000.0 / fs)
            else:
                beat["t_fiducial_source"] = "NEUROKIT_DWT_BASELINE_CONSENSUS_REJECTED"
        elif beat.get("t_fiducial_source") == "NEUROKIT_DWT_PENDING_BASELINE_CONSENSUS":
            beat["t_fiducial_source"] = "NEUROKIT_DWT_BASELINE_CONSENSUS_NOT_MET"

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
        "raw_p_peaks_samples": [int(v + a0) for v in p_peak_all.tolist()],
        "raw_p_onsets_samples": [int(v + a0) for v in p_on_all.tolist()],
        "raw_p_offsets_samples": [int(v + a0) for v in p_off_all.tolist()],
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
        "p_fallback_summary": p_fallback_summary,
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
    avg_p_fiducial_confidence = float(
        np.mean([b.get("p_fiducial_confidence", 0.0) for b in beats])
    )
    for field, unit in fields:
        vals = [b[field] for b in beats if b.get(field) is not None]
        value, consistency, n = _robust_aggregate(vals)
        conf = lead_conf * consistency * avg_beat_quality * avg_fiducial_confidence
        if field in {"qt_ms", "t_amp_mv"}:
            conf *= avg_t_fiducial_confidence
        if field in {"p_duration_ms", "pr_ms", "p_amp_mv"}:
            conf *= avg_p_fiducial_confidence
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

    # A P fiducial returned by a delineator is only a candidate. PR may
    # be published only when atrial depolarization is reproducible and coupled
    # to QRS across beats. This prevents baseline/flutter/fibrillatory activity
    # from being hallucinated as a discrete P wave.
    p_candidates = []
    for beat in beats:
        p_on = beat.get("p_onset_sample")
        p_off = beat.get("p_offset_sample")
        pr_val = beat.get("pr_ms")
        p_dur = beat.get("p_duration_ms")
        p_amp = beat.get("p_amp_mv")
        if None in {p_on, p_off, pr_val, p_dur, p_amp}:
            continue
        try:
            pr_f = float(pr_val)
            p_dur_f = float(p_dur)
            p_amp_f = float(p_amp)
        except Exception:
            continue
        if not (
            50.0 <= pr_f <= 500.0
            and 25.0 <= p_dur_f <= 180.0
            and abs(p_amp_f) >= 0.010
        ):
            continue
        p_candidates.append({
            "pr_ms": pr_f,
            "p_duration_ms": p_dur_f,
            "p_amp_mv": p_amp_f,
        })

    # Morphology reproducibility: true discrete P waves should recur with a
    # similar shape/timing across beats. Scalar PR/amplitude agreement alone can
    # be fooled by fibrillatory/flutter baseline deflections.
    p_waveforms = []
    p_polarities = []
    for beat in beats:
        p_on_g = beat.get("p_onset_sample")
        p_off_g = beat.get("p_offset_sample")
        if p_on_g is None or p_off_g is None:
            continue
        try:
            a = int(p_on_g)
            b = int(p_off_g)
        except Exception:
            continue
        if a < 0 or b <= a or b >= len(x_full):
            continue
        seg = np.asarray(x_full[a:b + 1], dtype=float)
        if seg.size < max(5, int(round(0.025 * fs))) or not np.isfinite(seg).all():
            continue
        seg = seg - float(np.median(seg))
        peak = float(np.max(np.abs(seg)))
        if peak < 0.010:
            continue
        target_n = 41
        xp = np.linspace(0.0, 1.0, seg.size)
        xq = np.linspace(0.0, 1.0, target_n)
        rs = np.interp(xq, xp, seg)
        norm = float(np.linalg.norm(rs))
        if norm <= 1e-9:
            continue
        p_waveforms.append(rs / norm)
        p_polarities.append(1 if float(np.sum(rs)) >= 0.0 else -1)

    p_shape_median_correlation = None
    p_polarity_consistency = None
    if len(p_waveforms) >= 3:
        stack = np.vstack(p_waveforms)
        template = np.median(stack, axis=0)
        tnorm = float(np.linalg.norm(template))
        if tnorm > 1e-9:
            template = template / tnorm
            cors = [float(np.dot(row, template)) for row in stack]
            p_shape_median_correlation = float(np.median(cors))
        pos = sum(v > 0 for v in p_polarities)
        neg = sum(v < 0 for v in p_polarities)
        p_polarity_consistency = max(pos, neg) / max(len(p_polarities), 1)

    p_candidate_n = len(p_candidates)
    beat_n = max(len(beats), 1)
    p_coupling_fraction = float(p_candidate_n / beat_n)
    p_amp_median = (
        float(np.median([row["p_amp_mv"] for row in p_candidates]))
        if p_candidates else None
    )
    p_amp_abs_median = (
        float(np.median([abs(row["p_amp_mv"]) for row in p_candidates]))
        if p_candidates else None
    )
    p_pr_median = (
        float(np.median([row["pr_ms"] for row in p_candidates]))
        if p_candidates else None
    )
    p_pr_mad = (
        float(np.median(np.abs(
            np.asarray([row["pr_ms"] for row in p_candidates], dtype=float)
            - float(p_pr_median)
        )))
        if p_candidates and p_pr_median is not None else None
    )
    p_pr_cv = (
        float(np.std(
            np.asarray([row["pr_ms"] for row in p_candidates], dtype=float),
            ddof=1,
        ) / max(float(p_pr_median), 1e-9))
        if len(p_candidates) >= 2 and p_pr_median is not None else None
    )
    p_reproducible = bool(
        p_candidate_n >= 3
        and p_coupling_fraction >= 0.60
        and p_amp_abs_median is not None
        and p_amp_abs_median >= 0.010
        and (
            p_pr_mad is None
            or p_pr_mad <= 35.0
            or (p_pr_cv is not None and p_pr_cv <= 0.18)
        )
        and (
            p_shape_median_correlation is None
            or p_shape_median_correlation >= 0.60
        )
        and (
            p_polarity_consistency is None
            or p_polarity_consistency >= 0.75
        )
    )
    result["atrial_activity"] = {
        "p_candidate_n": int(p_candidate_n),
        "beat_n": int(len(beats)),
        "p_qrs_coupling_fraction": round(p_coupling_fraction, 6),
        "p_amp_median_mv": (
            round(float(p_amp_median), 6)
            if p_amp_median is not None else None
        ),
        "p_amp_abs_median_mv": (
            round(float(p_amp_abs_median), 6)
            if p_amp_abs_median is not None else None
        ),
        "pr_median_ms": (
            round(float(p_pr_median), 3)
            if p_pr_median is not None else None
        ),
        "pr_mad_ms": (
            round(float(p_pr_mad), 3)
            if p_pr_mad is not None else None
        ),
        "pr_cv": (
            round(float(p_pr_cv), 6)
            if p_pr_cv is not None and math.isfinite(float(p_pr_cv)) else None
        ),
        "p_wave_reproducible": p_reproducible,
        "p_shape_median_correlation": (
            round(float(p_shape_median_correlation), 6)
            if p_shape_median_correlation is not None else None
        ),
        "p_polarity_consistency": (
            round(float(p_polarity_consistency), 6)
            if p_polarity_consistency is not None else None
        ),
        "p_positive": bool(
            p_reproducible
            and p_amp_median is not None
            and float(p_amp_median) > 0.010
        ),
        "rule": (
            ">=3_VALID_P_AND_P_QRS_COUPLING>=0.60_AND_MEDIAN_ABS_P>=0.010mV"
            "_AND_PR_REPRODUCIBLE_AND_P_MORPHOLOGY_REPRODUCIBLE"
        ),
    }

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

def _crosslead_dispersion_metric(
    per_lead: Dict[str, Dict[str, Any]],
    metric_name: str,
    *,
    min_confidence: float = 0.45,
) -> Dict[str, Any]:
    rows = []
    for lead,item in per_lead.items():
        m = (item.get("metrics") or {}).get(metric_name) or {}
        try:
            value = float(m.get("value"))
            confidence = float(m.get("confidence") or 0.0)
        except Exception:
            continue
        if not math.isfinite(value) or confidence < min_confidence:
            continue
        rows.append((lead,value,confidence))
    if len(rows) < 2:
        return _metric(
            None,
            unit="ms",
            confidence=max([r[2] for r in rows],default=0.0),
            reason="LT_2_TRUSTED_LEADS_FOR_DISPERSION",
            extra={"source_leads":[r[0] for r in rows]},
        )
    values=np.asarray([r[1] for r in rows],dtype=float)
    dispersion=float(np.max(values)-np.min(values))
    mad=float(np.median(np.abs(values-np.median(values))))
    return _metric(
        dispersion,
        unit="ms",
        confidence=float(np.mean([r[2] for r in rows])),
        extra={
            "source_leads":[r[0] for r in rows],
            "source_n":len(rows),
            "min_ms":round(float(np.min(values)),6),
            "max_ms":round(float(np.max(values)),6),
            "cross_lead_mad_ms":round(mad,6),
            "consensus_mode":"MAX_MINUS_MIN_TRUSTED_LEADS",
        },
    )


def _global_qrs_metric(per_lead: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate global 12-lead QRS duration from a robust upper envelope.

    A median of lead-specific QRS durations is not physiologically equivalent
    to global QRS duration: terminal conduction delay may be visible in only a
    subset of leads. The global duration should reflect later visible offset
    while remaining resistant to a single noisy lead. We therefore use the
    median of the three longest non-outlying trusted lead measurements.
    """
    candidates: list[tuple[str, float, float]] = []
    for lead, item in per_lead.items():
        m = (item.get("metrics") or {}).get("qrs_ms") or {}
        value = m.get("value")
        confidence = float(m.get("confidence") or 0.0)
        if value is None or not math.isfinite(float(value)):
            continue
        value_f = float(value)
        if not 40.0 <= value_f <= 220.0 or confidence < 0.20:
            continue
        candidates.append((lead, value_f, confidence))

    trusted = [row for row in candidates if row[2] >= 0.45]
    rows = trusted if len(trusted) >= 3 else candidates
    if len(rows) < 3:
        return _consensus_metric(per_lead, "qrs_ms", unit="ms")

    values = np.asarray([v for _, v, _ in rows], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) if values.size >= 2 else 0.0

    # Reject only isolated high-side outliers. The floor of 30 ms prevents a
    # tight narrow-QRS cluster from accepting one spurious very long lead.
    robust_sigma = 1.4826 * mad
    upper_limit = median + max(30.0, 3.0 * robust_sigma)
    filtered = [row for row in rows if row[1] <= upper_limit]
    if len(filtered) < 3:
        filtered = rows

    upper = sorted(filtered, key=lambda row: row[1], reverse=True)[:3]
    upper_values = np.asarray([v for _, v, _ in upper], dtype=float)
    upper_value = float(np.median(upper_values))
    upper_conf = float(np.mean([c for _, _, c in upper]))
    spread = float(np.max(upper_values) - np.min(upper_values))
    agreement = float(
        np.clip(1.0 - spread / max(upper_value, 60.0), 0.45, 1.0)
    )
    source_factor = float(np.clip(len(filtered) / 6.0, 0.60, 1.0))
    confidence = upper_conf * agreement * source_factor

    return _metric(
        upper_value,
        unit="ms",
        confidence=confidence,
        extra={
            "source_leads": [lead for lead, _, _ in upper],
            "source_n": len(upper),
            "all_eligible_leads": [lead for lead, _, _ in filtered],
            "all_eligible_n": len(filtered),
            "cross_lead_median_ms": round(median, 6),
            "cross_lead_mad_ms": round(mad, 6),
            "upper_envelope_spread_ms": round(spread, 6),
            "consensus_mode": "ROBUST_UPPER_ENVELOPE_QRS",
            "definition": (
                "MEDIAN_OF_THREE_LONGEST_NON_OUTLYING_TRUSTED_LEAD_DURATIONS"
            ),
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


def _global_atrial_activity(
    per_lead: Dict[str, Dict[str, Any]],
    rhythm_lead: str | None,
) -> Dict[str, Any]:
    reproducible_leads: list[str] = []
    for lead in LEADS:
        atrial = (per_lead.get(lead) or {}).get("atrial_activity") or {}
        if bool(atrial.get("p_wave_reproducible")):
            reproducible_leads.append(lead)

    rhythm_item = per_lead.get(str(rhythm_lead or "")) or {}
    rhythm_atrial = rhythm_item.get("atrial_activity") or {}
    rhythm_qrs = int(rhythm_item.get("r_count") or 0)
    rhythm_fraction = float(rhythm_atrial.get("p_qrs_coupling_fraction") or 0.0)
    rhythm_reproducible = bool(rhythm_atrial.get("p_wave_reproducible"))

    # A long rhythm strip with many QRS but almost no P-QRS coupling is strong
    # negative evidence against publishing a PR. Multi-lead reproducibility can
    # rescue a poor single strip only when at least two independent leads agree.
    strong_rhythm_negative = bool(
        rhythm_qrs >= 5
        and float(rhythm_item.get("duration_s") or 0.0) >= 5.0
        and rhythm_fraction < 0.35
    )
    multilead_support = len(reproducible_leads) >= 2
    atrial_reproducible = bool(
        (rhythm_reproducible or multilead_support)
        and not (strong_rhythm_negative and not multilead_support)
    )

    lead_ii = per_lead.get("II") or {}
    ii_atrial = lead_ii.get("atrial_activity") or {}
    sinus_compatible = bool(
        atrial_reproducible
        and bool(ii_atrial.get("p_wave_reproducible"))
        and bool(ii_atrial.get("p_positive"))
        and float(ii_atrial.get("p_qrs_coupling_fraction") or 0.0) >= 0.70
    )

    return {
        "evaluable": True,
        "p_wave_reproducible": atrial_reproducible,
        "sinus_compatible": sinus_compatible,
        "reproducible_leads": reproducible_leads,
        "reproducible_lead_n": len(reproducible_leads),
        "rhythm_lead": rhythm_lead,
        "rhythm_p_qrs_coupling_fraction": round(rhythm_fraction, 6),
        "rhythm_p_wave_reproducible": rhythm_reproducible,
        "strong_rhythm_negative": strong_rhythm_negative,
        "lead_ii_p_qrs_coupling_fraction": float(
            ii_atrial.get("p_qrs_coupling_fraction") or 0.0
        ),
        "lead_ii_p_positive": bool(ii_atrial.get("p_positive")),
        "pr_reportable": atrial_reproducible,
        "reason": (
            None
            if atrial_reproducible
            else "P_WAVES_NOT_REPRODUCIBLE"
        ),
        "rule": (
            "PR_REQUIRES_REPRODUCIBLE_P_QRS_COUPLING; "
            "SINUS_COMPATIBLE_REQUIRES_REPRODUCIBLE_POSITIVE_P_IN_II"
        ),
    }


def _fascicular_conduction_pattern(
    per_lead: Dict[str, Dict[str, Any]],
    axis: Dict[str, Any],
    global_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Conservative LAFB/HBAI compatibility assessment from digital morphology."""
    axis_deg = axis.get("degrees")
    try:
        axis_deg = float(axis_deg) if axis_deg is not None else None
    except Exception:
        axis_deg = None

    def metric(lead: str, name: str) -> float | None:
        m = ((per_lead.get(lead) or {}).get("metrics") or {}).get(name) or {}
        v = m.get("value")
        try:
            return float(v) if v is not None and math.isfinite(float(v)) else None
        except Exception:
            return None

    def net_positive(lead: str) -> bool | None:
        area = metric(lead, "qrs_net_area_mv_ms")
        if area is not None:
            return bool(area > 0.0)
        r = metric(lead, "r_amp_mv")
        s = metric(lead, "s_amp_mv")
        if r is None or s is None:
            return None
        return abs(float(r)) > abs(float(s))

    def rs_pattern(lead: str) -> str | None:
        r = metric(lead, "r_amp_mv")
        s = metric(lead, "s_amp_mv")
        if r is None or s is None:
            return None
        if abs(r) >= abs(s) * 1.15:
            return "R_DOMINANT"
        if abs(s) >= abs(r) * 1.15:
            return "S_DOMINANT"
        return "BIPHASIC"

    axis_support = bool(
        axis_deg is not None and -90.0 <= axis_deg <= -45.0
    )
    i_positive = net_positive("I")
    avl_positive = net_positive("aVL")
    inferior_patterns = {
        lead: rs_pattern(lead) for lead in ("II", "III", "aVF")
    }
    inferior_s_n = sum(v == "S_DOMINANT" for v in inferior_patterns.values())
    superior_support = bool(i_positive is True and avl_positive is True)
    inferior_support = bool(inferior_s_n >= 2)

    qrs_metric = global_metrics.get("qrs_ms") or {}
    try:
        qrs_ms = (
            float(qrs_metric.get("value"))
            if qrs_metric.get("value") is not None
            else None
        )
    except Exception:
        qrs_ms = None

    q_i = metric("I", "q_amp_mv")
    q_avl = metric("aVL", "q_amp_mv")
    qdur_i = metric("I", "q_duration_ms")
    qdur_avl = metric("aVL", "q_duration_ms")
    small_q_superior = bool(
        any(
            q is not None
            and -0.15 <= q <= -0.01
            and (qd is None or qd < 40.0)
            for q, qd in ((q_i, qdur_i), (q_avl, qdur_avl))
        )
    )

    score_components = {
        "left_axis_minus45_to_minus90": 0.40 if axis_support else 0.0,
        "positive_qrs_I_and_aVL": 0.22 if superior_support else 0.0,
        "rS_in_at_least_two_inferior_leads": 0.25 if inferior_support else 0.0,
        "small_q_superior_support": 0.08 if small_q_superior else 0.0,
        "qrs_not_complete_bundle_branch_range": (
            0.05 if qrs_ms is not None and qrs_ms < 120.0 else 0.0
        ),
    }
    score = float(np.clip(sum(score_components.values()), 0.0, 1.0))
    lafb_compatible = bool(
        axis_support
        and superior_support
        and inferior_support
        and score >= 0.80
    )

    return {
        "evaluable": bool(axis_deg is not None),
        "classification": (
            "LAFB_COMPATIBLE"
            if lafb_compatible
            else "NO_FASCICULAR_PATTERN_ESTABLISHED"
        ),
        "confidence": round(score, 6),
        "diagnostic_claim_allowed": False,
        "axis_deg": axis_deg,
        "qrs_ms": qrs_ms,
        "criteria": {
            "axis_minus45_to_minus90": axis_support,
            "positive_qrs_I": i_positive,
            "positive_qrs_aVL": avl_positive,
            "inferior_rs_patterns": inferior_patterns,
            "inferior_s_dominant_n": inferior_s_n,
            "small_q_superior_support": small_q_superior,
        },
        "score_components": score_components,
        "source": "CALIBRATED_DIGITAL_SIGNAL_MORPHOLOGY",
        "note": (
            "Compatibility rule for left anterior fascicular block (HBAI/LAFB). "
            "Left axis deviation alone is insufficient; superior positive QRS "
            "and inferior rS morphology are also required."
        ),
    }


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
    atrial_activity = _global_atrial_activity(per_lead, rhythm_lead)
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
            "r_peaks_samples": list(src.get("r_peaks_samples") or []),
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
            "rr_regularity": "REGULAR" if regular else "IRREGULAR",
            "atrial_activity_reproducible": bool(
                atrial_activity.get("p_wave_reproducible")
            ),
            "sinus_compatible": bool(atrial_activity.get("sinus_compatible")),
            "p_qrs_coupling_fraction": atrial_activity.get(
                "rhythm_p_qrs_coupling_fraction"
            ),
            "confidence": src.get("confidence"),
            "regularity_rule": "RR_CV<=0.10_AND_RR_MAD_MEDIAN<=0.08",
        }

    rhythm_consensus = build_rhythm_consensus(per_lead, rhythm_lead)
    rr_pattern = rr_irregularity_score(rhythm)
    rhythm["rr_irregularity_analysis"] = rr_pattern
    if rr_pattern.get("score") is not None:
        rhythm["rr_irregularity_score"] = rr_pattern.get("score")
    rhythm["rate_consensus"] = rhythm_consensus

    p_duration_candidate = _consensus_metric(
        per_lead, "p_duration_ms", unit="ms"
    )
    pr_candidate = _consensus_metric(per_lead, "pr_ms", unit="ms")

    if bool(atrial_activity.get("pr_reportable")):
        p_duration_global = p_duration_candidate
        pr_global = pr_candidate
    else:
        p_duration_global = _metric(
            None,
            unit="ms",
            confidence=0.0,
            reason="P_WAVES_NOT_REPRODUCIBLE",
            extra={
                "candidate_value": p_duration_candidate.get("value"),
                "candidate_confidence": p_duration_candidate.get("confidence"),
                "atrial_gate": atrial_activity,
            },
        )
        pr_global = _metric(
            None,
            unit="ms",
            confidence=0.0,
            reason="P_WAVES_NOT_REPRODUCIBLE",
            extra={
                "candidate_value": pr_candidate.get("value"),
                "candidate_confidence": pr_candidate.get("confidence"),
                "candidate_source_leads": pr_candidate.get("source_leads"),
                "atrial_gate": atrial_activity,
            },
        )

    ectopy = analyze_ectopy(canonical_ecg, per_lead, rhythm)

    atrial_mechanism = analyze_native_atrial_mechanism(
        canonical_ecg,
        per_lead,
        rhythm,
        atrial_activity,
        ectopy=ectopy,
    )

    global_metrics = {
        "heart_rate_bpm": _metric(
            (
                rhythm_consensus.get("heart_rate_bpm")
                if rhythm_consensus.get("evaluable")
                else rhythm.get("heart_rate_bpm")
                if rhythm.get("evaluable")
                else None
            ),
            unit="bpm",
            confidence=(
                float(rhythm_consensus.get("confidence") or 0.0)
                if rhythm_consensus.get("evaluable")
                else float(rhythm.get("confidence") or 0.0)
            ),
            reason="RHYTHM_NOT_MEASURABLE",
            extra={
                "source": (
                    rhythm_consensus.get("source")
                    if rhythm_consensus.get("evaluable")
                    else "SELECTED_RHYTHM_LEAD"
                ),
                "rate_consensus": rhythm_consensus,
            },
        ),
        "qrs_ms": _global_qrs_metric(per_lead),
        "p_duration_ms": p_duration_global,
        "pr_ms": pr_global,
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

    hr_value = (
        float(global_metrics["heart_rate_bpm"].get("value"))
        if global_metrics["heart_rate_bpm"].get("value") is not None
        else None
    )
    qtc_framingham = (
        float(qt) + 154.0 * (1.0 - rr_s)
        if qt is not None and rr_s is not None and rr_s > 0
        else None
    )
    qtc_hodges = (
        float(qt) + 1.75 * (hr_value - 60.0)
        if qt is not None and hr_value is not None
        else None
    )
    qrs_value = global_metrics["qrs_ms"].get("value")
    jt = (
        float(qt) - float(qrs_value)
        if qt is not None and qrs_value is not None
        else None
    )
    jtc_fridericia = (
        float(qtc_fridericia) - float(qrs_value)
        if qtc_fridericia is not None and qrs_value is not None
        else None
    )
    derived_confidence = min(
        qtc_confidence,
        float(global_metrics["qrs_ms"].get("confidence") or 0.0),
    )
    global_metrics["qtc_framingham_ms"] = _metric(
        qtc_framingham,
        unit="ms",
        confidence=qtc_confidence,
        reason="QT_OR_RR_NOT_MEASURABLE",
    )
    global_metrics["qtc_hodges_ms"] = _metric(
        qtc_hodges,
        unit="ms",
        confidence=qtc_confidence,
        reason="QT_OR_HEART_RATE_NOT_MEASURABLE",
    )
    global_metrics["jt_ms"] = _metric(
        jt,
        unit="ms",
        confidence=derived_confidence,
        reason="QT_OR_QRS_NOT_MEASURABLE",
    )
    global_metrics["jtc_fridericia_ms"] = _metric(
        jtc_fridericia,
        unit="ms",
        confidence=derived_confidence,
        reason="QTC_OR_QRS_NOT_MEASURABLE",
    )
    global_metrics["qrs_dispersion_ms"] = _crosslead_dispersion_metric(
        per_lead, "qrs_ms"
    )
    global_metrics["qt_dispersion_ms"] = _crosslead_dispersion_metric(
        per_lead, "qt_ms"
    )

    qrs_morphology = analyze_qrs_morphology(
        canonical_ecg,
        per_lead,
        global_metrics,
    )

    av_conduction = analyze_av_conduction(
        per_lead,
        atrial_activity,
        global_metrics=global_metrics,
    )

    wide_complex_tachycardia = analyze_wide_complex_tachycardia(
        canonical_ecg,
        {
            "rhythm": rhythm,
            "global": global_metrics,
            "leads": per_lead,
            "atrial_activity": atrial_activity,
            "atrial_mechanism": atrial_mechanism,
        },
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

    fascicular_conduction = _fascicular_conduction_pattern(
        per_lead,
        axis,
        global_metrics,
    )

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

    result = {
        "version": MEASUREMENT_VERSION,
        "source": "CALIBRATED_DIGITAL_SIGNAL_ONLY",
        "fs": int(canonical_ecg.get("fs") or 500),
        "calibration": canonical_ecg.get("calibration") or {},
        "rhythm": rhythm,
        "rhythm_consensus": rhythm_consensus,
        "atrial_activity": atrial_activity,
        "atrial_mechanism": atrial_mechanism,
        "wide_complex_tachycardia": wide_complex_tachycardia,
        "fascicular_conduction": fascicular_conduction,
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

    signal_integrity = analyze_signal_integrity(canonical_ecg, per_lead)
    measurement_consensus = build_measurement_consensus(
        canonical_ecg,
        per_lead,
        global_metrics,
        rhythm,
        axis,
    )
    measurement_failure_audit = audit_measurement_consensus(
        measurement_consensus
    )
    feature_graph = build_ecg_feature_graph(
        per_lead=per_lead,
        global_metrics=global_metrics,
        rhythm=rhythm,
        axis=axis,
        atrial_activity=atrial_activity,
        atrial_mechanism=atrial_mechanism,
        wide_complex_tachycardia=wide_complex_tachycardia,
        fascicular_conduction=fascicular_conduction,
        measurement_consensus=measurement_consensus,
        signal_integrity=signal_integrity,
        ectopy=ectopy,
        qrs_morphology=qrs_morphology,
        av_conduction=av_conduction,
    )
    preexcitation = analyze_preexcitation(feature_graph, qrs_morphology)
    feature_graph["specialist_evidence"]["preexcitation"] = dict(preexcitation)
    crosslead_conduction = analyze_crosslead_conduction(feature_graph)

    high_recall_candidates = build_high_recall_candidates(
        feature_graph,
        crosslead_conduction,
        per_lead,
    )
    feature_graph["specialist_evidence"]["high_recall_candidates"] = dict(
        high_recall_candidates
    )

    consistency = evaluate_ecg_consistency(feature_graph, crosslead_conduction)
    domain_gates = build_domain_gates(
        feature_graph,
        crosslead_conduction,
        consistency,
    )
    evidence_fusion = fuse_candidate_evidence(
        high_recall_candidates,
        domain_gates,
    )
    specialist_reasoning = reason_ecg(
        feature_graph,
        crosslead_conduction,
        consistency,
        domain_gates=domain_gates,
        evidence_fusion=evidence_fusion,
    )

    result.update({
        "signal_integrity": signal_integrity,
        "measurement_consensus": measurement_consensus,
        "measurement_failure_audit": measurement_failure_audit,
        "ectopy": ectopy,
        "qrs_morphology": qrs_morphology,
        "av_conduction": av_conduction,
        "preexcitation": preexcitation,
        "feature_graph": feature_graph,
        "crosslead_conduction": crosslead_conduction,
        "high_recall_candidates": high_recall_candidates,
        "domain_gates": domain_gates,
        "evidence_fusion": evidence_fusion,
        "consistency": consistency,
        "specialist_reasoning": specialist_reasoning,
    })
    return result
