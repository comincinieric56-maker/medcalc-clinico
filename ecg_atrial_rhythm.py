from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy.signal import butter, correlate, detrend, sosfiltfilt, welch


PREFERRED_ATRIAL_LEADS = ("II", "V1")
FALLBACK_ATRIAL_LEADS = ("aVF", "V5", "V6", "III")
ATRIAL_ANALYZER_VERSION = "MEDCALC_NATIVE_ATRIAL_MECHANISM_V1"


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


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _linear_score(value: float | None, lo: float, hi: float) -> float:
    if value is None or not math.isfinite(float(value)) or hi <= lo:
        return 0.0
    return _clip01((float(value) - lo) / (hi - lo))


def _guideline_af_gate(
    *,
    p_reproducible: bool,
    rr_irregularity: float,
    broad_entropy_score: float,
    periodicity_score: float,
    fwave_score: float,
    flutter_guard: bool,
    ectopy_driven: bool,
    ectopy_burden: float,
) -> tuple[bool, bool]:
    """Return standard AF-pattern support and strong-AF override.

    This is intentionally a transparent evidence gate rather than a trained
    probability. AF requires absent reproducible P waves, irregular RR and
    disorganized atrial activity; organized flutter morphology is excluded.
    """
    p_absent = not bool(p_reproducible)
    disorganized_atrial = bool(
        broad_entropy_score >= 0.45
        or periodicity_score <= 0.45
        or fwave_score >= 0.35
    )
    guideline = bool(
        p_absent
        and rr_irregularity >= 0.45
        and disorganized_atrial
        and not flutter_guard
        and not ectopy_driven
    )
    strong_despite_ectopy = bool(
        p_absent
        and rr_irregularity >= 0.60
        and fwave_score >= 0.55
        and broad_entropy_score >= 0.50
        and not flutter_guard
        and ectopy_burden < 0.25
    )
    return guideline, strong_despite_ectopy


def _bandpass(x: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    if len(x) < max(40, int(round(1.0 * fs))):
        return np.asarray(x, dtype=float)
    nyq = 0.5 * float(fs)
    lo_n = max(0.001, float(lo) / nyq)
    hi_n = min(0.999, float(hi) / nyq)
    if hi_n <= lo_n:
        return np.asarray(x, dtype=float)
    sos = butter(3, [lo_n, hi_n], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, np.asarray(x, dtype=float))
    except Exception:
        return np.asarray(x, dtype=float)


def _cancel_qrs_template(
    x: np.ndarray,
    r_peaks_local: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Subtract a median ventricular QRS template from native signal.

    The cancellation window is intentionally narrow (-80/+110 ms) so atrial
    activity outside the ventricular depolarization is preserved. This derived
    residual is used only for atrial-mechanism analysis and never replaces the
    canonical clinical signal.
    """
    y = np.asarray(x, dtype=float).copy()
    if y.size < 10 or r_peaks_local.size < 3:
        return y, {
            "status": "SKIPPED",
            "reason": "LT_3_R_PEAKS",
            "beats_used": int(r_peaks_local.size),
        }

    pre = max(1, int(round(0.080 * fs)))
    post = max(1, int(round(0.110 * fs)))
    width = pre + post + 1

    beats = []
    valid_r = []
    for rp in r_peaks_local.astype(int):
        a = int(rp) - pre
        b = int(rp) + post + 1
        if a < 0 or b > len(y):
            continue
        seg = y[a:b]
        if seg.size != width or not np.isfinite(seg).all():
            continue
        local_baseline = float(np.median(np.r_[seg[: max(2, pre // 4)], seg[-max(2, post // 4):]]))
        beats.append(seg - local_baseline)
        valid_r.append(int(rp))

    if len(beats) < 3:
        return y, {
            "status": "SKIPPED",
            "reason": "LT_3_COMPLETE_QRS_TEMPLATES",
            "beats_used": int(len(beats)),
        }

    template = np.median(np.vstack(beats), axis=0)
    taper = np.ones(width, dtype=float)
    edge = max(2, int(round(0.018 * fs)))
    ramp = np.sin(np.linspace(0.0, np.pi / 2.0, edge)) ** 2
    taper[:edge] = ramp
    taper[-edge:] = ramp[::-1]

    accum = np.zeros_like(y, dtype=float)
    weight = np.zeros_like(y, dtype=float)
    for rp in valid_r:
        a = rp - pre
        b = rp + post + 1
        accum[a:b] += template * taper
        weight[a:b] += taper

    mask = weight > 1e-9
    residual = y.copy()
    residual[mask] = residual[mask] - accum[mask] / weight[mask]

    return residual, {
        "status": "APPLIED",
        "beats_used": int(len(valid_r)),
        "window_ms": [-80, 110],
        "method": "MEDIAN_QRS_TEMPLATE_SUBTRACTION",
    }


def _spectral_features(x: np.ndarray, fs: int) -> Dict[str, Any]:
    y = np.asarray(x, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < int(round(3.0 * fs)):
        return {"evaluable": False, "reason": "LT_3S_FINITE_ATRIAL_RESIDUAL"}

    y = detrend(y, type="linear")
    y = _bandpass(y, fs, 2.5, 15.0)
    if float(np.std(y)) < 1e-6:
        return {"evaluable": False, "reason": "NEAR_ZERO_ATRIAL_RESIDUAL"}

    nperseg = min(len(y), max(256, int(round(2.0 * fs))))
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, psd = welch(
        y,
        fs=float(fs),
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    psd = np.asarray(psd, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    def band_power(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs <= hi) & np.isfinite(psd)
        if np.sum(m) < 2:
            return 0.0
        return float(np.trapezoid(psd[m], freqs[m]))

    total_1_20 = band_power(1.0, 20.0)
    power_3_10 = band_power(3.0, 10.0)
    power_4_10 = band_power(4.0, 10.0)
    f_wave_fraction = (
        float(power_4_10 / total_1_20)
        if total_1_20 > 1e-12 else 0.0
    )

    atrial_mask = (freqs >= 3.0) & (freqs <= 10.0) & np.isfinite(psd)
    if np.sum(atrial_mask) < 3 or power_3_10 <= 1e-12:
        return {"evaluable": False, "reason": "INSUFFICIENT_3_10HZ_POWER"}

    fa = freqs[atrial_mask]
    pa = psd[atrial_mask]
    peak_i = int(np.argmax(pa))
    dominant_hz = float(fa[peak_i])
    peak_power = float(pa[peak_i])
    median_power = float(np.median(pa))
    peak_to_median = float(peak_power / max(median_power, 1e-12))

    def peak_band_ratio(half_width_hz: float) -> float:
        m = (fa >= dominant_hz - half_width_hz) & (fa <= dominant_hz + half_width_hz)
        if np.sum(m) < 1:
            return 0.0
        narrow = float(np.trapezoid(pa[m], fa[m])) if np.sum(m) >= 2 else float(pa[m][0])
        return float(narrow / max(power_3_10, 1e-12))

    narrow_1p2_ratio = peak_band_ratio(0.60)
    envelope_2p5_ratio = peak_band_ratio(1.25)

    prob = np.clip(pa, 0.0, None)
    denom = float(np.sum(prob))
    if denom > 0:
        prob = prob / denom
        spectral_entropy = float(
            -np.sum(prob * np.log(prob + 1e-12))
            / max(np.log(len(prob)), 1e-12)
        )
    else:
        spectral_entropy = 1.0

    filtered = _bandpass(y, fs, 3.0, 10.0)
    filtered = filtered - float(np.mean(filtered))
    ac = correlate(filtered, filtered, mode="full", method="fft")
    ac = ac[len(filtered) - 1:]
    if ac.size and ac[0] > 0:
        ac = ac / float(ac[0])
    min_lag = max(1, int(round(fs / 10.0)))
    max_lag = min(len(ac) - 1, int(round(fs / 3.0)))
    periodicity = 0.0
    periodicity_hz = None
    if max_lag > min_lag:
        z = ac[min_lag:max_lag + 1]
        if z.size:
            rel = int(np.nanargmax(z))
            lag = min_lag + rel
            periodicity = _clip01(float(z[rel]))
            periodicity_hz = float(fs / lag) if lag > 0 else None

    return {
        "evaluable": True,
        "dominant_frequency_hz": round(dominant_hz, 6),
        "atrial_rate_bpm_from_df": round(dominant_hz * 60.0, 3),
        "f_wave_power_fraction_4_10hz": round(f_wave_fraction, 6),
        "narrow_peak_ratio_1p2hz": round(narrow_1p2_ratio, 6),
        "peak_envelope_ratio_2p5hz": round(envelope_2p5_ratio, 6),
        "spectral_entropy_3_10hz": round(spectral_entropy, 6),
        "peak_to_median_power_ratio": round(peak_to_median, 6),
        "autocorrelation_periodicity": round(periodicity, 6),
        "autocorrelation_frequency_hz": (
            round(float(periodicity_hz), 6)
            if periodicity_hz is not None else None
        ),
        "residual_rms_mv": round(float(np.sqrt(np.mean(filtered ** 2))), 6),
        "method": "WELCH_PLUS_AUTOCORRELATION_AFTER_NARROW_QRS_CANCELLATION",
    }


def _lead_features(
    lead: str,
    canonical_item: Dict[str, Any],
    measurement_item: Dict[str, Any],
    fs: int,
) -> Dict[str, Any]:
    signal = _as_signal(canonical_item)
    quality = _as_quality(canonical_item, len(signal))
    usable = np.isfinite(signal) & (quality > 0)
    runs = _finite_runs(usable)
    if not runs:
        return {"lead": lead, "evaluable": False, "reason": "NO_USABLE_SIGNAL"}

    a, b = max(runs, key=lambda ab: ab[1] - ab[0])
    if (b - a) < int(round(3.0 * fs)):
        return {
            "lead": lead,
            "evaluable": False,
            "reason": "LT_3S_CONTIGUOUS_SIGNAL",
            "duration_s": round((b - a) / float(fs), 3),
        }

    x = np.asarray(signal[a:b], dtype=float)
    r_abs = np.asarray(measurement_item.get("r_peaks_samples") or [], dtype=int)
    r_local = r_abs[(r_abs >= a) & (r_abs < b)] - int(a)
    residual, cancellation = _cancel_qrs_template(x, r_local, fs)
    features = _spectral_features(residual, fs)

    lead_conf = float(measurement_item.get("confidence") or canonical_item.get("confidence") or 0.0)
    observed_fraction = float(np.mean(quality[a:b] > 0)) if b > a else 0.0
    confidence = _clip01(lead_conf * math.sqrt(max(observed_fraction, 0.0)))

    return {
        "lead": lead,
        "evaluable": bool(features.get("evaluable")),
        "duration_s": round((b - a) / float(fs), 3),
        "r_count": int(r_local.size),
        "confidence": round(confidence, 6),
        "qrs_cancellation": cancellation,
        "features": features,
        "source": "NATIVE_CALIBRATED_DIGITAL_SIGNAL",
    }


def analyze_native_atrial_mechanism(
    canonical_ecg: Dict[str, Any],
    per_lead_measurements: Dict[str, Dict[str, Any]],
    rhythm: Dict[str, Any],
    atrial_activity: Dict[str, Any],
    ectopy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Characterize native atrial activity independently of R27 and RR labels.

    Output scores are research compatibility scores, not calibrated
    probabilities. The module intentionally returns an indeterminate mechanism
    unless spectral/temporal separation is sufficiently strong.
    """
    fs = int(canonical_ecg.get("fs") or 500)
    lead_items = canonical_ecg.get("leads") or {}

    selected = []
    for lead in PREFERRED_ATRIAL_LEADS:
        if lead in lead_items:
            selected.append(lead)
    for lead in FALLBACK_ATRIAL_LEADS:
        if len(selected) >= 3:
            break
        if lead in lead_items and lead not in selected:
            selected.append(lead)

    lead_results = []
    for lead in selected:
        row = _lead_features(
            lead,
            dict(lead_items.get(lead) or {}),
            dict(per_lead_measurements.get(lead) or {}),
            fs,
        )
        lead_results.append(row)

    usable = [
        row for row in lead_results
        if row.get("evaluable")
        and float(row.get("confidence") or 0.0) >= 0.25
    ]
    if not usable:
        return {
            "version": ATRIAL_ANALYZER_VERSION,
            "evaluable": False,
            "mechanism": "NOT_EVALUABLE",
            "confidence": 0.0,
            "reason": "NO_NATIVE_ATRIAL_LEAD_WITH_USABLE_RESIDUAL",
            "lead_results": lead_results,
            "diagnostic_claim_allowed": False,
        }

    weights = np.asarray(
        [max(float(row.get("confidence") or 0.0), 0.05) for row in usable],
        dtype=float,
    )
    weights = weights / max(float(np.sum(weights)), 1e-9)

    def weighted_feature(name: str) -> float | None:
        vals = []
        ws = []
        for w, row in zip(weights, usable):
            v = ((row.get("features") or {}).get(name))
            if v is None:
                continue
            try:
                vf = float(v)
            except Exception:
                continue
            if not math.isfinite(vf):
                continue
            vals.append(vf)
            ws.append(float(w))
        if not vals:
            return None
        ws_arr = np.asarray(ws, dtype=float)
        ws_arr = ws_arr / max(float(np.sum(ws_arr)), 1e-9)
        return float(np.sum(np.asarray(vals, dtype=float) * ws_arr))

    df = weighted_feature("dominant_frequency_hz")
    fwave = weighted_feature("f_wave_power_fraction_4_10hz")
    narrow = weighted_feature("peak_envelope_ratio_2p5hz")
    entropy = weighted_feature("spectral_entropy_3_10hz")
    periodicity = weighted_feature("autocorrelation_periodicity")
    peak_ratio = weighted_feature("peak_to_median_power_ratio")

    dfs = [
        float((row.get("features") or {}).get("dominant_frequency_hz"))
        for row in usable
        if (row.get("features") or {}).get("dominant_frequency_hz") is not None
    ]
    df_spread = float(np.std(dfs, ddof=1)) if len(dfs) >= 2 else 0.0
    cross_lead_df_consistency = _clip01(1.0 - df_spread / 2.0)

    rr_cv = rhythm.get("rr_cv")
    rr_irregularity = _linear_score(
        float(rr_cv) if rr_cv is not None else None,
        0.04,
        0.18,
    )
    rr_regularity = 1.0 - rr_irregularity
    ectopy = ectopy or {}
    ectopy_driven = bool(ectopy.get("irregularity_may_be_ectopy_driven"))
    ectopy_burden = float(ectopy.get("premature_burden") or 0.0)
    hr = rhythm.get("heart_rate_bpm")
    tachy = _linear_score(float(hr) if hr is not None else None, 95.0, 150.0)

    atrial_rate = (float(df) * 60.0) if df is not None else None
    conduction_integer_closeness = 0.0
    nearest_ratio = None
    if (
        atrial_rate is not None
        and hr is not None
        and float(hr) > 0
        and 180.0 <= atrial_rate <= 600.0
    ):
        ratio = float(atrial_rate / float(hr))
        candidates = [2.0, 3.0, 4.0]
        nearest = min(candidates, key=lambda z: abs(ratio - z))
        nearest_ratio = nearest
        conduction_integer_closeness = _clip01(1.0 - abs(ratio - nearest) / 0.35)

    narrow_score = _linear_score(narrow, 0.28, 0.62)
    periodicity_score = _linear_score(periodicity, 0.20, 0.70)
    organized_entropy = _clip01(1.0 - float(entropy if entropy is not None else 1.0))
    fwave_score = _linear_score(fwave, 0.10, 0.55)
    broad_entropy_score = _linear_score(entropy, 0.55, 0.92)
    peak_strength = _linear_score(peak_ratio, 2.0, 10.0)

    flutter_score = _clip01(
        0.27 * narrow_score
        + 0.23 * periodicity_score
        + 0.16 * organized_entropy
        + 0.12 * cross_lead_df_consistency
        + 0.12 * conduction_integer_closeness
        + 0.10 * peak_strength
    )
    af_score = _clip01(
        0.27 * fwave_score
        + 0.20 * broad_entropy_score
        + 0.18 * (1.0 - periodicity_score)
        + 0.20 * rr_irregularity
        + 0.15 * (1.0 - cross_lead_df_consistency)
    )

    residual_atrial_strength = max(fwave_score, narrow_score, periodicity_score)
    other_svt_score = _clip01(
        0.35 * tachy
        + 0.30 * rr_regularity
        + 0.20 * (1.0 - residual_atrial_strength)
        + 0.15 * (1.0 - fwave_score)
    )

    if bool(atrial_activity.get("sinus_compatible")):
        mechanism = "SINUS_COMPATIBLE"
        confidence = 0.85
        reason = "REPRODUCIBLE_POSITIVE_P_IN_II_WITH_P_QRS_COUPLING"
    else:
        scores = {
            "AF_COMPATIBLE": af_score,
            "FLUTTER_OR_AT_COMPATIBLE": flutter_score,
            "OTHER_SVT_COMPATIBLE": other_svt_score,
        }
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_score = ordered[0]
        second_score = ordered[1][1]
        margin = float(top_score - second_score)

        strict_flutter = bool(
            top_name == "FLUTTER_OR_AT_COMPATIBLE"
            and top_score >= 0.62
            and margin >= 0.10
            and df is not None
            and 3.0 <= float(df) <= 7.0
            and narrow is not None
            and float(narrow) >= 0.40
        )
        # Guideline-shaped AF evidence: irregular ventricular response,
        # absence of reproducible discrete P waves, and disorganized atrial
        # activity. Spectral F-wave power supports the call but is not required
        # as a single mandatory gate because AF can have low-amplitude baseline
        # activity. Organized flutter-like periodicity is an explicit exclusion.
        p_absent = not bool(atrial_activity.get("p_wave_reproducible"))
        flutter_guard = bool(
            df is not None
            and 3.0 <= float(df) <= 7.0
            and narrow_score >= 0.55
            and periodicity_score >= 0.55
        )
        disorganized_atrial = bool(
            broad_entropy_score >= 0.45
            or periodicity_score <= 0.45
            or fwave_score >= 0.35
        )
        guideline_af_pattern, strong_af_despite_ectopy = _guideline_af_gate(
            p_reproducible=bool(atrial_activity.get("p_wave_reproducible")),
            rr_irregularity=rr_irregularity,
            broad_entropy_score=broad_entropy_score,
            periodicity_score=periodicity_score,
            fwave_score=fwave_score,
            flutter_guard=flutter_guard,
            ectopy_driven=ectopy_driven,
            ectopy_burden=ectopy_burden,
        )
        strict_af = bool(
            (
                top_name == "AF_COMPATIBLE"
                and top_score >= 0.58
                and margin >= 0.06
                and disorganized_atrial
            )
            or guideline_af_pattern
            or strong_af_despite_ectopy
        )
        strict_svt = bool(
            top_name == "OTHER_SVT_COMPATIBLE"
            and top_score >= 0.68
            and margin >= 0.12
        )

        if strict_flutter:
            mechanism = "FLUTTER_OR_AT_COMPATIBLE"
            confidence = top_score
            reason = "ORGANIZED_NARROW_PERIODIC_ATRIAL_ACTIVITY"
        elif strict_af:
            mechanism = "AF_COMPATIBLE"
            confidence = max(
                float(top_score if top_name == "AF_COMPATIBLE" else af_score),
                0.72 if guideline_af_pattern else 0.68,
            )
            reason = (
                "IRREGULAR_RR_ABSENT_REPRODUCIBLE_P_DISORGANIZED_ATRIAL_ACTIVITY"
                if guideline_af_pattern
                else "BROAD_DISORGANIZED_ATRIAL_ACTIVITY_WITH_IRREGULAR_RR"
            )
        elif strict_svt:
            mechanism = "OTHER_SVT_COMPATIBLE"
            confidence = top_score
            reason = "REGULAR_TACHYCARDIA_WITHOUT_STRONG_ATRIAL_SPECTRAL_SIGNATURE"
        else:
            mechanism = "ATRIAL_TACHYARRHYTHMIA_UNDETERMINED"
            confidence = max(0.20, min(top_score, 0.59))
            reason = "COMPATIBILITY_SCORES_NOT_SUFFICIENTLY_SEPARATED"

    return {
        "version": ATRIAL_ANALYZER_VERSION,
        "evaluable": True,
        "source": "NATIVE_CALIBRATED_DIGITAL_SIGNAL_DII_V1_FIRST",
        "mechanism": mechanism,
        "confidence": round(float(confidence), 6),
        "reason": reason,
        "diagnostic_claim_allowed": False,
        "compatibility_scores_not_probabilities": {
            "AF": round(float(af_score), 6),
            "FLUTTER_OR_AT": round(float(flutter_score), 6),
            "OTHER_SVT": round(float(other_svt_score), 6),
        },
        "aggregate_features": {
            "dominant_frequency_hz": round(float(df), 6) if df is not None else None,
            "atrial_rate_bpm_from_df": round(float(atrial_rate), 3) if atrial_rate is not None else None,
            "f_wave_power_fraction_4_10hz": round(float(fwave), 6) if fwave is not None else None,
            "peak_envelope_ratio_2p5hz": round(float(narrow), 6) if narrow is not None else None,
            "spectral_entropy_3_10hz": round(float(entropy), 6) if entropy is not None else None,
            "autocorrelation_periodicity": round(float(periodicity), 6) if periodicity is not None else None,
            "peak_to_median_power_ratio": round(float(peak_ratio), 6) if peak_ratio is not None else None,
            "cross_lead_df_spread_hz": round(float(df_spread), 6),
            "cross_lead_df_consistency": round(float(cross_lead_df_consistency), 6),
            "rr_cv": float(rr_cv) if rr_cv is not None else None,
            "rr_irregularity_score": round(float(rr_irregularity), 6),
            "nearest_integer_av_ratio": nearest_ratio,
            "integer_av_ratio_closeness": round(float(conduction_integer_closeness), 6),
            "ectopy_burden": round(float(ectopy_burden), 6),
            "ectopy_driven_irregularity": ectopy_driven,
            "guideline_af_pattern": bool('guideline_af_pattern' in locals() and guideline_af_pattern),
            "flutter_guard": bool('flutter_guard' in locals() and flutter_guard),
        },
        "lead_results": lead_results,
        "methodology_note": (
            "Research compatibility classifier using native digital atrial activity after "
            "narrow ventricular QRS template cancellation, Welch spectrum, 3-10 Hz "
            "organization, autocorrelation periodicity, cross-lead dominant-frequency "
            "consistency, RR behavior and P-QRS reproducibility. Scores are not calibrated "
            "probabilities and do not replace clinician ECG interpretation."
        ),
    }
