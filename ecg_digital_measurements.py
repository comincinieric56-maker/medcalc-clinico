from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _arr(waves: Mapping[str, Any], key: str) -> np.ndarray:
    value = waves.get(key)
    if value is None:
        return np.asarray([], dtype=int)
    z = np.asarray(value, dtype=float).reshape(-1)
    z = z[np.isfinite(z)]
    return z.astype(int)


def _signal(item: Mapping[str, Any]) -> np.ndarray:
    vals = item.get("signal_mv") or []
    return np.asarray([
        np.nan if v is None else float(v)
        for v in vals
    ], dtype=float)


def _finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    if x.size == 0:
        return []
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _longest_segment(x: np.ndarray) -> Tuple[int, int] | None:
    runs = _finite_runs(np.isfinite(x))
    if not runs:
        return None
    return max(runs, key=lambda ab: ab[1] - ab[0])


def _median(values: Sequence[float]) -> Optional[float]:
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    return float(np.median(z)) if z.size else None


def _mad(values: Sequence[float]) -> Optional[float]:
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    med = float(np.median(z))
    return float(np.median(np.abs(z - med)))


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _metric_confidence(
    *,
    lead_confidence: float,
    n: int,
    values: Sequence[float],
    tolerance: float,
    calibration_confidence: float,
) -> float:
    support = min(1.0, max(0.0, float(n) / 5.0))
    mad = _mad(values)
    consistency = 0.0 if mad is None else max(0.0, 1.0 - float(mad) / max(float(tolerance), 1e-6))
    return _clip01(
        0.45 * float(lead_confidence)
        + 0.20 * support
        + 0.20 * consistency
        + 0.15 * float(calibration_confidence)
    )


def _nk_delineation(x: np.ndarray, fs: int) -> Dict[str, Any]:
    import neurokit2 as nk

    y = np.asarray(x, dtype=float)
    if y.size < int(1.2 * fs):
        raise ValueError("segmento insuficiente")
    clean = nk.ecg_clean(y, sampling_rate=fs, method="neurokit")
    _, peaks = nk.ecg_peaks(clean, sampling_rate=fs)
    r = np.asarray(peaks.get("ECG_R_Peaks", []), dtype=int)
    if r.size < 2:
        raise ValueError("QRS insuficientes")
    _, waves = nk.ecg_delineate(
        clean,
        rpeaks=r,
        sampling_rate=fs,
        method="dwt",
        show=False,
        show_type="all",
    )
    return {"clean": clean, "r": r, "waves": waves or {}}


def _closest_preceding(values: np.ndarray, target: int, low: int, high: int) -> Optional[int]:
    if values.size == 0:
        return None
    cand = values[(values < target - low) & (values >= target - high)]
    return int(cand[-1]) if cand.size else None


def _closest_following(values: np.ndarray, target: int, low: int, high: int) -> Optional[int]:
    if values.size == 0:
        return None
    cand = values[(values > target + low) & (values <= target + high)]
    return int(cand[0]) if cand.size else None


def _baseline_for_beat(
    x: np.ndarray,
    *,
    r: int,
    qrs_on: Optional[int],
    p_on: Optional[int],
    p_off: Optional[int],
    prev_t_off: Optional[int],
    fs: int,
) -> Tuple[Optional[float], str]:
    # Prefer a true PR segment after P offset and before QRS onset.
    if p_off is not None and qrs_on is not None:
        a = p_off + int(round(0.012 * fs))
        b = qrs_on - int(round(0.012 * fs))
        if b - a >= int(round(0.020 * fs)):
            seg = x[max(0, a):min(len(x), b)]
            if np.isfinite(seg).sum() >= max(3, int(0.015 * fs)):
                return float(np.nanmedian(seg)), "PR"

    # Otherwise use the TP interval if the previous T offset is available.
    if prev_t_off is not None and p_on is not None:
        a = prev_t_off + int(round(0.020 * fs))
        b = p_on - int(round(0.020 * fs))
        if b - a >= int(round(0.025 * fs)):
            seg = x[max(0, a):min(len(x), b)]
            if np.isfinite(seg).sum() >= max(3, int(0.020 * fs)):
                return float(np.nanmedian(seg)), "TP"

    # Conservative fallback used only for relative amplitude, never presented
    # as a proven PR/TP baseline.
    a = max(0, r - int(round(0.20 * fs)))
    b = max(a + 1, r - int(round(0.12 * fs)))
    seg = x[a:b]
    if np.isfinite(seg).sum() >= 3:
        return float(np.nanmedian(seg)), "PRE_QRS_FALLBACK"
    return None, "UNAVAILABLE"


def _q_duration_ms(segment: np.ndarray, r_local: int, baseline: float, fs: int) -> Optional[float]:
    if r_local <= 1:
        return None
    pre = np.asarray(segment[:r_local], dtype=float) - float(baseline)
    if pre.size < 2 or not np.isfinite(pre).any():
        return None
    q_idx = int(np.nanargmin(pre))
    if pre[q_idx] >= -0.015:
        return 0.0
    threshold = min(-0.01, 0.20 * float(pre[q_idx]))
    a = q_idx
    while a > 0 and np.isfinite(pre[a - 1]) and pre[a - 1] < threshold:
        a -= 1
    b = q_idx
    while b + 1 < pre.size and np.isfinite(pre[b + 1]) and pre[b + 1] < threshold:
        b += 1
    return float((b - a + 1) * 1000.0 / fs)


def _lead_measurements(
    lead: str,
    item: Mapping[str, Any],
    *,
    calibration_confidence: float,
) -> Dict[str, Any]:
    x0 = _signal(item)
    fs = int(item.get("fs") or 500)
    lead_conf = float(item.get("confidence") or 0.0)
    span = _longest_segment(x0)
    if span is None or span[1] - span[0] < int(1.2 * fs):
        return {
            "lead": lead,
            "evaluable": False,
            "reason": "INSUFFICIENT_CONTIGUOUS_DIGITAL_SIGNAL",
            "confidence": lead_conf,
        }

    a0, b0 = span
    x = x0[a0:b0]
    try:
        nk = _nk_delineation(x, fs)
    except Exception as exc:
        return {
            "lead": lead,
            "evaluable": False,
            "reason": f"DELINEATION_FAILED: {exc}",
            "confidence": lead_conf,
            "duration_s": float((b0 - a0) / fs),
        }

    r = np.asarray(nk["r"], dtype=int)
    waves = nk["waves"]
    r_on = _arr(waves, "ECG_R_Onsets")
    r_off = _arr(waves, "ECG_R_Offsets")
    p_on = _arr(waves, "ECG_P_Onsets")
    p_off = _arr(waves, "ECG_P_Offsets")
    p_peak = _arr(waves, "ECG_P_Peaks")
    t_peak = _arr(waves, "ECG_T_Peaks")
    t_off = _arr(waves, "ECG_T_Offsets")

    rr_ms = np.diff(r).astype(float) * 1000.0 / fs
    qrs_values: List[float] = []
    p_durations: List[float] = []
    pr_values: List[float] = []
    qt_values: List[float] = []
    st_j: List[float] = []
    st_j40: List[float] = []
    st_j60: List[float] = []
    st_j80: List[float] = []
    r_amp: List[float] = []
    s_amp: List[float] = []
    q_amp: List[float] = []
    q_duration: List[float] = []
    t_amp: List[float] = []
    qrs_net_area: List[float] = []
    baselines: List[str] = []
    beat_audit: List[Dict[str, Any]] = []

    for bi, rp in enumerate(r):
        qon = _closest_preceding(r_on, int(rp), 0, int(0.16 * fs))
        qoff = _closest_following(r_off, int(rp), 0, int(0.18 * fs))
        pon = _closest_preceding(p_on, qon if qon is not None else int(rp), int(0.02 * fs), int(0.40 * fs))
        poff = _closest_preceding(p_off, qon if qon is not None else int(rp), 0, int(0.30 * fs))
        pp = _closest_preceding(p_peak, qon if qon is not None else int(rp), int(0.02 * fs), int(0.35 * fs))
        tp = _closest_following(t_peak, int(rp), int(0.10 * fs), int(0.65 * fs))
        toff = _closest_following(t_off, int(rp), int(0.12 * fs), int(0.80 * fs))
        prev_to = None
        if bi > 0:
            prev = t_off[t_off < int(rp)]
            prev_to = int(prev[-1]) if prev.size else None

        baseline, baseline_source = _baseline_for_beat(
            x,
            r=int(rp),
            qrs_on=qon,
            p_on=pon,
            p_off=poff,
            prev_t_off=prev_to,
            fs=fs,
        )
        if baseline is None:
            continue
        baselines.append(baseline_source)

        if qon is not None and qoff is not None and qoff > qon:
            qrs_ms = float((qoff - qon) * 1000.0 / fs)
            if 25.0 <= qrs_ms <= 260.0:
                qrs_values.append(qrs_ms)

            seg = x[qon:qoff + 1]
            if seg.size >= 3 and np.isfinite(seg).sum() >= 3:
                rel = seg - baseline
                r_local = int(np.clip(rp - qon, 0, len(seg) - 1))
                pre = rel[:r_local + 1]
                post = rel[r_local:]
                r_value = float(np.nanmax(rel))
                s_value = float(np.nanmin(post)) if post.size else None
                q_value = float(np.nanmin(pre)) if pre.size else None
                r_amp.append(r_value)
                if s_value is not None:
                    s_amp.append(s_value)
                if q_value is not None:
                    q_amp.append(q_value)
                qdur = _q_duration_ms(seg, r_local, baseline, fs)
                if qdur is not None:
                    q_duration.append(float(qdur))
                qrs_net_area.append(float(np.trapezoid(rel, dx=1.0 / fs)))

            # J point is the QRS offset. Sample ST at fixed offsets from J.
            for target, bucket in [
                (qoff, st_j),
                (qoff + int(round(0.040 * fs)), st_j40),
                (qoff + int(round(0.060 * fs)), st_j60),
                (qoff + int(round(0.080 * fs)), st_j80),
            ]:
                if 0 <= target < len(x) and np.isfinite(x[target]):
                    bucket.append(float(x[target] - baseline))

        if pon is not None and poff is not None and poff > pon:
            pd = float((poff - pon) * 1000.0 / fs)
            if 20.0 <= pd <= 220.0:
                p_durations.append(pd)

        if pon is not None and qon is not None and qon > pon:
            pr = float((qon - pon) * 1000.0 / fs)
            if 50.0 <= pr <= 420.0:
                pr_values.append(pr)

        if qon is not None and toff is not None and toff > qon:
            qt = float((toff - qon) * 1000.0 / fs)
            if 120.0 <= qt <= 800.0:
                qt_values.append(qt)

        if tp is not None and 0 <= tp < len(x) and np.isfinite(x[tp]):
            t_amp.append(float(x[tp] - baseline))

        beat_audit.append({
            "r_sample": int(rp + a0),
            "qrs_on_sample": None if qon is None else int(qon + a0),
            "qrs_off_sample": None if qoff is None else int(qoff + a0),
            "p_on_sample": None if pon is None else int(pon + a0),
            "p_off_sample": None if poff is None else int(poff + a0),
            "p_peak_sample": None if pp is None else int(pp + a0),
            "t_peak_sample": None if tp is None else int(tp + a0),
            "t_off_sample": None if toff is None else int(toff + a0),
            "baseline_source": baseline_source,
            "baseline_mv": float(baseline),
        })

    def summarize(values: Sequence[float], tolerance: float) -> Dict[str, Any]:
        value = _median(values)
        conf = _metric_confidence(
            lead_confidence=lead_conf,
            n=len(values),
            values=values,
            tolerance=tolerance,
            calibration_confidence=calibration_confidence,
        ) if value is not None else 0.0
        return {
            "value": value,
            "confidence": conf,
            "n": int(len(values)),
            "mad": _mad(values),
        }

    qrs_s = summarize(qrs_values, 18.0)
    p_s = summarize(p_durations, 22.0)
    pr_s = summarize(pr_values, 28.0)
    qt_s = summarize(qt_values, 35.0)
    stj_s = summarize(st_j, 0.08)
    st40_s = summarize(st_j40, 0.08)
    st60_s = summarize(st_j60, 0.08)
    st80_s = summarize(st_j80, 0.08)
    r_s = summarize(r_amp, 0.25)
    s_s = summarize(s_amp, 0.25)
    q_s = summarize(q_amp, 0.20)
    qd_s = summarize(q_duration, 18.0)
    t_s = summarize(t_amp, 0.25)
    area_s = summarize(qrs_net_area, 0.030)

    st_reference = st60_s if st60_s["value"] is not None else stj_s
    st_value = st_reference["value"]
    st_direction = "NO_MEDIBLE"
    if st_value is not None:
        if st_value > 0.10:
            st_direction = "ELEVATION"
        elif st_value < -0.10:
            st_direction = "DEPRESSION"
        else:
            st_direction = "ISOELECTRIC_COMPATIBLE"

    t_value = t_s["value"]
    t_polarity = "NO_MEDIBLE"
    if t_value is not None:
        if t_value > 0.05:
            t_polarity = "POSITIVE"
        elif t_value < -0.05:
            t_polarity = "NEGATIVE"
        else:
            t_polarity = "FLAT_OR_BIPHASIC"

    rr_mean = float(np.mean(rr_ms)) if rr_ms.size else None
    rr_sd = float(np.std(rr_ms, ddof=1)) if rr_ms.size >= 2 else None
    rr_cv = (
        float(rr_sd / rr_mean)
        if rr_sd is not None and rr_mean and rr_mean > 0 else None
    )
    rr_mad = (
        float(np.median(np.abs(rr_ms - np.median(rr_ms))))
        if rr_ms.size else None
    )
    rr_rmssd = (
        float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
        if rr_ms.size >= 2 else None
    )

    rs_ratio = None
    if r_s["value"] is not None and s_s["value"] is not None and abs(float(s_s["value"])) > 1e-6:
        rs_ratio = float(abs(float(r_s["value"])) / abs(float(s_s["value"])))

    return {
        "lead": lead,
        "evaluable": True,
        "source": "DIGITAL_SIGNAL_ONLY",
        "fs": fs,
        "duration_s": float((b0 - a0) / fs),
        "signal_confidence": lead_conf,
        "r_count": int(r.size),
        "r_peaks_samples": [int(v + a0) for v in r.tolist()],
        "rr_ms": [round(float(v), 3) for v in rr_ms.tolist()],
        "rr_mean_ms": rr_mean,
        "rr_sd_ms": rr_sd,
        "rr_cv": rr_cv,
        "rr_mad_ms": rr_mad,
        "rr_rmssd_ms": rr_rmssd,
        "heart_rate_bpm": (60000.0 / rr_mean) if rr_mean and rr_mean > 0 else None,
        "qrs_ms": qrs_s,
        "p_duration_ms": p_s,
        "pr_ms": pr_s,
        "qt_ms": qt_s,
        "j_point_mv": stj_s,
        "st_j40_mv": st40_s,
        "st_j60_mv": st60_s,
        "st_j80_mv": st80_s,
        "st_direction": st_direction,
        "r_amplitude_mv": r_s,
        "s_amplitude_mv": s_s,
        "rs_ratio": rs_ratio,
        "q_amplitude_mv": q_s,
        "q_duration_ms": qd_s,
        "t_amplitude_mv": t_s,
        "t_polarity": t_polarity,
        "qrs_net_area_mv_s": area_s,
        "baseline_sources": {
            key: int(baselines.count(key))
            for key in sorted(set(baselines))
        },
        "beat_fiducials": beat_audit,
    }


def _weighted_global(
    per_lead: Mapping[str, Mapping[str, Any]],
    key: str,
    *,
    min_confidence: float = 0.35,
) -> Dict[str, Any]:
    vals: List[Tuple[float, float, str]] = []
    for lead, item in per_lead.items():
        metric = item.get(key) or {}
        value = metric.get("value") if isinstance(metric, Mapping) else None
        conf = metric.get("confidence") if isinstance(metric, Mapping) else None
        if value is None or conf is None:
            continue
        if float(conf) < min_confidence:
            continue
        vals.append((float(value), float(conf), lead))

    if not vals:
        return {
            "value": None,
            "confidence": 0.0,
            "source_leads": [],
            "reason": "NO_RELIABLE_DIGITAL_CONSENSUS",
        }

    values = np.asarray([v for v, _, _ in vals], dtype=float)
    confs = np.asarray([c for _, c, _ in vals], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    global_conf = _clip01(
        0.65 * float(np.median(confs))
        + 0.20 * min(1.0, len(vals) / 6.0)
        + 0.15 * max(0.0, 1.0 - mad / max(abs(median) * 0.20, 15.0))
    )
    return {
        "value": median,
        "confidence": global_conf,
        "source_leads": [lead for _, _, lead in vals],
        "source_n": len(vals),
        "mad": mad,
    }


def _rhythm_metrics(per_lead: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = []
    for lead, item in per_lead.items():
        if not item.get("evaluable"):
            continue
        rr = np.asarray(item.get("rr_ms") or [], dtype=float)
        if rr.size < 4:
            continue
        duration = float(item.get("duration_s") or 0.0)
        conf = float(item.get("signal_confidence") or 0.0)
        score = duration * 100.0 + rr.size * 10.0 + conf * 50.0 + (80.0 if lead == "II" else 0.0)
        candidates.append((score, lead, item))

    if not candidates:
        return {
            "evaluable": False,
            "reason": "NO_DIGITAL_LEAD_WITH_SUFFICIENT_RR",
        }

    _, lead, item = max(candidates, key=lambda x: x[0])
    rr = np.asarray(item.get("rr_ms") or [], dtype=float)
    mean = float(np.mean(rr))
    sd = float(np.std(rr, ddof=1)) if rr.size >= 2 else 0.0
    cv = float(sd / mean) if mean > 0 else None
    med = float(np.median(rr))
    mad = float(np.median(np.abs(rr - med)))
    mad_ratio = float(mad / med) if med > 0 else None
    rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if rr.size >= 2 else None
    pnn50 = float(np.mean(np.abs(np.diff(rr)) > 50.0)) if rr.size >= 2 else None

    regular = bool(
        cv is not None and mad_ratio is not None
        and cv <= 0.10
        and mad_ratio <= 0.075
    )
    marked_irregularity = bool(
        cv is not None and mad_ratio is not None
        and cv >= 0.12
        and mad_ratio >= 0.08
    )
    conf = _clip01(
        0.55 * float(item.get("signal_confidence") or 0.0)
        + 0.25 * min(1.0, rr.size / 10.0)
        + 0.20 * min(1.0, float(item.get("duration_s") or 0.0) / 8.0)
    )

    return {
        "evaluable": True,
        "source": "DIGITAL_RR_INTERVALS",
        "lead": lead,
        "duration_s": float(item.get("duration_s") or 0.0),
        "r_count": int(item.get("r_count") or 0),
        "rr_ms": [round(float(v), 3) for v in rr.tolist()],
        "rr_mean_ms": mean,
        "rr_sd_ms": sd,
        "rr_cv": cv,
        "rr_mad_ms": mad,
        "rr_mad_ratio": mad_ratio,
        "rr_rmssd_ms": rmssd,
        "pnn50": pnn50,
        "heart_rate_bpm": 60000.0 / mean if mean > 0 else None,
        "regular": regular,
        "marked_irregularity": marked_irregularity,
        "confidence": conf,
        "rule": "RR_CV_AND_MAD_RATIO",
    }


def _axis(per_lead: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    def net(lead: str) -> Optional[float]:
        item = per_lead.get(lead) or {}
        metric = item.get("qrs_net_area_mv_s") or {}
        if metric.get("value") is None or float(metric.get("confidence") or 0.0) < 0.35:
            return None
        return float(metric["value"])

    i = net("I")
    avf = net("aVF")
    if i is None or avf is None or (abs(i) < 1e-8 and abs(avf) < 1e-8):
        return {"evaluable": False, "degrees": None, "confidence": 0.0}
    deg = float(math.degrees(math.atan2(avf, i)))
    conf = min(
        float((per_lead["I"].get("qrs_net_area_mv_s") or {}).get("confidence") or 0.0),
        float((per_lead["aVF"].get("qrs_net_area_mv_s") or {}).get("confidence") or 0.0),
    )
    return {
        "evaluable": True,
        "degrees": deg,
        "confidence": conf,
        "qrs_net_I_mv_s": i,
        "qrs_net_aVF_mv_s": avf,
    }


def _r_progression(per_lead: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    vals = []
    for lead in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        item = per_lead.get(lead) or {}
        metric = item.get("r_amplitude_mv") or {}
        value = metric.get("value")
        conf = float(metric.get("confidence") or 0.0)
        vals.append((lead, None if value is None else float(value), conf))
    usable = [(lead, value, conf) for lead, value, conf in vals if value is not None and conf >= 0.35]
    if len(usable) < 4:
        return {
            "evaluable": False,
            "values_mv": {lead: value for lead, value, _ in vals},
            "reason": "INSUFFICIENT_PRECORDIAL_R_AMPLITUDES",
        }
    y = np.asarray([value for _, value, _ in usable], dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    conf = float(np.median([c for _, _, c in usable]))
    return {
        "evaluable": True,
        "values_mv": {lead: value for lead, value, _ in vals},
        "slope_mv_per_lead": slope,
        "confidence": conf,
        "pattern": (
            "INCREASING_R_WAVE_TREND"
            if slope > 0.03
            else "FLAT_OR_DECREASING_R_WAVE_TREND"
        ),
    }


def _voltages(per_lead: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    def val(lead: str, key: str) -> Optional[float]:
        metric = ((per_lead.get(lead) or {}).get(key) or {})
        if metric.get("value") is None or float(metric.get("confidence") or 0.0) < 0.35:
            return None
        return float(metric["value"])

    rv5 = val("V5", "r_amplitude_mv")
    rv6 = val("V6", "r_amplitude_mv")
    sv1 = val("V1", "s_amplitude_mv")
    ravl = val("aVL", "r_amplitude_mv")
    sv3 = val("V3", "s_amplitude_mv")

    sokolow = None
    if sv1 is not None and (rv5 is not None or rv6 is not None):
        sokolow = abs(sv1) + max(v for v in [rv5, rv6] if v is not None)
    cornell = None
    if ravl is not None and sv3 is not None:
        cornell = ravl + abs(sv3)
    return {
        "sokolow_lyon_mv": sokolow,
        "cornell_voltage_mv": cornell,
        "source": "DIGITAL_AMPLITUDES",
    }


def build_digital_measurements(
    digital_ecg: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measure ECG physiology exclusively from reconstructed digital signals."""
    calibration = dict(digital_ecg.get("calibration") or {})
    calibration_conf = float(calibration.get("confidence") or 0.0)
    per_lead: Dict[str, Dict[str, Any]] = {}
    for lead in LEADS:
        per_lead[lead] = _lead_measurements(
            lead,
            (digital_ecg.get("leads") or {}).get(lead) or {},
            calibration_confidence=calibration_conf,
        )

    rhythm = _rhythm_metrics(per_lead)
    qrs = _weighted_global(per_lead, "qrs_ms")
    p_dur = _weighted_global(per_lead, "p_duration_ms")
    pr = _weighted_global(per_lead, "pr_ms")
    qt = _weighted_global(per_lead, "qt_ms")
    axis = _axis(per_lead)

    rr_s = None
    if rhythm.get("rr_mean_ms"):
        rr_s = float(rhythm["rr_mean_ms"]) / 1000.0
    qtc_b = None
    qtc_f = None
    if qt.get("value") is not None and rr_s and rr_s > 0:
        qt_s = float(qt["value"]) / 1000.0
        qtc_b = qt_s / math.sqrt(rr_s) * 1000.0
        qtc_f = qt_s / (rr_s ** (1.0 / 3.0)) * 1000.0

    st_by_lead: Dict[str, Any] = {}
    t_by_lead: Dict[str, Any] = {}
    for lead, item in per_lead.items():
        st_by_lead[lead] = {
            "j_mv": (item.get("j_point_mv") or {}).get("value"),
            "j40_mv": (item.get("st_j40_mv") or {}).get("value"),
            "j60_mv": (item.get("st_j60_mv") or {}).get("value"),
            "j80_mv": (item.get("st_j80_mv") or {}).get("value"),
            "direction": item.get("st_direction"),
            "confidence": max(
                float((item.get("st_j60_mv") or {}).get("confidence") or 0.0),
                float((item.get("j_point_mv") or {}).get("confidence") or 0.0),
            ),
            "baseline_sources": item.get("baseline_sources") or {},
        }
        t_by_lead[lead] = {
            "amplitude_mv": (item.get("t_amplitude_mv") or {}).get("value"),
            "polarity": item.get("t_polarity"),
            "confidence": float((item.get("t_amplitude_mv") or {}).get("confidence") or 0.0),
        }

    morphology = {
        "r_progression": _r_progression(per_lead),
        "voltages": _voltages(per_lead),
        "qrs_prolonged": (
            None if qrs.get("value") is None
            else bool(float(qrs["value"]) >= 120.0)
        ),
        "pathologic_q_candidates": [
            lead for lead, item in per_lead.items()
            if (item.get("q_duration_ms") or {}).get("value") is not None
            and float((item.get("q_duration_ms") or {}).get("value") or 0.0) >= 40.0
            and (item.get("q_amplitude_mv") or {}).get("value") is not None
            and abs(float((item.get("q_amplitude_mv") or {}).get("value") or 0.0)) >= 0.10
            and min(
                float((item.get("q_duration_ms") or {}).get("confidence") or 0.0),
                float((item.get("q_amplitude_mv") or {}).get("confidence") or 0.0),
            ) >= 0.45
        ],
    }

    return {
        "schema": "MEDCALC_DIGITAL_MEASUREMENTS_V2",
        "source": "CANONICAL_DIGITAL_ECG_ONLY",
        "image_measurements_used": False,
        "fs": int(digital_ecg.get("fs") or 500),
        "calibration": calibration,
        "rhythm": rhythm,
        "global": {
            "heart_rate_bpm": {
                "value": rhythm.get("heart_rate_bpm"),
                "confidence": rhythm.get("confidence", 0.0),
            },
            "qrs_ms": qrs,
            "p_duration_ms": p_dur,
            "pr_ms": pr,
            "qt_ms": qt,
            "qtc_bazett_ms": {
                "value": qtc_b,
                "confidence": min(float(qt.get("confidence") or 0.0), float(rhythm.get("confidence") or 0.0)),
            },
            "qtc_fridericia_ms": {
                "value": qtc_f,
                "confidence": min(float(qt.get("confidence") or 0.0), float(rhythm.get("confidence") or 0.0)),
            },
            "axis_deg": axis,
        },
        "st_by_lead": st_by_lead,
        "t_by_lead": t_by_lead,
        "leads": per_lead,
        "morphology": morphology,
        "measurement_precedence": "NUMERIC_DIGITAL_SIGNAL_GT_CLASSIFIER",
        "fail_closed_below_confidence": 0.45,
    }
