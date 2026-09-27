from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy.signal import find_peaks


LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
QRS_MORPHOLOGY_VERSION = "MEDCALC_QRS_MORPHOLOGY_V2"


def _as_signal(item: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
        dtype=float,
    )


def _resample(seg: np.ndarray, n: int = 201) -> np.ndarray:
    seg = np.asarray(seg, dtype=float)
    if seg.size < 3:
        return np.full(n, np.nan, dtype=float)
    xp = np.linspace(0.0, 1.0, seg.size)
    xq = np.linspace(0.0, 1.0, n)
    return np.interp(xq, xp, seg)


def _median_qrs(canonical_item: Dict[str, Any], measured: Dict[str, Any]) -> tuple[np.ndarray | None, float | None]:
    fs = int(canonical_item.get("fs") or measured.get("fs") or 500)
    x = _as_signal(canonical_item)
    waves = []
    durations = []
    for beat in measured.get("beats") or []:
        try:
            a = int(beat["qrs_onset_sample"])
            b = int(beat["qrs_offset_sample"])
            baseline = float(beat.get("baseline_mv") or 0.0)
            quality = float(beat.get("beat_quality") or 0.0)
        except Exception:
            continue
        if quality < 0.55 or a < 0 or b <= a or b >= len(x):
            continue
        seg = np.asarray(x[a:b + 1], dtype=float) - baseline
        if seg.size < 8 or not np.isfinite(seg).all():
            continue
        waves.append(_resample(seg))
        durations.append((b - a) * 1000.0 / fs)
    if len(waves) < 2:
        return None, None
    return np.median(np.vstack(waves), axis=0), float(np.median(durations))


def _crossing_duration_ms(w: np.ndarray, duration_ms: float, polarity: str) -> float | None:
    if w.size < 5:
        return None
    if polarity == "negative":
        mask = w < -0.02
    else:
        mask = w > 0.02
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    last = int(idx[-1])
    start = last
    while start > 0 and mask[start - 1]:
        start -= 1
    return float((last - start + 1) / max(len(w) - 1, 1) * duration_ms)


def _describe(w: np.ndarray, duration_ms: float) -> Dict[str, Any]:
    amp = float(np.nanmax(w) - np.nanmin(w))
    prominence = max(0.025, 0.08 * amp)
    pos, _ = find_peaks(w, prominence=prominence, distance=max(3, len(w)//12))
    neg, _ = find_peaks(-w, prominence=prominence, distance=max(3, len(w)//12))
    max_i = int(np.nanargmax(w))
    min_i = int(np.nanargmin(w))
    r_peak_time_ms = max_i / max(len(w)-1,1) * duration_ms

    initial_q = bool(np.nanmin(w[: max(5, len(w)//4)]) <= -0.03)
    terminal_pos = float(np.nanmax(w[int(0.60*len(w)):]))
    terminal_neg = float(np.nanmin(w[int(0.60*len(w)):]))
    r_prime = bool(
        len(pos) >= 2
        and int(pos[-1]) >= int(0.52 * len(w))
        and any(int(n) > int(pos[-2]) and int(n) < int(pos[-1]) for n in neg)
    )
    notch = bool(len(pos) >= 2 and abs(float(w[pos[-1]] - w[pos[-2]])) <= max(0.35*amp,0.08))
    s_terminal_ms = _crossing_duration_ms(w, duration_ms, "negative")
    terminal_r_ms = _crossing_duration_ms(w, duration_ms, "positive")

    qrs_polarity = (
        "R_DOMINANT"
        if abs(float(np.nanmax(w))) >= 1.15 * abs(float(np.nanmin(w)))
        else "S_DOMINANT"
        if abs(float(np.nanmin(w))) >= 1.15 * abs(float(np.nanmax(w)))
        else "BIPHASIC"
    )

    initial_n = max(4, int(round(len(w) * min(0.35, 40.0/max(duration_ms,1.0)))))
    initial = w[:initial_n]
    full_slope = np.nanmax(np.abs(np.diff(w))) if len(w) > 2 else 0.0
    init_slope = np.nanmax(np.abs(np.diff(initial))) if len(initial) > 2 else 0.0
    slope_ratio = float(init_slope / max(full_slope, 1e-9))
    delta_slur = bool(duration_ms >= 110.0 and slope_ratio < 0.55)

    return {
        "duration_ms": round(duration_ms, 3),
        "r_peak_time_ms": round(r_peak_time_ms, 3),
        "positive_peak_n": int(len(pos)),
        "negative_peak_n": int(len(neg)),
        "r_prime_present": r_prime,
        "notched_or_double_r": notch,
        "initial_q_present": initial_q,
        "terminal_positive_mv": round(terminal_pos, 6),
        "terminal_negative_mv": round(terminal_neg, 6),
        "terminal_s_duration_ms": round(s_terminal_ms, 3) if s_terminal_ms is not None else None,
        "terminal_r_duration_ms": round(terminal_r_ms, 3) if terminal_r_ms is not None else None,
        "qrs_polarity": qrs_polarity,
        "initial_slope_ratio": round(slope_ratio, 6),
        "delta_slur_compatible": delta_slur,
    }


def analyze_qrs_morphology(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
    global_metrics: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    leads = canonical_ecg.get("leads") or {}
    for lead in LEADS:
        w, dur = _median_qrs(dict(leads.get(lead) or {}), dict(per_lead.get(lead) or {}))
        if w is None or dur is None:
            rows[lead] = {"evaluable": False, "reason": "LT_2_HIGH_QUALITY_QRS_BEATS"}
            continue
        row = _describe(w, dur)
        row["evaluable"] = True
        row["source"] = "MEDIAN_DIGITAL_QRS"
        rows[lead] = row

    qrs = ((global_metrics.get("qrs_ms") or {}).get("value"))
    try:
        qrs_ms = float(qrs) if qrs is not None else None
    except Exception:
        qrs_ms = None

    return {
        "version": QRS_MORPHOLOGY_VERSION,
        "qrs_ms": qrs_ms,
        "per_lead": rows,
        "source": "CALIBRATED_DIGITAL_SIGNAL_MEDIAN_BEATS",
        "diagnostic_claim_allowed": False,
    }
