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


def _interval_consensus(
    values: List[float],
    *,
    max_dispersion_ms: float,
    min_sources: int = 2,
) -> tuple[float | None, Dict[str, Any]]:
    """Return a conservative multilead interval consensus.

    Raster ECG delineation can produce a plausible-looking value from a single
    poorly reconstructed lead.  Require agreement across leads before exposing
    PR/QRS/QT as a motor measurement.
    """
    z = np.asarray(values, dtype=float)
    z = z[np.isfinite(z)]
    if z.size < int(min_sources):
        return None, {
            "source_n": int(z.size),
            "reportable": False,
            "reason": "INSUFFICIENT_MULTILEAD_CONSENSUS",
        }

    median = float(np.median(z))
    if z.size >= 4:
        dispersion = float(np.percentile(z, 75) - np.percentile(z, 25))
        dispersion_name = "IQR"
    else:
        dispersion = float(np.max(z) - np.min(z))
        dispersion_name = "RANGE"

    reportable = bool(dispersion <= float(max_dispersion_ms))
    return (
        median if reportable else None,
        {
            "source_n": int(z.size),
            "median_ms": median,
            "dispersion_ms": dispersion,
            "dispersion_metric": dispersion_name,
            "max_dispersion_ms": float(max_dispersion_ms),
            "reportable": reportable,
            "reason": None if reportable else "INTERLEAD_DISAGREEMENT",
        },
    )


def _qt_interval_is_technically_valid(qt_ms: float | None, rr_s: float | None) -> bool:
    """Technical delineation gate, not a clinical long-QT criterion.

    A T offset that reaches essentially the next QRS is a common raster/DWT
    failure during tachycardia.  Reject those measurements instead of turning a
    bad T-end into an extreme QTc.
    """
    if qt_ms is None or rr_s is None:
        return False
    try:
        qt = float(qt_ms)
        rr_ms = float(rr_s) * 1000.0
    except Exception:
        return False
    if not (math.isfinite(qt) and math.isfinite(rr_ms) and rr_ms > 0):
        return False
    return bool(120.0 <= qt <= 800.0 and qt < 0.90 * rr_ms)


def _rhythm_metrics(signal_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    """Measure rhythm from the best *working* lead, not merely the longest lead.

    Digitized paper ECGs can reconstruct one lead poorly even when it has the
    longest visible span. The previous implementation strongly preferred II
    and stopped if NeuroKit failed there, which caused FC/PR/QRS/QT to remain
    empty while axis measurements from other leads were still available.

    This implementation evaluates every lead with >=2 s of contiguous signal,
    tests both polarities, and selects the candidate with the strongest valid
    QRS detection. Interval measurements are then pooled across every successful
    candidate lead when possible.
    """
    pref = ["II", "I", "V5", "V1", "V6", "III", "aVF", "aVL", "aVR", "V2", "V3", "V4"]

    candidates: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for pref_rank, lead in enumerate(pref):
        idx = LEADS.index(lead)
        span = _longest_span(signal_mv[:, idx])
        if span is None:
            failures.append({"lead": lead, "reason": "sin segmento finito"})
            continue

        a, b = span
        duration_s = (b - a) / fs
        if duration_s < 2.0:
            failures.append({
                "lead": lead,
                "duration_s": float(duration_s),
                "reason": "segmento <2 s",
            })
            continue

        raw = np.asarray(signal_mv[a:b, idx], dtype=float)

        # Try native and inverted polarity. Paper digitisation can occasionally
        # invert a reconstructed row; R-peak timing should not disappear solely
        # because of that orientation error.
        best_local = None
        for polarity in (1.0, -1.0):
            x = raw * polarity
            try:
                nk = _nk_delineation(x, fs)
            except Exception as exc:
                failures.append({
                    "lead": lead,
                    "polarity": int(polarity),
                    "reason": f"NeuroKit: {exc}",
                })
                continue

            r = np.asarray(nk.get("r", []), dtype=int)
            if len(r) < 3:
                failures.append({
                    "lead": lead,
                    "polarity": int(polarity),
                    "duration_s": float(duration_s),
                    "r_count": int(len(r)),
                    "reason": "<3 QRS",
                })
                continue

            rr = np.diff(r) / fs
            if rr.size == 0 or not np.isfinite(rr).all() or float(np.median(rr)) <= 0:
                continue

            med_rr = float(np.median(rr))
            hr = 60.0 / med_rr
            if not 20.0 <= hr <= 320.0:
                failures.append({
                    "lead": lead,
                    "polarity": int(polarity),
                    "heart_rate_bpm": float(hr),
                    "reason": "FC fuera de rango técnico",
                })
                continue

            cv = (
                float(np.std(rr, ddof=1) / np.mean(rr))
                if len(rr) >= 2 and np.mean(rr) > 0
                else None
            )

            # Prefer longer segments, more beats, lead II when equally valid,
            # and avoid extremely implausible RR dispersion caused by bad rows.
            rr_penalty = 0.0
            if cv is not None and cv > 0.80:
                rr_penalty = 1000.0
            score = (
                float(duration_s) * 100.0
                + float(len(r)) * 10.0
                + (80.0 if lead == "II" else 0.0)
                - float(pref_rank)
                - rr_penalty
            )

            item = {
                "score": float(score),
                "lead": lead,
                "idx": idx,
                "span": (int(a), int(b)),
                "duration_s": float(duration_s),
                "polarity": int(polarity),
                "x": x,
                "nk": nk,
                "r": r,
                "rr": rr,
                "med_rr": med_rr,
                "heart_rate_bpm": float(hr),
                "rr_cv": cv,
            }
            if best_local is None or item["score"] > best_local["score"]:
                best_local = item

        if best_local is not None:
            candidates.append(best_local)

    if not candidates:
        return {
            "lead": None,
            "evaluable": False,
            "reason": "Ninguna derivación permitió detectar ≥3 complejos QRS de forma robusta.",
            "candidate_failures": failures[-24:],
        }

    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    lead = best["lead"]
    idx = best["idx"]
    a, b = best["span"]
    duration_s = best["duration_s"]
    x = best["x"]
    nk = best["nk"]
    r = best["r"]
    rr = best["rr"]
    med_rr = best["med_rr"]
    hr = best["heart_rate_bpm"]
    cv = best["rr_cv"]
    regular = bool(cv is not None and cv <= 0.10)

    # Pool interval estimates across all successful leads. This prevents one
    # imperfect reconstructed row from suppressing every motor measurement.
    pr_values: List[float] = []
    qrs_values: List[float] = []
    qt_values: List[float] = []
    p_ratio_values: List[float] = []
    p_positive_values: List[bool] = []
    interval_sources: List[Dict[str, Any]] = []

    for cand in candidates:
        waves = cand["nk"].get("waves", {}) or {}
        cr = cand["r"]
        cr_on = _arr(waves, "ECG_R_Onsets")
        cr_off = _arr(waves, "ECG_R_Offsets")
        cp_on = _arr(waves, "ECG_P_Onsets")
        cp_peaks = _arr(waves, "ECG_P_Peaks")
        ct_off = _arr(waves, "ECG_T_Offsets")

        pr_samples = _nearest_preceding(
            cp_on,
            cr_on if cr_on.size else cr.astype(float),
            low=int(0.06 * fs),
            high=int(0.40 * fs),
        )
        pr = _median_ms(pr_samples, fs)

        qrs_samples = _pairwise_duration(
            cr_on,
            cr_off,
            max_gap=int(0.22 * fs),
        )
        qrs = _median_ms(qrs_samples, fs)

        qt_samples = _pairwise_duration(
            cr_on,
            ct_off,
            max_gap=int(0.80 * fs),
        )
        qt = _median_ms(qt_samples, fs)

        r_for_p = cr_on if cr_on.size else cr.astype(float)
        p_before = _nearest_preceding(
            cp_peaks,
            r_for_p,
            low=int(0.06 * fs),
            high=int(0.35 * fs),
        )
        p_ratio = float(len(p_before) / max(1, len(r_for_p)))

        # PR is not reportable when the same delineation does not recover
        # reproducible P waves before the QRS.
        if pr is not None and 40.0 <= pr <= 400.0 and p_ratio >= 0.50:
            pr_values.append(float(pr))
        if qrs is not None and 30.0 <= qrs <= 240.0:
            qrs_values.append(float(qrs))
        if _qt_interval_is_technically_valid(qt, cand.get("med_rr")):
            qt_values.append(float(qt))
        p_ratio_values.append(p_ratio)

        p_positive = None
        if cp_peaks.size and cand["lead"] == "II":
            baseline = float(np.nanmedian(cand["x"]))
            valid = [int(v) for v in cp_peaks if 0 <= int(v) < len(cand["x"])]
            if valid:
                amp = np.asarray(
                    [cand["x"][v] - baseline for v in valid],
                    dtype=float,
                )
                p_positive = bool(np.nanmedian(amp) > 0)
                p_positive_values.append(p_positive)

        interval_sources.append({
            "lead": cand["lead"],
            "duration_s": cand["duration_s"],
            "polarity": cand["polarity"],
            "r_count": int(len(cr)),
            "heart_rate_bpm": cand["heart_rate_bpm"],
            "rr_cv": cand["rr_cv"],
            "pr_ms": pr,
            "qrs_ms": qrs,
            "qt_ms": qt,
            "p_before_qrs_ratio": p_ratio,
            "p_positive_in_ii": p_positive,
        })

    pr_ms, pr_quality = _interval_consensus(
        pr_values,
        max_dispersion_ms=50.0,
        min_sources=2,
    )
    qrs_ms, qrs_quality = _interval_consensus(
        qrs_values,
        max_dispersion_ms=40.0,
        min_sources=2,
    )
    qt_ms, qt_quality = _interval_consensus(
        qt_values,
        max_dispersion_ms=60.0,
        min_sources=2,
    )

    # Final cycle-length sanity check protects against a consensus of equally
    # bad T-end detections during fast rhythms.
    if qt_ms is not None and not _qt_interval_is_technically_valid(qt_ms, med_rr):
        qt_ms = None
        qt_quality = dict(qt_quality)
        qt_quality.update({
            "reportable": False,
            "reason": "QT_APPROACHES_NEXT_QRS",
            "rr_ms": float(med_rr * 1000.0),
        })

    qtc_bazett_ms = (
        float(qt_ms / math.sqrt(med_rr))
        if qt_ms is not None and med_rr > 0
        else None
    )

    p_ratio = (
        float(np.median(p_ratio_values))
        if p_ratio_values
        else 0.0
    )
    p_positive = (
        bool(sum(1 for v in p_positive_values if v) >= (len(p_positive_values) / 2))
        if p_positive_values
        else None
    )

    # If P waves are not reproducible, a numerical PR is not exposed even if
    # a delineator emitted candidate onsets.
    if p_ratio < 0.50:
        pr_ms = None
        pr_quality = dict(pr_quality)
        pr_quality.update({
            "reportable": False,
            "reason": "P_WAVES_NOT_REPRODUCIBLE",
            "p_before_qrs_ratio": float(p_ratio),
        })

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
        "measurement_method": "MULTILEAD_NEUROKIT_DWT_WITH_POLARITY_RETRY",
        "candidate_leads_evaluable": int(len(candidates)),
        "duration_s": float(duration_s),
        "r_count": int(len(r)),
        "heart_rate_bpm": float(hr),
        "rr_cv": cv,
        "regular": regular,
        "sinus_compatible": sinus_compatible,
        "p_before_qrs_ratio": p_ratio,
        "p_positive_in_ii": p_positive,
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_bazett_ms": qtc_bazett_ms,
        "interval_quality": {
            "pr": pr_quality,
            "qrs": qrs_quality,
            "qt": qt_quality,
            "qt_rr_fraction": (
                float((qt_ms / 1000.0) / med_rr)
                if qt_ms is not None and med_rr > 0
                else None
            ),
        },
        "premature_pattern_count": int(premature),
        "r_peaks_local": [int(v) for v in r.tolist()],
        "span_start": int(a),
        "span_end": int(b),
        "interval_sources": interval_sources,
        "candidate_failures": failures[-24:],
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
    vals: Dict[str, float] = {}
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF"):
        value = _lead_qrs_net(signal_mv, fs, lead)
        if value is not None and math.isfinite(float(value)):
            vals[lead] = float(value)

    if "I" not in vals or "aVF" not in vals:
        return {
            "evaluable": False,
            "degrees": None,
            "category": "NO EVALUABLE",
            "reason": "I_OR_AVF_NOT_RELIABLE",
        }

    lead_i = vals["I"]
    lead_avf = vals["aVF"]
    if abs(lead_i) < 1e-8 and abs(lead_avf) < 1e-8:
        return {
            "evaluable": False,
            "degrees": None,
            "category": "NO EVALUABLE",
            "reason": "LIMB_QRS_TOO_SMALL",
        }

    scale = float(np.median([abs(v) for v in vals.values()])) if vals else 0.0
    residuals: List[float] = []
    if scale > 1e-9 and all(k in vals for k in ("I", "II", "III")):
        residuals.append(abs(vals["II"] - (vals["I"] + vals["III"])) / scale)
    if scale > 1e-9 and all(k in vals for k in ("aVR", "aVL", "aVF")):
        residuals.append(abs(vals["aVR"] + vals["aVL"] + vals["aVF"]) / scale)

    consistency = float(np.median(residuals)) if residuals else None
    if consistency is not None and consistency > 0.55:
        return {
            "evaluable": False,
            "degrees": None,
            "category": "NO EVALUABLE",
            "reason": "LIMB_LEAD_VECTOR_INCONSISTENCY",
            "consistency": consistency,
            "limb_qrs_net": vals,
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
        "consistency": consistency,
        "limb_qrs_net": vals,
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


def _rhythm_screen(rhythm: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based rhythm screen from the longest observed rhythm strip.

    This is deliberately separate from frozen R27. It may describe a pattern
    compatible with AF/SVT/sinus tachycardia, but it does not substitute
    missing R27 features or create a binary R27 diagnosis.
    """
    if not rhythm.get("evaluable"):
        return {
            "evaluable": False,
            "code": "NOT_EVALUABLE",
            "label": "RITMO NO EVALUABLE",
            "basis": [],
        }

    hr = rhythm.get("heart_rate_bpm")
    qrs = rhythm.get("qrs_ms")
    rr_cv = rhythm.get("rr_cv")
    regular = bool(rhythm.get("regular"))
    sinus = bool(rhythm.get("sinus_compatible"))
    p_ratio = rhythm.get("p_before_qrs_ratio")

    basis: List[str] = []
    if hr is not None:
        basis.append(f"FC MOTOR {float(hr):.0f} LPM")
    if rr_cv is not None:
        basis.append(f"RR CV {float(rr_cv):.3f}")
    if qrs is not None:
        basis.append(f"QRS MOTOR {float(qrs):.0f} MS")
    if p_ratio is not None:
        basis.append(f"P/QRS {float(p_ratio):.2f}")

    tachy = bool(hr is not None and float(hr) >= 100.0)
    narrow = bool(qrs is not None and float(qrs) < 120.0)
    p_poor = bool(p_ratio is None or float(p_ratio) < 0.50)
    irregular_marked = bool(rr_cv is not None and float(rr_cv) >= 0.12)

    if tachy and narrow and irregular_marked and p_poor:
        return {
            "evaluable": True,
            "code": "AF_COMPATIBLE",
            "label": "PATRÓN COMPATIBLE CON FIBRILACIÓN AURICULAR CON RESPUESTA VENTRICULAR RÁPIDA",
            "basis": basis,
        }

    if tachy and narrow and regular and sinus:
        return {
            "evaluable": True,
            "code": "SINUS_TACHY_COMPATIBLE",
            "label": "PATRÓN COMPATIBLE CON TAQUICARDIA SINUSAL",
            "basis": basis,
        }

    if tachy and narrow and regular and p_poor:
        label = "PATRÓN COMPATIBLE CON TAQUICARDIA SUPRAVENTRICULAR REGULAR DE QRS ESTRECHO"
        if hr is not None and 130 <= float(hr) <= 180:
            label += "; FLUTTER AURICULAR 2:1 NO EXCLUIDO"
        return {
            "evaluable": True,
            "code": "SVT_COMPATIBLE",
            "label": label,
            "basis": basis,
        }

    if tachy and narrow:
        return {
            "evaluable": True,
            "code": "NARROW_TACHY_UNCLASSIFIED",
            "label": "TAQUICARDIA DE QRS ESTRECHO NO CLASIFICADA POR EL SCREENING DE RITMO",
            "basis": basis,
        }

    if sinus and regular:
        return {
            "evaluable": True,
            "code": "SINUS_COMPATIBLE",
            "label": "PATRÓN COMPATIBLE CON RITMO SINUSAL REGULAR",
            "basis": basis,
        }

    return {
        "evaluable": True,
        "code": "RHYTHM_UNCLASSIFIED",
        "label": "RITMO NO CLASIFICADO POR EL SCREENING AUTOMATIZADO",
        "basis": basis,
    }


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
    if st_ok and t_ok:
        conclusion_parts.append("SIN CAMBIOS AGUDOS ST-T EVIDENTES EN LOS SEGMENTOS EVALUABLES")
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


def _lead_evidence(signal_mv: np.ndarray, fs: int) -> Dict[str, Any]:
    """Return compact per-lead waveform evidence for UI/PDF rendering."""
    evidence: Dict[str, Any] = {}

    for lead in LEADS:
        idx = LEADS.index(lead)
        span = _longest_span(signal_mv[:, idx])
        if span is None:
            evidence[lead] = {
                "evaluable": False,
                "reason": "sin segmento finito",
            }
            continue

        a, b = span
        x = np.asarray(signal_mv[a:b, idx], dtype=float)
        duration_s = float((b - a) / fs)
        if x.size < 2:
            evidence[lead] = {
                "evaluable": False,
                "reason": "segmento insuficiente",
                "duration_s": duration_s,
            }
            continue

        # Full observed segment preview, resampled to a compact fixed grid.
        n_trace = 360
        src_t = np.linspace(0.0, duration_s, x.size, endpoint=False)
        dst_t = np.linspace(0.0, duration_s, n_trace, endpoint=False)
        finite = np.isfinite(x)
        if int(finite.sum()) >= 2:
            trace = np.interp(dst_t, src_t[finite], x[finite])
        else:
            trace = np.full(n_trace, np.nan)

        complex_values = None
        complex_time = None
        complex_method = None
        r_local = None

        try:
            nk = _nk_delineation(x, fs)
            r = np.asarray(nk.get("r", []), dtype=int)
            if len(r):
                rp = int(r[len(r) // 2])
                left = int(round(0.24 * fs))
                right = int(round(0.46 * fs))
                c0 = max(0, rp - left)
                c1 = min(len(x), rp + right)
                if c1 - c0 >= int(0.35 * fs):
                    w = x[c0:c1]
                    t = (np.arange(c0, c1) - rp) / fs
                    n_complex = 280
                    dst = np.linspace(float(t[0]), float(t[-1]), n_complex)
                    wf = np.isfinite(w)
                    if int(wf.sum()) >= 2:
                        complex_values = np.interp(dst, t[wf], w[wf])
                        complex_time = dst
                        complex_method = "NEUROKIT_MEDIAN_R_PEAK"
                        r_local = int(rp)
        except Exception:
            pass

        if complex_values is None:
            # Evidence fallback only: center a 700 ms window on the largest
            # absolute deflection in the finite observed segment.
            xf = np.nan_to_num(x - np.nanmedian(x), nan=0.0)
            rp = int(np.argmax(np.abs(xf)))
            left = int(round(0.24 * fs))
            right = int(round(0.46 * fs))
            c0 = max(0, rp - left)
            c1 = min(len(x), rp + right)
            w = x[c0:c1]
            if len(w) >= 2 and np.isfinite(w).sum() >= 2:
                t = (np.arange(c0, c1) - rp) / fs
                n_complex = 280
                dst = np.linspace(float(t[0]), float(t[-1]), n_complex)
                wf = np.isfinite(w)
                complex_values = np.interp(dst, t[wf], w[wf])
                complex_time = dst
                complex_method = "MAX_ABSOLUTE_DEFLECTION_FALLBACK"
                r_local = int(rp)

        evidence[lead] = {
            "evaluable": True,
            "source_start_sample": int(a),
            "source_end_sample": int(b),
            "duration_s": duration_s,
            "trace_time_s": [round(float(v), 6) for v in dst_t.tolist()],
            "trace_mv": [
                None if not math.isfinite(float(v)) else round(float(v), 5)
                for v in trace.tolist()
            ],
            "representative_complex_time_s": (
                [round(float(v), 6) for v in complex_time.tolist()]
                if complex_time is not None else None
            ),
            "representative_complex_mv": (
                [round(float(v), 5) for v in complex_values.tolist()]
                if complex_values is not None else None
            ),
            "representative_complex_method": complex_method,
            "representative_center_sample_local": r_local,
        }

    return evidence


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
    rhythm_screen = _rhythm_screen(rhythm)
    axis = _axis_metrics(signal_mv, fs)
    repol = _repolarization_metrics(signal_mv, fs)
    formatted = _format_report(rhythm, axis, repol)
    evidence_by_lead = _lead_evidence(signal_mv, fs)

    measurement_summary = {
        "heart_rate_bpm": rhythm.get("heart_rate_bpm"),
        "rr_cv": rhythm.get("rr_cv"),
        "beat_n": rhythm.get("r_count"),
        "pr_ms": rhythm.get("pr_ms"),
        "qrs_ms": rhythm.get("qrs_ms"),
        "qt_ms": rhythm.get("qt_ms"),
        "qtc_bazett_ms": rhythm.get("qtc_bazett_ms"),
        "axis_deg": axis.get("degrees"),
        "p_before_qrs_ratio": rhythm.get("p_before_qrs_ratio"),
        "premature_pattern_count": rhythm.get("premature_pattern_count"),
        "interval_quality": rhythm.get("interval_quality"),
        "axis_consistency": axis.get("consistency"),
        "st_abnormal_leads": repol.get("st_abnormal_leads"),
        "t_unexpected_polarity_leads": repol.get("t_unexpected_polarity_leads"),
    }

    return {
        "version": "ECG_STRUCTURED_REPORT_V1",
        "source": "digitized_signal_only",
        "diagnostic_model": False,
        "sampling_rate_hz": int(fs),
        "rhythm": rhythm,
        "rhythm_screen": rhythm_screen,
        "axis": axis,
        "repolarization": repol,
        "measurement_summary": measurement_summary,
        "evidence_by_lead": evidence_by_lead,
        "formatted": formatted,
        "limitations": [
            "Reporte descriptivo automatizado derivado de la señal reconstruida desde foto/PDF.",
            "No convierte probabilidades R27 en diagnósticos ni usa thresholds no validados.",
            "Los campos no demostrables se informan como NO EVALUABLE.",
        ],
    }
