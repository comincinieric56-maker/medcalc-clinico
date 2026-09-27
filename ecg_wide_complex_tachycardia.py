from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np
from scipy.signal import find_peaks


WCT_ANALYZER_VERSION = "MEDCALC_WIDE_COMPLEX_TACHYCARDIA_V1"
PRECORDIAL_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6")
LIMB_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF")


def _clip01(v: float) -> float:
    return float(np.clip(float(v), 0.0, 1.0))


def _finite_float(value: Any) -> float | None:
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _as_signal(item: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
        dtype=float,
    )


def _resample_vector(x: np.ndarray, n: int = 120) -> np.ndarray | None:
    y = np.asarray(x, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 5:
        return None
    if not finite.all():
        idx = np.arange(len(y), dtype=float)
        y[~finite] = np.interp(idx[~finite], idx[finite], y[finite])
    if len(y) < 2:
        return None
    src = np.linspace(0.0, 1.0, len(y))
    dst = np.linspace(0.0, 1.0, int(n))
    out = np.interp(dst, src, y)
    out = out - float(np.mean(out))
    scale = float(np.linalg.norm(out))
    if scale <= 1e-9:
        return None
    return out / scale


def _representative_beat(
    canonical_item: Dict[str, Any],
    measurement_item: Dict[str, Any],
) -> tuple[np.ndarray, Dict[str, Any]] | None:
    signal = _as_signal(canonical_item)
    beats = list(measurement_item.get("beats") or [])
    if not beats or signal.size == 0:
        return None

    candidates = []
    for beat in beats:
        q_on = beat.get("qrs_onset_sample")
        q_off = beat.get("qrs_offset_sample")
        baseline = _finite_float(beat.get("baseline_mv"))
        qrs_ms = _finite_float(beat.get("qrs_ms"))
        quality = _finite_float(beat.get("beat_quality")) or 0.0
        if q_on is None or q_off is None or baseline is None or qrs_ms is None:
            continue
        q_on = int(q_on)
        q_off = int(q_off)
        if q_on < 0 or q_off >= len(signal) or q_off <= q_on:
            continue
        seg = signal[q_on:q_off + 1] - baseline
        if np.isfinite(seg).sum() < max(5, int(0.75 * len(seg))):
            continue
        candidates.append((float(qrs_ms), float(quality), seg, beat))

    if not candidates:
        return None

    median_qrs = float(np.median([row[0] for row in candidates]))
    candidates.sort(key=lambda row: (abs(row[0] - median_qrs), -row[1]))
    _, _, seg, beat = candidates[0]
    return np.asarray(seg, dtype=float), dict(beat)


def _lead_morphology(
    lead: str,
    canonical_item: Dict[str, Any],
    measurement_item: Dict[str, Any],
    fs: int,
) -> Dict[str, Any]:
    rep = _representative_beat(canonical_item, measurement_item)
    if rep is None:
        return {
            "lead": lead,
            "evaluable": False,
            "reason": "NO_REPRESENTATIVE_QRS",
        }

    seg, beat = rep
    finite = np.isfinite(seg)
    if finite.sum() < 5:
        return {
            "lead": lead,
            "evaluable": False,
            "reason": "INSUFFICIENT_QRS_SAMPLES",
        }
    if not finite.all():
        idx = np.arange(len(seg), dtype=float)
        seg[~finite] = np.interp(idx[~finite], idx[finite], seg[finite])

    # Mild smoothing only for morphology timing; amplitudes remain calibrated mV.
    if len(seg) >= 5:
        kernel = np.ones(3, dtype=float) / 3.0
        smooth = np.convolve(seg, kernel, mode="same")
    else:
        smooth = seg.copy()

    positive_peak = float(np.max(smooth))
    negative_peak = float(np.min(smooth))
    positive_mag = max(positive_peak, 0.0)
    negative_mag = max(-negative_peak, 0.0)
    total_mag = max(positive_mag, negative_mag, 1e-9)
    amplitude_pp = positive_mag + negative_mag

    polarity_ratio = (positive_mag - negative_mag) / max(amplitude_pp, 1e-9)
    if polarity_ratio >= 0.25:
        dominant_polarity = "POSITIVE"
    elif polarity_ratio <= -0.25:
        dominant_polarity = "NEGATIVE"
    else:
        dominant_polarity = "BIPHASIC"

    monophasic_positive = bool(
        positive_mag >= 0.08
        and negative_mag <= max(0.025, 0.20 * positive_mag)
    )
    monophasic_negative = bool(
        negative_mag >= 0.08
        and positive_mag <= max(0.025, 0.20 * negative_mag)
    )

    qrs_ms = _finite_float(beat.get("qrs_ms"))
    sample_ms = 1000.0 / float(fs)

    # First major deflection peak from QRS onset.
    prominence = max(0.015, 0.12 * total_mag)
    abs_seg = np.abs(smooth)
    peaks, props = find_peaks(
        abs_seg,
        prominence=prominence,
        distance=max(1, int(round(0.012 * fs))),
    )
    first_peak_i = int(peaks[0]) if len(peaks) else int(np.argmax(abs_seg))
    first_peak_ms = float(first_peak_i * sample_ms)

    # Brugada-style RS morphology: an R (positive peak) followed by S nadir.
    pos_peaks, _ = find_peaks(
        smooth,
        prominence=max(0.012, 0.10 * total_mag),
        distance=max(1, int(round(0.010 * fs))),
    )
    neg_peaks, _ = find_peaks(
        -smooth,
        prominence=max(0.012, 0.10 * total_mag),
        distance=max(1, int(round(0.010 * fs))),
    )
    rs_present = False
    rs_interval_ms = None
    r_index = None
    s_index = None
    for rp in pos_peaks:
        later_s = neg_peaks[neg_peaks > rp]
        if later_s.size == 0:
            continue
        sp = int(later_s[0])
        if smooth[int(rp)] >= 0.04 and smooth[sp] <= -0.04:
            rs_present = True
            r_index = int(rp)
            s_index = sp
            rs_interval_ms = float((sp - int(rp)) * sample_ms)
            break

    # Terminal polarity supports coarse BBB morphology checks.
    terminal_n = max(2, int(round(0.040 * fs)))
    terminal = smooth[-terminal_n:] if len(smooth) >= terminal_n else smooth
    terminal_mean = float(np.mean(terminal))
    terminal_polarity = (
        "POSITIVE"
        if terminal_mean > 0.015
        else "NEGATIVE"
        if terminal_mean < -0.015
        else "NEUTRAL"
    )

    initial_n = max(2, int(round(0.040 * fs)))
    initial = smooth[:initial_n] if len(smooth) >= initial_n else smooth
    initial_mean = float(np.mean(initial))

    return {
        "lead": lead,
        "evaluable": True,
        "qrs_ms": qrs_ms,
        "positive_peak_mv": round(positive_peak, 6),
        "negative_peak_mv": round(negative_peak, 6),
        "peak_to_peak_mv": round(amplitude_pp, 6),
        "dominant_polarity": dominant_polarity,
        "polarity_ratio": round(float(polarity_ratio), 6),
        "monophasic_positive": monophasic_positive,
        "monophasic_negative": monophasic_negative,
        "first_major_peak_ms": round(first_peak_ms, 3),
        "rs_present": rs_present,
        "rs_interval_ms": (
            round(float(rs_interval_ms), 3)
            if rs_interval_ms is not None else None
        ),
        "r_index": r_index,
        "s_index": s_index,
        "initial_40ms_mean_mv": round(initial_mean, 6),
        "terminal_40ms_mean_mv": round(terminal_mean, 6),
        "terminal_polarity": terminal_polarity,
        "beat_quality": _finite_float(beat.get("beat_quality")),
        "fiducial_source": beat.get("fiducial_source"),
        "source": "CALIBRATED_DIGITAL_SIGNAL",
    }


def _rhythm_qrs_morphology_stability(
    canonical_item: Dict[str, Any],
    measurement_item: Dict[str, Any],
) -> Dict[str, Any]:
    signal = _as_signal(canonical_item)
    beats = list(measurement_item.get("beats") or [])
    vectors = []
    widths = []
    for beat in beats:
        q_on = beat.get("qrs_onset_sample")
        q_off = beat.get("qrs_offset_sample")
        baseline = _finite_float(beat.get("baseline_mv"))
        qrs_ms = _finite_float(beat.get("qrs_ms"))
        if q_on is None or q_off is None or baseline is None or qrs_ms is None:
            continue
        q_on = int(q_on)
        q_off = int(q_off)
        if q_on < 0 or q_off >= len(signal) or q_off <= q_on:
            continue
        seg = signal[q_on:q_off + 1] - baseline
        vec = _resample_vector(seg)
        if vec is None:
            continue
        vectors.append(vec)
        widths.append(float(qrs_ms))

    if len(vectors) < 3:
        return {
            "evaluable": False,
            "reason": "LT_3_QRS_FOR_MORPHOLOGY_STABILITY",
        }

    stack = np.vstack(vectors)
    template = np.median(stack, axis=0)
    template = template - float(np.mean(template))
    norm = float(np.linalg.norm(template))
    if norm <= 1e-9:
        return {"evaluable": False, "reason": "ZERO_MEDIAN_QRS_TEMPLATE"}
    template = template / norm

    correlations = np.clip(stack @ template, -1.0, 1.0)
    median_corr = float(np.median(correlations))
    low_corr_fraction = float(np.mean(correlations < 0.75))

    widths_arr = np.asarray(widths, dtype=float)
    median_width = float(np.median(widths_arr))
    narrow_limit = min(100.0, 0.72 * median_width)
    capture_candidates = int(np.sum(widths_arr <= narrow_limit))
    fusion_candidates = int(
        np.sum(
            (widths_arr > narrow_limit)
            & (widths_arr <= 0.88 * median_width)
        )
    )

    return {
        "evaluable": True,
        "beat_n": int(len(vectors)),
        "median_morphology_correlation": round(median_corr, 6),
        "low_correlation_fraction": round(low_corr_fraction, 6),
        "monomorphic_compatible": bool(
            median_corr >= 0.82 and low_corr_fraction <= 0.25
        ),
        "median_qrs_ms": round(median_width, 3),
        "capture_candidate_n": capture_candidates,
        "fusion_candidate_n": fusion_candidates,
        "capture_or_fusion_candidate": bool(
            capture_candidates >= 1 or fusion_candidates >= 1
        ),
        "note": (
            "Capture/fusion flags are morphology-width candidates only; they are "
            "supportive evidence and are never treated as definitive alone."
        ),
    }


def analyze_wide_complex_tachycardia(
    canonical_ecg: Dict[str, Any],
    digital_measurements: Dict[str, Any],
) -> Dict[str, Any]:
    """Research classifier for tachycardia with QRS >=120 ms.

    The module integrates directly measurable morphology inspired by established
    wide-complex tachycardia algorithms. It deliberately returns a compatibility
    classification rather than a binary clinical diagnosis.
    """
    rhythm = digital_measurements.get("rhythm") or {}
    global_m = digital_measurements.get("global") or {}
    per_lead = digital_measurements.get("leads") or {}
    atrial = digital_measurements.get("atrial_activity") or {}
    atrial_mechanism = digital_measurements.get("atrial_mechanism") or {}
    canonical_leads = canonical_ecg.get("leads") or {}
    fs = int(canonical_ecg.get("fs") or 500)

    hr = _finite_float(rhythm.get("heart_rate_bpm"))
    qrs_metric = global_m.get("qrs_ms") or {}
    qrs_ms = _finite_float(qrs_metric.get("value"))
    qrs_conf = _finite_float(qrs_metric.get("confidence")) or 0.0

    if hr is None:
        return {
            "version": WCT_ANALYZER_VERSION,
            "evaluable": False,
            "classification": "NOT_EVALUABLE",
            "reason": "HEART_RATE_NOT_MEASURABLE",
            "diagnostic_claim_allowed": False,
        }

    tachycardia = bool(hr >= 100.0)
    if not tachycardia:
        return {
            "version": WCT_ANALYZER_VERSION,
            "evaluable": True,
            "classification": "NOT_TACHYCARDIA",
            "heart_rate_bpm": round(hr, 3),
            "qrs_ms": qrs_ms,
            "diagnostic_claim_allowed": False,
        }

    if qrs_ms is None:
        return {
            "version": WCT_ANALYZER_VERSION,
            "evaluable": False,
            "classification": "TACHYCARDIA_QRS_NOT_EVALUABLE",
            "heart_rate_bpm": round(hr, 3),
            "reason": "QRS_DURATION_NOT_MEASURABLE",
            "diagnostic_claim_allowed": False,
        }

    wide = bool(qrs_ms >= 120.0)
    if not wide:
        return {
            "version": WCT_ANALYZER_VERSION,
            "evaluable": True,
            "classification": "NARROW_COMPLEX_TACHYCARDIA",
            "heart_rate_bpm": round(hr, 3),
            "qrs_ms": round(qrs_ms, 3),
            "qrs_confidence": round(qrs_conf, 6),
            "diagnostic_claim_allowed": False,
        }

    morphology: Dict[str, Dict[str, Any]] = {}
    for lead in set(PRECORDIAL_LEADS + LIMB_LEADS):
        morphology[lead] = _lead_morphology(
            lead,
            dict(canonical_leads.get(lead) or {}),
            dict(per_lead.get(lead) or {}),
            fs,
        )

    precordial = [
        morphology[lead]
        for lead in PRECORDIAL_LEADS
        if morphology.get(lead, {}).get("evaluable")
    ]
    precordial_polarities = [
        row.get("dominant_polarity") for row in precordial
    ]
    all_positive_concordance = bool(
        len(precordial_polarities) >= 5
        and all(p == "POSITIVE" for p in precordial_polarities)
    )
    all_negative_concordance = bool(
        len(precordial_polarities) >= 5
        and all(p == "NEGATIVE" for p in precordial_polarities)
    )
    precordial_concordance = bool(
        all_positive_concordance or all_negative_concordance
    )

    rs_rows = [row for row in precordial if row.get("rs_present")]
    no_rs_precordial = bool(
        len(precordial) >= 5 and len(rs_rows) == 0
    )
    rs_intervals = [
        float(row["rs_interval_ms"])
        for row in rs_rows
        if row.get("rs_interval_ms") is not None
    ]
    max_rs_interval_ms = max(rs_intervals) if rs_intervals else None
    rs_interval_gt_100 = bool(
        max_rs_interval_ms is not None and max_rs_interval_ms > 100.0
    )

    avr = morphology.get("aVR") or {}
    lead_ii = morphology.get("II") or {}
    avr_monophasic_r = bool(
        avr.get("evaluable") and avr.get("monophasic_positive")
    )
    avr_first_peak_gt_40 = bool(
        avr.get("evaluable")
        and _finite_float(avr.get("first_major_peak_ms")) is not None
        and float(avr["first_major_peak_ms"]) > 40.0
    )
    ii_first_peak_gt_40 = bool(
        lead_ii.get("evaluable")
        and _finite_float(lead_ii.get("first_major_peak_ms")) is not None
        and float(lead_ii["first_major_peak_ms"]) > 40.0
    )

    limb_negative = bool(
        all(
            (morphology.get(lead) or {}).get("dominant_polarity") == "NEGATIVE"
            for lead in ("I", "II", "III")
        )
    )

    inferior = [
        (morphology.get(lead) or {}).get("dominant_polarity")
        for lead in ("II", "III", "aVF")
    ]
    superior = [
        (morphology.get(lead) or {}).get("dominant_polarity")
        for lead in ("I", "aVL")
    ]
    inferior_same = bool(
        len(inferior) == 3
        and inferior[0] in {"POSITIVE", "NEGATIVE"}
        and all(p == inferior[0] for p in inferior)
    )
    superior_same_opposite = bool(
        inferior_same
        and len(superior) == 2
        and all(
            p in {"POSITIVE", "NEGATIVE"} and p != inferior[0]
            for p in superior
        )
    )
    opposing_limb_polarity = bool(inferior_same and superior_same_opposite)

    # Coarse classic BBB morphology support; supportive only.
    v1 = morphology.get("V1") or {}
    v6 = morphology.get("V6") or {}
    lbbb_like = bool(
        v1.get("evaluable")
        and v6.get("evaluable")
        and v1.get("dominant_polarity") == "NEGATIVE"
        and v6.get("dominant_polarity") == "POSITIVE"
    )
    rbbb_like = bool(
        v1.get("evaluable")
        and v6.get("evaluable")
        and v1.get("dominant_polarity") == "POSITIVE"
        and v6.get("terminal_polarity") == "NEGATIVE"
    )
    typical_bbb_support = bool(lbbb_like or rbbb_like)

    rhythm_lead = str(rhythm.get("lead") or "II")
    stability = _rhythm_qrs_morphology_stability(
        dict(canonical_leads.get(rhythm_lead) or {}),
        dict(per_lead.get(rhythm_lead) or {}),
    )

    p_reproducible = bool(atrial.get("p_wave_reproducible"))
    p_coupling = _finite_float(atrial.get("rhythm_p_qrs_coupling_fraction"))
    av_dissociation_support = bool(
        p_reproducible
        and p_coupling is not None
        and p_coupling < 0.35
    )
    capture_fusion_support = bool(
        stability.get("capture_or_fusion_candidate")
        and stability.get("monomorphic_compatible")
    )

    atrial_mech = str(atrial_mechanism.get("mechanism") or "")
    atrial_svt_support = bool(
        atrial_mech in {
            "FLUTTER_OR_AT_COMPATIBLE",
            "OTHER_SVT_COMPATIBLE",
            "SINUS_COMPATIBLE",
        }
    )
    one_to_one_atrial_support = bool(
        p_reproducible
        and p_coupling is not None
        and p_coupling >= 0.70
    )

    # Evidence weights are intentionally conservative and transparent.
    vt_components = {
        "av_dissociation": 0.28 if av_dissociation_support else 0.0,
        "capture_or_fusion_candidate": 0.18 if capture_fusion_support else 0.0,
        "precordial_concordance": 0.18 if precordial_concordance else 0.0,
        "no_rs_precordial": 0.16 if no_rs_precordial else 0.0,
        "rs_interval_gt_100ms": 0.16 if rs_interval_gt_100 else 0.0,
        "avr_monophasic_r": 0.12 if avr_monophasic_r else 0.0,
        "limb_I_II_III_negative": 0.10 if limb_negative else 0.0,
        "opposing_limb_polarity": 0.10 if opposing_limb_polarity else 0.0,
        "lead_II_first_peak_gt_40ms": 0.08 if ii_first_peak_gt_40 else 0.0,
        "aVR_first_peak_gt_40ms": 0.08 if avr_first_peak_gt_40 else 0.0,
    }
    vt_score = _clip01(sum(vt_components.values()))

    svt_components = {
        "reproducible_1to1_atrial_activity": 0.30 if one_to_one_atrial_support else 0.0,
        "organized_atrial_mechanism": 0.22 if atrial_svt_support else 0.0,
        "typical_bbb_morphology_support": 0.22 if typical_bbb_support else 0.0,
        "no_strong_vt_morphology": 0.18 if vt_score < 0.30 else 0.0,
        "stable_monomorphic_qrs": 0.08 if stability.get("monomorphic_compatible") else 0.0,
    }
    svt_score = _clip01(sum(svt_components.values()))

    if not stability.get("evaluable"):
        morphology_mode = "UNKNOWN"
    elif stability.get("monomorphic_compatible"):
        morphology_mode = "MONOMORPHIC_COMPATIBLE"
    else:
        morphology_mode = "POLYMORPHIC_OR_UNSTABLE"

    strong_vt_feature = bool(
        av_dissociation_support
        or (
            capture_fusion_support
            and (
                precordial_concordance
                or no_rs_precordial
                or rs_interval_gt_100
                or avr_monophasic_r
            )
        )
    )

    margin = float(vt_score - svt_score)
    if morphology_mode == "POLYMORPHIC_OR_UNSTABLE":
        classification = "WIDE_COMPLEX_TACHYCARDIA_UNDETERMINED"
        confidence = min(max(vt_score, svt_score), 0.55)
        reason = "QRS_MORPHOLOGY_NOT_STABLY_MONOMORPHIC"
    elif strong_vt_feature and vt_score >= 0.45:
        classification = "VT_COMPATIBLE"
        confidence = max(0.65, min(0.95, 0.55 + 0.40 * vt_score))
        reason = "STRONG_DIRECT_VT_SUPPORT"
    elif vt_score >= 0.58 and margin >= 0.12:
        classification = "VT_COMPATIBLE"
        confidence = min(0.90, 0.55 + 0.45 * vt_score)
        reason = "MULTICRITERIA_VT_MORPHOLOGY_SUPPORT"
    elif svt_score >= 0.62 and (svt_score - vt_score) >= 0.15:
        classification = "SVT_ABERRANCY_OR_PREEXCITATION_COMPATIBLE"
        confidence = min(0.88, 0.55 + 0.40 * svt_score)
        reason = "ATRIAL_AND_BBB_SUPPORT_WITHOUT_STRONG_VT_CRITERIA"
    else:
        classification = "WIDE_COMPLEX_TACHYCARDIA_UNDETERMINED"
        confidence = min(max(vt_score, svt_score), 0.59)
        reason = "VT_AND_SVT_EVIDENCE_NOT_SUFFICIENTLY_SEPARATED"

    criteria = {
        "tachycardia_hr_ge_100": True,
        "qrs_ge_120ms": True,
        "precordial_concordance": precordial_concordance,
        "positive_precordial_concordance": all_positive_concordance,
        "negative_precordial_concordance": all_negative_concordance,
        "no_rs_in_precordials": no_rs_precordial,
        "max_rs_interval_ms": (
            round(float(max_rs_interval_ms), 3)
            if max_rs_interval_ms is not None else None
        ),
        "rs_interval_gt_100ms": rs_interval_gt_100,
        "avr_monophasic_r": avr_monophasic_r,
        "lead_II_first_peak_gt_40ms": ii_first_peak_gt_40,
        "aVR_first_peak_gt_40ms": avr_first_peak_gt_40,
        "limb_I_II_III_predominantly_negative": limb_negative,
        "opposing_limb_polarity": opposing_limb_polarity,
        "av_dissociation_support": av_dissociation_support,
        "capture_or_fusion_candidate": capture_fusion_support,
        "lbbb_like_support": lbbb_like,
        "rbbb_like_support": rbbb_like,
    }

    return {
        "version": WCT_ANALYZER_VERSION,
        "evaluable": True,
        "source": "CALIBRATED_DIGITAL_SIGNAL_ONLY",
        "heart_rate_bpm": round(hr, 3),
        "qrs_ms": round(qrs_ms, 3),
        "qrs_confidence": round(qrs_conf, 6),
        "wide_complex_tachycardia": True,
        "morphology_mode": morphology_mode,
        "classification": classification,
        "confidence": round(float(confidence), 6),
        "reason": reason,
        "diagnostic_claim_allowed": False,
        "compatibility_scores_not_probabilities": {
            "VT": round(vt_score, 6),
            "SVT_ABERRANCY_OR_PREEXCITATION": round(svt_score, 6),
        },
        "criteria": criteria,
        "vt_score_components": vt_components,
        "svt_score_components": svt_components,
        "rhythm_qrs_stability": stability,
        "lead_morphology": morphology,
        "methodology_note": (
            "Research compatibility classifier for wide-complex tachycardia using "
            "direct digital-signal morphology. Features include precordial concordance, "
            "RS presence/interval, aVR and limb-lead morphology, II/aVR time to first "
            "major deflection, atrioventricular dissociation evidence, capture/fusion "
            "candidates, QRS morphology stability and coarse BBB support. It does not "
            "convert any single criterion into a definitive diagnosis."
        ),
    }
