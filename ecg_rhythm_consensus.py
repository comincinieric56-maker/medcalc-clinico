from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


RHYTHM_CONSENSUS_VERSION = "MEDCALC_RHYTHM_CONSENSUS_V1"


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _score_linear(value: float | None, lo: float, hi: float) -> float:
    if value is None or hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def build_rhythm_consensus(
    per_lead: Dict[str, Dict[str, Any]],
    rhythm_lead: str | None,
) -> Dict[str, Any]:
    """Robust rate consensus while preserving single-lead RR timing.

    Absolute heart rate is aggregated across independent lead windows to reduce
    missed/double R-peak bias. RR irregularity remains sourced from the selected
    native rhythm lead because non-simultaneous layout windows must never be
    stitched into a synthetic RR series.
    """
    rows = []
    for lead, item in per_lead.items():
        if not item.get("evaluable"):
            continue
        hr = _finite(item.get("heart_rate_bpm"))
        conf = _finite(item.get("confidence")) or 0.0
        r_n = int(item.get("r_count") or 0)
        duration = _finite(item.get("duration_s")) or 0.0
        if hr is None or not 25.0 <= hr <= 250.0 or r_n < 3 or duration < 1.5:
            continue
        rows.append({
            "lead": lead,
            "heart_rate_bpm": hr,
            "confidence": conf,
            "r_count": r_n,
            "duration_s": duration,
        })

    trusted = [row for row in rows if row["confidence"] >= 0.45]
    use = trusted if len(trusted) >= 3 else rows

    if not use:
        return {
            "version": RHYTHM_CONSENSUS_VERSION,
            "evaluable": False,
            "heart_rate_bpm": None,
            "confidence": 0.0,
            "reason": "NO_MULTILEAD_RATE_CANDIDATES",
        }

    values = np.asarray([row["heart_rate_bpm"] for row in use], dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) if len(values) >= 2 else 0.0
    tolerance = max(8.0, 0.18 * med)
    inliers = [
        row for row in use
        if abs(float(row["heart_rate_bpm"]) - med) <= tolerance
    ]
    if len(inliers) >= 3:
        values = np.asarray([row["heart_rate_bpm"] for row in inliers], dtype=float)
        med = float(np.median(values))
        use = inliers
        mad = float(np.median(np.abs(values - med))) if len(values) >= 2 else 0.0

    relative_mad = mad / max(med, 1.0)
    n_factor = float(np.clip(len(use) / 6.0, 0.45, 1.0))
    agreement = float(np.clip(1.0 - relative_mad / 0.15, 0.25, 1.0))
    base_conf = float(np.median([row["confidence"] for row in use]))
    confidence = float(np.clip(base_conf * n_factor * agreement, 0.0, 1.0))

    selected = next(
        (row for row in rows if row["lead"] == rhythm_lead),
        None,
    )
    selected_hr = selected["heart_rate_bpm"] if selected else None
    selected_outlier = bool(
        selected_hr is not None
        and abs(float(selected_hr) - med) > max(10.0, 0.20 * med)
        and len(use) >= 3
    )

    return {
        "version": RHYTHM_CONSENSUS_VERSION,
        "evaluable": True,
        "heart_rate_bpm": round(med, 6),
        "confidence": round(confidence, 6),
        "source": "ROBUST_MULTILEAD_HEART_RATE_CONSENSUS",
        "source_leads": [row["lead"] for row in use],
        "source_n": len(use),
        "crosslead_mad_bpm": round(mad, 6),
        "selected_rhythm_lead": rhythm_lead,
        "selected_rhythm_lead_hr_bpm": (
            round(float(selected_hr), 6) if selected_hr is not None else None
        ),
        "selected_rhythm_lead_rate_outlier": selected_outlier,
        "rule": (
            "RATE_AGGREGATED_ACROSS_LEADS; RR_SEQUENCE_NEVER_STITCHED_ACROSS_"
            "NONSIMULTANEOUS_LAYOUT_WINDOWS"
        ),
    }


def rr_irregularity_score(rhythm: Dict[str, Any]) -> Dict[str, Any]:
    rr_med = _finite(rhythm.get("rr_median_ms"))
    cv = _finite(rhythm.get("rr_cv"))
    mad = _finite(rhythm.get("rr_mad_ms"))
    rmssd = _finite(rhythm.get("rr_rmssd_ms"))
    pnn50 = _finite(rhythm.get("rr_pnn50"))

    mad_ratio = (mad / rr_med) if mad is not None and rr_med not in (None, 0) else None
    rmssd_ratio = (
        rmssd / rr_med if rmssd is not None and rr_med not in (None, 0) else None
    )

    components = {
        "cv": _score_linear(cv, 0.06, 0.20),
        "mad_ratio": _score_linear(mad_ratio, 0.04, 0.15),
        "rmssd_ratio": _score_linear(rmssd_ratio, 0.06, 0.22),
        "pnn50": _score_linear(pnn50, 0.15, 0.65),
    }
    available = {
        key: value for key, value in components.items()
        if {
            "cv": cv,
            "mad_ratio": mad_ratio,
            "rmssd_ratio": rmssd_ratio,
            "pnn50": pnn50,
        }[key] is not None
    }
    if not available:
        return {
            "evaluable": False,
            "score": None,
            "components": components,
        }

    weights = {"cv": 0.35, "mad_ratio": 0.25, "rmssd_ratio": 0.25, "pnn50": 0.15}
    denom = sum(weights[k] for k in available)
    score = sum(weights[k] * available[k] for k in available) / max(denom, 1e-9)
    return {
        "evaluable": True,
        "score": round(float(np.clip(score, 0.0, 1.0)), 6),
        "components": {k: round(float(v), 6) for k, v in components.items()},
        "rr_mad_ratio": round(float(mad_ratio), 6) if mad_ratio is not None else None,
        "rr_rmssd_ratio": round(float(rmssd_ratio), 6) if rmssd_ratio is not None else None,
        "source": "ROBUST_SINGLE_RHYTHM_LEAD_RR_FEATURES",
    }
