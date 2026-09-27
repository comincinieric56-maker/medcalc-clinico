from __future__ import annotations

from typing import Any, Dict

import numpy as np


SIGNAL_INTEGRITY_VERSION = "MEDCALC_SIGNAL_INTEGRITY_V1"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _longest_run(mask: np.ndarray) -> int:
    x = np.asarray(mask, dtype=bool)
    if not x.any():
        return 0
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int(max((b - a for a, b in zip(starts, ends)), default=0))


def analyze_signal_integrity(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Audit reconstructed signal quality without modifying the waveform."""
    fs = int(canonical_ecg.get("fs") or 500)
    source = canonical_ecg.get("leads") or {}
    rows: Dict[str, Any] = {}

    for lead in LEADS:
        item = dict(source.get(lead) or {})
        signal = np.asarray(
            [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
            dtype=float,
        )
        q = np.asarray(item.get("quality_mask", []), dtype=np.uint8)
        if q.size != signal.size:
            qq = np.zeros(signal.size, dtype=np.uint8)
            qq[: min(q.size, signal.size)] = q[: min(q.size, signal.size)]
            q = qq

        n = int(signal.size)
        finite = np.isfinite(signal)
        usable = finite & (q > 0)
        observed = finite & (q == 2)
        interpolated = finite & (q == 1)
        longest_samples = _longest_run(usable)
        longest_s = longest_samples / float(fs) if fs > 0 else 0.0
        finite_fraction = float(np.mean(finite)) if n else 0.0
        observed_fraction = float(np.mean(observed)) if n else 0.0
        interpolated_fraction = float(np.mean(interpolated)) if n else 0.0

        z = signal[finite]
        if z.size >= 20:
            p01, p99 = np.percentile(z, [1.0, 99.0])
            robust_span = float(p99 - p01)
        else:
            robust_span = 0.0
        flatline = bool(z.size >= 20 and robust_span < 0.03)

        measured = dict(per_lead.get(lead) or {})
        r_count = int(measured.get("r_count") or 0)
        base_conf = float(measured.get("confidence") or 0.0)

        interval_eligible = bool(
            longest_s >= 1.5
            and observed_fraction >= 0.45
            and not flatline
        )
        morphology_eligible = bool(
            longest_s >= 0.8
            and observed_fraction >= 0.40
            and not flatline
        )
        rhythm_eligible = bool(
            longest_s >= 5.0
            and r_count >= 5
            and observed_fraction >= 0.45
            and not flatline
        )

        continuity_score = float(np.clip(longest_s / 5.0, 0.0, 1.0))
        quality_score = float(np.clip(
            0.45 * observed_fraction
            + 0.15 * finite_fraction
            + 0.20 * continuity_score
            + 0.20 * base_conf
            - 0.20 * min(interpolated_fraction, 0.20) / 0.20,
            0.0,
            1.0,
        ))
        if flatline:
            quality_score = min(quality_score, 0.15)

        flags: list[str] = []
        if finite_fraction < 0.50:
            flags.append("LOW_FINITE_COVERAGE")
        if observed_fraction < 0.45:
            flags.append("LOW_OBSERVED_COVERAGE")
        if interpolated_fraction > 0.15:
            flags.append("HIGH_INTERPOLATION_FRACTION")
        if longest_s < 1.5:
            flags.append("SHORT_CONTIGUOUS_SUPPORT")
        if flatline:
            flags.append("FLATLINE_OR_NEAR_FLAT_SIGNAL")

        rows[lead] = {
            "signal_samples": n,
            "finite_fraction": round(finite_fraction, 6),
            "observed_fraction": round(observed_fraction, 6),
            "interpolated_fraction": round(interpolated_fraction, 6),
            "longest_usable_duration_s": round(longest_s, 6),
            "robust_amplitude_span_mv": round(robust_span, 6),
            "flatline": flatline,
            "r_count": r_count,
            "measurement_confidence": round(base_conf, 6),
            "interval_eligible": interval_eligible,
            "morphology_eligible": morphology_eligible,
            "rhythm_eligible": rhythm_eligible,
            "quality_score": round(quality_score, 6),
            "flags": flags,
        }

    scores = [float(v["quality_score"]) for v in rows.values()]
    interval_leads = [k for k, v in rows.items() if v["interval_eligible"]]
    morphology_leads = [k for k, v in rows.items() if v["morphology_eligible"]]
    rhythm_leads = [k for k, v in rows.items() if v["rhythm_eligible"]]

    return {
        "version": SIGNAL_INTEGRITY_VERSION,
        "source": "CANONICAL_SIGNAL_QUALITY_MASK_AND_CONTIGUITY",
        "per_lead": rows,
        "overall_quality": round(float(np.median(scores)), 6) if scores else 0.0,
        "interval_eligible_leads": interval_leads,
        "morphology_eligible_leads": morphology_leads,
        "rhythm_eligible_leads": rhythm_leads,
        "fail_closed": True,
        "policy": (
            "QUALITY_AUDIT_DOES_NOT_FILL_MISSING_SIGNAL_OR_OVERRIDE_MEASUREMENTS"
        ),
    }
