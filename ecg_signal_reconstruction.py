from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {lead: i for i, lead in enumerate(LEADS)}
LAYOUTS = {
    "6x2": [
        ["I", "V1"],
        ["II", "V2"],
        ["III", "V3"],
        ["aVR", "V4"],
        ["aVL", "V5"],
        ["aVF", "V6"],
    ],
    "3x4": [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ],
    "12x1": [[lead] for lead in LEADS],
}


def _finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    if x.size == 0:
        return []
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _longest_run_fraction(mask: np.ndarray) -> float:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    if x.size == 0:
        return 0.0
    runs = _finite_runs(x)
    if not runs:
        return 0.0
    return float(max(b - a for a, b in runs) / x.size)


def _or_empty(value: Any):
    return [] if value is None else value


def _safe_float(value: Any) -> Optional[float]:
    try:
        z = float(value)
    except Exception:
        return None
    return z if math.isfinite(z) else None


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def resolve_calibration(
    *,
    pixel_spacing_mm: Mapping[str, Any] | None,
    machine_measurements: Mapping[str, Any] | None = None,
    pulse_calibration: Mapping[str, Any] | None = None,
    default_speed_mm_s: float = 25.0,
    default_gain_mm_mv: float = 10.0,
) -> Dict[str, Any]:
    """Resolve the physical ECG scale without using clinical morphology.

    Grid spacing from the digitizer establishes pixel->mm. Printed machine
    settings and/or a detected calibration pulse establish mm/s and mm/mV.
    Standard defaults are retained only as an explicitly low-confidence
    compatibility fallback; downstream measurements can fail closed on that
    confidence rather than silently treating an assumption as ground truth.
    """
    pixel_spacing_mm = dict(pixel_spacing_mm or {})
    machine_measurements = dict(machine_measurements or {})
    pulse_calibration = dict(pulse_calibration or {})

    mm_px_x = _safe_float(pixel_spacing_mm.get("x"))
    mm_px_y = _safe_float(pixel_spacing_mm.get("y"))
    grid_ok = bool(
        mm_px_x is not None and mm_px_y is not None
        and 0.01 <= mm_px_x <= 2.0
        and 0.01 <= mm_px_y <= 2.0
    )
    grid_conf = 0.96 if grid_ok else 0.0

    speed_candidates: List[Tuple[str, float, float]] = []
    machine_speed = _safe_float(machine_measurements.get("speed_mm_per_s"))
    if machine_speed is not None and 10.0 <= machine_speed <= 100.0:
        speed_candidates.append(("PRINTED_MACHINE_SETTING", machine_speed, 0.94))

    pulse_speed = _safe_float(pulse_calibration.get("speed_mm_s"))
    pulse_conf = _safe_float(pulse_calibration.get("confidence")) or 0.0
    if (
        pulse_speed is not None
        and 10.0 <= pulse_speed <= 100.0
        and bool(pulse_calibration.get("detected"))
    ):
        speed_candidates.append(("CALIBRATION_PULSE", pulse_speed, _clip01(pulse_conf)))

    gain_candidates: List[Tuple[str, float, float]] = []
    machine_gain = _safe_float(machine_measurements.get("gain_mm_per_mV"))
    if machine_gain is not None and 2.5 <= machine_gain <= 40.0:
        gain_candidates.append(("PRINTED_MACHINE_SETTING", machine_gain, 0.94))

    pulse_gain = _safe_float(pulse_calibration.get("gain_mm_mV"))
    if (
        pulse_gain is not None
        and 2.5 <= pulse_gain <= 40.0
        and bool(pulse_calibration.get("detected"))
    ):
        gain_candidates.append(("CALIBRATION_PULSE", pulse_gain, _clip01(pulse_conf)))

    def choose(
        candidates: List[Tuple[str, float, float]],
        default_value: float,
        default_label: str,
    ) -> Tuple[float, float, str, bool, List[Dict[str, Any]]]:
        audit = [
            {"source": src, "value": float(value), "confidence": float(conf)}
            for src, value, conf in candidates
        ]
        conflict = False
        if candidates:
            values = [v for _, v, _ in candidates]
            if len(values) >= 2:
                lo, hi = min(values), max(values)
                conflict = bool(lo > 0 and (hi - lo) / lo > 0.12)
            source, value, confidence = max(candidates, key=lambda x: x[2])
            if conflict:
                confidence = min(float(confidence), 0.45)
                source += "_CONFLICT"
            return float(value), _clip01(float(confidence)), source, conflict, audit
        return float(default_value), 0.35, default_label, False, audit

    speed, speed_conf, speed_source, speed_conflict, speed_audit = choose(
        speed_candidates,
        default_speed_mm_s,
        "ASSUMED_STANDARD_SPEED",
    )
    gain, gain_conf, gain_source, gain_conflict, gain_audit = choose(
        gain_candidates,
        default_gain_mm_mv,
        "ASSUMED_STANDARD_GAIN",
    )

    overall = min(grid_conf, speed_conf, gain_conf) if grid_ok else 0.0
    return {
        "schema": "MEDCALC_ECG_CALIBRATION_V2",
        "mm_per_pixel_x": mm_px_x,
        "mm_per_pixel_y": mm_px_y,
        "speed_mm_s": speed,
        "gain_mm_mV": gain,
        "grid_confidence": round(grid_conf, 4),
        "speed_confidence": round(speed_conf, 4),
        "gain_confidence": round(gain_conf, 4),
        "confidence": round(float(overall), 4),
        "speed_source": speed_source,
        "gain_source": gain_source,
        "speed_conflict": bool(speed_conflict),
        "gain_conflict": bool(gain_conflict),
        "candidates": {
            "speed": speed_audit,
            "gain": gain_audit,
        },
        "quantitative_scale_verified": bool(overall >= 0.50),
        "assumption_used": bool(
            speed_source.startswith("ASSUMED_")
            or gain_source.startswith("ASSUMED_")
        ),
    }


def _remove_isolated_centerline_spikes(
    y_px: np.ndarray,
    *,
    mm_per_pixel_y: float,
    isolated_jump_mm: float = 4.0,
    neighbour_agreement_mm: float = 1.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove only isolated impossible centerline spikes.

    Broad QRS excursions are preserved. A point is rejected only when it is far
    from both immediate neighbours while those neighbours agree with each other.
    """
    y = np.asarray(y_px, dtype=float).copy()
    rejected = np.zeros(y.size, dtype=bool)
    if y.size < 3:
        return y, rejected

    jump_px = float(isolated_jump_mm) / max(float(mm_per_pixel_y), 1e-9)
    agree_px = float(neighbour_agreement_mm) / max(float(mm_per_pixel_y), 1e-9)
    finite = np.isfinite(y)

    for i in range(1, y.size - 1):
        if not (finite[i - 1] and finite[i] and finite[i + 1]):
            continue
        neighbours_agree = abs(y[i - 1] - y[i + 1]) <= agree_px
        isolated = (
            abs(y[i] - y[i - 1]) >= jump_px
            and abs(y[i] - y[i + 1]) >= jump_px
        )
        if neighbours_agree and isolated:
            y[i] = np.nan
            rejected[i] = True
    return y, rejected


def _interpolate_small_gaps(
    values: np.ndarray,
    *,
    sample_period_ms: float,
    max_gap_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).copy()
    filled = np.zeros(x.size, dtype=bool)
    if x.size < 3:
        return x, filled
    finite = np.isfinite(x)
    for a, b in _finite_runs(~finite):
        if a == 0 or b >= x.size:
            continue
        gap_ms = float((b - a) * sample_period_ms)
        if gap_ms > float(max_gap_ms):
            continue
        if not (np.isfinite(x[a - 1]) and np.isfinite(x[b])):
            continue
        x[a:b] = np.linspace(x[a - 1], x[b], (b - a) + 2)[1:-1]
        filled[a:b] = True
    return x, filled


def _centerline_probability(
    signal_prob: np.ndarray | None,
    y_px: np.ndarray,
    x_indices: np.ndarray,
) -> np.ndarray:
    if signal_prob is None:
        return np.where(np.isfinite(y_px), 0.75, 0.0).astype(float)

    prob = np.asarray(signal_prob, dtype=float)
    if prob.ndim != 2:
        return np.where(np.isfinite(y_px), 0.75, 0.0).astype(float)

    h, w = prob.shape
    out = np.zeros(len(y_px), dtype=float)
    for i, (yy, xx) in enumerate(zip(y_px, x_indices)):
        if not math.isfinite(float(yy)):
            continue
        x = int(np.clip(int(xx), 0, w - 1))
        y = int(np.clip(int(round(float(yy))), 0, h - 1))
        y0, y1 = max(0, y - 2), min(h, y + 3)
        out[i] = float(np.nanmax(prob[y0:y1, x])) if y1 > y0 else 0.0
    return np.clip(out, 0.0, 1.0)


def _resample_preserving_gaps(
    time_ms: np.ndarray,
    values: np.ndarray,
    observed_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    fs: int,
    interpolated_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    t = np.asarray(time_ms, dtype=float)
    x = np.asarray(values, dtype=float)
    observed = np.asarray(observed_mask, dtype=bool)
    conf = np.asarray(confidence, dtype=float)
    interpolated = (
        np.asarray(interpolated_mask, dtype=bool)
        if interpolated_mask is not None
        else np.zeros(values.size, dtype=bool)
    )

    if t.size == 0:
        return {
            "time_ms": np.asarray([], dtype=float),
            "signal_mv": np.asarray([], dtype=float),
            "observed_mask": np.asarray([], dtype=bool),
            "interpolated_mask": np.asarray([], dtype=bool),
            "low_confidence_mask": np.asarray([], dtype=bool),
            "confidence_mask": np.asarray([], dtype=float),
        }

    duration_ms = max(0.0, float(t[-1] - t[0]))
    n = max(2, int(round(duration_ms * float(fs) / 1000.0)) + 1)
    dst_t = np.arange(n, dtype=float) * (1000.0 / float(fs))
    dst = np.full(n, np.nan, dtype=float)
    dst_conf = np.zeros(n, dtype=float)
    dst_obs = np.zeros(n, dtype=bool)
    dst_interp = np.zeros(n, dtype=bool)

    finite = np.isfinite(x)
    for a, b in _finite_runs(finite):
        if b - a < 2:
            continue
        ta = t[a:b] - t[0]
        xa = x[a:b]
        ca = conf[a:b]
        lo = int(np.searchsorted(dst_t, ta[0], side="left"))
        hi = int(np.searchsorted(dst_t, ta[-1], side="right"))
        if hi <= lo:
            continue
        seg_t = dst_t[lo:hi]
        dst[lo:hi] = np.interp(seg_t, ta, xa)
        dst_conf[lo:hi] = np.interp(seg_t, ta, ca)

        # Every resampled sample inside an originally observed run is
        # observed. Small internally interpolated gaps remain explicitly marked
        # as interpolated rather than being promoted to observed signal.
        src_obs = observed[a:b].astype(float)
        src_interp = interpolated[a:b].astype(float)
        obs_score = np.interp(seg_t, ta, src_obs)
        interp_score = np.interp(seg_t, ta, src_interp)
        dst_obs[lo:hi] = obs_score >= 0.50
        dst_interp[lo:hi] = interp_score > 0.05
        dst_obs[lo:hi] &= ~dst_interp[lo:hi]

    dst_conf = np.clip(dst_conf, 0.0, 1.0)
    return {
        "time_ms": dst_t,
        "signal_mv": dst,
        "observed_mask": dst_obs,
        "interpolated_mask": dst_interp,
        "low_confidence_mask": np.isfinite(dst) & (dst_conf < 0.45),
        "confidence_mask": dst_conf,
    }


def _lead_record_from_centerline(
    y_px: np.ndarray,
    *,
    x_offset: int,
    mm_per_pixel_x: float,
    mm_per_pixel_y: float,
    speed_mm_s: float,
    gain_mm_mV: float,
    fs: int,
    calibration_confidence: float,
    signal_prob: np.ndarray | None,
    max_interpolation_gap_ms: float,
    source: str,
) -> Dict[str, Any]:
    y0 = np.asarray(y_px, dtype=float).reshape(-1)
    x_indices = np.arange(y0.size, dtype=int) + int(x_offset)
    observed_raw = np.isfinite(y0)

    cleaned_y, rejected_spike = _remove_isolated_centerline_spikes(
        y0,
        mm_per_pixel_y=float(mm_per_pixel_y),
    )
    sample_period_ms = (
        float(mm_per_pixel_x) / float(speed_mm_s) * 1000.0
    )
    cleaned_y, interpolated = _interpolate_small_gaps(
        cleaned_y,
        sample_period_ms=sample_period_ms,
        max_gap_ms=float(max_interpolation_gap_ms),
    )

    finite = np.isfinite(cleaned_y)
    if np.any(finite):
        baseline_y = float(np.nanmedian(cleaned_y))
    else:
        baseline_y = 0.0
    signal_mv_raw = -(
        cleaned_y - baseline_y
    ) * float(mm_per_pixel_y) / float(gain_mm_mV)

    time_ms_raw = np.arange(cleaned_y.size, dtype=float) * sample_period_ms
    local_prob = _centerline_probability(signal_prob, y0, x_indices)
    local_prob[~np.isfinite(cleaned_y)] = 0.0

    resampled = _resample_preserving_gaps(
        time_ms_raw,
        signal_mv_raw,
        observed_raw & ~rejected_spike,
        local_prob,
        fs=int(fs),
        interpolated_mask=interpolated,
    )

    sig = resampled["signal_mv"]
    finite_out = np.isfinite(sig)
    coverage = float(finite_out.mean()) if finite_out.size else 0.0
    longest = _longest_run_fraction(finite_out)
    mean_seg_conf = (
        float(np.nanmedian(resampled["confidence_mask"][finite_out]))
        if finite_out.any()
        else 0.0
    )
    confidence = _clip01(
        0.50 * mean_seg_conf
        + 0.25 * coverage
        + 0.15 * longest
        + 0.10 * float(calibration_confidence)
    )

    return {
        "signal_mv": sig,
        "time_ms": resampled["time_ms"],
        "fs": int(fs),
        "duration_s": (
            float(resampled["time_ms"][-1] / 1000.0)
            if resampled["time_ms"].size else 0.0
        ),
        "source": source,
        "confidence": confidence,
        "observed_mask": resampled["observed_mask"],
        "interpolated_mask": resampled["interpolated_mask"],
        "low_confidence_mask": resampled["low_confidence_mask"],
        "confidence_mask": resampled["confidence_mask"],
        "coverage": coverage,
        "longest_contiguous_fraction": longest,
        "baseline_y_px": baseline_y,
        "raw_observed_fraction": (
            float(observed_raw.mean()) if observed_raw.size else 0.0
        ),
        "interpolated_fraction": (
            float(interpolated.mean()) if interpolated.size else 0.0
        ),
        "rejected_spike_fraction": (
            float(rejected_spike.mean()) if rejected_spike.size else 0.0
        ),
        "scale": {
            "mm_per_pixel_x": float(mm_per_pixel_x),
            "mm_per_pixel_y": float(mm_per_pixel_y),
            "speed_mm_s": float(speed_mm_s),
            "gain_mm_mV": float(gain_mm_mV),
        },
    }


def reconstruct_digital_ecg_from_rows(
    raw_rows: np.ndarray,
    *,
    layout: str,
    rhythm_strip: bool,
    active_x: Sequence[int] | None,
    pixel_spacing_mm: Mapping[str, Any],
    calibration: Mapping[str, Any],
    signal_prob: np.ndarray | None = None,
    fs: int = 500,
    max_interpolation_gap_ms: float = 24.0,
    row_sources: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Convert U-Net/digitizer centerlines into calibrated per-lead signals.

    Layout is used only here to map physical rows/columns to lead names. The
    returned clinical representation is layout-independent.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Layout no soportado por reconstrucción física: {layout}")

    rows = np.asarray(raw_rows, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"raw_rows debe ser 2-D; recibido {rows.shape}")

    mm_px_x = _safe_float(pixel_spacing_mm.get("x"))
    mm_px_y = _safe_float(pixel_spacing_mm.get("y"))
    speed = _safe_float(calibration.get("speed_mm_s"))
    gain = _safe_float(calibration.get("gain_mm_mV"))
    cal_conf = _safe_float(calibration.get("confidence")) or 0.0
    if not all(v is not None and v > 0 for v in [mm_px_x, mm_px_y, speed, gain]):
        raise ValueError("Escala física incompleta para reconstrucción digital.")

    row_means = np.nanmean(rows, axis=1)
    order = np.argsort(np.nan_to_num(row_means, nan=np.inf))
    rows = rows[order]

    matrix = LAYOUTS[layout]
    primary_n = len(matrix)
    if rows.shape[0] < primary_n:
        raise ValueError(
            f"Filas insuficientes para {layout}: {rows.shape[0]} < {primary_n}"
        )

    if active_x is not None and len(active_x) == 2:
        x0 = max(0, int(active_x[0]))
        x1 = min(rows.shape[1] - 1, int(active_x[1]))
    else:
        finite_cols = np.flatnonzero(np.any(np.isfinite(rows[:primary_n]), axis=0))
        if finite_cols.size >= 2:
            x0, x1 = int(finite_cols[0]), int(finite_cols[-1])
        else:
            x0, x1 = 0, rows.shape[1] - 1
    if x1 <= x0:
        raise ValueError("Active span inválido para reconstrucción.")

    n_cols = len(matrix[0])
    edges = np.linspace(x0, x1 + 1, n_cols + 1)
    edges = np.rint(edges).astype(int)
    edges[0], edges[-1] = x0, x1 + 1

    leads: Dict[str, Dict[str, Any]] = {}
    audit_segments: Dict[str, Any] = {}
    sources = list(row_sources or [])

    for r, lead_row in enumerate(matrix):
        row = rows[r]
        for c, lead in enumerate(lead_row):
            a, b = int(edges[c]), int(edges[c + 1])
            source = (
                sources[r]
                if r < len(sources)
                else "U_NET_DIGITIZER_CENTERLINE"
            )
            rec = _lead_record_from_centerline(
                row[a:b],
                x_offset=a,
                mm_per_pixel_x=float(mm_px_x),
                mm_per_pixel_y=float(mm_px_y),
                speed_mm_s=float(speed),
                gain_mm_mV=float(gain),
                fs=int(fs),
                calibration_confidence=float(cal_conf),
                signal_prob=signal_prob,
                max_interpolation_gap_ms=float(max_interpolation_gap_ms),
                source=str(source),
            )
            leads[lead] = rec
            audit_segments[lead] = {
                "row_index": int(r),
                "column_index": int(c),
                "x_start": a,
                "x_end": b,
            }

    rhythm_meta = None
    if rhythm_strip and rows.shape[0] >= primary_n + 1:
        rhythm_row = rows[-1]
        source = (
            sources[primary_n]
            if primary_n < len(sources)
            else "U_NET_DIGITIZER_RHYTHM_CENTERLINE"
        )
        rr = _lead_record_from_centerline(
            rhythm_row[x0:x1 + 1],
            x_offset=x0,
            mm_per_pixel_x=float(mm_px_x),
            mm_per_pixel_y=float(mm_px_y),
            speed_mm_s=float(speed),
            gain_mm_mV=float(gain),
            fs=int(fs),
            calibration_confidence=float(cal_conf),
            signal_prob=signal_prob,
            max_interpolation_gap_ms=float(max_interpolation_gap_ms),
            source=str(source),
        )
        primary_ii = leads.get("II") or {}
        if (
            rr.get("duration_s", 0.0) > float(primary_ii.get("duration_s") or 0.0)
            and rr.get("coverage", 0.0) >= 0.35
        ):
            rr["source"] = str(source) + "_LONG_RHYTHM"
            leads["II"] = rr
            rhythm_meta = {
                "lead": "II",
                "used_as_primary_ii": True,
                "duration_s": rr.get("duration_s"),
                "confidence": rr.get("confidence"),
            }

    missing = [lead for lead in LEADS if lead not in leads]
    for lead in missing:
        leads[lead] = {
            "signal_mv": np.asarray([], dtype=float),
            "time_ms": np.asarray([], dtype=float),
            "fs": int(fs),
            "duration_s": 0.0,
            "source": "NOT_RECOVERED",
            "confidence": 0.0,
            "observed_mask": np.asarray([], dtype=bool),
            "interpolated_mask": np.asarray([], dtype=bool),
            "low_confidence_mask": np.asarray([], dtype=bool),
            "confidence_mask": np.asarray([], dtype=float),
            "coverage": 0.0,
            "longest_contiguous_fraction": 0.0,
        }

    lead_conf = [
        float((leads.get(lead) or {}).get("confidence") or 0.0)
        for lead in LEADS
    ]
    recovered = [
        lead for lead in LEADS
        if float((leads.get(lead) or {}).get("coverage") or 0.0) >= 0.25
    ]
    return {
        "schema": "MEDCALC_DIGITAL_ECG_V2",
        "source": "U_NET_DIGITIZER_CENTERLINE",
        "source_layout": layout,
        "layout_used_only_for_reconstruction": True,
        "fs": int(fs),
        "units": {"time": "ms", "amplitude": "mV"},
        "calibration": dict(calibration),
        "leads": leads,
        "recovered_leads": recovered,
        "recovered_lead_count": len(recovered),
        "global_confidence": (
            float(np.median(lead_conf)) if lead_conf else 0.0
        ),
        "rhythm_strip": rhythm_meta,
        "audit_mapping": {
            "active_x": [int(x0), int(x1)],
            "segments": audit_segments,
            "pixel_spacing_mm": {
                "x": float(mm_px_x),
                "y": float(mm_px_y),
            },
            "y_axis_orientation": "IMAGE_Y_DOWN__ECG_MV_UP",
            "small_gap_interpolation_max_ms": float(max_interpolation_gap_ms),
            "nonrecoverable_regions_preserved_as_nan": True,
        },
    }


def digital_ecg_from_legacy_matrix(
    signal_uv: np.ndarray,
    *,
    fs: int = 500,
    lead_names: Sequence[str] = LEADS,
    calibration: Mapping[str, Any] | None = None,
    source: str = "LEGACY_CANONICAL_DIGITIZER",
) -> Dict[str, Any]:
    """Compatibility adapter for layouts produced directly by Open-ECG.

    This adapter is intentionally transitional. It keeps the clinical analyzer
    layout-independent even when the upstream identifier has already converted
    pixels to a canonical microvolt matrix.
    """
    x = np.asarray(signal_uv, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(lead_names):
        raise ValueError(f"Matriz ECG inesperada: {x.shape}")
    leads: Dict[str, Dict[str, Any]] = {}
    for j, lead in enumerate(lead_names):
        sig = x[:, j] / 1000.0
        finite = np.isfinite(sig)
        idx = np.flatnonzero(finite)
        if idx.size:
            a, b = int(idx[0]), int(idx[-1]) + 1
            segment = sig[a:b]
            obs = np.isfinite(segment)
            time_ms = np.arange(segment.size, dtype=float) * (1000.0 / fs)
        else:
            segment = np.asarray([], dtype=float)
            obs = np.asarray([], dtype=bool)
            time_ms = np.asarray([], dtype=float)
        coverage = float(obs.mean()) if obs.size else 0.0
        leads[str(lead)] = {
            "signal_mv": segment,
            "time_ms": time_ms,
            "fs": int(fs),
            "duration_s": float(time_ms[-1] / 1000.0) if time_ms.size else 0.0,
            "source": source,
            "confidence": _clip01(0.55 + 0.35 * coverage),
            "observed_mask": obs,
            "interpolated_mask": np.zeros(obs.size, dtype=bool),
            "low_confidence_mask": np.where(obs, False, False),
            "confidence_mask": np.where(obs, 0.70, 0.0),
            "coverage": coverage,
            "longest_contiguous_fraction": _longest_run_fraction(obs),
        }
    return {
        "schema": "MEDCALC_DIGITAL_ECG_V2",
        "source": source,
        "source_layout": "UPSTREAM_CANONICAL",
        "layout_used_only_for_reconstruction": True,
        "fs": int(fs),
        "units": {"time": "ms", "amplitude": "mV"},
        "calibration": dict(calibration or {}),
        "leads": leads,
        "recovered_leads": [
            lead for lead, item in leads.items()
            if float(item.get("coverage") or 0.0) >= 0.25
        ],
        "recovered_lead_count": sum(
            1 for item in leads.values()
            if float(item.get("coverage") or 0.0) >= 0.25
        ),
        "global_confidence": float(np.median([
            float(item.get("confidence") or 0.0) for item in leads.values()
        ])),
        "rhythm_strip": None,
        "audit_mapping": {
            "source": "UPSTREAM_CANONICAL_MATRIX",
            "nonrecoverable_regions_preserved_as_nan": True,
        },
    }


def pack_legacy_10s_uv(
    digital_ecg: Mapping[str, Any],
    *,
    target_duration_s: float = 10.0,
    fs: int = 500,
) -> np.ndarray:
    """Create the historical samples×12 NaN-masked matrix for compatibility.

    This adapter is not used for clinical measurement. It exists for the frozen
    R27/WFDB contract and legacy report surfaces.
    """
    n = int(round(float(target_duration_s) * int(fs)))
    out = np.full((n, len(LEADS)), np.nan, dtype=float)
    leads = digital_ecg.get("leads") or {}
    for j, lead in enumerate(LEADS):
        item = leads.get(lead) or {}
        sig = np.asarray(_or_empty(item.get("signal_mv")), dtype=float)
        if sig.size == 0:
            continue
        src_fs = int(item.get("fs") or fs)
        if src_fs != fs:
            src_t = np.arange(sig.size, dtype=float) / float(src_fs)
            dst_n = min(n, int(round((sig.size - 1) / src_fs * fs)) + 1)
            dst_t = np.arange(dst_n, dtype=float) / float(fs)
            finite = np.isfinite(sig)
            if finite.sum() >= 2:
                packed = np.interp(dst_t, src_t[finite], sig[finite])
            else:
                packed = np.full(dst_n, np.nan)
        else:
            packed = sig[:n]
        out[: min(n, packed.size), j] = packed[:n] * 1000.0
    return out


def _json_array(values: Any, *, decimals: int = 6) -> List[Any]:
    arr = np.asarray(values)
    out: List[Any] = []
    for value in arr.reshape(-1):
        if arr.dtype == bool:
            out.append(bool(value))
            continue
        try:
            z = float(value)
        except Exception:
            out.append(None)
            continue
        out.append(round(z, decimals) if math.isfinite(z) else None)
    return out


def digital_ecg_to_jsonable(digital_ecg: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(digital_ecg)
    leads_out: Dict[str, Any] = {}
    for lead, item0 in (digital_ecg.get("leads") or {}).items():
        item = dict(item0)
        item["signal_mv"] = _json_array(_or_empty(item.get("signal_mv")))
        item["time_ms"] = _json_array(_or_empty(item.get("time_ms")), decimals=3)
        item["observed_mask"] = _json_array(
            np.asarray(_or_empty(item.get("observed_mask")), dtype=bool)
        )
        item["interpolated_mask"] = _json_array(
            np.asarray(_or_empty(item.get("interpolated_mask")), dtype=bool)
        )
        item["low_confidence_mask"] = _json_array(
            np.asarray(_or_empty(item.get("low_confidence_mask")), dtype=bool)
        )
        item["confidence_mask"] = _json_array(
            _or_empty(item.get("confidence_mask")), decimals=4
        )
        for key in (
            "confidence", "duration_s", "coverage",
            "longest_contiguous_fraction", "raw_observed_fraction",
            "interpolated_fraction", "rejected_spike_fraction",
            "baseline_y_px",
        ):
            if key in item and item[key] is not None:
                try:
                    item[key] = round(float(item[key]), 6)
                except Exception:
                    pass
        leads_out[str(lead)] = item
    out["leads"] = leads_out
    try:
        out["global_confidence"] = round(float(out.get("global_confidence") or 0.0), 6)
    except Exception:
        pass
    return out


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    px_per_mm: float,
) -> None:
    small = max(1, int(round(px_per_mm)))
    for x in range(0, width, small):
        major = (x // small) % 5 == 0
        shade = 218 if major else 238
        draw.line([(x, 0), (x, height)], fill=(255, shade, shade), width=2 if major else 1)
    for y in range(0, height, small):
        major = (y // small) % 5 == 0
        shade = 218 if major else 238
        draw.line([(0, y), (width, y)], fill=(255, shade, shade), width=2 if major else 1)


def render_reconstructed_ecg_png(
    digital_ecg: Mapping[str, Any],
    *,
    speed_mm_s: float = 25.0,
    gain_mm_mV: float = 10.0,
    width_px: int = 1800,
    px_per_mm: float = 5.0,
) -> bytes:
    """Render the digital signal only for visualization/audit.

    No downstream measurement should consume this raster.
    """
    leads = digital_ecg.get("leads") or {}
    row_h_mm = 32.0
    margin_mm = 8.0
    cols = 2
    rows = 6
    panel_w = width_px // cols
    height = int(round((rows * row_h_mm + 2 * margin_mm) * px_per_mm))
    img = Image.new("RGB", (width_px, height), "white")
    draw = ImageDraw.Draw(img)
    _draw_grid(draw, width=width_px, height=height, px_per_mm=px_per_mm)

    font = ImageFont.load_default()
    layout = LAYOUTS["6x2"]
    for r, lead_row in enumerate(layout):
        for c, lead in enumerate(lead_row):
            item = leads.get(lead) or {}
            sig = np.asarray([
                np.nan if v is None else float(v)
                for v in _or_empty(item.get("signal_mv"))
            ], dtype=float)
            fs = int(item.get("fs") or digital_ecg.get("fs") or 500)
            x0 = int(c * panel_w + margin_mm * px_per_mm)
            x1 = int((c + 1) * panel_w - margin_mm * px_per_mm)
            baseline = int((margin_mm + r * row_h_mm + row_h_mm / 2.0) * px_per_mm)
            draw.text((x0, baseline - int(13 * px_per_mm)), lead, fill=(20, 30, 40), font=font)
            if sig.size < 2:
                continue
            t_s = np.arange(sig.size, dtype=float) / float(fs)
            x = x0 + t_s * float(speed_mm_s) * px_per_mm
            y = baseline - sig * float(gain_mm_mV) * px_per_mm
            finite = np.isfinite(sig) & (x <= x1)
            for a, b in _finite_runs(finite):
                if b - a < 2:
                    continue
                pts = [(int(round(xx)), int(round(yy))) for xx, yy in zip(x[a:b], y[a:b])]
                draw.line(pts, fill=(15, 15, 15), width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_segmentation_overlay_png(
    signal_prob: np.ndarray,
    rows: np.ndarray,
    *,
    width_px: int = 1500,
) -> bytes:
    """Audit overlay in rectified U-Net segmentation coordinates."""
    prob = np.asarray(signal_prob, dtype=float)
    if prob.ndim != 2:
        raise ValueError("signal_prob debe ser 2-D")
    prob = np.nan_to_num(prob, nan=0.0)
    p99 = float(np.percentile(prob, 99.5)) if prob.size else 1.0
    scale = max(p99, 1e-6)
    gray = np.clip(prob / scale * 210.0, 0, 210).astype(np.uint8)
    base = np.stack([255 - gray, 255 - gray, 255 - gray], axis=2)
    img = Image.fromarray(base, mode="RGB")
    if img.width != width_px:
        ratio = width_px / float(img.width)
        img = img.resize((width_px, max(1, int(round(img.height * ratio)))))
    draw = ImageDraw.Draw(img)
    sx = img.width / float(prob.shape[1])
    sy = img.height / float(prob.shape[0])

    for row in np.asarray(rows, dtype=float):
        finite = np.isfinite(row)
        for a, b in _finite_runs(finite):
            if b - a < 2:
                continue
            xs = np.arange(a, b, dtype=float) * sx
            ys = row[a:b] * sy
            pts = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
            draw.line(pts, fill=(0, 180, 200), width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def png_bytes_to_data_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")
