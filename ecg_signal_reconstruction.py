from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {lead: i for i, lead in enumerate(LEADS)}

LAYOUTS: dict[str, list[list[str]]] = {
    "3x4": [
        ["I", "aVR", "V1", "V4"],
        ["II", "aVL", "V2", "V5"],
        ["III", "aVF", "V3", "V6"],
    ],
    "6x2": [
        ["I", "V1"],
        ["II", "V2"],
        ["III", "V3"],
        ["aVR", "V4"],
        ["aVL", "V5"],
        ["aVF", "V6"],
    ],
    "12x1": [[lead] for lead in LEADS],
}

QUALITY_MISSING = 0
QUALITY_INTERPOLATED = 1
QUALITY_OBSERVED = 2
RECONSTRUCTION_VERSION = "MEDCALC_CALIBRATED_SIGNAL_V2"


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _nearest_standard(
    value: float | None,
    standards: Iterable[float],
    *,
    relative_tolerance: float = 0.20,
) -> float | None:
    if value is None or value <= 0:
        return None
    candidates = [float(v) for v in standards]
    nearest = min(candidates, key=lambda v: abs(v - value))
    if abs(nearest - value) / nearest <= float(relative_tolerance):
        return nearest
    return None


def resolve_calibration(
    pixel_spacing_mm: Dict[str, Any] | None,
    *,
    speed_mm_per_s: float | None = None,
    gain_mm_per_mv: float | None = None,
    speed_source: str | None = None,
    gain_source: str | None = None,
    allow_standard_assumption: bool = True,
) -> Dict[str, Any]:
    """Resolve the physical calibration used by the digital signal contract.

    Pixel spacing comes from the segmented ECG grid after geometric alignment.
    Paper speed/gain should come from printed calibration or explicit metadata.
    When they are absent, MEDCALC may keep the historical 25 mm/s and 10 mm/mV
    assumption for compatibility, but the assumption is explicit and lowers
    calibration confidence instead of being treated as measured truth.
    """
    px = dict(pixel_spacing_mm or {})
    mm_x = _finite_float(px.get("x"))
    mm_y = _finite_float(px.get("y"))
    if mm_x is None or mm_y is None or mm_x <= 0 or mm_y <= 0:
        raise ValueError("GRID_SCALE_UNAVAILABLE: mm/pixel x/y inválidos.")

    speed_raw = _finite_float(speed_mm_per_s)
    gain_raw = _finite_float(gain_mm_per_mv)

    speed = _nearest_standard(speed_raw, [12.5, 25.0, 50.0, 100.0])
    gain = _nearest_standard(gain_raw, [2.5, 5.0, 10.0, 20.0])

    speed_assumed = False
    gain_assumed = False

    if speed is None and allow_standard_assumption:
        speed = 25.0
        speed_assumed = True
    if gain is None and allow_standard_assumption:
        gain = 10.0
        gain_assumed = True

    if speed is None:
        raise ValueError("PAPER_SPEED_UNAVAILABLE.")
    if gain is None:
        raise ValueError("GAIN_UNAVAILABLE.")

    anisotropy = abs(mm_x - mm_y) / max((mm_x + mm_y) / 2.0, 1e-9)
    grid_conf = float(np.clip(0.98 - 0.80 * anisotropy, 0.55, 0.98))
    speed_conf = 0.55 if speed_assumed else 0.98
    gain_conf = 0.55 if gain_assumed else 0.98
    confidence = float(min(grid_conf, speed_conf, gain_conf))

    return {
        "version": RECONSTRUCTION_VERSION,
        "mm_per_pixel_x": float(mm_x),
        "mm_per_pixel_y": float(mm_y),
        "pixels_per_mm_x": float(1.0 / mm_x),
        "pixels_per_mm_y": float(1.0 / mm_y),
        "speed_mm_per_s": float(speed),
        "gain_mm_per_mv": float(gain),
        "speed_source": (
            "STANDARD_ASSUMPTION_25_MM_S"
            if speed_assumed else str(speed_source or "PRINTED_OR_EXTERNAL_CALIBRATION")
        ),
        "gain_source": (
            "STANDARD_ASSUMPTION_10_MM_MV"
            if gain_assumed else str(gain_source or "PRINTED_OR_EXTERNAL_CALIBRATION")
        ),
        "speed_assumed": bool(speed_assumed),
        "gain_assumed": bool(gain_assumed),
        "grid_anisotropy": float(anisotropy),
        "grid_confidence": grid_conf,
        "confidence": confidence,
        "fully_observed_calibration": bool(not speed_assumed and not gain_assumed),
    }


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _fill_small_gaps(
    y: np.ndarray,
    *,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill only short internal gaps and return an interpolation mask."""
    out = np.asarray(y, dtype=float).copy()
    interpolated = np.zeros(out.shape, dtype=bool)
    finite = np.isfinite(out)
    if finite.sum() < 2 or max_gap_samples <= 0:
        return out, interpolated

    missing = ~finite
    for a, b in _finite_runs(missing):
        gap = b - a
        if (
            gap <= int(max_gap_samples)
            and a > 0
            and b < len(out)
            and np.isfinite(out[a - 1])
            and np.isfinite(out[b])
        ):
            out[a:b] = np.linspace(
                float(out[a - 1]),
                float(out[b]),
                gap + 2,
                dtype=float,
            )[1:-1]
            interpolated[a:b] = True
    return out, interpolated


def _remove_isolated_impossible_jumps(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reject isolated centerline spikes without flattening genuine QRS slopes."""
    out = np.asarray(y, dtype=float).copy()
    rejected = np.zeros(out.shape, dtype=bool)
    finite = np.isfinite(out)
    if finite.sum() < 7:
        return out, rejected

    diff = np.abs(np.diff(out))
    fd = diff[np.isfinite(diff)]
    if fd.size == 0:
        return out, rejected
    typical = float(np.median(fd))
    mad = float(np.median(np.abs(fd - typical)))
    dynamic_jump = max(1.5, typical + 12.0 * max(mad, 0.02))

    for i in range(1, len(out) - 1):
        if not (np.isfinite(out[i - 1]) and np.isfinite(out[i]) and np.isfinite(out[i + 1])):
            continue
        neighbour_mid = 0.5 * (out[i - 1] + out[i + 1])
        isolated = abs(out[i] - neighbour_mid)
        continuity = abs(out[i - 1] - out[i + 1])
        if isolated >= dynamic_jump and continuity <= max(0.35, 0.30 * isolated):
            out[i] = np.nan
            rejected[i] = True
    return out, rejected


def _piecewise_resample(
    time_s: np.ndarray,
    values: np.ndarray,
    observed_mask: np.ndarray,
    interpolated_mask: np.ndarray,
    *,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(y) & np.isfinite(t)
    if t.size < 2 or not finite.any():
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=np.uint8),
            np.asarray([], dtype=float),
        )

    duration_s = max(0.0, float(t[-1] - t[0]))
    n = max(2, int(round(duration_s * int(fs))) + 1)
    target_t = np.arange(n, dtype=float) / float(fs)
    target = np.full(n, np.nan, dtype=float)
    quality = np.zeros(n, dtype=np.uint8)

    for a, b in _finite_runs(finite):
        if b - a < 2:
            continue
        seg_t = t[a:b] - t[0]
        seg_y = y[a:b]
        lo = int(max(0, math.ceil(float(seg_t[0]) * fs - 1e-9)))
        hi = int(min(n, math.floor(float(seg_t[-1]) * fs + 1e-9) + 1))
        if hi <= lo:
            continue
        tt = target_t[lo:hi]
        target[lo:hi] = np.interp(tt, seg_t, seg_y)

        source_indices = np.searchsorted(seg_t, tt, side="left")
        source_indices = np.clip(source_indices, 0, len(seg_t) - 1)
        absolute_indices = a + source_indices
        interp_here = interpolated_mask[absolute_indices]
        observed_here = observed_mask[absolute_indices] & ~interp_here
        quality[lo:hi][interp_here] = QUALITY_INTERPOLATED
        quality[lo:hi][observed_here] = QUALITY_OBSERVED

    return target, quality, target_t


def _row_factor(source: str | None) -> float:
    src = str(source or "").upper()
    if "OPEN_ECG" in src:
        return 0.98
    if "WEIGHTED_BAND" in src:
        return 0.90
    if "U-NET" in src or "UNET" in src:
        return 0.92
    return 0.88


def _quality_summary(
    quality: np.ndarray,
    *,
    calibration_confidence: float,
    layout_confidence: float,
    row_source: str | None,
) -> Dict[str, Any]:
    q = np.asarray(quality, dtype=np.uint8)
    if q.size == 0:
        return {
            "confidence": 0.0,
            "observed_fraction": 0.0,
            "interpolated_fraction": 0.0,
            "missing_fraction": 1.0,
        }
    observed = float(np.mean(q == QUALITY_OBSERVED))
    interpolated = float(np.mean(q == QUALITY_INTERPOLATED))
    missing = float(np.mean(q == QUALITY_MISSING))
    support = float(np.clip(observed + 0.55 * interpolated, 0.0, 1.0))
    confidence = float(
        np.clip(
            calibration_confidence
            * max(0.0, min(1.0, layout_confidence))
            * _row_factor(row_source)
            * math.sqrt(max(support, 0.0)),
            0.0,
            1.0,
        )
    )
    return {
        "confidence": confidence,
        "observed_fraction": observed,
        "interpolated_fraction": interpolated,
        "missing_fraction": missing,
    }


def _serialize_signal(values: np.ndarray) -> list[float | None]:
    return [
        None if not math.isfinite(float(v)) else round(float(v), 6)
        for v in np.asarray(values, dtype=float).tolist()
    ]


def _lead_from_segment(
    row_y_px: np.ndarray,
    *,
    x_start_px: int,
    x_end_px: int,
    calibration: Dict[str, Any],
    fs: int,
    max_small_gap_ms: float,
    layout_confidence: float,
    row_source: str | None,
) -> Dict[str, Any]:
    y = np.asarray(row_y_px, dtype=float).reshape(-1)
    x0 = max(0, int(x_start_px))
    x1 = min(len(y), int(x_end_px))
    if x1 - x0 < 3:
        return {
            "signal_mv": [],
            "quality_mask": [],
            "fs": int(fs),
            "duration_s": 0.0,
            "source": "digitized",
            "confidence": 0.0,
            "status": "NOT_MEASURABLE",
            "reason": "SEGMENT_TOO_SHORT",
        }

    seg = y[x0:x1]
    original_observed = np.isfinite(seg)
    if int(original_observed.sum()) < 3:
        return {
            "signal_mv": [],
            "quality_mask": [],
            "fs": int(fs),
            "duration_s": 0.0,
            "source": "digitized",
            "confidence": 0.0,
            "status": "NOT_MEASURABLE",
            "reason": "INSUFFICIENT_CENTERLINE",
        }

    mm_x = float(calibration["mm_per_pixel_x"])
    mm_y = float(calibration["mm_per_pixel_y"])
    speed = float(calibration["speed_mm_per_s"])
    gain = float(calibration["gain_mm_per_mv"])

    baseline_y = float(np.nanmedian(seg))
    signal_mv = -(seg - baseline_y) * mm_y / gain
    signal_mv, rejected = _remove_isolated_impossible_jumps(signal_mv)
    observed_after_reject = original_observed & ~rejected

    source_fs_equiv = speed / max(mm_x, 1e-9)
    max_gap_source = int(round((max_small_gap_ms / 1000.0) * source_fs_equiv))
    signal_mv, interpolated = _fill_small_gaps(
        signal_mv,
        max_gap_samples=max(1, max_gap_source),
    )

    source_time = np.arange(len(signal_mv), dtype=float) * mm_x / speed
    resampled, quality, target_time = _piecewise_resample(
        source_time,
        signal_mv,
        observed_after_reject,
        interpolated,
        fs=int(fs),
    )

    summary = _quality_summary(
        quality,
        calibration_confidence=float(calibration["confidence"]),
        layout_confidence=float(layout_confidence),
        row_source=row_source,
    )

    status = "MEASURED" if summary["observed_fraction"] >= 0.35 else "LOW_QUALITY"
    return {
        "signal_mv": _serialize_signal(resampled),
        "quality_mask": [int(v) for v in quality.tolist()],
        "fs": int(fs),
        "duration_s": round(float(target_time[-1]) if target_time.size else 0.0, 6),
        "source": "digitized",
        "confidence": round(float(summary["confidence"]), 6),
        "status": status,
        "observed_fraction": round(float(summary["observed_fraction"]), 6),
        "interpolated_fraction": round(float(summary["interpolated_fraction"]), 6),
        "missing_fraction": round(float(summary["missing_fraction"]), 6),
        "row_source": str(row_source or "UNSPECIFIED"),
        "baseline_y_px": round(float(baseline_y), 4),
        "rejected_spike_count": int(rejected.sum()),
        "source_pixel_count": int(len(seg)),
        "source_observed_pixel_count": int(original_observed.sum()),
        "max_small_gap_ms": float(max_small_gap_ms),
        "time_origin_ms": 0.0,
        "time_step_ms": round(1000.0 / float(fs), 6),
        "units": {"time": "ms", "amplitude": "mV"},
    }


def _pack_legacy_matrix(
    leads: Dict[str, Dict[str, Any]],
    *,
    fs: int,
    target_duration_s: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(float(target_duration_s) * int(fs)))
    matrix = np.full((n, 12), np.nan, dtype=np.float64)
    quality = np.zeros((n, 12), dtype=np.uint8)
    for j, lead in enumerate(LEADS):
        item = leads.get(lead) or {}
        sig = np.asarray(
            [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
            dtype=float,
        )
        q = np.asarray(item.get("quality_mask", []), dtype=np.uint8)
        if sig.size == 0:
            continue
        m = min(n, int(sig.size))
        matrix[:m, j] = sig[:m]
        if q.size:
            quality[: min(m, int(q.size)), j] = q[: min(m, int(q.size))]
    return matrix, quality


def reconstruct_canonical_ecg(
    physical_rows_y_px: np.ndarray,
    *,
    layout: str,
    rhythm_strip: bool,
    active_x: list[int] | tuple[int, int] | None,
    pixel_spacing_mm: Dict[str, Any],
    speed_mm_per_s: float | None,
    gain_mm_per_mv: float | None,
    fs: int = 500,
    layout_confidence: float = 1.0,
    row_sources: list[str] | None = None,
    speed_source: str | None = None,
    gain_source: str | None = None,
    max_small_gap_ms: float = 40.0,
) -> Dict[str, Any]:
    """Convert segmented centerlines into layout-independent calibrated ECG leads.

    The layout is used only to assign physical row/column ROIs to lead names.
    All downstream consumers receive the same per-lead digital contract.
    """
    base_layout = str(layout or "").split("+", 1)[0]
    if base_layout not in LAYOUTS:
        raise ValueError(f"UNSUPPORTED_LAYOUT_FOR_RECONSTRUCTION: {layout}")

    rows = np.asarray(physical_rows_y_px, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"physical_rows_y_px debe ser 2-D; recibido {rows.shape}")

    calibration = resolve_calibration(
        pixel_spacing_mm,
        speed_mm_per_s=speed_mm_per_s,
        gain_mm_per_mv=gain_mm_per_mv,
        speed_source=speed_source,
        gain_source=gain_source,
        allow_standard_assumption=True,
    )

    matrix_layout = LAYOUTS[base_layout]
    expected_rows = len(matrix_layout)
    if rows.shape[0] < expected_rows:
        raise ValueError(
            f"INSUFFICIENT_ROWS_FOR_{base_layout}: {rows.shape[0]} < {expected_rows}"
        )

    if active_x is None:
        ax0, ax1 = 0, rows.shape[1]
    else:
        ax0 = max(0, int(active_x[0]))
        ax1 = min(rows.shape[1], int(active_x[1]) + 1)
    if ax1 - ax0 < 10:
        raise ValueError("ACTIVE_SIGNAL_WIDTH_TOO_SMALL")

    row_sources = list(row_sources or [])
    leads: Dict[str, Dict[str, Any]] = {}
    n_cols = len(matrix_layout[0])
    x_edges = [
        int(round(ax0 + (ax1 - ax0) * k / n_cols))
        for k in range(n_cols + 1)
    ]
    x_edges[0], x_edges[-1] = ax0, ax1

    for r, lead_row in enumerate(matrix_layout):
        source = row_sources[r] if r < len(row_sources) else None
        for col, lead in enumerate(lead_row):
            item = _lead_from_segment(
                rows[r],
                x_start_px=x_edges[col],
                x_end_px=x_edges[col + 1],
                calibration=calibration,
                fs=int(fs),
                max_small_gap_ms=float(max_small_gap_ms),
                layout_confidence=float(layout_confidence),
                row_source=source,
            )
            item["lead"] = lead
            item["roi"] = {
                "row": int(r),
                "column": int(col),
                "x_start_px": int(x_edges[col]),
                "x_end_px": int(x_edges[col + 1]),
            }
            leads[lead] = item

    # A genuine long rhythm strip is a second observation of lead II. Prefer it
    # over the short layout cell only when it carries more usable temporal data.
    rhythm_index = expected_rows
    if bool(rhythm_strip) and rows.shape[0] > rhythm_index:
        source = row_sources[rhythm_index] if rhythm_index < len(row_sources) else None
        rhythm_item = _lead_from_segment(
            rows[rhythm_index],
            x_start_px=ax0,
            x_end_px=ax1,
            calibration=calibration,
            fs=int(fs),
            max_small_gap_ms=float(max_small_gap_ms),
            layout_confidence=float(layout_confidence),
            row_source=source,
        )
        rhythm_item["lead"] = "II"
        rhythm_item["roi"] = {
            "row": int(rhythm_index),
            "column": 0,
            "x_start_px": int(ax0),
            "x_end_px": int(ax1),
            "rhythm_strip": True,
        }
        current = leads.get("II") or {}
        current_support = float(current.get("duration_s") or 0.0) * float(
            current.get("observed_fraction") or 0.0
        )
        rhythm_support = float(rhythm_item.get("duration_s") or 0.0) * float(
            rhythm_item.get("observed_fraction") or 0.0
        )
        if rhythm_support > current_support:
            rhythm_item["source"] = "digitized_native_rhythm_strip"
            leads["II"] = rhythm_item

    for lead in LEADS:
        leads.setdefault(
            lead,
            {
                "lead": lead,
                "signal_mv": [],
                "quality_mask": [],
                "fs": int(fs),
                "duration_s": 0.0,
                "source": "digitized",
                "confidence": 0.0,
                "status": "NOT_MEASURABLE",
                "reason": "LEAD_NOT_RECOVERED",
            },
        )

    matrix_mv, matrix_quality = _pack_legacy_matrix(leads, fs=int(fs))
    coverage = {
        lead: round(float(np.mean(matrix_quality[:, i] > 0)), 6)
        for i, lead in enumerate(LEADS)
    }

    return {
        "version": RECONSTRUCTION_VERSION,
        "source": "U_NET_SEGMENTATION_CENTERLINE_CALIBRATED",
        "layout_input": str(layout),
        "layout_used_for_roi_assignment_only": True,
        "fs": int(fs),
        "calibration": calibration,
        "leads": leads,
        "lead_order": list(LEADS),
        "legacy_matrix_mv": matrix_mv,
        "legacy_quality_mask": matrix_quality,
        "legacy_target_duration_s": 10.0,
        "coverage_by_lead": coverage,
        "contract": (
            "PER_LEAD_SIGNAL_MV_FIXED_FS_WITH_QUALITY_MASK;"
            "LAYOUT_DECOUPLED_AFTER_ROI_ASSIGNMENT"
        ),
    }


def canonical_to_worker_payload(canonical: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON-safe canonical metadata while keeping matrix adapters private."""
    out = {
        k: v
        for k, v in canonical.items()
        if k not in {"legacy_matrix_mv", "legacy_quality_mask"}
    }
    return out
