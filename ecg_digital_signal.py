from __future__ import annotations

"""Canonical digital ECG signal model.

This module is the boundary between image digitisation and clinical analysis.
Image/layout code may decide *where* a trace is on paper, but downstream ECG
measurements consume calibrated digital signals in millivolts at a known sample
rate.  Missing paper signal remains missing; only short internal gaps may be
interpolated and the interpolation is explicitly tracked in confidence masks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

import math
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
    "standard_12x1": [[lead] for lead in LEADS],
    "cabrera_12x1": [
        ["aVL"], ["I"], ["-aVR"], ["II"], ["aVF"], ["III"],
        ["V1"], ["V2"], ["V3"], ["V4"], ["V5"], ["V6"],
    ],
}


@dataclass
class ECGCalibration:
    mm_per_pixel_x: float
    mm_per_pixel_y: float
    speed_mm_per_s: float = 25.0
    gain_mm_per_mv: float = 10.0
    speed_source: str = "ASSUMED_STANDARD"
    gain_source: str = "ASSUMED_STANDARD"
    grid_source: str = "UNET_GRID"
    confidence: float = 0.55

    @property
    def ms_per_pixel(self) -> float:
        return 1000.0 * float(self.mm_per_pixel_x) / float(self.speed_mm_per_s)

    @property
    def mv_per_pixel(self) -> float:
        return float(self.mm_per_pixel_y) / float(self.gain_mm_per_mv)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mm_per_pixel_x": float(self.mm_per_pixel_x),
            "mm_per_pixel_y": float(self.mm_per_pixel_y),
            "speed_mm_per_s": float(self.speed_mm_per_s),
            "gain_mm_per_mv": float(self.gain_mm_per_mv),
            "ms_per_pixel": float(self.ms_per_pixel),
            "mv_per_pixel": float(self.mv_per_pixel),
            "speed_source": self.speed_source,
            "gain_source": self.gain_source,
            "grid_source": self.grid_source,
            "confidence": float(self.confidence),
        }


@dataclass
class DigitalLead:
    name: str
    signal_mv: np.ndarray
    fs: int
    observed_mask: np.ndarray
    confidence_mask: np.ndarray
    duration_s: float
    source: str = "digitized"
    confidence: float = 0.0
    pixel_x_range: tuple[int, int] | None = None
    baseline_y_px: float | None = None
    interpolated_samples: int = 0
    polarity_factor: float = 1.0
    notes: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        finite = np.isfinite(self.signal_mv)
        return {
            "fs": int(self.fs),
            "duration_s": float(self.duration_s),
            "observed_duration_s": float(finite.sum() / max(self.fs, 1)),
            "source": self.source,
            "confidence": float(self.confidence),
            "observed_fraction": float(finite.mean()) if finite.size else 0.0,
            "pixel_x_range": (
                [int(self.pixel_x_range[0]), int(self.pixel_x_range[1])]
                if self.pixel_x_range is not None else None
            ),
            "baseline_y_px": (
                float(self.baseline_y_px)
                if self.baseline_y_px is not None and math.isfinite(self.baseline_y_px)
                else None
            ),
            "interpolated_samples": int(self.interpolated_samples),
            "polarity_factor": float(self.polarity_factor),
            "notes": list(self.notes),
        }


@dataclass
class DigitalECG:
    leads: dict[str, DigitalLead]
    fs: int
    calibration: ECGCalibration
    layout_source: str
    layout_name: str
    reconstruction_version: str = "MEDCALC_DIGITAL_ECG_V1"

    def to_canonical_matrix_mv(self, duration_s: float = 10.0) -> np.ndarray:
        n = int(round(float(duration_s) * self.fs))
        out = np.full((n, len(LEADS)), np.nan, dtype=np.float64)
        for lead in LEADS:
            item = self.leads.get(lead)
            if item is None:
                continue
            x = np.asarray(item.signal_mv, dtype=float).reshape(-1)
            m = min(n, x.size)
            if m > 0:
                out[:m, LEAD_INDEX[lead]] = x[:m]
        return out

    def to_canonical_matrix_uv(self, duration_s: float = 10.0) -> np.ndarray:
        return self.to_canonical_matrix_mv(duration_s=duration_s) * 1000.0

    def to_summary(self) -> dict[str, Any]:
        return {
            "version": self.reconstruction_version,
            "fs": int(self.fs),
            "units": {"time": "ms", "amplitude": "mV"},
            "layout_name": self.layout_name,
            "layout_source": self.layout_source,
            "calibration": self.calibration.to_dict(),
            "leads": {
                lead: (
                    self.leads[lead].to_summary()
                    if lead in self.leads
                    else {
                        "fs": int(self.fs),
                        "duration_s": 0.0,
                        "observed_duration_s": 0.0,
                        "source": "missing",
                        "confidence": 0.0,
                        "observed_fraction": 0.0,
                        "notes": ["LEAD_NOT_RECOVERED"],
                    }
                )
                for lead in LEADS
            },
        }


def _finite(value: Any) -> float | None:
    try:
        z = float(value)
    except Exception:
        return None
    return z if math.isfinite(z) else None


def resolve_calibration(
    *,
    mm_per_pixel_x: Any,
    mm_per_pixel_y: Any,
    speed_mm_per_s: Any = None,
    gain_mm_per_mv: Any = None,
    speed_source: str | None = None,
    gain_source: str | None = None,
    grid_source: str = "UNET_GRID_PIXEL_SIZE_FINDER",
) -> ECGCalibration:
    """Resolve physical paper calibration without hiding assumptions.

    Grid scale is required. Paper speed and gain are accepted when they are
    plausible; otherwise standard 25 mm/s and 10 mm/mV are used but explicitly
    marked as assumptions and assigned lower confidence.
    """
    px_x = _finite(mm_per_pixel_x)
    px_y = _finite(mm_per_pixel_y)
    if px_x is None or px_y is None or not (0.005 <= px_x <= 2.0) or not (0.005 <= px_y <= 2.0):
        raise ValueError(
            f"Escala de grid inválida: mm_per_pixel_x={mm_per_pixel_x}, "
            f"mm_per_pixel_y={mm_per_pixel_y}."
        )

    speed = _finite(speed_mm_per_s)
    gain = _finite(gain_mm_per_mv)

    speed_ok = bool(speed is not None and 5.0 <= speed <= 100.0)
    gain_ok = bool(gain is not None and 2.0 <= gain <= 40.0)

    if not speed_ok:
        speed = 25.0
        speed_source = "ASSUMED_STANDARD_25_MM_S"
    else:
        speed_source = speed_source or "DETECTED_PRINTED_CALIBRATION"

    if not gain_ok:
        gain = 10.0
        gain_source = "ASSUMED_STANDARD_10_MM_MV"
    else:
        gain_source = gain_source or "DETECTED_PRINTED_CALIBRATION"

    confidence = 0.95 if speed_ok and gain_ok else 0.78 if (speed_ok or gain_ok) else 0.58

    return ECGCalibration(
        mm_per_pixel_x=float(px_x),
        mm_per_pixel_y=float(px_y),
        speed_mm_per_s=float(speed),
        gain_mm_per_mv=float(gain),
        speed_source=str(speed_source),
        gain_source=str(gain_source),
        grid_source=str(grid_source),
        confidence=float(confidence),
    )


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    d = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _fill_small_internal_gaps(
    values: np.ndarray,
    *,
    max_gap_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only short internal gaps bracketed by observed samples."""
    y = np.asarray(values, dtype=float).copy()
    original_finite = np.isfinite(y)
    interpolated = np.zeros(y.size, dtype=bool)
    if y.size < 3 or max_gap_samples <= 0:
        return y, interpolated

    missing = ~original_finite
    for a, b in _finite_runs(missing):
        gap = b - a
        if (
            gap <= int(max_gap_samples)
            and a > 0
            and b < y.size
            and np.isfinite(y[a - 1])
            and np.isfinite(y[b])
        ):
            y[a:b] = np.linspace(y[a - 1], y[b], gap + 2)[1:-1]
            interpolated[a:b] = True
    return y, interpolated


def _centerline_probability(
    signal_prob: np.ndarray | None,
    line_y: np.ndarray,
    x_indices: np.ndarray,
) -> np.ndarray:
    if signal_prob is None:
        return np.where(np.isfinite(line_y), 0.75, 0.0).astype(np.float64)

    prob = np.asarray(signal_prob, dtype=float)
    if prob.ndim != 2 or prob.size == 0:
        return np.where(np.isfinite(line_y), 0.75, 0.0).astype(np.float64)

    h, w = prob.shape
    out = np.zeros(len(line_y), dtype=np.float64)
    for i, (yy, xx) in enumerate(zip(line_y, x_indices)):
        if not np.isfinite(yy):
            continue
        x = max(0, min(w - 1, int(round(float(xx)))))
        y = max(0, min(h - 1, int(round(float(yy)))))
        ya, yb = max(0, y - 2), min(h, y + 3)
        out[i] = float(np.nanmax(prob[ya:yb, x])) if yb > ya else float(prob[y, x])
    return np.clip(out, 0.0, 1.0)


def _resample_with_mask(
    time_s: np.ndarray,
    amplitude_mv: np.ndarray,
    confidence: np.ndarray,
    *,
    target_fs: int,
    nominal_duration_s: float,
    max_gap_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    n = max(1, int(round(float(nominal_duration_s) * int(target_fs))))
    dst_t = np.arange(n, dtype=float) / float(target_fs)

    src = np.asarray(amplitude_mv, dtype=float)
    src_conf = np.asarray(confidence, dtype=float)
    finite = np.isfinite(src) & np.isfinite(time_s)

    out = np.full(n, np.nan, dtype=np.float64)
    out_conf = np.zeros(n, dtype=np.float64)
    observed = np.zeros(n, dtype=bool)

    if finite.sum() >= 2:
        for a, b in _finite_runs(finite):
            if b - a < 2:
                continue
            tt = time_s[a:b]
            yy = src[a:b]
            cc = src_conf[a:b]
            lo, hi = float(tt[0]), float(tt[-1])
            take = (dst_t >= lo) & (dst_t <= hi)
            if not np.any(take):
                continue
            out[take] = np.interp(dst_t[take], tt, yy)
            out_conf[take] = np.interp(dst_t[take], tt, cc)
            observed[take] = True

    # Fill only short internal holes after physical resampling. Interpolated
    # samples stay distinguishable from directly observed samples.
    max_gap = int(round(float(max_gap_ms) * target_fs / 1000.0))
    filled, interp = _fill_small_internal_gaps(out, max_gap_samples=max_gap)
    if np.any(interp):
        idx = np.flatnonzero(interp)
        for j in idx:
            left = max(0, j - max_gap - 1)
            right = min(n, j + max_gap + 2)
            neigh = out_conf[left:right]
            positive = neigh[neigh > 0]
            out_conf[j] = 0.45 * (float(np.median(positive)) if positive.size else 0.5)
    return filled, observed, np.clip(out_conf, 0.0, 1.0), int(interp.sum())


def _normalise_layout_name(layout: str) -> str:
    value = str(layout or "").strip()
    value = value.replace("standard_", "")
    if "+" in value:
        value = value.split("+", 1)[0]
    if value in {"3x4", "6x2", "12x1", "cabrera_12x1"}:
        return value
    if "12x1" in value.lower():
        return "cabrera_12x1" if "cabrera" in value.lower() else "12x1"
    if "6x2" in value.lower():
        return "6x2"
    if "3x4" in value.lower():
        return "3x4"
    return value


def _lead_and_polarity(label: str) -> tuple[str, float]:
    label = str(label)
    if label.startswith("-"):
        return label[1:], -1.0
    return label, 1.0


def reconstruct_digital_ecg_from_rows(
    rows_y_px: np.ndarray,
    *,
    layout: str,
    active_x: Iterable[int],
    calibration: ECGCalibration,
    target_fs: int = 500,
    rhythm_strip: bool = False,
    rhythm_lead: str = "II",
    signal_prob: np.ndarray | None = None,
    layout_confidence: float | None = None,
    row_sources: list[str] | None = None,
    max_gap_ms: float = 20.0,
) -> DigitalECG:
    """Convert segmented centerlines into calibrated per-lead digital signals."""
    rows = np.asarray(rows_y_px, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"rows_y_px debe ser 2-D; recibido {rows.shape}.")

    layout_key = _normalise_layout_name(layout)
    matrix = LAYOUTS.get(layout_key)
    if matrix is None:
        raise ValueError(f"Layout digital no soportado: {layout}")

    expected_rows = len(matrix)
    if rows.shape[0] < expected_rows:
        raise ValueError(
            f"Filas insuficientes para {layout_key}: {rows.shape[0]} < {expected_rows}."
        )

    x0, x1 = [int(v) for v in active_x]
    x0 = max(0, min(rows.shape[1] - 1, x0))
    x1 = max(x0 + 1, min(rows.shape[1] - 1, x1))
    width = x1 - x0 + 1
    n_cols = max(len(r) for r in matrix)
    edges = [
        int(round(x0 + width * k / n_cols))
        for k in range(n_cols + 1)
    ]
    edges[0], edges[-1] = x0, x1 + 1

    layout_conf = float(
        np.clip(
            0.85 if layout_confidence is None else float(layout_confidence),
            0.0,
            1.0,
        )
    )
    row_sources = list(row_sources or [])
    leads: dict[str, DigitalLead] = {}

    for r, layout_row in enumerate(matrix):
        row = np.asarray(rows[r], dtype=float)
        for c, label in enumerate(layout_row):
            lead, polarity = _lead_and_polarity(label)
            if lead not in LEAD_INDEX:
                continue
            a, b = edges[c], edges[c + 1]
            if b - a < 3:
                continue

            line = row[a:b].copy()
            xx = np.arange(a, b, dtype=float)
            finite = np.isfinite(line)
            if finite.sum() < 2:
                continue

            baseline_y = float(np.nanmedian(line[finite]))
            amp_mv = (
                -(line - baseline_y)
                * float(calibration.mm_per_pixel_y)
                / float(calibration.gain_mm_per_mv)
                * float(polarity)
            )
            time_s = (
                (xx - float(a))
                * float(calibration.mm_per_pixel_x)
                / float(calibration.speed_mm_per_s)
            )
            nominal_duration_s = float(
                (b - a)
                * float(calibration.mm_per_pixel_x)
                / float(calibration.speed_mm_per_s)
            )

            conf = _centerline_probability(
                signal_prob,
                line,
                xx.astype(int),
            )
            resampled, observed, conf_out, interpolated_n = _resample_with_mask(
                time_s,
                amp_mv,
                conf,
                target_fs=int(target_fs),
                nominal_duration_s=nominal_duration_s,
                max_gap_ms=float(max_gap_ms),
            )
            finite_out = np.isfinite(resampled)
            coverage = float(finite_out.mean()) if finite_out.size else 0.0
            median_seg_conf = (
                float(np.median(conf_out[finite_out]))
                if np.any(finite_out) else 0.0
            )
            row_source = row_sources[r] if r < len(row_sources) else "CENTERLINE"
            source_factor = 0.96 if row_source == "OPEN_ECG_SIGNAL_EXTRACTOR" else 0.90
            overall = float(np.clip(
                median_seg_conf
                * math.sqrt(max(coverage, 0.0))
                * float(calibration.confidence)
                * layout_conf
                * source_factor,
                0.0,
                1.0,
            ))
            notes: list[str] = []
            if interpolated_n:
                notes.append(f"SMALL_GAPS_INTERPOLATED={interpolated_n}")
            if calibration.speed_source.startswith("ASSUMED"):
                notes.append("PAPER_SPEED_ASSUMED")
            if calibration.gain_source.startswith("ASSUMED"):
                notes.append("GAIN_ASSUMED")

            item = DigitalLead(
                name=lead,
                signal_mv=resampled,
                fs=int(target_fs),
                observed_mask=observed,
                confidence_mask=conf_out,
                duration_s=float(nominal_duration_s),
                source="digitized_centerline",
                confidence=overall,
                pixel_x_range=(int(a), int(b - 1)),
                baseline_y_px=baseline_y,
                interpolated_samples=interpolated_n,
                polarity_factor=float(polarity),
                notes=notes,
            )
            previous = leads.get(lead)
            if previous is None or item.confidence > previous.confidence:
                leads[lead] = item

    # A separately printed rhythm row is a longer observation of one lead and
    # should replace a shorter primary instance for temporal analysis.
    if rhythm_strip and rows.shape[0] > expected_rows and rhythm_lead in LEAD_INDEX:
        r = expected_rows
        row = np.asarray(rows[r], dtype=float)
        line = row[x0:x1 + 1].copy()
        xx = np.arange(x0, x1 + 1, dtype=float)
        finite = np.isfinite(line)
        if finite.sum() >= 2:
            baseline_y = float(np.nanmedian(line[finite]))
            amp_mv = (
                -(line - baseline_y)
                * float(calibration.mm_per_pixel_y)
                / float(calibration.gain_mm_per_mv)
            )
            time_s = (
                (xx - float(x0))
                * float(calibration.mm_per_pixel_x)
                / float(calibration.speed_mm_per_s)
            )
            duration_s = float(
                len(line)
                * float(calibration.mm_per_pixel_x)
                / float(calibration.speed_mm_per_s)
            )
            conf = _centerline_probability(signal_prob, line, xx.astype(int))
            resampled, observed, conf_out, interpolated_n = _resample_with_mask(
                time_s,
                amp_mv,
                conf,
                target_fs=int(target_fs),
                nominal_duration_s=duration_s,
                max_gap_ms=float(max_gap_ms),
            )
            finite_out = np.isfinite(resampled)
            coverage = float(finite_out.mean()) if finite_out.size else 0.0
            median_seg_conf = (
                float(np.median(conf_out[finite_out]))
                if np.any(finite_out) else 0.0
            )
            overall = float(np.clip(
                median_seg_conf
                * math.sqrt(max(coverage, 0.0))
                * float(calibration.confidence)
                * layout_conf
                * 0.94,
                0.0,
                1.0,
            ))
            rhythm_item = DigitalLead(
                name=rhythm_lead,
                signal_mv=resampled,
                fs=int(target_fs),
                observed_mask=observed,
                confidence_mask=conf_out,
                duration_s=duration_s,
                source="digitized_native_rhythm_strip",
                confidence=overall,
                pixel_x_range=(int(x0), int(x1)),
                baseline_y_px=baseline_y,
                interpolated_samples=interpolated_n,
                notes=(
                    [f"SMALL_GAPS_INTERPOLATED={interpolated_n}"]
                    if interpolated_n else []
                ),
            )
            previous = leads.get(rhythm_lead)
            if previous is None or rhythm_item.duration_s > previous.duration_s:
                leads[rhythm_lead] = rhythm_item

    return DigitalECG(
        leads=leads,
        fs=int(target_fs),
        calibration=calibration,
        layout_source="POST_UNET_CENTERLINE_CALIBRATION",
        layout_name=layout_key + ("+1R" if rhythm_strip else ""),
    )


def digital_ecg_from_canonical_uv(
    signal_uv: np.ndarray,
    *,
    fs: int,
    calibration: ECGCalibration,
    layout_name: str,
    layout_source: str,
    confidence_by_lead: dict[str, float] | None = None,
) -> DigitalECG:
    """Compatibility wrapper for already-canonical vendor output.

    This keeps legacy/nonstandard layout support while presenting the same
    DigitalECG interface to the clinical analyzer.
    """
    x = np.asarray(signal_uv, dtype=float)
    if x.ndim != 2 or x.shape[1] != 12:
        raise ValueError(f"Se esperaba samples x 12; recibido {x.shape}.")

    confidence_by_lead = confidence_by_lead or {}
    leads: dict[str, DigitalLead] = {}
    for i, lead in enumerate(LEADS):
        sig = x[:, i] / 1000.0
        finite = np.isfinite(sig)
        if not finite.any():
            continue
        conf = float(np.clip(confidence_by_lead.get(lead, 0.70), 0.0, 1.0))
        leads[lead] = DigitalLead(
            name=lead,
            signal_mv=sig.copy(),
            fs=int(fs),
            observed_mask=finite.copy(),
            confidence_mask=np.where(finite, conf, 0.0).astype(float),
            duration_s=float(len(sig) / fs),
            source="digitized_vendor_canonical",
            confidence=float(conf * calibration.confidence),
            notes=["LEGACY_CANONICAL_COMPATIBILITY_PATH"],
        )

    return DigitalECG(
        leads=leads,
        fs=int(fs),
        calibration=calibration,
        layout_source=str(layout_source),
        layout_name=str(layout_name),
    )
