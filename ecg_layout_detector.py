from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_INDEX = {lead: i for i, lead in enumerate(LEAD_ORDER)}

LAYOUT_3X4 = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]

LAYOUT_6X2 = [
    ["I", "V1"],
    ["II", "V2"],
    ["III", "V3"],
    ["aVR", "V4"],
    ["aVL", "V5"],
    ["aVF", "V6"],
]

DETECTOR_VERSION = "MEDCALC_LAYOUT_ROUTER_V1"


def _resize_for_detection(image: Image.Image, max_side: int = 1200) -> tuple[Image.Image, float]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = min(1.0, float(max_side) / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    return image, scale


def _ink_mask(rgb: np.ndarray) -> np.ndarray:
    """Return black/near-neutral printer ink while suppressing red ECG grid."""
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    neutral_dark = (mx < 170) & ((mx - mn) < 75)
    mask = neutral_dark.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _estimate_rotation_deg(rgb: np.ndarray) -> float:
    """Estimate paper skew from the red grid, not from waveform slopes."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    red = (
        (r > 145)
        & (r.astype(np.int16) > g.astype(np.int16) + 18)
        & (r.astype(np.int16) > b.astype(np.int16) + 18)
    )
    edges = cv2.Canny(red.astype(np.uint8) * 255, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(50, int(rgb.shape[1] * 0.08)),
        minLineLength=max(80, int(rgb.shape[1] * 0.20)),
        maxLineGap=max(10, int(rgb.shape[1] * 0.02)),
    )
    if lines is None:
        return 0.0

    angles: list[float] = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [float(v) for v in line]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 10:
            angles.append(angle)

    if len(angles) < 2:
        return 0.0
    return float(np.median(np.asarray(angles, dtype=float)))


def _candidate_rows(mask: np.ndarray) -> dict[str, Any]:
    h, w = mask.shape
    y0, y1 = int(round(h * 0.12)), int(round(h * 0.94))
    x0, x1 = int(round(w * 0.02)), int(round(w * 0.98))
    roi = mask[y0:y1, x0:x1]

    profile = roi.mean(axis=1).astype(np.float32)
    smooth = gaussian_filter1d(profile, sigma=max(1.2, h / 650.0))
    profile_max = float(np.max(smooth)) if smooth.size else 0.0

    peaks, props = find_peaks(
        smooth,
        distance=max(10, int(round(h * 0.07))),
        prominence=max(0.003, 0.08 * profile_max),
        height=max(float(np.percentile(smooth, 55)), 0.006) if smooth.size else 0.006,
    )

    half_band = max(3, int(round(h * 0.012)))
    rows: list[dict[str, Any]] = []
    heights = props.get("peak_heights", np.zeros(len(peaks)))
    prominences = props.get("prominences", np.zeros(len(peaks)))

    for j, peak in enumerate(peaks):
        yy = int(peak + y0)
        ya, yb = max(0, yy - half_band), min(h, yy + half_band + 1)
        band = mask[ya:yb, x0:x1]
        support = float(band.any(axis=0).mean()) if band.size else 0.0
        if support < 0.30:
            continue
        rows.append(
            {
                "center_y": yy,
                "strength": float(heights[j]),
                "prominence": float(prominences[j]),
                "horizontal_support": support,
            }
        )

    return {
        "rows": rows,
        "profile_max": profile_max,
        "search_roi": [x0, y0, x1, y1],
    }


def _classify_row_geometry(rows: list[dict[str, Any]], height: int) -> dict[str, Any]:
    centers = np.asarray([r["center_y"] for r in rows], dtype=int)
    recovery: Optional[str] = None
    rhythm_strip = False
    layout: Optional[str] = None
    primary = centers

    if len(centers) == 7:
        layout, rhythm_strip, primary = "6x2", True, centers[:6]
    elif len(centers) == 6:
        layout, primary = "6x2", centers
    elif len(centers) == 4:
        layout, rhythm_strip, primary = "3x4", True, centers[:3]
    elif len(centers) == 3:
        layout, primary = "3x4", centers
    elif len(centers) == 8:
        top_extra = centers[0] <= int(0.12 * height)
        bottom_extra = centers[-1] >= int(0.90 * height)
        if top_extra and bottom_extra:
            layout, rhythm_strip, primary = "6x2", True, centers[1:7]
            recovery = "TRIM_TOP_HEADER_AND_BOTTOM_RHYTHM_EXTRAS"
    elif len(centers) == 5:
        layout = None

    if layout is None:
        return {
            "layout": None,
            "rhythm_strip": False,
            "primary_centers_y": [],
            "recovery": recovery,
            "geometry_score": 0.0,
            "spacing_cv": None,
        }

    diffs = np.diff(primary.astype(float))
    spacing_cv = (
        float(np.std(diffs) / np.mean(diffs))
        if len(diffs) >= 2 and float(np.mean(diffs)) > 0
        else 0.20
    )

    primary_n = len(primary)
    strengths = [float(r["strength"]) for r in rows[:primary_n]]
    supports = [float(r["horizontal_support"]) for r in rows[:primary_n]]
    max_strength = max(strengths) if strengths else 1.0

    count_score = 1.0 if recovery is None else 0.88
    spacing_score = float(np.clip(1.0 - spacing_cv / 0.20, 0.0, 1.0))
    strength_score = (
        float(np.clip(np.median(strengths) / max(max_strength, 1e-6), 0.0, 1.0))
        if strengths
        else 0.0
    )
    support_score = float(np.clip(np.median(supports), 0.0, 1.0)) if supports else 0.0

    geometry_score = (
        0.55 * count_score
        + 0.25 * spacing_score
        + 0.10 * strength_score
        + 0.10 * support_score
    )

    return {
        "layout": layout,
        "rhythm_strip": bool(rhythm_strip),
        "primary_centers_y": [int(v) for v in primary.tolist()],
        "recovery": recovery,
        "geometry_score": float(np.clip(geometry_score, 0.0, 0.995)),
        "spacing_cv": spacing_cv,
    }


def _active_span_and_regions(
    mask: np.ndarray,
    layout: str,
    primary_centers: list[int],
    rhythm_strip: bool,
    all_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    h, w = mask.shape
    centers = np.asarray(primary_centers, dtype=float)
    spacing = (
        float(np.median(np.diff(centers)))
        if len(centers) >= 2
        else h * 0.12
    )
    row_half = max(4, int(round(0.36 * spacing)))

    primary_mask = np.zeros_like(mask, dtype=bool)
    for center in centers:
        ya = max(0, int(round(center)) - row_half)
        yb = min(h, int(round(center)) + row_half + 1)
        primary_mask[ya:yb] |= mask[ya:yb]

    _, xs = np.where(primary_mask)
    if xs.size < 20:
        ax0, ax1 = int(0.03 * w), int(0.97 * w)
    else:
        ax0, ax1 = int(np.quantile(xs, 0.01)), int(np.quantile(xs, 0.99))
        pad = max(3, int(round(w * 0.008)))
        ax0, ax1 = max(0, ax0 - pad), min(w - 1, ax1 + pad)
    if ax1 <= ax0:
        ax0, ax1 = 0, w - 1

    matrix = LAYOUT_6X2 if layout == "6x2" else LAYOUT_3X4
    n_cols = 2 if layout == "6x2" else 4
    n_rows = 6 if layout == "6x2" else 3

    if len(centers) >= 2:
        mids = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
        edge_half = 0.5 * float(np.median(np.diff(centers)))
        y_edges = [max(0, int(round(centers[0] - edge_half)))]
        y_edges += [int(round(v)) for v in mids]
        y_edges += [min(h, int(round(centers[-1] + edge_half)))]
    else:
        y_edges = [0, h]

    x_edges = [
        int(round(ax0 + (ax1 - ax0 + 1) * k / n_cols))
        for k in range(n_cols + 1)
    ]
    x_edges[0], x_edges[-1] = ax0, ax1 + 1

    regions: list[dict[str, Any]] = []
    for row in range(min(n_rows, len(y_edges) - 1)):
        for col in range(n_cols):
            regions.append(
                {
                    "lead": matrix[row][col],
                    "row": row,
                    "column": col,
                    "bbox": [
                        int(x_edges[col]),
                        int(y_edges[row]),
                        int(x_edges[col + 1]),
                        int(y_edges[row + 1]),
                    ],
                }
            )

    rhythm_region = None
    if rhythm_strip and len(all_rows) > n_rows:
        center = float(all_rows[-1]["center_y"])
        rhythm_region = [
            int(ax0),
            max(0, int(round(center - row_half))),
            int(ax1 + 1),
            min(h, int(round(center + row_half + 1))),
        ]

    return {
        "active_x": [int(ax0), int(ax1)],
        "lead_regions": regions,
        "rhythm_region": rhythm_region,
    }


def _vote_for(layout: Optional[str], confidence: float) -> dict[str, float]:
    if layout == "6x2":
        return {"6x2": float(confidence), "3x4": float(1.0 - confidence)}
    if layout == "3x4":
        return {"3x4": float(confidence), "6x2": float(1.0 - confidence)}
    return {"3x4": 0.0, "6x2": 0.0}


def fuse_layout_votes(
    geometry: Optional[dict[str, float]] = None,
    lead_labels: Optional[dict[str, float]] = None,
    open_ecg: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Fuse available voters using the MEDCALC 0.50/0.35/0.15 policy.

    Missing voters are not treated as negative evidence; weights are
    renormalized across voters that actually produced a result.
    """
    sources = [
        (0.50, geometry, "geometry"),
        (0.35, lead_labels, "lead_labels"),
        (0.15, open_ecg, "open_ecg"),
    ]
    scores = {"3x4": 0.0, "6x2": 0.0}
    denominator = 0.0
    used: list[str] = []

    for weight, vote, source_name in sources:
        if not vote:
            continue
        top = max(float(vote.get("3x4", 0.0)), float(vote.get("6x2", 0.0)))
        if top <= 0:
            continue
        denominator += weight
        used.append(source_name)
        for layout in scores:
            scores[layout] += weight * float(vote.get(layout, 0.0))

    if denominator <= 0:
        return {
            "layout": None,
            "confidence": 0.0,
            "scores": scores,
            "sources_used": [],
        }

    scores = {k: float(v / denominator) for k, v in scores.items()}
    winner = max(scores, key=scores.get)
    return {
        "layout": winner,
        "confidence": float(scores[winner]),
        "scores": scores,
        "sources_used": used,
    }


def detect_ecg_layout(image_or_path: Image.Image | str | Path) -> dict[str, Any]:
    """Detect 3x4 versus 6x2 before U-Net inference using image geometry.

    This detector intentionally does not use OCR as its primary layout signal.
    """
    image = image_or_path if isinstance(image_or_path, Image.Image) else Image.open(image_or_path)
    original_size = list(image.size)
    image, scale = _resize_for_detection(image)
    rgb = np.asarray(image, dtype=np.uint8)
    mask = _ink_mask(rgb)

    row_info = _candidate_rows(mask)
    classified = _classify_row_geometry(row_info["rows"], mask.shape[0])
    layout = classified["layout"]
    geometry_score = float(classified["geometry_score"])
    fused = fuse_layout_votes(geometry=_vote_for(layout, geometry_score))

    if layout is not None:
        region_info = _active_span_and_regions(
            mask,
            layout,
            classified["primary_centers_y"],
            bool(classified["rhythm_strip"]),
            row_info["rows"],
        )
        rows = 6 if layout == "6x2" else 3
        columns = 2 if layout == "6x2" else 4
        regions = region_info["lead_regions"]
    else:
        rows = columns = None
        region_info = {
            "active_x": [0, mask.shape[1] - 1],
            "lead_regions": [],
            "rhythm_region": None,
        }
        regions = []

    confidence = float(fused["confidence"])
    if layout is None or confidence < 0.65:
        route = "UNKNOWN"
    elif confidence >= 0.85:
        route = (
            "6X2_ACTIVE_SPAN_CANONICALIZER"
            if layout == "6x2"
            else "STANDARD_3X4_DIGITIZER"
        )
    else:
        route = "AMBIGUOUS_LAYOUT_RESOLVER"

    return {
        "detector_version": DETECTOR_VERSION,
        "layout": layout,
        "confidence": confidence,
        "rows": rows,
        "columns": columns,
        "leads_detected": int(len(regions)),
        "rhythm_strip": bool(classified.get("rhythm_strip", False)),
        "rotation_deg": _estimate_rotation_deg(rgb),
        "route": route,
        "geometry_score": geometry_score,
        "lead_label_score": None,
        "open_ecg_score": None,
        "vote_scores": fused["scores"],
        "vote_sources_used": fused["sources_used"],
        "row_centers_y": [int(r["center_y"]) for r in row_info["rows"]],
        "primary_centers_y": classified.get("primary_centers_y", []),
        "row_spacing_cv": classified.get("spacing_cv"),
        "layout_recovery": classified.get("recovery"),
        "active_x": region_info["active_x"],
        "lead_regions": regions,
        "rhythm_region": region_info["rhythm_region"],
        "detection_image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "original_image_size": [int(original_size[0]), int(original_size[1])],
        "scale_from_original": float(scale),
    }


def _interpolate_preserving_nan(row: np.ndarray, target_n: int) -> np.ndarray:
    row = np.asarray(row, dtype=np.float64)
    n = row.size
    out = np.full(target_n, np.nan, dtype=np.float64)
    finite = np.isfinite(row)
    if int(finite.sum()) < 2:
        return out

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_n, dtype=np.float64)
    xf = x[finite]
    yf = row[finite]
    valid_target = (x_new >= xf[0]) & (x_new <= xf[-1])
    out[valid_target] = np.interp(x_new[valid_target], xf, yf)
    return out


def canonicalize_extracted_rows(
    raw_lines: Any,
    *,
    avg_pixel_per_mm: float,
    layout: str,
    rhythm_strip: bool,
    target_num_samples: int = 5000,
    required_valid_samples: int = 2,
    active_x: Optional[list[int]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map extracted physical rows to the frozen 12-lead 10 s grid.

    This is the deterministic MEDCALC canonicalizer used after a high-confidence
    preflight layout decision. It preserves observed time spans: 6x2 leads occupy
    5 s, 3x4 leads occupy 2.5 s, and a long lead-II strip occupies 10 s.
    Unprinted portions remain NaN.
    """
    try:
        import torch

        if isinstance(raw_lines, torch.Tensor):
            rows = raw_lines.detach().cpu().numpy().astype(np.float64)
        else:
            rows = np.asarray(raw_lines, dtype=np.float64)
    except Exception:
        rows = np.asarray(raw_lines, dtype=np.float64)

    if rows.ndim != 2:
        raise ValueError(f"raw_lines debe ser 2-D; recibido {rows.shape}.")
    if not np.isfinite(float(avg_pixel_per_mm)) or float(avg_pixel_per_mm) <= 0:
        raise ValueError("avg_pixel_per_mm inválido.")
    if layout not in {"3x4", "6x2"}:
        raise ValueError(f"Layout no soportado: {layout}")

    if active_x is not None and len(active_x) == 2:
        x0 = max(0, int(active_x[0]))
        x1 = min(rows.shape[1] - 1, int(active_x[1]))
        if x1 > x0:
            rows = rows[:, x0 : x1 + 1]

    row_means = np.nanmean(rows, axis=1)
    order = np.argsort(np.nan_to_num(row_means, nan=np.inf))
    rows = rows[order]

    expected_primary = 6 if layout == "6x2" else 3
    if rows.shape[0] < expected_primary:
        raise RuntimeError(
            f"Filas insuficientes para {layout}: {rows.shape[0]} < {expected_primary}."
        )

    if rhythm_strip and rows.shape[0] >= expected_primary + 1:
        primary_rows = rows[:expected_primary]
        rhythm_row = rows[-1]
    else:
        primary_rows = rows[:expected_primary]
        rhythm_row = None

    work = primary_rows.copy()
    if rhythm_row is not None:
        work = np.vstack([work, rhythm_row[None, :]])

    means = np.nanmean(work, axis=1, keepdims=True)
    work = -(work - means) * (0.1 / float(avg_pixel_per_mm)) * 1000.0

    valid_per_col = np.sum(np.isfinite(work), axis=0)
    valid_cols = np.flatnonzero(valid_per_col >= int(required_valid_samples))
    if valid_cols.size >= 2:
        work = work[:, int(valid_cols[0]) : int(valid_cols[-1]) + 1]

    rows_500 = np.vstack(
        [
            _interpolate_preserving_nan(row, int(target_num_samples))
            for row in work
        ]
    )

    canonical = np.full((12, int(target_num_samples)), np.nan, dtype=np.float64)
    matrix = LAYOUT_6X2 if layout == "6x2" else LAYOUT_3X4
    n_cols = 2 if layout == "6x2" else 4
    edges = [int(round(target_num_samples * k / n_cols)) for k in range(n_cols + 1)]
    edges[0], edges[-1] = 0, int(target_num_samples)

    for r, layout_row in enumerate(matrix):
        for c, lead in enumerate(layout_row):
            a, b = edges[c], edges[c + 1]
            canonical[LEAD_INDEX[lead], a:b] = rows_500[r, a:b]

    if rhythm_row is not None and rows_500.shape[0] > expected_primary:
        canonical[LEAD_INDEX["II"], :] = rows_500[expected_primary, :]

    coverage = np.isfinite(canonical).mean(axis=1)
    meta = {
        "canonicalizer": (
            "6X2_ACTIVE_SPAN_CANONICALIZER"
            if layout == "6x2"
            else "3X4_ACTIVE_SPAN_CANONICALIZER"
        ),
        "layout": layout,
        "rhythm_strip": bool(rhythm_row is not None),
        "source_row_count": int(rows.shape[0]),
        "primary_row_count": int(expected_primary),
        "target_num_samples": int(target_num_samples),
        "coverage_by_lead": {
            lead: float(coverage[i]) for i, lead in enumerate(LEAD_ORDER)
        },
    }
    return canonical, meta


# ---------------------------------------------------------------------------
# Post-segmentation row geometry
# ---------------------------------------------------------------------------
# The pre-U-Net detector decides 3x4 versus 6x2. These helpers then use the
# segmentation U-Net probability map only to locate the exact physical rows and
# active horizontal span. Lead-name OCR/U-Net is not required on a confident
# route.


def _longest_true_run(mask: np.ndarray) -> Optional[tuple[int, int]]:
    best: Optional[tuple[int, int]] = None
    start: Optional[int] = None
    for i, value in enumerate(np.asarray(mask, dtype=bool)):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(mask) - 1):
            end = i if value and i == len(mask) - 1 else i - 1
            item = (int(start), int(end))
            if best is None or item[1] - item[0] > best[1] - best[0]:
                best = item
            start = None
    return best


def detect_signal_active_x(
    signal_prob: np.ndarray,
    threshold: float = 0.12,
) -> tuple[int, int, dict[str, Any]]:
    from scipy.ndimage import binary_closing

    prob = np.asarray(signal_prob, dtype=np.float32)
    h, w = prob.shape
    mask = prob >= float(threshold)
    col = mask.mean(axis=0).astype(np.float32)
    smooth = gaussian_filter1d(col, sigma=max(2.0, w / 500.0))

    positive = smooth[smooth > 0]
    if len(positive) == 0:
        return int(0.03 * w), int(0.92 * w), {"fallback": True}

    cutoff = max(0.0025, float(np.percentile(positive, 8)) * 0.40)
    active = smooth > cutoff
    close_width = max(5, int(round(w * 0.025)))
    active = binary_closing(active, structure=np.ones(close_width, dtype=bool))

    run = _longest_true_run(active)
    if run is None or (run[1] - run[0] + 1) < 0.55 * w:
        x0, x1 = int(0.03 * w), int(0.92 * w)
        fallback = True
    else:
        x0, x1 = run
        fallback = False

    return int(x0), int(x1), {
        "threshold": float(cutoff),
        "fallback": bool(fallback),
        "span_fraction": float((x1 - x0 + 1) / max(w, 1)),
    }


def _quarter_signal_peaks(mask: np.ndarray, x0: int, x1: int) -> list[dict[str, Any]]:
    h, _ = mask.shape
    y0 = int(round(0.02 * h))
    y1 = int(round(0.89 * h))

    results: list[dict[str, Any]] = []
    for quarter in range(4):
        xa = int(round(x0 + (x1 - x0 + 1) * quarter / 4.0))
        xb = int(round(x0 + (x1 - x0 + 1) * (quarter + 1) / 4.0))
        xb = max(xa + 2, min(xb, mask.shape[1]))

        profile = mask[y0:y1, xa:xb].mean(axis=1).astype(np.float32)
        profile = gaussian_filter1d(profile, sigma=max(1.0, h / 900.0))

        if float(profile.max()) <= 1e-9:
            peaks = np.array([], dtype=int)
            heights = np.array([], dtype=float)
            prominences = np.array([], dtype=float)
        else:
            peaks, props = find_peaks(
                profile,
                distance=max(8, int(round(h * 0.065))),
                prominence=max(0.004, 0.055 * float(profile.max())),
                height=max(float(np.percentile(profile, 65)), 0.012),
            )
            heights = props.get("peak_heights", np.zeros(len(peaks)))
            prominences = props.get("prominences", np.zeros(len(peaks)))

        results.append(
            {
                "quarter": int(quarter),
                "x0": int(xa),
                "x1": int(xb),
                "peaks_y": [int(p + y0) for p in peaks],
                "heights": [float(v) for v in heights],
                "prominences": [float(v) for v in prominences],
                "count": int(len(peaks)),
            }
        )
    return results


def _primary_signal_centers(
    quarter_info: list[dict[str, Any]],
    expected: int,
    mask: np.ndarray,
    x0: int,
    x1: int,
) -> tuple[list[float], str]:
    exact = [q["peaks_y"] for q in quarter_info if q["count"] == expected]
    if exact:
        matrix = np.asarray(exact, dtype=float)
        centers = np.median(matrix, axis=0)
        return [float(v) for v in centers], "QUARTER_RANK_MEDIAN"

    h, _ = mask.shape
    y0 = int(round(0.02 * h))
    y1 = int(round(0.89 * h))
    profile = mask[y0:y1, x0 : x1 + 1].mean(axis=1).astype(np.float32)
    profile = gaussian_filter1d(profile, sigma=max(1.0, h / 900.0))

    peaks, props = find_peaks(
        profile,
        distance=max(8, int(round(h * (0.060 if expected == 6 else 0.120)))),
        prominence=max(0.004, 0.050 * float(profile.max())),
        height=max(float(np.percentile(profile, 60)), 0.010),
    )
    if len(peaks) < expected:
        raise RuntimeError(
            f"Layout {expected} filas, pero la máscara U-Net sólo produjo "
            f"{len(peaks)} bandas globales."
        )

    scores = props.get("peak_heights", np.ones(len(peaks))) + props.get(
        "prominences", np.zeros(len(peaks))
    )
    chosen = np.argsort(scores)[-expected:]
    centers = np.sort(peaks[chosen] + y0)
    return [float(v) for v in centers], "GLOBAL_STRONGEST_PEAKS"


def _find_signal_rhythm_center(
    mask: np.ndarray,
    centers: list[float],
    x0: int,
    x1: int,
) -> Optional[float]:
    h, _ = mask.shape
    if not centers:
        return None

    spacing = (
        float(np.median(np.diff(np.asarray(centers, dtype=float))))
        if len(centers) >= 2
        else h * 0.12
    )
    start = int(round(centers[-1] + 0.42 * spacing))
    end = int(round(0.995 * h))
    if end - start < 10:
        return None

    profile = mask[start:end, x0 : x1 + 1].mean(axis=1).astype(np.float32)
    profile = gaussian_filter1d(profile, sigma=max(1.0, h / 900.0))
    if float(profile.max()) < 0.010:
        return None

    peaks, props = find_peaks(
        profile,
        distance=max(8, int(round(h * 0.055))),
        prominence=max(0.003, 0.04 * float(profile.max())),
        height=max(float(np.percentile(profile, 60)), 0.008),
    )
    if len(peaks) == 0:
        candidate = start + int(np.argmax(profile))
    else:
        heights = props.get("peak_heights", np.ones(len(peaks)))
        candidate = start + int(peaks[int(np.argmax(heights))])

    half_band = max(3, int(round(0.018 * h)))
    ya = max(0, candidate - half_band)
    yb = min(h, candidate + half_band + 1)
    support = mask[ya:yb, x0 : x1 + 1].any(axis=0).mean()
    if float(support) < 0.35:
        return None
    return float(candidate)


def detect_rows_from_signal_probability(
    signal_prob: np.ndarray,
    *,
    layout: str,
    rhythm_strip_hint: bool,
    threshold: float = 0.12,
) -> dict[str, Any]:
    if layout not in {"3x4", "6x2"}:
        raise ValueError(f"Layout no soportado: {layout}")

    prob = np.asarray(signal_prob, dtype=np.float32)
    mask = prob >= float(threshold)
    x0, x1, active_debug = detect_signal_active_x(prob, threshold=threshold)
    quarters = _quarter_signal_peaks(mask, x0, x1)
    expected = 6 if layout == "6x2" else 3
    centers, method = _primary_signal_centers(quarters, expected, mask, x0, x1)

    rhythm_center = (
        _find_signal_rhythm_center(mask, centers, x0, x1)
        if rhythm_strip_hint
        else None
    )

    return {
        "layout": layout,
        "active_x": [int(x0), int(x1)],
        "active_x_debug": active_debug,
        "primary_centers_y": [float(v) for v in centers],
        "primary_center_method": method,
        "rhythm_center_y": rhythm_center,
        "quarter_counts": [int(q["count"]) for q in quarters],
        "quarter_details": quarters,
    }


def _merge_line_cluster(lines: np.ndarray, indices: list[int]) -> np.ndarray:
    subset = np.asarray(lines[indices], dtype=float)
    out = np.full(subset.shape[1], np.nan, dtype=float)
    for x in range(subset.shape[1]):
        vals = subset[:, x]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[x] = float(np.median(vals))
    return out


def _prepare_candidate_lines(
    lines: np.ndarray,
    active_x: list[int],
    height: int,
) -> list[dict[str, Any]]:
    x0, x1 = [int(v) for v in active_x]
    records: list[dict[str, Any]] = []
    for i, line in enumerate(np.asarray(lines, dtype=float)):
        valid = np.isfinite(line)
        if int(valid.sum()) < max(30, int(0.10 * (x1 - x0 + 1))):
            continue
        active_valid = valid[x0 : x1 + 1]
        active_fraction = float(active_valid.mean()) if len(active_valid) else 0.0
        if active_fraction < 0.12:
            continue
        ymed = float(np.nanmedian(line[x0 : x1 + 1]))
        if not (0 <= ymed < height):
            continue
        records.append(
            {
                "index": int(i),
                "median_y": ymed,
                "active_fraction": active_fraction,
            }
        )
    records.sort(key=lambda rec: rec["median_y"])
    return records


def _assign_lines_to_centers(
    lines: np.ndarray,
    records: list[dict[str, Any]],
    centers: list[float],
    height: int,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    if not centers:
        return {}, {"status": "NO_CENTERS"}

    spacing = (
        float(np.median(np.diff(np.asarray(centers, dtype=float))))
        if len(centers) >= 2
        else height * 0.12
    )

    clusters: list[dict[str, Any]] = []
    tolerance = max(8.0, 0.18 * spacing)
    for rec in records:
        if not clusters or abs(rec["median_y"] - clusters[-1]["median_y"]) > tolerance:
            clusters.append(
                {
                    "members": [rec["index"]],
                    "ys": [rec["median_y"]],
                    "coverage": [rec["active_fraction"]],
                    "median_y": rec["median_y"],
                }
            )
        else:
            clusters[-1]["members"].append(rec["index"])
            clusters[-1]["ys"].append(rec["median_y"])
            clusters[-1]["coverage"].append(rec["active_fraction"])
            clusters[-1]["median_y"] = float(np.median(clusters[-1]["ys"]))

    merged: list[dict[str, Any]] = []
    for cluster in clusters:
        line = _merge_line_cluster(lines, cluster["members"])
        merged.append(
            {
                "line": line,
                "median_y": float(np.nanmedian(line)),
                "coverage": float(np.mean(np.isfinite(line))),
                "members": cluster["members"],
            }
        )

    if not merged:
        return {}, {"status": "NO_OFFICIAL_LINES"}

    cost = np.zeros((len(centers), len(merged)), dtype=float)
    for i, center in enumerate(centers):
        for j, rec in enumerate(merged):
            dy = abs(rec["median_y"] - center) / max(spacing, 1e-9)
            coverage_penalty = max(0.0, 0.45 - rec["coverage"])
            cost[i, j] = dy + 0.5 * coverage_penalty

    row_idx, col_idx = linear_sum_assignment(cost)
    assigned: dict[int, np.ndarray] = {}
    for i, j in zip(row_idx, col_idx):
        if cost[i, j] <= 0.48:
            assigned[int(i)] = merged[int(j)]["line"]

    return assigned, {
        "status": "OPEN_ECG_SIGNAL_EXTRACTOR_ASSIGNED",
        "cluster_count": int(len(merged)),
        "assigned_count": int(len(assigned)),
        "spacing_px": float(spacing),
        "clusters": [
            {
                "median_y": float(rec["median_y"]),
                "coverage": float(rec["coverage"]),
                "members": [int(v) for v in rec["members"]],
            }
            for rec in merged
        ],
    }


def _weighted_band_fallback(
    signal_prob: np.ndarray,
    center_y: float,
    spacing: float,
) -> np.ndarray:
    prob = np.asarray(signal_prob, dtype=np.float32)
    h, w = prob.shape
    half = max(12, int(round(0.48 * spacing)))
    y0 = max(0, int(round(center_y)) - half)
    y1 = min(h, int(round(center_y)) + half + 1)

    sub = prob[y0:y1]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    weights = np.where(sub >= 0.05, sub, 0.0)
    mass = weights.sum(axis=0)
    line = np.full(w, np.nan, dtype=np.float32)
    cutoff = (
        max(0.08, float(np.percentile(mass[mass > 0], 8)))
        if np.any(mass > 0)
        else 0.08
    )
    good = mass > cutoff
    if np.any(good):
        line[good] = (
            (weights[:, good] * rows).sum(axis=0)
            / np.maximum(mass[good], 1e-9)
        )
    return line


def build_rows_from_signal_probability(
    signal_prob: np.ndarray,
    raw_lines: Any,
    signal_geometry: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    try:
        import torch

        if isinstance(raw_lines, torch.Tensor):
            official = raw_lines.detach().cpu().numpy().astype(np.float64)
        else:
            official = np.asarray(raw_lines, dtype=np.float64)
    except Exception:
        official = np.asarray(raw_lines, dtype=np.float64)

    h, _ = np.asarray(signal_prob).shape
    centers = list(signal_geometry["primary_centers_y"])
    rhythm_center = signal_geometry.get("rhythm_center_y")
    all_centers = centers + (
        [float(rhythm_center)] if rhythm_center is not None else []
    )

    records = _prepare_candidate_lines(
        official,
        signal_geometry["active_x"],
        h,
    )
    assigned, assignment_debug = _assign_lines_to_centers(
        official,
        records,
        all_centers,
        h,
    )

    spacing = (
        float(np.median(np.diff(np.asarray(centers, dtype=float))))
        if len(centers) >= 2
        else h * 0.12
    )

    row_lines: list[np.ndarray] = []
    sources: list[str] = []
    for i, center in enumerate(all_centers):
        if i in assigned:
            row_lines.append(np.asarray(assigned[i], dtype=float))
            sources.append("OPEN_ECG_SIGNAL_EXTRACTOR")
        else:
            row_lines.append(
                _weighted_band_fallback(
                    np.asarray(signal_prob, dtype=np.float32),
                    float(center),
                    float(spacing),
                )
            )
            sources.append("WEIGHTED_BAND_FALLBACK")

    return np.asarray(row_lines, dtype=np.float64), sources, {
        "assignment": assignment_debug,
        "source_count": int(len(row_lines)),
    }
