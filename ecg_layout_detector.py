from __future__ import annotations

import io
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

LAYOUT_12X1 = [[lead] for lead in LEAD_ORDER]

DETECTOR_VERSION = "MEDCALC_LAYOUT_ROUTER_V2"


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
    # OpenCV may return Hough lines as (N,1,4) or (N,4) depending on build.
    for line in np.asarray(lines).reshape(-1, 4):
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

    if len(centers) == 13:
        layout, rhythm_strip, primary = "12x1", True, centers[:12]
    elif len(centers) == 12:
        layout, primary = "12x1", centers
    elif len(centers) == 7:
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

    if layout == "6x2":
        matrix, n_cols, n_rows = LAYOUT_6X2, 2, 6
    elif layout == "3x4":
        matrix, n_cols, n_rows = LAYOUT_3X4, 4, 3
    elif layout == "12x1":
        matrix, n_cols, n_rows = LAYOUT_12X1, 1, 12
    else:
        raise ValueError(f"Layout no soportado: {layout}")

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
    names = ("3x4", "6x2", "12x1")
    if layout not in names:
        return {name: 0.0 for name in names}
    conf = float(np.clip(confidence, 0.0, 1.0))
    remainder = (1.0 - conf) / 2.0
    return {
        name: (conf if name == layout else remainder)
        for name in names
    }


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
    scores = {"3x4": 0.0, "6x2": 0.0, "12x1": 0.0}
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
        rows = 6 if layout == "6x2" else 3 if layout == "3x4" else 12
        columns = 2 if layout == "6x2" else 4 if layout == "3x4" else 1
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
            if layout == "3x4"
            else "12X1_DIGITAL_RECONSTRUCTION"
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


def detect_ecg_layout_source(
    source_name: str,
    source_bytes: bytes,
    pdf_page_index: int = 0,
) -> dict[str, Any]:
    """Lightweight layout preflight directly from an uploaded photo or PDF."""
    name = (source_name or "").lower()
    if name.endswith(".pdf"):
        import pymupdf

        doc = pymupdf.open(stream=source_bytes, filetype="pdf")
        try:
            if doc.page_count < 1:
                raise ValueError("PDF sin páginas.")
            idx = min(max(int(pdf_page_index), 0), int(doc.page_count) - 1)
            page = doc.load_page(idx)
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(120.0 / 72.0, 120.0 / 72.0),
                alpha=False,
            )
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        finally:
            doc.close()
    else:
        image = ImageOps.exif_transpose(
            Image.open(io.BytesIO(source_bytes))
        ).convert("RGB")

    return detect_ecg_layout(image)


def _interpolate_preserving_nan(row: np.ndarray, target_n: int) -> np.ndarray:
    """Resample each observed run independently without bridging missing ECG.

    The previous implementation interpolated from the first finite sample to the
    last finite sample. Any internal NaN gap was therefore replaced by a
    straight line, creating artificial ramps in the reconstructed ECG. This
    implementation keeps every missing interval missing and only interpolates
    inside contiguous observed runs.
    """
    row = np.asarray(row, dtype=np.float64).reshape(-1)
    n = int(row.size)
    target_n = int(target_n)
    out = np.full(target_n, np.nan, dtype=np.float64)
    if n < 1 or target_n < 1:
        return out

    finite = np.isfinite(row)
    if not finite.any():
        return out

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, target_n, dtype=np.float64)

    transitions = np.diff(
        np.r_[False, finite, False].astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    for start, end in zip(starts, ends):
        start = int(start)
        end = int(end)
        run_x = x[start:end]
        run_y = row[start:end]
        if run_y.size == 0:
            continue

        if run_y.size == 1:
            # Preserve an isolated observed sample without creating support in
            # neighbouring missing columns.
            j = int(np.argmin(np.abs(x_new - run_x[0])))
            out[j] = float(run_y[0])
            continue

        target_mask = (x_new >= run_x[0]) & (x_new <= run_x[-1])
        if np.any(target_mask):
            out[target_mask] = np.interp(
                x_new[target_mask],
                run_x,
                run_y,
            )

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
    if layout not in {"3x4", "6x2", "12x1"}:
        raise ValueError(f"Layout no soportado: {layout}")

    if active_x is not None and len(active_x) == 2:
        x0 = max(0, int(active_x[0]))
        x1 = min(rows.shape[1] - 1, int(active_x[1]))
        if x1 > x0:
            rows = rows[:, x0 : x1 + 1]

    row_means = np.nanmean(rows, axis=1)
    order = np.argsort(np.nan_to_num(row_means, nan=np.inf))
    rows = rows[order]

    expected_primary = 6 if layout == "6x2" else 3 if layout == "3x4" else 12
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
    if layout == "6x2":
        matrix, n_cols = LAYOUT_6X2, 2
    elif layout == "3x4":
        matrix, n_cols = LAYOUT_3X4, 4
    else:
        matrix, n_cols = LAYOUT_12X1, 1
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
            if layout == "3x4"
            else "12X1_ACTIVE_SPAN_CANONICALIZER"
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
        distance=max(
            5,
            int(round(h * (
                0.035 if expected == 12
                else 0.060 if expected == 6
                else 0.120
            ))),
        ),
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
    if layout not in {"3x4", "6x2", "12x1"}:
        raise ValueError(f"Layout no soportado: {layout}")

    prob = np.asarray(signal_prob, dtype=np.float32)
    mask = prob >= float(threshold)
    x0, x1, active_debug = detect_signal_active_x(prob, threshold=threshold)
    quarters = _quarter_signal_peaks(mask, x0, x1)
    expected = 6 if layout == "6x2" else 3 if layout == "3x4" else 12
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


def recover_signal_geometry_from_preflight(
    signal_prob: np.ndarray,
    layout_preflight: dict[str, Any],
    *,
    threshold: float = 0.08,
    aligned_active_x: list[int] | None = None,
) -> dict[str, Any]:
    """Map trusted page geometry into the aligned U-Net probability map.

    The preflight stage supplies the expected physical row order. This helper
    does not fabricate traces: each mapped row is locally re-centered on actual
    U-Net signal probability and later extracted only from segmented support.
    """
    layout = str(layout_preflight.get("layout") or "")
    expected = 6 if layout == "6x2" else 3 if layout == "3x4" else None
    if expected is None:
        raise ValueError(f"PREFLIGHT_LAYOUT_NOT_GUIDABLE:{layout}")

    centers_src = np.asarray(
        layout_preflight.get("primary_centers_y") or [],
        dtype=float,
    )
    det_size = layout_preflight.get("detection_image_size")
    if centers_src.size != expected:
        raise RuntimeError(
            f"PREFLIGHT_CENTER_COUNT_MISMATCH:{centers_src.size}/{expected}"
        )
    if (
        not isinstance(det_size, (list, tuple))
        or len(det_size) != 2
        or float(det_size[0]) <= 0
        or float(det_size[1]) <= 0
    ):
        raise RuntimeError("PREFLIGHT_DETECTION_SIZE_UNAVAILABLE")

    prob = np.asarray(signal_prob, dtype=np.float32)
    if prob.ndim != 2 or prob.size == 0:
        raise RuntimeError("INVALID_SIGNAL_PROBABILITY")
    h, w = prob.shape
    det_w, det_h = float(det_size[0]), float(det_size[1])

    active_src = layout_preflight.get("active_x") or [0, det_w - 1]
    x0 = int(round(float(active_src[0]) * w / det_w))
    x1 = int(round(float(active_src[1]) * w / det_w))
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0, min(w - 1, x1))

    if aligned_active_x is not None:
        if (len(aligned_active_x) != 2
                or not 0 <= aligned_active_x[0] < aligned_active_x[1] < w):
            raise ValueError("INVALID_ALIGNED_ACTIVE_X")
        x0, x1 = map(int, aligned_active_x)

    expected_y = centers_src * float(h) / det_h
    spacing = (
        float(np.median(np.diff(np.sort(expected_y))))
        if expected_y.size >= 2
        else 0.12 * h
    )
    mask = prob >= float(threshold)
    centers: list[float] = []
    support_rows: list[float] = []
    search_half = max(6, int(round(0.34 * max(spacing, 1.0))))
    smooth_n = max(3, int(round(0.04 * max(spacing, 1.0))))
    if smooth_n % 2 == 0:
        smooth_n += 1

    for y_expected in expected_y:
        ya = max(0, int(round(y_expected)) - search_half)
        yb = min(h, int(round(y_expected)) + search_half + 1)
        if yb <= ya:
            raise RuntimeError("PREFLIGHT_ROW_SEARCH_WINDOW_EMPTY")

        # Horizontal persistence of segmented trace is more robust than raw
        # probability magnitude in the presence of ECG grid lines.
        profile = mask[ya:yb, x0 : x1 + 1].mean(axis=1).astype(float)
        if profile.size >= smooth_n:
            kernel = np.ones(smooth_n, dtype=float) / float(smooth_n)
            smooth = np.convolve(profile, kernel, mode="same")
        else:
            smooth = profile
        local = int(np.nanargmax(smooth))
        center = float(ya + local)

        band_half = max(3, int(round(0.18 * max(spacing, 1.0))))
        sy0 = max(0, int(round(center)) - band_half)
        sy1 = min(h, int(round(center)) + band_half + 1)
        support = float(mask[sy0:sy1, x0 : x1 + 1].any(axis=0).mean())
        if support < 0.10:
            raise RuntimeError(
                f"PREFLIGHT_ROW_LOW_UNET_SUPPORT:{support:.3f}"
            )
        centers.append(center)
        support_rows.append(support)

    ordered = np.asarray(centers, dtype=float)
    if np.any(np.diff(ordered) <= 0):
        raise RuntimeError("PREFLIGHT_GUIDED_ROWS_NOT_ORDERED")

    quarters = _quarter_signal_peaks(mask, x0, x1)
    geometry = {
        "layout": layout,
        "active_x": [int(x0), int(x1)],
        "active_x_debug": {
            "source": ("EXTRACTOR_ALIGNED_CANVAS_PIXELS" if aligned_active_x is not None
                       else "PREFLIGHT_GEOMETRY_MAPPED_TO_ALIGNED_UNET"),
            "threshold": float(threshold),
        },
        "primary_centers_y": [float(v) for v in centers],
        "primary_center_method": "PREFLIGHT_GUIDED_LOCAL_UNET_RECENTER",
        "primary_row_support": [round(float(v), 6) for v in support_rows],
        "rhythm_center_y": None,
        "quarter_counts": [int(q["count"]) for q in quarters],
        "quarter_details": quarters,
        "preflight_confidence": float(
            layout_preflight.get("confidence") or 0.0
        ),
    }

    if bool(layout_preflight.get("rhythm_strip")):
        recovered, source = recover_rhythm_center_from_preflight(
            prob,
            geometry,
            layout_preflight,
            threshold=max(0.05, float(threshold)),
        )
        if recovered is not None:
            geometry["rhythm_center_y"] = float(recovered)
            geometry["rhythm_center_recovery"] = source

    return geometry


def recover_rhythm_center_from_preflight(
    signal_prob: np.ndarray,
    signal_geometry: dict[str, Any],
    layout_preflight: dict[str, Any],
    *,
    threshold: float = 0.08,
) -> tuple[float | None, str]:
    """Recover a long rhythm strip when the post-U-Net detector misses it.

    The preflight detector sees the full page before neural alignment and often
    identifies the +1R row reliably.  This helper maps that region into the
    aligned probability-map coordinates, then searches a narrow lower-page band
    for actual signal support.  It never fabricates a waveform; it only supplies
    a row center to the existing observed-pixel extractor.
    """
    existing = signal_geometry.get("rhythm_center_y")
    if existing is not None:
        return float(existing), "SIGNAL_PROBABILITY"

    if not bool(layout_preflight.get("rhythm_strip")):
        return None, "NO_RHYTHM_HINT"

    region = layout_preflight.get("rhythm_region")
    det_size = layout_preflight.get("detection_image_size")
    if (
        not isinstance(region, (list, tuple))
        or len(region) != 4
        or not isinstance(det_size, (list, tuple))
        or len(det_size) != 2
    ):
        return None, "PREFLIGHT_GEOMETRY_UNAVAILABLE"

    prob = np.asarray(signal_prob, dtype=np.float32)
    if prob.ndim != 2 or prob.size == 0:
        return None, "INVALID_SIGNAL_PROBABILITY"

    h, w = prob.shape
    det_w, det_h = float(det_size[0]), float(det_size[1])
    if det_h <= 0 or det_w <= 0:
        return None, "INVALID_PREFLIGHT_SIZE"

    expected_y = 0.5 * (float(region[1]) + float(region[3])) * h / det_h

    centers = np.asarray(signal_geometry.get("primary_centers_y") or [], dtype=float)
    if centers.size:
        spacing = (
            float(np.median(np.diff(np.sort(centers))))
            if centers.size >= 2
            else 0.10 * h
        )
        min_y = float(np.max(centers) + max(3.0, 0.22 * spacing))
    else:
        spacing = 0.10 * h
        min_y = 0.55 * h

    x0, x1 = [int(v) for v in signal_geometry.get("active_x", [0, w - 1])]
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0, min(w - 1, x1))

    search_half = max(8, int(round(0.10 * h)))
    ya = max(int(round(min_y)), int(round(expected_y)) - search_half, 0)
    yb = min(h - 1, int(round(expected_y)) + search_half)
    if yb <= ya:
        return None, "PREFLIGHT_RHYTHM_OUTSIDE_ALIGNED_PAGE"

    mask = prob >= float(threshold)
    profile = mask[ya : yb + 1, x0 : x1 + 1].mean(axis=1).astype(float)
    if profile.size == 0 or not np.isfinite(profile).any():
        return None, "NO_RHYTHM_SUPPORT"

    smooth_n = max(3, int(round(0.012 * h)))
    if smooth_n % 2 == 0:
        smooth_n += 1
    kernel = np.ones(smooth_n, dtype=float) / smooth_n
    smooth = np.convolve(profile, kernel, mode="same")
    candidate = ya + int(np.nanargmax(smooth))

    half_band = max(3, int(round(0.018 * h)))
    sy0 = max(0, candidate - half_band)
    sy1 = min(h, candidate + half_band + 1)
    support = float(mask[sy0:sy1, x0 : x1 + 1].any(axis=0).mean())
    if support < 0.20:
        return None, "PREFLIGHT_RHYTHM_LOW_SUPPORT"

    if candidate <= min_y:
        return None, "PREFLIGHT_RHYTHM_OVERLAPS_PRIMARY_ROWS"

    return float(candidate), "PREFLIGHT_GEOMETRY_RECOVERY"


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
    """Trace a physical ECG row directly from the U-Net probability map.

    This path is intentionally geometric: it converts only pixels supported by
    the segmentation model into a y-coordinate centerline. A vertical proximity
    weight keeps neighbouring rows/text from pulling the centroid away from the
    expected physical row, while still allowing large QRS deflections.
    """
    prob = np.asarray(signal_prob, dtype=np.float32)
    h, w = prob.shape
    half = max(12, int(round(0.48 * spacing)))
    y0 = max(0, int(round(center_y)) - half)
    y1 = min(h, int(round(center_y)) + half + 1)

    sub = prob[y0:y1]
    rows = np.arange(y0, y1, dtype=np.float32)[:, None]
    sigma = max(4.0, 0.42 * float(spacing))
    proximity = np.exp(
        -0.5 * ((rows - float(center_y)) / sigma) ** 2
    ).astype(np.float32)

    # Keep weak-but-coherent trace support. The proximity weighting suppresses
    # most remote grid/text probability without requiring a high hard cutoff.
    weights = np.where(sub >= 0.025, sub * proximity, 0.0)
    mass = weights.sum(axis=0)
    line = np.full(w, np.nan, dtype=np.float32)
    cutoff = (
        max(0.025, float(np.percentile(mass[mass > 0], 5)))
        if np.any(mass > 0)
        else 0.025
    )
    good = mass >= cutoff
    if np.any(good):
        line[good] = (
            (weights[:, good] * rows).sum(axis=0)
            / np.maximum(mass[good], 1e-9)
        )
    return line


def build_weighted_rows_counterfactual(
    signal_prob: np.ndarray,
    signal_geometry: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Development-only all-weighted centerline reconstruction.

    This never changes the primary row selection. It reconstructs the same
    physical rows directly from the aligned U-Net probability map so benchmarks
    can determine whether Open-ECG row extraction itself contributes interval
    distortion.
    """
    prob = np.asarray(signal_prob, dtype=np.float32)
    h, w = prob.shape
    centers = list(signal_geometry.get("primary_centers_y") or [])
    rhythm_center = signal_geometry.get("rhythm_center_y")
    all_centers = centers + (
        [float(rhythm_center)] if rhythm_center is not None else []
    )
    if not all_centers:
        raise RuntimeError("WEIGHTED_COUNTERFACTUAL_NO_ROW_CENTERS")

    spacing = (
        float(np.median(np.diff(np.asarray(centers, dtype=float))))
        if len(centers) >= 2
        else h * 0.12
    )
    active_x = [int(v) for v in signal_geometry.get("active_x") or [0, w - 1]]

    rows: list[np.ndarray] = []
    coverage: list[float] = []
    for center in all_centers:
        line = np.asarray(
            _weighted_band_fallback(prob, float(center), float(spacing)),
            dtype=np.float64,
        ).reshape(-1)
        if int(line.size) != int(w):
            line = _interpolate_preserving_nan(line, int(w))
        rows.append(line)
        coverage.append(_active_line_coverage(line, active_x))

    stacked = np.vstack(rows).astype(np.float64, copy=False)
    return (
        stacked,
        ["WEIGHTED_BAND_COUNTERFACTUAL"] * len(rows),
        {
            "row_count": int(len(rows)),
            "output_shape": [int(v) for v in stacked.shape],
            "active_coverage_by_row": [
                round(float(v), 6) for v in coverage
            ],
            "min_active_coverage": (
                round(float(min(coverage)), 6) if coverage else 0.0
            ),
        },
    )


def _map_extracted_line_to_active_span(
    line: np.ndarray,
    *,
    target_width: int,
    active_x: list[int],
) -> np.ndarray:
    """Map an extractor-local row onto the physical active ECG x-span.

    Open-ECG raw centerlines are commonly emitted on the cropped signal span,
    not on the full aligned U-Net canvas. Stretching that local row to the full
    canvas changes the time scale and systematically lengthens ECG intervals.
    Preserve the trusted physical geometry instead: resample only to active_x
    and leave the non-signal margins unobserved.
    """
    src = np.asarray(line, dtype=np.float64).reshape(-1)
    out = np.full(int(target_width), np.nan, dtype=np.float64)
    if src.size == 0 or int(target_width) <= 0:
        return out

    x0, x1 = [int(v) for v in active_x]
    x0 = max(0, min(int(target_width) - 1, x0))
    x1 = max(x0, min(int(target_width) - 1, x1))
    active_width = int(x1 - x0 + 1)
    if active_width < 2:
        return out

    mapped = (
        src
        if int(src.size) == active_width
        else _interpolate_preserving_nan(src, active_width)
    )
    out[x0 : x1 + 1] = mapped
    return out


def _active_line_coverage(line: np.ndarray, active_x: list[int]) -> float:
    x = np.asarray(line, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    x0, x1 = [int(v) for v in active_x]
    x0 = max(0, min(x.size - 1, x0))
    x1 = max(x0, min(x.size - 1, x1))
    return float(np.isfinite(x[x0 : x1 + 1]).mean())


def _longest_true_run_fraction(mask: np.ndarray) -> float:
    x = np.asarray(mask, dtype=bool).reshape(-1)
    if x.size == 0 or not x.any():
        return 0.0
    transitions = np.diff(np.r_[False, x, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    longest = max((int(b - a) for a, b in zip(starts, ends)), default=0)
    return float(longest / x.size)


def _active_line_longest_run_fraction(
    line: np.ndarray,
    active_x: list[int],
) -> float:
    """Longest contiguous observed fraction inside the active ECG width."""
    x = np.asarray(line, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    x0, x1 = [int(v) for v in active_x]
    x0 = max(0, min(x.size - 1, x0))
    x1 = max(x0, min(x.size - 1, x1))
    return _longest_true_run_fraction(np.isfinite(x[x0 : x1 + 1]))


def _active_half_min_longest_run_fraction(
    line: np.ndarray,
    active_x: list[int],
) -> float:
    """Minimum contiguous observed fraction across the two 6x2 half-rows.

    A 6x2 physical row contains two sequential leads. Total row coverage can look
    acceptable while one half is badly fragmented (the exact V3/V4 failure that
    blocked R27). This metric evaluates the weakest half independently.
    """
    x = np.asarray(line, dtype=float).reshape(-1)
    if x.size == 0:
        return 0.0
    x0, x1 = [int(v) for v in active_x]
    x0 = max(0, min(x.size - 1, x0))
    x1 = max(x0, min(x.size - 1, x1))
    seg = np.isfinite(x[x0 : x1 + 1])
    if seg.size < 4:
        return 0.0
    mid = seg.size // 2
    left = seg[:mid]
    right = seg[mid:]
    return float(
        min(
            _longest_true_run_fraction(left),
            _longest_true_run_fraction(right),
        )
    )


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

    # The Open-ECG extractor and the aligned probability map can have
    # different horizontal lengths after perspective alignment/cropping.
    # Normalise every recovered physical row onto the probability-map width
    # before stacking them. The rows are y-coordinate trajectories; linear
    # interpolation over finite samples preserves their geometry while keeping
    # unobserved leading/trailing regions as NaN.
    target_width = int(np.asarray(signal_prob).shape[1])

    row_lines: list[np.ndarray] = []
    sources: list[str] = []
    source_widths: list[int] = []
    source_quality: list[dict[str, Any]] = []

    active_x = [int(v) for v in signal_geometry["active_x"]]
    rhythm_index = len(centers) if rhythm_center is not None else None

    for i, center in enumerate(all_centers):
        fallback = np.asarray(
            _weighted_band_fallback(
                np.asarray(signal_prob, dtype=np.float32),
                float(center),
                float(spacing),
            ),
            dtype=np.float64,
        ).reshape(-1)
        if int(fallback.size) != target_width:
            fallback = _interpolate_preserving_nan(fallback, target_width)

        official_line = None
        official_cov = 0.0
        if i in assigned:
            official_line = np.asarray(assigned[i], dtype=np.float64).reshape(-1)
            source_widths.append(int(official_line.size))
            if int(official_line.size) != target_width:
                official_line = _map_extracted_line_to_active_span(
                    official_line,
                    target_width=target_width,
                    active_x=active_x,
                )
            official_cov = _active_line_coverage(official_line, active_x)
        else:
            source_widths.append(int(fallback.size))

        fallback_cov = _active_line_coverage(fallback, active_x)
        official_longest_run = (
            _active_line_longest_run_fraction(official_line, active_x)
            if official_line is not None
            else 0.0
        )
        fallback_longest_run = _active_line_longest_run_fraction(
            fallback,
            active_x,
        )
        official_half_run = (
            _active_half_min_longest_run_fraction(official_line, active_x)
            if official_line is not None and len(centers) == 6
            else None
        )
        fallback_half_run = (
            _active_half_min_longest_run_fraction(fallback, active_x)
            if len(centers) == 6
            else None
        )

        # The official Open-ECG row can be correctly centred yet contain only a
        # short fragment. That is exactly what produced a false "+1R observed"
        # with only ~1.9 s of lead II. Prefer the probability-map centerline
        # when it materially recovers more of the *same observed row*.
        use_fallback = official_line is None
        selection_reason = (
            "NO_OFFICIAL_LINE"
            if official_line is None
            else (
                "OFFICIAL_LINE_ACTIVE_SPAN_ALIGNED"
                if source_widths[-1] != target_width
                else "OFFICIAL_LINE_RETAINED"
            )
        )
        if official_line is not None:
            if i == rhythm_index:
                coverage_rescue = bool(
                    official_cov < 0.70
                    and fallback_cov >= max(0.35, official_cov + 0.10)
                )
                contiguity_rescue = bool(
                    official_longest_run < 0.55
                    and fallback_longest_run >= 0.55
                    and fallback_longest_run >= official_longest_run + 0.10
                )
                use_fallback = bool(coverage_rescue or contiguity_rescue)
                if contiguity_rescue:
                    selection_reason = "RHYTHM_CONTIGUITY_RECOVERY"
                elif coverage_rescue:
                    selection_reason = "RHYTHM_COVERAGE_RECOVERY"
            else:
                severe_fragmentation = bool(
                    official_cov < 0.45
                    and fallback_cov >= max(0.35, official_cov + 0.20)
                )
                material_coverage_gain = bool(
                    fallback_cov >= 0.80
                    and fallback_cov >= official_cov + 0.03
                )
                contiguous_half_rescue = bool(
                    official_half_run is not None
                    and fallback_half_run is not None
                    and official_half_run < 0.30
                    and fallback_half_run >= 0.30
                )
                use_fallback = bool(
                    severe_fragmentation
                    or material_coverage_gain
                    or contiguous_half_rescue
                )
                if contiguous_half_rescue:
                    selection_reason = "CONTIGUOUS_HALF_ROW_RESCUE"
                elif severe_fragmentation:
                    selection_reason = "SEVERE_FRAGMENTATION_RECOVERY"
                elif material_coverage_gain:
                    selection_reason = "MATERIAL_COVERAGE_GAIN"

        if use_fallback:
            line = fallback
            source = "WEIGHTED_BAND_FALLBACK"
        else:
            line = official_line
            source = "OPEN_ECG_SIGNAL_EXTRACTOR"

        if line is None or int(line.size) != target_width:
            raise RuntimeError(
                f"No fue posible normalizar la fila {i} a {target_width} columnas."
            )

        selected_cov = _active_line_coverage(line, active_x)
        row_lines.append(line)
        sources.append(source)
        source_quality.append({
            "row_index": int(i),
            "is_rhythm_row": bool(i == rhythm_index),
            "selected_source": source,
            "selection_reason": selection_reason,
            "selected_active_coverage": round(float(selected_cov), 6),
            "official_active_coverage": round(float(official_cov), 6),
            "fallback_active_coverage": round(float(fallback_cov), 6),
            "official_longest_run_fraction": round(
                float(official_longest_run), 6
            ),
            "fallback_longest_run_fraction": round(
                float(fallback_longest_run), 6
            ),
            "official_min_half_longest_run_fraction": (
                round(float(official_half_run), 6)
                if official_half_run is not None else None
            ),
            "fallback_min_half_longest_run_fraction": (
                round(float(fallback_half_run), 6)
                if fallback_half_run is not None else None
            ),
        })

    if not row_lines:
        raise RuntimeError("No se recuperaron filas físicas del ECG.")

    stacked = np.vstack(row_lines).astype(np.float64, copy=False)

    return stacked, sources, {
        "assignment": assignment_debug,
        "source_count": int(len(row_lines)),
        "target_width": int(target_width),
        "source_widths": source_widths,
        "source_quality": source_quality,
        "output_shape": [int(v) for v in stacked.shape],
    }


# ---------------------------------------------------------------------------
# General post-U-Net layout hypothesis router
# ---------------------------------------------------------------------------

def _finite_longest_run_fraction(line: np.ndarray) -> float:
    return _longest_true_run_fraction(np.isfinite(np.asarray(line, dtype=float)))


def evaluate_layout_hypothesis(
    signal_prob: np.ndarray,
    raw_lines: Any,
    *,
    avg_pixel_per_mm: float,
    layout: str,
    threshold: float = 0.12,
) -> dict[str, Any]:
    """Evaluate one standard ECG layout using only observed segmented signal.

    The hypothesis is scored after U-Net segmentation.  No preflight row count
    is allowed to decide the layout.  Geometry, Open-ECG row assignment,
    observed-row coverage, lead recovery and within-lead continuity all
    contribute independently.
    """
    if layout not in {"3x4", "6x2", "12x1"}:
        raise ValueError(f"Layout no soportado: {layout}")

    expected_rows = 6 if layout == "6x2" else 3 if layout == "3x4" else 12
    expected_lead_fraction = 0.50 if layout == "6x2" else 0.25 if layout == "3x4" else 1.0

    geometry = detect_rows_from_signal_probability(
        signal_prob,
        layout=layout,
        rhythm_strip_hint=True,
        threshold=threshold,
    )
    centers = np.asarray(geometry.get("primary_centers_y") or [], dtype=float)
    if centers.size != expected_rows:
        raise RuntimeError(
            f"{layout}: filas U-Net {int(centers.size)} != {expected_rows}."
        )

    rows, sources, row_debug = build_rows_from_signal_probability(
        signal_prob,
        raw_lines,
        geometry,
    )

    rhythm_detected = geometry.get("rhythm_center_y") is not None
    canonical_uv, canonical_meta = canonicalize_extracted_rows(
        rows,
        avg_pixel_per_mm=float(avg_pixel_per_mm),
        layout=layout,
        rhythm_strip=bool(rhythm_detected),
        target_num_samples=5000,
        required_valid_samples=2,
        active_x=geometry.get("active_x"),
    )
    if canonical_uv.shape != (12, 5000):
        raise RuntimeError(
            f"{layout}: forma canónica inesperada {canonical_uv.shape}."
        )

    coverage = np.isfinite(canonical_uv).mean(axis=1)
    recovered_leads = int(np.sum(coverage >= 0.10))
    usable_leads = int(np.sum(coverage >= (0.70 * expected_lead_fraction)))

    spacing = np.diff(np.sort(centers))
    spacing_mean = float(np.mean(spacing)) if spacing.size else 0.0
    spacing_cv = (
        float(np.std(spacing) / spacing_mean)
        if spacing_mean > 0
        else 1.0
    )
    spacing_score = float(np.clip(1.0 - spacing_cv / 0.35, 0.0, 1.0))

    quarter_counts = [
        int(v) for v in (geometry.get("quarter_counts") or [])
    ]
    if quarter_counts:
        quarter_scores = [
            max(0.0, 1.0 - abs(float(n) - expected_rows) / expected_rows)
            for n in quarter_counts
        ]
        quarter_score = float(np.mean(quarter_scores))
    else:
        quarter_score = 0.0

    assignment = row_debug.get("assignment") or {}
    assigned_count = int(assignment.get("assigned_count") or 0)
    assignment_score = float(
        np.clip(assigned_count / max(expected_rows, 1), 0.0, 1.0)
    )

    qualities = [
        q for q in (row_debug.get("source_quality") or [])
        if not bool(q.get("is_rhythm_row"))
    ]
    selected_coverages = np.asarray(
        [float(q.get("selected_active_coverage") or 0.0) for q in qualities],
        dtype=float,
    )
    median_row_coverage = (
        float(np.median(selected_coverages))
        if selected_coverages.size else 0.0
    )
    min_row_coverage = (
        float(np.min(selected_coverages))
        if selected_coverages.size else 0.0
    )
    row_coverage_score = float(
        np.clip((median_row_coverage - 0.25) / 0.65, 0.0, 1.0)
    )

    lead_recovery_score = float(recovered_leads / 12.0)
    lead_continuity = []
    for lead_idx in range(12):
        run_fraction = _finite_longest_run_fraction(canonical_uv[lead_idx])
        lead_continuity.append(
            float(np.clip(run_fraction / expected_lead_fraction, 0.0, 1.0))
        )
    continuity_score = float(np.median(lead_continuity))

    rhythm_quality = 0.0
    if rhythm_detected and qualities:
        rhythm_items = [
            q for q in (row_debug.get("source_quality") or [])
            if bool(q.get("is_rhythm_row"))
        ]
        if rhythm_items:
            rhythm_quality = float(
                np.clip(
                    float(rhythm_items[0].get("selected_active_coverage") or 0.0),
                    0.0,
                    1.0,
                )
            )

    score = float(
        0.24 * quarter_score
        + 0.14 * spacing_score
        + 0.18 * assignment_score
        + 0.16 * row_coverage_score
        + 0.16 * lead_recovery_score
        + 0.12 * continuity_score
    )
    if rhythm_detected:
        score = float(min(1.0, score + 0.02 * rhythm_quality))

    hard_failures: list[str] = []
    if recovered_leads < 10:
        hard_failures.append(f"recovered_leads={recovered_leads}")
    if assigned_count < max(2, expected_rows - 1):
        hard_failures.append(f"assigned_rows={assigned_count}/{expected_rows}")
    if median_row_coverage < 0.35:
        hard_failures.append(
            f"median_row_coverage={median_row_coverage:.3f}"
        )
    if quarter_score < 0.45:
        hard_failures.append(f"quarter_score={quarter_score:.3f}")
    if spacing_score < 0.35:
        hard_failures.append(f"spacing_score={spacing_score:.3f}")

    accepted = bool(score >= 0.66 and not hard_failures)

    return {
        "layout": layout,
        "accepted": accepted,
        "score": round(score, 6),
        "hard_failures": hard_failures,
        "metrics": {
            "expected_rows": int(expected_rows),
            "quarter_counts": quarter_counts,
            "quarter_score": round(quarter_score, 6),
            "spacing_cv": round(spacing_cv, 6),
            "spacing_score": round(spacing_score, 6),
            "assigned_rows": int(assigned_count),
            "assignment_score": round(assignment_score, 6),
            "median_row_coverage": round(median_row_coverage, 6),
            "min_row_coverage": round(min_row_coverage, 6),
            "recovered_leads": int(recovered_leads),
            "usable_leads": int(usable_leads),
            "lead_recovery_score": round(lead_recovery_score, 6),
            "continuity_score": round(continuity_score, 6),
            "rhythm_detected": bool(rhythm_detected),
            "rhythm_quality": round(rhythm_quality, 6),
        },
        "geometry": geometry,
        "row_sources": sources,
        "row_debug": row_debug,
        "canonical_uv": canonical_uv,
        "canonical_meta": canonical_meta,
        "physical_rows_y_px": rows,
    }


def route_layout_hypotheses(
    signal_prob: np.ndarray,
    raw_lines: Any,
    *,
    avg_pixel_per_mm: float,
    threshold: float = 0.12,
    min_winner_score: float = 0.66,
    min_margin: float = 0.07,
) -> dict[str, Any]:
    """Choose 3x4, 6x2 or 12x1 from post-segmentation evidence.

    If evidence is weak or two accepted hypotheses are too close, no layout is
    selected and the caller must use the neural layout identifier/fail closed.
    """
    candidates: list[dict[str, Any]] = []
    errors: dict[str, str] = {}

    for layout in ("3x4", "6x2", "12x1"):
        try:
            item = evaluate_layout_hypothesis(
                signal_prob,
                raw_lines,
                avg_pixel_per_mm=float(avg_pixel_per_mm),
                layout=layout,
                threshold=threshold,
            )
            candidates.append(item)
        except Exception as exc:
            errors[layout] = str(exc)

    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    accepted = [item for item in ranked if bool(item.get("accepted"))]

    selected = None
    margin = None
    decision = "NO_ACCEPTED_HYPOTHESIS"

    if accepted:
        winner = accepted[0]
        runner_score = (
            float(ranked[1].get("score") or 0.0)
            if len(ranked) >= 2 else 0.0
        )
        margin = float(float(winner["score"]) - runner_score)
        competing_accepted = len(accepted) >= 2
        if (
            float(winner["score"]) >= float(min_winner_score)
            and (not competing_accepted or margin >= float(min_margin))
        ):
            selected = winner
            decision = "SELECTED"
        else:
            decision = "AMBIGUOUS_ACCEPTED_HYPOTHESES"

    audit = []
    for item in ranked:
        audit.append(
            {
                "layout": item["layout"],
                "accepted": bool(item["accepted"]),
                "score": float(item["score"]),
                "hard_failures": list(item.get("hard_failures") or []),
                "metrics": item.get("metrics") or {},
            }
        )

    return {
        "router_version": "MEDCALC_POST_UNET_LAYOUT_HYPOTHESES_V2",
        "decision": decision,
        "selected_layout": (
            selected.get("layout") if selected is not None else None
        ),
        "selected_score": (
            float(selected.get("score")) if selected is not None else None
        ),
        "margin": round(float(margin), 6) if margin is not None else None,
        "min_winner_score": float(min_winner_score),
        "min_margin": float(min_margin),
        "candidates": audit,
        "errors": errors,
        "_selected_candidate": selected,
    }


def route_temporal_rhythm_reference(
    signal_prob: np.ndarray,
    raw_lines: Any,
    *,
    avg_pixel_per_mm: float,
    layout_hint: str,
    threshold: float = 0.12,
    min_rhythm_coverage: float = 0.55,
) -> dict[str, Any]:
    """Recover a genuine long rhythm strip independently of full-layout QC.

    The primary 2000 px route has already selected the standard layout.  At
    1200 px we therefore do not require every primary row to be re-confirmed
    before using a separately observed long strip for temporal RR analysis.
    This prevents a weak V3/V4/row assignment from suppressing rhythm when the
    bottom strip itself is well observed.

    Only the long strip is returned.  Other leads are intentionally blanked so
    this route cannot leak into morphology, axis or R27.
    """
    layout = str(layout_hint or "").split("+", 1)[0]
    if layout not in {"3x4", "6x2", "12x1"}:
        raise RuntimeError(
            f"Layout primario no soportado para referencia temporal: {layout_hint}"
        )

    candidate = evaluate_layout_hypothesis(
        signal_prob,
        raw_lines,
        avg_pixel_per_mm=float(avg_pixel_per_mm),
        layout=layout,
        threshold=threshold,
    )

    metrics = candidate.get("metrics") or {}
    geometry = candidate.get("geometry") or {}
    if not bool(metrics.get("rhythm_detected")):
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY_REJECTED: no se detectó tira larga observada."
        )

    rhythm_quality = float(metrics.get("rhythm_quality") or 0.0)
    if rhythm_quality < float(min_rhythm_coverage):
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY_REJECTED: cobertura horizontal insuficiente "
            f"({rhythm_quality:.3f} < {float(min_rhythm_coverage):.3f})."
        )

    canonical = np.asarray(candidate.get("canonical_uv"), dtype=np.float64)
    if canonical.shape != (12, 5000):
        raise RuntimeError(
            f"TEMPORAL_STRIP_ONLY_REJECTED: forma canónica {canonical.shape}."
        )

    ii_index = LEAD_INDEX["II"]
    rhythm = np.asarray(canonical[ii_index], dtype=np.float64)
    finite = np.isfinite(rhythm)
    observed_fraction = float(finite.mean())
    longest_fraction = _longest_true_run_fraction(finite)

    if observed_fraction < float(min_rhythm_coverage):
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY_REJECTED: II observado "
            f"{observed_fraction:.3f} < {float(min_rhythm_coverage):.3f}."
        )
    if longest_fraction < float(min_rhythm_coverage):
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY_REJECTED: tramo continuo II "
            f"{longest_fraction:.3f} < {float(min_rhythm_coverage):.3f}."
        )

    rhythm_only = np.full((12, 5000), np.nan, dtype=np.float64)
    rhythm_only[ii_index] = rhythm

    rhythm_items = [
        q for q in (candidate.get("row_debug") or {}).get("source_quality", [])
        if bool(q.get("is_rhythm_row"))
    ]
    selected_source = (
        rhythm_items[0].get("selected_source")
        if rhythm_items else None
    )

    return {
        "layout_hint": layout,
        "source": "LOW_MEMORY_1200_TEMPORAL_STRIP_ONLY",
        "signal_uv": rhythm_only,
        "rhythm_lead": "II",
        "rhythm_strip_observed": True,
        "rhythm_strip_coverage": round(observed_fraction, 6),
        "rhythm_strip_longest_contiguous_fraction": round(
            longest_fraction,
            6,
        ),
        "rhythm_strip_center_y": geometry.get("rhythm_center_y"),
        "rhythm_row_source": selected_source,
        "candidate_score": float(candidate.get("score") or 0.0),
        "candidate_accepted_as_full_layout": bool(candidate.get("accepted")),
        "candidate_hard_failures": list(candidate.get("hard_failures") or []),
        "candidate_metrics": metrics,
    }
