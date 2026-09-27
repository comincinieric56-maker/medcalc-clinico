from __future__ import annotations

import io
import math
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _grid_color(grid: str) -> tuple[int, int, int]:
    mapping = {
        "red": (244, 205, 205),
        "green": (208, 236, 211),
        "gray": (220, 220, 220),
    }
    return mapping.get(str(grid).lower(), mapping["red"])


def _signal(item: Dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
        dtype=float,
    )


def render_calibrated_ecg_png(
    canonical_signal: Dict[str, Any],
    *,
    paper_speed_mm_per_s: float = 25.0,
    display_gain_mm_per_mv: float = 10.0,
    px_per_mm: float = 3.2,
    grid: str = "red",
    max_duration_s: float = 10.0,
) -> bytes:
    """Render the reconstructed digital signal on standard ECG paper.

    This renderer is audit-only. No clinical measurement is taken from its
    pixels; it visualizes the already calibrated per-lead arrays.
    """
    leads = canonical_signal.get("leads") or {}
    fs = int(canonical_signal.get("fs") or 500)

    duration = 0.0
    for lead in LEADS:
        item = leads.get(lead) or {}
        duration = max(duration, float(item.get("duration_s") or 0.0))
    duration = min(float(max_duration_s), max(duration, 2.5))

    left_mm = 12.0
    right_mm = 5.0
    top_mm = 7.0
    bottom_mm = 5.0
    row_mm = 18.0
    signal_width_mm = duration * float(paper_speed_mm_per_s)
    width_mm = left_mm + signal_width_mm + right_mm
    height_mm = top_mm + len(LEADS) * row_mm + bottom_mm

    w = int(round(width_mm * px_per_mm))
    h = int(round(height_mm * px_per_mm))
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    gc = _grid_color(grid)
    major = tuple(max(0, c - 25) for c in gc)
    for mm in np.arange(0.0, width_mm + 0.001, 1.0):
        x = int(round(mm * px_per_mm))
        is_major = abs((mm / 5.0) - round(mm / 5.0)) < 1e-6
        draw.line((x, 0, x, h), fill=major if is_major else gc, width=2 if is_major else 1)
    for mm in np.arange(0.0, height_mm + 0.001, 1.0):
        y = int(round(mm * px_per_mm))
        is_major = abs((mm / 5.0) - round(mm / 5.0)) < 1e-6
        draw.line((0, y, w, y), fill=major if is_major else gc, width=2 if is_major else 1)

    ink = (25, 25, 25)
    muted = (75, 75, 75)
    x0 = left_mm * px_per_mm
    n_max = int(round(duration * fs))

    for row, lead in enumerate(LEADS):
        baseline = (top_mm + (row + 0.5) * row_mm) * px_per_mm
        item = leads.get(lead) or {}
        sig = _signal(item)
        q = np.asarray(item.get("quality_mask", []), dtype=np.uint8)
        if q.size != sig.size:
            qq = np.zeros(sig.size, dtype=np.uint8)
            qq[: min(len(qq), len(q))] = q[: min(len(qq), len(q))]
            q = qq

        draw.text((2, baseline - 5), lead, fill=ink)
        conf = float(item.get("confidence") or 0.0)
        status = str(item.get("status") or "")
        draw.text(
            (2, baseline + 5),
            f"{status} c={conf:.2f}",
            fill=muted,
        )

        n = min(sig.size, n_max)
        if n < 2:
            draw.text((x0 + 8, baseline - 5), "NO MEDIBLE", fill=(130, 70, 20))
            continue

        t = np.arange(n, dtype=float) / fs
        xx = x0 + t * float(paper_speed_mm_per_s) * px_per_mm
        yy = baseline - sig[:n] * float(display_gain_mm_per_mv) * px_per_mm

        finite = np.isfinite(sig[:n])
        d = np.diff(np.r_[False, finite, False].astype(np.int8))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            pts = [(float(x), float(y)) for x, y in zip(xx[a:b], yy[a:b])]
            draw.line(pts, fill=ink, width=2)

        # Low-confidence/interpolated samples receive an unobtrusive marker
        # beneath the baseline so missing-data provenance remains visible.
        if q.size:
            interp = np.flatnonzero(q[:n] == 1)
            missing = np.flatnonzero(q[:n] == 0)
            if interp.size:
                for idx in interp[:: max(1, len(interp)//80 + 1)]:
                    x = float(xx[idx])
                    draw.line((x, baseline + 5, x, baseline + 7), fill=(180, 125, 20), width=1)
            if missing.size:
                for idx in missing[:: max(1, len(missing)//80 + 1)]:
                    x = float(xx[idx])
                    draw.line((x, baseline + 8, x, baseline + 10), fill=(170, 60, 60), width=1)

    draw.text(
        (int(x0), 2),
        (
            f"RECONSTRUCCIÓN DIGITAL · {paper_speed_mm_per_s:g} mm/s · "
            f"{display_gain_mm_per_mv:g} mm/mV · {fs} Hz"
        ),
        fill=ink,
    )

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
