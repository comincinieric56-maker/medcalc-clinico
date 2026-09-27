from __future__ import annotations

"""Visual audit renderers for calibrated ECG reconstruction.

These functions are deliberately downstream of DigitalECG and are never fed
back into the measurement engine.
"""

import io
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ecg_digital_signal import DigitalECG, LEADS


def _grid_colors() -> tuple[str, str]:
    return "#F4DADA", "#E7B8B8"


def render_reconstructed_ecg_png(
    ecg: DigitalECG,
    *,
    duration_s: float = 10.0,
    pixels_per_mm: float = 3.5,
    row_height_mm: float = 22.0,
    left_margin_mm: float = 12.0,
    top_margin_mm: float = 10.0,
) -> bytes:
    """Render a canonical 12x1 ECG at the signal's physical speed/gain."""
    speed = float(ecg.calibration.speed_mm_per_s)
    gain = float(ecg.calibration.gain_mm_per_mv)
    width_mm = left_margin_mm + float(duration_s) * speed + 8.0
    height_mm = top_margin_mm + len(LEADS) * row_height_mm + 8.0
    width = max(600, int(round(width_mm * pixels_per_mm)))
    height = max(400, int(round(height_mm * pixels_per_mm)))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    minor, major = _grid_colors()
    minor_px = max(1, int(round(pixels_per_mm)))
    major_px = max(minor_px, int(round(5.0 * pixels_per_mm)))
    for x in range(0, width, minor_px):
        draw.line(
            [(x, 0), (x, height)],
            fill=major if x % major_px == 0 else minor,
            width=2 if x % major_px == 0 else 1,
        )
    for y in range(0, height, minor_px):
        draw.line(
            [(0, y), (width, y)],
            fill=major if y % major_px == 0 else minor,
            width=2 if y % major_px == 0 else 1,
        )

    x0 = int(round(left_margin_mm * pixels_per_mm))
    top = int(round(top_margin_mm * pixels_per_mm))
    x_scale = speed * pixels_per_mm
    y_scale = gain * pixels_per_mm

    for row, lead_name in enumerate(LEADS):
        baseline = top + int(round((row + 0.5) * row_height_mm * pixels_per_mm))
        draw.text(
            (3, baseline - int(4.5 * pixels_per_mm)),
            lead_name,
            fill="#13212B",
        )
        item = ecg.leads.get(lead_name)
        if item is None:
            continue
        x = np.asarray(item.signal_mv, dtype=float)
        if x.size < 2:
            continue
        fs = int(item.fs)
        time_s = np.arange(x.size, dtype=float) / float(fs)
        take = time_s <= float(duration_s)
        time_s = time_s[take]
        x = x[take]

        current: list[tuple[int, int]] = []
        for t, value in zip(time_s, x):
            if not np.isfinite(value):
                if len(current) >= 2:
                    draw.line(current, fill="#101820", width=2)
                current = []
                continue
            px = x0 + int(round(float(t) * x_scale))
            py = baseline - int(round(float(value) * y_scale))
            current.append((px, py))
        if len(current) >= 2:
            draw.line(current, fill="#101820", width=2)

    draw.text(
        (x0, 2),
        f"{speed:g} mm/s   {gain:g} mm/mV   {ecg.fs} Hz   DIGITAL RECONSTRUCTION",
        fill="#13212B",
    )
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def render_centerline_overlay_png(
    aligned_image_rgb: Any,
    rows_y_px: np.ndarray,
    *,
    active_x: list[int] | tuple[int, int] | None = None,
    row_sources: list[str] | None = None,
    max_width: int = 1400,
) -> bytes:
    """Overlay extracted centerlines on the perspective-corrected source ROI."""
    if hasattr(aligned_image_rgb, "detach"):
        arr = aligned_image_rgb.detach().cpu().numpy()
    else:
        arr = np.asarray(aligned_image_rgb)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr[:3], (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"aligned_image_rgb shape invalid: {arr.shape}")
    if arr.dtype != np.uint8:
        if np.nanmax(arr) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr[..., :3]).convert("RGB")
    rows = np.asarray(rows_y_px, dtype=float)
    if rows.ndim != 2:
        raise ValueError(f"rows_y_px shape invalid: {rows.shape}")

    # The audit thumbnail returned by the model may have been downscaled from
    # the probability-map width. Scale centerline coordinates consistently.
    scale_x = image.width / float(rows.shape[1])
    if active_x and len(active_x) >= 2:
        x0, x1 = int(active_x[0]), int(active_x[1])
    else:
        x0, x1 = 0, rows.shape[1] - 1

    finite_y = rows[np.isfinite(rows)]
    nominal_h = (
        float(np.nanmax(finite_y) + 1)
        if finite_y.size else float(image.height)
    )
    scale_y = image.height / max(nominal_h, 1.0)

    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    palette = [
        (0, 120, 255, 220),
        (255, 80, 60, 220),
        (0, 160, 100, 220),
        (180, 70, 210, 220),
        (230, 150, 0, 220),
        (0, 170, 190, 220),
        (255, 30, 140, 220),
    ]
    row_sources = list(row_sources or [])

    for r in range(rows.shape[0]):
        line = rows[r]
        color = palette[r % len(palette)]
        current: list[tuple[int, int]] = []
        for xx in range(max(0, x0), min(rows.shape[1], x1 + 1)):
            yy = line[xx]
            if not np.isfinite(yy):
                if len(current) >= 2:
                    draw.line(current, fill=color, width=2)
                current = []
                continue
            current.append((
                int(round(xx * scale_x)),
                int(round(float(yy) * scale_y)),
            ))
        if len(current) >= 2:
            draw.line(current, fill=color, width=2)
        label = f"row {r+1}"
        if r < len(row_sources):
            label += " " + str(row_sources[r]).replace("_", " ")[:24]
        if current:
            draw.text(
                (current[0][0] + 4, max(0, current[0][1] - 12)),
                label,
                fill=color,
            )

    if overlay.width > int(max_width):
        ratio = float(max_width) / overlay.width
        overlay = overlay.resize(
            (int(max_width), max(1, int(round(overlay.height * ratio)))),
            Image.Resampling.LANCZOS,
        )

    out = io.BytesIO()
    overlay.convert("RGB").save(out, format="PNG", optimize=True)
    return out.getvalue()
