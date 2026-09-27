from __future__ import annotations

"""Synthetic ECG validation harness for the digital-signal-first pipeline.

The harness generates ECGs with known fiducials, renders them to paper, applies
controlled image degradations and scores MEDCALC output against ground truth.
It is intentionally independent from any one patient ECG.
"""

import argparse
import io
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ecg_digital_signal import (
    DigitalECG,
    DigitalLead,
    LEADS,
    resolve_calibration,
)
from ecg_signal_measurements import measure_digital_ecg


LEAD_SCALE = {
    "I": 0.85,
    "II": 1.00,
    "III": 0.65,
    "aVR": -0.65,
    "aVL": 0.45,
    "aVF": 0.75,
    "V1": 0.55,
    "V2": 0.85,
    "V3": 1.05,
    "V4": 1.20,
    "V5": 1.05,
    "V6": 0.85,
}

ST_DEFAULT_MV = {
    "I": 0.00,
    "II": 0.00,
    "III": -0.08,
    "aVR": 0.05,
    "aVL": 0.00,
    "aVF": -0.10,
    "V1": -0.03,
    "V2": -0.12,
    "V3": -0.14,
    "V4": -0.13,
    "V5": -0.09,
    "V6": -0.04,
}

LAYOUT_ROWS = {
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


def _gaussian(t: np.ndarray, center: float, sigma: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def generate_known_ecg(
    *,
    fs: int = 500,
    duration_s: float = 10.0,
    rr_ms: float = 800.0,
    p_duration_ms: float = 100.0,
    pr_ms: float = 160.0,
    qrs_ms: float = 90.0,
    qt_ms: float = 380.0,
    st_by_lead: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create deterministic 12-lead ECG with known timing and ST values."""
    st_by_lead = dict(ST_DEFAULT_MV if st_by_lead is None else st_by_lead)
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    rr_s = rr_ms / 1000.0
    p_dur = p_duration_ms / 1000.0
    pr_s = pr_ms / 1000.0
    qrs_s = qrs_ms / 1000.0
    qt_s = qt_ms / 1000.0

    qrs_times = np.arange(0.8, duration_s - 0.45, rr_s)
    signals: dict[str, np.ndarray] = {}

    for lead in LEADS:
        scale = float(LEAD_SCALE[lead])
        st = float(st_by_lead.get(lead, 0.0))
        y = np.zeros(n, dtype=float)

        for qrs0 in qrs_times:
            p_on = qrs0 - pr_s
            p_center = p_on + p_dur / 2.0
            p_amp = 0.12 * (1.0 if lead != "aVR" else -0.8) * max(0.45, abs(scale))
            y += _gaussian(t, p_center, max(0.018, p_dur / 5.0), p_amp)

            # QRS fiducials are explicitly tied to qrs0 and qrs0+qrs_s.
            q_c = qrs0 + 0.16 * qrs_s
            r_c = qrs0 + 0.42 * qrs_s
            s_c = qrs0 + 0.72 * qrs_s
            y += _gaussian(t, q_c, max(0.006, 0.08 * qrs_s), -0.12 * abs(scale))
            y += _gaussian(t, r_c, max(0.007, 0.09 * qrs_s), 1.05 * scale)
            y += _gaussian(t, s_c, max(0.007, 0.09 * qrs_s), -0.28 * scale)

            j = qrs0 + qrs_s
            t_on = qrs0 + max(qrs_s + 0.07, 0.18)
            t_peak = qrs0 + min(qt_s - 0.09, 0.30)
            t_end = qrs0 + qt_s

            plateau = (t >= j) & (t <= t_on)
            y[plateau] += st
            decay = (t > t_on) & (t < t_peak)
            if np.any(decay):
                frac = (t[decay] - t_on) / max(t_peak - t_on, 1e-6)
                y[decay] += st * (1.0 - 0.7 * frac)

            t_amp = 0.32 * (1.0 if lead != "aVR" else -1.0) * max(0.50, abs(scale))
            t_sigma = max(0.035, (t_end - t_on) / 5.0)
            y += _gaussian(t, t_peak, t_sigma, t_amp)

        # Low-amplitude deterministic baseline wander.
        y += 0.012 * np.sin(2 * np.pi * 0.18 * t + 0.2 * LEADS.index(lead))
        signals[lead] = y.astype(np.float64)

    ground_truth = {
        "fs": int(fs),
        "duration_s": float(duration_s),
        "rr_ms": float(rr_ms),
        "heart_rate_bpm": float(60000.0 / rr_ms),
        "p_duration_ms": float(p_duration_ms),
        "pr_ms": float(pr_ms),
        "qrs_ms": float(qrs_ms),
        "qt_ms": float(qt_ms),
        "qtc_bazett_ms": float(qt_ms / math.sqrt(rr_ms / 1000.0)),
        "st_j60_mv_by_lead": {lead: float(st_by_lead.get(lead, 0.0)) for lead in LEADS},
        "r_amplitude_reference_mv_by_lead": {
            lead: float(1.05 * LEAD_SCALE[lead]) for lead in LEADS
        },
    }
    return signals, ground_truth


def make_digital_ecg_from_known(
    signals: dict[str, np.ndarray],
    *,
    fs: int = 500,
    speed_mm_s: float = 25.0,
    gain_mm_mv: float = 10.0,
) -> DigitalECG:
    calibration = resolve_calibration(
        mm_per_pixel_x=0.2,
        mm_per_pixel_y=0.2,
        speed_mm_per_s=speed_mm_s,
        gain_mm_per_mv=gain_mm_mv,
        speed_source="SYNTHETIC_GROUND_TRUTH",
        gain_source="SYNTHETIC_GROUND_TRUTH",
        grid_source="SYNTHETIC_GROUND_TRUTH",
    )
    leads = {}
    for lead in LEADS:
        x = np.asarray(signals[lead], dtype=float)
        finite = np.isfinite(x)
        leads[lead] = DigitalLead(
            name=lead,
            signal_mv=x,
            fs=int(fs),
            observed_mask=finite,
            confidence_mask=np.where(finite, 1.0, 0.0),
            duration_s=float(len(x) / fs),
            source="synthetic_ground_truth",
            confidence=1.0,
        )
    return DigitalECG(
        leads=leads,
        fs=int(fs),
        calibration=calibration,
        layout_source="SYNTHETIC_GROUND_TRUTH",
        layout_name="12x1",
    )


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    ppm: float,
    color: str,
) -> None:
    minor = max(1, int(round(ppm)))
    major = max(minor, int(round(5 * ppm)))
    for x in range(0, width, minor):
        draw.line(
            [(x, 0), (x, height)],
            fill=color,
            width=1 if x % major else 2,
        )
    for y in range(0, height, minor):
        draw.line(
            [(0, y), (width, y)],
            fill=color,
            width=1 if y % major else 2,
        )


def render_paper_ecg(
    signals: dict[str, np.ndarray],
    *,
    layout: str = "6x2",
    rhythm_strip: bool = False,
    fs: int = 500,
    speed_mm_s: float = 25.0,
    gain_mm_mv: float = 10.0,
    pixels_per_mm: float = 4.0,
    grid_color: str = "#f3b6bd",
    trace_width: int = 2,
) -> Image.Image:
    layout = str(layout)
    if layout not in LAYOUT_ROWS:
        raise ValueError(f"Unsupported synthetic layout: {layout}")
    rows = LAYOUT_ROWS[layout]
    n_rows = len(rows) + (1 if rhythm_strip else 0)
    page_width_mm = 270.0
    row_height_mm = 25.0
    top_mm = 14.0
    bottom_mm = 12.0
    left_mm = 10.0
    right_mm = 10.0
    width = int(round(page_width_mm * pixels_per_mm))
    height = int(round((top_mm + bottom_mm + n_rows * row_height_mm) * pixels_per_mm))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    _draw_grid(
        draw,
        width=width,
        height=height,
        ppm=pixels_per_mm,
        color=grid_color,
    )

    usable_width_mm = page_width_mm - left_mm - right_mm
    x0_px = int(round(left_mm * pixels_per_mm))
    y0_px = int(round(top_mm * pixels_per_mm))
    amp_px_per_mv = gain_mm_mv * pixels_per_mm

    for r, row_leads in enumerate(rows):
        base_y = y0_px + int(round((r + 0.5) * row_height_mm * pixels_per_mm))
        cols = len(row_leads)
        col_width_mm = usable_width_mm / cols
        seg_duration_s = col_width_mm / speed_mm_s
        for col, lead in enumerate(row_leads):
            sig = np.asarray(signals[lead], dtype=float)
            n = min(len(sig), int(round(seg_duration_s * fs)))
            xa = x0_px + int(round(col * col_width_mm * pixels_per_mm))
            xb = x0_px + int(round((col + 1) * col_width_mm * pixels_per_mm))
            xs = np.linspace(xa, xb - 1, n)
            ys = base_y - sig[:n] * amp_px_per_mv
            pts = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
            if len(pts) >= 2:
                draw.line(pts, fill="#111111", width=int(trace_width), joint="curve")
            draw.text((xa + 3, base_y - int(10 * pixels_per_mm)), lead, fill="#111111")

    if rhythm_strip:
        base_y = y0_px + int(round((len(rows) + 0.5) * row_height_mm * pixels_per_mm))
        lead = "II"
        sig = np.asarray(signals[lead], dtype=float)
        max_duration_s = usable_width_mm / speed_mm_s
        n = min(len(sig), int(round(max_duration_s * fs)))
        xa = x0_px
        xb = x0_px + int(round(usable_width_mm * pixels_per_mm))
        xs = np.linspace(xa, xb - 1, n)
        ys = base_y - sig[:n] * amp_px_per_mv
        pts = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
        draw.line(pts, fill="#111111", width=int(trace_width), joint="curve")
        draw.text((xa + 3, base_y - int(10 * pixels_per_mm)), "II", fill="#111111")

    draw.text(
        (x0_px, max(2, int(2 * pixels_per_mm))),
        f"{speed_mm_s:g} mm/s   {gain_mm_mv:g} mm/mV",
        fill="#111111",
    )
    return image


def degrade_image(image: Image.Image, mode: str) -> Image.Image:
    mode = str(mode or "clean").lower()
    if mode == "clean":
        return image.copy()
    if mode == "rotate":
        return image.rotate(3.0, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.2))
    if mode == "lowres":
        small = image.resize(
            (max(320, image.width // 3), max(240, image.height // 3)),
            Image.Resampling.BILINEAR,
        )
        return small.resize(image.size, Image.Resampling.BILINEAR)
    if mode == "jpeg":
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=42)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if mode == "perspective":
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([
            [0.03 * w, 0.02 * h],
            [0.98 * w, 0.06 * h],
            [0.95 * w, 0.96 * h],
            [0.05 * w, 0.99 * h],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            arr,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return Image.fromarray(warped)
    if mode == "noise":
        arr = np.asarray(image).astype(np.float32)
        rng = np.random.default_rng(12345)
        arr += rng.normal(0.0, 9.0, arr.shape)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if mode == "combined":
        out = degrade_image(image, "perspective")
        out = out.rotate(1.8, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white")
        out = out.filter(ImageFilter.GaussianBlur(radius=0.8))
        return degrade_image(out, "jpeg")
    raise ValueError(f"Unknown degradation: {mode}")


def _abs_error(measured: Any, truth: float) -> float | None:
    try:
        value = float(measured)
    except Exception:
        return None
    return abs(value - float(truth)) if math.isfinite(value) else None


def score_measurements(
    measurements: dict[str, Any],
    ground_truth: dict[str, Any],
    *,
    recovered_fraction_by_lead: dict[str, float] | None = None,
) -> dict[str, Any]:
    intervals = measurements.get("intervals") or {}
    rhythm = measurements.get("rhythm") or {}
    st_by_lead = measurements.get("st_by_lead") or {}
    morphology = measurements.get("morphology") or {}

    def interval(name: str) -> Any:
        return (intervals.get(name) or {}).get("value")

    st_errors = []
    for lead in LEADS:
        measured = (st_by_lead.get(lead) or {}).get("j60_mv")
        truth = (ground_truth.get("st_j60_mv_by_lead") or {}).get(lead)
        err = _abs_error(measured, truth) if truth is not None else None
        if err is not None:
            st_errors.append(err)

    amp_errors = []
    r_map = (morphology.get("r_amplitude_mv") or {})
    for lead in LEADS:
        truth = (ground_truth.get("r_amplitude_reference_mv_by_lead") or {}).get(lead)
        err = _abs_error(r_map.get(lead), truth) if truth is not None else None
        if err is not None:
            amp_errors.append(err)

    coverage = recovered_fraction_by_lead or {}
    recovered_pct = (
        100.0 * sum(float(coverage.get(lead) or 0.0) >= 0.30 for lead in LEADS) / 12.0
        if coverage else None
    )

    return {
        "MAE_QRS_ms": _abs_error(interval("qrs_ms"), ground_truth["qrs_ms"]),
        "MAE_PR_ms": _abs_error(interval("pr_ms"), ground_truth["pr_ms"]),
        "MAE_QT_ms": _abs_error(interval("qt_ms"), ground_truth["qt_ms"]),
        "MAE_ST_mV": float(np.mean(st_errors)) if st_errors else None,
        "ERROR_RR_ms": _abs_error(rhythm.get("rr_median_ms"), ground_truth["rr_ms"]),
        "ERROR_FC_bpm": _abs_error(rhythm.get("heart_rate_bpm"), ground_truth["heart_rate_bpm"]),
        "ERROR_AMPLITUDE_R_mV": float(np.mean(amp_errors)) if amp_errors else None,
        "RECOVERED_LEADS_PERCENT": recovered_pct,
        "measured": {
            "qrs_ms": interval("qrs_ms"),
            "pr_ms": interval("pr_ms"),
            "qt_ms": interval("qt_ms"),
            "rr_median_ms": rhythm.get("rr_median_ms"),
            "heart_rate_bpm": rhythm.get("heart_rate_bpm"),
        },
    }


def score_worker_meta(meta_path: Path, truth_path: Path, output_path: Path) -> dict[str, Any]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    structured = meta.get("structured_report") or {}
    measurements = structured.get("digital_measurements") or {}
    coverage = (meta.get("signal") or {}).get("observed_fraction_by_lead") or {}
    score = score_measurements(
        measurements,
        truth,
        recovered_fraction_by_lead=coverage,
    )
    score["status"] = meta.get("status")
    score["layout"] = (meta.get("signal") or {}).get("layout_name")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(score, indent=2), encoding="utf-8")
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--out-dir", required=True)
    gen.add_argument("--layout", choices=["6x2", "3x4", "12x1"], default="6x2")
    gen.add_argument("--rhythm-strip", action="store_true")
    gen.add_argument(
        "--degradation",
        choices=["clean", "rotate", "blur", "lowres", "jpeg", "perspective", "noise", "combined"],
        default="clean",
    )
    gen.add_argument(
        "--grid",
        choices=["red", "green", "gray"],
        default="red",
    )
    gen.add_argument("--trace-width", type=int, default=2)

    sig = sub.add_parser("signal-check")
    sig.add_argument("--output", required=True)

    score = sub.add_parser("score-meta")
    score.add_argument("--meta", required=True)
    score.add_argument("--truth", required=True)
    score.add_argument("--output", required=True)

    args = ap.parse_args()

    if args.command == "generate":
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        signals, truth = generate_known_ecg()
        color = {
            "red": "#f3b6bd",
            "green": "#c8e1c5",
            "gray": "#d8d8d8",
        }[args.grid]
        paper = render_paper_ecg(
            signals,
            layout=args.layout,
            rhythm_strip=bool(args.rhythm_strip),
            grid_color=color,
            trace_width=int(args.trace_width),
        )
        paper = degrade_image(paper, args.degradation)
        image_path = out / "synthetic_ecg.png"
        truth_path = out / "ground_truth.json"
        paper.save(image_path)
        truth.update({
            "layout": args.layout,
            "rhythm_strip": bool(args.rhythm_strip),
            "degradation": args.degradation,
            "grid": args.grid,
        })
        truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
        print(json.dumps({
            "image": str(image_path),
            "ground_truth": str(truth_path),
        }))
        return 0

    if args.command == "signal-check":
        signals, truth = generate_known_ecg()
        ecg = make_digital_ecg_from_known(signals)
        measurements = measure_digital_ecg(ecg)
        result = score_measurements(measurements, truth)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0

    result = score_worker_meta(
        Path(args.meta),
        Path(args.truth),
        Path(args.output),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
