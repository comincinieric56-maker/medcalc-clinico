from __future__ import annotations

import argparse
import io
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LAYOUTS = {
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


@dataclass(frozen=True)
class SyntheticTruth:
    fs: int = 500
    heart_rate_bpm: float = 75.0
    rr_ms: float = 800.0
    p_duration_ms: float = 80.0
    pr_ms: float = 160.0
    qrs_ms: float = 100.0
    qt_ms: float = 400.0
    st_j60_mv: float = -0.12
    r_amp_mv: float = 1.0
    s_amp_mv: float = -0.35
    q_amp_mv: float = -0.12
    t_amp_mv: float = 0.30


def _gaussian(t: np.ndarray, center: float, sigma: float, amplitude: float) -> np.ndarray:
    return float(amplitude) * np.exp(-0.5 * ((t - float(center)) / float(sigma)) ** 2)


def generate_ground_truth_ecg(
    *,
    duration_s: float = 10.0,
    fs: int = 500,
    heart_rate_bpm: float = 75.0,
    st_by_lead: Dict[str, float] | None = None,
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Create deterministic digital ECGs with known fiducial intervals.

    This is a validation signal, not a physiological simulator. Its purpose is
    to provide exact numeric ground truth for paper-rendering/recovery tests.
    """
    truth = SyntheticTruth(
        fs=int(fs),
        heart_rate_bpm=float(heart_rate_bpm),
        rr_ms=60000.0 / float(heart_rate_bpm),
    )
    rr_s = truth.rr_ms / 1000.0
    t = np.arange(int(round(duration_s * fs)), dtype=float) / float(fs)

    lead_scale = {
        "I": 0.85, "II": 1.00, "III": 0.70,
        "aVR": -0.75, "aVL": 0.55, "aVF": 0.80,
        "V1": 0.55, "V2": 0.75, "V3": 0.95,
        "V4": 1.15, "V5": 1.10, "V6": 0.90,
    }
    default_st = {lead: truth.st_j60_mv for lead in LEADS}
    default_st.update(st_by_lead or {})

    out = {lead: np.zeros_like(t) for lead in LEADS}
    first_r = 0.60
    r_times = np.arange(first_r, duration_s - 0.45, rr_s)

    for lead in LEADS:
        scale = float(lead_scale[lead])
        x = np.zeros_like(t)
        for r in r_times:
            # Ground-truth fiducials relative to R.
            p_on = r - 0.20
            p_off = r - 0.12
            qrs_on = r - 0.04
            qrs_off = r + 0.06
            t_off = r + 0.36

            x += _gaussian(t, r - 0.16, 0.022, 0.12 * scale)
            x += _gaussian(t, r - 0.030, 0.010, truth.q_amp_mv * scale)
            x += _gaussian(t, r, 0.010, truth.r_amp_mv * scale)
            x += _gaussian(t, r + 0.032, 0.012, truth.s_amp_mv * scale)

            st = float(default_st[lead])
            st_mask = (t >= qrs_off) & (t <= r + 0.18)
            x[st_mask] += st

            x += _gaussian(t, r + 0.25, 0.055, truth.t_amp_mv * scale)

        # Small deterministic baseline wander makes filtering realistic without
        # changing the known fiducial timing.
        x += 0.015 * np.sin(2 * np.pi * 0.22 * t + 0.2 * LEADS.index(lead))
        out[lead] = x

    ground_truth = {
        **asdict(truth),
        "duration_s": float(duration_s),
        "r_times_s": [round(float(v), 6) for v in r_times.tolist()],
        "st_by_lead_mv": {lead: float(default_st[lead]) for lead in LEADS},
        "lead_scale": lead_scale,
    }
    return out, ground_truth


def _grid_rgb(name: str) -> tuple[int, int, int]:
    values = {
        "red": (235, 190, 190),
        "green": (190, 225, 195),
        "gray": (210, 210, 210),
        "none": (255, 255, 255),
    }
    return values.get(str(name).lower(), values["red"])


def render_ecg_paper(
    signals: Dict[str, np.ndarray],
    *,
    fs: int,
    layout: str,
    rhythm_strip: bool = False,
    speed_mm_per_s: float = 25.0,
    gain_mm_per_mv: float = 10.0,
    px_per_mm: float = 4.0,
    grid_color: str = "red",
    trace_width: int = 2,
    add_text: bool = True,
) -> Image.Image:
    """Render a canonical digital ECG onto standard paper.

    Layout changes only placement; waveform time/amplitude remain physically
    calibrated from speed and gain.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unsupported layout: {layout}")
    matrix = LAYOUTS[layout]
    rows = len(matrix)
    cols = len(matrix[0])
    segment_s = 10.0 / cols

    margin_mm = 8.0
    row_height_mm = 26.0
    signal_width_mm = 10.0 * float(speed_mm_per_s)
    width_mm = 2 * margin_mm + signal_width_mm
    height_mm = 2 * margin_mm + rows * row_height_mm + (row_height_mm if rhythm_strip else 0.0)

    width = int(round(width_mm * px_per_mm))
    height = int(round(height_mm * px_per_mm))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    grid = _grid_rgb(grid_color)
    if grid_color.lower() != "none":
        for mm in np.arange(0, width_mm + 0.001, 1.0):
            x = int(round(mm * px_per_mm))
            major = abs(mm % 5.0) < 1e-6
            g = tuple(max(0, min(255, c - (25 if major else 0))) for c in grid)
            draw.line((x, 0, x, height), fill=g, width=1 if not major else 2)
        for mm in np.arange(0, height_mm + 0.001, 1.0):
            y = int(round(mm * px_per_mm))
            major = abs(mm % 5.0) < 1e-6
            g = tuple(max(0, min(255, c - (25 if major else 0))) for c in grid)
            draw.line((0, y, width, y), fill=g, width=1 if not major else 2)

    start_x = margin_mm * px_per_mm
    row_h = row_height_mm * px_per_mm
    segment_px = signal_width_mm * px_per_mm / cols

    for row, lead_row in enumerate(matrix):
        baseline = margin_mm * px_per_mm + (row + 0.52) * row_h
        for col, lead in enumerate(lead_row):
            sig = np.asarray(signals[lead], dtype=float)
            start = int(round(col * segment_s * fs))
            stop = int(round((col + 1) * segment_s * fs))
            segment = sig[start:stop]
            if segment.size < 2:
                continue
            xx = start_x + col * segment_px + np.arange(segment.size) * (
                speed_mm_per_s * px_per_mm / fs
            )
            yy = baseline - segment * gain_mm_per_mv * px_per_mm
            pts = [(float(x), float(y)) for x, y in zip(xx, yy)]
            draw.line(pts, fill=(20, 20, 20), width=max(1, int(trace_width)))
            if add_text:
                draw.text((start_x + col * segment_px + 4, baseline - 11 * px_per_mm), lead, fill=(25, 25, 25))

    if rhythm_strip:
        lead = "II"
        baseline = margin_mm * px_per_mm + (rows + 0.52) * row_h
        sig = np.asarray(signals[lead], dtype=float)
        xx = start_x + np.arange(sig.size) * (speed_mm_per_s * px_per_mm / fs)
        yy = baseline - sig * gain_mm_per_mv * px_per_mm
        draw.line([(float(x), float(y)) for x, y in zip(xx, yy)], fill=(20, 20, 20), width=max(1, int(trace_width)))
        if add_text:
            draw.text((start_x + 4, baseline - 11 * px_per_mm), "II", fill=(25, 25, 25))

    if add_text:
        draw.text((int(start_x), 4), f"{speed_mm_per_s:g}mm/s   {gain_mm_per_mv:g}mm/mV", fill=(20, 20, 20))
    return image


def degrade_ecg_image(
    image: Image.Image,
    *,
    rotation_deg: float = 0.0,
    perspective: float = 0.0,
    blur_sigma: float = 0.0,
    noise_sd: float = 0.0,
    jpeg_quality: int | None = None,
    scale: float = 1.0,
    text_overlay: bool = False,
    seed: int = 7,
) -> Image.Image:
    """Apply deterministic scan/photo degradations for validation."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    if abs(float(perspective)) > 1e-6:
        p = float(np.clip(perspective, 0.0, 0.20))
        dx, dy = p * w, p * h
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([
            [0.55 * dx, 0.15 * dy],
            [w - 1 - 0.20 * dx, 0.65 * dy],
            [w - 1 - 0.60 * dx, h - 1 - 0.15 * dy],
            [0.15 * dx, h - 1 - 0.55 * dy],
        ])
        H = cv2.getPerspectiveTransform(src, dst)
        rgb = cv2.warpPerspective(rgb, H, (w, h), borderValue=(255, 255, 255))

    if abs(float(rotation_deg)) > 1e-6:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0)
        rgb = cv2.warpAffine(rgb, M, (w, h), borderValue=(255, 255, 255))

    if float(blur_sigma) > 0:
        sigma = float(blur_sigma)
        k = max(3, int(round(6 * sigma + 1)) | 1)
        rgb = cv2.GaussianBlur(rgb, (k, k), sigmaX=sigma)

    if float(noise_sd) > 0:
        rng = np.random.default_rng(int(seed))
        noise = rng.normal(0.0, float(noise_sd), size=rgb.shape)
        rgb = np.clip(rgb.astype(float) + noise, 0, 255).astype(np.uint8)

    if text_overlay:
        cv2.putText(
            rgb,
            "ECG VALIDATION 12345",
            (int(0.12 * w), int(0.22 * h)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.5, w / 1800.0),
            (45, 45, 45),
            2,
            cv2.LINE_AA,
        )

    if float(scale) != 1.0:
        nw = max(64, int(round(w * float(scale))))
        nh = max(64, int(round(h * float(scale))))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    out = Image.fromarray(rgb)
    if jpeg_quality is not None:
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=int(jpeg_quality))
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
    return out


def _metric_value(recovered: Dict[str, Any], name: str) -> float | None:
    item = ((recovered.get("global") or {}).get(name) or {})
    try:
        return float(item["value"]) if item.get("value") is not None else None
    except Exception:
        return None


def score_recovered_measurements(
    ground_truth: Dict[str, Any],
    recovered: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare recovered digital measurements with known synthetic truth."""
    expected = {
        "qrs_ms": ground_truth.get("qrs_ms"),
        "pr_ms": ground_truth.get("pr_ms"),
        "qt_ms": ground_truth.get("qt_ms"),
        "heart_rate_bpm": ground_truth.get("heart_rate_bpm"),
    }
    errors: Dict[str, float | None] = {}
    for name, truth in expected.items():
        got = _metric_value(recovered, name)
        errors[f"MAE_{name.upper()}"] = (
            abs(float(got) - float(truth))
            if got is not None and truth is not None else None
        )

    rhythm = recovered.get("rhythm") or {}
    rr_got = rhythm.get("rr_median_ms")
    errors["ERROR_RR_MS"] = (
        abs(float(rr_got) - float(ground_truth["rr_ms"]))
        if rr_got is not None else None
    )

    st_errors = []
    amp_errors = []
    recovered_leads = 0
    for lead in LEADS:
        lead_result = (recovered.get("leads") or {}).get(lead) or {}
        if lead_result.get("evaluable"):
            recovered_leads += 1
        st = ((lead_result.get("metrics") or {}).get("st_j60_mv") or {}).get("value")
        target_st = (ground_truth.get("st_by_lead_mv") or {}).get(lead)
        if st is not None and target_st is not None:
            st_errors.append(abs(float(st) - float(target_st)))
        r_amp = ((lead_result.get("metrics") or {}).get("r_amp_mv") or {}).get("value")
        scale = (ground_truth.get("lead_scale") or {}).get(lead)
        if r_amp is not None and scale is not None:
            target_r = float(ground_truth["r_amp_mv"]) * float(scale)
            amp_errors.append(abs(float(r_amp) - target_r))

    errors["MAE_ST_MV"] = float(np.mean(st_errors)) if st_errors else None
    errors["MAE_AMPLITUDE_MV"] = float(np.mean(amp_errors)) if amp_errors else None
    errors["LEAD_RECOVERY_PERCENT"] = 100.0 * recovered_leads / len(LEADS)
    return errors


def build_validation_matrix(output_dir: Path) -> list[Dict[str, Any]]:
    """Generate a cross-layout/degradation paper ECG validation suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    signals, truth = generate_ground_truth_ecg()

    cases = []
    degradations = [
        {"name": "clean", "kwargs": {}},
        {"name": "rotated", "kwargs": {"rotation_deg": 4.0}},
        {"name": "oblique", "kwargs": {"perspective": 0.07}},
        {"name": "scan_blur", "kwargs": {"blur_sigma": 1.1, "noise_sd": 3.0}},
        {"name": "low_res", "kwargs": {"scale": 0.55, "jpeg_quality": 65}},
        {"name": "text_overlay", "kwargs": {"text_overlay": True}},
    ]
    grid_colors = ["red", "green", "gray", "none"]

    for layout in ("6x2", "3x4", "12x1"):
        for rhythm_strip in (False, True):
            if layout == "12x1" and rhythm_strip:
                continue
            for grid_color in grid_colors:
                base = render_ecg_paper(
                    signals,
                    fs=int(truth["fs"]),
                    layout=layout,
                    rhythm_strip=rhythm_strip,
                    grid_color=grid_color,
                )
                for degradation in degradations:
                    image = degrade_ecg_image(base, **degradation["kwargs"])
                    stem = f"{layout}_{'strip' if rhythm_strip else 'nostrip'}_{grid_color}_{degradation['name']}"
                    path = output_dir / f"{stem}.png"
                    image.save(path)
                    cases.append({
                        "case_id": stem,
                        "path": str(path),
                        "layout": layout,
                        "rhythm_strip": rhythm_strip,
                        "grid_color": grid_color,
                        "degradation": degradation,
                        "ground_truth": truth,
                    })

    (output_dir / "manifest.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="ecg_validation_cases")
    args = ap.parse_args()
    cases = build_validation_matrix(Path(args.output_dir))
    print(json.dumps({
        "case_count": len(cases),
        "layouts": sorted(set(c["layout"] for c in cases)),
        "output_dir": str(Path(args.output_dir).resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
