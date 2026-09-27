from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ecg_digital_measurements import build_digital_measurements
from ecg_signal_reconstruction import LEADS, LAYOUTS, digital_ecg_to_jsonable


def _gaussian(t: np.ndarray, center: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - center) / max(width, 1e-6)) ** 2)


def generate_synthetic_digital_ecg(
    *,
    fs: int = 500,
    duration_s: float = 10.0,
    heart_rate_bpm: float = 75.0,
    pr_ms: float = 160.0,
    qrs_ms: float = 92.0,
    qt_ms: float = 390.0,
    st_by_lead_mv: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Create a deterministic digital ECG with known timing/amplitude truth.

    The waveform is synthetic and intended for engineering validation, not
    physiological simulation or clinical training.
    """
    st_by_lead_mv = dict(st_by_lead_mv or {})
    n = int(round(duration_s * fs))
    t = np.arange(n, dtype=float) / fs
    rr = 60.0 / float(heart_rate_bpm)
    qrs_s = qrs_ms / 1000.0
    pr_s = pr_ms / 1000.0
    qt_s = qt_ms / 1000.0

    lead_scale = {
        "I": 0.85, "II": 1.00, "III": 0.70,
        "aVR": -0.65, "aVL": 0.55, "aVF": 0.80,
        "V1": 0.45, "V2": 0.75, "V3": 1.00,
        "V4": 1.15, "V5": 1.05, "V6": 0.90,
    }

    leads: Dict[str, Any] = {}
    beat_centers = np.arange(0.7, duration_s - 0.4, rr)
    for lead in LEADS:
        scale = lead_scale[lead]
        sig = np.zeros(n, dtype=float)
        for r0 in beat_centers:
            p_center = r0 - pr_s + 0.055
            sig += _gaussian(t, p_center, 0.020, 0.12 * np.sign(scale or 1.0))

            # QRS width is controlled by Q/S centers around the R peak.
            q_center = r0 - qrs_s * 0.30
            s_center = r0 + qrs_s * 0.32
            sig += _gaussian(t, q_center, max(0.007, qrs_s * 0.11), -0.18 * abs(scale))
            sig += _gaussian(t, r0, max(0.008, qrs_s * 0.10), 1.15 * scale)
            sig += _gaussian(t, s_center, max(0.009, qrs_s * 0.12), -0.30 * abs(scale))

            j_time = r0 + qrs_s * 0.50
            t_center = r0 + min(0.30, max(0.20, qt_s * 0.62))
            st = float(st_by_lead_mv.get(lead, 0.0))
            st_start = int(max(0, round(j_time * fs)))
            st_end = int(min(n, round((t_center - 0.06) * fs)))
            if st_end > st_start:
                sig[st_start:st_end] += st
            sig += _gaussian(t, t_center, 0.055, 0.32 * np.sign(scale or 1.0))

        leads[lead] = {
            "signal_mv": sig,
            "time_ms": np.arange(n, dtype=float) * 1000.0 / fs,
            "fs": fs,
            "duration_s": duration_s,
            "source": "SYNTHETIC_GROUND_TRUTH",
            "confidence": 1.0,
            "observed_mask": np.ones(n, dtype=bool),
            "confidence_mask": np.ones(n, dtype=float),
            "coverage": 1.0,
            "longest_contiguous_fraction": 1.0,
        }

    return {
        "schema": "MEDCALC_DIGITAL_ECG_V2",
        "source": "SYNTHETIC_GROUND_TRUTH",
        "source_layout": "DIGITAL_NATIVE",
        "layout_used_only_for_reconstruction": False,
        "fs": fs,
        "units": {"time": "ms", "amplitude": "mV"},
        "calibration": {
            "schema": "MEDCALC_ECG_CALIBRATION_V2",
            "mm_per_pixel_x": 0.2,
            "mm_per_pixel_y": 0.2,
            "speed_mm_s": 25.0,
            "gain_mm_mV": 10.0,
            "confidence": 1.0,
            "quantitative_scale_verified": True,
            "speed_source": "SYNTHETIC_GROUND_TRUTH",
            "gain_source": "SYNTHETIC_GROUND_TRUTH",
        },
        "leads": leads,
        "recovered_leads": list(LEADS),
        "recovered_lead_count": 12,
        "global_confidence": 1.0,
        "ground_truth": {
            "heart_rate_bpm": heart_rate_bpm,
            "rr_ms": 60000.0 / heart_rate_bpm,
            "pr_ms": pr_ms,
            "qrs_ms": qrs_ms,
            "qt_ms": qt_ms,
            "st_by_lead_mv": st_by_lead_mv,
        },
    }


def render_paper_ecg(
    digital_ecg: Mapping[str, Any],
    *,
    layout: str = "6x2",
    rhythm_strip: bool = True,
    speed_mm_s: float = 25.0,
    gain_mm_mV: float = 10.0,
    px_per_mm: float = 5.0,
    grid_color: str = "red",
    background: str = "white",
) -> Image.Image:
    if layout not in {"6x2", "3x4", "12x1"}:
        raise ValueError(layout)

    small = max(1, int(round(px_per_mm)))
    width_mm = 270.0
    row_mm = 28.0
    if layout == "6x2":
        rows = 7 if rhythm_strip else 6
        matrix = LAYOUTS["6x2"]
    elif layout == "3x4":
        rows = 4 if rhythm_strip else 3
        matrix = LAYOUTS["3x4"]
    else:
        rows = 12
        matrix = [[lead] for lead in LEADS]
        rhythm_strip = False

    width = int(round(width_mm * px_per_mm))
    height = int(round((rows * row_mm + 12.0) * px_per_mm))
    img = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(img)

    palette = {
        "red": ((255, 235, 235), (240, 190, 190)),
        "green": ((235, 248, 235), (185, 220, 185)),
        "gray": ((242, 242, 242), (205, 205, 205)),
    }
    minor, major = palette.get(grid_color, palette["red"])
    for x in range(0, width, small):
        draw.line((x, 0, x, height), fill=major if (x // small) % 5 == 0 else minor, width=1)
    for y in range(0, height, small):
        draw.line((0, y, width, y), fill=major if (y // small) % 5 == 0 else minor, width=1)

    margin_px = int(round(8.0 * px_per_mm))
    usable_w = width - 2 * margin_px
    ncols = len(matrix[0])

    def draw_lead(lead: str, r: int, c: int, ncols_local: int, duration_limit: float | None = None):
        item = (digital_ecg.get("leads") or {}).get(lead) or {}
        raw = item.get("signal_mv")
        if raw is None:
            return
        sig = np.asarray(raw, dtype=float)
        fs = int(item.get("fs") or digital_ecg.get("fs") or 500)
        if duration_limit is not None:
            sig = sig[: int(round(duration_limit * fs))]
        panel_w = usable_w / ncols_local
        x0 = margin_px + c * panel_w
        baseline = int(round((6.0 + (r + 0.5) * row_mm) * px_per_mm))
        t = np.arange(sig.size, dtype=float) / fs
        x = x0 + t * speed_mm_s * px_per_mm
        x_max = margin_px + (c + 1) * panel_w
        y = baseline - sig * gain_mm_mV * px_per_mm
        finite = np.isfinite(sig) & (x < x_max)
        idx = np.flatnonzero(finite)
        if idx.size >= 2:
            pts = [(int(round(x[i])), int(round(y[i]))) for i in idx]
            draw.line(pts, fill=(20, 20, 20), width=max(1, int(round(px_per_mm * 0.35))))
        draw.text((int(x0 + 4), baseline - int(11 * px_per_mm)), lead, fill=(20, 20, 20))

    if layout == "12x1":
        for r, lead in enumerate(LEADS):
            draw_lead(lead, r, 0, 1, duration_limit=10.0)
    else:
        total_time = usable_w / px_per_mm / speed_mm_s
        per_panel = total_time / ncols
        for r, lead_row in enumerate(matrix):
            for c, lead in enumerate(lead_row):
                draw_lead(lead, r, c, ncols, duration_limit=per_panel)
        if rhythm_strip:
            draw_lead("II", len(matrix), 0, 1, duration_limit=total_time)

    return img


def augment_paper_image(
    image: Image.Image,
    *,
    rotation_deg: float = 0.0,
    perspective: float = 0.0,
    blur_sigma: float = 0.0,
    noise_sd: float = 0.0,
    jpeg_quality: int | None = None,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]

    if abs(rotation_deg) > 1e-6:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotation_deg, 1.0)
        arr = cv2.warpAffine(arr, m, (w, h), borderValue=(255, 255, 255))

    if perspective > 0:
        d = float(perspective) * min(w, h)
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = np.float32([[d, d * 0.4], [w - 1 - d * 0.3, 0], [w - 1, h - 1 - d], [0, h - 1]])
        m = cv2.getPerspectiveTransform(src, dst)
        arr = cv2.warpPerspective(arr, m, (w, h), borderValue=(255, 255, 255))

    if blur_sigma > 0:
        k = max(3, int(round(blur_sigma * 6)) | 1)
        arr = cv2.GaussianBlur(arr, (k, k), blur_sigma)

    if noise_sd > 0:
        rng = np.random.default_rng(20260927)
        noise = rng.normal(0.0, noise_sd, arr.shape)
        arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)

    out = Image.fromarray(arr)
    if jpeg_quality is not None:
        import io
        buf = io.BytesIO()
        out.save(buf, format="JPEG", quality=int(jpeg_quality))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return out


def validation_metrics(
    recovered: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
) -> Dict[str, Any]:
    dm = recovered.get("digital_measurements") if "digital_measurements" in recovered else recovered
    glob = dm.get("global") or {}
    truth = ground_truth.get("ground_truth") or ground_truth

    def err(key: str, truth_key: str | None = None):
        truth_value = truth.get(truth_key or key)
        metric = glob.get(key) or {}
        recovered_value = metric.get("value") if isinstance(metric, Mapping) else metric
        if truth_value is None or recovered_value is None:
            return None
        return abs(float(recovered_value) - float(truth_value))

    st_errors = []
    recovered_st = dm.get("st_by_lead") or {}
    for lead, true_value in (truth.get("st_by_lead_mv") or {}).items():
        item = recovered_st.get(lead) or {}
        got = item.get("j60_mv")
        if got is not None:
            st_errors.append(abs(float(got) - float(true_value)))

    return {
        "MAE_QRS_MS": err("qrs_ms"),
        "MAE_PR_MS": err("pr_ms"),
        "MAE_QT_MS": err("qt_ms"),
        "MAE_ST_MV": float(np.mean(st_errors)) if st_errors else None,
        "ERROR_FC_BPM": err("heart_rate_bpm"),
        "ERROR_RR_MS": (
            abs(float((dm.get("rhythm") or {}).get("rr_mean_ms")) - float(truth["rr_ms"]))
            if (dm.get("rhythm") or {}).get("rr_mean_ms") is not None and truth.get("rr_ms") is not None
            else None
        ),
        "RECOVERED_LEAD_PERCENT": (
            100.0 * float(ground_truth.get("recovered_lead_count") or 0) / 12.0
            if ground_truth.get("recovered_lead_count") is not None
            else None
        ),
    }


def build_validation_case_matrix() -> Sequence[Dict[str, Any]]:
    cases = []
    for layout in ["6x2", "3x4", "12x1"]:
        for grid in ["red", "green", "gray"]:
            cases.append({
                "layout": layout,
                "grid": grid,
                "rotation_deg": 0.0,
                "perspective": 0.0,
                "blur_sigma": 0.0,
                "noise_sd": 0.0,
            })
    cases.extend([
        {"layout": "6x2", "grid": "red", "rotation_deg": 4.0, "perspective": 0.0, "blur_sigma": 0.0, "noise_sd": 0.0},
        {"layout": "6x2", "grid": "red", "rotation_deg": 0.0, "perspective": 0.035, "blur_sigma": 0.0, "noise_sd": 0.0},
        {"layout": "3x4", "grid": "gray", "rotation_deg": -3.0, "perspective": 0.025, "blur_sigma": 0.8, "noise_sd": 3.0},
        {"layout": "12x1", "grid": "green", "rotation_deg": 2.0, "perspective": 0.02, "blur_sigma": 1.2, "noise_sd": 5.0},
    ])
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="validation_ecg")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    truth = generate_synthetic_digital_ecg(
        st_by_lead_mv={"V2": -0.14, "V3": -0.12, "aVR": 0.11}
    )
    (out / "ground_truth.json").write_text(
        json.dumps(digital_ecg_to_jsonable(truth), indent=2),
        encoding="utf-8",
    )
    dm = build_digital_measurements(truth)
    (out / "direct_measurements.json").write_text(
        json.dumps(dm, indent=2),
        encoding="utf-8",
    )

    manifest = []
    for i, case in enumerate(build_validation_case_matrix(), start=1):
        img = render_paper_ecg(
            truth,
            layout=case["layout"],
            rhythm_strip=case["layout"] in {"6x2", "3x4"},
            grid_color=case["grid"],
        )
        img = augment_paper_image(
            img,
            rotation_deg=case["rotation_deg"],
            perspective=case["perspective"],
            blur_sigma=case["blur_sigma"],
            noise_sd=case["noise_sd"],
            jpeg_quality=70 if case["noise_sd"] else None,
        )
        name = f"case_{i:03d}_{case['layout']}_{case['grid']}.png"
        img.save(out / name)
        manifest.append({"file": name, **case})

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out), "cases": len(manifest)}, indent=2))


if __name__ == "__main__":
    main()
