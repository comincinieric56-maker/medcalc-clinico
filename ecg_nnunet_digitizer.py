from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import signal as scipy_signal

# Pretrained digitizer provenance:
# Krones F, Walker B, Lyons T, Mahdi A. ECG-Digitiser.
# PhysioNet Challenge 2024 winning solution, BSD-2-Clause.
UPSTREAM_REPO = "felixkrones/ECG-Digitiser"
UPSTREAM_COMMIT = "e6f62aa776f105e4c7b04f21669da4d4f0df370b"
MODEL_NAME = "M3"
MODEL_DATASET = "Dataset500_Signals"
MODEL_TRAINER = "nnUNetTrainer__nnUNetPlans__2d"
MODEL_FOLD = "fold_all"
MODEL_CHECKPOINT = "checkpoint_final.pth"
MODEL_CHECKPOINT_SHA256 = "8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb"
MODEL_CHECKPOINT_SIZE = 474_901_894

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LABELS = {lead: i + 1 for i, lead in enumerate(LEADS)}

# Same vertical positioning constants used by the public winning pipeline.
Y_SHIFT_RATIO = {
    "I": 12.6 / 21.59,
    "II": 9 / 21.59,
    "III": 5.4 / 21.59,
    "aVR": 12.6 / 21.59,
    "aVL": 9 / 21.59,
    "aVF": 5.4 / 21.59,
    "V1": 12.59 / 21.59,
    "V2": 9 / 21.59,
    "V3": 5.4 / 21.59,
    "V4": 12.59 / 21.59,
    "V5": 9 / 21.59,
    "V6": 5.4 / 21.59,
    "full": 2.1 / 21.59,
}

_CACHE = Path(tempfile.gettempdir()) / "medcalc_ecg_digitizer_m3"
_MODEL_ROOT = (
    _CACHE
    / "models"
    / MODEL_NAME
    / "nnUNet_results"
    / MODEL_DATASET
    / MODEL_TRAINER
)
_MODEL_FOLD_DIR = _MODEL_ROOT / MODEL_FOLD
_LOCK = threading.Lock()


class ECGDigitizerError(RuntimeError):
    pass


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_sha256: str | None = None, expected_size: int | None = None) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()

    with requests.get(url, stream=True, timeout=(20, 600)) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    if expected_size is not None and tmp.stat().st_size != expected_size:
        tmp.unlink(missing_ok=True)
        raise ECGDigitizerError(
            f"Peso nnU-Net incompleto. Esperado={expected_size}; actual={tmp.stat().st_size if tmp.exists() else 'NA'}"
        )

    if expected_sha256 is not None:
        actual = _sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ECGDigitizerError(
                f"SHA256 del peso nnU-Net inválido. Esperado={expected_sha256}; actual={actual}"
            )
    tmp.replace(dest)


def ensure_pretrained_digitizer() -> Path:
    """Materializa únicamente los artefactos M3 necesarios para inferencia."""
    with _LOCK:
        dataset_json = _MODEL_ROOT / "dataset.json"
        plans_json = _MODEL_ROOT / "plans.json"
        checkpoint = _MODEL_FOLD_DIR / MODEL_CHECKPOINT

        base_raw = (
            "https://raw.githubusercontent.com/"
            f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/models/{MODEL_NAME}/nnUNet_results/"
            f"{MODEL_DATASET}/{MODEL_TRAINER}"
        )
        checkpoint_url = (
            "https://media.githubusercontent.com/media/"
            f"{UPSTREAM_REPO}/{UPSTREAM_COMMIT}/models/{MODEL_NAME}/nnUNet_results/"
            f"{MODEL_DATASET}/{MODEL_TRAINER}/{MODEL_FOLD}/{MODEL_CHECKPOINT}"
        )

        if not dataset_json.is_file():
            _download(base_raw + "/dataset.json", dataset_json)
        if not plans_json.is_file():
            _download(base_raw + "/plans.json", plans_json)
        if (
            not checkpoint.is_file()
            or checkpoint.stat().st_size != MODEL_CHECKPOINT_SIZE
            or _sha256(checkpoint) != MODEL_CHECKPOINT_SHA256
        ):
            _download(
                checkpoint_url,
                checkpoint,
                expected_sha256=MODEL_CHECKPOINT_SHA256,
                expected_size=MODEL_CHECKPOINT_SIZE,
            )

        # Fail closed on malformed metadata.
        dataset = json.loads(dataset_json.read_text(encoding="utf-8"))
        if dataset.get("file_ending") != ".png":
            raise ECGDigitizerError("Modelo M3 no declara entrada PNG esperada.")
        labels = dataset.get("labels") or {}
        for lead, value in LABELS.items():
            if int(labels.get(lead, -1)) != value:
                raise ECGDigitizerError(f"Mapeo de etiqueta inesperado para {lead}.")

        return _MODEL_ROOT.parent.parent.parent


def _rgb_bytes_to_png(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def _rotation_angle(rgb: np.ndarray) -> float:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Scale Hough threshold with image width instead of assuming one fixed resolution.
    threshold = max(250, int(round(rgb.shape[1] * 0.55)))
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    if lines is None:
        return 0.0

    angles: List[float] = []
    for line in lines[:100]:
        rho, theta = line[0]
        deg = theta * 180.0 / np.pi
        dev = -(90.0 - deg)
        if abs(dev) <= 15.0:
            angles.append(dev)

    if len(angles) < 3:
        return 0.0
    return float(np.median(angles))


def _rotate_keep_size(rgb: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.05:
        return rgb
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        rgb,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _predict_mask(rgb: np.ndarray, work: Path) -> np.ndarray:
    model_root = ensure_pretrained_digitizer()

    inp = work / "nn_input"
    out = work / "nn_output"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    image_path = inp / "00000_temp_0000.png"
    Image.fromarray(rgb).save(image_path, format="PNG")

    env = dict(os.environ)
    env["nnUNet_results"] = str(model_root)

    cmd = [
        "nnUNetv2_predict",
        "-d", MODEL_DATASET,
        "-i", str(inp),
        "-o", str(out),
        "-f", "all",
        "-tr", "nnUNetTrainer",
        "-c", "2d",
        "-p", "nnUNetPlans",
        "-device", "cpu",
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    if proc.returncode != 0:
        raise ECGDigitizerError(
            "nnU-Net falló.\n"
            f"STDOUT:\n{proc.stdout[-6000:]}\n"
            f"STDERR:\n{proc.stderr[-6000:]}"
        )

    mask_path = out / "00000_temp.png"
    if not mask_path.is_file():
        raise ECGDigitizerError("nnU-Net terminó sin producir máscara PNG.")

    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.int16)


def _component_bbox(binary: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(binary)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _centerline(binary_crop: np.ndarray) -> np.ndarray:
    h, w = binary_crop.shape
    y = np.full(w, np.nan, dtype=float)
    for x in range(w):
        rows = np.flatnonzero(binary_crop[:, x])
        if rows.size:
            y[x] = float(np.mean(rows))

    good = np.flatnonzero(np.isfinite(y))
    if good.size < max(20, int(0.15 * w)):
        return y

    # Interpolate only within the observed x support.
    lo, hi = int(good[0]), int(good[-1])
    xi = np.arange(lo, hi + 1)
    y[lo : hi + 1] = np.interp(xi, good, y[good])
    return y


def _resample_1d(values: np.ndarray, n: int) -> np.ndarray:
    good = np.flatnonzero(np.isfinite(values))
    if good.size < 2:
        return np.full(n, np.nan, dtype=float)
    x_old = good.astype(float) / max(1.0, len(values) - 1.0)
    x_new = np.linspace(x_old[0], x_old[-1], n)
    return np.interp(x_new, x_old, values[good])


def _vectorize_mask(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    widths = []
    bboxes: Dict[str, Tuple[int, int, int, int] | None] = {}
    for lead, label in LABELS.items():
        bbox = _component_bbox(mask == label)
        bboxes[lead] = bbox
        if bbox is not None:
            widths.append(bbox[2] - bbox[0])

    if len(widths) < 10:
        raise ECGDigitizerError(
            f"Segmentación incompleta: sólo {len(widths)}/12 derivaciones detectadas."
        )

    med = float(np.median(widths))
    short_widths = [w for w in widths if w < 2.0 * med]
    if not short_widths:
        raise ECGDigitizerError("No fue posible inferir la escala temporal desde la segmentación.")

    short_width = float(np.mean(short_widths))
    sec_per_pixel = 2.5 / short_width
    mm_per_pixel = 25.0 * sec_per_pixel
    mv_per_pixel = mm_per_pixel / 10.0

    signals: Dict[str, np.ndarray] = {}
    durations: Dict[str, float] = {}
    coverages: Dict[str, float] = {}

    image_h = float(rgb.shape[0])

    for lead in LEADS:
        bbox = bboxes[lead]
        if bbox is None:
            signals[lead] = np.full(5000, np.nan, dtype=np.float32)
            durations[lead] = 0.0
            coverages[lead] = 0.0
            continue

        x0, y0, x1, y1 = bbox
        crop = (mask[y0:y1, x0:x1] == LABELS[lead])
        center = _centerline(crop)
        width_sec = (x1 - x0) * sec_per_pixel

        if width_sec > 5.0:
            duration = 10.0
            y_shift = Y_SHIFT_RATIO["full"]
        else:
            duration = 2.5
            y_shift = Y_SHIFT_RATIO[lead]

        baseline_global_y = (1.0 - y_shift) * image_h
        center_global = y0 + center
        mv = (baseline_global_y - center_global) * mv_per_pixel

        n = int(round(duration * 500.0))
        resampled = _resample_1d(mv, n).astype(np.float32)

        full = np.full(5000, np.nan, dtype=np.float32)
        if duration >= 9.5:
            full[: min(5000, len(resampled))] = resampled[:5000]
        else:
            # Preserve exactly what the page contains; do not invent the missing 7.5 s.
            full[: min(1250, len(resampled))] = resampled[:1250]

        signals[lead] = full
        durations[lead] = duration
        coverages[lead] = float(np.mean(np.isfinite(full)))

    matrix = np.column_stack([signals[l] for l in LEADS]).astype(np.float32)

    # R27 requires a full 10 s 12-lead signal. Standard 3x4 printouts do not contain it.
    eligible = all(durations[l] >= 9.5 and coverages[l] >= 0.95 for l in LEADS)

    return {
        "signals_500": matrix,
        "lead_names": list(LEADS),
        "duration_sec_by_lead": durations,
        "finite_coverage_by_lead": coverages,
        "sec_per_pixel": float(sec_per_pixel),
        "mV_per_pixel": float(mv_per_pixel),
        "r27_temporal_coverage_eligible": bool(eligible),
    }


def _mask_overlay(rgb: np.ndarray, mask: np.ndarray) -> bytes:
    palette = np.array(
        [
            [0,0,0],
            [230,25,75],[60,180,75],[255,225,25],[0,130,200],
            [245,130,48],[145,30,180],[70,240,240],[240,50,230],
            [210,245,60],[250,190,190],[0,128,128],[230,190,255],
        ],
        dtype=np.uint8,
    )
    colors = palette[np.clip(mask, 0, 12)]
    overlay = rgb.copy().astype(float)
    fg = mask > 0
    overlay[fg] = 0.55 * overlay[fg] + 0.45 * colors[fg]
    out = io.BytesIO()
    Image.fromarray(np.clip(overlay,0,255).astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


def digitize_with_pretrained_nnunet(image_bytes: bytes) -> Dict[str, Any]:
    """Segment ECG traces with the pretrained M3 nnU-Net and vectorize measured pixels.

    The function does not synthesize unobserved time. For a standard 3x4 printout,
    each non-rhythm lead usually has only ~2.5 s of observed data; such a page is
    therefore digitizable but not eligible for the frozen R27 10 s/12-lead runtime.
    """
    rgb = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"), dtype=np.uint8)
    angle = _rotation_angle(rgb)
    rotated = _rotate_keep_size(rgb, angle)

    with tempfile.TemporaryDirectory(prefix="medcalc_nnunet_ecg_") as tmp:
        mask = _predict_mask(rotated, Path(tmp))

    vec = _vectorize_mask(rotated, mask)
    vec.update(
        {
            "rotation_deg": float(angle),
            "mask_overlay_bytes": _mask_overlay(rotated, mask),
            "model_source": UPSTREAM_REPO,
            "model_commit": UPSTREAM_COMMIT,
            "model_name": MODEL_NAME,
            "checkpoint_sha256": MODEL_CHECKPOINT_SHA256,
        }
    )
    return vec


def make_r27_wfdb_payload(signals_500: np.ndarray, lead_names: List[str]) -> Dict[str, bytes]:
    """Create 500 Hz + 100 Hz WFDB bytes only for complete 10 s 12-lead signals."""
    import wfdb

    x = np.asarray(signals_500, dtype=float)
    if x.shape != (5000, 12):
        raise ECGDigitizerError(f"Forma 500 Hz inesperada: {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ECGDigitizerError(
            "No se puede construir entrada R27: faltan muestras observadas en una o más derivaciones."
        )
    if list(lead_names) != LEADS:
        raise ECGDigitizerError("Orden de derivaciones inesperado.")

    # Deterministic anti-aliased 500→100 Hz conversion for the image-derived adapter.
    x100 = scipy_signal.resample_poly(x, up=1, down=5, axis=0)
    if x100.shape != (1000, 12):
        raise ECGDigitizerError(f"Forma 100 Hz inesperada: {x100.shape}")

    with tempfile.TemporaryDirectory(prefix="medcalc_r27_wfdb_") as tmp:
        root = Path(tmp)
        hr_name = "photo_hr"
        lr_name = "photo_lr"

        common = dict(
            units=["mV"] * 12,
            sig_name=LEADS,
            fmt=["16"] * 12,
            adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
        )
        wfdb.wrsamp(hr_name, fs=500, p_signal=x, write_dir=str(root), **common)
        wfdb.wrsamp(lr_name, fs=100, p_signal=x100, write_dir=str(root), **common)

        return {
            "hr_hea_name": hr_name + ".hea",
            "hr_hea_bytes": (root / (hr_name + ".hea")).read_bytes(),
            "hr_dat_name": hr_name + ".dat",
            "hr_dat_bytes": (root / (hr_name + ".dat")).read_bytes(),
            "lr_hea_name": lr_name + ".hea",
            "lr_hea_bytes": (root / (lr_name + ".hea")).read_bytes(),
            "lr_dat_name": lr_name + ".dat",
            "lr_dat_bytes": (root / (lr_name + ".dat")).read_bytes(),
        }
