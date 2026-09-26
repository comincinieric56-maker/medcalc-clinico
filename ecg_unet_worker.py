from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _render_source_to_png(source: Path, page_index: int, destination: Path) -> dict:
    ext = source.suffix.lower()

    if ext == ".pdf":
        import fitz

        doc = fitz.open(stream=source.read_bytes(), filetype="pdf")
        try:
            if getattr(doc, "needs_pass", False):
                raise RuntimeError("PDF protegido con contraseña.")
            if doc.page_count < 1:
                raise RuntimeError("PDF sin páginas.")
            if not 0 <= page_index < doc.page_count:
                raise RuntimeError(
                    f"Página PDF fuera de rango: {page_index + 1}/{doc.page_count}."
                )
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72.0, 300 / 72.0), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            page_count = int(doc.page_count)
        finally:
            doc.close()
        source_type = "pdf"
    else:
        image = Image.open(source).convert("RGB")
        page_count = 1
        source_type = "image"

    # Bound the raster before classical rectification. The neural model can
    # resample internally; keeping the source finite prevents pathological RAM use.
    max_dim = 4200
    scale = min(1.0, max_dim / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=96)

    # Perspective correction runs in this child process so OpenCV/Numpy memory
    # disappears before R27 starts.
    from ecg_photo_engine import prepare_ecg_image, rectify_ecg_photo

    prepared, _, _ = prepare_ecg_image(buf.getvalue(), max_dimension=4200)
    rectified, rect_meta = rectify_ecg_photo(prepared)

    rect = Image.open(io.BytesIO(rectified)).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rect.save(destination, format="PNG", optimize=True)

    return {
        "source_type": source_type,
        "pdf_page_index": int(page_index) if source_type == "pdf" else None,
        "pdf_page_count": page_count,
        "input_width": int(image.width),
        "input_height": int(image.height),
        "rectified_width": int(rect.width),
        "rectified_height": int(rect.height),
        "rectification": rect_meta,
    }


def _run_digitiser(digitiser_root: Path, image_dir: Path, output500: Path) -> None:
    output500.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONPATH": str(digitiser_root)
            + os.pathsep
            + str(Path(__file__).resolve().parent)
            + os.pathsep
            + env.get("PYTHONPATH", ""),
            # nnU-Net compilation can consume substantial RAM on CPU.
            "nnUNet_compile": "false",
        }
    )

    cmd = [
        sys.executable,
        "-m",
        "src.run.digitize",
        "-d",
        str(image_dir),
        "-m",
        str(digitiser_root / "models" / "M3"),
        "-o",
        str(output500),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(digitiser_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=1500,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "PhysioNet ECG-Digitiser falló.\n"
            f"STDOUT:\n{proc.stdout[-12000:]}\n"
            f"STDERR:\n{proc.stderr[-12000:]}"
        )


def _lead_coverage(signal: np.ndarray) -> float:
    if signal.size == 0:
        return 0.0
    finite = np.isfinite(signal)
    if not finite.any():
        return 0.0
    x = np.nan_to_num(signal.astype(float), nan=0.0)
    # Upstream writes unavailable portions as exact zero after nan_to_num.
    # Real ECG baselines may cross zero, but long exact-zero sections are the
    # key signal that a printed 3x4 layout did not contain that time interval.
    observed = np.abs(x) > 1e-8
    return float(np.mean(observed))


def _validate_and_make_100hz(output500: Path, output100: Path, record_name: str) -> dict:
    import wfdb
    from scipy.signal import resample_poly

    base500 = output500 / record_name
    rec = wfdb.rdrecord(str(base500))
    sig = np.asarray(rec.p_signal, dtype=np.float64)
    sig_names = list(rec.sig_name or [])

    if int(round(float(rec.fs))) != 500:
        raise RuntimeError(f"Frecuencia U-Net inesperada: {rec.fs} Hz; se esperaban 500 Hz.")
    if sig.ndim != 2 or sig.shape[0] != 5000:
        raise RuntimeError(
            f"Longitud U-Net inesperada: {sig.shape}; se esperaban 5000 muestras."
        )
    if sig_names != LEADS:
        raise RuntimeError(
            "El U-Net no recuperó exactamente las 12 derivaciones estándar en orden. "
            f"Recuperadas={sig_names}"
        )
    if sig.shape[1] != 12:
        raise RuntimeError(f"Número de derivaciones inesperado: {sig.shape[1]}")

    finite_fraction = float(np.mean(np.isfinite(sig)))
    if finite_fraction < 0.999:
        raise RuntimeError(
            f"Señal no finita tras digitalización: fracción finita={finite_fraction:.5f}."
        )

    coverage = {lead: _lead_coverage(sig[:, i]) for i, lead in enumerate(LEADS)}
    min_coverage = min(coverage.values())

    # R27 was frozen on 10 s x 12 leads. A conventional 3x4 printout contains
    # only 2.5 s for most leads. Never tile, extrapolate or invent the missing
    # 7.5 s merely to satisfy the R27 tensor shape.
    r27_compatible = bool(min_coverage >= 0.90)

    sig100 = resample_poly(sig, up=1, down=5, axis=0)
    if sig100.shape != (1000, 12):
        raise RuntimeError(f"Resample 100 Hz inesperado: {sig100.shape}")

    output100.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name,
        fs=100,
        units=["mV"] * 12,
        sig_name=LEADS,
        p_signal=np.asarray(sig100, dtype=np.float64),
        write_dir=str(output100),
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
    )

    return {
        "shape_500": [int(v) for v in sig.shape],
        "shape_100": [int(v) for v in sig100.shape],
        "fs_500": 500,
        "fs_100": 100,
        "sig_names": sig_names,
        "finite_fraction": finite_fraction,
        "observed_fraction_by_lead": {k: round(v, 4) for k, v in coverage.items()},
        "min_observed_fraction": round(min_coverage, 4),
        "r27_input_compatible": r27_compatible,
        "r27_compatibility_rule": "12 leads x 10 s; >=90% non-zero observed coverage per lead",
        "resampling_note": (
            "100 Hz is derived from the reconstructed 500 Hz signal with scipy.signal.resample_poly(1,5). "
            "This photo-domain adapter is not equivalent to established external validation of R27."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--digitiser-root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pdf-page-index", type=int, default=0)
    args = ap.parse_args()

    digitiser_root = Path(args.digitiser_root).resolve()
    source = Path(args.source).resolve()
    output_root = Path(args.output_root).resolve()
    meta_path = Path(args.meta).resolve()

    image_dir = output_root / "input"
    output500 = output_root / "wfdb500"
    output100 = output_root / "wfdb100"
    image_dir.mkdir(parents=True, exist_ok=True)

    record_name = "medcalc_photo_ecg"
    image_path = image_dir / f"{record_name}.png"

    meta: dict = {
        "status": "STARTED",
        "digitiser": "felixkrones/ECG-Digitiser M3",
        "digitiser_commit": "e6f62aa776f105e4c7b04f21669da4d4f0df370b",
        "model_sha256": "8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb",
    }

    try:
        meta["image"] = _render_source_to_png(
            source,
            int(args.pdf_page_index),
            image_path,
        )
        _run_digitiser(digitiser_root, image_dir, output500)
        signal_meta = _validate_and_make_100hz(output500, output100, record_name)
        meta["signal"] = signal_meta
        meta["wfdb_500_base"] = str(output500 / record_name)
        meta["wfdb_100_base"] = str(output100 / record_name)

        if signal_meta["r27_input_compatible"]:
            meta["status"] = "PASS"
            meta["reason"] = "Digitalización completa compatible con el contrato temporal de entrada R27."
        else:
            meta["status"] = "DIGITIZED_ONLY"
            meta["reason"] = (
                "La imagen fue digitalizada, pero no contiene 10 s observados para las 12 derivaciones. "
                "R27 no se ejecuta porque completar segmentos ausentes sería fabricar señal."
            )
    except Exception as exc:
        meta["status"] = "FAIL"
        meta["reason"] = str(exc)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        raise

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
