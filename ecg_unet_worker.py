from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _prepare_source_image(source: Path, page_index: int, destination: Path) -> dict:
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
            pix = page.get_pixmap(
                matrix=fitz.Matrix(300.0 / 72.0, 300.0 / 72.0),
                alpha=False,
            )
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            page_count = int(doc.page_count)
        finally:
            doc.close()
        source_type = "pdf"
    else:
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        source_type = "image"
        page_count = 1

    original_size = image.size

    # Keep enough resolution for the grid while bounding worst-case RAM before
    # the U-Net performs its own resampling.
    max_dimension = 4200
    scale = min(1.0, float(max_dimension) / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)

    return {
        "source_type": source_type,
        "pdf_page_index": int(page_index) if source_type == "pdf" else None,
        "pdf_page_count": page_count,
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "processed_width": int(image.width),
        "processed_height": int(image.height),
    }


def _load_digitizer(
    vendor_root: Path,
    segmentation_model: Path,
    lead_model: Path,
):
    import torch

    sys.path.insert(0, str(vendor_root))

    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper

    config_path = vendor_root / "src" / "config" / "inference_wrapper_george-moody-2024.yml"
    layout_path = vendor_root / "src" / "config" / "lead_layouts_all.yml"
    lead_unet_config = vendor_root / "src" / "config" / "lead_name_unet.yml"

    cfg = get_cfg(str(config_path))

    # CPU-only inference for Streamlit Community Cloud.
    cfg.MODEL.KWARGS.device = "cpu"
    cfg.MODEL.KWARGS.resample_size = 2000
    cfg.MODEL.KWARGS.apply_dewarping = False
    cfg.MODEL.KWARGS.enable_timing = False

    inner = cfg.MODEL.KWARGS.config
    inner.SEGMENTATION_MODEL.weight_path = str(segmentation_model)
    inner.LAYOUT_IDENTIFIER.config_path = str(layout_path)
    inner.LAYOUT_IDENTIFIER.unet_config_path = str(lead_unet_config)
    inner.LAYOUT_IDENTIFIER.unet_weight_path = str(lead_model)
    inner.LAYOUT_IDENTIFIER.KWARGS.device = "cpu"
    inner.LAYOUT_IDENTIFIER.KWARGS.possibly_flipped = True

    # R27's frozen signal contract is 10 s at 500 Hz.
    # This sets the output grid length only; unobserved printed portions remain
    # NaN in the canonical lead tensor and are checked before R27 can run.
    inner.LAYOUT_IDENTIFIER.KWARGS.target_num_samples = 5000
    inner.LAYOUT_IDENTIFIER.KWARGS.required_valid_samples = 2

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = InferenceWrapper(**cfg.MODEL.KWARGS)
    model.eval()
    return model


def _digitize_image(image_path: Path, model) -> tuple[np.ndarray, dict]:
    import torch
    from torchvision.io import decode_image

    image = decode_image(str(image_path), mode="RGB")[:3].unsqueeze(0)

    with torch.inference_mode():
        result = model(
            image,
            layout_should_include_substring=None,
        )

    canonical = result.get("signal", {}).get("canonical_lines")
    if canonical is None:
        raise RuntimeError("El U-Net no produjo canonical_lines.")

    signal_uv = canonical.detach().cpu().numpy().astype(np.float64)
    if signal_uv.shape != (12, 5000):
        raise RuntimeError(
            f"Forma canónica inesperada: {signal_uv.shape}; se esperaba (12, 5000)."
        )

    signal_uv = signal_uv.T  # samples x leads

    finite = np.isfinite(signal_uv)
    coverage = finite.mean(axis=0)

    layout_name = str(result.get("layout_name") or "")
    layout_cost = result.get("signal", {}).get("layout_matching_cost")
    try:
        layout_cost = float(layout_cost)
    except Exception:
        layout_cost = None

    pixel = result.get("pixel_spacing_mm") or {}

    meta = {
        "shape_500_candidate": [int(v) for v in signal_uv.shape],
        "sig_names": LEADS,
        "observed_fraction_by_lead": {
            lead: round(float(coverage[i]), 6)
            for i, lead in enumerate(LEADS)
        },
        "min_observed_fraction": round(float(np.min(coverage)), 6),
        "all_samples_observed": bool(np.all(finite)),
        "layout_name": layout_name,
        "layout_matching_cost": layout_cost,
        "pixel_spacing_mm": {
            "x": float(pixel["x"]) if pixel.get("x") is not None else None,
            "y": float(pixel["y"]) if pixel.get("y") is not None else None,
        },
        "units_from_digitizer": "uV",
        "target_samples": 5000,
    }
    return signal_uv, meta


def _write_wfdb_pair(
    signal_uv: np.ndarray,
    output500: Path,
    output100: Path,
    record_name: str,
) -> dict:
    import wfdb
    from scipy.signal import resample_poly

    if signal_uv.shape != (5000, 12):
        raise RuntimeError(f"Forma inesperada antes de WFDB: {signal_uv.shape}")
    if not np.isfinite(signal_uv).all():
        raise RuntimeError(
            "La señal contiene muestras no observadas. No se permite imputarlas antes de R27."
        )

    # Open ECG Digitizer reports microvolts. PTB-XL / R27 reads physical ECG
    # amplitudes in millivolts through WFDB.
    signal500_mv = signal_uv / 1000.0

    if not np.isfinite(signal500_mv).all():
        raise RuntimeError("Conversión uV→mV produjo valores no finitos.")

    signal100_mv = resample_poly(signal500_mv, up=1, down=5, axis=0)
    if signal100_mv.shape != (1000, 12):
        raise RuntimeError(f"Resample 100 Hz inesperado: {signal100_mv.shape}")
    if not np.isfinite(signal100_mv).all():
        raise RuntimeError("Resample 100 Hz produjo valores no finitos.")

    output500.mkdir(parents=True, exist_ok=True)
    output100.mkdir(parents=True, exist_ok=True)

    common = dict(
        units=["mV"] * 12,
        sig_name=LEADS,
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
    )

    wfdb.wrsamp(
        record_name,
        fs=500,
        p_signal=signal500_mv,
        write_dir=str(output500),
        **common,
    )
    wfdb.wrsamp(
        record_name,
        fs=100,
        p_signal=signal100_mv,
        write_dir=str(output100),
        **common,
    )

    return {
        "shape_500": [5000, 12],
        "shape_100": [1000, 12],
        "fs_500": 500,
        "fs_100": 100,
        "units_for_r27": "mV",
        "adapter_100hz": "scipy.signal.resample_poly(up=1, down=5)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-root", required=True)
    ap.add_argument("--segmentation-model", required=True)
    ap.add_argument("--lead-model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pdf-page-index", type=int, default=0)
    args = ap.parse_args()

    vendor_root = Path(args.vendor_root).resolve()
    segmentation_model = Path(args.segmentation_model).resolve()
    lead_model = Path(args.lead_model).resolve()
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
        "digitizer": "Ahus-AIM/Open-ECG-Digitizer",
        "digitizer_commit": "97a15087d4abcda843da8c58ee74b1d8f47e6f9a",
        "segmentation_model_sha256": "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b",
        "lead_model_sha256": "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2",
        "license": "CC BY-SA 4.0",
    }

    try:
        meta["image"] = _prepare_source_image(
            source,
            int(args.pdf_page_index),
            image_path,
        )

        model = _load_digitizer(
            vendor_root,
            segmentation_model,
            lead_model,
        )
        signal_uv, signal_meta = _digitize_image(image_path, model)

        # Release neural model memory before any later R27 process exists.
        del model

        meta["signal"] = signal_meta

        # Fail closed: a conventional printed 3x4 ECG normally contains only
        # 2.5 s of most leads. The U-Net is allowed to digitize that visible
        # information, but R27 may not receive fabricated missing samples.
        if not signal_meta["all_samples_observed"]:
            meta["status"] = "DIGITIZED_ONLY"
            meta["reason"] = (
                "El trazado fue digitalizado, pero no existen 10 s observados para "
                "las 12 derivaciones. R27 no se ejecuta porque MEDCALC no repite, "
                "interpola ni inventa segmentos no impresos."
            )
        else:
            wfdb_meta = _write_wfdb_pair(
                signal_uv,
                output500,
                output100,
                record_name,
            )
            meta["signal"].update(wfdb_meta)
            meta["signal"]["r27_input_compatible"] = True
            meta["signal"]["r27_compatibility_rule"] = (
                "12 standard leads; exactly 5000 observed finite samples/lead at 500 Hz"
            )
            meta["signal"]["photo_domain_warning"] = (
                "The 100 Hz representation is derived from the reconstructed 500 Hz "
                "signal. This photo-domain adapter has not established external "
                "validation for the frozen R27 models."
            )
            meta["wfdb_500_base"] = str(output500 / record_name)
            meta["wfdb_100_base"] = str(output100 / record_name)
            meta["status"] = "PASS"
            meta["reason"] = (
                "Digitalización completa: 10 s observados y finitos en las 12 derivaciones."
            )

    except Exception as exc:
        meta["status"] = "FAIL"
        meta["reason"] = str(exc)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
