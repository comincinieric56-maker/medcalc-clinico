from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from r27_local_runtime import R27LocalError, run_r27_local


OPEN_ECG_REPO = "https://github.com/Ahus-AIM/Open-ECG-Digitizer"
OPEN_ECG_COMMIT = "97a15087d4abcda843da8c58ee74b1d8f47e6f9a"
OPEN_ECG_LICENSE = "CC BY-SA 4.0"

SEGMENTATION_MODEL_SHA256 = "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b"
SEGMENTATION_MODEL_SIZE = 90_464_067
LEAD_MODEL_SHA256 = "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2"
LEAD_MODEL_SIZE = 23_296_757

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "ecg_digitizer_assets"
VENDOR_ROOT = ROOT / "ecg_digitizer_vendor"

SEGMENTATION_MODEL = ASSET_ROOT / "unet_weights_07072025.pt"
LEAD_MODEL = ASSET_ROOT / "lead_name_unet_weights_07072025.pt"
PROVENANCE_PATH = ASSET_ROOT / "PROVENANCE.json"
LICENSE_PATH = ASSET_ROOT / "LICENSE_OPEN_ECG_DIGITIZER.txt"

_RUN_LOCK = threading.Lock()


class ECGDigitiserError(RuntimeError):
    pass


def _sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _verify_file(path: Path, expected_size: int, expected_sha: str) -> None:
    if not path.is_file():
        raise ECGDigitiserError(f"Artefacto del digitalizador faltante: {path.name}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ECGDigitiserError(
            f"Tamaño inválido para {path.name}: {actual_size}; esperado {expected_size}."
        )
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ECGDigitiserError(
            f"SHA-256 inválido para {path.name}. Esperado={expected_sha}; actual={actual_sha}"
        )


def verify_digitiser_assets() -> Dict[str, Any]:
    required_source = [
        VENDOR_ROOT / "src" / "model" / "inference_wrapper.py",
        VENDOR_ROOT / "src" / "model" / "unet.py",
        VENDOR_ROOT / "src" / "model" / "signal_extractor.py",
        VENDOR_ROOT / "src" / "model" / "lead_identifier.py",
        VENDOR_ROOT / "src" / "config" / "inference_wrapper_george-moody-2024.yml",
        VENDOR_ROOT / "src" / "config" / "lead_layouts_all.yml",
        VENDOR_ROOT / "src" / "config" / "lead_name_unet.yml",
    ]
    for path in required_source:
        if not path.is_file():
            raise ECGDigitiserError(f"Fuente vendorizada faltante: {path.relative_to(ROOT)}")

    _verify_file(
        SEGMENTATION_MODEL,
        SEGMENTATION_MODEL_SIZE,
        SEGMENTATION_MODEL_SHA256,
    )
    _verify_file(
        LEAD_MODEL,
        LEAD_MODEL_SIZE,
        LEAD_MODEL_SHA256,
    )

    if not LICENSE_PATH.is_file() or not PROVENANCE_PATH.is_file():
        raise ECGDigitiserError("Falta licencia/provenance del digitalizador.")

    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    if provenance.get("source_commit") != OPEN_ECG_COMMIT:
        raise ECGDigitiserError("Commit de procedencia del digitalizador no coincide.")

    return {
        "ready": True,
        "source_repository": OPEN_ECG_REPO,
        "source_commit": OPEN_ECG_COMMIT,
        "license": OPEN_ECG_LICENSE,
        "segmentation_model_sha256": SEGMENTATION_MODEL_SHA256,
        "lead_model_sha256": LEAD_MODEL_SHA256,
        "segmentation_model_size": SEGMENTATION_MODEL_SIZE,
        "lead_model_size": LEAD_MODEL_SIZE,
    }


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise ECGDigitiserError(f"Archivo esperado no generado: {path}")
    return path.read_bytes()


def digitize_photo_pdf_and_run_r27(
    github_token: str,
    *,
    source_name: str,
    source_bytes: bytes,
    age: float,
    sex: str,
    pdf_page_index: int = 0,
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """Photo/PDF -> Open ECG Digitizer U-Net -> 500/100 Hz -> frozen R27.

    The neural digitizer executes in a separate subprocess. That child exits
    before R27 is started so both model stacks are not resident concurrently.

    R27 exact models remain frozen. For photo/PDF records without true 10 s on
    all 12 leads, MEDCALC may use the explicitly labeled research-only
    R27-TILED adapter: each incomplete lead's longest observed contiguous
    segment is repeated exactly to 10 s. This transformation is not validated
    as equivalent to real 10 s input and never authorizes binary diagnosis.
    """
    if not github_token:
        raise ECGDigitiserError("Falta R27_GITHUB_TOKEN en Streamlit Secrets.")
    if str(sex) not in {"0", "1"}:
        raise ECGDigitiserError("sex debe ser el código congelado 0 o 1.")
    if not 0.0 <= float(age) <= 120.0:
        raise ECGDigitiserError("Edad fuera del rango aceptado.")
    if not source_bytes:
        raise ECGDigitiserError("Archivo foto/PDF vacío.")

    asset_status = verify_digitiser_assets()

    with _RUN_LOCK:
        with tempfile.TemporaryDirectory(prefix="medcalc_photo_r27_") as tmp:
            request_root = Path(tmp)
            ext = Path(source_name or "").suffix.lower()
            if ext not in {".pdf", ".jpg", ".jpeg", ".png", ".webp"}:
                raise ECGDigitiserError("Formato no admitido. Use PDF/JPG/JPEG/PNG/WEBP.")

            source_path = request_root / ("source" + ext)
            source_path.write_bytes(source_bytes)

            out_root = request_root / "digitized"
            out_root.mkdir(parents=True, exist_ok=True)
            meta_path = request_root / "digitizer_meta.json"

            worker = ROOT / "ecg_unet_worker.py"
            if not worker.is_file():
                raise ECGDigitiserError("Falta ecg_unet_worker.py.")

            env = dict(os.environ)
            env.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--vendor-root",
                    str(VENDOR_ROOT),
                    "--segmentation-model",
                    str(SEGMENTATION_MODEL),
                    "--lead-model",
                    str(LEAD_MODEL),
                    "--source",
                    str(source_path),
                    "--output-root",
                    str(out_root),
                    "--meta",
                    str(meta_path),
                    "--pdf-page-index",
                    str(int(pdf_page_index)),
                    "--allow-r27-tiled",
                ],
                cwd=str(ROOT),
                env=env,
                # Inherit Streamlit stdout/stderr so the Cloud log shows the
                # exact stage reached if the container is killed by a resource
                # limit. The worker writes structured failure metadata as well.
                text=True,
                timeout=int(timeout_seconds),
            )

            if proc.returncode != 0:
                reason = ""
                if meta_path.is_file():
                    try:
                        reason = str(
                            json.loads(meta_path.read_text(encoding="utf-8")).get("reason")
                            or ""
                        )
                    except Exception:
                        reason = ""
                raise ECGDigitiserError(
                    "El digitalizador U-Net falló."
                    + (f"\nDetalle: {reason}" if reason else "")
                    + f"\nCódigo de salida: {proc.returncode}"
                )

            if not meta_path.is_file():
                raise ECGDigitiserError("El digitalizador terminó sin metadata.")

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["assets"] = asset_status
            status = str(meta.get("status") or "")

            if status == "FAIL":
                raise ECGDigitiserError(
                    "Digitalización rechazada por control de integridad: "
                    + str(meta.get("reason") or "sin detalle")
                )

            if status == "DIGITIZED_ONLY":
                return {
                    "payload": None,
                    "digitizer": meta,
                    "digitizer_stdout_tail": "",
                }

            if status not in {"PASS", "PASS_TILED"}:
                raise ECGDigitiserError(f"Estado inesperado del digitalizador: {status}")

            hr_base = Path(meta["wfdb_500_base"])
            lr_base = Path(meta["wfdb_100_base"])
            hr_hea = Path(str(hr_base) + ".hea")
            hr_dat = Path(str(hr_base) + ".dat")
            lr_hea = Path(str(lr_base) + ".hea")
            lr_dat = Path(str(lr_base) + ".dat")

            try:
                payload = run_r27_local(
                    github_token,
                    age=float(age),
                    sex=str(sex),
                    hr_hea_name=hr_hea.name,
                    hr_hea_bytes=_read_bytes(hr_hea),
                    hr_dat_name=hr_dat.name,
                    hr_dat_bytes=_read_bytes(hr_dat),
                    lr_hea_name=lr_hea.name,
                    lr_hea_bytes=_read_bytes(lr_hea),
                    lr_dat_name=lr_dat.name,
                    lr_dat_bytes=_read_bytes(lr_dat),
                    timeout_seconds=900,
                )
            except R27LocalError as exc:
                # The U-Net/digitizer has already completed successfully. Preserve
                # its structured report instead of discarding it just because the
                # separate frozen R27 runtime failed to materialize/execute.
                return {
                    "payload": None,
                    "digitizer": meta,
                    "digitizer_stdout_tail": "",
                    "r27_error": str(exc),
                    "r27_status": "RUNTIME_FAILED_AFTER_DIGITIZATION",
                }
            except Exception as exc:
                return {
                    "payload": None,
                    "digitizer": meta,
                    "digitizer_stdout_tail": "",
                    "r27_error": f"R27 falló tras la digitalización: {exc}",
                    "r27_status": "RUNTIME_FAILED_AFTER_DIGITIZATION",
                }

            signal_meta = meta.get("signal") or {}
            payload = dict(payload)
            payload["input_adapter"] = {
                "mode": signal_meta.get("r27_input_mode"),
                "r27_tiled": bool(signal_meta.get("r27_tiled", False)),
                "research_only": bool(signal_meta.get("r27_tiled", False)),
                "validated_equivalent_to_real_10s": (
                    False if signal_meta.get("r27_tiled") else True
                ),
                "provenance": signal_meta.get("r27_tiled_provenance"),
                "source_layout": signal_meta.get("layout_name"),
                "source_observed_fraction_by_lead": signal_meta.get(
                    "observed_fraction_by_lead"
                ),
            }

            return {
                "payload": payload,
                "digitizer": meta,
                "digitizer_stdout_tail": "",
            }


def digitiser_status() -> Dict[str, Any]:
    return verify_digitiser_assets()
