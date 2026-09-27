from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict

import requests

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

REMOTE_ACTION_REPO = "comincinieric56-maker/medcalc-r27-backend"
REMOTE_ACTION_WORKFLOW = "remote-ecg-analyze.yml"


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


def _annotate_r27_payload(payload: Dict[str, Any] | None, meta: Dict[str, Any]) -> Dict[str, Any] | None:
    if payload is None:
        return None

    signal_meta = meta.get("signal") or {}
    out = dict(payload)
    tiled_input = bool(signal_meta.get("r27_tiled", False))
    temporal_rhythm_modules = [
        "AF", "FLUTTER", "SVT", "SINUS", "SINUS_TACHY",
        "SINUS_ARRHYTHMIA", "PVC", "PAC", "BIGEMINY", "TRIGEMINY",
        "AVB1", "AVB2", "AVB3",
    ]
    out["input_adapter"] = {
        "mode": signal_meta.get("r27_input_mode"),
        "r27_tiled": tiled_input,
        "research_only": tiled_input,
        "validated_equivalent_to_real_10s": False if tiled_input else True,
        "provenance": signal_meta.get("r27_tiled_provenance"),
        "source_layout": signal_meta.get("layout_name"),
        "source_observed_fraction_by_lead": signal_meta.get(
            "observed_fraction_by_lead"
        ),
        "source_observed_seconds_by_lead": signal_meta.get(
            "observed_seconds_by_lead"
        ),
        "native_signal_contract": signal_meta.get("native_signal_contract"),
        "digital_signal_schema": signal_meta.get("digital_signal_schema"),
        "clinical_measurement_source": signal_meta.get(
            "clinical_measurement_source"
        ),
        "measurement_precedence": "NUMERIC_DIGITAL_SIGNAL_GT_CLASSIFIER",
        "calibration": signal_meta.get("calibration"),
        "rhythm_strip_observed": bool(
            signal_meta.get("rhythm_strip_observed", False)
        ),
        "rhythm_strip_lead": signal_meta.get("rhythm_strip_lead"),
        "rhythm_strip_center_source": signal_meta.get(
            "rhythm_strip_center_source"
        ),
        "temporal_rhythm_modules_interpretable": not tiled_input,
        "suppressed_temporal_modules": temporal_rhythm_modules if tiled_input else [],
    }

    if tiled_input:
        modules = out.get("modules") or {}
        for key in temporal_rhythm_modules:
            item = modules.get(key)
            if isinstance(item, dict):
                item["interpretability"] = "NOT_INTERPRETABLE_R27_TILED"
    return out


def remote_digitizer_status(
    api_url: str,
    api_token: str | None = None,
    *,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    base = str(api_url or "").strip().rstrip("/")
    if not base:
        raise ECGDigitiserError("Falta ECG_R27_API_URL en Streamlit Secrets.")
    headers = {}
    if api_token:
        headers["Authorization"] = "Bearer " + str(api_token)
    try:
        response = requests.get(
            base + "/health",
            headers=headers,
            timeout=int(timeout_seconds),
        )
    except requests.RequestException as exc:
        raise ECGDigitiserError(f"Backend ECG no disponible: {exc}") from exc
    if response.status_code >= 400:
        raise ECGDigitiserError(
            f"Backend ECG respondió HTTP {response.status_code}: "
            + response.text[-1200:]
        )
    try:
        return dict(response.json())
    except Exception as exc:
        raise ECGDigitiserError("Backend ECG devolvió /health inválido.") from exc


def _github_api_headers(github_token: str) -> dict[str, str]:
    token = str(github_token or "").strip()
    if not token:
        raise ECGDigitiserError(
            "Falta R27_GITHUB_TOKEN. Se necesita para iniciar el runner privado "
            "de GitHub Actions y leer su artefacto de resultado."
        )
    return {
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def digitize_photo_pdf_github_actions(
    api_url: str,
    api_token: str | None,
    github_token: str,
    *,
    source_name: str,
    source_bytes: bytes,
    age: float,
    sex: str,
    pdf_page_index: int = 0,
    timeout_seconds: int = 2400,
    poll_seconds: int = 8,
) -> Dict[str, Any]:
    """Run the heavy ECG pipeline on a private GitHub-hosted runner.

    Streamlit only stages the source, dispatches the private workflow and polls
    for a short-lived artifact. Neither the U-Net nor frozen R27 is loaded in
    the Streamlit process or the lightweight staging backend.
    """
    base = str(api_url or "").strip().rstrip("/")
    if not base:
        raise ECGDigitiserError("Falta ECG_R27_API_URL para staging del ECG.")
    if str(sex) not in {"0", "1"}:
        raise ECGDigitiserError("sex debe ser el código congelado 0 o 1.")
    if not 0.0 <= float(age) <= 120.0:
        raise ECGDigitiserError("Edad fuera del rango aceptado.")
    if not source_bytes:
        raise ECGDigitiserError("Archivo foto/PDF vacío.")

    ext = Path(source_name or "").suffix.lower()
    content_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if ext not in content_types:
        raise ECGDigitiserError("Formato no admitido. Use PDF/JPG/JPEG/PNG/WEBP.")

    stage_headers = {}
    if api_token:
        stage_headers["Authorization"] = "Bearer " + str(api_token)

    try:
        stage_response = requests.post(
            base + "/v1/ecg/jobs/source",
            headers=stage_headers,
            files={
                "source": (
                    "ecg" + ext,
                    source_bytes,
                    content_types[ext],
                )
            },
            timeout=(30, 120),
        )
    except requests.RequestException as exc:
        raise ECGDigitiserError(
            "No fue posible transferir el ECG al staging remoto. "
            f"Detalle: {exc}"
        ) from exc

    if stage_response.status_code >= 400:
        detail = stage_response.text[-4000:]
        try:
            parsed = stage_response.json()
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise ECGDigitiserError(
            f"Staging ECG respondió HTTP {stage_response.status_code}: {detail}"
        )

    try:
        stage = dict(stage_response.json())
        job_id = str(stage["job_id"])
        source_url = str(stage["source_url"])
    except Exception as exc:
        raise ECGDigitiserError("Staging ECG devolvió una respuesta inválida.") from exc

    gh_headers = _github_api_headers(github_token)
    dispatch_url = (
        f"https://api.github.com/repos/{REMOTE_ACTION_REPO}/actions/workflows/"
        f"{REMOTE_ACTION_WORKFLOW}/dispatches"
    )
    dispatch_payload = {
        "ref": "main",
        "inputs": {
            "job_id": job_id,
            "source_url": source_url,
            "source_ext": ext,
            "age": str(float(age)),
            "sex": str(sex),
            "pdf_page_index": str(int(pdf_page_index)),
        },
    }

    try:
        dispatch = requests.post(
            dispatch_url,
            headers=gh_headers,
            json=dispatch_payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise ECGDigitiserError(
            "No fue posible iniciar GitHub Actions para el ECG. "
            f"Detalle: {exc}"
        ) from exc

    if dispatch.status_code != 204:
        raise ECGDigitiserError(
            "GitHub Actions rechazó el trabajo ECG "
            f"(HTTP {dispatch.status_code}). El R27_GITHUB_TOKEN debe tener "
            "lectura del repositorio privado y permiso Actions: write. "
            + dispatch.text[-3000:]
        )

    artifact_name = "medcalc-ecg-" + job_id
    artifacts_url = (
        f"https://api.github.com/repos/{REMOTE_ACTION_REPO}/actions/artifacts"
    )
    deadline = time.monotonic() + int(timeout_seconds)
    artifact = None

    while time.monotonic() < deadline:
        try:
            listing = requests.get(
                artifacts_url,
                headers=gh_headers,
                params={"name": artifact_name, "per_page": 10},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise ECGDigitiserError(
                f"Fallo consultando resultado de GitHub Actions: {exc}"
            ) from exc

        if listing.status_code >= 400:
            raise ECGDigitiserError(
                "No fue posible consultar los artefactos del runner ECG "
                f"(HTTP {listing.status_code}): {listing.text[-3000:]}"
            )

        data = listing.json()
        candidates = [
            item
            for item in (data.get("artifacts") or [])
            if item.get("name") == artifact_name and not item.get("expired", False)
        ]
        if candidates:
            artifact = sorted(
                candidates,
                key=lambda item: str(item.get("created_at") or ""),
                reverse=True,
            )[0]
            break
        time.sleep(max(3, int(poll_seconds)))

    if artifact is None:
        raise ECGDigitiserError(
            "GitHub Actions no produjo el resultado ECG dentro del tiempo límite. "
            f"Trabajo: {job_id}"
        )

    archive_url = str(artifact.get("archive_download_url") or "")
    if not archive_url:
        raise ECGDigitiserError("El artefacto ECG no tiene URL de descarga.")

    try:
        archive = requests.get(
            archive_url,
            headers=gh_headers,
            timeout=(30, 180),
        )
    except requests.RequestException as exc:
        raise ECGDigitiserError(
            f"No fue posible descargar el resultado ECG: {exc}"
        ) from exc
    if archive.status_code >= 400:
        raise ECGDigitiserError(
            f"Descarga del resultado ECG falló HTTP {archive.status_code}: "
            + archive.text[-3000:]
        )

    try:
        with zipfile.ZipFile(io.BytesIO(archive.content), "r") as zf:
            names = zf.namelist()
            result_name = next(
                name for name in names
                if Path(name).name == "result.json"
            )
            result = json.loads(zf.read(result_name).decode("utf-8"))
    except Exception as exc:
        raise ECGDigitiserError(
            "El artefacto del runner ECG no contiene result.json válido."
        ) from exc

    if result.get("job_error"):
        raise ECGDigitiserError(
            "El runner privado de ECG falló: " + str(result["job_error"])
        )

    meta = result.get("digitizer") or {}
    meta["execution_location"] = "GITHUB_ACTIONS_PRIVATE_RUNNER"
    meta["streamlit_loaded_unet"] = False
    result["digitizer"] = meta
    result["payload"] = _annotate_r27_payload(result.get("payload"), meta)
    result["remote_backend"] = True
    result["remote_compute"] = "GITHUB_ACTIONS"
    result["remote_job_id"] = job_id
    return result


def digitize_photo_pdf_remote(
    api_url: str,
    api_token: str | None,
    *,
    source_name: str,
    source_bytes: bytes,
    age: float,
    sex: str,
    pdf_page_index: int = 0,
    timeout_seconds: int = 1800,
) -> Dict[str, Any]:
    """Run photo/PDF U-Net + measurements + optional R27 outside Streamlit."""
    base = str(api_url or "").strip().rstrip("/")
    if not base:
        raise ECGDigitiserError(
            "Falta ECG_R27_API_URL. El U-Net remoto es obligatorio para "
            "foto/PDF y no se ejecutará dentro de Streamlit."
        )
    if str(sex) not in {"0", "1"}:
        raise ECGDigitiserError("sex debe ser el código congelado 0 o 1.")
    if not 0.0 <= float(age) <= 120.0:
        raise ECGDigitiserError("Edad fuera del rango aceptado.")
    if not source_bytes:
        raise ECGDigitiserError("Archivo foto/PDF vacío.")

    ext = Path(source_name or "").suffix.lower()
    content_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    if ext not in content_types:
        raise ECGDigitiserError("Formato no admitido. Use PDF/JPG/JPEG/PNG/WEBP.")

    headers = {}
    if api_token:
        headers["Authorization"] = "Bearer " + str(api_token)

    files = {
        "source": (
            Path(source_name or ("ecg" + ext)).name,
            source_bytes,
            content_types[ext],
        )
    }
    data = {
        "age": str(float(age)),
        "sex": str(sex),
        "release": "RESEARCH_PROBABILITY_ONLY_RELEASE",
        "pdf_page_index": str(int(pdf_page_index)),
        "run_r27": "1",
    }

    try:
        response = requests.post(
            base + "/v1/ecg/analyze",
            headers=headers,
            files=files,
            data=data,
            timeout=(30, int(timeout_seconds)),
        )
    except requests.RequestException as exc:
        raise ECGDigitiserError(
            "El backend remoto de ECG no respondió. "
            "Streamlit no ejecutó el U-Net localmente. "
            f"Detalle: {exc}"
        ) from exc

    if response.status_code >= 400:
        detail = response.text[-8000:]
        try:
            parsed = response.json()
            if isinstance(parsed, dict) and parsed.get("detail"):
                detail = str(parsed["detail"])
        except Exception:
            pass
        raise ECGDigitiserError(
            f"Backend ECG respondió HTTP {response.status_code}: {detail}"
        )

    try:
        result = dict(response.json())
    except Exception as exc:
        raise ECGDigitiserError("Backend ECG devolvió JSON inválido.") from exc

    meta = result.get("digitizer") or {}
    meta["execution_location"] = "REMOTE_BACKEND"
    meta["streamlit_loaded_unet"] = False
    result["digitizer"] = meta
    result["payload"] = _annotate_r27_payload(result.get("payload"), meta)
    result["remote_backend"] = True
    return result


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
                    # Limit glibc arena proliferation in the child process.
                    # This reduces CPU-side fragmentation without constraining
                    # the Streamlit parent process.
                    "MALLOC_ARENA_MAX": "2",
                }
            )

            worker_cmd = [
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
            ]

            def _run_digitizer_worker(extra_args: list[str] | None = None):
                return subprocess.run(
                    worker_cmd + list(extra_args or []),
                    cwd=str(ROOT),
                    env=env,
                    # The neural stack lives only in this child. If it exits,
                    # PyTorch memory is returned to the OS before R27 starts.
                    text=True,
                    timeout=int(timeout_seconds),
                )

            def _worker_reason() -> str:
                if not meta_path.is_file():
                    return ""
                try:
                    return str(
                        json.loads(meta_path.read_text(encoding="utf-8")).get("reason")
                        or ""
                    )
                except Exception:
                    return ""

            proc = _run_digitizer_worker()
            reason = _worker_reason()
            low_memory_retry = False

            if proc.returncode != 0:
                reason_l = reason.casefold()
                resource_failure = bool(
                    proc.returncode in {-9, 137}
                    or "out of memory" in reason_l
                    or "cannot allocate memory" in reason_l
                    or "defaultcpuallocator" in reason_l
                    or "memoryerror" in reason_l
                )
                if resource_failure:
                    # A high-resolution child may be killed without taking down
                    # Streamlit itself. Retry once with the historical compact
                    # path, after deleting partial worker products.
                    low_memory_retry = True
                    if out_root.exists():
                        shutil.rmtree(out_root, ignore_errors=True)
                    out_root.mkdir(parents=True, exist_ok=True)
                    try:
                        meta_path.unlink()
                    except FileNotFoundError:
                        pass
                    proc = _run_digitizer_worker(["--force-low-memory"])
                    reason = _worker_reason()

            if proc.returncode != 0:
                raise ECGDigitiserError(
                    "El digitalizador U-Net falló."
                    + (f"\nDetalle: {reason}" if reason else "")
                    + f"\nCódigo de salida: {proc.returncode}"
                )

            if not meta_path.is_file():
                raise ECGDigitiserError("El digitalizador terminó sin metadata.")

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["assets"] = asset_status
            meta["worker_low_memory_retry"] = bool(low_memory_retry)
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

            payload = _annotate_r27_payload(payload, meta)

            return {
                "payload": payload,
                "digitizer": meta,
                "digitizer_stdout_tail": "",
            }


def digitiser_status() -> Dict[str, Any]:
    return verify_digitiser_assets()
