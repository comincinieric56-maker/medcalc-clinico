from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

from r27_local_runtime import R27LocalError, run_r27_local

# Licensed PhysioNet Challenge 2024 winner.
DIGITISER_REPO = "https://github.com/felixkrones/ECG-Digitiser.git"
DIGITISER_COMMIT = "e6f62aa776f105e4c7b04f21669da4d4f0df370b"
DIGITISER_CACHE = Path(tempfile.gettempdir()) / f"medcalc_ecg_digitiser_{DIGITISER_COMMIT[:12]}"

MODEL_REL = Path(
    "models/M3/nnUNet_results/Dataset500_Signals/"
    "nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth"
)
MODEL_SHA256 = "8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb"
MODEL_SIZE = 474_901_894
MODEL_URL = (
    "https://media.githubusercontent.com/media/felixkrones/ECG-Digitiser/"
    f"{DIGITISER_COMMIT}/{MODEL_REL.as_posix()}"
)

_SOURCE_LOCK = threading.Lock()
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


def _run_checked(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise ECGDigitiserError(
            f"Comando de digitalización falló ({proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout[-8000:]}\n"
            f"STDERR:\n{proc.stderr[-8000:]}"
        )
    return proc


def _download_model(destination: Path) -> None:
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)

    with requests.get(MODEL_URL, stream=True, timeout=(30, 900)) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    size = partial.stat().st_size
    if size != MODEL_SIZE:
        partial.unlink(missing_ok=True)
        raise ECGDigitiserError(
            f"Checkpoint U-Net incompleto: {size} bytes; esperado {MODEL_SIZE}."
        )

    actual = _sha256_file(partial)
    if actual != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise ECGDigitiserError(
            f"SHA del checkpoint U-Net no coincide. Esperado={MODEL_SHA256}; actual={actual}"
        )

    partial.replace(destination)


def ensure_digitiser_assets() -> Path:
    """Materializa código + checkpoint exactos del digitalizador ganador 2024."""
    with _SOURCE_LOCK:
        marker = DIGITISER_CACHE / ".medcalc_digitiser_ready.json"

        if marker.is_file():
            try:
                meta = json.loads(marker.read_text(encoding="utf-8"))
                model = DIGITISER_CACHE / MODEL_REL
                if (
                    meta.get("commit") == DIGITISER_COMMIT
                    and model.is_file()
                    and model.stat().st_size == MODEL_SIZE
                    and _sha256_file(model) == MODEL_SHA256
                ):
                    return DIGITISER_CACHE
            except Exception:
                pass

        if DIGITISER_CACHE.exists():
            shutil.rmtree(DIGITISER_CACHE, ignore_errors=True)
        DIGITISER_CACHE.mkdir(parents=True, exist_ok=False)

        # Fetch only runtime-relevant paths. Git LFS pointers are intentionally
        # left as pointers and the one checkpoint we need is fetched separately
        # from GitHub's media endpoint and verified by SHA-256.
        _run_checked(["git", "init"], cwd=DIGITISER_CACHE, timeout=60)
        _run_checked(
            ["git", "remote", "add", "origin", DIGITISER_REPO],
            cwd=DIGITISER_CACHE,
            timeout=60,
        )
        _run_checked(
            [
                "git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
                "fetch", "--depth", "1", "--filter=blob:none", "origin", DIGITISER_COMMIT,
            ],
            cwd=DIGITISER_CACHE,
            timeout=600,
        )
        _run_checked(
            [
                "git", "-c", "filter.lfs.smudge=", "-c", "filter.lfs.required=false",
                "checkout", "FETCH_HEAD", "--", "src", "config.py", "models/M3", "LICENSE",
            ],
            cwd=DIGITISER_CACHE,
            timeout=300,
        )

        model = DIGITISER_CACHE / MODEL_REL
        _download_model(model)

        # Verify the source commit fetched before removing git metadata.
        head = _run_checked(
            ["git", "rev-parse", "FETCH_HEAD"],
            cwd=DIGITISER_CACHE,
            timeout=30,
        ).stdout.strip()
        if head != DIGITISER_COMMIT:
            raise ECGDigitiserError(
                f"Commit del digitalizador inesperado: {head} != {DIGITISER_COMMIT}"
            )

        shutil.rmtree(DIGITISER_CACHE / ".git", ignore_errors=True)

        marker.write_text(
            json.dumps(
                {
                    "repo": DIGITISER_REPO,
                    "commit": DIGITISER_COMMIT,
                    "model_sha256": MODEL_SHA256,
                    "model_size": MODEL_SIZE,
                    "license": "BSD-2-Clause",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return DIGITISER_CACHE


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
    """FOTO/PDF -> U-Net/nnU-Net -> WFDB 500/100 Hz -> R27.

    The U-Net worker is a separate process and exits before R27 starts. This
    prevents the segmentation model and R27 model stack from occupying memory
    at the same time in the Streamlit process.
    """
    if not github_token:
        raise ECGDigitiserError("Falta R27_GITHUB_TOKEN en Streamlit Secrets.")
    if str(sex) not in {"0", "1"}:
        raise ECGDigitiserError("sex debe ser el código congelado 0 o 1.")
    if not 0.0 <= float(age) <= 120.0:
        raise ECGDigitiserError("Edad fuera del rango aceptado.")
    if not source_bytes:
        raise ECGDigitiserError("Archivo foto/PDF vacío.")

    digitiser_root = ensure_digitiser_assets()

    with _RUN_LOCK:
        with tempfile.TemporaryDirectory(prefix="medcalc_photo_r27_") as tmp:
            root = Path(tmp)
            ext = Path(source_name or "").suffix.lower()
            if ext not in {".pdf", ".jpg", ".jpeg", ".png", ".webp"}:
                raise ECGDigitiserError("Formato no admitido. Use PDF/JPG/JPEG/PNG/WEBP.")

            source_path = root / ("source" + ext)
            source_path.write_bytes(source_bytes)

            out_root = root / "digitized"
            out_root.mkdir(parents=True, exist_ok=True)
            meta_path = root / "digitizer_meta.json"

            worker = Path(__file__).resolve().parent / "ecg_unet_worker.py"
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
                }
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(worker),
                    "--digitiser-root", str(digitiser_root),
                    "--source", str(source_path),
                    "--output-root", str(out_root),
                    "--meta", str(meta_path),
                    "--pdf-page-index", str(int(pdf_page_index)),
                ],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=int(timeout_seconds),
            )

            if proc.returncode != 0:
                raise ECGDigitiserError(
                    "El digitalizador U-Net falló.\n"
                    f"STDOUT:\n{proc.stdout[-10000:]}\n"
                    f"STDERR:\n{proc.stderr[-10000:]}"
                )

            if not meta_path.is_file():
                raise ECGDigitiserError("El digitalizador terminó sin metadata.")

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status = str(meta.get("status") or "")

            if status == "FAIL":
                raise ECGDigitiserError(
                    "Digitalización rechazada por control de integridad: "
                    + str(meta.get("reason") or "sin detalle")
                )

            # A conventional 3x4 printout is often a valid ECG image but does
            # not contain a full 10 s trace for all 12 leads. Return the
            # digitization metadata without calling R27 rather than fabricating
            # the absent samples.
            if status == "DIGITIZED_ONLY":
                return {
                    "payload": None,
                    "digitizer": meta,
                    "digitizer_stdout_tail": proc.stdout[-3000:],
                }

            if status != "PASS":
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
            except R27LocalError:
                raise
            except Exception as exc:
                raise ECGDigitiserError(f"R27 falló tras la digitalización: {exc}") from exc

            return {
                "payload": payload,
                "digitizer": meta,
                "digitizer_stdout_tail": proc.stdout[-3000:],
            }


def digitiser_status() -> dict[str, Any]:
    root = ensure_digitiser_assets()
    return {
        "ready": True,
        "source_commit": DIGITISER_COMMIT,
        "model_sha256": MODEL_SHA256,
        "model_size": MODEL_SIZE,
        "cache_root": str(root),
        "pipeline": "PhysioNet Challenge 2024 winner nnU-Net M3 -> 500 Hz -> resample_poly 100 Hz -> R27",
    }
