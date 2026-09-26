from __future__ import annotations

import csv
import hashlib
import hmac
import importlib.util
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict

R27_SOURCE_REPO = "comincinieric56-maker/medcalc-r27-backend"
R27_SOURCE_COMMIT = "ffb4980570a4efd4c54cb0d326c94858f3905711"
R27_SOURCE_URL = f"https://github.com/{R27_SOURCE_REPO}.git"

EXPECTED_RELEASE_TYPE = "RESEARCH_PROBABILITY_ONLY_RELEASE"
EXPECTED_CLINICAL_STATUS = "CLINICAL_DEPLOYMENT_BLOCKED"

ALL35 = [
    "SINUS","AF","FLUTTER","SINUS_BRADY","SINUS_TACHY","SINUS_ARRHYTHMIA",
    "PVC","PAC","BIGEMINY","TRIGEMINY","AVB1","AVB2","AVB3",
    "RBBB_COMPLETE","RBBB_INCOMPLETE","LBBB","LBBB_INCOMPLETE","IVCD",
    "LAFB","LPFB","WPW","LVH","RVH","LAE","RAE","LOW_VOLTAGE","Q_WAVE",
    "LONG_QT","ST_DEPRESSION","ST_ELEVATION","ISCHEMIA_GENERIC",
    "MI_HISTORY_Q_SCREEN","PACEMAKER","SVT","NORMAL_ECG",
]

# Immutable identities already closed in R24-R27.
CRITICAL_SHA256 = {
    "compat/MEDCALC/V32_13P6E_FINAL_E2E_DEV_FREEZE_16/FREEZE/sources/P6A_SOURCE.py":
        "f7e9db45a1b64403546f68dabcfb036875f8c2099ea3d3178e9790148f483113",
    "compat/MEDCALC/V32_13P6E_FINAL_E2E_DEV_FREEZE_16/FREEZE/sources/P6B_SOURCE.py":
        "495cd71314354190a81536b7702fae3aa6cba93aa56072e36a1695fd18670ce9",
    "compat/MEDCALC/V32_13P6E_FINAL_E2E_DEV_FREEZE_16/FREEZE/models/V31_CPU_100HZ_MULTI_SEED31092026_RECOVERY.pt":
        "df2e957922fcefe9adfba4e15acfe6e3e955cd6cd5f850bd86db0f32eea98325",
    "compat/MEDCALC/runtime/V27/MEDCALC_ECG_V27_LEAD_TERRITORY_ENGINE.py":
        "b6a8f0f56cfbd293bc8e69aa5e8f483b42b42c8c1f9c36533fd31225ab123364",
    "compat/MEDCALC/runtime/V27/ecg_v24_criteria_engine.py":
        "35b706456ed706060d64dc3fe01bf4c49431afb1301de4a9c6544f4fda20ea29",
    "compat/MEDCALC/V38_EXACT_V37_RUNTIME_RECOVERY_PARITY_1/runtime/MEDCALC_V37_EXACT_RECOVERED_RUNTIME.py":
        "4fcf1e0dd79c103216588179fc800eeffa39ea8942d50d07c51d434c61c79cce",
    "compat/MEDCALC/V32_13P4_EXACT_COMPOSITE_ADAPTER_FREEZE/EXACT_ADAPTER_BUNDLE/EXACT_COMPOSITE_ADAPTER_POLICY.json":
        "d9aa0081c60ca8666159ffef37e7021c273a01647f694c9ddebbd6ed201f903f",
}

_CACHE_ROOT = Path(tempfile.gettempdir()) / f"medcalc_r27_{R27_SOURCE_COMMIT[:12]}"
_READY_MARKER = _CACHE_ROOT / ".medcalc_r27_ready.json"
_INSTALL_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()


class R27LocalError(RuntimeError):
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


def _run_checked(cmd, *, cwd: Path | None = None, env: dict | None = None, timeout: int = 900):
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise R27LocalError(
            f"Comando R27 falló ({proc.returncode}).\n"
            f"STDOUT:\n{proc.stdout[-6000:]}\n"
            f"STDERR:\n{proc.stderr[-6000:]}"
        )
    return proc


def _verify_runtime_root(root: Path) -> None:
    required = [
        root / "r27_single_request_runner.py",
        root / "MEDCALC_R27_PROBABILITY_ONLY_GATE.py",
        root / "reference" / "ptb_id9_row.json",
        root / "compat" / "MEDCALC",
    ]
    for path in required:
        if not path.exists():
            raise R27LocalError(f"Artefacto R27 faltante: {path}")

    for rel, expected in CRITICAL_SHA256.items():
        path = root / rel
        if not path.is_file():
            raise R27LocalError(f"Artefacto congelado faltante: {rel}")
        actual = _sha256_file(path)
        if not hmac.compare_digest(actual, expected):
            raise R27LocalError(
                f"SHA R27 inválido para {rel}. Esperado={expected}; actual={actual}"
            )


def ensure_r27_runtime(github_token: str) -> Path:
    """Materializa una copia local exacta del backend R27 congelado.

    El repositorio fuente permanece privado. El token sólo se pasa a git mediante
    GIT_ASKPASS y nunca se escribe dentro del repositorio clonado.
    """
    if not github_token:
        raise R27LocalError("Falta R27_GITHUB_TOKEN en Streamlit Secrets.")

    with _INSTALL_LOCK:
        if _READY_MARKER.is_file():
            try:
                meta = json.loads(_READY_MARKER.read_text(encoding="utf-8"))
                if meta.get("source_commit") == R27_SOURCE_COMMIT:
                    _verify_runtime_root(_CACHE_ROOT)
                    return _CACHE_ROOT
            except Exception:
                pass

        if _CACHE_ROOT.exists():
            shutil.rmtree(_CACHE_ROOT, ignore_errors=True)
        _CACHE_ROOT.mkdir(parents=True, exist_ok=False)

        askpass = Path(tempfile.gettempdir()) / f"medcalc_r27_askpass_{os.getpid()}.sh"
        askpass.write_text(
            '#!/bin/sh\n'
            'case "$1" in\n'
            '  *Username*) echo "x-access-token" ;;\n'
            '  *Password*) printf "%s\\n" "$R27_GITHUB_TOKEN" ;;\n'
            '  *) echo "" ;;\n'
            'esac\n',
            encoding="utf-8",
        )
        askpass.chmod(0o700)

        env = dict(os.environ)
        env["GIT_ASKPASS"] = str(askpass)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["R27_GITHUB_TOKEN"] = github_token

        try:
            _run_checked(["git", "init"], cwd=_CACHE_ROOT, env=env, timeout=60)
            _run_checked(
                ["git", "remote", "add", "origin", R27_SOURCE_URL],
                cwd=_CACHE_ROOT,
                env=env,
                timeout=60,
            )
            _run_checked(
                ["git", "fetch", "--depth", "1", "origin", R27_SOURCE_COMMIT],
                cwd=_CACHE_ROOT,
                env=env,
                timeout=600,
            )
            _run_checked(
                ["git", "checkout", "--detach", "FETCH_HEAD"],
                cwd=_CACHE_ROOT,
                env=env,
                timeout=120,
            )
            head = _run_checked(
                ["git", "rev-parse", "HEAD"],
                cwd=_CACHE_ROOT,
                env=env,
                timeout=30,
            ).stdout.strip()
        finally:
            askpass.unlink(missing_ok=True)
            env["R27_GITHUB_TOKEN"] = ""

        if head != R27_SOURCE_COMMIT:
            raise R27LocalError(
                f"Commit R27 inesperado. Esperado={R27_SOURCE_COMMIT}; actual={head}"
            )

        # El checkout ya fue verificado; remover metadatos git ahorra disco y evita
        # que el token/remote queden asociados a la copia de trabajo.
        shutil.rmtree(_CACHE_ROOT / ".git", ignore_errors=True)

        _verify_runtime_root(_CACHE_ROOT)
        _READY_MARKER.write_text(
            json.dumps(
                {
                    "source_repo": R27_SOURCE_REPO,
                    "source_commit": R27_SOURCE_COMMIT,
                    "release_type": EXPECTED_RELEASE_TYPE,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return _CACHE_ROOT


def _safe_stem(filename: str, expected_ext: str) -> str:
    name = Path(filename or "").name
    if not name.lower().endswith(expected_ext):
        raise R27LocalError(f"Se esperaba un archivo {expected_ext}.")
    stem = name[:-len(expected_ext)]
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    if not stem or any(ch not in allowed for ch in stem):
        raise R27LocalError(f"Nombre WFDB no seguro: {name}")
    return stem


def _copy_reference_ptb(runtime_root: Path, destination_root: Path) -> dict:
    ref = runtime_root / "reference"
    row = json.loads((ref / "ptb_id9_row.json").read_text(encoding="utf-8"))
    ref_ptb = ref / "ptb-xl-1.0.3"
    for field in ("filename_hr", "filename_lr"):
        rel_base = str(row[field])
        for ext in (".hea", ".dat"):
            source = ref_ptb / (rel_base + ext)
            destination = destination_root / (rel_base + ext)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return row


def _write_request_files(
    ptb_root: Path,
    *,
    hr_stem: str,
    hr_hea: bytes,
    hr_dat: bytes,
    lr_stem: str,
    lr_hea: bytes,
    lr_dat: bytes,
) -> tuple[str, str]:
    hr_rel = Path("request") / "hr" / hr_stem
    lr_rel = Path("request") / "lr" / lr_stem

    for rel, hea, dat in (
        (hr_rel, hr_hea, hr_dat),
        (lr_rel, lr_hea, lr_dat),
    ):
        base = ptb_root / rel
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(str(base) + ".hea").write_bytes(hea)
        Path(str(base) + ".dat").write_bytes(dat)

    return str(hr_rel), str(lr_rel)


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def _prepare_database_and_meta(
    runtime_root: Path,
    ptb_root: Path,
    *,
    age: float,
    sex: int,
    filename_hr: str,
    filename_lr: str,
) -> Path:
    id9 = _copy_reference_ptb(runtime_root, ptb_root)
    request = {key: None for key in id9.keys()}
    request.update(
        {
            "ecg_id": 990000001,
            "patient_id": 990000001,
            "age": float(age),
            "sex": int(sex),
            "filename_hr": filename_hr,
            "filename_lr": filename_lr,
        }
    )

    fieldnames = list(id9.keys())
    with (ptb_root / "ptbxl_database.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({k: _csv_cell(id9.get(k)) for k in fieldnames})
        writer.writerow({k: _csv_cell(request.get(k)) for k in fieldnames})

    meta_path = ptb_root.parent / "request_meta.json"
    meta_path.write_text(
        json.dumps(request, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return meta_path


def _load_gate(runtime_root: Path):
    gate_path = runtime_root / "MEDCALC_R27_PROBABILITY_ONLY_GATE.py"
    spec = importlib.util.spec_from_file_location("medcalc_r27_gate_local", gate_path)
    if spec is None or spec.loader is None:
        raise R27LocalError("No fue posible cargar el gate R27.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_probability_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("release_type") != EXPECTED_RELEASE_TYPE:
        raise R27LocalError("release_type R27 inesperado.")
    if payload.get("clinical_deployment_status") != EXPECTED_CLINICAL_STATUS:
        raise R27LocalError("clinical_deployment_status R27 inesperado.")

    modules = payload.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(ALL35):
        raise R27LocalError("El resultado R27 no contiene exactamente los 35 módulos.")

    for module in ALL35:
        item = modules[module]
        p = float(item["probability"])
        if not 0.0 <= p <= 1.0:
            raise R27LocalError(f"{module}: probability fuera de [0,1].")
        if item.get("threshold") is not None:
            raise R27LocalError(f"{module}: threshold no autorizado.")
        if item.get("binary_classification") is not None:
            raise R27LocalError(f"{module}: clasificación binaria no autorizada.")
        if item.get("diagnostic_label") is not None:
            raise R27LocalError(f"{module}: etiqueta diagnóstica no autorizada.")
        if item.get("clinical_diagnostic_claim_allowed") is not False:
            raise R27LocalError(f"{module}: claim clínico no autorizado.")

    return payload


def run_r27_local(
    github_token: str,
    *,
    age: float,
    sex: str,
    hr_hea_name: str,
    hr_hea_bytes: bytes,
    hr_dat_name: str,
    hr_dat_bytes: bytes,
    lr_hea_name: str,
    lr_hea_bytes: bytes,
    lr_dat_name: str,
    lr_dat_bytes: bytes,
    timeout_seconds: int = 900,
) -> Dict[str, Any]:
    if not 0.0 <= float(age) <= 120.0:
        raise R27LocalError("Edad fuera del rango aceptado.")
    if str(sex) not in {"0", "1"}:
        raise R27LocalError("sex debe ser el código congelado 0 o 1.")

    hr_hea_stem = _safe_stem(hr_hea_name, ".hea")
    hr_dat_stem = _safe_stem(hr_dat_name, ".dat")
    lr_hea_stem = _safe_stem(lr_hea_name, ".hea")
    lr_dat_stem = _safe_stem(lr_dat_name, ".dat")

    if hr_hea_stem != hr_dat_stem:
        raise R27LocalError("Los archivos 500 Hz .hea/.dat deben compartir stem.")
    if lr_hea_stem != lr_dat_stem:
        raise R27LocalError("Los archivos 100 Hz .hea/.dat deben compartir stem.")

    runtime_root = ensure_r27_runtime(github_token)

    with _RUN_LOCK:
        with tempfile.TemporaryDirectory(prefix="medcalc_r27_local_") as tmp:
            request_root = Path(tmp)
            ptb_root = request_root / "ptb-xl-1.0.3"
            ptb_root.mkdir(parents=True, exist_ok=True)

            filename_hr, filename_lr = _write_request_files(
                ptb_root,
                hr_stem=hr_hea_stem,
                hr_hea=hr_hea_bytes,
                hr_dat=hr_dat_bytes,
                lr_stem=lr_hea_stem,
                lr_hea=lr_hea_bytes,
                lr_dat=lr_dat_bytes,
            )

            meta_path = _prepare_database_and_meta(
                runtime_root,
                ptb_root,
                age=float(age),
                sex=int(sex),
                filename_hr=filename_hr,
                filename_lr=filename_lr,
            )

            result_path = request_root / "bridge_result.json"
            env = dict(os.environ)
            env.update(
                {
                    "MEDCALC_COMPAT_ROOT": str(runtime_root / "compat" / "MEDCALC"),
                    "R27_PTB_ROOT": str(ptb_root),
                    "R27_RUN_OUT": str(request_root / "upstream_work"),
                    "R27_R23_OUT": str(request_root / "r23_work"),
                    "R27_REQUEST_META_JSON": str(meta_path),
                    "R27_RESULT_JSON": str(result_path),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )

            proc = subprocess.run(
                [os.sys.executable, str(runtime_root / "r27_single_request_runner.py")],
                cwd=str(runtime_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=int(timeout_seconds),
            )

            if proc.returncode != 0:
                raise R27LocalError(
                    "R27 exact runtime falló.\n"
                    f"STDOUT:\n{proc.stdout[-8000:]}\n"
                    f"STDERR:\n{proc.stderr[-8000:]}"
                )
            if not result_path.is_file():
                raise R27LocalError("R27 terminó sin producir bridge_result.json.")

            bridge_result = json.loads(result_path.read_text(encoding="utf-8"))
            gate = _load_gate(runtime_root)
            payload = gate.probability_only_view(bridge_result)
            gate.assert_probability_only_payload(payload)
            return validate_probability_payload(payload)


def runtime_status(github_token: str) -> dict:
    root = ensure_r27_runtime(github_token)
    return {
        "ready": True,
        "source_repo": R27_SOURCE_REPO,
        "source_commit": R27_SOURCE_COMMIT,
        "cache_root": str(root),
        "critical_sha_n": len(CRITICAL_SHA256),
        "modules": len(ALL35),
        "release_type": EXPECTED_RELEASE_TYPE,
    }
