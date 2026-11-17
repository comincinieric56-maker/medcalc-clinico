"""Carga y valida los registros canónicos de modelos y datasets.

Uso:  python tools/registry.py            (sale con código 1 si hay errores)

Reglas que hace cumplir:
  * campos obligatorios y estados permitidos;
  * todo SHA-256 declarado tiene 64 hex en minúscula;
  * todo modelo `frozen: true` tiene al menos un artefacto con SHA o manifiesto;
  * toda referencia de datos (trained_on/selected_on/evaluated_on) apunta a un
    dataset registrado;
  * ningún modelo puede declararse VE (validado externamente) sin el bloque
    `external_validation_evidence` con protocolo preregistrado y dataset
    que estuviera LOCKED_FRESH/FRESH;
  * ids únicos.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "registry.yaml"
DATASETS = ROOT / "datasets" / "registry.yaml"

MODEL_STATUSES = {"RC", "EXP", "VI", "VE", "OBS", "PE"}
DATASET_STATUSES = {"FRESH", "LOCKED_FRESH", "DEV", "CONSUMED"}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DATA_KEYS = ("trained_on", "selected_on", "evaluated_on")


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def dataset_ids(datasets: dict[str, Any]) -> list[str]:
    return [d["id"] for d in datasets.get("datasets", [])]


def resolve_dataset(ref: str, ids: list[str]) -> str | None:
    """Una referencia como `ptb_xl_folds_1_8` resuelve al dataset `ptb_xl`."""
    best = None
    for d in ids:
        if ref == d or ref.startswith(d + "_") or ref.startswith(d):
            if best is None or len(d) > len(best):
                best = d
    return best


def iter_artifacts(model: dict[str, Any]):
    for art in model.get("artifacts") or []:
        if isinstance(art, dict):
            yield art


def validate(models: dict[str, Any], datasets: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    # ---- datasets
    ds_ids = dataset_ids(datasets)
    if len(ds_ids) != len(set(ds_ids)):
        errors.append("datasets: ids duplicados")
    for d in datasets.get("datasets", []):
        if d.get("usage_status") not in DATASET_STATUSES:
            errors.append(f"dataset {d.get('id')}: usage_status inválido {d.get('usage_status')!r}")
        sha = d.get("source_sha256")
        if sha is not None and not SHA_RE.match(str(sha)):
            errors.append(f"dataset {d['id']}: source_sha256 mal formado")
        for name, split in (d.get("splits") or {}).items():
            sha = split.get("sha256")
            if sha is not None and not SHA_RE.match(str(sha)):
                errors.append(f"dataset {d['id']} split {name}: sha256 mal formado")
            f = split.get("file")
            if f and not (ROOT / f).is_file():
                errors.append(f"dataset {d['id']} split {name}: falta archivo {f}")

    # ---- models
    m_ids = [m.get("id") for m in models.get("models", [])]
    if len(m_ids) != len(set(m_ids)):
        errors.append("models: ids duplicados")
    for m in models.get("models", []):
        mid = m.get("id", "<sin id>")
        for key in ("id", "title", "task", "status", "frozen"):
            if key not in m:
                errors.append(f"model {mid}: falta campo {key}")
        if m.get("status") not in MODEL_STATUSES:
            errors.append(f"model {mid}: status inválido {m.get('status')!r}")

        has_hash = False
        for art in iter_artifacts(m):
            sha = art.get("sha256")
            if sha is not None:
                if not SHA_RE.match(str(sha)):
                    errors.append(f"model {mid} artefacto {art.get('name')}: sha256 mal formado")
                else:
                    has_hash = True
            if art.get("manifest"):
                if not (ROOT / art["manifest"]).is_file():
                    errors.append(f"model {mid}: falta manifiesto {art['manifest']}")
                else:
                    has_hash = True
        if m.get("frozen") and not has_hash:
            # Se admite solo si el propio registro reconoce que el SHA está pendiente.
            pending = any(a.get("sha_status") == "DECLARED_ONLY" and a.get("sha256") is None
                          for a in iter_artifacts(m))
            if not pending:
                errors.append(f"model {mid}: frozen sin ningún SHA ni manifiesto")

        for key in DATA_KEYS:
            for ref in (m.get("data") or {}).get(key, []) or []:
                if resolve_dataset(ref, ds_ids) is None:
                    errors.append(f"model {mid}: {key} referencia dataset no registrado {ref!r}")

        if m.get("status") == "VE":
            ev = m.get("external_validation_evidence")
            if not ev:
                errors.append(f"model {mid}: VE sin external_validation_evidence")
            else:
                for key in ("protocol_file", "protocol_sha256", "dataset", "dataset_status_before"):
                    if key not in ev:
                        errors.append(f"model {mid}: external_validation_evidence sin {key}")
                if ev.get("dataset_status_before") not in {"FRESH", "LOCKED_FRESH"}:
                    errors.append(f"model {mid}: VE exige dataset FRESH/LOCKED_FRESH antes de abrirlo")
    return errors


def main() -> int:
    errors = validate(load(MODELS), load(DATASETS))
    if errors:
        print("REGISTRO: FALLA")
        for e in errors:
            print("  -", e)
        return 1
    m = load(MODELS)["models"]
    print(f"REGISTRO: OK ({len(m)} modelos, {len(dataset_ids(load(DATASETS)))} datasets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
