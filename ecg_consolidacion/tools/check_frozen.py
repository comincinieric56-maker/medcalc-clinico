"""Verifica que nada congelado haya cambiado.

El candado `frozen/FROZEN.lock.json` contiene:
  * el SHA-256 de cada artefacto de todo modelo `frozen: true` del registro;
  * el SHA-256 de cada archivo local protegido (splits, protocolos, manifiestos).

Uso:
  python tools/check_frozen.py           # verifica (CI)
  python tools/check_frozen.py --write   # regenera el candado (solo el responsable,
                                         # en un PR aprobado por CODEOWNERS)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import MODELS, ROOT, iter_artifacts, load  # noqa: E402

LOCK = ROOT / "frozen" / "FROZEN.lock.json"
PROTECTED_GLOBS = (
    "datasets/splits_*.json",
    "protocols/*.yaml",
    "models/manifests/*.csv",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_lock() -> dict:
    models = load(MODELS)["models"]
    frozen_models: dict[str, dict[str, str | None]] = {}
    for m in models:
        if not m.get("frozen"):
            continue
        arts = {}
        for a in iter_artifacts(m):
            arts[str(a.get("name"))] = a.get("sha256")
        frozen_models[m["id"]] = arts
    files = {}
    for pattern in PROTECTED_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            files[str(p.relative_to(ROOT))] = sha256_file(p)
    return {"schema": "MEDCALC_EKG_FROZEN_LOCK_V1", "models": frozen_models, "files": files}


def diff(expected: dict, actual: dict) -> list[str]:
    problems = []
    for mid, arts in expected.get("models", {}).items():
        if mid not in actual["models"]:
            problems.append(f"modelo congelado eliminado o descongelado: {mid}")
            continue
        for name, sha in arts.items():
            got = actual["models"][mid].get(name, "<ausente>")
            if got != sha:
                problems.append(f"{mid}/{name}: SHA cambió {sha} -> {got}")
    for path, sha in expected.get("files", {}).items():
        got = actual["files"].get(path, "<ausente>")
        if got != sha:
            problems.append(f"archivo protegido modificado o eliminado: {path}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    actual = build_lock()
    if args.write:
        LOCK.parent.mkdir(exist_ok=True)
        LOCK.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Candado escrito: {len(actual['models'])} modelos, {len(actual['files'])} archivos")
        return 0
    if not LOCK.is_file():
        print("FROZEN: FALLA — no existe frozen/FROZEN.lock.json")
        return 1
    expected = json.loads(LOCK.read_text(encoding="utf-8"))
    problems = diff(expected, actual)
    if problems:
        print("FROZEN: FALLA")
        for p in problems:
            print("  -", p)
        return 1
    new_frozen = set(actual["models"]) - set(expected["models"])
    if new_frozen:
        print(f"FROZEN: aviso — modelos congelados nuevos sin candado: {sorted(new_frozen)}; ejecutar --write en un PR aprobado")
        return 1
    print(f"FROZEN: OK ({len(expected['models'])} modelos, {len(expected['files'])} archivos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
