"""Falla si el código intenta tocar conjuntos reservados o consumidos.

Escanea el código ejecutable (engine/, benchmarks/, tools/, ui/) con los patrones
de `datasets/registry.yaml: reserved_path_patterns`. Un archivo puede declarar
una excepción aprobada con una línea:

    # reserved-access-approved: <dataset_id> PR#<número>

La excepción solo es válida si el PR la aprueba el responsable (CODEOWNERS).

Uso:  python tools/check_reserved_paths.py [archivos...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import DATASETS, ROOT, load  # noqa: E402

SCAN_DIRS = ("engine", "benchmarks", "tools", "ui")
SCAN_SUFFIXES = {".py", ".ipynb", ".sh", ".yaml", ".yml", ".json", ".js", ".ts", ".tsx"}
SELF_EXEMPT = {"tools/check_reserved_paths.py", "tools/registry.py"}
APPROVAL_RE = re.compile(r"reserved-access-approved:\s*([a-z0-9_]+)\s+PR#\d+")


def compile_rules():
    raw = load(DATASETS).get("reserved_path_patterns") or {}
    rules = {}
    for ds, spec in raw.items():
        rules[ds] = {
            "any": [re.compile(p) for p in spec.get("any", [])],
            "all": [re.compile(p) for p in spec.get("all", [])],
        }
    return rules


def candidate_files(argv: list[str]) -> list[Path]:
    if argv:
        paths = [ROOT / a for a in argv]
    else:
        paths = [p for d in SCAN_DIRS for p in (ROOT / d).rglob("*") if p.is_file()]
    out = []
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        if rel in SELF_EXEMPT or not p.is_file():
            continue
        if not any(rel.startswith(d + "/") for d in SCAN_DIRS):
            continue
        if p.suffix in SCAN_SUFFIXES:
            out.append(p)
    return out


def scan_text(text: str, rules) -> list[str]:
    approved = set(APPROVAL_RE.findall(text))
    hits = []
    for ds, r in rules.items():
        if ds in approved:
            continue
        if r["any"] and any(p.search(text) for p in r["any"]):
            hits.append(ds)
        elif r["all"] and all(p.search(text) for p in r["all"]):
            hits.append(ds)
    return hits


def main(argv: list[str]) -> int:
    rules = compile_rules()
    problems = []
    for p in candidate_files(argv):
        hits = scan_text(p.read_text(encoding="utf-8", errors="replace"), rules)
        for ds in hits:
            problems.append(f"{p.relative_to(ROOT)} -> conjunto reservado '{ds}'")
    if problems:
        print("RESERVADOS: FALLA (acceso no aprobado a conjuntos reservados/consumidos)")
        for pr in problems:
            print("  -", pr)
        return 1
    print("RESERVADOS: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
