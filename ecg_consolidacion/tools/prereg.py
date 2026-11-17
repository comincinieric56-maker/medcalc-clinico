"""Preregistro de protocolos.

Un protocolo (YAML en protocols/) se registra ANTES de producir resultados:

    python tools/prereg.py register protocols/MI_PROTOCOLO.yaml --by "nombre"

Esto añade una fila a protocols/PREREGISTRATION_LEDGER.csv con su SHA-256 y la
fecha UTC. Ese commit debe fusionarse antes del primer commit con resultados.
Desde entonces el archivo es inmutable (lo vigila tools/check_frozen.py).

    python tools/prereg.py verify      # CI: el ledger coincide con los archivos
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "protocols" / "PREREGISTRATION_LEDGER.csv"
FIELDS = ["protocol_file", "sha256", "registered_utc", "registered_by", "note"]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_ledger() -> list[dict]:
    if not LEDGER.is_file():
        return []
    with LEDGER.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def register(path: str, by: str, note: str) -> int:
    p = (ROOT / path).resolve()
    rel = p.relative_to(ROOT).as_posix()
    if not rel.startswith("protocols/"):
        print("El protocolo debe estar en protocols/")
        return 1
    if any(r["protocol_file"] == rel for r in read_ledger()):
        print(f"{rel} ya está registrado; un protocolo registrado no se modifica: crea una versión nueva.")
        return 1
    new = not LEDGER.is_file()
    with LEDGER.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"protocol_file": rel, "sha256": sha256_file(p),
                    "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "registered_by": by, "note": note})
    print(f"Registrado {rel}")
    return 0


def verify() -> int:
    problems = []
    for r in read_ledger():
        p = ROOT / r["protocol_file"]
        if not p.is_file():
            problems.append(f"falta {r['protocol_file']}")
        elif sha256_file(p) != r["sha256"]:
            problems.append(f"protocolo modificado tras registro: {r['protocol_file']}")
    if problems:
        print("PREREGISTRO: FALLA")
        for x in problems:
            print("  -", x)
        return 1
    print(f"PREREGISTRO: OK ({len(read_ledger())} protocolos)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("path")
    r.add_argument("--by", required=True)
    r.add_argument("--note", default="")
    sub.add_parser("verify")
    a = ap.parse_args()
    return register(a.path, a.by, a.note) if a.cmd == "register" else verify()


if __name__ == "__main__":
    sys.exit(main())
