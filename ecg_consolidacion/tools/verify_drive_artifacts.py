"""Verificación de SOLO LECTURA de los SHA-256 del registro contra Google Drive.

Pensado para ejecutarse en Google Colab con Drive montado:

    from google.colab import drive; drive.mount('/content/drive')
    !git clone --depth 1 https://github.com/comincinieric56-maker/medcalc-clinico /content/medcalc-clinico
    !pip -q install pyyaml
    !python /content/medcalc-clinico/ecg_consolidacion/tools/verify_drive_artifacts.py \
        --drive-root /content/drive/MyDrive/MEDCALC --out /content/verify.json

Qué hace:
  * Para cada artefacto con `drive_path`, calcula el SHA-256 del archivo.
  * Para cada artefacto con `manifest` + `manifest_root`, verifica TODAS las
    filas del manifiesto (p. ej. los 35 joblib de R27).
  * Para artefactos con `search_name` y sin ruta, busca por nombre bajo
    --drive-root y reporta todas las copias y sus SHA.
  * No escribe, mueve ni borra nada en Drive. El informe se escribe en --out
    (por defecto, en el disco local de Colab) y nunca sobrescribe un archivo.

Salida: código 0 si no hay discrepancias; 1 si alguna; 2 si faltan archivos.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import MODELS, ROOT, iter_artifacts, load  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: Path, expected: str | None) -> dict:
    if not path.is_file():
        return {"path": str(path), "result": "MISSING"}
    got = sha256_file(path)
    if expected is None:
        return {"path": str(path), "sha256": got, "result": "RECORDED"}
    return {"path": str(path), "sha256": got, "expected": expected,
            "result": "OK" if got == expected else "MISMATCH"}


def find_by_name(root: Path, name: str, limit: int = 20) -> list[Path]:
    out = []
    for p in root.rglob(name):
        out.append(p)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", type=Path, default=Path("/content/drive/MyDrive/MEDCALC"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-search", action="store_true", help="no buscar por nombre (más rápido)")
    args = ap.parse_args()

    root: Path = args.drive_root
    if not root.is_dir():
        print(f"No existe {root}. ¿Está Drive montado?")
        return 2

    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "drive_root": str(root), "items": []}
    for m in load(MODELS)["models"]:
        for a in iter_artifacts(m):
            base = {"model": m["id"], "artifact": a.get("name")}
            if a.get("drive_path"):
                report["items"].append({**base, **check(root / a["drive_path"], a.get("sha256"))})
            elif a.get("manifest") and a.get("manifest_root"):
                with (ROOT / a["manifest"]).open(encoding="utf-8") as fh:
                    for row in csv.DictReader(fh):
                        p = root / a["manifest_root"] / row["relative_path"]
                        report["items"].append({**base, "manifest_row": row["relative_path"],
                                                **check(p, row["sha256"])})
            elif a.get("search_name") and not args.no_search:
                hits = find_by_name(root, a["search_name"])
                if not hits:
                    report["items"].append({**base, "result": "MISSING", "search_name": a["search_name"]})
                for p in hits:
                    report["items"].append({**base, "search_hit": True, **check(p, a.get("sha256"))})

    counts: dict[str, int] = {}
    for it in report["items"]:
        counts[it["result"]] = counts.get(it["result"], 0) + 1
    report["counts"] = counts

    out = args.out or Path(f"/content/medcalc_verify_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    if out.exists():
        print(f"{out} ya existe; no se sobrescribe.")
        return 2
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    for it in report["items"]:
        if it["result"] in {"MISMATCH", "MISSING"}:
            print(it["result"], it.get("model"), it.get("manifest_row") or it.get("artifact"), it.get("path", ""))
    print(f"Informe: {out}")
    if counts.get("MISMATCH"):
        return 1
    if counts.get("MISSING"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
