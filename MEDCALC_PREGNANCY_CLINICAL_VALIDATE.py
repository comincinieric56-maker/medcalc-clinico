#!/usr/bin/env python3
"""Mechanical safety gate for MEDCALC pregnancy clinical catalogue.

This validator checks identity, schema, draft/review separation and provenance
completeness. PASS DOES NOT CERTIFY CLINICAL ACCURACY, dosage, trimester claims,
product equivalence, reviewer approval or authorization to publish to Supabase.

Usage:
  python MEDCALC_PREGNANCY_CLINICAL_VALIDATE.py --strict
  python MEDCALC_PREGNANCY_CLINICAL_VALIDATE.py --strict --csv path/to/file.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_CSV = Path(__file__).with_name("MEDCALC_PREGNANCY_CLINICAL_GUIDANCE_V1.csv")
REQUIRED_COLUMNS = (
    "med_id", "generic_name", "status", "recommendation", "risk_summary",
    "clinical_considerations", "pregnancy_indication_note", "fetal_neonatal_risk",
    "evidence_level", "trimester_1", "trimester_2", "trimester_3",
    "source_authority", "source_url", "reviewed_at",
)
RECOMMENDATIONS = {
    "PREFERRED", "COMPATIBLE", "USE_WITH_CAUTION", "AVOID",
    "CONTRAINDICATED", "SPECIALIST_ONLY", "INSUFFICIENT_DATA",
}
EVIDENCE_LEVELS = {"HIGH", "MODERATE", "LOW", "LIMITED", "UNKNOWN"}
STATUS = {"REVIEWED", "PENDING_REVIEW"}
MED_RE = re.compile(r"^MED-\d{4}$")


def generic_or_invalid_url(value: str) -> bool:
    if not value or value.strip() != value:
        return True
    try:
        parts = urlsplit(value)
    except ValueError:
        return True
    if parts.scheme != "https" or not parts.netloc or parts.username:
        return True
    host = (parts.hostname or "").lower()
    if not host or "." not in host:
        return True
    path = parts.path.strip("/")
    if not path:
        return True
    if host in ("dailymed.nlm.nih.gov", "www.dailymed.nlm.nih.gov") and (
        path in ("dailymed", "dailymed/") or
        (path == "dailymed/drugInfo.cfm" and not parts.query) or
        (path == "dailymed/lookup.cfm" and not parts.query)
    ):
        return True
    return False


def audit(path: Path) -> dict:
    problems: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return {"counts": {}, "errors": [f"Archivo no encontrado: {path}"], "warnings": []}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, restkey="__EXTRA__", restval=None)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            problems.append("Esquema de 15 columnas no coincide con REQUIRED_COLUMNS")
        rows = list(reader)
    ids: set[str] = set()
    status_counts: Counter[str] = Counter()
    for n, row in enumerate(rows, start=2):
        id_ = str(row.get("med_id") or "").strip()
        status = str(row.get("status") or "").strip()
        rec = str(row.get("recommendation") or "").strip()
        source_url = str(row.get("source_url") or "").strip()
        label = f"fila {n} / {id_ or 'SIN MED-ID'}"
        if "__EXTRA__" in row or any(value is None for value in row.values()):
            problems.append(f"{label}: anchura CSV inválida")
        if not MED_RE.fullmatch(id_):
            problems.append(f"{label}: MED-ID inválido")
        elif id_ in ids:
            problems.append(f"{label}: MED-ID duplicado")
        ids.add(id_)
        if not (row.get("generic_name") or "").strip():
            problems.append(f"{label}: principio activo vacío")
        if status not in STATUS:
            problems.append(f"{label}: estado desconocido {status!r}")
        status_counts[status] += 1
        if rec not in RECOMMENDATIONS:
            problems.append(f"{label}: recomendación desconocida {rec!r}")
        level = str(row.get("evidence_level") or "").strip()
        if level not in EVIDENCE_LEVELS:
            problems.append(f"{label}: evidencia desconocida {level!r}")
        if status == "PENDING_REVIEW":
            if rec != "INSUFFICIENT_DATA":
                problems.append(f"{label}: pendiente con recomendación publicable")
            if any(row.get(c) for c in ("trimester_1", "trimester_2", "trimester_3")):
                problems.append(f"{label}: pendiente con clasificación por trimestre")
        if status == "REVIEWED":
            if rec == "INSUFFICIENT_DATA":
                problems.append(f"{label}: REVIEWED sin recomendación específica")
            if level == "UNKNOWN":
                problems.append(f"{label}: REVIEWED sin nivel de evidencia")
            if generic_or_invalid_url(source_url):
                problems.append(f"{label}: REVIEWED sin fuente HTTPS específica")
            for field in (
                "risk_summary", "clinical_considerations",
                "pregnancy_indication_note", "fetal_neonatal_risk",
                "source_authority", "reviewed_at",
            ):
                if not str(row.get(field) or "").strip():
                    problems.append(f"{label}: campo obligatorio vacío {field}")
            date = str(row.get("reviewed_at") or "")
            try:
                dt.date.fromisoformat(date)
            except ValueError:
                problems.append(f"{label}: reviewed_at no es fecha ISO")
            authority = (row.get("source_authority") or "").upper()
            if any(t in authority for t in ("NO MATCH", "NO EQUIVALENTE", "IDENTIDAD NO")):
                problems.append(f"{label}: fuente identificada como no equivalente")
            if any(row.get(c) for c in ("trimester_1", "trimester_2", "trimester_3")):
                warnings.append(f"{label}: comprobar manualmente evidencia POR TRIMESTRE")
    return {
        "counts": {"rows": len(rows), "unique_med_ids": len(ids), **dict(status_counts)},
        "errors": problems,
        "warnings": warnings,
        "limitations": [
            "Una URL específica no demuestra por sí misma que el contenido respalde cada afirmación.",
            "El sello REVIEWED no representa revisión clínica independiente ni autoriza publicación.",
            "Las afirmaciones por trimestre y las equivalencias de forma/sal requieren revisión humana.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--strict", action="store_true", help="Nonzero exit if structural/source defects")
    args = parser.parse_args()
    report = audit(args.csv)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
