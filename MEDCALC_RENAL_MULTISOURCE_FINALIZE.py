#!/usr/bin/env python3
"""
MEDCALC · RENAL MULTIFUENTE · FINALIZER
Combina la salida de:
- MEDCALC actual (estado del catálogo)
- RxNorm/RxNav
- DailyMed SPL
- openFDA Drug Labeling
- fuentes locales del repositorio:
    renal_biblio_verificada_2025.csv
    ajuste_renal.csv

No convierte bibliografía histórica/local en CURRENT_REFERENCE por sí sola.
Solo la usa como capa de soporte/corroboración y para priorizar pendientes.
"""

from __future__ import annotations
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
OUT = Path("generated_renal_multisource")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).upper()
    return re.sub(r"\s+", " ", s).strip()

def pick_col(rows, candidates):
    if not rows:
        return None
    cols = list(rows[0].keys())
    bynorm = {norm(c): c for c in cols}
    for cand in candidates:
        if norm(cand) in bynorm:
            return bynorm[norm(cand)]
    return None

def local_index(rows, name_candidates, medid_candidates):
    idx_name = {}
    idx_mid = {}
    if not rows:
        return idx_name, idx_mid
    nc = pick_col(rows, name_candidates)
    mc = pick_col(rows, medid_candidates)
    for r in rows:
        if nc and r.get(nc):
            idx_name.setdefault(norm(r[nc]), []).append(r)
        if mc and r.get(mc):
            idx_mid.setdefault(str(r[mc]).strip(), []).append(r)
    return idx_name, idx_mid

catalog = read_csv("MEDCALC_RENAL_MASTER_CATALOGO_1122.csv")
if len(catalog) != 1122:
    raise SystemExit(f"SEGURIDAD: catálogo esperado 1122; encontrado {len(catalog)}.")

v2 = read_csv("generated_renal_master_v2/renal_v2_full_audit.csv")
v3 = read_csv("generated_renal_master_v3/renal_v3_identity_accepted.csv")
v4 = read_csv("generated_renal_master_v4_openfda/renal_v4_full_audit.csv")

if len(v2) != 1122:
    raise SystemExit(f"SEGURIDAD: V2 esperado 1122; encontrado {len(v2)}.")
if len(v3) != 180:
    raise SystemExit(f"SEGURIDAD: V3 esperado 180; encontrado {len(v3)}.")
if len(v4) != 879:
    raise SystemExit(f"SEGURIDAD: V4 esperado 879; encontrado {len(v4)}.")

# Local repo sources: best effort.
local_biblio = read_csv("renal_biblio_verificada_2025.csv")
local_adjust = read_csv("ajuste_renal.csv")

b_name, b_mid = local_index(
    local_biblio,
    ["principio_activo", "generic_name", "medicamento", "drug_name_source", "nombre"],
    ["med_id", "id_revision", "medid"],
)
a_name, a_mid = local_index(
    local_adjust,
    ["principio_activo", "generic_name", "medicamento", "nombre", "farmaco"],
    ["med_id", "id_revision", "medid"],
)

v2_by = {r["med_id"]: r for r in v2}
v3_ids = {r["med_id"] for r in v3}
v4_by = {r["med_id"]: r for r in v4}

rows = []
for c in catalog:
    mid = c["med_id"]
    name = c["generic_name"]
    x2 = v2_by.get(mid, {})
    x4 = v4_by.get(mid, {})

    existing = x2.get("v2_decision") == "EXISTING_VERIFIED"
    daily = mid in v3_ids
    openfda = x4.get("v4_decision") == "ACCEPT"

    local_b = bool(b_mid.get(mid) or b_name.get(norm(name)))
    local_a = bool(a_mid.get(mid) or a_name.get(norm(name)))

    current_sources = int(existing) + int(daily) + int(openfda)
    support_sources = int(local_b) + int(local_a)

    if existing:
        resolution = "EXISTING_VERIFIED_MEDCALC"
    elif daily and openfda:
        resolution = "CURRENT_MULTI_OFFICIAL_CONCORDANT"
    elif daily:
        resolution = "CURRENT_DAILYMED_IDENTITY_CONFIRMED"
    elif openfda:
        resolution = "CURRENT_OPENFDA_IDENTITY_CONFIRMED"
    elif local_b or local_a:
        resolution = "LOCAL_SUPPORT_ONLY_NEEDS_CURRENT_SOURCE"
    else:
        resolution = "UNRESOLVED"

    # Local sources can support prioritization but never upgrade current status.
    if resolution in {"UNRESOLVED", "LOCAL_SUPPORT_ONLY_NEEDS_CURRENT_SOURCE"}:
        next_action = (
            "CURRENT_SOURCE_REQUIRED_WITH_LOCAL_SUPPORT"
            if (local_b or local_a)
            else "CURRENT_SOURCE_REQUIRED"
        )
    else:
        next_action = "STRUCTURALLY_RESOLVED"

    rows.append({
        "med_id": mid,
        "generic_name": name,
        "resolution": resolution,
        "current_source_count": current_sources,
        "local_support_count": support_sources,
        "existing_medcalc_verified": existing,
        "dailymed_v3_confirmed": daily,
        "openfda_v4_confirmed": openfda,
        "local_renal_biblio_2025": local_b,
        "local_ajuste_renal_csv": local_a,
        "v2_reason": x2.get("v2_reason", ""),
        "v4_reason": x4.get("v4_reason", ""),
        "next_action": next_action,
    })

fields = list(rows[0].keys())
with (OUT / "renal_multisource_matrix_1122.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

resolved = [r for r in rows if r["next_action"] == "STRUCTURALLY_RESOLVED"]
pending = [r for r in rows if r["next_action"] != "STRUCTURALLY_RESOLVED"]

with (OUT / "renal_multisource_resolved.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(resolved)

with (OUT / "renal_multisource_pending.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(pending)

counts = {}
for r in rows:
    counts[r["resolution"]] = counts.get(r["resolution"], 0) + 1

summary = {
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "catalog_rows": 1122,
    "resolved_structurally": len(resolved),
    "still_needs_current_source": len(pending),
    "resolution_counts": dict(sorted(counts.items())),
    "local_repository_support": {
        "renal_biblio_verificada_2025_rows": len(local_biblio),
        "ajuste_renal_rows": len(local_adjust),
        "pending_with_local_support": sum(
            1 for r in pending if r["local_support_count"] > 0
        ),
    },
    "source_roles": {
        "RxNorm_RxNav": "drug identity normalization",
        "DailyMed_SPL": "current regulatory renal labeling",
        "openFDA_Drug_Label": "current FDA SPL renal labeling",
        "DrugsFDA_openFDA_harmonization": "regulatory/product identity support used by FDA harmonization layer",
        "MEDCALC_existing": "existing validated CURRENT_AUTO/CURRENT_REFERENCE/TDM",
        "renal_biblio_verificada_2025": "local secondary support; never upgrades alone",
        "ajuste_renal_csv": "local secondary support; never upgrades alone",
    },
    "safety": {
        "local_source_alone_can_close_current_reference": False,
        "new_numeric_automatic_rules_created": False,
        "existing_current_auto_preserved": True,
    },
}
(OUT / "renal_multisource_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

# V4 already generates the single load SQL for V3+V4 current sources.
src_sql = Path("generated_renal_master_v4_openfda/MEDCALC_RENAL_MASTER_FINAL_V4.sql")
if not src_sql.exists():
    raise SystemExit("Falta SQL FINAL V4.")
sql = src_sql.read_text(encoding="utf-8")
header = f"""-- =====================================================================
-- MEDCALC RENAL MULTIFUENTE · CANDIDATO FINAL DE CARGA
-- Matriz multifuentе generada: {summary['generated_at_utc']}
--
-- La matriz de 1.122 MED-ID se encuentra en:
-- generated_renal_multisource/renal_multisource_matrix_1122.csv
--
-- ESTE SQL SOLO CARGA EVIDENCIA REGULATORIA ACTUAL CONFIRMADA.
-- Las fuentes locales 2025 se usan como soporte y NO se convierten
-- automáticamente en CURRENT_REFERENCE.
-- =====================================================================

"""
(OUT / "MEDCALC_RENAL_MULTISOURCE_FINAL.sql").write_text(
    header + sql, encoding="utf-8"
)

print(json.dumps(summary, ensure_ascii=False, indent=2))
