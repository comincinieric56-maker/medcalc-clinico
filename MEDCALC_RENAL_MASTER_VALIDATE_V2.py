#!/usr/bin/env python3
# MEDCALC · RENAL MASTER · VALIDACIÓN ESTRICTA V2
#
# Entrada:
#   generated_renal_master/renal_master_evidence.csv
#
# Salida:
#   generated_renal_master_v2/
#
# Objetivo:
#   Reducir falsos positivos del primer barrido automático.
#   NO consulta APIs. NO modifica Supabase.
#
# Criterios estrictos:
# - Rechaza snippets genéricos ("Targeted renal text from SPL").
# - Rechaza monocomponente enlazado a etiqueta combinada.
# - Exige sección titulada específicamente renal/kidney.
# - Exige acción clínica renal explícita:
#     no ajuste / ajuste de dosis / umbral CrCl-eGFR-GFR /
#     no recomendado / pauta alrededor de diálisis.
# - Rechaza PK-only, cautelas inespecíficas y texto de sobredosis.
# - Todo lo aceptado sigue siendo CURRENT_REFERENCE, automatizable=FALSE.
#
# NO crea reglas numéricas automáticas nuevas.

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DIRECT_RENAL_TITLE = re.compile(
    r"\b("
    r"renal impairment|renal function|kidney impairment|kidney function|"
    r"hepatic and renal impairment|patients? with renal impairment|"
    r"dosage .* renal impairment|dose .* renal impairment"
    r")\b",
    re.I,
)

RENAL_TERM = re.compile(
    r"\b("
    r"renal impairment|renal function|kidney function|"
    r"creatinine clearance|\bcrcl\b|\begfr\b|\bgfr\b|"
    r"hemodialysis|haemodialysis|dialysis"
    r")\b",
    re.I,
)

NO_ADJUST = re.compile(
    r"\b("
    r"no (?:dose|dosage) adjustment(?: is)? (?:necessary|required|recommended)?|"
    r"does not require (?:dose|dosage) adjustment|"
    r"no adjustment .* renal impairment|"
    r"dose adjustment is not (?:necessary|required)"
    r")\b",
    re.I,
)

NOT_RECOMMENDED = re.compile(
    r"\b(not recommended|should not be used|avoid use|contraindicat)\b",
    re.I,
)

RENAL_THRESHOLD = re.compile(
    r"(?:crcl|creatinine clearance|egfr|gfr)"
    r"[^.;:\n]{0,100}"
    r"(?:<|>|≤|≥|less than|greater than|between|\d)",
    re.I,
)

DOSE_ACTION = re.compile(
    r"\b("
    r"dose|dosage|dosing|administer|administration|"
    r"reduce|reduction|interval|every|once daily|twice daily|"
    r"q\d+h|\d+(?:\.\d+)?\s*(?:mg|mcg|g)"
    r")\b",
    re.I,
)

DIALYSIS_DOSING = re.compile(
    r"(?:administer|give|dose|supplement)"
    r"[^.;:\n]{0,140}(?:after|before|following)?"
    r"[^.;:\n]{0,100}(?:hemodialysis|dialysis)"
    r"|"
    r"(?:hemodialysis|dialysis)"
    r"[^.;:\n]{0,140}(?:administer|give|dose|supplement)",
    re.I,
)

DOSE_ADJUSTMENT_TEXT = re.compile(
    r"\b("
    r"dose adjustment|dosage adjustment|"
    r"adjust(?:ment)? of (?:the )?dose|"
    r"reduce(?:d)? dose|"
    r"increase .* interval|decrease .* dose"
    r")\b",
    re.I,
)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("&", " / ")
    s = re.sub(r"[(),;:+\-]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def catalog_is_combo(name: str) -> bool:
    raw = str(name or "")
    n = norm(raw)
    return "/" in raw or " + " in raw or bool(re.search(r"\bY\b", n))


def title_is_combo(title: str) -> bool:
    # DailyMed titles routinely write active combination ingredients with AND.
    # Company text in [...] is removed before inspection.
    t = str(title or "").upper().split("[", 1)[0]
    return " AND " in t or "/" in t or " + " in t


def sqlq(value: Optional[str]) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def strict_decision(row: dict) -> tuple[str, str]:
    status = str(row.get("harvest_status") or "")

    if status == "ALREADY_VERIFIED_IN_MEDCALC":
        return "EXISTING_VERIFIED", "EXISTING"

    if status != "HIGH_MATCH_RENAL_EVIDENCE":
        return "REVIEW", status or "UNKNOWN_FIRST_PASS_STATUS"

    sections = str(row.get("renal_section_titles") or "")
    text = str(row.get("renal_text") or "")
    label_title = str(row.get("label_title") or "")
    generic_name = str(row.get("generic_name") or "")

    if sections.strip() == "Targeted renal text from SPL":
        return "REJECT", "FALLBACK_SNIPPET_NOT_RENAL_DOSING_SECTION"

    if not catalog_is_combo(generic_name) and title_is_combo(label_title):
        return "REJECT", "SINGLE_INGREDIENT_MATCHED_COMBINATION_LABEL"

    if not DIRECT_RENAL_TITLE.search(sections):
        return "REJECT", "NO_DIRECT_RENAL_SECTION_TITLE"

    if not RENAL_TERM.search(text):
        return "REJECT", "RENAL_TERMS_ABSENT"

    if NO_ADJUST.search(text):
        return "ACCEPT", "STRICT_NO_ADJUSTMENT"

    if NOT_RECOMMENDED.search(text):
        severe_renal = re.search(
            r"(severe|end.stage|advanced)[^.;]{0,100}renal impairment",
            text,
            re.I,
        )
        if RENAL_THRESHOLD.search(text) or severe_renal:
            return "ACCEPT", "STRICT_NOT_RECOMMENDED"

    if RENAL_THRESHOLD.search(text) and DOSE_ACTION.search(text):
        return "ACCEPT", "STRICT_RENAL_DOSING"

    if DIALYSIS_DOSING.search(text):
        return "ACCEPT", "STRICT_DIALYSIS_DOSING"

    # Require renal term and explicit dose-adjustment action within a local window.
    for m in RENAL_TERM.finditer(text):
        lo = max(0, m.start() - 220)
        hi = min(len(text), m.end() + 500)
        if DOSE_ADJUSTMENT_TEXT.search(text[lo:hi]):
            return "ACCEPT", "STRICT_RENAL_DOSING_TEXT"

    return "REJECT", "PK_CAUTION_OR_DIALYSIS_ONLY_NO_DOSING_ACTION"


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def audit_values(rows: list[dict]) -> str:
    vals = []
    for r in rows:
        vals.append(
            "("
            + ",".join(
                [
                    sqlq(r.get("med_id")),
                    sqlq(r.get("generic_name")),
                    sqlq(r.get("v2_decision")),
                    sqlq(r.get("v2_reason")),
                    sqlq(r.get("harvest_status")),
                    sqlq(r.get("match_confidence")),
                    sqlq(r.get("setid")),
                    sqlq(r.get("published_date")),
                    sqlq(truncate(r.get("label_title") or "", 1200)),
                    sqlq(r.get("source_url")),
                    sqlq(truncate(r.get("renal_section_titles") or "", 1200)),
                ]
            )
            + ")"
        )
    return ",\n".join(vals)


def accepted_values(rows: list[dict]) -> str:
    vals = []
    for r in rows:
        vals.append(
            "("
            + ",".join(
                [
                    sqlq(r.get("med_id")),
                    sqlq(r.get("generic_name")),
                    sqlq(r.get("v2_reason")),
                    sqlq(r.get("setid")),
                    sqlq(r.get("published_date")),
                    sqlq(truncate(r.get("label_title") or "", 1500)),
                    sqlq(r.get("source_url")),
                    sqlq(truncate(r.get("renal_section_titles") or "", 1500)),
                    sqlq(truncate(r.get("renal_text") or "", 12000)),
                ]
            )
            + ")"
        )
    return ",\n".join(vals)


def make_sql(all_rows: list[dict], accepted: list[dict]) -> str:
    av = audit_values(all_rows)
    ev = accepted_values(accepted)
    if not ev:
        ev = "('','','','','','','','','')"

    generated = datetime.now(timezone.utc).isoformat()

    return f"""-- =====================================================================
-- MEDCALC · RENAL MASTER · VALIDACIÓN ESTRICTA V2
-- Generado UTC: {generated}
--
-- IMPORTANTE
--   Este SQL reemplaza al SQL de la primera pasada.
--   NO ejecutar MEDCALC_RENAL_MAESTRO_GENERADO.sql de V1.
--
-- V2:
--   • Rechaza snippets renales inespecíficos.
--   • Rechaza monocomponente -> etiqueta combinada.
--   • Exige sección renal directa.
--   • Exige acción renal de dosificación explícita.
--   • No crea reglas automáticas nuevas.
--   • Conserva CURRENT_AUTO existente.
-- =====================================================================


CREATE TABLE IF NOT EXISTS public.renal_master_validation_v2 (
    medication_id uuid PRIMARY KEY REFERENCES public.medications(id) ON DELETE CASCADE,
    med_id text NOT NULL,
    generic_name text NOT NULL,
    v2_decision text NOT NULL,
    v2_reason text NOT NULL,
    first_pass_status text,
    match_confidence text,
    setid text,
    published_date text,
    label_title text,
    source_url text,
    renal_section_titles text,
    validated_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);


WITH raw(
 med_id,generic_name,v2_decision,v2_reason,first_pass_status,match_confidence,
 setid,published_date,label_title,source_url,renal_section_titles
) AS (
VALUES
{av}
)
INSERT INTO public.renal_master_validation_v2
(
 medication_id,med_id,generic_name,v2_decision,v2_reason,
 first_pass_status,match_confidence,setid,published_date,label_title,
 source_url,renal_section_titles,validated_at,updated_at
)
SELECT
 m.id,r.med_id,r.generic_name,r.v2_decision,r.v2_reason,
 r.first_pass_status,r.match_confidence,r.setid,r.published_date,
 r.label_title,r.source_url,r.renal_section_titles,NOW(),NOW()
FROM raw r
JOIN public.medications m ON m.med_id=r.med_id
ON CONFLICT (medication_id) DO UPDATE
SET generic_name=EXCLUDED.generic_name,
    v2_decision=EXCLUDED.v2_decision,
    v2_reason=EXCLUDED.v2_reason,
    first_pass_status=EXCLUDED.first_pass_status,
    match_confidence=EXCLUDED.match_confidence,
    setid=EXCLUDED.setid,
    published_date=EXCLUDED.published_date,
    label_title=EXCLUDED.label_title,
    source_url=EXCLUDED.source_url,
    renal_section_titles=EXCLUDED.renal_section_titles,
    validated_at=NOW(),
    updated_at=NOW();


-- =====================================================================
-- SOLO LOS ACCEPT V2 PUEDEN CREAR NUEVA EVIDENCIA CLÍNICA
-- =====================================================================

WITH evidence(
 med_id,generic_name,v2_reason,setid,published_date,label_title,
 source_url,renal_section_titles,renal_text
) AS (
VALUES
{ev}
)
INSERT INTO public.sources
(id,title,organization,url,page,source_type,last_verified,created_at)
SELECT
 gen_random_uuid(),
 CONCAT('DailyMed — ',e.generic_name,' — renal labeling V2'),
 'U.S. National Library of Medicine · DailyMed',
 e.source_url,
 NULLIF(e.renal_section_titles,''),
 'RENAL_REFERENCIA',
 CURRENT_DATE,
 NOW()
FROM evidence e
WHERE e.med_id<>''
  AND NOT EXISTS (
      SELECT 1 FROM public.sources s WHERE s.url=e.source_url
  );


WITH evidence(
 med_id,generic_name,v2_reason,setid,published_date,label_title,
 source_url,renal_section_titles,renal_text
) AS (
VALUES
{ev}
)
INSERT INTO public.renal_bibliography
(
 id,medication_id,drug_name_source,normal_dose,adjustment_method,
 recommendations,verified,status,source_id,created_at,updated_at
)
SELECT
 gen_random_uuid(),
 m.id,
 m.generic_name,
 NULL,
 CONCAT('CURRENT_DAILYMED_STRICT_V2:',e.v2_reason),
 e.renal_text,
 TRUE,
 'PUBLISHED',
 s.id,
 NOW(),
 NOW()
FROM evidence e
JOIN public.medications m ON m.med_id=e.med_id
JOIN public.sources s ON s.url=e.source_url
WHERE e.med_id<>''
  AND NOT EXISTS (
      SELECT 1
      FROM public.renal_bibliography rb
      WHERE rb.medication_id=m.id
        AND rb.source_id=s.id
        AND rb.status='PUBLISHED'
  );


WITH evidence(
 med_id,generic_name,v2_reason,setid,published_date,label_title,
 source_url,renal_section_titles,renal_text
) AS (
VALUES
{ev}
)
INSERT INTO public.renal_rules
(
 id,medication_id,indication,population,route,renal_metric,range_text,
 lower_limit,upper_limit,lower_inclusive,upper_inclusive,
 adjusted_regimen,rule_type,notes,automatizable,status,source_id,reviewed_at
)
SELECT
 gen_random_uuid(),
 m.id,
 CONCAT(m.generic_name,' — ficha renal regulatoria actual · V2'),
 'Adulto / según ficha regulatoria',
 NULL,
 NULL,
 'Referencia clínica actual',
 NULL,NULL,FALSE,FALSE,
 e.renal_text,
 CASE
   WHEN e.v2_reason='STRICT_NO_ADJUSTMENT' THEN 'NO_AJUSTE'
   ELSE 'PRECAUCION'
 END,
 CONCAT(
   'DailyMed SPL ',e.setid,' · ',COALESCE(e.published_date,''),' · ',
   e.v2_reason,
   '. Referencia regulatoria no automatizable; no sustituye CURRENT_AUTO.'
 ),
 FALSE,
 'PUBLISHED',
 s.id,
 NOW()
FROM evidence e
JOIN public.medications m ON m.med_id=e.med_id
JOIN public.sources s ON s.url=e.source_url
WHERE e.med_id<>''
  AND NOT EXISTS (
      SELECT 1
      FROM public.renal_rules rr
      WHERE rr.medication_id=m.id
        AND rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V2')
        AND rr.source_id=s.id
        AND rr.status IN ('PUBLISHED','PENDING_REVIEW')
  );


-- Certificar únicamente las reglas V2 recién definidas.
INSERT INTO public.renal_rule_validation
(rule_id,validation_class,evidence_note,validated_at,updated_at)
SELECT
 rr.id,
 'CURRENT_REFERENCE',
 CONCAT(
   'RENAL MASTER V2: evidencia regulatoria DailyMed con filtro estricto. ',
   COALESCE(v.v2_reason,'')
 ),
 NOW(),NOW()
FROM public.renal_rules rr
JOIN public.medications m ON m.id=rr.medication_id
JOIN public.renal_master_validation_v2 v ON v.medication_id=m.id
JOIN public.sources s ON s.id=rr.source_id
WHERE v.v2_decision='ACCEPT'
  AND rr.status='PUBLISHED'
  AND COALESCE(rr.automatizable,FALSE)=FALSE
  AND rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V2')
  AND s.url=v.source_url
ON CONFLICT (rule_id) DO UPDATE
SET validation_class='CURRENT_REFERENCE',
    evidence_note=EXCLUDED.evidence_note,
    validated_at=NOW(),
    updated_at=NOW();


-- Cerrar como referencia únicamente si no existe cierre CURRENT_AUTO.
INSERT INTO public.renal_phase6_review
(
 medication_id,med_id,generic_name,original_priority,review_batch,
 phase6_status,disposition,source_id,decision_note,validated_at,updated_at
)
SELECT
 m.id,m.med_id,m.generic_name,2,8,
 'CLOSED_CURRENT_REFERENCE',
 'REFERENCIA_ACTUAL_VALIDADA',
 rr.source_id,
 rr.adjusted_regimen,
 NOW(),NOW()
FROM public.medications m
JOIN public.renal_master_validation_v2 v
  ON v.medication_id=m.id AND v.v2_decision='ACCEPT'
JOIN public.renal_rules rr
  ON rr.medication_id=m.id
 AND rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V2')
JOIN public.renal_rule_validation rv
  ON rv.rule_id=rr.id AND rv.validation_class='CURRENT_REFERENCE'
WHERE NOT EXISTS (
    SELECT 1
    FROM public.renal_phase6_review p
    WHERE p.medication_id=m.id
      AND p.phase6_status='CLOSED_CURRENT_AUTO'
)
ON CONFLICT (medication_id) DO UPDATE
SET review_batch=CASE
      WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
      THEN public.renal_phase6_review.review_batch
      ELSE 8
    END,
    phase6_status=CASE
      WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
      THEN public.renal_phase6_review.phase6_status
      ELSE 'CLOSED_CURRENT_REFERENCE'
    END,
    disposition=CASE
      WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
      THEN public.renal_phase6_review.disposition
      ELSE 'REFERENCIA_ACTUAL_VALIDADA'
    END,
    source_id=CASE
      WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
      THEN public.renal_phase6_review.source_id
      ELSE EXCLUDED.source_id
    END,
    decision_note=CASE
      WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
      THEN public.renal_phase6_review.decision_note
      ELSE EXCLUDED.decision_note
    END,
    validated_at=NOW(),
    updated_at=NOW();


INSERT INTO public.medication_module_status (medication_id,renal_status)
SELECT medication_id,'PUBLISHED'
FROM public.renal_master_validation_v2
WHERE v2_decision IN ('ACCEPT','EXISTING_VERIFIED')
ON CONFLICT (medication_id) DO UPDATE
SET renal_status='PUBLISHED';


-- =====================================================================
-- RESULTADO FINAL V2
-- =====================================================================

SELECT
 COUNT(*) AS medicamentos_auditados,
 COUNT(*) FILTER (WHERE v2_decision='EXISTING_VERIFIED') AS existentes_verificados,
 COUNT(*) FILTER (WHERE v2_decision='ACCEPT') AS nuevas_referencias_v2,
 COUNT(*) FILTER (WHERE v2_decision='REJECT') AS falsos_positivos_rechazados,
 COUNT(*) FILTER (WHERE v2_decision='REVIEW') AS requieren_fuente_secundaria_revision
FROM public.renal_master_validation_v2;
"""


def main() -> None:
    in_path = Path("generated_renal_master/renal_master_evidence.csv")
    out_dir = Path("generated_renal_master_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise SystemExit(
            "No existe generated_renal_master/renal_master_evidence.csv. "
            "Ejecute primero MEDCALC Renal Master 1122."
        )

    rows = load_rows(in_path)

    if len(rows) != 1122:
        raise SystemExit(
            f"SEGURIDAD: se esperaban 1122 filas de evidencia; se encontraron {len(rows)}."
        )

    for r in rows:
        decision, reason = strict_decision(r)
        r["v2_decision"] = decision
        r["v2_reason"] = reason

    accepted = [r for r in rows if r["v2_decision"] == "ACCEPT"]
    rejected = [r for r in rows if r["v2_decision"] == "REJECT"]
    review = [r for r in rows if r["v2_decision"] == "REVIEW"]
    existing = [r for r in rows if r["v2_decision"] == "EXISTING_VERIFIED"]

    fields = list(rows[0].keys())

    write_csv(out_dir / "renal_v2_full_audit.csv", rows, fields)
    write_csv(out_dir / "renal_v2_accepted_current_reference.csv", accepted, fields)
    write_csv(out_dir / "renal_v2_rejected_false_positives.csv", rejected, fields)
    write_csv(out_dir / "renal_v2_needs_secondary_source.csv", review, fields)

    reason_counts = {}
    for r in rows:
        reason_counts[r["v2_reason"]] = reason_counts.get(r["v2_reason"], 0) + 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": len(rows),
        "existing_verified": len(existing),
        "strict_v2_accepted_current_reference": len(accepted),
        "first_pass_false_positives_rejected": len(rejected),
        "needs_secondary_source_or_review": len(review),
        "decision_total_check": len(existing) + len(accepted) + len(rejected) + len(review),
        "reason_counts": dict(sorted(reason_counts.items())),
        "safety": {
            "new_automatic_rules_created": False,
            "existing_current_auto_preserved": True,
            "targeted_snippet_fallback_allowed": False,
            "single_ingredient_to_combination_label_allowed": False,
            "direct_renal_section_required": True,
            "explicit_renal_dosing_action_required": True,
        },
    }

    (out_dir / "renal_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (out_dir / "MEDCALC_RENAL_MAESTRO_VALIDADO_V2.sql").write_text(
        make_sql(rows, accepted),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
