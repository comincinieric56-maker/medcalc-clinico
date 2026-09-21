#!/usr/bin/env python3
"""
MEDCALC · RENAL MASTER · V3 IDENTITY GATE
=========================================

Input:
  generated_renal_master_v2/renal_v2_accepted_current_reference.csv

Purpose:
  Validate that each DailyMed SPL belongs to the SAME drug identity as the
  MED-ID before allowing the V2 renal evidence to be loaded.

This is deliberately conservative.

Required conditions for AUTO-PASS identity:
  1. Catalog generic vs RxNorm-resolved name must remain highly concordant
     after removal of common salt words.
  2. Combination products cannot collapse into a single ingredient.
  3. The exact DailyMed SETID selected in V1/V2 must also be returned by a
     DailyMed lookup using the exact RxCUI resolved from the RxNorm name.
  4. If any identity check is uncertain, the item is moved to REVIEW and is
     NOT included in the SQL loader.

No numeric renal rule is generated.
All new accepted content remains CURRENT_REFERENCE, automatizable=FALSE.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from rapidfuzz import fuzz

RXNORM = "https://rxnav.nlm.nih.gov/REST"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

SALT_WORDS = {
    "HYDROCHLORIDE","HCL","SODIUM","POTASSIUM","CALCIUM","MAGNESIUM",
    "MESYLATE","BESYLATE","MALEATE","FUMARATE","SUCCINATE","ACETATE",
    "TARTRATE","CITRATE","PHOSPHATE","SULFATE","BROMIDE","CHLORIDE",
    "DIHYDRATE","MONOHYDRATE","TRIHYDRATE","HEMIHYDRATE","ANHYDROUS",
    "PAMOATE","XINAFOATE","LACTATE","GLUCONATE",
    "CLORHIDRATO","SODICO","SODICA","POTASICO","POTASICA","CALCICO",
    "CALCICA","MESILATO","BESILATO","MALEATO","FUMARATO","SUCCINATO",
    "ACETATO","TARTRATO","CITRATO","FOSFATO","SULFATO","BROMURO",
    "CLORURO","PAMOATO","LACTATO","GLUCONATO","DE",
}

# Counterions / formulation terms must never be accepted as the drug identity.
BAD_RXNORM_IDENTITIES = {
    "MALATE","CHLORIDE","SODIUM","POTASSIUM","CALCIUM","MAGNESIUM",
    "PHOSPHATE","SULFATE","CITRATE","ACETATE","FUMARATE","MALEATE",
    "TARTRATE","LACTATE","GLUCONATE","BROMIDE","WATER",
}

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("&", " / ")
    s = re.sub(r"[^A-Z0-9/+ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def strip_salts(s: str) -> str:
    toks = [
        t for t in norm(s).split()
        if t not in SALT_WORDS and t not in {"ACID","ACIDO"}
    ]
    return " ".join(toks)

def is_combo(s: str) -> bool:
    raw = str(s or "")
    return "/" in raw or " + " in raw or bool(re.search(r"\bY\b", norm(raw)))

def combo_parts(s: str) -> list[str]:
    return [
        strip_salts(x)
        for x in re.split(r"\s*/\s*|\s+\+\s+|\s+\bY\b\s+", str(s or ""), flags=re.I)
        if strip_salts(x)
    ]

def name_score(a: str, b: str) -> float:
    a = strip_salts(a)
    b = strip_salts(b)
    if not a or not b:
        return 0.0
    return float(max(fuzz.WRatio(a,b), fuzz.token_set_ratio(a,b)))

def sqlq(v: Optional[str]) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"

def trunc(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n-1] + "…"

class HTTP:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "MEDCALC-RenalIdentityV3/1.0"

    def get_json(self, url, params=None, tries=4):
        last = None
        for i in range(tries):
            try:
                r = self.s.get(url, params=params, timeout=35)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                time.sleep(0.12)
                return r.json()
            except Exception as e:
                last = e
                time.sleep(1.0 + i)
        raise last

def exact_rxcuis(http: HTTP, rxnorm_name: str) -> list[str]:
    j = http.get_json(
        f"{RXNORM}/rxcui.json",
        params={"name": rxnorm_name, "search": 2, "allsrc": 0},
    ) or {}
    return list(((j.get("idGroup") or {}).get("rxnormId")) or [])

def dailymed_setids_for_rxcui(http: HTTP, rxcui: str) -> set[str]:
    j = http.get_json(
        f"{DAILYMED}/spls.json",
        params={"rxcui": rxcui, "pagesize": 100, "page": 1},
    ) or {}
    return {
        str(x.get("setid") or "").lower()
        for x in (j.get("data") or [])
        if x.get("setid")
    }

def identity_check(http: HTTP, row: dict) -> tuple[str,str,str]:
    generic = row.get("generic_name") or ""
    rxname = row.get("rxnorm_name") or ""
    setid = str(row.get("setid") or "").lower().strip()

    if not generic or not rxname or not setid:
        return "REVIEW", "MISSING_IDENTITY_FIELDS", ""

    rx_core = strip_salts(rxname)
    if rx_core in BAD_RXNORM_IDENTITIES:
        return "REJECT", "RXNORM_RESOLVED_TO_COUNTERION_NOT_DRUG", ""

    score = name_score(generic, rxname)
    if score < 90:
        return "REVIEW", f"CATALOG_RXNORM_IDENTITY_SCORE_{score:.1f}_LT90", ""

    # Combinations must not resolve to a single unrelated ingredient.
    parts = combo_parts(generic)
    if is_combo(generic):
        if len(parts) < 2:
            return "REVIEW", "COMBINATION_PARSE_UNCERTAIN", ""
        # The RxNorm name must carry evidence for every catalog component.
        rxnorm_norm = strip_salts(rxname)
        part_scores = [
            max(
                fuzz.partial_ratio(p, rxnorm_norm),
                fuzz.token_set_ratio(p, rxnorm_norm),
            )
            for p in parts
        ]
        if any(x < 80 for x in part_scores):
            return "REVIEW", "COMBINATION_COLLAPSED_OR_MISSING_COMPONENT", ""

    rxcuis = exact_rxcuis(http, rxname)
    if not rxcuis:
        return "REVIEW", "RXNORM_NAME_NO_LONGER_RESOLVES_EXACTLY", ""

    # Crucial cross-check: the chosen label must actually map to the exact
    # RxNorm concept, not merely contain a similar substring.
    supporting = []
    for rxcui in rxcuis[:8]:
        ids = dailymed_setids_for_rxcui(http, rxcui)
        if setid in ids:
            supporting.append(rxcui)

    if not supporting:
        return "REJECT", "DAILYMED_SETID_NOT_MAPPED_TO_EXACT_RXNORM_CONCEPT", ",".join(rxcuis)

    return "ACCEPT", "IDENTITY_CONFIRMED_RXNORM_DAILYMED_SETID", ",".join(supporting)

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in fields})

def accepted_values(rows: list[dict]) -> str:
    vals=[]
    for r in rows:
        vals.append("(" + ",".join([
            sqlq(r.get("med_id")),
            sqlq(r.get("generic_name")),
            sqlq(r.get("v3_reason")),
            sqlq(r.get("v3_supporting_rxcui")),
            sqlq(r.get("setid")),
            sqlq(r.get("published_date")),
            sqlq(trunc(r.get("label_title") or "",1500)),
            sqlq(r.get("source_url")),
            sqlq(trunc(r.get("renal_section_titles") or "",1500)),
            sqlq(trunc(r.get("renal_text") or "",12000)),
        ]) + ")")
    return ",\n".join(vals)

def audit_values(rows: list[dict]) -> str:
    vals=[]
    for r in rows:
        vals.append("(" + ",".join([
            sqlq(r.get("med_id")),
            sqlq(r.get("generic_name")),
            sqlq(r.get("v3_decision")),
            sqlq(r.get("v3_reason")),
            sqlq(r.get("v3_supporting_rxcui")),
            sqlq(r.get("setid")),
            sqlq(r.get("source_url")),
        ]) + ")")
    return ",\n".join(vals)

def make_sql(all_rows:list[dict], accepted:list[dict]) -> str:
    av=audit_values(all_rows)
    ev=accepted_values(accepted)
    if not ev:
        ev="('','','','','','','','','','')"
    return f"""-- =====================================================================
-- MEDCALC · RENAL MASTER · SQL VALIDADO V3
-- IDENTITY GATE: RxNorm exact concept + DailyMed SETID mapping
-- Generado UTC: {datetime.now(timezone.utc).isoformat()}
--
-- NO EJECUTAR LOS SQL V1 O V2.
-- ESTE ES EL ÚNICO CANDIDATO DE CARGA DE LA CADENA MASTER.
--
-- Las nuevas filas siguen siendo CURRENT_REFERENCE y automatizable=FALSE.
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.renal_master_identity_v3 (
    medication_id uuid PRIMARY KEY REFERENCES public.medications(id) ON DELETE CASCADE,
    med_id text NOT NULL,
    generic_name text NOT NULL,
    v3_decision text NOT NULL,
    v3_reason text NOT NULL,
    supporting_rxcui text,
    setid text,
    source_url text,
    checked_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

WITH x(med_id,generic_name,v3_decision,v3_reason,supporting_rxcui,setid,source_url) AS (
VALUES
{av}
)
INSERT INTO public.renal_master_identity_v3
(medication_id,med_id,generic_name,v3_decision,v3_reason,supporting_rxcui,setid,source_url,checked_at,updated_at)
SELECT m.id,x.med_id,x.generic_name,x.v3_decision,x.v3_reason,x.supporting_rxcui,x.setid,x.source_url,NOW(),NOW()
FROM x
JOIN public.medications m ON m.med_id=x.med_id
ON CONFLICT (medication_id) DO UPDATE
SET generic_name=EXCLUDED.generic_name,
    v3_decision=EXCLUDED.v3_decision,
    v3_reason=EXCLUDED.v3_reason,
    supporting_rxcui=EXCLUDED.supporting_rxcui,
    setid=EXCLUDED.setid,
    source_url=EXCLUDED.source_url,
    checked_at=NOW(),
    updated_at=NOW();

WITH e(med_id,generic_name,v3_reason,supporting_rxcui,setid,published_date,label_title,source_url,renal_section_titles,renal_text) AS (
VALUES
{ev}
)
INSERT INTO public.sources
(id,title,organization,url,page,source_type,last_verified,created_at)
SELECT gen_random_uuid(),
       CONCAT('DailyMed — ',e.generic_name,' — renal labeling V3'),
       'U.S. National Library of Medicine · DailyMed',
       e.source_url,
       NULLIF(e.renal_section_titles,''),
       'RENAL_REFERENCIA',
       CURRENT_DATE,
       NOW()
FROM e
WHERE e.med_id<>''
  AND NOT EXISTS (SELECT 1 FROM public.sources s WHERE s.url=e.source_url);

WITH e(med_id,generic_name,v3_reason,supporting_rxcui,setid,published_date,label_title,source_url,renal_section_titles,renal_text) AS (
VALUES
{ev}
)
INSERT INTO public.renal_bibliography
(id,medication_id,drug_name_source,normal_dose,adjustment_method,recommendations,verified,status,source_id,created_at,updated_at)
SELECT gen_random_uuid(),m.id,m.generic_name,NULL,
       'CURRENT_DAILYMED_IDENTITY_V3',
       e.renal_text,TRUE,'PUBLISHED',src.id,NOW(),NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
JOIN LATERAL (
    SELECT s.id
    FROM public.sources s
    WHERE s.url=e.source_url
    ORDER BY s.created_at DESC NULLS LAST,s.id
    LIMIT 1
) src ON TRUE
WHERE e.med_id<>''
  AND NOT EXISTS (
      SELECT 1 FROM public.renal_bibliography rb
      WHERE rb.medication_id=m.id AND rb.source_id=src.id AND rb.status='PUBLISHED'
  );

WITH e(med_id,generic_name,v3_reason,supporting_rxcui,setid,published_date,label_title,source_url,renal_section_titles,renal_text) AS (
VALUES
{ev}
)
INSERT INTO public.renal_rules
(id,medication_id,indication,population,route,renal_metric,range_text,
 lower_limit,upper_limit,lower_inclusive,upper_inclusive,
 adjusted_regimen,rule_type,notes,automatizable,status,source_id,reviewed_at)
SELECT gen_random_uuid(),m.id,
       CONCAT(m.generic_name,' — ficha renal regulatoria actual · V3'),
       'Adulto / según ficha regulatoria',
       NULL,NULL,'Referencia clínica actual',
       NULL,NULL,FALSE,FALSE,
       e.renal_text,
       'PRECAUCION',
       CONCAT('DailyMed SPL ',e.setid,' · RxCUI ',e.supporting_rxcui,
              ' · identidad verificada V3. Referencia no automatizable.'),
       FALSE,'PUBLISHED',src.id,NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
JOIN LATERAL (
    SELECT s.id
    FROM public.sources s
    WHERE s.url=e.source_url
    ORDER BY s.created_at DESC NULLS LAST,s.id
    LIMIT 1
) src ON TRUE
WHERE e.med_id<>''
  AND NOT EXISTS (
      SELECT 1 FROM public.renal_rules rr
      WHERE rr.medication_id=m.id
        AND rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V3')
        AND rr.status IN ('PUBLISHED','PENDING_REVIEW')
  );

INSERT INTO public.renal_rule_validation
(rule_id,validation_class,evidence_note,validated_at,updated_at)
SELECT rr.id,'CURRENT_REFERENCE',
       'RENAL MASTER V3: identidad farmacológica verificada por RxNorm exacto + mapping SETID DailyMed; referencia no automatizable.',
       NOW(),NOW()
FROM public.renal_rules rr
JOIN public.medications m ON m.id=rr.medication_id
JOIN public.renal_master_identity_v3 v ON v.medication_id=m.id AND v.v3_decision='ACCEPT'
WHERE rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V3')
  AND rr.status='PUBLISHED'
ON CONFLICT (rule_id) DO UPDATE
SET validation_class='CURRENT_REFERENCE',
    evidence_note=EXCLUDED.evidence_note,
    validated_at=NOW(),
    updated_at=NOW();

-- Nunca degrada CURRENT_AUTO.
INSERT INTO public.renal_phase6_review
(medication_id,med_id,generic_name,original_priority,review_batch,
 phase6_status,disposition,source_id,decision_note,validated_at,updated_at)
SELECT m.id,m.med_id,m.generic_name,2,9,
       'CLOSED_CURRENT_REFERENCE','REFERENCIA_ACTUAL_VALIDADA',
       rr.source_id,rr.adjusted_regimen,NOW(),NOW()
FROM public.medications m
JOIN public.renal_master_identity_v3 v ON v.medication_id=m.id AND v.v3_decision='ACCEPT'
JOIN public.renal_rules rr
  ON rr.medication_id=m.id
 AND rr.indication=CONCAT(m.generic_name,' — ficha renal regulatoria actual · V3')
JOIN public.renal_rule_validation rv
  ON rv.rule_id=rr.id AND rv.validation_class='CURRENT_REFERENCE'
WHERE NOT EXISTS (
    SELECT 1 FROM public.renal_phase6_review p
    WHERE p.medication_id=m.id AND p.phase6_status='CLOSED_CURRENT_AUTO'
)
ON CONFLICT (medication_id) DO UPDATE
SET review_batch=CASE WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
                      THEN public.renal_phase6_review.review_batch ELSE 9 END,
    phase6_status=CASE WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
                       THEN public.renal_phase6_review.phase6_status ELSE 'CLOSED_CURRENT_REFERENCE' END,
    disposition=CASE WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
                     THEN public.renal_phase6_review.disposition ELSE 'REFERENCIA_ACTUAL_VALIDADA' END,
    source_id=CASE WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
                   THEN public.renal_phase6_review.source_id ELSE EXCLUDED.source_id END,
    decision_note=CASE WHEN public.renal_phase6_review.phase6_status='CLOSED_CURRENT_AUTO'
                       THEN public.renal_phase6_review.decision_note ELSE EXCLUDED.decision_note END,
    validated_at=NOW(),updated_at=NOW();

INSERT INTO public.medication_module_status(medication_id,renal_status)
SELECT medication_id,'PUBLISHED'
FROM public.renal_master_identity_v3
WHERE v3_decision='ACCEPT'
ON CONFLICT (medication_id) DO UPDATE SET renal_status='PUBLISHED';

SELECT
 COUNT(*) AS candidatos_v2_evaluados,
 COUNT(*) FILTER (WHERE v3_decision='ACCEPT') AS identidad_confirmada_v3,
 COUNT(*) FILTER (WHERE v3_decision='REJECT') AS identidad_incorrecta_rechazada,
 COUNT(*) FILTER (WHERE v3_decision='REVIEW') AS identidad_requiere_revision
FROM public.renal_master_identity_v3;
"""

def main():
    inp=Path("generated_renal_master_v2/renal_v2_accepted_current_reference.csv")
    out=Path("generated_renal_master_v3")
    out.mkdir(parents=True,exist_ok=True)

    if not inp.exists():
        raise SystemExit("Falta generated_renal_master_v2/renal_v2_accepted_current_reference.csv")

    with inp.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))

    if len(rows) != 205:
        raise SystemExit(f"SEGURIDAD: se esperaban 205 candidatos V2; hay {len(rows)}.")

    http=HTTP()
    for i,r in enumerate(rows,1):
        try:
            dec,reason,support=identity_check(http,r)
        except Exception as exc:
            dec,reason,support="REVIEW","IDENTITY_API_ERROR",""
            r["v3_error"]=repr(exc)
        r["v3_decision"]=dec
        r["v3_reason"]=reason
        r["v3_supporting_rxcui"]=support
        print(f"[{i:03d}/205] {r['med_id']} {r['generic_name']}: {dec} {reason}")

    fields=list(rows[0].keys())
    accept=[r for r in rows if r["v3_decision"]=="ACCEPT"]
    reject=[r for r in rows if r["v3_decision"]=="REJECT"]
    review=[r for r in rows if r["v3_decision"]=="REVIEW"]

    write_csv(out/"renal_v3_identity_full.csv",rows,fields)
    write_csv(out/"renal_v3_identity_accepted.csv",accept,fields)
    write_csv(out/"renal_v3_identity_rejected.csv",reject,fields)
    write_csv(out/"renal_v3_identity_review.csv",review,fields)

    reasons={}
    for r in rows:
        reasons[r["v3_reason"]]=reasons.get(r["v3_reason"],0)+1

    summary={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "v2_candidates":len(rows),
        "v3_identity_confirmed":len(accept),
        "v3_identity_rejected":len(reject),
        "v3_identity_review":len(review),
        "total_check":len(accept)+len(reject)+len(review),
        "reason_counts":dict(sorted(reasons.items())),
        "known_false_positive_guards":{
            "counterion_as_drug_rejected":True,
            "dailyMed_setid_must_map_to_exact_rxnorm_concept":True,
            "combination_collapse_rejected_or_review":True,
            "minimum_catalog_rxnorm_identity_score":90
        }
    }
    (out/"renal_v3_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"MEDCALC_RENAL_MAESTRO_VALIDADO_V3.sql").write_text(make_sql(rows,accept),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
