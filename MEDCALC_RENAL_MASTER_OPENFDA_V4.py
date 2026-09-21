#!/usr/bin/env python3
"""
MEDCALC · RENAL MASTER · V4 SECONDARY OFFICIAL SOURCE
=====================================================

Inputs already produced in the repository:
  generated_renal_master_v2/renal_v2_full_audit.csv
  generated_renal_master_v3/renal_v3_identity_accepted.csv

Goal:
  Process ALL unresolved MED-ID in one run using official FDA/openFDA labeling
  as a secondary regulatory source.

Resolved before V4:
  - V2 EXISTING_VERIFIED
  - V3 ACCEPT

Everything else is processed here.

Safety:
  - Exact RxNorm identity or very strict exact substance/generic identity.
  - Combination integrity enforced.
  - Overdosage and adverse-event text are excluded from renal dosing evidence.
  - Explicit renal dosing action required.
  - New rows remain CURRENT_REFERENCE and automatizable=FALSE.
  - No automatic numeric rules are created.
  - Existing CURRENT_AUTO/current verified content is preserved at SQL runtime.
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
from urllib.parse import quote

import requests
from rapidfuzz import fuzz

RXNORM = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_LABEL = "https://api.fda.gov/drug/label.json"

SALT_MAP = {
    "ACIDO":"ACID","CLORHIDRATO":"HYDROCHLORIDE","CLORURO":"CHLORIDE",
    "BROMURO":"BROMIDE","HIDROBROMURO":"HYDROBROMIDE","SULFATO":"SULFATE",
    "SODICO":"SODIUM","SODICA":"SODIUM","DISODICO":"DISODIUM","DISODICA":"DISODIUM",
    "POTASICO":"POTASSIUM","POTASICA":"POTASSIUM","CALCICO":"CALCIUM",
    "CALCICA":"CALCIUM","CALCIO":"CALCIUM","MAGNESIO":"MAGNESIUM",
    "FOSFATO":"PHOSPHATE","CITRATO":"CITRATE","TARTRATO":"TARTRATE",
    "ACETATO":"ACETATE","FUMARATO":"FUMARATE","MALEATO":"MALEATE",
    "SUCCINATO":"SUCCINATE","MESILATO":"MESYLATE","BESILATO":"BESYLATE",
    "LACTATO":"LACTATE","GLUCONATO":"GLUCONATE","OXIDO":"OXIDE",
    "HIDROXIDO":"HYDROXIDE",
}
SALT_WORDS = {
    "HYDROCHLORIDE","HCL","SODIUM","POTASSIUM","CALCIUM","MAGNESIUM",
    "MESYLATE","BESYLATE","MALEATE","FUMARATE","SUCCINATE","ACETATE",
    "TARTRATE","CITRATE","PHOSPHATE","SULFATE","BROMIDE","CHLORIDE",
    "DIHYDRATE","MONOHYDRATE","TRIHYDRATE","ANHYDROUS","PAMOATE",
    "XINAFOATE","LACTATE","GLUCONATE","DISODIUM",
    "CLORHIDRATO","SODICO","SODICA","POTASICO","POTASICA","CALCICO",
    "CALCICA","MESILATO","BESILATO","MALEATO","FUMARATO","SUCCINATO",
    "ACETATO","TARTRATO","CITRATO","FOSFATO","SULFATO","BROMURO",
    "CLORURO","PAMOATO","LACTATO","GLUCONATO","DE",
}
BAD_IDENTITIES = {
    "MALATE","CHLORIDE","SODIUM","POTASSIUM","CALCIUM","MAGNESIUM",
    "PHOSPHATE","SULFATE","CITRATE","ACETATE","FUMARATE","MALEATE",
    "TARTRATE","LACTATE","GLUCONATE","BROMIDE","WATER","DISODIUM"
}

RENAL = re.compile(
    r"\b(renal impairment|renal function|kidney function|kidney impairment|"
    r"creatinine clearance|\bcrcl\b|\begfr\b|\bgfr\b|"
    r"hemodialysis|haemodialysis|dialysis|end.stage renal|renal insufficien)\b",
    re.I,
)
NO_ADJUST = re.compile(
    r"\b(no (?:dose|dosage) adjustment(?: is)? (?:necessary|required|recommended)?|"
    r"does not require (?:dose|dosage) adjustment|"
    r"dose adjustment is not (?:necessary|required)|"
    r"no adjustment .* renal)\b", re.I
)
NOT_RECOMMENDED = re.compile(
    r"\b(not recommended|should not be used|avoid use|contraindicat)\b", re.I
)
THRESHOLD = re.compile(
    r"(?:crcl|creatinine clearance|egfr|gfr)"
    r"[^.;:\n]{0,120}"
    r"(?:<|>|≤|≥|less than|greater than|between|\d)", re.I
)
DOSE_ACTION = re.compile(
    r"\b(dose|dosage|dosing|administer|administration|reduce|reduction|"
    r"interval|every|once daily|twice daily|q\d+h|"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g))\b", re.I
)
DIALYSIS_DOSING = re.compile(
    r"(?:administer|give|dose|supplement)[^.;:\n]{0,180}"
    r"(?:hemodialysis|haemodialysis|dialysis)|"
    r"(?:hemodialysis|haemodialysis|dialysis)[^.;:\n]{0,180}"
    r"(?:administer|give|dose|supplement)", re.I
)
ADJUSTMENT = re.compile(
    r"\b(dose adjustment|dosage adjustment|adjust(?:ment)? of (?:the )?dose|"
    r"reduced dose|reduce the dose|increase .* interval|decrease .* dose)\b", re.I
)

# Label sections permitted as dosing evidence.
SAFE_FIELDS = [
    "renal_impairment",
    "dosage_and_administration",
    "use_in_specific_populations",
    "clinical_pharmacology",
    "pharmacokinetics",
    "warnings_and_cautions",
    "precautions",
    "warnings",
]
# Explicitly NOT read as renal dose evidence:
# overdosage, adverse_reactions, laboratory_tests, nonclinical_toxicology.

def norm(s:str)->str:
    s=unicodedata.normalize("NFKD",str(s or ""))
    s="".join(c for c in s if not unicodedata.combining(c))
    s=s.upper().replace("&"," / ")
    s=re.sub(r"[^A-Z0-9/+ ]+"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def translate_salts(s:str)->str:
    return " ".join(SALT_MAP.get(x,x) for x in norm(s).split())

def strip_salts(s:str)->str:
    toks=[x for x in norm(s).split() if x not in SALT_WORDS and x not in {"ACID","ACIDO"}]
    return " ".join(toks)

def is_combo(s:str)->bool:
    raw=str(s or "")
    return "/" in raw or " + " in raw or bool(re.search(r"\bY\b",norm(raw)))

def combo_parts(s:str)->list[str]:
    return [
        strip_salts(x)
        for x in re.split(r"\s*/\s*|\s+\+\s+|\s+\bY\b\s+",str(s or ""),flags=re.I)
        if strip_salts(x)
    ]

def similarity(a:str,b:str)->float:
    a=strip_salts(a); b=strip_salts(b)
    if not a or not b: return 0.0
    return float(max(fuzz.WRatio(a,b),fuzz.token_set_ratio(a,b)))

def sqlq(v:Optional[str])->str:
    if v is None: return "NULL"
    return "'" + str(v).replace("'","''") + "'"

def trunc(s:str,n:int)->str:
    s=str(s or "")
    return s if len(s)<=n else s[:n-1]+"…"

class HTTP:
    def __init__(self):
        self.s=requests.Session()
        self.s.headers["User-Agent"]="MEDCALC-RenalOpenFDA-V4/1.0"

    def get_json(self,url,params=None,tries=4):
        last=None
        for i in range(tries):
            try:
                r=self.s.get(url,params=params,timeout=35)
                if r.status_code==404:
                    time.sleep(0.28)
                    return None
                r.raise_for_status()
                time.sleep(0.28)
                return r.json()
            except Exception as exc:
                last=exc
                time.sleep(1.2*(i+1))
        raise last

def exact_rxcuis(http:HTTP,name:str)->list[str]:
    if not name: return []
    j=http.get_json(f"{RXNORM}/rxcui.json",params={"name":name,"search":2,"allsrc":0}) or {}
    return list(((j.get("idGroup") or {}).get("rxnormId")) or [])

def phrase(term:str)->str:
    term=str(term or "").replace('"','').strip()
    return f'"{term}"'

def label_search(http:HTTP,search:str)->list[dict]:
    j=http.get_json(OPENFDA_LABEL,params={"search":search,"limit":100}) or {}
    return list(j.get("results") or [])

def candidate_labels(http:HTTP,row:dict,rxcuis:list[str])->list[dict]:
    queries=[]
    for rxcui in rxcuis[:6]:
        queries.append(f'openfda.rxcui:{phrase(rxcui)}')

    names=[]
    for n in [
        row.get("rxnorm_name"),
        translate_salts(row.get("generic_name") or ""),
        translate_salts(row.get("normalized_name") or ""),
    ]:
        n=str(n or "").strip()
        if n and n not in names:
            names.append(n)

    for n in names[:4]:
        queries.append(f'openfda.generic_name:{phrase(n)}')
        queries.append(f'openfda.substance_name:{phrase(n)}')

    seen=set(); out=[]
    for q in queries:
        try:
            items=label_search(http,q)
        except Exception:
            continue
        for x in items:
            # SPL set id is the best dedup key; fallback to id/effective_time.
            of=x.get("openfda") or {}
            sid=((of.get("spl_set_id") or [""])[0] if isinstance(of.get("spl_set_id"),list) else of.get("spl_set_id")) or ""
            key=sid or str(x.get("id") or "") or json.dumps(of,sort_keys=True)[:500]
            if key in seen: continue
            seen.add(key); out.append(x)
    return out

def listify(v):
    if v is None: return []
    if isinstance(v,list): return [str(x) for x in v]
    return [str(v)]

def identity_ok(row:dict,rec:dict,rxcuis:list[str])->tuple[bool,float,str]:
    of=rec.get("openfda") or {}
    cand_rxcui=set(listify(of.get("rxcui")))
    generic=listify(of.get("generic_name"))
    substances=listify(of.get("substance_name"))
    names=generic+substances

    catalog=row.get("generic_name") or ""
    rxname=row.get("rxnorm_name") or ""
    core=strip_salts(rxname)
    if core in BAD_IDENTITIES:
        return False,0.0,"COUNTERION_IDENTITY"

    # Strongest identity: exact RxCUI crosswalk in openFDA.
    intersect=set(rxcuis).intersection(cand_rxcui)
    score=max([similarity(catalog,n) for n in names] or [0.0])
    if intersect:
        if is_combo(catalog):
            parts=combo_parts(catalog)
            joined=" ".join(names)
            for p in parts:
                if max(fuzz.partial_ratio(p,strip_salts(joined)),fuzz.token_set_ratio(p,strip_salts(joined)))<80:
                    return False,score,"RXCUI_MATCH_BUT_COMBINATION_COMPONENT_MISSING"
        return True,max(score,99.0),"OPENFDA_EXACT_RXCUI"

    # Fallback only when names are near-exact.
    if not names:
        return False,0.0,"OPENFDA_NO_IDENTITY_NAMES"

    best=max(score, similarity(catalog, rxname))
    if best < 96:
        return False,best,"NAME_IDENTITY_LT96"

    joined=" ".join(names)
    if is_combo(catalog):
        parts=combo_parts(catalog)
        for p in parts:
            if max(fuzz.partial_ratio(p,strip_salts(joined)),fuzz.token_set_ratio(p,strip_salts(joined)))<88:
                return False,best,"COMBINATION_COMPONENT_MISSING"
    else:
        # Reject a clearly combined label for a monocomponent.
        if " AND " in norm(joined) or "/" in joined:
            return False,best,"MONOCOMPONENT_TO_COMBINATION"

    return True,best,"OPENFDA_EXACT_NAME_FALLBACK"

def effective(rec:dict)->int:
    raw=str(rec.get("effective_time") or "0")
    try: return int(re.sub(r"\D","",raw)[:8] or 0)
    except: return 0

def renal_snippets(rec:dict)->tuple[str,str]:
    snippets=[]
    used_fields=[]
    for field in SAFE_FIELDS:
        vals=listify(rec.get(field))
        for val in vals:
            if not RENAL.search(val):
                continue
            # renal_impairment can be used in full; other broad sections only
            # contribute local windows around renal terms.
            if field=="renal_impairment":
                text=val
            else:
                wins=[]
                for m in RENAL.finditer(val):
                    lo=max(0,m.start()-420); hi=min(len(val),m.end()+1250)
                    wins.append(val[lo:hi])
                    if len(wins)>=5: break
                text=" | ".join(wins)
            if text:
                snippets.append(f"{field}: {text}")
                used_fields.append(field)
    return " | ".join(dict.fromkeys(used_fields)), trunc("\n\n".join(snippets),12000)

def clinical_action(text:str)->tuple[bool,str]:
    if not text:
        return False,"NO_RENAL_TEXT"

    if NO_ADJUST.search(text):
        return True,"OPENFDA_NO_ADJUSTMENT"

    if NOT_RECOMMENDED.search(text):
        if THRESHOLD.search(text) or re.search(r"(severe|end.stage|advanced)[^.;]{0,120}renal impairment",text,re.I):
            return True,"OPENFDA_NOT_RECOMMENDED"

    if THRESHOLD.search(text) and DOSE_ACTION.search(text):
        return True,"OPENFDA_RENAL_DOSING"

    if DIALYSIS_DOSING.search(text):
        return True,"OPENFDA_DIALYSIS_DOSING"

    for m in RENAL.finditer(text):
        lo=max(0,m.start()-220); hi=min(len(text),m.end()+600)
        if ADJUSTMENT.search(text[lo:hi]):
            return True,"OPENFDA_RENAL_DOSING_TEXT"

    return False,"RENAL_TEXT_WITHOUT_EXPLICIT_DOSING_ACTION"

def select_record(row:dict,cands:list[dict],rxcuis:list[str]):
    scored=[]
    for rec in cands:
        ok,score,reason=identity_ok(row,rec,rxcuis)
        if not ok: continue
        fields,text=renal_snippets(rec)
        action,action_reason=clinical_action(text)
        scored.append((1 if action else 0,score,effective(rec),rec,reason,fields,text,action_reason))
    if not scored:
        return None
    scored.sort(key=lambda x:(x[0],x[1],x[2]),reverse=True)
    return scored[0]

def source_url(rec:dict)->str:
    of=rec.get("openfda") or {}
    sids=listify(of.get("spl_set_id"))
    if sids and sids[0]:
        sid=sids[0]
        return f'https://api.fda.gov/drug/label.json?search=openfda.spl_set_id:%22{sid}%22&limit=1'
    apps=listify(of.get("application_number"))
    if apps and apps[0]:
        return f'https://api.fda.gov/drug/label.json?search=openfda.application_number:%22{quote(apps[0])}%22&limit=1'
    return "https://open.fda.gov/apis/drug/label/"

def process(http:HTTP,row:dict)->dict:
    out=dict(row)
    rxname=row.get("rxnorm_name") or ""
    try:
        rxcuis=exact_rxcuis(http,rxname) if rxname else []
    except Exception as exc:
        rxcuis=[]
        out["v4_rxnorm_error"]=repr(exc)

    if strip_salts(rxname) in BAD_IDENTITIES:
        rxcuis=[]

    try:
        cands=candidate_labels(http,row,rxcuis)
    except Exception as exc:
        out.update(v4_decision="REVIEW",v4_reason="OPENFDA_API_ERROR",v4_error=repr(exc))
        return out

    if not cands:
        out.update(v4_decision="REVIEW",v4_reason="NO_OPENFDA_LABEL")
        return out

    sel=select_record(row,cands,rxcuis)
    if not sel:
        out.update(v4_decision="REVIEW",v4_reason="NO_OPENFDA_IDENTITY_CONFIRMED_LABEL")
        return out

    action,score,eff,rec,identity_reason,fields,text,action_reason=sel
    of=rec.get("openfda") or {}
    sids=listify(of.get("spl_set_id"))
    out["v4_identity_reason"]=identity_reason
    out["v4_identity_score"]=f"{score:.1f}"
    out["v4_effective_time"]=str(rec.get("effective_time") or "")
    out["v4_spl_set_id"]=sids[0] if sids else ""
    out["v4_source_url"]=source_url(rec)
    out["v4_renal_fields"]=fields
    out["v4_renal_text"]=text
    out["v4_openfda_rxcui"]=",".join(listify(of.get("rxcui")))
    out["v4_openfda_generic_name"]=" | ".join(listify(of.get("generic_name")))
    out["v4_openfda_substance_name"]=" | ".join(listify(of.get("substance_name")))

    if action:
        out["v4_decision"]="ACCEPT"
        out["v4_reason"]=action_reason
    else:
        out["v4_decision"]="REVIEW"
        out["v4_reason"]=action_reason
    return out

def read_csv(path:Path)->list[dict]:
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path:Path,rows:list[dict],fields:list[str]):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in fields})

def sqlq(v:Optional[str])->str:
    if v is None:return "NULL"
    return "'" + str(v).replace("'","''") + "'"

def sql_rows_v3(rows:list[dict])->list[dict]:
    out=[]
    for r in rows:
        x={
            "med_id":r.get("med_id"),
            "generic_name":r.get("generic_name"),
            "source_kind":"DAILYMED_V3",
            "reason":r.get("v2_reason") or "V3_IDENTITY_CONFIRMED",
            "source_url":r.get("source_url"),
            "source_title":f"DailyMed — {r.get('generic_name')} — renal labeling final",
            "source_org":"U.S. National Library of Medicine · DailyMed",
            "source_locator":r.get("renal_section_titles"),
            "renal_text":r.get("renal_text"),
            "setid":r.get("setid"),
        }
        out.append(x)
    return out

def sql_rows_v4(rows:list[dict])->list[dict]:
    out=[]
    for r in rows:
        x={
            "med_id":r.get("med_id"),
            "generic_name":r.get("generic_name"),
            "source_kind":"OPENFDA_V4",
            "reason":r.get("v4_reason"),
            "source_url":r.get("v4_source_url"),
            "source_title":f"openFDA — {r.get('generic_name')} — renal labeling final",
            "source_org":"U.S. Food and Drug Administration · openFDA",
            "source_locator":r.get("v4_renal_fields"),
            "renal_text":r.get("v4_renal_text"),
            "setid":r.get("v4_spl_set_id"),
        }
        out.append(x)
    return out

def values(rows:list[dict])->str:
    vals=[]
    for r in rows:
        vals.append("(" + ",".join([
            sqlq(r.get("med_id")),sqlq(r.get("generic_name")),
            sqlq(r.get("source_kind")),sqlq(r.get("reason")),
            sqlq(r.get("source_url")),sqlq(r.get("source_title")),
            sqlq(r.get("source_org")),sqlq(trunc(r.get("source_locator") or "",1800)),
            sqlq(trunc(r.get("renal_text") or "",12000)),sqlq(r.get("setid")),
        ]) + ")")
    return ",\n".join(vals)

def make_final_sql(v3_rows:list[dict],v4_rows:list[dict])->str:
    combined=sql_rows_v3(v3_rows)+sql_rows_v4(v4_rows)
    vv=values(combined)
    return f"""-- =====================================================================
-- MEDCALC · RENAL MASTER · FINAL V4
-- Combina:
--   V3 DailyMed identity-confirmed
--   V4 openFDA secondary official source
--
-- NO EJECUTAR SQL V1/V2/V3.
-- Este SQL NO crea nuevas reglas automáticas.
-- Preserva reglas/fuentes renales ya validadas.
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.renal_master_final_v4 (
    medication_id uuid PRIMARY KEY REFERENCES public.medications(id) ON DELETE CASCADE,
    med_id text NOT NULL,
    generic_name text NOT NULL,
    source_kind text NOT NULL,
    decision_reason text NOT NULL,
    source_url text NOT NULL,
    setid text,
    loaded_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

WITH e(med_id,generic_name,source_kind,reason,source_url,source_title,source_org,source_locator,renal_text,setid) AS (
VALUES
{vv}
)
INSERT INTO public.renal_master_final_v4
(medication_id,med_id,generic_name,source_kind,decision_reason,source_url,setid,loaded_at,updated_at)
SELECT m.id,e.med_id,e.generic_name,e.source_kind,e.reason,e.source_url,e.setid,NOW(),NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
ON CONFLICT (medication_id) DO UPDATE
SET generic_name=EXCLUDED.generic_name,
    source_kind=EXCLUDED.source_kind,
    decision_reason=EXCLUDED.decision_reason,
    source_url=EXCLUDED.source_url,
    setid=EXCLUDED.setid,
    updated_at=NOW();

-- Fuentes
WITH e(med_id,generic_name,source_kind,reason,source_url,source_title,source_org,source_locator,renal_text,setid) AS (
VALUES
{vv}
)
INSERT INTO public.sources
(id,title,organization,url,page,source_type,last_verified,created_at)
SELECT gen_random_uuid(),e.source_title,e.source_org,e.source_url,
       NULLIF(e.source_locator,''),'RENAL_REFERENCIA',CURRENT_DATE,NOW()
FROM e
WHERE e.source_url<>''
AND NOT EXISTS (SELECT 1 FROM public.sources s WHERE s.url=e.source_url);

-- Bibliografía: se omite si el medicamento YA tiene bibliografía renal verificada.
WITH e(med_id,generic_name,source_kind,reason,source_url,source_title,source_org,source_locator,renal_text,setid) AS (
VALUES
{vv}
)
INSERT INTO public.renal_bibliography
(id,medication_id,drug_name_source,normal_dose,adjustment_method,
 recommendations,verified,status,source_id,created_at,updated_at)
SELECT gen_random_uuid(),m.id,m.generic_name,NULL,
       CONCAT('RENAL_MASTER_FINAL_V4:',e.source_kind,':',e.reason),
       e.renal_text,TRUE,'PUBLISHED',src.id,NOW(),NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
JOIN LATERAL (
  SELECT s.id FROM public.sources s
  WHERE s.url=e.source_url
  ORDER BY s.created_at DESC NULLS LAST,s.id LIMIT 1
) src ON TRUE
WHERE NOT EXISTS (
  SELECT 1 FROM public.renal_bibliography rb
  WHERE rb.medication_id=m.id
    AND rb.status='PUBLISHED'
    AND COALESCE(rb.verified,FALSE)=TRUE
);

-- Regla CURRENT_REFERENCE no automática.
-- Se omite si ya existe una regla renal actual validada para ese medicamento.
WITH e(med_id,generic_name,source_kind,reason,source_url,source_title,source_org,source_locator,renal_text,setid) AS (
VALUES
{vv}
)
INSERT INTO public.renal_rules
(id,medication_id,indication,population,route,renal_metric,range_text,
 lower_limit,upper_limit,lower_inclusive,upper_inclusive,
 adjusted_regimen,rule_type,notes,automatizable,status,source_id,reviewed_at)
SELECT gen_random_uuid(),m.id,
       CONCAT(m.generic_name,' — referencia renal regulatoria · MASTER FINAL V4'),
       'Adulto / según ficha regulatoria',
       NULL,NULL,'Referencia clínica actual',
       NULL,NULL,FALSE,FALSE,
       e.renal_text,
       CASE WHEN e.reason IN ('STRICT_NO_ADJUSTMENT','OPENFDA_NO_ADJUSTMENT')
            THEN 'NO_AJUSTE' ELSE 'PRECAUCION' END,
       CONCAT(e.source_kind,' · ',e.reason,' · SETID ',COALESCE(e.setid,''),
              '. Referencia no automatizable.'),
       FALSE,'PUBLISHED',src.id,NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
JOIN LATERAL (
  SELECT s.id FROM public.sources s
  WHERE s.url=e.source_url
  ORDER BY s.created_at DESC NULLS LAST,s.id LIMIT 1
) src ON TRUE
WHERE NOT EXISTS (
  SELECT 1
  FROM public.renal_rules rr
  JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
  WHERE rr.medication_id=m.id
    AND rr.status='PUBLISHED'
    AND rv.validation_class IN ('CURRENT_AUTO','CURRENT_REFERENCE','TDM')
);

INSERT INTO public.renal_rule_validation
(rule_id,validation_class,evidence_note,validated_at,updated_at)
SELECT rr.id,'CURRENT_REFERENCE',
       'RENAL MASTER FINAL V4: identidad farmacológica y acción renal regulatoria verificadas; referencia no automatizable.',
       NOW(),NOW()
FROM public.renal_rules rr
JOIN public.medications m ON m.id=rr.medication_id
JOIN public.renal_master_final_v4 f ON f.medication_id=m.id
WHERE rr.indication=CONCAT(m.generic_name,' — referencia renal regulatoria · MASTER FINAL V4')
  AND rr.status='PUBLISHED'
ON CONFLICT (rule_id) DO UPDATE
SET validation_class='CURRENT_REFERENCE',
    evidence_note=EXCLUDED.evidence_note,
    validated_at=NOW(),updated_at=NOW();

-- No degradar CURRENT_AUTO/TDM/referencia ya existente.
INSERT INTO public.renal_phase6_review
(medication_id,med_id,generic_name,original_priority,review_batch,
 phase6_status,disposition,source_id,decision_note,validated_at,updated_at)
SELECT m.id,m.med_id,m.generic_name,2,10,
       'CLOSED_CURRENT_REFERENCE','REFERENCIA_ACTUAL_VALIDADA',
       rr.source_id,rr.adjusted_regimen,NOW(),NOW()
FROM public.medications m
JOIN public.renal_master_final_v4 f ON f.medication_id=m.id
JOIN public.renal_rules rr
  ON rr.medication_id=m.id
 AND rr.indication=CONCAT(m.generic_name,' — referencia renal regulatoria · MASTER FINAL V4')
JOIN public.renal_rule_validation rv
  ON rv.rule_id=rr.id AND rv.validation_class='CURRENT_REFERENCE'
WHERE NOT EXISTS (
  SELECT 1 FROM public.renal_phase6_review p
  WHERE p.medication_id=m.id
    AND p.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
)
ON CONFLICT (medication_id) DO UPDATE
SET review_batch=CASE
    WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.review_batch ELSE 10 END,
    phase6_status=CASE
    WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.phase6_status ELSE 'CLOSED_CURRENT_REFERENCE' END,
    disposition=CASE
    WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.disposition ELSE 'REFERENCIA_ACTUAL_VALIDADA' END,
    source_id=CASE
    WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.source_id ELSE EXCLUDED.source_id END,
    decision_note=CASE
    WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.decision_note ELSE EXCLUDED.decision_note END,
    validated_at=NOW(),updated_at=NOW();

INSERT INTO public.medication_module_status(medication_id,renal_status)
SELECT medication_id,'PUBLISHED'
FROM public.renal_master_final_v4
ON CONFLICT (medication_id) DO UPDATE SET renal_status='PUBLISHED';

SELECT
 (SELECT COUNT(*) FROM public.renal_master_final_v4) AS nuevas_referencias_master_final,
 COUNT(*) FILTER (WHERE f.source_kind='DAILYMED_V3') AS desde_dailymed_v3,
 COUNT(*) FILTER (WHERE f.source_kind='OPENFDA_V4') AS desde_openfda_v4
FROM public.renal_master_final_v4 f;
"""

def main():
    v2_path=Path("generated_renal_master_v2/renal_v2_full_audit.csv")
    v3_path=Path("generated_renal_master_v3/renal_v3_identity_accepted.csv")
    out=Path("generated_renal_master_v4_openfda")
    out.mkdir(parents=True,exist_ok=True)

    if not v2_path.exists(): raise SystemExit("Falta V2 full audit.")
    if not v3_path.exists(): raise SystemExit("Falta V3 identity accepted.")

    v2=read_csv(v2_path)
    v3=read_csv(v3_path)

    if len(v2)!=1122: raise SystemExit(f"Se esperaban 1122 filas V2; hay {len(v2)}.")
    if len(v3)!=180: raise SystemExit(f"Se esperaban 180 aceptados V3; hay {len(v3)}.")

    existing={r["med_id"] for r in v2 if r.get("v2_decision")=="EXISTING_VERIFIED"}
    v3ids={r["med_id"] for r in v3}
    resolved=existing|v3ids
    unresolved=[r for r in v2 if r.get("med_id") not in resolved]

    if len(existing)!=63:
        raise SystemExit(f"Se esperaban 63 existentes; hay {len(existing)}.")
    if len(unresolved)!=879:
        raise SystemExit(f"Se esperaban 879 no resueltos para V4; hay {len(unresolved)}.")

    http=HTTP()
    results=[]
    for i,r in enumerate(unresolved,1):
        try:
            x=process(http,r)
        except Exception as exc:
            x=dict(r)
            x.update(v4_decision="REVIEW",v4_reason="UNHANDLED_V4_ERROR",v4_error=repr(exc))
        results.append(x)
        print(f"[{i:03d}/879] {x['med_id']} {x['generic_name']}: {x.get('v4_decision')} {x.get('v4_reason')}")

        if i%25==0:
            fields=list(dict.fromkeys(k for row in results for k in row.keys()))
            write_csv(out/"renal_v4_partial.csv",results,fields)

    fields=list(dict.fromkeys(k for row in results for k in row.keys()))
    accepted=[r for r in results if r.get("v4_decision")=="ACCEPT"]
    pending=[r for r in results if r.get("v4_decision")!="ACCEPT"]

    write_csv(out/"renal_v4_full_audit.csv",results,fields)
    write_csv(out/"renal_v4_openfda_accepted.csv",accepted,fields)
    write_csv(out/"renal_v4_still_unresolved.csv",pending,fields)

    reasons={}
    for r in results:
        reasons[r.get("v4_reason") or "UNKNOWN"]=reasons.get(r.get("v4_reason") or "UNKNOWN",0)+1

    final_resolved=63+180+len(accepted)
    summary={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "catalog_rows":1122,
        "existing_verified":63,
        "dailyMed_v3_identity_confirmed":180,
        "v4_unresolved_input":879,
        "openFDA_v4_accepted":len(accepted),
        "still_unresolved_after_v4":len(pending),
        "total_structurally_resolved_after_v4":final_resolved,
        "total_check":final_resolved+len(pending),
        "reason_counts":dict(sorted(reasons.items())),
        "safety":{
            "official_secondary_source":"FDA/openFDA drug labeling",
            "exact_rxcui_preferred":True,
            "strict_name_fallback_min_score":96,
            "combination_integrity_required":True,
            "overdosage_excluded":True,
            "adverse_reactions_excluded":True,
            "explicit_renal_dosing_action_required":True,
            "new_automatic_rules_created":False
        }
    }
    (out/"renal_v4_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    (out/"MEDCALC_RENAL_MASTER_FINAL_V4.sql").write_text(make_final_sql(v3,accepted),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
