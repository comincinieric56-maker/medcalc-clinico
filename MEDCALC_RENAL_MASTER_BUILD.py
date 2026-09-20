#!/usr/bin/env python3
"""
MEDCALC · RENAL MASTER · FULL CATALOG
====================================
Processes the entire MED-ID catalog in one run.

Data sources
------------
- RxNorm / RxNav API: terminology reconciliation.
- DailyMed API v2: current Structured Product Labels.

Safety model
------------
- Existing MEDCALC automatic rules are never deleted or overwritten.
- A fuzzy lexical match alone is never enough to publish a renal rule.
- Only HIGH-confidence DailyMed matches may create a new CURRENT_REFERENCE.
- Generated rules are deliberately NON-AUTOMATIC. Existing validated
  CURRENT_AUTO rules remain the calculation layer.
- "No renal section" is NEVER interpreted as "no dose adjustment".
- Unmatched or ambiguous records are stored in the audit table only.
- Combination products require all catalog ingredients to be represented
  in the matched title/canonical term before HIGH confidence is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

import requests
from lxml import etree
from rapidfuzz import fuzz

RXNORM = "https://rxnav.nlm.nih.gov/REST"
DAILYMED = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

SALT_TRANSLATIONS = {
    "ACIDO": "ACID",
    "CLORHIDRATO": "HYDROCHLORIDE",
    "CLORHIDRATOS": "HYDROCHLORIDE",
    "CLORURO": "CHLORIDE",
    "BROMURO": "BROMIDE",
    "HIDROBROMURO": "HYDROBROMIDE",
    "SULFATO": "SULFATE",
    "SODICO": "SODIUM",
    "SODICA": "SODIUM",
    "DISODICO": "DISODIUM",
    "DISODICA": "DISODIUM",
    "POTASICO": "POTASSIUM",
    "POTASICA": "POTASSIUM",
    "CALCICO": "CALCIUM",
    "CALCICA": "CALCIUM",
    "CALCIO": "CALCIUM",
    "MAGNESIO": "MAGNESIUM",
    "MAGNESICO": "MAGNESIUM",
    "FOSFATO": "PHOSPHATE",
    "CITRATO": "CITRATE",
    "TARTRATO": "TARTRATE",
    "ACETATO": "ACETATE",
    "FUMARATO": "FUMARATE",
    "MALEATO": "MALEATE",
    "SUCCINATO": "SUCCINATE",
    "MESILATO": "MESYLATE",
    "BESILATO": "BESYLATE",
    "LACTATO": "LACTATE",
    "GLUCONATO": "GLUCONATE",
    "OXIDO": "OXIDE",
    "HIDROXIDO": "HYDROXIDE",
    "HIERRO": "IRON",
    "ALUMINIO": "ALUMINUM",
}

STOPWORDS = {
    "DE","DEL","LA","EL","LOS","LAS","Y","CON","PARA","COMO",
    "MG","ML","IV","PO","ORAL","INYECTABLE","INHALADO","TOPICO","TOPICA",
    "TABLETA","TABLETAS","CAPSULA","CAPSULAS","SOLUCION","SUSPENSION",
    "LIBERACION","PROLONGADA","EXTENDIDA","INMEDIATA",
}

RENAL_TITLE_RE = re.compile(
    r"\b(renal|kidney|hemodialysis|haemodialysis|dialysis|dialytic|nephro)\b",
    re.I,
)
RENAL_TEXT_RE = re.compile(
    r"\b(renal impairment|renal function|kidney function|creatinine clearance|"
    r"\bcrcl\b|egfr|gfr|hemodialysis|haemodialysis|dialysis|end.stage renal|"
    r"renal failure|renal insufficien)\b",
    re.I,
)
DOSING_HINT_RE = re.compile(
    r"\b(dose|dosage|dosing|adjust|reduce|reduction|interval|every\s+\d+|"
    r"once daily|twice daily|not recommended|contraindicat|no dose adjustment|"
    r"no dosage adjustment|administer after.*dialysis)\b",
    re.I,
)
NO_ADJUST_RE = re.compile(
    r"\b(no (?:dose|dosage) adjustment|does not require (?:dose|dosage) adjustment|"
    r"no adjustment .* renal|dose adjustment is not (?:necessary|required))\b",
    re.I,
)
NOT_RECOMMENDED_RE = re.compile(
    r"\b(not recommended|should not be used|avoid use|contraindicat)\b", re.I
)

@dataclass
class Result:
    med_id: str
    generic_name: str
    normalized_name: str
    previous_status: str
    previous_rules: int
    previous_auto: int
    previous_verified_biblio: int
    harvest_status: str = ""
    match_confidence: str = ""
    matched_term: str = ""
    rxnorm_name: str = ""
    rxnorm_score: float = 0.0
    setid: str = ""
    spl_version: str = ""
    published_date: str = ""
    label_title: str = ""
    source_url: str = ""
    renal_section_titles: str = ""
    renal_text: str = ""
    rule_type: str = ""
    query_trace: str = ""
    error: str = ""

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = s.replace("&", " / ")
    s = re.sub(r"[(),;:+\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def translate_salts(s: str) -> str:
    words = norm(s).split()
    return " ".join(SALT_TRANSLATIONS.get(w, w) for w in words)

def significant_tokens(s: str) -> list[str]:
    out = []
    for w in norm(s).split():
        if w in STOPWORDS:
            continue
        if len(w) <= 2:
            continue
        if w in SALT_TRANSLATIONS:
            continue
        out.append(w)
    return out

def ingredient_parts(s: str) -> list[str]:
    s = norm(s)
    parts = re.split(r"\s*/\s*|\s+\+\s+", s)
    return [p.strip() for p in parts if p.strip()]

def safe_sql(s: Optional[str]) -> str:
    if s is None:
        return "NULL"
    return "'" + str(s).replace("'", "''") + "'"

def truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n-1] + "…"

class HTTP:
    def __init__(self, sleep: float = 0.2):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": "MEDCALC-RenalMaster/1.0 (clinical evidence harvester)"
        })
        self.sleep = sleep

    def get_json(self, url: str, params=None, attempts=4):
        last = None
        for i in range(attempts):
            try:
                r = self.s.get(url, params=params, timeout=35)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                time.sleep(self.sleep)
                return r.json()
            except Exception as e:
                last = e
                time.sleep((i + 1) * 1.2)
        raise last

    def get_text(self, url: str, params=None, attempts=4):
        last = None
        for i in range(attempts):
            try:
                r = self.s.get(url, params=params, timeout=45)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                time.sleep(self.sleep)
                return r.text
            except Exception as e:
                last = e
                time.sleep((i + 1) * 1.2)
        raise last

def rxnorm_candidates(http: HTTP, name: str) -> list[dict]:
    candidates = []
    seen = set()

    # Exact/normalized first.
    j = http.get_json(
        f"{RXNORM}/rxcui.json",
        params={"name": name, "search": 2, "allsrc": 0},
    ) or {}
    ids = (((j.get("idGroup") or {}).get("rxnormId")) or [])
    for rxcui in ids[:5]:
        if rxcui in seen:
            continue
        seen.add(rxcui)
        p = http.get_json(f"{RXNORM}/rxcui/{rxcui}/properties.json") or {}
        props = p.get("properties") or {}
        candidates.append({
            "rxcui": rxcui,
            "name": props.get("name") or "",
            "score": 100.0,
            "method": "RXNORM_EXACT_OR_NORMALIZED",
        })

    if candidates:
        return candidates

    # Approximate fallback.
    j = http.get_json(
        f"{RXNORM}/approximateTerm.json",
        params={"term": name, "maxEntries": 8, "option": 1},
    ) or {}
    arr = (((j.get("approximateGroup") or {}).get("candidate")) or [])
    for c in arr:
        rxcui = str(c.get("rxcui") or "")
        if not rxcui or rxcui in seen:
            continue
        seen.add(rxcui)
        candidates.append({
            "rxcui": rxcui,
            "name": c.get("name") or "",
            "score": float(c.get("score") or 0),
            "method": "RXNORM_APPROX",
        })
    return candidates[:8]

def lexical_similarity(catalog: str, candidate: str) -> float:
    a = translate_salts(catalog)
    b = norm(candidate)
    return float(max(
        fuzz.token_set_ratio(a, b),
        fuzz.WRatio(a, b),
        fuzz.partial_ratio(a, b),
    ))

def combo_coverage(catalog: str, candidate: str) -> bool:
    parts = ingredient_parts(catalog)
    if len(parts) <= 1:
        return True
    b = norm(candidate)
    for p in parts:
        toks = significant_tokens(p)
        if not toks:
            continue
        best = max((fuzz.partial_ratio(t, b) for t in toks), default=0)
        if best < 72:
            return False
    return True

def dailymed_spls(http: HTTP, term: str, rxcui: str = "") -> list[dict]:
    queries = []
    # Prefer generic-name query.
    queries.append({"drug_name": term, "name_type": "g", "pagesize": 100, "page": 1})
    if rxcui:
        queries.append({"rxcui": rxcui, "pagesize": 100, "page": 1})

    all_items = []
    seen = set()
    for params in queries:
        j = http.get_json(f"{DAILYMED}/spls.json", params=params) or {}
        for x in j.get("data") or []:
            sid = str(x.get("setid") or "")
            if not sid or sid in seen:
                continue
            seen.add(sid)
            all_items.append(x)
    return all_items

def parse_date(s: str):
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    return datetime(1900, 1, 1)

def choose_spl(items: list[dict], catalog_name: str, matched_term: str) -> Optional[dict]:
    if not items:
        return None
    scored = []
    for x in items:
        title = x.get("title") or ""
        sim = max(
            lexical_similarity(catalog_name, title),
            lexical_similarity(matched_term, title),
        )
        if not combo_coverage(catalog_name, title):
            sim -= 25
        scored.append((sim, parse_date(x.get("published_date") or ""), x))
    scored.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return scored[0][2]

def clean_text(node) -> str:
    txt = " ".join(t.strip() for t in node.itertext() if str(t).strip())
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()

def extract_renal_sections(xml_text: str) -> tuple[list[str], str]:
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_text.encode("utf-8", "ignore"), parser=parser)

    candidates = []
    for sec in root.xpath('//*[local-name()="section"]'):
        titles = sec.xpath('./*[local-name()="title"]')
        title = clean_text(titles[0]) if titles else ""
        body_nodes = sec.xpath('./*[local-name()="text"]')
        body = clean_text(body_nodes[0]) if body_nodes else ""
        if not body:
            continue

        score = 0
        if RENAL_TITLE_RE.search(title):
            score += 100
        if RENAL_TEXT_RE.search(body):
            score += 30
        if re.search(r"\b(dosage|administration|specific populations|clinical pharmacology)\b", title, re.I):
            score += 15
        if DOSING_HINT_RE.search(body):
            score += 20

        # We do not want a generic adverse event section merely mentioning AKI.
        if score >= 50:
            candidates.append((score, title, body))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    used = set()
    for score, title, body in candidates:
        fingerprint = re.sub(r"\W+", "", (title + body[:300]).lower())
        if fingerprint in used:
            continue
        used.add(fingerprint)
        selected.append((title, body))
        if len(selected) >= 4:
            break

    if not selected:
        # Last-resort targeted snippets from the full document.
        full = clean_text(root)
        snippets = []
        for m in RENAL_TEXT_RE.finditer(full):
            lo = max(0, m.start() - 450)
            hi = min(len(full), m.end() + 1100)
            snip = full[lo:hi]
            if DOSING_HINT_RE.search(snip):
                snippets.append(snip)
            if len(snippets) >= 3:
                break
        if snippets:
            return ["Targeted renal text from SPL"], truncate(" | ".join(snippets), 8000)
        return [], ""

    titles = [x[0] or "Renal section" for x in selected]
    text = "\n\n".join(
        f"{(t or 'Renal section')}: {b}" for t, b in selected
    )
    return titles, truncate(text, 12000)

def classify_rule_type(text: str) -> str:
    if NO_ADJUST_RE.search(text):
        return "NO_AJUSTE"
    if NOT_RECOMMENDED_RE.search(text):
        return "PRECAUCION"
    if re.search(r"\b(hemodialysis|haemodialysis|dialysis)\b", text, re.I):
        return "PRECAUCION"
    return "PRECAUCION"

def confidence(catalog: str, rxname: str, rxscore: float, label_title: str) -> str:
    sim_rx = lexical_similarity(catalog, rxname)
    sim_label = lexical_similarity(catalog, label_title)
    combo_ok = combo_coverage(catalog, rxname + " " + label_title)

    # High confidence requires both lexical consistency and combination integrity.
    if combo_ok and max(sim_rx, sim_label) >= 88 and (rxscore >= 80 or sim_rx >= 92):
        return "HIGH"
    if combo_ok and max(sim_rx, sim_label) >= 75:
        return "MEDIUM"
    return "LOW"

def process_row(http: HTTP, row: dict) -> Result:
    r = Result(
        med_id=row["med_id"],
        generic_name=row["generic_name"],
        normalized_name=row.get("normalized_name") or row["generic_name"],
        previous_status=row.get("renal_audit_status") or "",
        previous_rules=int(float(row.get("renal_rules_published") or 0)),
        previous_auto=int(float(row.get("renal_rules_automatic") or 0)),
        previous_verified_biblio=int(float(row.get("renal_biblio_verified") or 0)),
    )

    # Already verified drugs are still included in the final audit, but API calls
    # are unnecessary unless they have no verified current bibliography.
    if r.previous_verified_biblio > 0 and r.previous_status in {
        "HAS_AUTOMATIC_RULE", "HAS_PUBLISHED_NONAUTO_RULE"
    }:
        r.harvest_status = "ALREADY_VERIFIED_IN_MEDCALC"
        r.match_confidence = "EXISTING"
        return r

    queries = []
    for q in [
        r.generic_name,
        r.normalized_name,
        translate_salts(r.generic_name),
        translate_salts(r.normalized_name),
    ]:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in queries:
            queries.append(q)

    rx_all = []
    for q in queries[:4]:
        try:
            for c in rxnorm_candidates(http, q):
                c["query"] = q
                c["lexical"] = lexical_similarity(r.generic_name, c.get("name") or "")
                rx_all.append(c)
        except Exception as e:
            r.query_trace += f"RXERR[{q}]={e};"

    # Deduplicate and rank.
    uniq = {}
    for c in rx_all:
        key = c["rxcui"]
        old = uniq.get(key)
        if old is None or (c["lexical"], c["score"]) > (old["lexical"], old["score"]):
            uniq[key] = c
    rx_all = sorted(
        uniq.values(),
        key=lambda c: (combo_coverage(r.generic_name, c["name"]), c["lexical"], c["score"]),
        reverse=True,
    )[:8]

    if not rx_all:
        r.harvest_status = "NO_RXNORM_MATCH"
        r.match_confidence = "NONE"
        r.query_trace += "|".join(queries)
        return r

    best_bundle = None
    for c in rx_all[:5]:
        term = c.get("name") or c.get("query") or ""
        try:
            items = dailymed_spls(http, term, c.get("rxcui") or "")
        except Exception as e:
            r.query_trace += f"DMERR[{term}]={e};"
            continue
        spl = choose_spl(items, r.generic_name, term)
        if not spl:
            continue

        label_sim = lexical_similarity(r.generic_name, spl.get("title") or "")
        bundle_score = (
            1 if combo_coverage(r.generic_name, term + " " + (spl.get("title") or "")) else 0,
            max(c["lexical"], label_sim),
            c["score"],
            parse_date(spl.get("published_date") or ""),
        )
        if best_bundle is None or bundle_score > best_bundle[0]:
            best_bundle = (bundle_score, c, spl)

    if not best_bundle:
        c = rx_all[0]
        r.harvest_status = "RXNORM_MATCH_NO_DAILYMED_SPL"
        r.matched_term = c.get("query") or ""
        r.rxnorm_name = c.get("name") or ""
        r.rxnorm_score = c.get("score") or 0
        r.match_confidence = "MEDIUM" if c.get("lexical", 0) >= 80 else "LOW"
        return r

    _, c, spl = best_bundle
    r.matched_term = c.get("query") or ""
    r.rxnorm_name = c.get("name") or ""
    r.rxnorm_score = float(c.get("score") or 0)
    r.setid = spl.get("setid") or ""
    r.spl_version = str(spl.get("spl_version") or "")
    r.published_date = spl.get("published_date") or ""
    r.label_title = spl.get("title") or ""
    r.source_url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={r.setid}"
    r.match_confidence = confidence(
        r.generic_name, r.rxnorm_name, r.rxnorm_score, r.label_title
    )

    if r.match_confidence != "HIGH":
        r.harvest_status = "AMBIGUOUS_DAILYMED_MATCH"
        return r

    try:
        xml = http.get_text(f"{DAILYMED}/spls/{r.setid}.xml")
        if not xml:
            r.harvest_status = "DAILYMED_XML_NOT_FOUND"
            return r
        titles, renal_text = extract_renal_sections(xml)
    except Exception as e:
        r.harvest_status = "DAILYMED_XML_PARSE_ERROR"
        r.error = str(e)
        return r

    r.renal_section_titles = " | ".join(titles)
    r.renal_text = renal_text

    if not renal_text:
        r.harvest_status = "HIGH_MATCH_NO_EXPLICIT_RENAL_DOSING_TEXT"
        return r

    r.rule_type = classify_rule_type(renal_text)
    r.harvest_status = "HIGH_MATCH_RENAL_EVIDENCE"
    return r

def write_csv(path: Path, rows: list[Result]):
    fields = list(asdict(rows[0]).keys()) if rows else list(Result.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in rows:
            w.writerow(asdict(x))

def sql_values(rows: Iterable[Result]) -> str:
    vals = []
    for r in rows:
        vals.append(
            "(" + ",".join([
                safe_sql(r.med_id),
                safe_sql(r.generic_name),
                safe_sql(r.harvest_status),
                safe_sql(r.match_confidence),
                safe_sql(r.matched_term),
                safe_sql(r.rxnorm_name),
                str(float(r.rxnorm_score)),
                safe_sql(r.setid),
                safe_sql(r.spl_version),
                safe_sql(r.published_date),
                safe_sql(truncate(r.label_title, 1500)),
                safe_sql(r.source_url),
                safe_sql(truncate(r.renal_section_titles, 1500)),
                safe_sql(truncate(r.renal_text, 12000)),
                safe_sql(r.rule_type),
            ]) + ")"
        )
    return ",\n".join(vals)

def generate_sql(rows: list[Result]) -> str:
    all_values = sql_values(rows)
    evidence = [r for r in rows if r.harvest_status == "HIGH_MATCH_RENAL_EVIDENCE"]
    evidence_values = sql_values(evidence)

    return f"""-- =====================================================================
-- MEDCALC · RENAL MASTER · FULL CATALOG
-- Generated automatically from RxNorm + DailyMed.
-- Generated UTC: {datetime.now(timezone.utc).isoformat()}
--
-- SAFETY:
--   * Does not delete existing renal rules.
--   * New clinical rows are CURRENT_REFERENCE and automatizable=FALSE.
--   * Existing CURRENT_AUTO rules remain untouched.
--   * Ambiguous matches and labels without explicit renal dosing text
--     are recorded only in renal_master_harvest.
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.renal_master_harvest (
    medication_id uuid PRIMARY KEY REFERENCES public.medications(id) ON DELETE CASCADE,
    med_id text NOT NULL,
    generic_name text NOT NULL,
    harvest_status text NOT NULL,
    match_confidence text,
    matched_term text,
    rxnorm_name text,
    rxnorm_score numeric,
    setid text,
    spl_version text,
    published_date text,
    label_title text,
    source_url text,
    renal_section_titles text,
    renal_text text,
    rule_type text,
    harvested_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

WITH raw(
 med_id,generic_name,harvest_status,match_confidence,matched_term,rxnorm_name,
 rxnorm_score,setid,spl_version,published_date,label_title,source_url,
 renal_section_titles,renal_text,rule_type
) AS (
VALUES
{all_values}
)
INSERT INTO public.renal_master_harvest(
 medication_id,med_id,generic_name,harvest_status,match_confidence,matched_term,
 rxnorm_name,rxnorm_score,setid,spl_version,published_date,label_title,source_url,
 renal_section_titles,renal_text,rule_type,harvested_at,updated_at
)
SELECT
 m.id,r.med_id,r.generic_name,r.harvest_status,r.match_confidence,r.matched_term,
 r.rxnorm_name,r.rxnorm_score,r.setid,r.spl_version,r.published_date,r.label_title,
 r.source_url,r.renal_section_titles,r.renal_text,r.rule_type,NOW(),NOW()
FROM raw r
JOIN public.medications m ON m.med_id=r.med_id
ON CONFLICT (medication_id) DO UPDATE
SET generic_name=EXCLUDED.generic_name,
    harvest_status=EXCLUDED.harvest_status,
    match_confidence=EXCLUDED.match_confidence,
    matched_term=EXCLUDED.matched_term,
    rxnorm_name=EXCLUDED.rxnorm_name,
    rxnorm_score=EXCLUDED.rxnorm_score,
    setid=EXCLUDED.setid,
    spl_version=EXCLUDED.spl_version,
    published_date=EXCLUDED.published_date,
    label_title=EXCLUDED.label_title,
    source_url=EXCLUDED.source_url,
    renal_section_titles=EXCLUDED.renal_section_titles,
    renal_text=EXCLUDED.renal_text,
    rule_type=EXCLUDED.rule_type,
    updated_at=NOW();

-- Sources for HIGH-confidence renal evidence only.
WITH raw(
 med_id,generic_name,harvest_status,match_confidence,matched_term,rxnorm_name,
 rxnorm_score,setid,spl_version,published_date,label_title,source_url,
 renal_section_titles,renal_text,rule_type
) AS (
VALUES
{evidence_values if evidence_values else "('','','','','','',0,'','','','','','','','','')"}
)
INSERT INTO public.sources
(id,title,organization,url,page,source_type,last_verified,created_at)
SELECT
 gen_random_uuid(),
 CONCAT('DailyMed — ', r.generic_name, ' — current renal labeling'),
 'U.S. National Library of Medicine · DailyMed',
 r.source_url,
 NULLIF(r.renal_section_titles,''),
 'RENAL_REFERENCIA',
 CURRENT_DATE,
 NOW()
FROM raw r
WHERE r.harvest_status='HIGH_MATCH_RENAL_EVIDENCE'
  AND r.source_url<>''
  AND NOT EXISTS (
    SELECT 1 FROM public.sources s WHERE s.url=r.source_url
  );

-- Verified renal bibliography from the current label.
WITH raw(
 med_id,generic_name,harvest_status,match_confidence,matched_term,rxnorm_name,
 rxnorm_score,setid,spl_version,published_date,label_title,source_url,
 renal_section_titles,renal_text,rule_type
) AS (
VALUES
{evidence_values if evidence_values else "('','','','','','',0,'','','','','','','','','')"}
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
 'CURRENT_DAILYMED_LABEL_REFERENCE',
 r.renal_text,
 TRUE,
 'PUBLISHED',
 s.id,
 NOW(),
 NOW()
FROM raw r
JOIN public.medications m ON m.med_id=r.med_id
JOIN public.sources s ON s.url=r.source_url
WHERE r.harvest_status='HIGH_MATCH_RENAL_EVIDENCE'
  AND NOT EXISTS (
    SELECT 1
    FROM public.renal_bibliography rb
    WHERE rb.medication_id=m.id
      AND rb.source_id=s.id
      AND rb.status='PUBLISHED'
  );

-- Current reference rule: exact regulatory renal text, deliberately not automatic.
WITH raw(
 med_id,generic_name,harvest_status,match_confidence,matched_term,rxnorm_name,
 rxnorm_score,setid,spl_version,published_date,label_title,source_url,
 renal_section_titles,renal_text,rule_type
) AS (
VALUES
{evidence_values if evidence_values else "('','','','','','',0,'','','','','','','','','')"}
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
 CONCAT(m.generic_name,' — ficha renal regulatoria actual'),
 'Adulto / según ficha regulatoria',
 NULL,
 NULL,
 'Referencia clínica actual',
 NULL,NULL,FALSE,FALSE,
 r.renal_text,
 CASE WHEN r.rule_type='NO_AJUSTE' THEN 'NO_AJUSTE' ELSE 'PRECAUCION' END,
 CONCAT('DailyMed SPL ',r.setid,' · ',COALESCE(r.published_date,''),'. ',
        'Referencia no automatizable; no sustituye una pauta CURRENT_AUTO ya validada.'),
 FALSE,
 'PUBLISHED',
 s.id,
 NOW()
FROM raw r
JOIN public.medications m ON m.med_id=r.med_id
JOIN public.sources s ON s.url=r.source_url
WHERE r.harvest_status='HIGH_MATCH_RENAL_EVIDENCE'
  AND NOT EXISTS (
    SELECT 1
    FROM public.renal_rules rr
    WHERE rr.medication_id=m.id
      AND COALESCE(rr.range_text,'')='Referencia clínica actual'
      AND rr.source_id=s.id
      AND rr.status IN ('PUBLISHED','PENDING_REVIEW')
  );

-- Validation class for generated references.
INSERT INTO public.renal_rule_validation
(rule_id,validation_class,evidence_note,validated_at,updated_at)
SELECT
 rr.id,
 'CURRENT_REFERENCE',
 'RENAL MASTER: referencia regulatoria actual obtenida automáticamente de DailyMed con coincidencia farmacológica HIGH. No automatizada.',
 NOW(),NOW()
FROM public.renal_rules rr
JOIN public.sources s ON s.id=rr.source_id
WHERE rr.status='PUBLISHED'
  AND COALESCE(rr.automatizable,FALSE)=FALSE
  AND s.source_type='RENAL_REFERENCIA'
  AND s.title LIKE 'DailyMed — % — current renal labeling'
ON CONFLICT (rule_id) DO UPDATE
SET validation_class='CURRENT_REFERENCE',
    evidence_note=EXCLUDED.evidence_note,
    validated_at=NOW(),
    updated_at=NOW();

-- Do NOT downgrade existing CURRENT_AUTO closures.
INSERT INTO public.renal_phase6_review
(
 medication_id,med_id,generic_name,original_priority,review_batch,
 phase6_status,disposition,source_id,decision_note,validated_at,updated_at
)
SELECT
 m.id,m.med_id,m.generic_name,2,99,
 'CLOSED_CURRENT_REFERENCE',
 'REFERENCIA_ACTUAL_VALIDADA',
 rr.source_id,
 rr.adjusted_regimen,
 NOW(),NOW()
FROM public.medications m
JOIN public.renal_rules rr ON rr.medication_id=m.id
JOIN public.renal_rule_validation rv
  ON rv.rule_id=rr.id AND rv.validation_class='CURRENT_REFERENCE'
JOIN public.renal_master_harvest h
  ON h.medication_id=m.id
WHERE h.harvest_status='HIGH_MATCH_RENAL_EVIDENCE'
  AND NOT EXISTS (
    SELECT 1
    FROM public.renal_phase6_review p
    WHERE p.medication_id=m.id
      AND p.phase6_status='CLOSED_CURRENT_AUTO'
  )
ON CONFLICT (medication_id) DO UPDATE
SET review_batch=99,
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
FROM public.renal_master_harvest
WHERE harvest_status='HIGH_MATCH_RENAL_EVIDENCE'
ON CONFLICT (medication_id) DO UPDATE
SET renal_status='PUBLISHED';

-- Final summary.
SELECT
 COUNT(*) AS catalogo_procesado,
 COUNT(*) FILTER (WHERE harvest_status='ALREADY_VERIFIED_IN_MEDCALC') AS ya_verificados,
 COUNT(*) FILTER (WHERE harvest_status='HIGH_MATCH_RENAL_EVIDENCE') AS nueva_evidencia_renal_high,
 COUNT(*) FILTER (WHERE harvest_status='HIGH_MATCH_NO_EXPLICIT_RENAL_DOSING_TEXT') AS high_sin_texto_renal_explicito,
 COUNT(*) FILTER (WHERE harvest_status='AMBIGUOUS_DAILYMED_MATCH') AS coincidencia_ambigua,
 COUNT(*) FILTER (WHERE harvest_status='RXNORM_MATCH_NO_DAILYMED_SPL') AS rxnorm_sin_dailymed,
 COUNT(*) FILTER (WHERE harvest_status='NO_RXNORM_MATCH') AS sin_rxnorm,
 COUNT(*) FILTER (WHERE harvest_status LIKE '%ERROR%') AS errores
FROM public.renal_master_harvest;
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.catalog, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    http = HTTP(args.sleep)
    results = []
    total = len(rows)

    for i, row in enumerate(rows, 1):
        try:
            res = process_row(http, row)
        except Exception as e:
            res = Result(
                med_id=row["med_id"],
                generic_name=row["generic_name"],
                normalized_name=row.get("normalized_name") or row["generic_name"],
                previous_status=row.get("renal_audit_status") or "",
                previous_rules=int(float(row.get("renal_rules_published") or 0)),
                previous_auto=int(float(row.get("renal_rules_automatic") or 0)),
                previous_verified_biblio=int(float(row.get("renal_biblio_verified") or 0)),
                harvest_status="UNHANDLED_ERROR",
                match_confidence="NONE",
                error=repr(e),
            )
        results.append(res)
        print(
            f"[{i:04d}/{total:04d}] {res.med_id} {res.generic_name}: "
            f"{res.harvest_status} {res.match_confidence}"
        )

        # Checkpoint every 25 records.
        if i % 25 == 0:
            write_csv(out / "renal_master_evidence.partial.csv", results)

    write_csv(out / "renal_master_evidence.csv", results)

    evidence = [r for r in results if r.harvest_status == "HIGH_MATCH_RENAL_EVIDENCE"]
    ambiguous = [r for r in results if r.harvest_status in {
        "AMBIGUOUS_DAILYMED_MATCH","RXNORM_MATCH_NO_DAILYMED_SPL","NO_RXNORM_MATCH",
        "HIGH_MATCH_NO_EXPLICIT_RENAL_DOSING_TEXT","DAILYMED_XML_NOT_FOUND",
        "DAILYMED_XML_PARSE_ERROR","UNHANDLED_ERROR",
    }]
    write_csv(out / "renal_current_reference_ready.csv", evidence)
    write_csv(out / "renal_needs_secondary_source_or_review.csv", ambiguous)

    sql = generate_sql(results)
    (out / "MEDCALC_RENAL_MAESTRO_GENERADO.sql").write_text(sql, encoding="utf-8")

    counts = {}
    for r in results:
        counts[r.harvest_status] = counts.get(r.harvest_status, 0) + 1

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": len(results),
        "high_confidence_current_renal_evidence": len(evidence),
        "needs_secondary_source_or_review": len(ambiguous),
        "status_counts": dict(sorted(counts.items())),
        "source_policy": {
            "terminology": "RxNorm/RxNav",
            "regulatory_label": "DailyMed SPL",
            "automatic_rules_created": False,
            "existing_current_auto_preserved": True,
        },
    }
    (out / "renal_master_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
