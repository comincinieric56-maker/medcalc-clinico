#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL MULTISOURCE V5
=====================================

Una sola ejecución para el catálogo completo.

Reutiliza evidencia ya generada:
- MEDCALC existente
- RxNorm / RxNav
- DailyMed SPL
- openFDA Drug Label

Y evalúa TODOS los pendientes, en la misma corrida, contra:
- Drugs@FDA (corroboración regulatoria/identidad)
- Health Canada DPD + Product Monograph PDF (evidencia renal)
- EMA official Product Information PDF (evidencia renal)
- fuentes locales del repositorio:
    renal_biblio_verificada_2025.csv
    ajuste_renal.csv
    renal_biblio_ocr_indice.csv

Seguridad clínica:
- NO crea nuevas reglas numéricas automáticas.
- Toda nueva evidencia queda CURRENT_REFERENCE, automatizable=FALSE.
- La identidad farmacológica debe ser exacta/estricta.
- Combinaciones deben conservar todos los ingredientes.
- Una simple mención renal NO basta: se exige acción clínica renal explícita.
- Sobredosis/eventos adversos aislados no califican.
- NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND ≠ "no requiere ajuste".
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rapidfuzz import fuzz

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(".")
OUT = Path("generated_renal_global_v5")
OUT.mkdir(parents=True, exist_ok=True)

CATALOG = Path("MEDCALC_RENAL_MASTER_CATALOGO_1122.csv")
MATRIX_V4 = Path("generated_renal_multisource/renal_multisource_matrix_1122.csv")
V3_ACCEPT = Path("generated_renal_master_v3/renal_v3_identity_accepted.csv")
V4_ACCEPT = Path("generated_renal_master_v4_openfda/renal_v4_openfda_accepted.csv")
V2_FULL = Path("generated_renal_master_v2/renal_v2_full_audit.csv")

LOCAL_FILES = {
    "renal_biblio_verificada_2025": Path("renal_biblio_verificada_2025.csv"),
    "ajuste_renal": Path("ajuste_renal.csv"),
    "renal_biblio_ocr_indice": Path("renal_biblio_ocr_indice.csv"),
}

# ---------------------------------------------------------------------
# Official endpoints
# ---------------------------------------------------------------------

HC_ACTIVE = "https://health-products.canada.ca/api/drug/activeingredient/"
HC_PRODUCT = "https://health-products.canada.ca/api/drug/drugproduct/"
HC_STATUS = "https://health-products.canada.ca/api/drug/status/"
HC_INFO = "https://health-products.canada.ca/dpd-bdpp/info"

DRUGSFDA = "https://api.fda.gov/drug/drugsfda.json"

EMA_SEARCH = "https://www.ema.europa.eu/en/search"

# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

TRANSLATE = {
    "CLORHIDRATO": "HYDROCHLORIDE",
    "HIDROCLORURO": "HYDROCHLORIDE",
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
    "ACIDO": "ACID",
}

SALT_WORDS = {
    "HYDROCHLORIDE","HCL","SODIUM","POTASSIUM","CALCIUM","MAGNESIUM",
    "MESYLATE","BESYLATE","MALEATE","FUMARATE","SUCCINATE","ACETATE",
    "TARTRATE","CITRATE","PHOSPHATE","SULFATE","BROMIDE","CHLORIDE",
    "DIHYDRATE","MONOHYDRATE","TRIHYDRATE","ANHYDROUS","PAMOATE",
    "XINAFOATE","LACTATE","GLUCONATE","DISODIUM","HYDROBROMIDE",
    "CLORHIDRATO","HIDROCLORURO","SODICO","SODICA","POTASICO","POTASICA",
    "CALCICO","CALCICA","MESILATO","BESILATO","MALEATO","FUMARATO",
    "SUCCINATO","ACETATO","TARTRATO","CITRATO","FOSFATO","SULFATO",
    "BROMURO","CLORURO","PAMOATO","LACTATO","GLUCONATO","DE",
}

def norm(s: Any) -> str:
    s = unicodedata.normalize("NFKD", str(s or ""))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper().replace("&", " / ")
    s = re.sub(r"[^A-Z0-9/+ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def translate_name(s: str) -> str:
    return " ".join(TRANSLATE.get(x, x) for x in norm(s).split())

def strip_salts(s: str) -> str:
    return " ".join(
        x for x in translate_name(s).split()
        if x not in SALT_WORDS and x not in {"ACID", "ACIDO"}
    )

def is_combo(s: str) -> bool:
    raw = str(s or "")
    return "/" in raw or " + " in raw or bool(re.search(r"\bY\b", norm(raw)))

def split_combo(s: str) -> list[str]:
    parts = re.split(r"\s*/\s*|\s+\+\s+|\s+\bY\b\s+", str(s or ""), flags=re.I)
    return [strip_salts(x) for x in parts if strip_salts(x)]

def sim(a: str, b: str) -> float:
    a, b = strip_salts(a), strip_salts(b)
    if not a or not b:
        return 0.0
    return float(max(fuzz.WRatio(a, b), fuzz.token_set_ratio(a, b)))

def exact_component_match(catalog_name: str, source_ingredients: list[str]) -> bool:
    src = [strip_salts(x) for x in source_ingredients if strip_salts(x)]
    if not src:
        return False

    if is_combo(catalog_name):
        parts = split_combo(catalog_name)
        if len(parts) < 2:
            return False
        for p in parts:
            if max((sim(p, s) for s in src), default=0) < 92:
                return False
        # prevent collapse into a single ingredient
        return len(src) >= len(parts)

    target = strip_salts(catalog_name)
    if len(src) > 1:
        # A monocomponent cannot be accepted from a true combination label.
        close = [s for s in src if sim(target, s) >= 92]
        if len(close) != 1:
            return False
        # If other substantive ingredients are present, reject.
        others = [s for s in src if sim(target, s) < 92 and len(s) >= 3]
        if others:
            return False
    return max((sim(target, s) for s in src), default=0) >= 92

# ---------------------------------------------------------------------
# Renal clinical-action detector
# ---------------------------------------------------------------------

RENAL = re.compile(
    r"\b(renal impairment|renal insufficien|renal function|kidney impairment|"
    r"kidney function|creatinine clearance|\bcrcl\b|\begfr\b|\bgfr\b|"
    r"hemodialysis|haemodialysis|dialysis|end.stage renal)\b", re.I
)
NO_ADJUST = re.compile(
    r"\b(no (?:dose|dosage) adjustment(?: is)? (?:necessary|required|recommended)?|"
    r"does not require (?:dose|dosage) adjustment|"
    r"dose adjustment is not (?:necessary|required)|"
    r"no adjustment[^.;]{0,100}renal)\b", re.I
)
NOT_RECOMMENDED = re.compile(
    r"\b(not recommended|should not be used|avoid use|contraindicat)\b", re.I
)
THRESHOLD = re.compile(
    r"(?:crcl|creatinine clearance|egfr|gfr)"
    r"[^.;:\n]{0,140}(?:<|>|≤|≥|less than|greater than|between|\d)", re.I
)
DOSE_ACTION = re.compile(
    r"\b(dose|dosage|dosing|administer|administration|reduce|reduction|"
    r"interval|every|once daily|twice daily|q\d+h|"
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml))\b", re.I
)
ADJUST = re.compile(
    r"\b(dose adjustment|dosage adjustment|adjust(?:ment)? of (?:the )?dose|"
    r"reduced dose|reduce the dose|increase[^.;]{0,80}interval|"
    r"decrease[^.;]{0,80}dose)\b", re.I
)
DIALYSIS_DOSING = re.compile(
    r"(?:administer|give|dose|supplement)[^.;:\n]{0,200}"
    r"(?:hemodialysis|haemodialysis|dialysis)|"
    r"(?:hemodialysis|haemodialysis|dialysis)[^.;:\n]{0,200}"
    r"(?:administer|give|dose|supplement)", re.I
)
OVERDOSE = re.compile(r"\b(overdosage|overdose|poisoning)\b", re.I)

def renal_windows(text: str, radius_before: int = 600, radius_after: int = 1800) -> str:
    text = re.sub(r"\s+", " ", str(text or ""))
    wins = []
    for m in RENAL.finditer(text):
        lo = max(0, m.start() - radius_before)
        hi = min(len(text), m.end() + radius_after)
        w = text[lo:hi]
        # Exclude windows that are clearly only an overdose section.
        if OVERDOSE.search(w[:350]) and not re.search(
            r"(renal impairment|creatinine clearance|crcl|egfr|gfr)", w, re.I
        ):
            continue
        wins.append(w)
        if len(wins) >= 8:
            break
    return "\n---\n".join(wins)[:12000]

def clinical_action(text: str) -> tuple[bool, str]:
    text = str(text or "")
    if not RENAL.search(text):
        return False, "NO_RENAL_TEXT"

    if NO_ADJUST.search(text):
        return True, "NO_ADJUSTMENT_EXPLICIT"

    if NOT_RECOMMENDED.search(text):
        if THRESHOLD.search(text) or re.search(
            r"(severe|end.stage|advanced)[^.;]{0,140}renal impairment", text, re.I
        ):
            return True, "RENAL_NOT_RECOMMENDED_EXPLICIT"

    if THRESHOLD.search(text) and DOSE_ACTION.search(text):
        return True, "RENAL_THRESHOLD_DOSING_EXPLICIT"

    if DIALYSIS_DOSING.search(text):
        return True, "DIALYSIS_DOSING_EXPLICIT"

    for m in RENAL.finditer(text):
        lo = max(0, m.start() - 250)
        hi = min(len(text), m.end() + 700)
        if ADJUST.search(text[lo:hi]):
            return True, "RENAL_DOSING_TEXT_EXPLICIT"

    return False, "RENAL_MENTION_WITHOUT_EXPLICIT_DOSING_ACTION"

def action_family(reason: str) -> str:
    r = str(reason or "")
    if "NO_ADJUSTMENT" in r:
        return "NO_ADJUSTMENT"
    if "NOT_RECOMMENDED" in r:
        return "NOT_RECOMMENDED"
    if "DIALYSIS" in r:
        return "DIALYSIS"
    if "DOSING" in r or "THRESHOLD" in r:
        return "ADJUST_DOSING"
    return "OTHER"

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path: Path, rows: list[dict], fields: Optional[list[str]] = None):
    if fields is None:
        fields = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})

def unwrap_json(j: Any) -> list[dict]:
    if j is None:
        return []
    if isinstance(j, list):
        return [x for x in j if isinstance(x, dict)]
    if isinstance(j, dict):
        if isinstance(j.get("result"), list):
            return [x for x in j["result"] if isinstance(x, dict)]
        if isinstance(j.get("results"), list):
            return [x for x in j["results"] if isinstance(x, dict)]
        return [j]
    return []

def trunc(s: Any, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n-1] + "…"

def sqlq(v: Optional[str]) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''") + "'"

# ---------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------

def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "MEDCALC-RenalGlobalV5/1.0 (+clinical evidence audit)",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return s

def get(session: requests.Session, url: str, *, params=None, timeout=35, tries=4):
    last = None
    for i in range(tries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2.5 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            time.sleep(1.0 + i * 1.5)
    if last:
        raise last
    return None

# ---------------------------------------------------------------------
# Local-source indexes
# ---------------------------------------------------------------------

def pick_col(rows: list[dict], candidates: list[str]) -> Optional[str]:
    if not rows:
        return None
    cols = list(rows[0].keys())
    m = {norm(c): c for c in cols}
    for c in candidates:
        if norm(c) in m:
            return m[norm(c)]
    return None

def build_local_index(path: Path) -> tuple[dict, int]:
    rows = read_csv(path)
    idx = {}
    if not rows:
        return idx, 0
    name_col = pick_col(
        rows,
        ["principio_activo","generic_name","medicamento","drug_name_source",
         "nombre","farmaco","fármaco","medication"]
    )
    medid_col = pick_col(rows, ["med_id","medid","id_revision","medication_id"])
    for r in rows:
        if medid_col and r.get(medid_col):
            idx.setdefault(("MEDID", str(r[medid_col]).strip()), 0)
            idx[("MEDID", str(r[medid_col]).strip())] += 1
        if name_col and r.get(name_col):
            idx.setdefault(("NAME", norm(r[name_col])), 0)
            idx[("NAME", norm(r[name_col]))] += 1
    return idx, len(rows)

LOCAL_INDEXES = {}
LOCAL_ROW_COUNTS = {}
for key, path in LOCAL_FILES.items():
    ix, n = build_local_index(path)
    LOCAL_INDEXES[key] = ix
    LOCAL_ROW_COUNTS[key] = n

def local_support(med_id: str, generic_name: str) -> dict:
    out = {}
    for key, ix in LOCAL_INDEXES.items():
        out[key] = bool(
            ix.get(("MEDID", med_id))
            or ix.get(("NAME", norm(generic_name)))
        )
    return out

# ---------------------------------------------------------------------
# Drugs@FDA
# ---------------------------------------------------------------------

def drugsfda_check(session: requests.Session, generic_name: str) -> dict:
    api_key = os.getenv("OPENFDA_API_KEY", "").strip()

    # Without an API key openFDA allows 1,000 requests/day per IP.
    # V5 has <=810 pending MED-ID, so we deliberately use at most ONE
    # Drugs@FDA request per medication when no key is configured.
    if is_combo(generic_name):
        parts = split_combo(generic_name)
        search_name = parts[0] if parts else strip_salts(generic_name)
    else:
        search_name = strip_salts(generic_name)

    if not search_name:
        return {
            "drugsfda_status": "NO_EXACT_IDENTITY_MATCH",
            "drugsfda_application_number": "",
            "drugsfda_identity_score": "",
        }

    if api_key:
        names = [translate_name(generic_name), strip_salts(generic_name), search_name]
        queries = []
        for n in names:
            if n:
                queries.append(f'openfda.generic_name:"{n}"')
                queries.append(f'products.active_ingredients.name:"{n}"')
        queries = list(dict.fromkeys(queries))
    else:
        queries = [f'products.active_ingredients.name:"{search_name}"']

    best = None
    best_score = 0.0
    for q in queries:
        params = {"search": q, "limit": 20}
        if api_key:
            params["api_key"] = api_key
        try:
            r = get(session, DRUGSFDA, params=params)
            if not r:
                continue
            for rec in (r.json().get("results") or []):
                ingredients = []
                for p in rec.get("products") or []:
                    for ai in p.get("active_ingredients") or []:
                        if ai.get("name"):
                            ingredients.append(ai["name"])
                of = rec.get("openfda") or {}
                ingredients += of.get("substance_name") or []
                ingredients = list(dict.fromkeys(ingredients))
                if not exact_component_match(generic_name, ingredients):
                    continue
                score = max((sim(generic_name, x) for x in ingredients), default=0)
                if score > best_score:
                    best_score = score
                    best = rec
        except Exception:
            continue

    if not best:
        return {
            "drugsfda_status": "NO_EXACT_IDENTITY_MATCH",
            "drugsfda_application_number": "",
            "drugsfda_identity_score": "",
        }

    return {
        "drugsfda_status": "EXACT_IDENTITY_CORROBORATED",
        "drugsfda_application_number": best.get("application_number", ""),
        "drugsfda_identity_score": f"{best_score:.1f}",
    }

# ---------------------------------------------------------------------
# Health Canada DPD + current Product Monograph
# ---------------------------------------------------------------------

def hc_search_codes(session: requests.Session, generic_name: str) -> list[int]:
    parts = split_combo(generic_name) if is_combo(generic_name) else [strip_salts(generic_name)]
    if not parts:
        return []

    code_sets = []
    for p in parts:
        # DPD uses INN/English names. Salt-stripped base is usually safer.
        try:
            r = get(
                session, HC_ACTIVE,
                params={"ingredientname": p, "lang": "en", "type": "json"}
            )
            rows = unwrap_json(r.json() if r else None)
        except Exception:
            rows = []

        codes = set()
        for x in rows:
            ing = x.get("ingredient_name") or x.get("ingredientname") or ""
            code = x.get("drug_code") or x.get("drugcode")
            if code is None:
                continue
            if sim(p, ing) >= 90:
                try:
                    codes.add(int(code))
                except Exception:
                    pass
        code_sets.append(codes)

    if not code_sets or any(not x for x in code_sets):
        return []
    common = set.intersection(*code_sets)
    return sorted(common)[:60]

def hc_ingredients_for_code(session: requests.Session, code: int) -> list[str]:
    try:
        r = get(
            session, HC_ACTIVE,
            params={"id": code, "lang": "en", "type": "json"}
        )
        rows = unwrap_json(r.json() if r else None)
    except Exception:
        return []
    vals = []
    for x in rows:
        v = x.get("ingredient_name") or x.get("ingredientname")
        if v:
            vals.append(str(v))
    return list(dict.fromkeys(vals))

def hc_product_for_code(session: requests.Session, code: int) -> dict:
    try:
        r = get(
            session, HC_PRODUCT,
            params={"id": code, "lang": "en", "type": "json"}
        )
        rows = unwrap_json(r.json() if r else None)
        return rows[0] if rows else {}
    except Exception:
        return {}

def hc_status_for_code(session: requests.Session, code: int) -> str:
    try:
        r = get(
            session, HC_STATUS,
            params={"id": code, "lang": "en", "type": "json"}
        )
        rows = unwrap_json(r.json() if r else None)
        if not rows:
            return ""
        x = rows[0]
        return str(
            x.get("status_name")
            or x.get("status")
            or x.get("current_status")
            or ""
        )
    except Exception:
        return ""

def pdf_text_from_url(session: requests.Session, url: str) -> str:
    r = get(session, url, timeout=60, tries=3)
    if not r:
        return ""
    data = r.content
    if not data.startswith(b"%PDF"):
        return ""
    reader = PdfReader(io.BytesIO(data))
    pieces = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if RENAL.search(t):
            pieces.append(t)
        if len("\n".join(pieces)) > 18000:
            break
    return "\n".join(pieces)

def hc_monograph_for_code(session: requests.Session, code: int) -> dict:
    page_url = f"{HC_INFO}?code={code}&lang=eng"
    try:
        r = get(session, HC_INFO, params={"code": code, "lang": "eng"})
        if not r:
            return {}
        html = r.text
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return {}

    best_href = ""
    for a in soup.find_all("a", href=True):
        txt = " ".join(a.stripped_strings).lower()
        href = a["href"]
        blob = (txt + " " + href.lower())
        if ".pdf" in blob and ("monograph" in blob or "product" in blob):
            best_href = urljoin(r.url, href)
            break

    # Some pages expose a PDF link with generic text.
    if not best_href:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if ".pdf" in href.lower():
                best_href = urljoin(r.url, href)
                break

    if not best_href:
        return {
            "page_url": r.url,
            "pdf_url": "",
            "monograph_date": "",
            "renal_text": "",
        }

    page_text = " ".join(soup.stripped_strings)
    m = re.search(
        r"Product Monograph.*?Date:\s*(\d{4}-\d{2}-\d{2})",
        page_text, re.I
    )
    date = m.group(1) if m else ""

    try:
        full_text = pdf_text_from_url(session, best_href)
    except Exception:
        full_text = ""

    renal_text = renal_windows(full_text)
    return {
        "page_url": r.url,
        "pdf_url": best_href,
        "monograph_date": date,
        "renal_text": renal_text,
    }

def health_canada_check(session: requests.Session, generic_name: str) -> dict:
    codes = hc_search_codes(session, generic_name)
    candidates = []
    for code in codes[:24]:
        ingredients = hc_ingredients_for_code(session, code)
        if not exact_component_match(generic_name, ingredients):
            continue
        product = hc_product_for_code(session, code)
        status = hc_status_for_code(session, code)
        candidates.append((code, ingredients, product, status))

    if not candidates:
        return {
            "hc_status": "NO_EXACT_DPD_PRODUCT",
            "hc_reason": "",
            "hc_drug_code": "",
            "hc_din": "",
            "hc_product_name": "",
            "hc_monograph_url": "",
            "hc_monograph_date": "",
            "hc_renal_text": "",
        }

    # Prefer marketed/approved, newest product update.
    def rank(c):
        code, ingredients, product, status = c
        s = norm(status)
        active = 2 if ("MARKETED" in s or "APPROVED" in s) else 1 if "DORMANT" in s else 0
        date = str(product.get("last_update_date") or "")
        return (active, date, code)

    candidates.sort(key=rank, reverse=True)

    for code, ingredients, product, status in candidates[:8]:
        mono = hc_monograph_for_code(session, code)
        renal_text = mono.get("renal_text") or ""
        ok, reason = clinical_action(renal_text)
        if ok:
            return {
                "hc_status": "ACCEPT",
                "hc_reason": reason,
                "hc_drug_code": str(code),
                "hc_din": str(product.get("drug_identification_number") or ""),
                "hc_product_name": str(product.get("brand_name") or ""),
                "hc_product_status": status,
                "hc_monograph_url": mono.get("pdf_url") or mono.get("page_url") or "",
                "hc_monograph_date": mono.get("monograph_date") or "",
                "hc_renal_text": renal_text,
            }

    # Exact identity existed but no explicit renal dosing found.
    code, ingredients, product, status = candidates[0]
    mono = hc_monograph_for_code(session, code)
    renal_text = mono.get("renal_text") or ""
    ok, reason = clinical_action(renal_text)
    return {
        "hc_status": "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
        "hc_reason": reason,
        "hc_drug_code": str(code),
        "hc_din": str(product.get("drug_identification_number") or ""),
        "hc_product_name": str(product.get("brand_name") or ""),
        "hc_product_status": status,
        "hc_monograph_url": mono.get("pdf_url") or mono.get("page_url") or "",
        "hc_monograph_date": mono.get("monograph_date") or "",
        "hc_renal_text": renal_text,
    }

# ---------------------------------------------------------------------
# EMA official Product Information
# ---------------------------------------------------------------------

def ema_identity_page_ok(generic_name: str, page_text: str) -> bool:
    page_n = strip_salts(page_text)
    if is_combo(generic_name):
        return all(
            fuzz.partial_ratio(p, page_n) >= 90
            for p in split_combo(generic_name)
        )
    target = strip_salts(generic_name)
    return bool(target) and fuzz.partial_ratio(target, page_n) >= 94

def ema_candidate_pages(session: requests.Session, generic_name: str) -> list[str]:
    urls = []
    try:
        r = get(
            session, EMA_SEARCH,
            params={"search_api_fulltext": strip_salts(generic_name)}
        )
        if not r:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(r.url, a["href"])
            if "/en/medicines/human/" in href:
                href = href.split("#", 1)[0].split("?", 1)[0]
                if href not in urls:
                    urls.append(href)
            if len(urls) >= 8:
                break
    except Exception:
        pass
    return urls

def ema_product_pdf(session: requests.Session, page_url: str, generic_name: str) -> dict:
    try:
        r = get(session, page_url)
        if not r:
            return {}
        soup = BeautifulSoup(r.text, "html.parser")
        text = " ".join(soup.stripped_strings)
        if not ema_identity_page_ok(generic_name, text):
            return {}
    except Exception:
        return {}

    pdfs = []
    for a in soup.find_all("a", href=True):
        href = urljoin(r.url, a["href"])
        txt = " ".join(a.stripped_strings).lower()
        blob = (txt + " " + href.lower())
        if ".pdf" not in blob:
            continue
        score = 0
        if "product information" in blob:
            score += 5
        if "product-information" in blob:
            score += 5
        if "annex" in blob:
            score += 2
        if "_en" in blob or "en.pdf" in blob:
            score += 1
        if "assessment" in blob or "overview" in blob:
            score -= 4
        pdfs.append((score, href))

    pdfs.sort(reverse=True)
    for score, pdf in pdfs[:5]:
        if score < 1:
            continue
        try:
            full = pdf_text_from_url(session, pdf)
        except Exception:
            continue
        renal_text = renal_windows(full)
        ok, reason = clinical_action(renal_text)
        if ok:
            return {
                "ema_status": "ACCEPT",
                "ema_reason": reason,
                "ema_product_page": r.url,
                "ema_pdf_url": pdf,
                "ema_renal_text": renal_text,
            }

    return {
        "ema_status": "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
        "ema_reason": "NO_EXPLICIT_RENAL_ACTION_IN_RETRIEVED_PI",
        "ema_product_page": r.url,
        "ema_pdf_url": "",
        "ema_renal_text": "",
    }

def ema_check(session: requests.Session, generic_name: str) -> dict:
    pages = ema_candidate_pages(session, generic_name)
    exact_seen = False
    for page in pages:
        rec = ema_product_pdf(session, page, generic_name)
        if not rec:
            continue
        exact_seen = True
        if rec.get("ema_status") == "ACCEPT":
            return rec
    if exact_seen:
        return {
            "ema_status": "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
            "ema_reason": "NO_EXPLICIT_RENAL_ACTION_IN_RETRIEVED_PI",
            "ema_product_page": "",
            "ema_pdf_url": "",
            "ema_renal_text": "",
        }
    return {
        "ema_status": "NO_EXACT_EMA_PRODUCT",
        "ema_reason": "",
        "ema_product_page": "",
        "ema_pdf_url": "",
        "ema_renal_text": "",
    }

# ---------------------------------------------------------------------
# Process one unresolved MED-ID
# ---------------------------------------------------------------------

def process_one(row: dict) -> dict:
    med_id = row["med_id"]
    generic_name = row["generic_name"]
    s = new_session()

    out = {
        "med_id": med_id,
        "generic_name": generic_name,
    }
    out.update(local_support(med_id, generic_name))

    # Independent sources: query all in the same task.
    try:
        out.update(drugsfda_check(s, generic_name))
    except Exception as exc:
        out.update(
            drugsfda_status="ERROR",
            drugsfda_application_number="",
            drugsfda_identity_score="",
            drugsfda_error=repr(exc),
        )

    try:
        out.update(health_canada_check(s, generic_name))
    except Exception as exc:
        out.update(
            hc_status="ERROR",
            hc_reason="",
            hc_drug_code="",
            hc_din="",
            hc_product_name="",
            hc_monograph_url="",
            hc_monograph_date="",
            hc_renal_text="",
            hc_error=repr(exc),
        )

    try:
        out.update(ema_check(s, generic_name))
    except Exception as exc:
        out.update(
            ema_status="ERROR",
            ema_reason="",
            ema_product_page="",
            ema_pdf_url="",
            ema_renal_text="",
            ema_error=repr(exc),
        )

    # Final adjudication for new current evidence.
    evidence = []
    if out.get("hc_status") == "ACCEPT":
        evidence.append(("HEALTH_CANADA", out.get("hc_reason", "")))
    if out.get("ema_status") == "ACCEPT":
        evidence.append(("EMA", out.get("ema_reason", "")))

    if len(evidence) >= 2:
        families = {action_family(x[1]) for x in evidence}
        if len(families) == 1:
            out["v5_resolution"] = "CURRENT_MULTI_OFFICIAL_CONCORDANT"
            out["v5_primary_source"] = "HEALTH_CANADA"
            out["v5_reason"] = evidence[0][1]
        else:
            out["v5_resolution"] = "OFFICIAL_SOURCES_ACTION_CONFLICT"
            out["v5_primary_source"] = ""
            out["v5_reason"] = "ACTION_FAMILY_CONFLICT"
    elif len(evidence) == 1:
        out["v5_resolution"] = "CURRENT_SINGLE_OFFICIAL_EXACT"
        out["v5_primary_source"] = evidence[0][0]
        out["v5_reason"] = evidence[0][1]
    else:
        local_any = any(
            bool(out.get(k))
            for k in LOCAL_FILES.keys()
        )
        hc_exact = out.get("hc_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
        ema_exact = out.get("ema_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
        fda_exact = out.get("drugsfda_status") == "EXACT_IDENTITY_CORROBORATED"

        if hc_exact or ema_exact:
            out["v5_resolution"] = "NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        elif local_any or fda_exact:
            out["v5_resolution"] = "CURRENT_SOURCE_NOT_FOUND_WITH_SUPPORT"
        else:
            out["v5_resolution"] = "UNRESOLVED"
        out["v5_primary_source"] = ""
        out["v5_reason"] = ""

    return out

# ---------------------------------------------------------------------
# Build one final matrix for all 1122
# ---------------------------------------------------------------------

def existing_resolution_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    catalog = read_csv(CATALOG)
    matrix = read_csv(MATRIX_V4)
    v3 = read_csv(V3_ACCEPT)
    v4 = read_csv(V4_ACCEPT)
    v2 = read_csv(V2_FULL)

    if len(catalog) != 1122:
        raise SystemExit(f"SEGURIDAD: catálogo esperado 1122; encontrado {len(catalog)}")
    if len(matrix) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz previa esperada 1122; encontrada {len(matrix)}")

    return catalog, matrix, v3, v4

def evidence_sql_rows(v3: list[dict], v4: list[dict], new_rows: list[dict]) -> list[dict]:
    rows = []

    # DailyMed V3 (already identity-confirmed)
    for r in v3:
        rows.append({
            "med_id": r.get("med_id"),
            "generic_name": r.get("generic_name"),
            "source_kind": "DAILYMED_V3",
            "reason": r.get("v2_reason") or "IDENTITY_CONFIRMED",
            "source_url": r.get("source_url"),
            "source_title": f"DailyMed — {r.get('generic_name')} — renal labeling",
            "source_org": "U.S. National Library of Medicine · DailyMed",
            "source_locator": r.get("renal_section_titles"),
            "renal_text": r.get("renal_text"),
            "external_id": r.get("setid"),
        })

    # openFDA V4
    for r in v4:
        rows.append({
            "med_id": r.get("med_id"),
            "generic_name": r.get("generic_name"),
            "source_kind": "OPENFDA_V4",
            "reason": r.get("v4_reason"),
            "source_url": r.get("v4_source_url"),
            "source_title": f"openFDA — {r.get('generic_name')} — renal labeling",
            "source_org": "U.S. Food and Drug Administration · openFDA",
            "source_locator": r.get("v4_renal_fields"),
            "renal_text": r.get("v4_renal_text"),
            "external_id": r.get("v4_spl_set_id"),
        })

    # V5 Health Canada / EMA: only new resolved MED-ID.
    for r in new_rows:
        resolution = r.get("v5_resolution")
        if resolution not in {
            "CURRENT_SINGLE_OFFICIAL_EXACT",
            "CURRENT_MULTI_OFFICIAL_CONCORDANT",
        }:
            continue

        # Primary current reference.
        primary = r.get("v5_primary_source")
        if primary == "HEALTH_CANADA":
            rows.append({
                "med_id": r.get("med_id"),
                "generic_name": r.get("generic_name"),
                "source_kind": "HEALTH_CANADA_V5",
                "reason": r.get("hc_reason"),
                "source_url": r.get("hc_monograph_url"),
                "source_title": f"Health Canada — {r.get('generic_name')} — Product Monograph",
                "source_org": "Health Canada · Drug Product Database",
                "source_locator": f"DIN {r.get('hc_din','')} · drug_code {r.get('hc_drug_code','')}",
                "renal_text": r.get("hc_renal_text"),
                "external_id": r.get("hc_din"),
            })
        elif primary == "EMA":
            rows.append({
                "med_id": r.get("med_id"),
                "generic_name": r.get("generic_name"),
                "source_kind": "EMA_V5",
                "reason": r.get("ema_reason"),
                "source_url": r.get("ema_pdf_url") or r.get("ema_product_page"),
                "source_title": f"EMA — {r.get('generic_name')} — Product Information",
                "source_org": "European Medicines Agency",
                "source_locator": "Official Product Information",
                "renal_text": r.get("ema_renal_text"),
                "external_id": "",
            })

    # Deduplicate by MED-ID; preserve earlier accepted current evidence.
    # The SQL will also check live DB before inserting.
    dedup = {}
    priority = {
        "DAILYMED_V3": 4,
        "OPENFDA_V4": 3,
        "HEALTH_CANADA_V5": 2,
        "EMA_V5": 1,
    }
    for r in rows:
        mid = r.get("med_id")
        if not mid:
            continue
        if mid not in dedup or priority.get(r["source_kind"], 0) > priority.get(dedup[mid]["source_kind"], 0):
            dedup[mid] = r
    return list(dedup.values())

def build_sql(evidence: list[dict]) -> str:
    vals = []
    for r in evidence:
        vals.append("(" + ",".join([
            sqlq(r.get("med_id")),
            sqlq(r.get("generic_name")),
            sqlq(r.get("source_kind")),
            sqlq(r.get("reason")),
            sqlq(r.get("source_url")),
            sqlq(r.get("source_title")),
            sqlq(r.get("source_org")),
            sqlq(trunc(r.get("source_locator") or "", 1800)),
            sqlq(trunc(r.get("renal_text") or "", 12000)),
            sqlq(r.get("external_id")),
        ]) + ")")
    vv = ",\n".join(vals)
    if not vv:
        vv = "('','','','','','','','','','')"

    return f"""-- =====================================================================
-- MEDCALC · RENAL GLOBAL MULTISOURCE V5 · SQL FINAL CANDIDATO
-- Generado UTC: {datetime.now(timezone.utc).isoformat()}
--
-- FUENTES:
--   MEDCALC existente + RxNorm/RxNav + DailyMed + openFDA +
--   Drugs@FDA + Health Canada DPD/Product Monograph + EMA Product Info +
--   bibliografía local.
--
-- SEGURIDAD:
--   * NO crea reglas numéricas automáticas nuevas.
--   * Sólo carga referencias actuales con identidad estricta y acción renal.
--   * Si el medicamento ya tiene CURRENT_AUTO/CURRENT_REFERENCE/TDM
--     validado en Supabase, NO crea otra regla.
-- =====================================================================

CREATE TABLE IF NOT EXISTS public.renal_global_multisource_v5 (
    medication_id uuid PRIMARY KEY REFERENCES public.medications(id) ON DELETE CASCADE,
    med_id text NOT NULL,
    generic_name text NOT NULL,
    source_kind text NOT NULL,
    decision_reason text NOT NULL,
    source_url text NOT NULL,
    external_id text,
    loaded_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

WITH e(
 med_id,generic_name,source_kind,reason,source_url,source_title,
 source_org,source_locator,renal_text,external_id
) AS (
VALUES
{vv}
)
INSERT INTO public.renal_global_multisource_v5
(medication_id,med_id,generic_name,source_kind,decision_reason,source_url,external_id,loaded_at,updated_at)
SELECT m.id,e.med_id,e.generic_name,e.source_kind,e.reason,e.source_url,e.external_id,NOW(),NOW()
FROM e
JOIN public.medications m ON m.med_id=e.med_id
WHERE e.med_id<>''
ON CONFLICT (medication_id) DO UPDATE
SET generic_name=EXCLUDED.generic_name,
    source_kind=EXCLUDED.source_kind,
    decision_reason=EXCLUDED.decision_reason,
    source_url=EXCLUDED.source_url,
    external_id=EXCLUDED.external_id,
    updated_at=NOW();

-- Sources
WITH e(
 med_id,generic_name,source_kind,reason,source_url,source_title,
 source_org,source_locator,renal_text,external_id
) AS (
VALUES
{vv}
)
INSERT INTO public.sources
(id,title,organization,url,page,source_type,last_verified,created_at)
SELECT gen_random_uuid(),e.source_title,e.source_org,e.source_url,
       NULLIF(e.source_locator,''),'RENAL_REFERENCIA',CURRENT_DATE,NOW()
FROM e
WHERE e.med_id<>'' AND e.source_url<>''
AND NOT EXISTS (
    SELECT 1 FROM public.sources s WHERE s.url=e.source_url
);

-- Verified bibliography: only if live DB does not already contain a verified
-- published renal bibliography entry for this medicine/source.
WITH e(
 med_id,generic_name,source_kind,reason,source_url,source_title,
 source_org,source_locator,renal_text,external_id
) AS (
VALUES
{vv}
)
INSERT INTO public.renal_bibliography
(id,medication_id,drug_name_source,normal_dose,adjustment_method,
 recommendations,verified,status,source_id,created_at,updated_at)
SELECT gen_random_uuid(),m.id,m.generic_name,NULL,
       CONCAT('RENAL_GLOBAL_V5:',e.source_kind,':',e.reason),
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
      SELECT 1
      FROM public.renal_bibliography rb
      WHERE rb.medication_id=m.id
        AND rb.source_id=src.id
        AND rb.status='PUBLISHED'
        AND COALESCE(rb.verified,FALSE)=TRUE
  );

-- Current reference rule only when no validated CURRENT_AUTO,
-- CURRENT_REFERENCE or TDM exists already.
WITH e(
 med_id,generic_name,source_kind,reason,source_url,source_title,
 source_org,source_locator,renal_text,external_id
) AS (
VALUES
{vv}
)
INSERT INTO public.renal_rules
(id,medication_id,indication,population,route,renal_metric,range_text,
 lower_limit,upper_limit,lower_inclusive,upper_inclusive,
 adjusted_regimen,rule_type,notes,automatizable,status,source_id,reviewed_at)
SELECT gen_random_uuid(),m.id,
       CONCAT(m.generic_name,' — referencia renal regulatoria · GLOBAL V5'),
       'Adulto / según ficha regulatoria',
       NULL,NULL,'Referencia clínica actual',
       NULL,NULL,FALSE,FALSE,
       e.renal_text,
       CASE
         WHEN e.reason LIKE '%NO_ADJUSTMENT%' THEN 'NO_AJUSTE'
         ELSE 'PRECAUCION'
       END,
       CONCAT(e.source_kind,' · ',e.reason,
              ' · referencia actual no automatizable'),
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
       'RENAL GLOBAL MULTISOURCE V5: identidad farmacológica estricta + recomendación renal regulatoria explícita; referencia no automatizable.',
       NOW(),NOW()
FROM public.renal_rules rr
JOIN public.medications m ON m.id=rr.medication_id
WHERE rr.indication=CONCAT(m.generic_name,' — referencia renal regulatoria · GLOBAL V5')
  AND rr.status='PUBLISHED'
ON CONFLICT (rule_id) DO UPDATE
SET validation_class='CURRENT_REFERENCE',
    evidence_note=EXCLUDED.evidence_note,
    validated_at=NOW(),updated_at=NOW();

-- Preserve CURRENT_AUTO and TDM closures.
INSERT INTO public.renal_phase6_review
(medication_id,med_id,generic_name,original_priority,review_batch,
 phase6_status,disposition,source_id,decision_note,validated_at,updated_at)
SELECT m.id,m.med_id,m.generic_name,2,11,
       'CLOSED_CURRENT_REFERENCE','REFERENCIA_ACTUAL_VALIDADA',
       rr.source_id,rr.adjusted_regimen,NOW(),NOW()
FROM public.medications m
JOIN public.renal_rules rr
  ON rr.medication_id=m.id
 AND rr.indication=CONCAT(m.generic_name,' — referencia renal regulatoria · GLOBAL V5')
JOIN public.renal_rule_validation rv
  ON rv.rule_id=rr.id
 AND rv.validation_class='CURRENT_REFERENCE'
WHERE NOT EXISTS (
    SELECT 1 FROM public.renal_phase6_review p
    WHERE p.medication_id=m.id
      AND p.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
)
ON CONFLICT (medication_id) DO UPDATE
SET review_batch=CASE
      WHEN public.renal_phase6_review.phase6_status IN ('CLOSED_CURRENT_AUTO','CLOSED_TDM')
      THEN public.renal_phase6_review.review_batch ELSE 11 END,
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
SELECT DISTINCT rr.medication_id,'PUBLISHED'
FROM public.renal_rules rr
JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
WHERE rr.status='PUBLISHED'
  AND rv.validation_class IN ('CURRENT_AUTO','CURRENT_REFERENCE','TDM')
ON CONFLICT (medication_id) DO UPDATE
SET renal_status='PUBLISHED';

-- Verification
SELECT
 COUNT(DISTINCT m.id) FILTER (
   WHERE EXISTS (
     SELECT 1 FROM public.renal_rules rr
     JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
     WHERE rr.medication_id=m.id
       AND rr.status='PUBLISHED'
       AND rv.validation_class IN ('CURRENT_AUTO','CURRENT_REFERENCE','TDM')
   )
 ) AS medicamentos_con_regla_renal_validada,
 COUNT(*) AS medicamentos_activos_totales
FROM public.medications m
WHERE COALESCE(m.active,TRUE)=TRUE;
"""

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    catalog, matrix, v3, v4 = existing_resolution_rows()
    by_mid = {r["med_id"]: r for r in matrix}

    unresolved = [
        c for c in catalog
        if by_mid.get(c["med_id"], {}).get("next_action") != "STRUCTURALLY_RESOLVED"
    ]

    # Current repo state should contain the 810 pending from multisource V4.
    # Do not hard-fail on a slightly different count; preserve 1122 invariant.
    print(f"Catalog: {len(catalog)}")
    print(f"Pending entering V5: {len(unresolved)}")

    results = []
    max_workers = int(os.getenv("RENAL_V5_WORKERS", "5"))
    max_workers = min(max(max_workers, 1), 8)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, r): r for r in unresolved}
        done = 0
        for fut in as_completed(futures):
            base = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "med_id": base["med_id"],
                    "generic_name": base["generic_name"],
                    "v5_resolution": "UNRESOLVED",
                    "v5_primary_source": "",
                    "v5_reason": "",
                    "v5_error": repr(exc),
                }
                rec.update(local_support(base["med_id"], base["generic_name"]))
            results.append(rec)
            print(
                f"[{done:03d}/{len(unresolved)}] "
                f"{rec['med_id']} {rec['generic_name']} -> "
                f"{rec.get('v5_resolution')}"
            )
            if done % 25 == 0:
                write_csv(OUT / "renal_global_v5_partial.csv", sorted(results, key=lambda x: x["med_id"]))

    results.sort(key=lambda x: x["med_id"])
    write_csv(OUT / "renal_global_v5_pending_input_full_results.csv", results)

    # Final 1122 matrix
    new_by = {r["med_id"]: r for r in results}
    final = []
    for c in catalog:
        mid = c["med_id"]
        old = by_mid.get(mid, {})
        row = {
            "med_id": mid,
            "generic_name": c["generic_name"],
            "prior_resolution": old.get("resolution", ""),
        }

        if old.get("next_action") == "STRUCTURALLY_RESOLVED":
            row["global_resolution"] = old.get("resolution") or "STRUCTURALLY_RESOLVED_PRE_V5"
            row["global_primary_source"] = (
                "MEDCALC" if old.get("existing_medcalc_verified") in ("True", "true", True)
                else "DAILYMED" if old.get("dailymed_v3_confirmed") in ("True", "true", True)
                else "OPENFDA" if old.get("openfda_v4_confirmed") in ("True", "true", True)
                else "PREVIOUS"
            )
            row["global_current_resolved"] = True
        else:
            nr = new_by.get(mid, {})
            row.update(nr)
            row["global_resolution"] = nr.get("v5_resolution", "UNRESOLVED")
            row["global_primary_source"] = nr.get("v5_primary_source", "")
            row["global_current_resolved"] = nr.get("v5_resolution") in {
                "CURRENT_SINGLE_OFFICIAL_EXACT",
                "CURRENT_MULTI_OFFICIAL_CONCORDANT",
            }

        final.append(row)

    if len(final) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz final !=1122 ({len(final)})")

    write_csv(OUT / "renal_global_matrix_1122.csv", final)

    resolved = [
        r for r in final
        if r.get("global_current_resolved") is True
        or r.get("global_resolution") in {
            "EXISTING_VERIFIED_MEDCALC",
            "CURRENT_DAILYMED_IDENTITY_CONFIRMED",
            "CURRENT_OPENFDA_IDENTITY_CONFIRMED",
            "CURRENT_MULTI_OFFICIAL_CONCORDANT",
            "CURRENT_SINGLE_OFFICIAL_EXACT",
        }
    ]
    pending = [r for r in final if r not in resolved]

    write_csv(OUT / "renal_global_resolved.csv", resolved)
    write_csv(OUT / "renal_global_still_pending.csv", pending)

    # Counts
    resolution_counts = {}
    for r in final:
        k = r.get("global_resolution") or "UNKNOWN"
        resolution_counts[k] = resolution_counts.get(k, 0) + 1

    v5_counts = {}
    for r in results:
        k = r.get("v5_resolution") or "UNKNOWN"
        v5_counts[k] = v5_counts.get(k, 0) + 1

    source_counts = {
        "health_canada_accept": sum(r.get("hc_status") == "ACCEPT" for r in results),
        "ema_accept": sum(r.get("ema_status") == "ACCEPT" for r in results),
        "drugsfda_exact_identity": sum(
            r.get("drugsfda_status") == "EXACT_IDENTITY_CORROBORATED"
            for r in results
        ),
        "local_biblio_support": sum(
            bool(r.get("renal_biblio_verificada_2025")) for r in results
        ),
        "local_ajuste_support": sum(
            bool(r.get("ajuste_renal")) for r in results
        ),
        "local_ocr_support": sum(
            bool(r.get("renal_biblio_ocr_indice")) for r in results
        ),
    }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": 1122,
        "prior_structurally_resolved": 1122 - len(unresolved),
        "v5_pending_input": len(unresolved),
        "global_current_resolved": len(resolved),
        "global_still_pending": len(pending),
        "total_check": len(resolved) + len(pending),
        "resolution_counts": dict(sorted(resolution_counts.items())),
        "v5_pending_resolution_counts": dict(sorted(v5_counts.items())),
        "source_counts_on_v5_pending": source_counts,
        "local_repository_row_counts": LOCAL_ROW_COUNTS,
        "source_policy": {
            "RxNorm_RxNav": "identity normalization from earlier stages",
            "DailyMed_SPL": "current official US labeling from earlier stages",
            "openFDA_Drug_Label": "current FDA SPL labeling from earlier stages",
            "DrugsFDA": "independent FDA approval/product identity corroboration",
            "Health_Canada_DPD_Product_Monograph": "current official Canadian product monograph renal evidence",
            "EMA_Product_Information": "current official EU product information renal evidence when available",
            "local_files": "secondary support only; never close CURRENT_REFERENCE alone",
        },
        "safety": {
            "new_numeric_automatic_rules_created": False,
            "existing_current_auto_preserved": True,
            "explicit_renal_dosing_action_required": True,
            "exact_or_strict_drug_identity_required": True,
            "combination_integrity_required": True,
            "no_explicit_renal_recommendation_does_not_mean_no_adjustment": True,
        },
    }

    (OUT / "renal_global_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    evidence = evidence_sql_rows(v3, v4, results)
    (OUT / "MEDCALC_RENAL_GLOBAL_MULTISOURCE_FINAL_V5.sql").write_text(
        build_sql(evidence),
        encoding="utf-8"
    )

    # Read-only verification SQL
    verify_sql = """-- MEDCALC RENAL GLOBAL V5 · VERIFICACIÓN POST-CARGA
SELECT
  COUNT(*) FILTER (WHERE renal_status='PUBLISHED') AS renal_published,
  COUNT(*) AS total_module_rows
FROM public.medication_module_status;

SELECT
  rv.validation_class,
  COUNT(DISTINCT rr.medication_id) AS medicamentos
FROM public.renal_rules rr
JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
WHERE rr.status='PUBLISHED'
GROUP BY rv.validation_class
ORDER BY rv.validation_class;

SELECT
  COUNT(DISTINCT m.id) AS activos,
  COUNT(DISTINCT m.id) FILTER (
    WHERE EXISTS (
      SELECT 1 FROM public.renal_rules rr
      JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
      WHERE rr.medication_id=m.id
        AND rr.status='PUBLISHED'
        AND rv.validation_class IN ('CURRENT_AUTO','CURRENT_REFERENCE','TDM')
    )
  ) AS con_regla_validada
FROM public.medications m
WHERE COALESCE(m.active,TRUE)=TRUE;
"""
    (OUT / "MEDCALC_RENAL_GLOBAL_V5_VERIFICAR.sql").write_text(
        verify_sql, encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
