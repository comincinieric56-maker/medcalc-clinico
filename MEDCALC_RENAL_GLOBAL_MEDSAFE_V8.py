#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V8 · MEDSAFE NEW ZEALAND
================================================

Cuarta pasada regulatoria sobre los MED-ID que continúan pendientes después de
ISP Chile V7.

Fuente:
- Medsafe New Zealand
- Data Sheets profesionales publicadas en el portal oficial.

Seguridad:
- búsqueda por principio activo;
- identidad farmacológica estricta cuando el índice declara el ingrediente;
- comprobación adicional dentro del Data Sheet;
- exige una conducta renal explícita;
- no crea reglas numéricas automáticas;
- toda evidencia nueva queda CURRENT_REFERENCE.
"""

from __future__ import annotations

import csv
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rapidfuzz import fuzz

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import (
    exact_component_match,
    is_combo,
    norm,
    split_combo,
    strip_salts,
    renal_windows,
    clinical_action,
    read_csv,
    write_csv,
)

V7 = Path("generated_renal_global_v7")
OUT = Path("generated_renal_global_v8")
OUT.mkdir(parents=True, exist_ok=True)

PENDING = V7 / "renal_global_still_pending.csv"
MATRIX = V7 / "renal_global_matrix_1122.csv"

SEARCH_URL = "https://www.medsafe.govt.nz/medicines/infosearch.asp"
ALT_SEARCH_URL = "https://www.medsafe.govt.nz/DbSearch/InfoSearch"
DATASHEET_INDEX = "https://www.medsafe.govt.nz/profs/datasheet/datasheet.htm"

HEADERS = {
    "User-Agent": "MEDCALC-Renal-Medsafe-V8/1.0",
    "Accept-Language": "en-NZ,en;q=0.9",
}


def new_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get(s, url, *, params=None, timeout=45, tries=4):
    last = None
    for i in range(tries):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2.5 * (i + 1))
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            last = exc
            time.sleep(1.2 + 1.8 * i)
    if last:
        raise last
    return None


def clean_text(value):
    s = re.sub(r"<[^>]+>", " ", str(value or ""))
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def ingredient_parts(text):
    raw = str(text or "")
    parts = re.split(r"\s*(?:\+|;|,|/|\band\b)\s*", raw, flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def identity_from_declared(generic_name, ingredient_text):
    parts = ingredient_parts(ingredient_text)
    if not parts:
        return False
    try:
        return exact_component_match(generic_name, parts)
    except Exception:
        return False


def identity_from_document(generic_name, text):
    """Fallback estricto cuando el índice no declara ingrediente separado."""
    head = strip_salts(clean_text(text)[:12000])
    if not head:
        return False
    if is_combo(generic_name):
        parts = split_combo(generic_name)
        return bool(parts) and all(fuzz.partial_ratio(p, head) >= 92 for p in parts)
    target = strip_salts(generic_name)
    return bool(target) and fuzz.partial_ratio(target, head) >= 96


def parse_search_results(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    out = []

    for tr in soup.find_all("tr"):
        cells = [clean_text(x.get_text(" ", strip=True)) for x in tr.find_all(["td", "th"])]
        if not cells:
            continue
        blob = " ".join(cells).lower()
        if "active ingredient" in blob and ("trade" in blob or "medicine" in blob):
            continue

        links = []
        for a in tr.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            txt = clean_text(a.get_text(" ", strip=True))
            if href not in [x["url"] for x in links]:
                links.append({"url": href, "text": txt})

        ds_links = [
            x for x in links
            if (
                "datasheet" in x["url"].lower()
                or "data sheet" in x["text"].lower()
                or x["url"].lower().endswith(".pdf")
            )
        ]
        if not ds_links:
            continue

        product = cells[0] if cells else ""
        active = cells[1] if len(cells) > 1 else ""
        out.append({
            "product_name": product,
            "active_ingredient": active,
            "links": [x["url"] for x in ds_links],
            "source_page": page_url,
        })

    # Some result pages are list-based rather than table-based.
    if not out:
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            txt = clean_text(a.get_text(" ", strip=True))
            low = (txt + " " + href).lower()
            if "datasheet" in low or href.lower().endswith(".pdf"):
                out.append({
                    "product_name": txt,
                    "active_ingredient": "",
                    "links": [href],
                    "source_page": page_url,
                })

    dedup = []
    seen = set()
    for row in out:
        for link in row["links"]:
            if link in seen:
                continue
            seen.add(link)
            item = dict(row)
            item["links"] = [link]
            dedup.append(item)
    return dedup


def search_medsafe(s, generic_name):
    candidates = []
    terms = [str(generic_name or "").strip()]
    base = strip_salts(generic_name)
    if base and base not in terms:
        terms.append(base)

    for term in terms[:2]:
        if not term:
            continue
        for url in (SEARCH_URL, ALT_SEARCH_URL):
            try:
                r = get(
                    s,
                    url,
                    params={"txtMedicine": term, "Medicine": term},
                    timeout=40,
                    tries=3,
                )
            except Exception:
                r = None
            if not r:
                continue
            parsed = parse_search_results(r.text, r.url)
            candidates.extend(parsed)
            if parsed:
                break
        if candidates:
            break

    # Keep exact ingredient matches first.
    exact = [
        r for r in candidates
        if r.get("active_ingredient")
        and identity_from_declared(generic_name, r.get("active_ingredient"))
    ]
    unknown = [r for r in candidates if not r.get("active_ingredient")]
    return exact + unknown


def extract_pdf_text(data):
    if not data.startswith(b"%PDF"):
        return ""
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            parts.append(t)
        if len("\n".join(parts)) > 160000:
            break
    return "\n".join(parts)


def document_text(s, url):
    try:
        r = get(s, url, timeout=75, tries=3)
    except Exception:
        return "", ""
    if not r:
        return "", ""

    ctype = (r.headers.get("content-type") or "").lower()
    if "pdf" in ctype or r.content.startswith(b"%PDF"):
        return extract_pdf_text(r.content), r.url

    soup = BeautifulSoup(r.text, "html.parser")
    # If an HTML landing page links a PDF, prefer that.
    for a in soup.find_all("a", href=True):
        href = urljoin(r.url, a["href"])
        blob = (clean_text(a.get_text(" ", strip=True)) + " " + href).lower()
        if ".pdf" in blob and ("data" in blob or "sheet" in blob or "datasheet" in blob):
            try:
                rr = get(s, href, timeout=75, tries=2)
            except Exception:
                rr = None
            if rr and ("pdf" in (rr.headers.get("content-type") or "").lower() or rr.content.startswith(b"%PDF")):
                return extract_pdf_text(rr.content), rr.url

    return clean_text(soup.get_text(" ", strip=True)), r.url


def process_one(row):
    med_id = row.get("med_id") or ""
    name = row.get("generic_name") or ""
    s = new_session()

    try:
        candidates = search_medsafe(s, name)
    except Exception as exc:
        return {
            "med_id": med_id,
            "generic_name": name,
            "medsafe_status": "ERROR",
            "medsafe_error": repr(exc),
            "v8_resolution": "UNRESOLVED",
        }

    if not candidates:
        return {
            "med_id": med_id,
            "generic_name": name,
            "medsafe_status": "NO_EXACT_MEDSAFE_PRODUCT",
            "v8_resolution": "UNRESOLVED",
        }

    exact_seen = False
    first = None
    for cand in candidates[:16]:
        declared = cand.get("active_ingredient") or ""
        if declared and not identity_from_declared(name, declared):
            continue

        for link in cand.get("links") or []:
            text, final_url = document_text(s, link)
            if not text:
                continue
            if declared:
                identity_ok = True
            else:
                identity_ok = identity_from_document(name, text)
            if not identity_ok:
                continue

            exact_seen = True
            renal = renal_windows(text)
            ok, reason = clinical_action(renal)
            rec = {
                "med_id": med_id,
                "generic_name": name,
                "medsafe_status": "ACCEPT" if ok else "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
                "medsafe_reason": reason,
                "medsafe_product_name": cand.get("product_name"),
                "medsafe_active_ingredient": declared,
                "medsafe_source_url": final_url or link,
                "medsafe_renal_text": renal,
            }
            if first is None:
                first = rec
            if ok:
                rec["v8_resolution"] = "CURRENT_SINGLE_OFFICIAL_EXACT"
                rec["v8_primary_source"] = "MEDSAFE_NZ"
                rec["v8_reason"] = reason
                return rec

    if exact_seen and first:
        first["v8_resolution"] = "NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        first["v8_primary_source"] = ""
        first["v8_reason"] = first.get("medsafe_reason") or ""
        return first

    return {
        "med_id": med_id,
        "generic_name": name,
        "medsafe_status": "NO_EXACT_MEDSAFE_PRODUCT",
        "v8_resolution": "UNRESOLVED",
    }


def main():
    if not PENDING.exists() or not MATRIX.exists():
        raise SystemExit("Faltan outputs ISP V7; V8 debe ejecutarse después de V7.")

    pending = read_csv(PENDING)
    matrix = read_csv(MATRIX)
    if len(matrix) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V7 esperada 1122; encontrada {len(matrix)}")

    print(f"Pending entering Medsafe V8: {len(pending)}")
    results = []
    workers = 5

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, r): r for r in pending}
        done = 0
        for fut in as_completed(futures):
            base = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "med_id": base.get("med_id"),
                    "generic_name": base.get("generic_name"),
                    "medsafe_status": "ERROR",
                    "medsafe_error": repr(exc),
                    "v8_resolution": "UNRESOLVED",
                }
            results.append(rec)
            print(
                f"[{done:03d}/{len(pending)}] "
                f"{rec.get('med_id')} {rec.get('generic_name')} -> "
                f"{rec.get('v8_resolution')}"
            )
            if done % 25 == 0:
                write_csv(
                    OUT / "renal_global_v8_medsafe_partial.csv",
                    sorted(results, key=lambda x: x.get("med_id") or ""),
                )

    results.sort(key=lambda x: x.get("med_id") or "")
    write_csv(OUT / "renal_global_v8_medsafe_results.csv", results)

    by_mid = {r.get("med_id"): r for r in results}
    final = []
    for old in matrix:
        row = dict(old)
        mid = old.get("med_id")
        nr = by_mid.get(mid)
        prior_resolved = str(old.get("global_current_resolved_v7")).lower() == "true"
        if nr:
            row.update(nr)
            if nr.get("v8_resolution") == "CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v8"] = nr["v8_resolution"]
                row["global_primary_source_v8"] = "MEDSAFE_NZ"
                row["global_current_resolved_v8"] = True
            else:
                row["global_resolution_v8"] = old.get("global_resolution_v7") or nr.get("v8_resolution")
                row["global_primary_source_v8"] = old.get("global_primary_source_v7") or ""
                row["global_current_resolved_v8"] = prior_resolved
        else:
            row["global_resolution_v8"] = old.get("global_resolution_v7")
            row["global_primary_source_v8"] = old.get("global_primary_source_v7")
            row["global_current_resolved_v8"] = prior_resolved
        final.append(row)

    if len(final) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V8 !=1122 ({len(final)})")

    write_csv(OUT / "renal_global_matrix_1122.csv", final)
    resolved = [
        r for r in final
        if r.get("global_current_resolved_v8") is True
        or str(r.get("global_current_resolved_v8")).lower() == "true"
    ]
    still = [r for r in final if r not in resolved]
    write_csv(OUT / "renal_global_resolved.csv", resolved)
    write_csv(OUT / "renal_global_still_pending.csv", still)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": 1122,
        "v7_pending_input": len(pending),
        "medsafe_accept": sum(r.get("medsafe_status") == "ACCEPT" for r in results),
        "medsafe_exact_without_explicit_action": sum(
            r.get("medsafe_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
            for r in results
        ),
        "medsafe_no_exact_product": sum(
            r.get("medsafe_status") == "NO_EXACT_MEDSAFE_PRODUCT"
            for r in results
        ),
        "medsafe_errors": sum(r.get("medsafe_status") == "ERROR" for r in results),
        "global_current_resolved_v8": len(resolved),
        "global_still_pending_v8": len(still),
        "total_check": len(resolved) + len(still),
        "source": {
            "name": "Medsafe New Zealand",
            "role": "official medicine Data Sheets",
            "search": SEARCH_URL,
        },
        "safety": {
            "new_numeric_automatic_rules_created": False,
            "explicit_renal_action_required": True,
            "exact_drug_identity_required": True,
            "combination_integrity_required": True,
            "absence_of_explicit_recommendation_is_not_no_adjustment": True,
        },
    }
    (OUT / "renal_global_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
