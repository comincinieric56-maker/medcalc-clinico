#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V7 · ISP CHILE
=====================================

Tercera pasada regulatoria sobre los MED-ID que siguen pendientes después de
AEMPS/CIMA V6.

Fuente:
- Instituto de Salud Pública de Chile (ISP)
- Folleto de información al profesional / Resumen de características del producto

Política clínica:
- El índice del ISP solo confirma identidad/producto.
- Para cerrar una ficha renal se exige recuperar el documento profesional
  asociado y encontrar una conducta renal explícita.
- Una mención aislada de riñón/creatinina no basta.
- No se crean reglas numéricas automáticas.
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

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import exact_component_match, write_csv, read_csv
from MEDCALC_RENAL_GLOBAL_CIMA_V6 import (
    clinical_action_es,
    renal_windows_es,
    clean_text,
)

V6 = Path("generated_renal_global_v6")
OUT = Path("generated_renal_global_v7")
OUT.mkdir(parents=True, exist_ok=True)

PENDING = V6 / "renal_global_still_pending.csv"
MATRIX = V6 / "renal_global_matrix_1122.csv"

ISP_BASE = "https://www.ispch.cl/anamed/medicamentos/informacion-al-profesional/folletos-informacion/"
ISP_FALLBACK = "https://www.ispch.cl/informacion-al-profesional-v2/"

HEADERS = {
    "User-Agent": "MEDCALC-Renal-ISP-V7/1.0",
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.6",
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


def parse_year(text):
    m = re.search(r"\b(20\d{2})\b", str(text or ""))
    return int(m.group(1)) if m else 0


def parse_isp_page(html, page_url):
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) < 3:
            continue
        cells = [clean_text(td.get_text(" ", strip=True)) for td in tds]
        # Skip header.
        blob = " ".join(cells[:5]).lower()
        if "principio activo" in blob and ("registro" in blob or "nombre" in blob):
            continue

        name = cells[0] if len(cells) > 0 else ""
        reg = cells[1] if len(cells) > 1 else ""
        active = cells[2] if len(cells) > 2 else ""
        matter = cells[3] if len(cells) > 3 else ""
        year_text = cells[4] if len(cells) > 4 else ""
        if not name or not active:
            continue

        links = []
        for a in tr.find_all("a", href=True):
            href = urljoin(page_url, a.get("href"))
            if href not in links:
                links.append(href)

        out.append({
            "product_name": name,
            "registro": reg,
            "active_ingredient": active,
            "matter": matter,
            "year": parse_year(year_text or name),
            "links": links,
            "index_url": page_url,
        })
    return out


def crawl_index():
    s = new_session()
    rows = []
    seen = set()
    empty = 0

    # Prefer current canonical path; fall back to legacy WordPress path.
    for base in (ISP_BASE, ISP_FALLBACK):
        rows.clear()
        seen.clear()
        empty = 0
        for page in range(1, 380):
            url = base if page == 1 else urljoin(base, f"page/{page}/")
            try:
                r = get(s, url, timeout=40, tries=3)
            except Exception:
                r = None
            if not r:
                empty += 1
                if empty >= 3 and page > 5:
                    break
                continue
            parsed = parse_isp_page(r.text, r.url)
            if not parsed:
                empty += 1
                if empty >= 3 and page > 5:
                    break
                continue
            empty = 0
            for row in parsed:
                key = (row["registro"], row["product_name"], row["year"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        if rows:
            return rows
    return []


def active_components(text):
    parts = re.split(r"\s*(?://|/|\+|\by\b)\s*", str(text or ""), flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def exact_isp_identity(generic_name, active_text):
    comps = active_components(active_text)
    if not comps:
        return False
    try:
        return exact_component_match(generic_name, comps)
    except Exception:
        return False


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
        if len("\n".join(parts)) > 120000:
            break
    return "\n".join(parts)


def candidate_document_links(s, row):
    links = list(row.get("links") or [])
    expanded = []

    for link in links[:6]:
        low = link.lower()
        if any(low.endswith(ext) or f"{ext}?" in low for ext in (".pdf", ".doc", ".docx")):
            expanded.append(link)
            continue
        try:
            r = get(s, link, timeout=40, tries=2)
        except Exception:
            continue
        if not r:
            continue
        ctype = (r.headers.get("content-type") or "").lower()
        if "application/pdf" in ctype or r.content.startswith(b"%PDF"):
            expanded.append(link)
            continue
        if "html" not in ctype and not r.text:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = urljoin(r.url, a.get("href"))
            blob = (clean_text(a.get_text(" ", strip=True)) + " " + href).lower()
            if (
                ".pdf" in blob
                or "folleto" in blob
                or "profesional" in blob
                or "ficha tecnica" in blob
                or "ficha técnica" in blob
                or "resumen de caracteristicas" in blob
                or "resumen de características" in blob
            ):
                if href not in expanded:
                    expanded.append(href)

    # Deduplicate preserving order.
    out = []
    for x in expanded:
        if x not in out:
            out.append(x)
    return out[:12]


def document_text(s, url):
    try:
        r = get(s, url, timeout=70, tries=3)
    except Exception:
        return ""
    if not r:
        return ""
    ctype = (r.headers.get("content-type") or "").lower()
    if "pdf" in ctype or r.content.startswith(b"%PDF"):
        return extract_pdf_text(r.content)
    if "html" in ctype or "<html" in r.text[:500].lower():
        soup = BeautifulSoup(r.text, "html.parser")
        return clean_text(soup.get_text(" ", strip=True))
    return ""


def process_one(row, isp_rows):
    med_id = row.get("med_id") or ""
    name = row.get("generic_name") or ""
    s = new_session()

    candidates = [r for r in isp_rows if exact_isp_identity(name, r.get("active_ingredient"))]
    candidates.sort(key=lambda r: (r.get("year") or 0, r.get("registro") or ""), reverse=True)

    if not candidates:
        return {
            "med_id": med_id,
            "generic_name": name,
            "isp_status": "NO_EXACT_ISP_PRODUCT",
            "v7_resolution": "UNRESOLVED",
        }

    first = candidates[0]
    exact_without_doc = False

    for cand in candidates[:10]:
        links = candidate_document_links(s, cand)
        if not links:
            exact_without_doc = True
            continue
        for link in links:
            text = document_text(s, link)
            if not text:
                continue
            renal = renal_windows_es(text)
            ok, reason = clinical_action_es(renal)
            if ok:
                return {
                    "med_id": med_id,
                    "generic_name": name,
                    "isp_status": "ACCEPT",
                    "isp_reason": reason.replace("CIMA_", "ISP_"),
                    "isp_product_name": cand.get("product_name"),
                    "isp_registro": cand.get("registro"),
                    "isp_active_ingredient": cand.get("active_ingredient"),
                    "isp_year": cand.get("year"),
                    "isp_index_url": cand.get("index_url"),
                    "isp_document_url": link,
                    "isp_renal_text": renal,
                    "v7_resolution": "CURRENT_SINGLE_OFFICIAL_EXACT",
                    "v7_primary_source": "ISP_CHILE",
                    "v7_reason": reason.replace("CIMA_", "ISP_"),
                }
            exact_without_doc = True

    return {
        "med_id": med_id,
        "generic_name": name,
        "isp_status": (
            "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
            if exact_without_doc
            else "EXACT_PRODUCT_NO_RETRIEVABLE_PROFESSIONAL_DOCUMENT"
        ),
        "isp_reason": "",
        "isp_product_name": first.get("product_name"),
        "isp_registro": first.get("registro"),
        "isp_active_ingredient": first.get("active_ingredient"),
        "isp_year": first.get("year"),
        "isp_index_url": first.get("index_url"),
        "isp_document_url": "",
        "isp_renal_text": "",
        "v7_resolution": "UNRESOLVED",
        "v7_primary_source": "",
        "v7_reason": "",
    }


def main():
    if not PENDING.exists() or not MATRIX.exists():
        raise SystemExit("Faltan outputs CIMA V6; ISP V7 debe ejecutarse después de V6.")

    pending = read_csv(PENDING)
    matrix = read_csv(MATRIX)
    if len(matrix) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V6 esperada 1122; encontrada {len(matrix)}")

    isp_rows = crawl_index()
    print(f"ISP index rows: {len(isp_rows)}")
    if not isp_rows:
        raise SystemExit("No se pudo recuperar el índice oficial ISP Chile.")

    # Persist source index for auditability.
    audit_rows = []
    for r in isp_rows:
        x = dict(r)
        x["links"] = " | ".join(r.get("links") or [])
        audit_rows.append(x)
    write_csv(OUT / "isp_professional_leaflet_index.csv", audit_rows)

    results = []
    workers = 5
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, r, isp_rows): r for r in pending}
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
                    "isp_status": "ERROR",
                    "isp_error": repr(exc),
                    "v7_resolution": "UNRESOLVED",
                }
            results.append(rec)
            print(
                f"[{done:03d}/{len(pending)}] {rec.get('med_id')} "
                f"{rec.get('generic_name')} -> {rec.get('isp_status')}"
            )
            if done % 25 == 0:
                write_csv(
                    OUT / "renal_global_v7_isp_partial.csv",
                    sorted(results, key=lambda x: x.get("med_id") or ""),
                )

    results.sort(key=lambda x: x.get("med_id") or "")
    write_csv(OUT / "renal_global_v7_isp_results.csv", results)

    by_mid = {r.get("med_id"): r for r in results}
    final = []
    for old in matrix:
        row = dict(old)
        nr = by_mid.get(old.get("med_id"))
        prior_resolved = str(old.get("global_current_resolved_v6")).lower() == "true"
        if nr:
            row.update(nr)
            if nr.get("v7_resolution") == "CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v7"] = nr["v7_resolution"]
                row["global_primary_source_v7"] = "ISP_CHILE"
                row["global_current_resolved_v7"] = True
            else:
                row["global_resolution_v7"] = old.get("global_resolution_v6")
                row["global_primary_source_v7"] = old.get("global_primary_source_v6")
                row["global_current_resolved_v7"] = prior_resolved
        else:
            row["global_resolution_v7"] = old.get("global_resolution_v6")
            row["global_primary_source_v7"] = old.get("global_primary_source_v6")
            row["global_current_resolved_v7"] = prior_resolved
        final.append(row)

    if len(final) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V7 !=1122 ({len(final)})")

    write_csv(OUT / "renal_global_matrix_1122.csv", final)
    resolved = [
        r for r in final
        if r.get("global_current_resolved_v7") is True
        or str(r.get("global_current_resolved_v7")).lower() == "true"
    ]
    still = [r for r in final if r not in resolved]
    write_csv(OUT / "renal_global_resolved.csv", resolved)
    write_csv(OUT / "renal_global_still_pending.csv", still)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": 1122,
        "v6_pending_input": len(pending),
        "isp_index_rows": len(isp_rows),
        "isp_accept": sum(r.get("isp_status") == "ACCEPT" for r in results),
        "isp_exact_without_explicit_action": sum(
            r.get("isp_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
            for r in results
        ),
        "isp_exact_without_retrievable_document": sum(
            r.get("isp_status") == "EXACT_PRODUCT_NO_RETRIEVABLE_PROFESSIONAL_DOCUMENT"
            for r in results
        ),
        "isp_no_exact_product": sum(
            r.get("isp_status") == "NO_EXACT_ISP_PRODUCT" for r in results
        ),
        "isp_errors": sum(r.get("isp_status") == "ERROR" for r in results),
        "global_current_resolved_v7": len(resolved),
        "global_still_pending_v7": len(still),
        "total_check": len(resolved) + len(still),
        "source": {
            "name": "Instituto de Salud Pública de Chile",
            "dataset": "Folleto de información al profesional / Resumen de características del producto",
            "url": ISP_BASE,
        },
        "safety": {
            "index_match_alone_can_close": False,
            "professional_document_required": True,
            "explicit_renal_action_required": True,
            "new_numeric_automatic_rules_created": False,
        },
    }
    (OUT / "renal_global_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
