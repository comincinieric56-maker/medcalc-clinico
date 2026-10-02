#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V10 · MHRA UNITED KINGDOM
================================================

Sexta pasada regulatoria sobre los MED-ID pendientes después de ANSM V9.

Fuente:
- MHRA Products (UK)
- Summary of Product Characteristics (SmPC)

Política:
- búsqueda por principio activo;
- identidad estricta en página/documento;
- recomendación renal explícita obligatoria;
- no crea reglas numéricas automáticas;
- toda evidencia nueva queda CURRENT_REFERENCE.
"""

from __future__ import annotations

import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from rapidfuzz import fuzz

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import (
    is_combo,
    split_combo,
    strip_salts,
    renal_windows,
    clinical_action,
    read_csv,
    write_csv,
)

V9=Path("generated_renal_global_v9")
OUT=Path("generated_renal_global_v10")
OUT.mkdir(parents=True,exist_ok=True)

PENDING=V9/"renal_global_still_pending.csv"
MATRIX=V9/"renal_global_matrix_1122.csv"

BASE="https://products.mhra.gov.uk"
HEADERS={
    "User-Agent":"MEDCALC-Renal-MHRA-V10/1.0",
    "Accept-Language":"en-GB,en;q=0.9",
}


def new_session():
    s=requests.Session()
    s.headers.update(HEADERS)
    return s


def get(s,url,*,params=None,timeout=45,tries=4):
    last=None
    for i in range(tries):
        try:
            r=s.get(url,params=params,timeout=timeout)
            if r.status_code==404:
                return None
            if r.status_code==429:
                time.sleep(2.5*(i+1)); continue
            r.raise_for_status()
            return r
        except Exception as exc:
            last=exc
            time.sleep(1.2+1.8*i)
    if last: raise last
    return None


def clean_text(v):
    s=re.sub(r"<[^>]+>"," ",str(v or ""))
    s=s.replace("&nbsp;"," ").replace("&amp;","&")
    return re.sub(r"\s+"," ",s).strip()


def identity_ok(name,text):
    hay=strip_salts(clean_text(text)[:20000])
    if not hay: return False
    if is_combo(name):
        parts=split_combo(name)
        return bool(parts) and all(fuzz.partial_ratio(p,hay)>=92 for p in parts)
    target=strip_salts(name)
    return bool(target) and fuzz.partial_ratio(target,hay)>=96


def extract_pdf(data):
    if not data.startswith(b"%PDF"): return ""
    reader=PdfReader(io.BytesIO(data))
    parts=[]
    for p in reader.pages:
        try: t=p.extract_text() or ""
        except Exception: t=""
        if t: parts.append(t)
        if len("\n".join(parts))>170000: break
    return "\n".join(parts)


def search_pages(s,name):
    terms=[str(name or "").strip()]
    base=strip_salts(name)
    if base and base not in terms: terms.append(base)
    urls=[]
    for term in terms[:2]:
        if not term: continue
        for page in range(1,5):
            try:
                r=get(s,f"{BASE}/search/",params={"search":term,"page":page},timeout=40,tries=3)
            except Exception:
                r=None
            if not r: continue
            soup=BeautifulSoup(r.text,"html.parser")
            found=0
            for a in soup.find_all("a",href=True):
                href=urljoin(r.url,a["href"])
                txt=clean_text(a.get_text(" ",strip=True))
                low=(txt+" "+href).lower()
                if "products.mhra.gov.uk" not in href:
                    continue
                if (
                    "/product" in href.lower()
                    or "/products/" in href.lower()
                    or "spc" in low
                    or "summary of product characteristics" in low
                ):
                    if href not in urls:
                        urls.append(href); found+=1
            if found==0 and page>1: break
        if urls: break
    return urls[:40]


def expand_spc_links(s,url):
    try:
        r=get(s,url,timeout=45,tries=3)
    except Exception:
        return []
    if not r: return []
    soup=BeautifulSoup(r.text,"html.parser")
    links=[]
    # Current MHRA site may expose document links as PDFs or routes with doc=SPC.
    for a in soup.find_all("a",href=True):
        href=urljoin(r.url,a["href"])
        txt=clean_text(a.get_text(" ",strip=True))
        low=(txt+" "+href).lower()
        if (
            "spc" in low
            or "summary of product characteristics" in low
            or "summary-of-product-characteristics" in low
        ):
            if href not in links: links.append(href)
    # If the search result itself is an SPC page, keep it.
    page_text=clean_text(soup.get_text(" ",strip=True))
    if "summary of product characteristics" in page_text.lower() and r.url not in links:
        links.append(r.url)
    return links[:12]


def document_text(s,url):
    try:
        r=get(s,url,timeout=70,tries=3)
    except Exception:
        return "",""
    if not r: return "",""
    ctype=(r.headers.get("content-type") or "").lower()
    if "pdf" in ctype or r.content.startswith(b"%PDF"):
        return extract_pdf(r.content),r.url
    soup=BeautifulSoup(r.text,"html.parser")
    # Follow direct PDF within document page when present.
    for a in soup.find_all("a",href=True):
        href=urljoin(r.url,a["href"])
        blob=(clean_text(a.get_text(" ",strip=True))+" "+href).lower()
        if ".pdf" in blob and ("spc" in blob or "product" in blob or "summary" in blob):
            try:
                rr=get(s,href,timeout=70,tries=2)
            except Exception:
                rr=None
            if rr and ("pdf" in (rr.headers.get("content-type") or "").lower() or rr.content.startswith(b"%PDF")):
                return extract_pdf(rr.content),rr.url
    return clean_text(soup.get_text(" ",strip=True)),r.url


def process_one(row):
    med_id=row.get("med_id") or ""
    name=row.get("generic_name") or ""
    s=new_session()
    try:
        pages=search_pages(s,name)
    except Exception as exc:
        return {"med_id":med_id,"generic_name":name,"mhra_status":"ERROR","mhra_error":repr(exc),"v10_resolution":"UNRESOLVED"}

    if not pages:
        return {"med_id":med_id,"generic_name":name,"mhra_status":"NO_MHRA_RESULT","v10_resolution":"UNRESOLVED"}

    exact_seen=False
    first=None
    for page in pages[:24]:
        try:
            p=get(s,page,timeout=40,tries=2)
        except Exception:
            p=None
        if not p: continue
        ptext=clean_text(BeautifulSoup(p.text,"html.parser").get_text(" ",strip=True))
        if not identity_ok(name,ptext):
            continue
        exact_seen=True
        links=expand_spc_links(s,page)
        if not links:
            links=[page]
        for link in links:
            text,final_url=document_text(s,link)
            if not text or not identity_ok(name,text):
                continue
            renal=renal_windows(text)
            ok,reason=clinical_action(renal)
            rec={
                "med_id":med_id,
                "generic_name":name,
                "mhra_status":"ACCEPT" if ok else "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
                "mhra_reason":reason,
                "mhra_product_page":page,
                "mhra_spc_url":final_url or link,
                "mhra_renal_text":renal,
            }
            if first is None: first=rec
            if ok:
                rec["v10_resolution"]="CURRENT_SINGLE_OFFICIAL_EXACT"
                rec["v10_primary_source"]="MHRA_UK"
                rec["v10_reason"]=reason
                return rec

    if exact_seen and first:
        first["v10_resolution"]="NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        first["v10_primary_source"]=""
        first["v10_reason"]=first.get("mhra_reason") or ""
        return first
    return {"med_id":med_id,"generic_name":name,"mhra_status":"NO_EXACT_MHRA_PRODUCT","v10_resolution":"UNRESOLVED"}


def main():
    if not PENDING.exists() or not MATRIX.exists():
        raise SystemExit("Faltan outputs ANSM V9; V10 debe ejecutarse después de V9.")
    pending=read_csv(PENDING)
    matrix=read_csv(MATRIX)
    if len(matrix)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V9 esperada 1122; encontrada {len(matrix)}")

    print(f"Pending entering MHRA V10: {len(pending)}")
    results=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures={ex.submit(process_one,r):r for r in pending}
        done=0
        for fut in as_completed(futures):
            base=futures[fut]; done+=1
            try: rec=fut.result()
            except Exception as exc:
                rec={"med_id":base.get("med_id"),"generic_name":base.get("generic_name"),"mhra_status":"ERROR","mhra_error":repr(exc),"v10_resolution":"UNRESOLVED"}
            results.append(rec)
            print(f"[{done:03d}/{len(pending)}] {rec.get('med_id')} {rec.get('generic_name')} -> {rec.get('v10_resolution')}")
            if done%25==0:
                write_csv(OUT/"renal_global_v10_mhra_partial.csv",sorted(results,key=lambda x:x.get("med_id") or ""))

    results.sort(key=lambda x:x.get("med_id") or "")
    write_csv(OUT/"renal_global_v10_mhra_results.csv",results)
    by_mid={r.get("med_id"):r for r in results}
    final=[]
    for old in matrix:
        row=dict(old); mid=old.get("med_id"); nr=by_mid.get(mid)
        prior=str(old.get("global_current_resolved_v9")).lower()=="true"
        if nr:
            row.update(nr)
            if nr.get("v10_resolution")=="CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v10"]=nr["v10_resolution"]
                row["global_primary_source_v10"]="MHRA_UK"
                row["global_current_resolved_v10"]=True
            else:
                row["global_resolution_v10"]=old.get("global_resolution_v9") or nr.get("v10_resolution")
                row["global_primary_source_v10"]=old.get("global_primary_source_v9") or ""
                row["global_current_resolved_v10"]=prior
        else:
            row["global_resolution_v10"]=old.get("global_resolution_v9")
            row["global_primary_source_v10"]=old.get("global_primary_source_v9")
            row["global_current_resolved_v10"]=prior
        final.append(row)

    if len(final)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V10 !=1122 ({len(final)})")

    write_csv(OUT/"renal_global_matrix_1122.csv",final)
    resolved=[r for r in final if r.get("global_current_resolved_v10") is True or str(r.get("global_current_resolved_v10")).lower()=="true"]
    still=[r for r in final if r not in resolved]
    write_csv(OUT/"renal_global_resolved.csv",resolved)
    write_csv(OUT/"renal_global_still_pending.csv",still)

    summary={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "catalog_rows":1122,
        "v9_pending_input":len(pending),
        "mhra_accept":sum(r.get("mhra_status")=="ACCEPT" for r in results),
        "mhra_exact_without_explicit_action":sum(r.get("mhra_status")=="EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION" for r in results),
        "mhra_no_exact_product":sum(r.get("mhra_status") in {"NO_EXACT_MHRA_PRODUCT","NO_MHRA_RESULT"} for r in results),
        "mhra_errors":sum(r.get("mhra_status")=="ERROR" for r in results),
        "global_current_resolved_v10":len(resolved),
        "global_still_pending_v10":len(still),
        "total_check":len(resolved)+len(still),
        "source":{"name":"MHRA Products · United Kingdom","url":BASE,"role":"official SmPC"},
        "safety":{
            "new_numeric_automatic_rules_created":False,
            "explicit_renal_action_required":True,
            "exact_drug_identity_required":True,
            "combination_integrity_required":True,
            "absence_of_explicit_recommendation_is_not_no_adjustment":True,
        },
    }
    (OUT/"renal_global_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
