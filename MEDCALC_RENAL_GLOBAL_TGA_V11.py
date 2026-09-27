#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V11 · TGA AUSTRALIA
==========================================

Séptima pasada regulatoria sobre los MED-ID pendientes después de MHRA V10.

Fuente:
- Therapeutic Goods Administration (TGA)
- Australian Register of Therapeutic Goods (ARTG)
- Product Information (PI) aprobado por TGA.

Seguridad:
- identidad farmacológica estricta;
- integridad de combinaciones;
- PI profesional obligatorio;
- acción renal explícita obligatoria;
- ninguna regla numérica automática nueva.
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

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import (
    exact_component_match,
    strip_salts,
    renal_windows,
    clinical_action,
    read_csv,
    write_csv,
)

V10=Path("generated_renal_global_v10")
OUT=Path("generated_renal_global_v11")
OUT.mkdir(parents=True,exist_ok=True)

PENDING=V10/"renal_global_still_pending.csv"
MATRIX=V10/"renal_global_matrix_1122.csv"

BASE="https://www.tga.gov.au"
HEADERS={
    "User-Agent":"MEDCALC-Renal-TGA-V11/1.0",
    "Accept-Language":"en-AU,en;q=0.9",
}
ARTG_RE=re.compile(r"/resources/artg/(\d+)")


def new_session():
    s=requests.Session()
    s.headers.update(HEADERS)
    return s


def get(s,url,*,params=None,timeout=45,tries=4):
    last=None
    for i in range(tries):
        try:
            r=s.get(url,params=params,timeout=timeout)
            if r.status_code==404: return None
            if r.status_code==429:
                time.sleep(2.5*(i+1)); continue
            r.raise_for_status()
            return r
        except Exception as exc:
            last=exc; time.sleep(1.2+1.8*i)
    if last: raise last
    return None


def clean(v):
    return re.sub(r"\s+"," ",str(v or "")).strip()


def candidate_search_urls(term):
    q=quote(term)
    return [
        f"{BASE}/search?search={q}",
        f"{BASE}/search?keywords={q}",
        f"{BASE}/search/site/{q}",
    ]


def search_artg(s,name):
    terms=[str(name or "").strip()]
    base=strip_salts(name)
    if base and base not in terms: terms.append(base)

    out=[]
    seen=set()
    for term in terms[:2]:
        for url in candidate_search_urls(term):
            try: r=get(s,url,timeout=40,tries=2)
            except Exception: r=None
            if not r: continue
            soup=BeautifulSoup(r.text,"html.parser")
            for a in soup.find_all("a",href=True):
                href=urljoin(r.url,a["href"])
                m=ARTG_RE.search(href)
                if not m: continue
                artg=m.group(1)
                if artg in seen: continue
                seen.add(artg)
                out.append(href.split("#",1)[0].split("?",1)[0])
            if out: break
        if out: break
    return out[:40]


def parse_ingredients_from_page(soup):
    text=soup.get_text("\n",strip=True)
    lines=[clean(x) for x in text.splitlines() if clean(x)]
    vals=[]
    try:
        i=next(i for i,x in enumerate(lines) if x.lower()=="ingredients")
    except StopIteration:
        return vals
    stop_words={
        "licence category","licence status","summary","product information",
        "consumer medicine information","related information","sponsor",
    }
    for x in lines[i+1:i+16]:
        if x.lower() in stop_words: break
        if len(x)>160: continue
        vals.append(x)
    return vals


def product_info_links(page_url,soup):
    out=[]
    for a in soup.find_all("a",href=True):
        href=urljoin(page_url,a["href"])
        txt=clean(a.get_text(" ",strip=True))
        blob=(txt+" "+href).lower()
        if "product information" in blob and (".pdf" in blob or "download" in blob or "/sites/" in blob):
            if href not in out: out.append(href)
    return out[:10]


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


def document_text(s,url):
    try: r=get(s,url,timeout=75,tries=3)
    except Exception: return "",""
    if not r: return "",""
    ctype=(r.headers.get("content-type") or "").lower()
    if "pdf" in ctype or r.content.startswith(b"%PDF"):
        return extract_pdf(r.content),r.url
    soup=BeautifulSoup(r.text,"html.parser")
    return clean(soup.get_text(" ",strip=True)),r.url


def process_one(row):
    med_id=row.get("med_id") or ""
    name=row.get("generic_name") or ""
    s=new_session()

    try: pages=search_artg(s,name)
    except Exception as exc:
        return {"med_id":med_id,"generic_name":name,"tga_status":"ERROR","tga_error":repr(exc),"v11_resolution":"UNRESOLVED"}

    if not pages:
        return {"med_id":med_id,"generic_name":name,"tga_status":"NO_ARTG_RESULT","v11_resolution":"UNRESOLVED"}

    first=None
    exact_seen=False
    for page in pages[:24]:
        try: r=get(s,page,timeout=45,tries=2)
        except Exception: r=None
        if not r: continue
        soup=BeautifulSoup(r.text,"html.parser")
        ingredients=parse_ingredients_from_page(soup)
        if not ingredients:
            continue
        try:
            if not exact_component_match(name,ingredients):
                continue
        except Exception:
            continue

        exact_seen=True
        artg=(ARTG_RE.search(r.url).group(1) if ARTG_RE.search(r.url) else "")
        links=product_info_links(r.url,soup)
        for link in links:
            text,final=document_text(s,link)
            if not text: continue
            renal=renal_windows(text)
            ok,reason=clinical_action(renal)
            rec={
                "med_id":med_id,
                "generic_name":name,
                "tga_status":"ACCEPT" if ok else "EXACT_ARTG_NO_EXPLICIT_RENAL_ACTION",
                "tga_reason":reason,
                "tga_artg_id":artg,
                "tga_artg_page":r.url,
                "tga_pi_url":final or link,
                "tga_ingredients":" | ".join(ingredients),
                "tga_renal_text":renal,
            }
            if first is None: first=rec
            if ok:
                rec["v11_resolution"]="CURRENT_SINGLE_OFFICIAL_EXACT"
                rec["v11_primary_source"]="TGA_AU"
                rec["v11_reason"]=reason
                return rec

    if exact_seen and first:
        first["v11_resolution"]="NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        first["v11_primary_source"]=""
        first["v11_reason"]=first.get("tga_reason") or ""
        return first

    if exact_seen:
        return {"med_id":med_id,"generic_name":name,"tga_status":"EXACT_ARTG_PI_UNAVAILABLE","v11_resolution":"UNRESOLVED"}
    return {"med_id":med_id,"generic_name":name,"tga_status":"NO_EXACT_ARTG_PRODUCT","v11_resolution":"UNRESOLVED"}


def main():
    if not PENDING.exists() or not MATRIX.exists():
        raise SystemExit("Faltan outputs MHRA V10; V11 debe ejecutarse después de V10.")
    pending=read_csv(PENDING)
    matrix=read_csv(MATRIX)
    if len(matrix)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V10 esperada 1122; encontrada {len(matrix)}")

    print(f"Pending entering TGA V11: {len(pending)}")
    results=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures={ex.submit(process_one,r):r for r in pending}
        done=0
        for fut in as_completed(futures):
            base=futures[fut]; done+=1
            try: rec=fut.result()
            except Exception as exc:
                rec={"med_id":base.get("med_id"),"generic_name":base.get("generic_name"),"tga_status":"ERROR","tga_error":repr(exc),"v11_resolution":"UNRESOLVED"}
            results.append(rec)
            print(f"[{done:03d}/{len(pending)}] {rec.get('med_id')} {rec.get('generic_name')} -> {rec.get('v11_resolution')}")
            if done%25==0:
                write_csv(OUT/"renal_global_v11_tga_partial.csv",sorted(results,key=lambda x:x.get("med_id") or ""))

    results.sort(key=lambda x:x.get("med_id") or "")
    write_csv(OUT/"renal_global_v11_tga_results.csv",results)
    by_mid={r.get("med_id"):r for r in results}
    final=[]
    for old in matrix:
        row=dict(old); mid=old.get("med_id"); nr=by_mid.get(mid)
        prior=str(old.get("global_current_resolved_v10")).lower()=="true"
        if nr:
            row.update(nr)
            if nr.get("v11_resolution")=="CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v11"]=nr["v11_resolution"]
                row["global_primary_source_v11"]="TGA_AU"
                row["global_current_resolved_v11"]=True
            else:
                row["global_resolution_v11"]=old.get("global_resolution_v10") or nr.get("v11_resolution")
                row["global_primary_source_v11"]=old.get("global_primary_source_v10") or ""
                row["global_current_resolved_v11"]=prior
        else:
            row["global_resolution_v11"]=old.get("global_resolution_v10")
            row["global_primary_source_v11"]=old.get("global_primary_source_v10")
            row["global_current_resolved_v11"]=prior
        final.append(row)

    if len(final)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V11 !=1122 ({len(final)})")

    write_csv(OUT/"renal_global_matrix_1122.csv",final)
    resolved=[r for r in final if r.get("global_current_resolved_v11") is True or str(r.get("global_current_resolved_v11")).lower()=="true"]
    still=[r for r in final if r not in resolved]
    write_csv(OUT/"renal_global_resolved.csv",resolved)
    write_csv(OUT/"renal_global_still_pending.csv",still)

    summary={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "catalog_rows":1122,
        "v10_pending_input":len(pending),
        "tga_accept":sum(r.get("tga_status")=="ACCEPT" for r in results),
        "tga_exact_without_explicit_action":sum(r.get("tga_status")=="EXACT_ARTG_NO_EXPLICIT_RENAL_ACTION" for r in results),
        "tga_no_exact_product":sum(r.get("tga_status") in {"NO_ARTG_RESULT","NO_EXACT_ARTG_PRODUCT"} for r in results),
        "tga_errors":sum(r.get("tga_status")=="ERROR" for r in results),
        "global_current_resolved_v11":len(resolved),
        "global_still_pending_v11":len(still),
        "total_check":len(resolved)+len(still),
        "source":{"name":"Therapeutic Goods Administration · ARTG/Product Information","url":BASE,"role":"official Australian Product Information"},
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
