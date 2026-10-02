#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V9 · ANSM / BDPM FRANCE
===============================================

Quinta pasada regulatoria sobre los MED-ID que continúan pendientes después de
Medsafe V8.

Fuente oficial:
- Base de Données Publique des Médicaments (BDPM) · France
- CIS_bdpm.txt (specialités)
- CIS_COMPO_bdpm.txt (composition)
- RCP professionnel associé à chaque code CIS.

Sécurité:
- identité par composition BDPM;
- intégrité des associations;
- recommandation rénale explicite obligatoire;
- aucune règle numérique automatique créée;
- toute nouvelle preuve reste CURRENT_REFERENCE.
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

import requests
from bs4 import BeautifulSoup

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import (
    exact_component_match,
    read_csv,
    write_csv,
)

V8=Path("generated_renal_global_v8")
OUT=Path("generated_renal_global_v9")
OUT.mkdir(parents=True,exist_ok=True)

PENDING=V8/"renal_global_still_pending.csv"
MATRIX=V8/"renal_global_matrix_1122.csv"

BASE="https://base-donnees-publique.medicaments.gouv.fr"
CIS_URL=f"{BASE}/download/file/CIS_bdpm.txt"
COMPO_URL=f"{BASE}/download/file/CIS_COMPO_bdpm.txt"

HEADERS={
    "User-Agent":"MEDCALC-Renal-ANSM-V9/1.0",
    "Accept-Language":"fr-FR,fr;q=0.9,en;q=0.5",
}

RENAL_FR=re.compile(
    r"\b(insuffisance r[eé]nale|fonction r[eé]nale|d[eé]ficience r[eé]nale|"
    r"alt[eé]ration de la fonction r[eé]nale|clairance de la cr[eé]atinine|"
    r"cr[eé]atinine clearance|\bcrcl\b|\begfr\b|\bdfg\b|"
    r"d[eé]bit de filtration glom[eé]rulaire|h[eé]modialyse|dialyse|"
    r"insuffisance r[eé]nale terminale)\b",re.I
)
NO_ADJUST_FR=re.compile(
    r"\b(aucun ajustement (?:de la )?(?:dose|posologie) n['’]est (?:n[eé]cessaire|requis)|"
    r"aucune adaptation posologique n['’]est (?:n[eé]cessaire|requise)|"
    r"ne n[eé]cessite pas d['’]ajustement (?:de la )?(?:dose|posologie)|"
    r"pas d['’]ajustement (?:de la )?(?:dose|posologie) n[eé]cessaire|"
    r"aucune modification de dose n['’]est n[eé]cessaire)\b",re.I
)
NOT_REC_FR=re.compile(
    r"\b(non recommand[eé]|ne doit pas [eê]tre utilis[eé]|[eé]viter l['’]utilisation|"
    r"contre-indiqu[eé]|ne doit pas [eê]tre administr[eé])\b",re.I
)
THRESHOLD_FR=re.compile(
    r"(?:clairance de la cr[eé]atinine|crcl|egfr|dfg|d[eé]bit de filtration glom[eé]rulaire)"
    r"[^.;:\n]{0,180}(?:<|>|≤|≥|inf[eé]rieur|sup[eé]rieur|entre|\d)",re.I
)
DOSE_FR=re.compile(
    r"\b(dose|posologie|administrer|administration|r[eé]duire|r[eé]duction|"
    r"adapter|adaptation|intervalle|toutes les|une fois par jour|deux fois par jour|"
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml))\b",re.I
)
DIALYSIS_FR=re.compile(
    r"(?:administrer|dose|suppl[eé]ment|adapter)[^.;:\n]{0,220}(?:h[eé]modialyse|dialyse)|"
    r"(?:h[eé]modialyse|dialyse)[^.;:\n]{0,220}(?:administrer|dose|suppl[eé]ment|adapter)",
    re.I
)
ADJUST_FR=re.compile(
    r"\b(adaptation posologique|ajustement de la dose|adapter la dose|"
    r"r[eé]duire la dose|r[eé]duction de la dose|augmenter l['’]intervalle|"
    r"modifier la dose|modifier l['’]intervalle)\b",re.I
)


def new_session():
    s=requests.Session()
    s.headers.update(HEADERS)
    return s


def get(s,url,*,timeout=50,tries=4):
    last=None
    for i in range(tries):
        try:
            r=s.get(url,timeout=timeout)
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


def decode_bytes(data):
    for enc in ("utf-8-sig","cp1252","latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("latin-1",errors="replace")


def load_bdpm():
    s=new_session()
    rcis=get(s,CIS_URL,timeout=80)
    rcomp=get(s,COMPO_URL,timeout=80)
    if not rcis or not rcomp:
        raise RuntimeError("No fue posible descargar CIS_bdpm/CIS_COMPO_bdpm.")

    cis={}
    for row in csv.reader(io.StringIO(decode_bytes(rcis.content)),delimiter="\t"):
        if len(row)<2: continue
        code=row[0].strip()
        cis[code]={
            "cis":code,
            "name":row[1].strip() if len(row)>1 else "",
            "form":row[2].strip() if len(row)>2 else "",
            "route":row[3].strip() if len(row)>3 else "",
            "amm_status":row[4].strip() if len(row)>4 else "",
            "procedure":row[5].strip() if len(row)>5 else "",
            "market_status":row[6].strip() if len(row)>6 else "",
            "amm_date":row[7].strip() if len(row)>7 else "",
        }

    comps={}
    for row in csv.reader(io.StringIO(decode_bytes(rcomp.content)),delimiter="\t"):
        if len(row)<4: continue
        code=row[0].strip()
        substance=row[3].strip()
        nature=row[6].strip().upper() if len(row)>6 else ""
        if not code or not substance: continue
        # SA: substance active; ST/FT: therapeutic fraction in older exports.
        if nature and nature not in {"SA","ST","FT"}:
            continue
        comps.setdefault(code,[])
        if substance not in comps[code]:
            comps[code].append(substance)

    return cis,comps


def renal_windows_fr(text,before=650,after=2300):
    text=re.sub(r"\s+"," ",str(text or ""))
    wins=[]
    for m in RENAL_FR.finditer(text):
        lo=max(0,m.start()-before); hi=min(len(text),m.end()+after)
        w=text[lo:hi]
        if w not in wins: wins.append(w)
        if len(wins)>=10: break
    return "\n---\n".join(wins)[:15000]


def clinical_action_fr(text):
    text=str(text or "")
    if not RENAL_FR.search(text):
        return False,"ANSM_NO_RENAL_TEXT"
    if NO_ADJUST_FR.search(text):
        return True,"ANSM_NO_ADJUSTMENT_EXPLICIT"
    if NOT_REC_FR.search(text):
        if THRESHOLD_FR.search(text) or re.search(
            r"(grave|s[eé]v[eè]re|terminale)[^.;]{0,160}insuffisance r[eé]nale",text,re.I
        ):
            return True,"ANSM_RENAL_NOT_RECOMMENDED_EXPLICIT"
    if THRESHOLD_FR.search(text) and DOSE_FR.search(text):
        return True,"ANSM_RENAL_THRESHOLD_DOSING_EXPLICIT"
    if DIALYSIS_FR.search(text):
        return True,"ANSM_DIALYSIS_DOSING_EXPLICIT"
    for m in RENAL_FR.finditer(text):
        lo=max(0,m.start()-300); hi=min(len(text),m.end()+900)
        if ADJUST_FR.search(text[lo:hi]):
            return True,"ANSM_RENAL_DOSING_TEXT_EXPLICIT"
    return False,"ANSM_RENAL_MENTION_WITHOUT_EXPLICIT_DOSING_ACTION"


def rcp_text(s,cis):
    urls=[
        f"{BASE}/affichageDoc.php?specid={cis}&typedoc=R",
        f"{BASE}/medicament/{cis}/extrait",
    ]
    for url in urls:
        try:
            r=get(s,url,timeout=50,tries=3)
        except Exception:
            r=None
        if not r: continue
        soup=BeautifulSoup(r.text,"html.parser")
        text=" ".join(soup.stripped_strings)
        if "RÉSUMÉ DES CARACTÉRISTIQUES" in text.upper() or "RESUME DES CARACTERISTIQUES" in text.upper():
            return re.sub(r"\s+"," ",text).strip(),r.url
    return "",""


def rank_product(meta):
    market=str(meta.get("market_status") or "").upper()
    amm=str(meta.get("amm_status") or "").upper()
    score=0
    if "COMMERCIAL" in market: score+=3
    if "AUTORIS" in amm or "ACTIVE" in amm or "VALIDE" in amm: score+=2
    return (score,str(meta.get("amm_date") or ""),str(meta.get("cis") or ""))


def process_one(row,cis_map,comp_map):
    med_id=row.get("med_id") or ""
    name=row.get("generic_name") or ""
    candidates=[]
    for code,ingredients in comp_map.items():
        try:
            if exact_component_match(name,ingredients):
                candidates.append((cis_map.get(code,{ "cis":code }),ingredients))
        except Exception:
            continue

    if not candidates:
        return {
            "med_id":med_id,"generic_name":name,
            "ansm_status":"NO_EXACT_BDPM_PRODUCT",
            "v9_resolution":"UNRESOLVED",
        }

    candidates.sort(key=lambda x:rank_product(x[0]),reverse=True)
    s=new_session()
    first=None
    for meta,ingredients in candidates[:16]:
        code=meta.get("cis")
        text,url=rcp_text(s,code)
        if not text: continue
        renal=renal_windows_fr(text)
        ok,reason=clinical_action_fr(renal)
        rec={
            "med_id":med_id,
            "generic_name":name,
            "ansm_status":"ACCEPT" if ok else "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
            "ansm_reason":reason,
            "ansm_cis":code,
            "ansm_product_name":meta.get("name"),
            "ansm_ingredients":" | ".join(ingredients),
            "ansm_source_url":url,
            "ansm_renal_text":renal,
            "ansm_market_status":meta.get("market_status"),
            "ansm_amm_status":meta.get("amm_status"),
        }
        if first is None: first=rec
        if ok:
            rec["v9_resolution"]="CURRENT_SINGLE_OFFICIAL_EXACT"
            rec["v9_primary_source"]="ANSM_BDPM"
            rec["v9_reason"]=reason
            return rec

    if first:
        first["v9_resolution"]="NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        first["v9_primary_source"]=""
        first["v9_reason"]=first.get("ansm_reason") or ""
        return first

    return {
        "med_id":med_id,"generic_name":name,
        "ansm_status":"EXACT_PRODUCT_RCP_UNAVAILABLE",
        "v9_resolution":"UNRESOLVED",
    }


def main():
    if not PENDING.exists() or not MATRIX.exists():
        raise SystemExit("Faltan outputs Medsafe V8; V9 debe ejecutarse después de V8.")
    pending=read_csv(PENDING)
    matrix=read_csv(MATRIX)
    if len(matrix)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V8 esperada 1122; encontrada {len(matrix)}")

    cis_map,comp_map=load_bdpm()
    print(f"BDPM specialties: {len(cis_map)} · compositions: {len(comp_map)}")
    print(f"Pending entering ANSM V9: {len(pending)}")

    results=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures={ex.submit(process_one,r,cis_map,comp_map):r for r in pending}
        done=0
        for fut in as_completed(futures):
            base=futures[fut]; done+=1
            try:
                rec=fut.result()
            except Exception as exc:
                rec={
                    "med_id":base.get("med_id"),"generic_name":base.get("generic_name"),
                    "ansm_status":"ERROR","ansm_error":repr(exc),
                    "v9_resolution":"UNRESOLVED",
                }
            results.append(rec)
            print(f"[{done:03d}/{len(pending)}] {rec.get('med_id')} {rec.get('generic_name')} -> {rec.get('v9_resolution')}")
            if done%25==0:
                write_csv(OUT/"renal_global_v9_ansm_partial.csv",sorted(results,key=lambda x:x.get("med_id") or ""))

    results.sort(key=lambda x:x.get("med_id") or "")
    write_csv(OUT/"renal_global_v9_ansm_results.csv",results)

    by_mid={r.get("med_id"):r for r in results}
    final=[]
    for old in matrix:
        row=dict(old); mid=old.get("med_id"); nr=by_mid.get(mid)
        prior=str(old.get("global_current_resolved_v8")).lower()=="true"
        if nr:
            row.update(nr)
            if nr.get("v9_resolution")=="CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v9"]=nr["v9_resolution"]
                row["global_primary_source_v9"]="ANSM_BDPM"
                row["global_current_resolved_v9"]=True
            else:
                row["global_resolution_v9"]=old.get("global_resolution_v8") or nr.get("v9_resolution")
                row["global_primary_source_v9"]=old.get("global_primary_source_v8") or ""
                row["global_current_resolved_v9"]=prior
        else:
            row["global_resolution_v9"]=old.get("global_resolution_v8")
            row["global_primary_source_v9"]=old.get("global_primary_source_v8")
            row["global_current_resolved_v9"]=prior
        final.append(row)

    if len(final)!=1122:
        raise SystemExit(f"SEGURIDAD: matriz V9 !=1122 ({len(final)})")

    write_csv(OUT/"renal_global_matrix_1122.csv",final)
    resolved=[r for r in final if r.get("global_current_resolved_v9") is True or str(r.get("global_current_resolved_v9")).lower()=="true"]
    still=[r for r in final if r not in resolved]
    write_csv(OUT/"renal_global_resolved.csv",resolved)
    write_csv(OUT/"renal_global_still_pending.csv",still)

    summary={
        "generated_at_utc":datetime.now(timezone.utc).isoformat(),
        "catalog_rows":1122,
        "v8_pending_input":len(pending),
        "ansm_accept":sum(r.get("ansm_status")=="ACCEPT" for r in results),
        "ansm_exact_without_explicit_action":sum(r.get("ansm_status")=="EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION" for r in results),
        "ansm_no_exact_product":sum(r.get("ansm_status")=="NO_EXACT_BDPM_PRODUCT" for r in results),
        "ansm_errors":sum(r.get("ansm_status")=="ERROR" for r in results),
        "global_current_resolved_v9":len(resolved),
        "global_still_pending_v9":len(still),
        "total_check":len(resolved)+len(still),
        "source":{
            "name":"Base de Données Publique des Médicaments · ANSM/France",
            "specialties":CIS_URL,
            "composition":COMPO_URL,
            "role":"official RCP / composition database",
        },
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
