#!/usr/bin/env python3
"""MEDCALC pregnancy regulatory audit.
Builds a reviewable pregnancy evidence dataset from the canonical MED-ID catalogue.
No TGA-to-FDA conversion is ever performed. Letter categories are retained only
when explicitly present in the source text.
"""
from __future__ import annotations
import csv, json, re, time, unicodedata, urllib.parse, urllib.request
from pathlib import Path

CATALOG=Path("MEDCALC_RENAL_MASTER_CATALOGO_1122.csv")
OUT=Path("generated_pregnancy_global_v1")
OUT.mkdir(exist_ok=True)
UA={"User-Agent":"MEDCALC-clinico pregnancy evidence audit/1.0"}

def norm(s):
    s=unicodedata.normalize("NFKD",str(s or "")).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+"," ",s).strip()

def get_json(url, timeout=30):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.load(r)

def pregnancy_text(result):
    vals=[]
    for key in ("pregnancy","pregnancy_or_breast_feeding","teratogenic_effects","labor_and_delivery"):
        v=result.get(key)
        if isinstance(v,list): vals.extend(str(x) for x in v)
        elif v: vals.append(str(v))
    return "\n".join(vals).strip()

def category(text):
    m=re.search(r"pregnancy\s+(?:category|category\s*[:\-]?)\s*([ABCDX])\b",text,re.I)
    if not m: m=re.search(r"pregnancy\s*category\s*([ABCDX])\b",text,re.I)
    return m.group(1).upper() if m else ""

def search_openfda(name):
    # exact generic identity first; never accept a fuzzy identity.
    q=urllib.parse.quote(f'openfda.generic_name:"{name}"')
    url=f"https://api.fda.gov/drug/label.json?search={q}&limit=10"
    try: data=get_json(url)
    except Exception as e: return None,str(e)
    candidates=[]
    nn=norm(name)
    for x in data.get("results",[]):
        gens=x.get("openfda",{}).get("generic_name",[])
        if any(norm(g)==nn for g in gens):
            txt=pregnancy_text(x)
            if txt: candidates.append((x,txt))
    if not candidates: return None,"no exact generic pregnancy label"
    candidates.sort(key=lambda z:z[0].get("effective_time",""),reverse=True)
    return candidates[0],None

def main():
    with CATALOG.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    out=[]; accepted=0
    for i,r in enumerate(rows,1):
        med_id=r.get("med_id") or r.get("MED_ID") or ""
        name=r.get("generic_name") or r.get("nombre") or r.get("medicamento") or ""
        rec={"med_id":med_id,"generic_name":name,"status":"UNRESOLVED","source":"openFDA/DailyMed",
             "source_url":"","effective_time":"","fda_historical_category":"","pregnancy_text":"",
             "trimester_specific":False,"identity_method":"EXACT_GENERIC"}
        if name:
            hit,err=search_openfda(name)
            if hit:
                txt=pregnancy_text(hit); rec.update(status="REGULATORY_TEXT_FOUND",
                    source_url="https://dailymed.nlm.nih.gov/",effective_time=hit.get("effective_time",""),
                    fda_historical_category=category(txt),pregnancy_text=txt)
                accepted+=1
            else: rec["note"]=err
        out.append(rec)
        if i%50==0: print(f"{i}/{len(rows)} accepted={accepted}",flush=True)
        time.sleep(0.05)
    fields=sorted({k for r in out for k in r})
    with (OUT/"pregnancy_v1_full_audit.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(out)
    with (OUT/"pregnancy_v1_regulatory_found.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([r for r in out if r["status"]=="REGULATORY_TEXT_FOUND"])
    with (OUT/"pregnancy_v1_unresolved.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([r for r in out if r["status"]=="UNRESOLVED"])
    summary={"total":len(out),"regulatory_text_found":accepted,"unresolved":len(out)-accepted,
             "safety_rules":["NO_TGA_TO_FDA_MAPPING","EXACT_GENERIC_IDENTITY_ONLY","NO_TRIMESTER_INFERENCE","FDA_LETTERS_ONLY_IF_EXPLICIT_IN_SOURCE"]}
    (OUT/"pregnancy_v1_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
