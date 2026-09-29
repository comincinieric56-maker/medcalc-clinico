#!/usr/bin/env python3
"""MEDCALC pregnancy regulatory audit.
Builds a reviewable pregnancy evidence dataset from the canonical MED-ID catalogue.
No TGA-to-FDA conversion is ever performed. Letter categories are retained only
when explicitly present in the source text.
"""
from __future__ import annotations
import csv, json, os, re, time, unicodedata, urllib.parse, urllib.request, zipfile
from pathlib import Path

CATALOG=Path("MEDCALC_RENAL_MASTER_CATALOGO_1122.csv")
OUT=Path("generated_pregnancy_global_v1")
OUT.mkdir(exist_ok=True)
UA={"User-Agent":"MEDCALC-clinico pregnancy evidence audit/1.0"}
BULK_DIR=Path(os.getenv("OPENFDA_LABEL_BULK_DIR","")) if os.getenv("OPENFDA_LABEL_BULK_DIR") else None
REGULATORY_ALIASES={
    "cefadroxilo":"cefadroxil",
    # Reviewed Spanish/USAN-INN equivalents. These are identity translations only;
    # they do not imply pregnancy safety or any FDA category.
    "acetazolamida":"acetazolamide",
    "acetilcisteina":"acetylcysteine",
    "acido acetil salicilico":"aspirin",
    "acido ascorbico":"ascorbic acid",
    "acido folico":"folic acid",
    "acido mefenamico":"mefenamic acid",
    "acido tranexamico":"tranexamic acid",
    "albendazol":"albendazole",
    "amiodarona":"amiodarone",
    "amitriptilina":"amitriptyline",
    "azatioprina":"azathioprine",
    "benzocaina":"benzocaine",
    "budesonida":"budesonide",
    "capecitabina":"capecitabine",
    "carbamazepina":"carbamazepine",
    "cetirizina":"cetirizine",
    "cianocobalamina":"cyanocobalamin",
    "ciclofosfamida":"cyclophosphamide",
    "clomifeno":"clomiphene",
    "clomipramina":"clomipramine",
    "clorambucilo":"chlorambucil",
    "clorfenamina":"chlorpheniramine",
    # AEMPS/CIMA multilingual product authorizations explicitly document these
    # Spanish/non-Spanish active-name correspondences; identity only, no safety claim.
    "darifenacina":"darifenacin",
    "metocarbamol":"methocarbamol",
    "gabapentina":"gabapentin",
    "isotretinoina":"isotretinoin",
    "ciprofloxacina":"ciprofloxacin",
    "itraconazol":"itraconazole",
    "voriconazol":"voriconazole",
}

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

def build_bulk_generic_names():
    """Return normalized generic names actually present in pregnancy-bearing labels."""
    if not BULK_DIR or not BULK_DIR.exists(): return {}
    names={}
    for zp in sorted(BULK_DIR.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                if not member.endswith(".json"): continue
                with z.open(member) as fh: data=json.load(fh)
                for x in data.get("results",[]):
                    if not pregnancy_text(x): continue
                    for g in x.get("openfda",{}).get("generic_name",[]):
                        ng=norm(g)
                        if ng and ng not in names: names[ng]=g
    return names

def build_bulk_index(names):
    """Index only requested exact/reviewed generic names from openFDA label ZIPs."""
    if not BULK_DIR or not BULK_DIR.exists():
        return {}
    wanted={norm(REGULATORY_ALIASES.get(norm(n),n)) for n in names if n}
    idx={}
    for zp in sorted(BULK_DIR.glob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                if not member.endswith(".json"): continue
                with z.open(member) as fh:
                    data=json.load(fh)
                for x in data.get("results",[]):
                    gens=x.get("openfda",{}).get("generic_name",[])
                    matched=[norm(g) for g in gens if norm(g) in wanted]
                    if not matched or not pregnancy_text(x): continue
                    for key in matched:
                        old=idx.get(key)
                        if old is None or x.get("effective_time","") > old.get("effective_time",""):
                            idx[key]=x
    return idx

def search_openfda(name):
    # exact generic identity first; never accept a fuzzy identity.
    name=REGULATORY_ALIASES.get(norm(name),name)
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
    return candidates[0][0],None

def main():
    with CATALOG.open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    bulk=build_bulk_index([r.get("generic_name") or r.get("nombre") or r.get("medicamento") or "" for r in rows])
    regulatory_names=build_bulk_generic_names() if BULK_DIR else {}
    (OUT/"openfda_pregnancy_generic_name_index.json").write_text(
        json.dumps(regulatory_names,ensure_ascii=False,sort_keys=True),encoding="utf-8")
    out=[]; accepted=0
    for i,r in enumerate(rows,1):
        med_id=r.get("med_id") or r.get("MED_ID") or ""
        name=r.get("generic_name") or r.get("nombre") or r.get("medicamento") or ""
        rec={"med_id":med_id,"generic_name":name,"status":"UNRESOLVED","source":"openFDA/DailyMed",
             "source_url":"","effective_time":"","fda_historical_category":"","pregnancy_text":"",
             "trimester_specific":False,
             "identity_method":"EXPLICIT_REVIEWED_ALIAS" if norm(name) in REGULATORY_ALIASES else "EXACT_GENERIC",
             "regulatory_query_name":REGULATORY_ALIASES.get(norm(name),name),
             "set_id":"","application_number":"","manufacturer_name":""}
        if name:
            qname=REGULATORY_ALIASES.get(norm(name),name)
            hit=bulk.get(norm(qname)) if bulk else None
            err=None if hit else ("no exact generic pregnancy label in bulk" if bulk else None)
            if not bulk: hit,err=search_openfda(name)
            if hit:
                txt=pregnancy_text(hit); rec.update(status="REGULATORY_TEXT_FOUND",
                    source_url=(f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={hit.get('set_id')}" if hit.get("set_id") else "https://dailymed.nlm.nih.gov/"),
                    effective_time=hit.get("effective_time",""),
                    set_id=hit.get("set_id",""),
                    application_number="; ".join(hit.get("openfda",{}).get("application_number",[]) or []),
                    manufacturer_name="; ".join(hit.get("openfda",{}).get("manufacturer_name",[]) or []),
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
