#!/usr/bin/env python3
import csv,json,re,sys
from collections import Counter
from pathlib import Path
SALT_WORDS={"acetato","besilato","bromhidrato","calcio","citrato","clorhidrato","fosfato","fumarato","hidrobromuro","hidrocloruro","maleato","mesilato","potasica","potasico","sodica","sodico","succinato","tartrato"}
def classify(r):
    if r.get("status")=="REGULATORY_TEXT_FOUND": return "FDA_DAILYMED_LABEL_FOUND"
    n=(r.get("generic_name") or "").lower()
    if any(x in n for x in (" + "," / ","+")): return "COMBINATION_REQUIRES_EXACT_PRODUCT_IDENTITY"
    if set(re.findall(r"[a-záéíóúñ]+",n)) & SALT_WORDS: return "SALT_ESTER_REQUIRES_VERIFIED_EQUIVALENCE"
    return "NO_EXACT_OPENFDA_LABEL_FOUND"
def main(src,outdir):
    rows=list(csv.DictReader(Path(src).open(encoding="utf-8-sig")))
    assert len(rows)==1122, f"expected 1122 rows, got {len(rows)}"
    for r in rows:r["resolution_status"]=classify(r)
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0])
    with (out/"pregnancy_v1_resolution_status.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    counts=dict(sorted(Counter(r["resolution_status"] for r in rows).items()))
    summary={"total":len(rows),"resolution_counts":counts,"safety_rules":["NO_FUZZY_AUTOACCEPT","NO_TGA_TO_FDA_MAPPING","NO_TRIMESTER_INFERENCE","UNRESOLVED_IS_NOT_EVIDENCE"]}
    (out/"pregnancy_v1_resolution_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":
    if len(sys.argv)!=3:raise SystemExit("usage: classifier.py FULL_AUDIT.csv OUTDIR")
    main(sys.argv[1],sys.argv[2])
