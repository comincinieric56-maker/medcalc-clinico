#!/usr/bin/env python3
"""Rank identity candidates for human/regulatory review. Never auto-accepts."""
import csv,json,sys,unicodedata,re
from difflib import SequenceMatcher
from pathlib import Path
def norm(s):
    s=unicodedata.normalize("NFKD",str(s or "")).encode("ascii","ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+"," ",s).strip()
def main(queue,index,out):
    rows=list(csv.DictReader(Path(queue).open(encoding="utf-8-sig")))
    names=json.loads(Path(index).read_text(encoding="utf-8"))
    keys=list(names)
    result=[]
    for r in rows:
        if r.get("candidate_review_reason")!="REVIEW_SPANISH_INN_USAN_EQUIVALENT": continue
        q=norm(r.get("generic_name"))
        scored=sorted(((SequenceMatcher(None,q,k).ratio(),k) for k in keys),reverse=True)[:5]
        for rank,(score,k) in enumerate(scored,1):
            result.append({"med_id":r.get("med_id",""),"generic_name":r.get("generic_name",""),
             "candidate_rank":rank,"candidate_generic_name":names[k],"similarity":f"{score:.4f}",
             "decision":"REVIEW_REQUIRED","identity_accepted":"false"})
    fields=["med_id","generic_name","candidate_rank","candidate_generic_name","similarity","decision","identity_accepted"]
    with Path(out).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(result)
    print(json.dumps({"source_med_ids":len({r["med_id"] for r in result}),"candidate_rows":len(result),
      "autoaccepted":0,"rule":"SIMILARITY_IS_DISCOVERY_ONLY"},indent=2))
if __name__=="__main__":
    if len(sys.argv)!=4: raise SystemExit("usage: rank.py QUEUE.csv INDEX.json OUT.csv")
    main(*sys.argv[1:])
