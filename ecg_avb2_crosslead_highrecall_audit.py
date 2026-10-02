from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,TARGETS,_canonical,_download,_ensure_record,_fast_gate_holdout_ids,
    _target_positive,select_records,
)
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_CROSSLEAD_HIGHRECALL_AUDIT_V1"

def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try: return fn(*args)
        except Exception:
            if attempt>=attempts: raise
            time.sleep(10*attempt)

def highrecall(events,r_samples,fs):
    p=np.unique(np.asarray([int(round(float(e["time_ms"])*fs/1000.0)) for e in events if e.get("time_ms") is not None],dtype=int))
    r=np.unique(np.asarray([int(x) for x in r_samples],dtype=int))
    out={"evaluable":False,"atrial_regular":False,"candidate":False,"code":None,"max_drop":0,"conducted_n":0,"dropped_n":0}
    if len(p)<4 or len(r)<3 or fs<=0: return out
    pp=np.diff(p)*1000.0/fs
    if pp.size<2 or np.mean(pp)<=0: return out
    pp_cv=float(np.std(pp,ddof=1)/np.mean(pp))
    atrial_regular=pp_cv<=0.18
    mappings=[{"p":int(pi),"conducted":False,"pr_ms":None} for pi in p]
    claimed=set()
    min_pr=int(np.floor(70.0*fs/1000.0)); max_pr=int(np.ceil(550.0*fs/1000.0))
    for ri0 in r:
        ri=int(ri0)
        idx=np.where((p<ri)&((ri-p)>=min_pr)&((ri-p)<=max_pr))[0]
        chosen=None
        for z in idx[::-1]:
            j=int(z)
            if j not in claimed:
                chosen=j; break
        if chosen is None: continue
        pi=int(p[chosen])
        mappings[chosen]={"p":pi,"conducted":True,"pr_ms":float((ri-pi)*1000.0/fs)}
        claimed.add(chosen)
    conducted=[m for m in mappings if m["conducted"]]
    dropped=[m for m in mappings if not m["conducted"]]
    pr=np.asarray([m["pr_ms"] for m in conducted],dtype=float)
    pr_mad=float(np.median(np.abs(pr-np.median(pr)))) if pr.size>=2 else None
    stable=bool(pr.size>=3 and pr_mad is not None and pr_mad<=40.0)
    run=max_drop=0
    for m in mappings:
        if m["conducted"]: run=0
        else: run+=1; max_drop=max(max_drop,run)
    ratio=len(p)/max(len(r),1)
    progressive=False
    for j,m in enumerate(mappings):
        if m["conducted"] or j<3: continue
        prior=[float(x["pr_ms"]) for x in mappings[max(0,j-3):j] if x["conducted"] and x["pr_ms"] is not None]
        if len(prior)>=3 and all((b-a)>5.0 for a,b in zip(prior[:-1],prior[1:])):
            progressive=True; break
    code=None
    if atrial_regular and len(dropped)>=1 and len(conducted)>=2:
        if 1.60<=ratio<=2.40: code="TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
        elif max_drop>=2: code="HIGH_GRADE_AV_BLOCK_COMPATIBLE"
        elif stable: code="MOBITZ_II_COMPATIBLE"
        elif progressive: code="MOBITZ_I_WENCKEBACH_COMPATIBLE"
        else: code="SECOND_DEGREE_AV_BLOCK_CANDIDATE"
    out.update({
        "evaluable":True,"pp_cv":pp_cv,"atrial_regular":atrial_regular,
        "candidate":code is not None,"code":code,"max_drop":max_drop,
        "conducted_n":len(conducted),"dropped_n":len(dropped),"ratio":ratio,
    })
    return out

def run(workdir,output,process_fold=None):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"; statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)
    meta=pd.read_csv(meta_path)
    selected,_=select_records(meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids())
    aliases=set(TARGETS["AVB2"]["scp"])
    positives=selected[selected["_codes"].map(lambda c:_target_positive(c,aliases))].copy()
    if process_fold is not None:
        positives=positives[positives["_fold"].astype(int)==int(process_fold)].copy()

    counts=Counter(); code_counts=Counter(); errors=[]
    root=workdir/"records"
    for _,row in positives.iterrows():
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),int(row["ecg_id"]))
            a=analyze_canonical_ecg(canonical)
            ev=recover_crosslead_atrial_candidates(canonical,a.get("leads") or {})
            observed=list((ev.get("observed_consensus") or {}).get("events") or [])
            seeded=list(ev.get("recovered_events") or [])
            unseeded=list(ev.get("unseeded_events") or [])
            combined=[{"time_ms":float(t)} for t in (ev.get("combined_event_times_ms") or [])]
            r=list((a.get("rhythm") or {}).get("r_peaks_samples") or [])
            fs=int(canonical.get("fs") or a.get("fs") or 500)
            counts["n"]+=1
            for name,events in {"observed":observed,"seeded":seeded,"unseeded":unseeded,"combined":combined}.items():
                q=highrecall(events,r,fs)
                counts[f"{name}_evaluable_n"]+=int(q["evaluable"])
                counts[f"{name}_atrial_regular_le018_n"]+=int(q["atrial_regular"])
                counts[f"{name}_candidate_n"]+=int(q["candidate"])
                counts[f"{name}_ge2_drop_n"]+=int(int(q["max_drop"])>=2)
                counts[f"{name}_conducted_ge2_n"]+=int(int(q["conducted_n"])>=2)
                if q["code"]: code_counts[f"{name}:{q['code']}"]+=1
        except Exception as exc:
            errors.append(type(exc).__name__)
    out={
        "version":VERSION,"dataset":"PTB-XL","role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,"process_fold":process_fold,"fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,"selection_avb2_positive_n":int(len(positives)),
        "analysis_error_n":len(errors),"analysis_error_types":dict(Counter(errors)),
        "counts":dict(counts),"code_counts":dict(code_counts),
        "policy":"EXACT_EXISTING_HIGH_RECALL_AV_SEQUENCE_THRESHOLDS_AND_MAPPING; AUDIT_ONLY; NO_NEW_THRESHOLDS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--process-fold",type=int,choices=[1,2,3,4,5,6,7,8])
    args=ap.parse_args()
    run(args.workdir,args.output,args.process_fold)
