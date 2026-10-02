from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _canonical, _download, _ensure_record,
    _fast_gate_holdout_ids, _target_positive, select_records,
)
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_independent_atrial_evidence_audit import _shadow_harmonic_relation
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_HARMONIC_BRANCH_SPECIFICITY_AUDIT_V1"


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _topology(events:list[dict], r_samples:list[int], fs:int) -> dict:
    p=np.unique(np.asarray([
        int(round(float(e["time_ms"])*fs/1000.0))
        for e in events if e.get("time_ms") is not None
    ],dtype=int))
    r=np.unique(np.asarray([int(x) for x in r_samples],dtype=int))
    out={
        "evaluable":False,"conducted_n":0,"dropped_n":0,"max_drop":0,
        "ratio":None,"stable_pr":False,"progressive_pr":False,"raw_branch":None,
    }
    if fs<=0 or len(p)<4 or len(r)<3:
        return out

    mappings=[{"p":int(pi),"conducted":False,"pr_ms":None} for pi in p]
    claimed=set()
    min_pr=int(np.floor(70.0*fs/1000.0))
    max_pr=int(np.ceil(550.0*fs/1000.0))
    for ri0 in r:
        ri=int(ri0)
        idx=np.where(
            (p<ri)
            & ((ri-p)>=min_pr)
            & ((ri-p)<=max_pr)
        )[0]
        chosen=None
        for z in idx[::-1]:
            j=int(z)
            if j not in claimed:
                chosen=j
                break
        if chosen is None:
            continue
        pi=int(p[chosen])
        mappings[chosen]={
            "p":pi,
            "conducted":True,
            "pr_ms":float((ri-pi)*1000.0/fs),
        }
        claimed.add(chosen)

    conducted=[m for m in mappings if m["conducted"]]
    dropped=[m for m in mappings if not m["conducted"]]
    pr=np.asarray([m["pr_ms"] for m in conducted],dtype=float)
    pr_mad=float(np.median(np.abs(pr-np.median(pr)))) if pr.size>=2 else None
    stable=bool(pr.size>=3 and pr_mad is not None and pr_mad<=40.0)

    max_drop=0
    run=0
    for m in mappings:
        if m["conducted"]:
            run=0
        else:
            run+=1
            max_drop=max(max_drop,run)

    progressive=False
    for j,m in enumerate(mappings):
        if m["conducted"] or j<3:
            continue
        prior=[
            float(x["pr_ms"])
            for x in mappings[max(0,j-3):j]
            if x["conducted"] and x["pr_ms"] is not None
        ]
        if len(prior)>=3 and all((b-a)>5.0 for a,b in zip(prior[:-1],prior[1:])):
            progressive=True
            break

    ratio=len(p)/max(len(r),1)
    raw_branch=None
    if len(dropped)>=1 and len(conducted)>=2:
        if 1.60<=ratio<=2.40:
            raw_branch="TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
        elif max_drop>=2:
            raw_branch="HIGH_GRADE_AV_BLOCK_COMPATIBLE"
        elif stable:
            raw_branch="MOBITZ_II_COMPATIBLE"
        elif progressive:
            raw_branch="MOBITZ_I_WENCKEBACH_COMPATIBLE"
        else:
            raw_branch="SECOND_DEGREE_AV_BLOCK_CANDIDATE"

    out.update({
        "evaluable":True,
        "conducted_n":len(conducted),
        "dropped_n":len(dropped),
        "max_drop":max_drop,
        "ratio":float(ratio),
        "stable_pr":stable,
        "progressive_pr":progressive,
        "raw_branch":raw_branch,
    })
    return out


def _apply(dst:Counter, branch_counts:Counter, events:list[dict], rhythm:dict, fs:int, prefix:str) -> None:
    topo=_topology(events,list(rhythm.get("r_peaks_samples") or []),fs)
    h=_shadow_harmonic_relation({"unseeded_events":events},rhythm,fs)
    harmonic=bool(h.get("harmonic_consistent"))
    dst[f"{prefix}_n"]+=1
    dst[f"{prefix}_evaluable_n"]+=int(bool(topo["evaluable"]))
    dst[f"{prefix}_harmonic_consistent_n"]+=int(harmonic)
    dst[f"{prefix}_dropped_any_n"]+=int(int(topo["dropped_n"])>=1)
    dst[f"{prefix}_conducted_ge2_n"]+=int(int(topo["conducted_n"])>=2)
    dst[f"{prefix}_stable_pr_n"]+=int(bool(topo["stable_pr"]))
    dst[f"{prefix}_progressive_pr_n"]+=int(bool(topo["progressive_pr"]))
    dst[f"{prefix}_raw_branch_any_n"]+=int(topo["raw_branch"] is not None)
    dst[f"{prefix}_harmonic_branch_any_n"]+=int(
        harmonic and topo["raw_branch"] is not None
    )
    if topo["raw_branch"] is not None:
        branch_counts[f"{prefix}:RAW:{topo['raw_branch']}"]+=1
        if harmonic:
            branch_counts[f"{prefix}:HARMONIC:{topo['raw_branch']}"]+=1


def run(workdir:Path,output:Path,process_fold:int):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"
    statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)

    meta=pd.read_csv(meta_path)
    selected,summary=select_records(
        meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids()
    )
    neg_ids=set(int(x) for x in summary["negative_control_ecg_ids"])
    aliases=set(TARGETS["AVB2"]["scp"])
    rows=selected[
        (selected["_fold"].astype(int)==int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(neg_ids)
            | selected["_codes"].map(lambda c:_target_positive(c,aliases))
        )
    ].copy()

    groups={
        "AVB2":{"counts":Counter(),"branches":Counter()},
        "CONTROL":{"counts":Counter(),"branches":Counter()},
    }
    errors=Counter()
    root=workdir/"records"

    for _,row in rows.iterrows():
        ecg_id=int(row["ecg_id"])
        group="CONTROL" if ecg_id in neg_ids else "AVB2"
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            )
            analysis=analyze_canonical_ecg(canonical)
            evidence=recover_crosslead_atrial_candidates(
                canonical,analysis.get("leads") or {}
            )
            observed=list((evidence.get("observed_consensus") or {}).get("events") or [])
            combined=[
                {"time_ms":float(t)}
                for t in (evidence.get("combined_event_times_ms") or [])
            ]
            rhythm=dict(analysis.get("rhythm") or {})
            fs=int(canonical.get("fs") or analysis.get("fs") or 500)

            dst=groups[group]
            dst["counts"]["case_n"]+=1
            _apply(dst["counts"],dst["branches"],observed,rhythm,fs,"observed")
            _apply(dst["counts"],dst["branches"],combined,rhythm,fs,"combined")
        except Exception as exc:
            errors[f"{group}:{type(exc).__name__}"]+=1

    out={
        "version":VERSION,
        "dataset":"PTB-XL",
        "role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,
        "process_fold":int(process_fold),
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "groups":{
            g:{
                "counts":dict(v["counts"]),
                "branch_counts":dict(v["branches"]),
            } for g,v in groups.items()
        },
        "analysis_error_n":sum(errors.values()),
        "analysis_error_types":dict(errors),
        "policy":"EXISTING_HARMONIC_CONSISTENCY_AS_AUDIT_GATE_PLUS_EXACT_EXISTING_HIGH_RECALL_AV_BRANCH_THRESHOLDS; NO_EVENT_SYNTHESIS; NO_NEW_THRESHOLDS; ALL_FIXED_CLEAN_CONTROLS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--process-fold",type=int,choices=[1,2,3,4,5,6,7,8],required=True)
    args=ap.parse_args()
    run(args.workdir,args.output,args.process_fold)
