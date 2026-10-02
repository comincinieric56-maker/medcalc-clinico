from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _canonical, _download, _ensure_record,
    _fast_gate_holdout_ids, _target_positive, select_records,
)
from ecg_avb2_crosslead_highrecall_audit import highrecall
from ecg_candidate_detectors import PREFERRED_AV_LEADS
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_REAL_PERLEAD_SUPPORT_AUDIT_V1"


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _case_support(analysis:dict) -> tuple[dict[str,int], Counter]:
    per_lead=analysis.get("leads") or {}
    support={
        "evaluable":0,
        "atrial_regular":0,
        "ge2_drop":0,
        "conducted_ge2":0,
        "regular_drop2_conducted2":0,
        "candidate":0,
    }
    codes=Counter()
    for lead in PREFERRED_AV_LEADS:
        item=per_lead.get(lead) or {}
        if not item.get("evaluable"):
            continue
        fs=int(item.get("fs") or analysis.get("fs") or 500)
        p=sorted(set(int(x) for x in (item.get("raw_p_peaks_samples") or [])))
        r=sorted(set(int(x) for x in (item.get("r_peaks_samples") or [])))
        events=[{"time_ms":1000.0*float(x)/float(fs)} for x in p]
        q=highrecall(events,r,fs)
        if not q.get("evaluable"):
            continue
        support["evaluable"]+=1
        regular=bool(q.get("atrial_regular"))
        drop2=int(q.get("max_drop") or 0)>=2
        conducted2=int(q.get("conducted_n") or 0)>=2
        candidate=bool(q.get("candidate"))
        support["atrial_regular"]+=int(regular)
        support["ge2_drop"]+=int(drop2)
        support["conducted_ge2"]+=int(conducted2)
        support["regular_drop2_conducted2"]+=int(regular and drop2 and conducted2)
        support["candidate"]+=int(candidate)
        if q.get("code"):
            codes[f"{lead}:{q['code']}"]+=1
    return support,codes


def run(workdir:Path,output:Path,process_fold:int):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"; statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)

    meta=pd.read_csv(meta_path)
    selected,summary=select_records(meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids())
    neg_ids=set(int(x) for x in summary["negative_control_ecg_ids"])
    avb2_aliases=set(TARGETS["AVB2"]["scp"])
    rows=selected[
        (selected["_fold"].astype(int)==int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(neg_ids)
            | selected["_codes"].map(lambda c:_target_positive(c,avb2_aliases))
        )
    ].copy()

    groups={
        "AVB2":{"counts":Counter(),"hist":Counter(),"code_counts":Counter()},
        "CONTROL":{"counts":Counter(),"hist":Counter(),"code_counts":Counter()},
    }
    errors=Counter(); root=workdir/"records"

    for _,row in rows.iterrows():
        ecg_id=int(row["ecg_id"])
        group="CONTROL" if ecg_id in neg_ids else "AVB2"
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id)
            analysis=analyze_canonical_ecg(canonical)
            support,codes=_case_support(analysis)
            dst=groups[group]
            dst["counts"]["case_n"]+=1
            for key,value in support.items():
                dst["counts"][f"case_any_{key}_n"]+=int(value>0)
                dst["hist"][f"{key}_support_{int(value)}"]+=1
            dst["counts"]["case_any_regular_drop2_conducted2_n"]+=int(
                support["regular_drop2_conducted2"]>0
            )
            dst["counts"]["case_any_candidate_n"]+=int(support["candidate"]>0)
            dst["code_counts"].update(codes)
        except Exception as exc:
            errors[f"{group}:{type(exc).__name__}"]+=1

    out={
        "version":VERSION,
        "dataset":"PTB-XL",
        "role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,
        "process_fold":int(process_fold),
        "preferred_av_leads":list(PREFERRED_AV_LEADS),
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "groups":{
            g:{
                "counts":dict(v["counts"]),
                "support_histogram":dict(v["hist"]),
                "code_counts":dict(v["code_counts"]),
            } for g,v in groups.items()
        },
        "analysis_error_n":sum(errors.values()),
        "analysis_error_types":dict(errors),
        "policy":"EXACT_EXISTING_PER_LEAD_HIGH_RECALL_MAPPING; SUPPORT_DISTRIBUTION_ONLY; NO_SUPPORT_THRESHOLD_SELECTION; ALL_FIXED_CLEAN_CONTROLS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
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
