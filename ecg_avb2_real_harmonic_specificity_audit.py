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
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_independent_atrial_evidence_audit import _shadow_harmonic_relation
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_REAL_HARMONIC_SPECIFICITY_AUDIT_V1"


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _apply(dst:Counter, events:list[dict], rhythm:dict, fs:int, prefix:str) -> None:
    wrapped={"unseeded_events":list(events or [])}
    h=_shadow_harmonic_relation(wrapped,rhythm,fs)
    q=highrecall(events,list(rhythm.get("r_peaks_samples") or []),fs)
    dst[f"{prefix}_n"]+=1
    dst[f"{prefix}_event_ge3_n"]+=int(len(events or [])>=3)
    dst[f"{prefix}_harmonic_evaluable_n"]+=int(bool(h.get("evaluable")))
    dst[f"{prefix}_harmonic_consistent_n"]+=int(bool(h.get("harmonic_consistent")))
    dst[f"{prefix}_faster_than_ventricular_n"]+=int(bool(h.get("faster_than_ventricular")))
    dst[f"{prefix}_harmonic_faster_n"]+=int(bool(h.get("harmonic_consistent_and_faster")))
    dst[f"{prefix}_ventricular_regular_n"]+=int(bool(h.get("ventricular_regular")))
    dst[f"{prefix}_phase_dissociation_n"]+=int(bool(h.get("phase_dissociation")))
    dst[f"{prefix}_complete_block_mechanism_n"]+=int(bool(h.get("complete_block_mechanism")))
    drop2=bool(int(q.get("max_drop") or 0)>=2)
    conducted2=bool(int(q.get("conducted_n") or 0)>=2)
    dst[f"{prefix}_ge2_drop_n"]+=int(drop2)
    dst[f"{prefix}_conducted_ge2_n"]+=int(conducted2)
    dst[f"{prefix}_harmonic_drop2_conducted2_n"]+=int(
        bool(h.get("harmonic_consistent")) and drop2 and conducted2
    )
    dst[f"{prefix}_harmonic_faster_drop2_conducted2_n"]+=int(
        bool(h.get("harmonic_consistent_and_faster")) and drop2 and conducted2
    )


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

    groups={"AVB2":Counter(),"CONTROL":Counter()}
    errors=Counter()
    root=workdir/"records"
    for _,row in rows.iterrows():
        ecg_id=int(row["ecg_id"])
        group="CONTROL" if ecg_id in neg_ids else "AVB2"
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id)
            a=analyze_canonical_ecg(canonical)
            ev=recover_crosslead_atrial_candidates(canonical,a.get("leads") or {})
            observed=list((ev.get("observed_consensus") or {}).get("events") or [])
            combined=[{"time_ms":float(t)} for t in (ev.get("combined_event_times_ms") or [])]
            rhythm=dict(a.get("rhythm") or {})
            fs=int(canonical.get("fs") or a.get("fs") or 500)
            groups[group]["case_n"]+=1
            _apply(groups[group],observed,rhythm,fs,"observed")
            _apply(groups[group],combined,rhythm,fs,"combined")
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
        "groups":{k:dict(v) for k,v in groups.items()},
        "analysis_error_types":dict(errors),
        "analysis_error_n":sum(errors.values()),
        "policy":"EXACT_EXISTING_HARMONIC_RELATION_PLUS_EXISTING_HIGH_RECALL_DROP_MAPPING; AGGREGATE_AUDIT_ONLY; ALL_FIXED_CLEAN_CONTROLS; NO_NEW_THRESHOLDS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
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
