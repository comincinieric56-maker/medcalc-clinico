from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical, _download,
    _ensure_record, _hash, _published_codes, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg

TARGET="FLUTTER"
CODE="FLUTTER_OR_AT_COMPATIBLE"
CLEAN_N=160


def _metrics(rows, pred):
    tp=fn=fp=tn=0
    for r in rows:
        y=bool(r["reference"])
        p=bool(pred(r))
        if y and p: tp+=1
        elif y and not p: fn+=1
        elif (not y) and p: fp+=1
        else: tn+=1
    return {
        "n":len(rows),"tp":tp,"fn":fn,"fp":fp,"tn":tn,
        "sensitivity":tp/(tp+fn) if tp+fn else None,
        "specificity":tn/(tn+fp) if tn+fp else None,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-ptbxl-development-cache"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_FLUTTER_FOLD9_CONFIRM.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    fold9=_adult_rows(meta,[9])
    flutter=TARGETS[TARGET]
    af=TARGETS["AF"]

    pos=fold9[
        fold9["_codes"].map(lambda x:_target_positive(x,flutter["scp"]))
    ].copy()
    af_ref=fold9[
        fold9["_codes"].map(lambda x:_target_positive(x,af["scp"]))
        & ~fold9["_codes"].map(lambda x:_target_positive(x,flutter["scp"]))
    ].copy()
    clean=fold9[~fold9["_codes"].map(_any_target_positive)].copy()
    clean=clean.sort_values(["_hash","ecg_id"]).head(CLEAN_N)

    frames=[]
    for group,df in [
        ("FLUTTER_POSITIVE",pos),
        ("AF_REFERENCE",af_ref),
        ("CLEAN_CONTROL",clean),
    ]:
        z=df.copy(); z["_group"]=group; frames.append(z)
    frame=pd.concat(frames,ignore_index=True).drop_duplicates(subset=["ecg_id","_group"])

    expected=set(flutter["medcalc"])
    rows=[]; errors=[]; root=args.workdir/"records"
    for i,row in frame.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            fused=dict(((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE) or {})
            rows.append({
                "group":str(row["_group"]),
                "reference":_target_positive(dict(row["_codes"]),flutter["scp"]),
                "current_final":bool(expected & set(_published_codes(analysis))),
                "fused_publishable":bool(fused.get("publishable")),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_FLUTTER_FOLD9 {i+1}/{len(frame)}",flush=True)

    baseline=_metrics(rows,lambda r:r["current_final"])
    projected=_metrics(rows,lambda r:r["current_final"] or r["fused_publishable"])
    result={
        "version":"MEDCALC_FLUTTER_FOLD9_CONFIRMATION_V1",
        "role":"POST_SELECTION_CONFIRMATION_ONLY",
        "policy_frozen_before_confirmation":True,
        "policy":"PRESERVE_ALREADY_PUBLISHABLE_FUSED_FLUTTER_AS_SECONDARY_RHYTHM_FINDING",
        "tuning_data_used_in_this_run":False,
        "fold":9,
        "group_n":{
            g:sum(r["group"]==g for r in rows)
            for g in ("FLUTTER_POSITIVE","AF_REFERENCE","CLEAN_CONTROL")
        },
        "baseline_current_final":baseline,
        "projected_current_or_fused_flutter":projected,
        "incremental_by_group":{
            g:sum(
                r["group"]==g and (not r["current_final"]) and r["fused_publishable"]
                for r in rows
            )
            for g in ("FLUTTER_POSITIVE","AF_REFERENCE","CLEAN_CONTROL")
        },
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"Flutter fold9 confirmation had {len(errors)} errors")


if __name__=="__main__":
    main()
