from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _fast_gate_holdout_ids, _hash,
    _published_codes, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg

FOLDS=[1,2,3,4,5,6,7,8]
POSITIVE_N=80
NEGATIVE_N=80
CODE="LPFB_COMPATIBLE"

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
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-lpfb-gate"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_LPFB_GATE.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["LPFB"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"]).head(POSITIVE_N)
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    rows=[]; errors=[]; root=args.workdir/"records"
    expected=set(spec["medcalc"])
    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            final=bool(expected & set(_published_codes(analysis)))
            specialist=str((analysis.get("fascicular_conduction") or {}).get("classification") or "")==CODE
            rows.append({
                "reference":_target_positive(dict(row["_codes"]),spec["scp"]),
                "current_final":final,
                "specialist_confirmed":specialist,
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_LPFB_GATE {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_LPFB_SPECIALIST_SAFETY_GATE_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "baseline_current_final":_metrics(rows,lambda r:r["current_final"]),
        "safety_gate_current_final_and_specialist":_metrics(
            rows,lambda r:r["current_final"] and r["specialist_confirmed"]
        ),
        "specialist_standalone":_metrics(rows,lambda r:r["specialist_confirmed"]),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors: raise SystemExit(f"LPFB gate audit had {len(errors)} errors")

if __name__=="__main__":
    main()
