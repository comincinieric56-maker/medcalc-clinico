from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _adult_rows,
    _any_target_positive,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _hash,
    _published_codes,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
CODE="VENTRICULAR_PREEXCITATION_COMPATIBLE"


def _policy(analysis:dict[str,Any])->bool:
    pre=dict(analysis.get("preexcitation") or {})
    crit=dict(pre.get("criteria") or {})
    short=set(str(x) for x in (crit.get("short_pr_leads") or []))
    delta=set(str(x) for x in (crit.get("rescue_delta_wide_leads") or []))
    overlap=short & delta
    union=short | delta
    return bool(
        crit.get("reproducible_p")
        and len(short)>=2
        and len(delta)>=2
        and len(overlap)>=1
        and len(union)>=3
    )


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-wpw-distributed"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_WPW_DISTRIBUTED.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["WPW"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"])
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    negative_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    pos_n=0; baseline_tp=0; recoverable=0
    neg_n=len(neg); baseline_fp=0; incremental_fp=0
    errors=[]; root=args.workdir/"records"
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
            trigger=_policy(analysis)
            is_pos=_target_positive(dict(row["_codes"]),spec["scp"])
            if is_pos:
                pos_n+=1
                baseline_tp+=int(final)
                recoverable+=int((not final) and trigger)
            elif ecg_id in negative_ids:
                baseline_fp+=int(final)
                incremental_fp+=int((not final) and trigger)
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_WPW_DISTRIBUTED {i+1}/{len(selected)}",flush=True)

    projected_tp=baseline_tp+recoverable
    projected_fp=baseline_fp+incremental_fp
    result={
        "version":"MEDCALC_WPW_DISTRIBUTED_SUPPORT_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "policy":{
            "short_pr_leads_ge":2,
            "rescue_delta_wide_leads_ge":2,
            "same_lead_overlap_ge":1,
            "distinct_support_leads_ge":3,
            "reproducible_p_required":True,
            "thresholds_unchanged":True,
        },
        "positive_n":pos_n,
        "baseline_final_positive_n":baseline_tp,
        "recoverable_positive_n":recoverable,
        "projected_final_positive_n":projected_tp,
        "projected_final_sensitivity":projected_tp/pos_n if pos_n else None,
        "negative_control_n":neg_n,
        "baseline_negative_fp_n":baseline_fp,
        "incremental_negative_trigger_n":incremental_fp,
        "projected_negative_fp_n":projected_fp,
        "projected_specificity":(neg_n-projected_fp)/neg_n if neg_n else None,
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
        "interpretation":"AGGREGATE_COUNTERFACTUAL_ONLY_NO_DIAGNOSTIC_CHANGE",
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"WPW distributed audit had {len(errors)} errors")


if __name__=="__main__":
    main()
