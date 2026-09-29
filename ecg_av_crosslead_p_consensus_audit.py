from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _fast_gate_holdout_ids, _hash,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg

FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
CLUSTER_MS=50.0


def _clusters(analysis:dict[str,Any], support_n:int)->list[float]:
    leads=analysis.get("leads") or {}
    events=[]
    fs_values=[]
    for lead,item in leads.items():
        fs=int(item.get("fs") or analysis.get("fs") or 500)
        fs_values.append(fs)
        for p in item.get("raw_p_peaks_samples") or []:
            events.append((float(p)*1000.0/max(fs,1),str(lead)))
    events.sort()
    if not events:
        return []
    groups=[]
    cur=[]
    for t,lead in events:
        if not cur or t-cur[-1][0] <= CLUSTER_MS:
            cur.append((t,lead))
        else:
            groups.append(cur); cur=[(t,lead)]
    if cur:
        groups.append(cur)
    out=[]
    for g in groups:
        by_lead={}
        for t,lead in g:
            by_lead.setdefault(lead,[]).append(t)
        if len(by_lead) < support_n:
            continue
        # One contribution per lead prevents one noisy lead from dominating.
        vals=[float(np.median(v)) for v in by_lead.values()]
        out.append(float(np.median(vals)))
    return out


def _summary_for_support(analysis:dict[str,Any], support_n:int)->dict[str,Any]:
    p=_clusters(analysis,support_n)
    pp=np.diff(np.asarray(p,dtype=float)) if len(p)>=2 else np.asarray([],dtype=float)
    pp_med=float(np.median(pp)) if pp.size else None
    pp_cv=(
        float(np.std(pp,ddof=1)/np.mean(pp))
        if pp.size>=2 and float(np.mean(pp))>0 else None
    )
    regular=bool(len(p)>=4 and pp_cv is not None and pp_cv<=0.12)

    rhythm=analysis.get("rhythm") or {}
    r=np.asarray(rhythm.get("r_peaks_samples") or [],dtype=float)
    fs=int(analysis.get("fs") or 500)
    # rhythm r_peaks_samples are in samples from the canonical signal.
    rr=np.diff(r)*1000.0/max(fs,1) if r.size>=2 else np.asarray([],dtype=float)
    rr_med=float(np.median(rr)) if rr.size else None
    ventricular_rate=60000.0/rr_med if rr_med and rr_med>0 else None
    atrial_rate=60000.0/pp_med if pp_med and pp_med>0 else None
    ratio=len(p)/max(int(r.size),1)
    rate_gt_125=bool(
        atrial_rate is not None and ventricular_rate is not None
        and atrial_rate>1.25*ventricular_rate
    )

    av=analysis.get("av_conduction") or {}
    selected_lead=str(av.get("lead") or "")
    selected_count=0
    if selected_lead:
        selected_count=len(((analysis.get("leads") or {}).get(selected_lead) or {}).get("raw_p_peaks_samples") or [])

    return {
        "evaluable":len(p)>=4,
        "p_count":len(p),
        "qrs_count":int(r.size),
        "regular":regular,
        "p_qrs_count_ratio":ratio,
        "approx_2_to_1_count_ratio":bool(1.75<=ratio<=2.25 and len(p)>=4 and r.size>=2),
        "atrial_rate_gt_1_25_ventricular":rate_gt_125,
        "extra_p_vs_selected_ge1":bool(len(p)>=selected_count+1),
        "extra_p_vs_selected_ge2":bool(len(p)>=selected_count+2),
    }


def _flags(analysis:dict[str,Any])->dict[str,bool]:
    s2=_summary_for_support(analysis,2)
    s3=_summary_for_support(analysis,3)
    return {
        "S2_EVALUABLE":s2["evaluable"],
        "S2_REGULAR":s2["regular"],
        "S2_APPROX_2_TO_1":s2["approx_2_to_1_count_ratio"],
        "S2_ATRIAL_RATE_GT_1_25_VENTRICULAR":s2["atrial_rate_gt_1_25_ventricular"],
        "S2_EXTRA_P_GE1":s2["extra_p_vs_selected_ge1"],
        "S2_EXTRA_P_GE2":s2["extra_p_vs_selected_ge2"],
        "S3_EVALUABLE":s3["evaluable"],
        "S3_REGULAR":s3["regular"],
        "S3_APPROX_2_TO_1":s3["approx_2_to_1_count_ratio"],
        "S3_ATRIAL_RATE_GT_1_25_VENTRICULAR":s3["atrial_rate_gt_1_25_ventricular"],
        "S3_EXTRA_P_GE1":s3["extra_p_vs_selected_ge1"],
        "S3_EXTRA_P_GE2":s3["extra_p_vs_selected_ge2"],
    }


def _summ(rows:list[dict[str,Any]], group:str)->dict[str,Any]:
    z=[r for r in rows if r["group"]==group]
    names=sorted(next(iter(z))["flags"]) if z else []
    return {
        "n":len(z),
        "flag_counts":{name:sum(bool(r["flags"][name]) for r in z) for name in names},
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-av-crosslead-p"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AV_CROSSLEAD_P.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    parts=[]
    for target in ("AVB2","AVB3"):
        spec=TARGETS[target]
        z=adult[adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))].copy()
        z=z.sort_values(["_hash","ecg_id"]); z["_group"]=target
        parts.append(z)
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    neg["_group"]="CLEAN_CONTROL"
    selected=pd.concat(parts+[neg],ignore_index=True).drop_duplicates(subset=["ecg_id","_group"])

    rows=[]; errors=[]; root=args.workdir/"records"
    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            rows.append({"group":str(row["_group"]),"flags":_flags(analysis)})
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_AV_CROSSLEAD_P {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_CROSSLEAD_P_CONSENSUS_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "cluster_window_ms":CLUSTER_MS,
        "AVB2":_summ(rows,"AVB2"),
        "AVB3":_summ(rows,"AVB3"),
        "clean_control":_summ(rows,"CLEAN_CONTROL"),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
        "note":"Uses only existing raw_p_peaks across leads; no new P detector and no diagnostic change.",
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"crosslead P audit had {len(errors)} errors")


if __name__=="__main__":
    main()
