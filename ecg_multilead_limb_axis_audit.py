from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
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
LIMB_ANGLES={
    "I":0.0,
    "II":60.0,
    "III":120.0,
    "aVR":-150.0,
    "aVL":-30.0,
    "aVF":90.0,
}

def _net_area(analysis, lead):
    m=(((analysis.get("leads") or {}).get(lead) or {}).get("metrics") or {}).get("qrs_net_area_mv_ms") or {}
    v=m.get("value")
    try:
        x=float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None

def _fit_axis(analysis):
    rows=[]; vals=[]
    for lead,deg in LIMB_ANGLES.items():
        v=_net_area(analysis,lead)
        if v is None:
            continue
        phi=math.radians(deg)
        rows.append([math.cos(phi),math.sin(phi)])
        vals.append(v)
    if len(vals)<4:
        return {"evaluable":False,"lead_n":len(vals)}
    X=np.asarray(rows,dtype=float)
    y=np.asarray(vals,dtype=float)
    coef, *_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@coef
    theta=math.degrees(math.atan2(float(coef[1]),float(coef[0])))
    # Normalize to conventional [-180,180].
    if theta>180: theta-=360
    if theta<=-180: theta+=360
    ss_res=float(np.sum((y-pred)**2))
    ss_tot=float(np.sum((y-np.mean(y))**2))
    r2=(1.0-ss_res/ss_tot) if ss_tot>1e-12 else None
    return {
        "evaluable":True,
        "lead_n":len(vals),
        "degrees":round(theta,6),
        "r2":round(float(r2),6) if r2 is not None else None,
    }

def _policy(analysis,target):
    fit=_fit_axis(analysis)
    if not fit.get("evaluable"):
        return False
    deg=float(fit["degrees"])
    f=analysis.get("fascicular_conduction") or {}
    c=f.get("criteria") or {}
    if target=="LAFB":
        return bool(
            -90.0<=deg<=-45.0
            and c.get("positive_qrs_I")
            and c.get("positive_qrs_aVL")
            and int(c.get("inferior_s_dominant_n") or 0)>=2
            and c.get("qrs_lt_120ms")
        )
    return bool(
        90.0<=deg<=180.0
        and int(c.get("superior_s_dominant_n") or 0)>=2
        and int(c.get("inferior_r_dominant_n") or 0)>=2
        and c.get("qrs_lt_120ms")
    )

def _score(rows,target):
    spec=TARGETS[target]
    expected=set(spec["medcalc"])
    positives=[r for r in rows if _target_positive(r["codes"],spec["scp"])]
    negatives=[r for r in rows if r["is_clean_control"]]
    btp=sum(bool(expected & set(r["published"])) for r in positives)
    bfp=sum(bool(expected & set(r["published"])) for r in negatives)
    rec=sum(
        (not bool(expected & set(r["published"]))) and bool(r["policy"][target])
        for r in positives
    )
    inc=sum(
        (not bool(expected & set(r["published"]))) and bool(r["policy"][target])
        for r in negatives
    )
    tp=btp+rec; fp=bfp+inc
    return {
        "positive_n":len(positives),
        "baseline_final_positive_n":btp,
        "recoverable_positive_n":rec,
        "projected_final_positive_n":tp,
        "projected_final_sensitivity":tp/len(positives) if positives else None,
        "negative_control_n":len(negatives),
        "baseline_negative_fp_n":bfp,
        "incremental_negative_trigger_n":inc,
        "projected_negative_fp_n":fp,
        "projected_specificity":(len(negatives)-fp)/len(negatives) if negatives else None,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-axis-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AXIS_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    parts=[]
    for target in ("LAFB","LPFB"):
        spec=TARGETS[target]
        z=adult[adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))].copy()
        parts.append(z.sort_values(["_hash","ecg_id"]).head(POSITIVE_N))
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    neg_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat(parts+[neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    rows=[]; errors=[]; root=args.workdir/"records"
    current_axis_abs_diff=[]
    fit_r2=[]
    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            fit=_fit_axis(analysis)
            cur=(analysis.get("axis") or {}).get("degrees")
            if fit.get("evaluable") and cur is not None:
                diff=abs(float(fit["degrees"])-float(cur))
                diff=min(diff,360.0-diff)
                current_axis_abs_diff.append(diff)
            if fit.get("r2") is not None:
                fit_r2.append(float(fit["r2"]))
            rows.append({
                "codes":dict(row["_codes"]),
                "is_clean_control":ecg_id in neg_ids,
                "published":sorted(_published_codes(analysis)),
                "policy":{
                    "LAFB":_policy(analysis,"LAFB"),
                    "LPFB":_policy(analysis,"LPFB"),
                },
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_AXIS_AUDIT {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_MULTILEAD_LIMB_AXIS_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "method":"LEAST_SQUARES_VECTOR_FIT_TO_QRS_NET_AREA_I_II_III_aVR_aVL_aVF",
        "minimum_limb_leads":4,
        "axis_fit_r2_median":float(np.median(fit_r2)) if fit_r2 else None,
        "current_vs_fit_axis_abs_diff_median_deg":float(np.median(current_axis_abs_diff)) if current_axis_abs_diff else None,
        "LAFB":_score(rows,"LAFB"),
        "LPFB":_score(rows,"LPFB"),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors: raise SystemExit(f"axis audit had {len(errors)} errors")

if __name__=="__main__":
    main()
