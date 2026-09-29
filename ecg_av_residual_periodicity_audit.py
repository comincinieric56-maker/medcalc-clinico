from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
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
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=120
P_RICH_LEADS=("II","V1","aVF","III","I","V2")
MIN_PERIOD_MS=300.0
MAX_PERIOD_MS=1500.0
MIN_ACF_STRENGTH=0.25
MIN_CONCORDANT_LEADS=3


def _ventricular_template_residual(
    x:np.ndarray,
    r_peaks:list[int],
    fs:int,
)->np.ndarray|None:
    y=np.asarray(x,dtype=float)
    r=np.asarray(sorted(set(int(v) for v in r_peaks)),dtype=int)
    pre=int(round(0.25*fs))
    post=int(round(0.45*fs))
    segs=[]
    valid_r=[]
    for rp in r:
        a=rp-pre; b=rp+post+1
        if a<0 or b>len(y):
            continue
        seg=y[a:b]
        if not np.isfinite(seg).all():
            continue
        edge=max(3,int(round(0.025*fs)))
        baseline=float(np.median(np.r_[seg[:edge],seg[-edge:]]))
        segs.append(seg-baseline)
        valid_r.append(rp)
    if len(segs)<4:
        return None
    template=np.median(np.vstack(segs),axis=0)
    residual=np.asarray(y,dtype=float).copy()
    weight=np.zeros_like(residual)
    acc=np.zeros_like(residual)
    for rp in valid_r:
        a=rp-pre; b=rp+post+1
        acc[a:b]+=template
        weight[a:b]+=1.0
    mask=weight>0
    residual[mask]-=acc[mask]/weight[mask]
    med=float(np.nanmedian(residual))
    residual=np.nan_to_num(residual-med,nan=0.0,posinf=0.0,neginf=0.0)
    return residual


def _acf_period(residual:np.ndarray,fs:int)->dict[str,Any]:
    # Downsample after a simple moving-average anti-alias step. Only periodic
    # structure in the physiologic atrial range is audited; this is not a
    # fiducial detector and cannot publish a diagnosis.
    target_fs=100
    step=max(1,int(round(fs/target_fs)))
    if step>1:
        kernel=np.ones(step,dtype=float)/float(step)
        z=np.convolve(residual,kernel,mode="same")[::step]
        eff_fs=fs/step
    else:
        z=residual.copy()
        eff_fs=float(fs)
    z=z-float(np.mean(z))
    sd=float(np.std(z))
    if not np.isfinite(sd) or sd<1e-6:
        return {"evaluable":False,"reason":"LOW_RESIDUAL_VARIANCE"}
    z=z/sd
    n=len(z)
    fft_n=1
    while fft_n<2*n:
        fft_n*=2
    f=np.fft.rfft(z,fft_n)
    ac=np.fft.irfft(f*np.conj(f),fft_n)[:n]
    denom=np.arange(n,0,-1,dtype=float)
    ac=ac/np.maximum(denom,1.0)
    if ac[0]<=0:
        return {"evaluable":False,"reason":"INVALID_ACF"}
    ac=ac/ac[0]
    min_lag=max(1,int(round(MIN_PERIOD_MS*eff_fs/1000.0)))
    max_lag=min(n-2,int(round(MAX_PERIOD_MS*eff_fs/1000.0)))
    if max_lag<=min_lag:
        return {"evaluable":False,"reason":"SHORT_SIGNAL"}
    region=ac[min_lag:max_lag+1]
    rel=int(np.argmax(region))
    lag=min_lag+rel
    strength=float(ac[lag])
    period_ms=lag*1000.0/eff_fs
    return {
        "evaluable":True,
        "period_ms":round(period_ms,3),
        "acf_strength":round(strength,6),
        "strong":bool(strength>=MIN_ACF_STRENGTH),
    }


def _audit_record(rec,analysis:dict[str,Any])->dict[str,Any]:
    sig=np.asarray(rec.p_signal,dtype=float)
    names=[str(x) for x in rec.sig_name]
    fs=int(round(float(rec.fs)))
    periods=[]
    by_lead={}
    rr_values=[]
    for lead in P_RICH_LEADS:
        if lead not in names:
            continue
        item=dict((analysis.get("leads") or {}).get(lead) or {})
        r=list(item.get("r_peaks_samples") or [])
        if len(r)>=3:
            rr=np.diff(np.asarray(r,dtype=float))*1000.0/fs
            rr=rr[np.isfinite(rr)&(rr>0)]
            if rr.size:
                rr_values.extend(rr.tolist())
        residual=_ventricular_template_residual(sig[:,names.index(lead)],r,fs)
        if residual is None:
            by_lead[lead]={"evaluable":False,"reason":"LT_4_TEMPLATE_BEATS"}
            continue
        row=_acf_period(residual,fs)
        by_lead[lead]=row
        if row.get("strong"):
            periods.append((lead,float(row["period_ms"]),float(row["acf_strength"])))
    rr_med=float(np.median(rr_values)) if rr_values else None
    if periods:
        pvals=np.asarray([p for _,p,_ in periods],dtype=float)
        period_med=float(np.median(pvals))
        concordant=[
            (lead,p,s) for lead,p,s in periods
            if abs(p-period_med)/max(period_med,1.0)<=0.10
        ]
    else:
        period_med=None
        concordant=[]
    atrial_faster=bool(
        rr_med is not None and period_med is not None
        and rr_med/max(period_med,1.0)>=1.20
    )
    ratio=(rr_med/period_med) if rr_med and period_med else None
    trigger=bool(len(concordant)>=MIN_CONCORDANT_LEADS and atrial_faster)
    return {
        "strong_lead_n":len(periods),
        "concordant_lead_n":len(concordant),
        "residual_atrial_period_ms":round(period_med,3) if period_med is not None else None,
        "ventricular_rr_median_ms":round(rr_med,3) if rr_med is not None else None,
        "atrial_to_ventricular_rate_ratio":round(ratio,6) if ratio is not None else None,
        "atrial_faster_by_ge20pct":atrial_faster,
        "crosslead_periodicity_trigger":trigger,
    }


def _score(rows:list[dict[str,Any]],target:str,negative_ids:set[int])->dict[str,Any]:
    spec=TARGETS[target]
    positives=[r for r in rows if _target_positive(r.get("codes") or {},spec["scp"])]
    pos_trigger=sum(bool((r.get("periodicity") or {}).get("crosslead_periodicity_trigger")) for r in positives)
    neg=[r for r in rows if int(r["ecg_id"]) in negative_ids]
    neg_trigger=sum(bool((r.get("periodicity") or {}).get("crosslead_periodicity_trigger")) for r in neg)
    return {
        "positive_n":len(positives),
        "positive_trigger_n":pos_trigger,
        "positive_trigger_fraction":pos_trigger/len(positives) if positives else None,
        "negative_control_n":len(neg),
        "negative_trigger_n":neg_trigger,
        "negative_trigger_fraction":neg_trigger/len(neg) if neg else None,
        "specificity_if_used_as_gate":(len(neg)-neg_trigger)/len(neg) if neg else None,
        "interpretation":"OBSERVABILITY_AUDIT_ONLY_NOT_A_DIAGNOSTIC_RULE",
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-av-residual-periodicity"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AV_RESIDUAL_PERIODICITY.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    parts=[]; availability={}
    for target in ("AVB2","AVB3"):
        spec=TARGETS[target]
        pos=adult[adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))].copy()
        pos=pos.sort_values(["_hash","ecg_id"])
        availability[target]=len(pos)
        parts.append(pos)
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    negative_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat(parts+[neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    rows=[]; errors=[]; root=args.workdir/"records"
    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            rows.append({
                "ecg_id":ecg_id,
                "codes":dict(row["_codes"]),
                "periodicity":_audit_record(rec,analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%20==0:
            print(f"MEDCALC_AV_RESIDUAL_PERIODICITY {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_RESIDUAL_PERIODICITY_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_available_by_target":availability,
        "negative_selected_n":len(neg),
        "records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "parameters":{
            "p_rich_leads":list(P_RICH_LEADS),
            "period_ms_range":[MIN_PERIOD_MS,MAX_PERIOD_MS],
            "min_acf_strength":MIN_ACF_STRENGTH,
            "min_concordant_leads":MIN_CONCORDANT_LEADS,
            "min_atrial_to_ventricular_rate_ratio":1.20,
        },
        "AVB2":_score(rows,"AVB2",negative_ids),
        "AVB3":_score(rows,"AVB3",negative_ids),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AV residual periodicity audit had {len(errors)} errors")


if __name__=="__main__":
    main()
