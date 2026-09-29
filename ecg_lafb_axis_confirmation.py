from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _canonical, _download, _ensure_record,
    _published_codes, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


TARGET="LAFB"
LIMB_ANGLES={"I":0.0,"II":60.0,"III":120.0,"aVR":-150.0,"aVL":-30.0,"aVF":90.0}


def _net_area(analysis, lead):
    m=(((analysis.get("leads") or {}).get(lead) or {}).get("metrics") or {}).get("qrs_net_area_mv_ms") or {}
    try:
        x=float(m.get("value"))
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
        return None
    X=np.asarray(rows,dtype=float)
    y=np.asarray(vals,dtype=float)
    coef,*_=np.linalg.lstsq(X,y,rcond=None)
    theta=math.degrees(math.atan2(float(coef[1]),float(coef[0])))
    if theta>180: theta-=360
    if theta<=-180: theta+=360
    return theta


def _policy(analysis):
    deg=_fit_axis(analysis)
    if deg is None:
        return False
    c=(analysis.get("fascicular_conduction") or {}).get("criteria") or {}
    return bool(
        -90.0<=deg<=-45.0
        and c.get("positive_qrs_I")
        and c.get("positive_qrs_aVL")
        and int(c.get("inferior_s_dominant_n") or 0)>=2
        and c.get("qrs_lt_120ms")
    )


def _metrics(rows,pred):
    tp=fn=fp=tn=0
    for r in rows:
        y=bool(r["reference"]); p=bool(pred(r))
        if y and p: tp+=1
        elif y and not p: fn+=1
        elif (not y) and p: fp+=1
        else: tn+=1
    return {
        "n":len(rows),"tp":tp,"fn":fn,"fp":fp,"tn":tn,
        "sensitivity":tp/(tp+fn) if tp+fn else None,
        "specificity":tn/(tn+fp) if tn+fp else None,
    }


def _evaluate(rows):
    return {
        "baseline_current_final":_metrics(rows,lambda r:r["current_final"]),
        "frozen_policy_standalone":_metrics(rows,lambda r:r["policy"]),
        "projected_baseline_or_policy":_metrics(rows,lambda r:r["current_final"] or r["policy"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-lafb-axis-confirm"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_LAFB_AXIS_CONFIRM.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    spec=TARGETS[TARGET]

    fold9=_adult_rows(meta,[9])
    f9_pos=fold9[fold9["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    f9_neg=fold9[
        ~fold9["_codes"].map(
            lambda x:any(_target_positive(x,TARGETS[t]["scp"]) for t in TARGETS)
        )
    ].copy().sort_values(["_hash","ecg_id"]).head(160)
    f9=pd.concat([f9_pos,f9_neg],ignore_index=True).drop_duplicates(subset=["ecg_id"])

    manifest=json.loads(Path("ecg_fast_gate_100_manifest.json").read_text())
    fast_ids={int(r["ecg_id"]) for r in manifest["cases"]}
    dev=_adult_rows(meta,[1,2,3,4,5,6,7,8])
    fg=dev[dev["ecg_id"].astype(int).isin(fast_ids)].copy()

    out={"fold9":[],"fast_gate_100":[]}
    errors=[]; root=args.workdir/"records"; expected=set(spec["medcalc"])
    for label,frame in [("fold9",f9),("fast_gate_100",fg)]:
        for i,row in frame.iterrows():
            ecg_id=int(row["ecg_id"])
            try:
                local=_ensure_record(root,str(row["filename_hr"]))
                rec=wfdb.rdrecord(str(local))
                analysis=analyze_canonical_ecg(_canonical(
                    rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
                ))
                out[label].append({
                    "reference":_target_positive(dict(row["_codes"]),spec["scp"]),
                    "current_final":bool(expected & set(_published_codes(analysis))),
                    "policy":_policy(analysis),
                })
            except Exception as exc:
                errors.append({"set":label,"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
            if (i+1)%25==0:
                print(f"MEDCALC_LAFB_AXIS_CONFIRM {label} {i+1}/{len(frame)}",flush=True)

    result={
        "version":"MEDCALC_LAFB_AXIS_CONFIRMATION_V1",
        "role":"POST_SELECTION_CONFIRMATION_ONLY",
        "policy_frozen_before_confirmation":True,
        "policy":"LS_VECTOR_AXIS_GE4_LIMB_LEADS_-90_TO_-45_PLUS_EXISTING_LAFB_MORPHOLOGY_AND_QRS_LT120",
        "tuning_data_used_in_this_run":False,
        "fold9":_evaluate(out["fold9"]),
        "fast_gate_100":_evaluate(out["fast_gate_100"]),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"LAFB axis confirmation had {len(errors)} errors")


if __name__=="__main__":
    main()
