from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _adult_rows,
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


def _lead_observability(analysis:dict[str,Any])->dict[str,Any]:
    leads=dict(analysis.get("leads") or {})
    counts=Counter()
    for item_raw in leads.values():
        item=dict(item_raw or {})
        p=np.unique(np.asarray(item.get("raw_p_peaks_samples") or [],dtype=int))
        r=np.unique(np.asarray(item.get("r_peaks_samples") or [],dtype=int))
        if item.get("evaluable"):
            counts["lead_evaluable_n"]+=1
        if len(p)>=4:
            counts["p_ge4_n"]+=1
        if len(r)>=3:
            counts["r_ge3_n"]+=1
        if item.get("evaluable") and len(p)>=4 and len(r)>=3:
            counts["av_input_eligible_n"]+=1
            atrial=dict(item.get("atrial_activity") or {})
            if atrial.get("p_wave_reproducible"):
                counts["eligible_p_repro_n"]+=1
            fs=int(item.get("fs") or 500)
            pp=np.diff(p)*1000.0/max(fs,1)
            pp_cv=(
                float(np.std(pp,ddof=1)/np.mean(pp))
                if pp.size>=2 and float(np.mean(pp))>0
                else None
            )
            if pp_cv is not None and pp_cv<=0.12:
                counts["eligible_organized_p_n"]+=1
    return dict(counts)


def _summarize(rows:list[dict[str,Any]],target:str)->dict[str,Any]:
    spec=TARGETS[target]
    expected=set(spec["medcalc"])
    positives=[
        r for r in rows
        if _target_positive(r.get("codes") or {},spec["scp"])
    ]
    av_class=Counter()
    av_reason=Counter()
    eval_n=0
    final_n=0
    any_eligible_n=0
    ge2_eligible_n=0
    any_repro_n=0
    any_org_n=0
    non_eval_with_input=0

    for r in positives:
        av=dict(r.get("av") or {})
        obs=dict(r.get("obs") or {})
        if av.get("evaluable"):
            eval_n+=1
        av_class[str(av.get("classification") or "MISSING")]+=1
        if str(av.get("reason") or ""):
            av_reason[str(av.get("reason"))]+=1
        final_n+=int(bool(expected & set(r.get("published_codes") or [])))
        eligible=int(obs.get("av_input_eligible_n") or 0)
        repro=int(obs.get("eligible_p_repro_n") or 0)
        org=int(obs.get("eligible_organized_p_n") or 0)
        any_eligible_n+=int(eligible>=1)
        ge2_eligible_n+=int(eligible>=2)
        any_repro_n+=int(repro>=1)
        any_org_n+=int(org>=1)
        non_eval_with_input+=int((not av.get("evaluable")) and eligible>=1)

    return {
        "positive_n":len(positives),
        "baseline_final_positive_n":final_n,
        "av_evaluable_n":eval_n,
        "any_av_input_eligible_lead_n_cases":any_eligible_n,
        "ge2_av_input_eligible_leads_n_cases":ge2_eligible_n,
        "any_reproducible_p_eligible_lead_n_cases":any_repro_n,
        "any_organized_p_eligible_lead_n_cases":any_org_n,
        "non_evaluable_despite_eligible_input_n":non_eval_with_input,
        "classification_counts":dict(sorted(av_class.items())),
        "non_evaluable_reason_counts":dict(sorted(av_reason.items())),
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-av-failure-anatomy"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AV_FAILURE_ANATOMY.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    parts=[]
    availability={}
    for target in ("AVB2","AVB3"):
        spec=TARGETS[target]
        pos=adult[
            adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))
        ].copy()
        pos=pos.sort_values(["_hash","ecg_id"])
        availability[target]=len(pos)
        parts.append(pos)

    selected=pd.concat(parts,ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()
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
                "published_codes":sorted(_published_codes(analysis)),
                "av":dict(analysis.get("av_conduction") or {}),
                "obs":_lead_observability(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        print(f"MEDCALC_AV_FAILURE_ANATOMY {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_FAILURE_ANATOMY_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_available_by_target":availability,
        "records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "AVB2":_summarize(rows,"AVB2"),
        "AVB3":_summarize(rows,"AVB3"),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AV failure anatomy had {len(errors)} analysis errors")


if __name__=="__main__":
    main()
