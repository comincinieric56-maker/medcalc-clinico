from __future__ import annotations

import argparse
import json
from collections import Counter
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
from ecg_av_conduction import analyze_av_conduction
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
AVB2_CODES={
    "MOBITZ_I_WENCKEBACH_COMPATIBLE",
    "MOBITZ_II_COMPATIBLE",
    "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
    "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
}
AVB3_CODES={"COMPLETE_AV_BLOCK_COMPATIBLE"}


def _per_lead_av_support(analysis: dict[str, Any]) -> dict[str, Any]:
    per_lead=dict(analysis.get("leads") or {})
    global_atrial=dict(analysis.get("atrial_activity") or {})
    global_metrics=dict(analysis.get("global") or {})
    classifications: dict[str,str]={}
    for lead,item_raw in per_lead.items():
        item=dict(item_raw or {})
        if not bool(item.get("evaluable")):
            continue
        p=list(item.get("raw_p_peaks_samples") or [])
        r=list(item.get("r_peaks_samples") or [])
        if len(p)<4 or len(r)<3:
            continue
        result=analyze_av_conduction(
            {str(lead):item},
            global_atrial,
            global_metrics=global_metrics,
        )
        if not bool(result.get("evaluable")):
            continue
        cls=str(result.get("classification") or "")
        if cls and cls!="NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED":
            classifications[str(lead)]=cls

    counts=Counter(classifications.values())
    avb2_same_class_max=max(
        [n for cls,n in counts.items() if cls in AVB2_CODES] or [0]
    )
    avb2_family_n=sum(1 for cls in classifications.values() if cls in AVB2_CODES)
    avb3_n=sum(1 for cls in classifications.values() if cls in AVB3_CODES)
    return {
        "evaluable_lead_n": len([
            lead for lead,item_raw in per_lead.items()
            if bool((item_raw or {}).get("evaluable"))
            and len((item_raw or {}).get("raw_p_peaks_samples") or [])>=4
            and len((item_raw or {}).get("r_peaks_samples") or [])>=3
        ]),
        "avb2_family_support_lead_n": avb2_family_n,
        "avb2_same_class_max_lead_n": avb2_same_class_max,
        "avb2_same_class_ge2": avb2_same_class_max>=2,
        "avb3_support_lead_n": avb3_n,
        "avb3_ge2": avb3_n>=2,
    }


def _score(
    rows:list[dict[str,Any]],
    target:str,
    negative_ids:set[int],
) -> dict[str,Any]:
    spec=TARGETS[target]
    expected=set(spec["medcalc"])
    positives=[
        r for r in rows
        if _target_positive(r.get("codes") or {},spec["scp"])
    ]
    baseline_tp=sum(
        bool(expected & set(r.get("published_codes") or []))
        for r in positives
    )
    baseline_fp=sum(
        bool(expected & set(r.get("published_codes") or []))
        for r in rows if int(r["ecg_id"]) in negative_ids
    )
    trigger_key="avb2_same_class_ge2" if target=="AVB2" else "avb3_ge2"
    recoverable=0
    negative_trigger=0
    for r in rows:
        final_hit=bool(expected & set(r.get("published_codes") or []))
        if final_hit:
            continue
        trigger=bool((r.get("multilead_av") or {}).get(trigger_key))
        if not trigger:
            continue
        if _target_positive(r.get("codes") or {},spec["scp"]):
            recoverable+=1
        elif int(r["ecg_id"]) in negative_ids:
            negative_trigger+=1

    projected_tp=baseline_tp+recoverable
    projected_fp=baseline_fp+negative_trigger
    return {
        "target":target,
        "policy":(
            "GE2_LEADS_SAME_AVB2_CLASSIFICATION"
            if target=="AVB2"
            else "GE2_LEADS_COMPLETE_AV_BLOCK_COMPATIBLE"
        ),
        "positive_n":len(positives),
        "baseline_final_positive_n":baseline_tp,
        "baseline_final_sensitivity":(
            baseline_tp/len(positives) if positives else None
        ),
        "recoverable_positive_n":recoverable,
        "projected_final_positive_n_if_all_triggers_publish":projected_tp,
        "projected_final_sensitivity_if_all_triggers_publish":(
            projected_tp/len(positives) if positives else None
        ),
        "negative_control_n":len(negative_ids),
        "baseline_negative_fp_n":baseline_fp,
        "incremental_negative_trigger_n":negative_trigger,
        "projected_negative_fp_upper_n_if_all_triggers_publish":projected_fp,
        "projected_specificity_lower_bound_if_all_triggers_publish":(
            (len(negative_ids)-projected_fp)/len(negative_ids)
            if negative_ids else None
        ),
        "interpretation":"AGGREGATE_COUNTERFACTUAL_ONLY_NO_DIAGNOSTIC_CHANGE",
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-av-multilead-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AV_MULTILEAD_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    positive_parts=[]
    availability={}
    for target in ("AVB2","AVB3"):
        spec=TARGETS[target]
        pos=adult[
            adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))
        ].copy()
        pos=pos.sort_values(["_hash","ecg_id"])
        availability[target]=len(pos)
        positive_parts.append(pos)

    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    negative_ids=set(int(x) for x in neg["ecg_id"].tolist())

    selected=pd.concat(positive_parts+[neg],ignore_index=True)
    selected=selected.drop_duplicates(subset=["ecg_id"]).copy()
    rows=[]
    errors=[]
    records_root=args.workdir/"records"

    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(records_root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(
                _canonical(
                    rec.p_signal,
                    int(round(float(rec.fs))),
                    list(rec.sig_name),
                    ecg_id,
                )
            )
            rows.append({
                "ecg_id":ecg_id,
                "codes":dict(row["_codes"]),
                "published_codes":sorted(_published_codes(analysis)),
                "multilead_av":_per_lead_av_support(analysis),
            })
        except Exception as exc:
            errors.append({
                "ecg_id":ecg_id,
                "error":f"{type(exc).__name__}:{exc}",
            })
        if (i+1)%25==0:
            print(f"MEDCALC_AV_MULTILEAD_AUDIT {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_MULTILEAD_CONCORDANCE_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_available_by_target":availability,
        "negative_selected_n":len(neg),
        "records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "AVB2":_score(rows,"AVB2",negative_ids),
        "AVB3":_score(rows,"AVB3",negative_ids),
        "case_level_results_emitted":False,
    }
    args.output.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AV multilead audit had {len(errors)} analysis errors")


if __name__=="__main__":
    main()
