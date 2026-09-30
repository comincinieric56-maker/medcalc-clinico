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
    _any_target_positive,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg

FOLDS = [1,2,3,4,5,6,7,8]
NEGATIVE_N = 160
CLUSTER_MS = 50.0


def _mapping_flags(av: dict[str, Any]) -> dict[str, bool]:
    cls = str(av.get("classification") or "")
    p = int(av.get("p_count") or 0)
    q = int(av.get("qrs_count") or 0)
    conducted = int(av.get("conducted_p_n") or 0)
    dropped = int(av.get("nonconducted_p_n") or 0)
    coupling = float(av.get("p_qrs_coupling_fraction") or 0.0)
    atrial_regular = bool(av.get("atrial_sequence_regular"))
    ventricular_regular = bool(av.get("ventricular_sequence_regular"))
    stable_pr = bool(av.get("stable_pr"))
    dissoc = bool(av.get("av_dissociation_phase"))
    ar = av.get("atrial_rate_bpm")
    vr = av.get("ventricular_rate_bpm")
    try:
        faster_125 = bool(ar is not None and vr is not None and float(ar) > 1.25 * float(vr))
    except Exception:
        faster_125 = False
    highgrade_input = bool(atrial_regular and dropped >= 1 and conducted >= 2)
    complete_core = bool(
        atrial_regular and ventricular_regular and p >= 5 and q >= 3 and faster_125
    )
    return {
        "AV_EVALUABLE": bool(av.get("evaluable")),
        "FIRST_DEGREE_CLASSIFIED": cls == "FIRST_DEGREE_AV_DELAY_COMPATIBLE",
        "NO_HIGH_GRADE_CLASSIFIED": cls == "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED",
        "ANY_HIGH_GRADE_CLASSIFIED": cls in {
            "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
            "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
            "MOBITZ_II_COMPATIBLE",
            "MOBITZ_I_WENCKEBACH_COMPATIBLE",
            "COMPLETE_AV_BLOCK_COMPATIBLE",
        },
        "DROPPED_GE1": dropped >= 1,
        "CONDUCTED_GE2": conducted >= 2,
        "ATRIAL_REGULAR": atrial_regular,
        "VENTRICULAR_REGULAR": ventricular_regular,
        "STABLE_PR": stable_pr,
        "AV_DISSOCIATION_PHASE": dissoc,
        "ATRIAL_RATE_GT_1_25_VENTRICULAR": faster_125,
        "P_QRS_COUNT_DIFF_GE1": abs(p-q) >= 1,
        "P_QRS_COUNT_DIFF_GE2": abs(p-q) >= 2,
        "COUPLING_LT_0_80": coupling < 0.80,
        "HIGH_GRADE_INPUT_PATTERN": highgrade_input,
        "HIGH_GRADE_INPUT_BUT_NO_HIGH_GRADE_CLASS": bool(
            highgrade_input and cls == "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"
        ),
        "COMPLETE_CORE_PATTERN": complete_core,
        "COMPLETE_CORE_FAILS_PHASE": bool(complete_core and not dissoc),
        "COMPLETE_CORE_BLOCKED_BY_STABLE_PR": bool(complete_core and stable_pr),
        "FIRST_DEGREE_WITH_COUNT_DIFF": bool(
            cls == "FIRST_DEGREE_AV_DELAY_COMPATIBLE" and abs(p-q) >= 1
        ),
        "FIRST_DEGREE_WITH_LOW_COUPLING": bool(
            cls == "FIRST_DEGREE_AV_DELAY_COMPATIBLE" and coupling < 0.80
        ),
    }


def _clusters(analysis: dict[str, Any], support_n: int) -> list[float]:
    leads = analysis.get("leads") or {}
    events: list[tuple[float,str]] = []
    for lead,item in leads.items():
        fs = int(item.get("fs") or analysis.get("fs") or 500)
        for p in item.get("raw_p_peaks_samples") or []:
            events.append((float(p) * 1000.0 / max(fs,1), str(lead)))
    events.sort()
    if not events:
        return []
    groups: list[list[tuple[float,str]]] = []
    cur: list[tuple[float,str]] = []
    for t,lead in events:
        if not cur or t-cur[-1][0] <= CLUSTER_MS:
            cur.append((t,lead))
        else:
            groups.append(cur)
            cur=[(t,lead)]
    if cur:
        groups.append(cur)
    out=[]
    for g in groups:
        by_lead: dict[str,list[float]] = {}
        for t,lead in g:
            by_lead.setdefault(lead,[]).append(t)
        if len(by_lead) < support_n:
            continue
        vals=[float(np.median(v)) for v in by_lead.values()]
        out.append(float(np.median(vals)))
    return out


def _crosslead(analysis: dict[str, Any], support_n: int) -> dict[str, Any]:
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
    rr=np.diff(r)*1000.0/max(fs,1) if r.size>=2 else np.asarray([],dtype=float)
    rr_med=float(np.median(rr)) if rr.size else None
    ventricular_rate=60000.0/rr_med if rr_med and rr_med>0 else None
    atrial_rate=60000.0/pp_med if pp_med and pp_med>0 else None
    ratio=len(p)/max(int(r.size),1)
    rate_gt_125=bool(
        atrial_rate is not None and ventricular_rate is not None
        and atrial_rate > 1.25*ventricular_rate
    )
    av=analysis.get("av_conduction") or {}
    selected_lead=str(av.get("lead") or "")
    selected_count=0
    if selected_lead:
        selected_count=len(
            ((analysis.get("leads") or {}).get(selected_lead) or {})
            .get("raw_p_peaks_samples") or []
        )
    return {
        "evaluable": len(p)>=4,
        "p_count": len(p),
        "qrs_count": int(r.size),
        "regular": regular,
        "approx_2_to_1": bool(1.75<=ratio<=2.25 and len(p)>=4 and r.size>=2),
        "atrial_rate_gt_1_25_ventricular": rate_gt_125,
        "extra_p_ge1": bool(len(p)>=selected_count+1),
        "extra_p_ge2": bool(len(p)>=selected_count+2),
    }


def _row_flags(analysis: dict[str, Any]) -> dict[str, bool]:
    m=_mapping_flags(dict(analysis.get("av_conduction") or {}))
    s2=_crosslead(analysis,2)
    s3=_crosslead(analysis,3)
    out=dict(m)
    out.update({
        "S2_EVALUABLE":s2["evaluable"],
        "S2_REGULAR":s2["regular"],
        "S2_APPROX_2_TO_1":s2["approx_2_to_1"],
        "S2_RATE_GT_1_25":s2["atrial_rate_gt_1_25_ventricular"],
        "S2_EXTRA_P_GE1":s2["extra_p_ge1"],
        "S2_EXTRA_P_GE2":s2["extra_p_ge2"],
        "S3_EVALUABLE":s3["evaluable"],
        "S3_REGULAR":s3["regular"],
        "S3_APPROX_2_TO_1":s3["approx_2_to_1"],
        "S3_RATE_GT_1_25":s3["atrial_rate_gt_1_25_ventricular"],
        "S3_EXTRA_P_GE1":s3["extra_p_ge1"],
        "S3_EXTRA_P_GE2":s3["extra_p_ge2"],
        "S2_REGULAR_AND_RATE_GT_1_25":bool(s2["regular"] and s2["atrial_rate_gt_1_25_ventricular"]),
        "S3_REGULAR_AND_RATE_GT_1_25":bool(s3["regular"] and s3["atrial_rate_gt_1_25_ventricular"]),
        "S2_REGULAR_AND_APPROX_2_TO_1":bool(s2["regular"] and s2["approx_2_to_1"]),
        "S3_REGULAR_AND_APPROX_2_TO_1":bool(s3["regular"] and s3["approx_2_to_1"]),
        "S2_EXTRA_P_GE1_AND_RATE_GT_1_25":bool(s2["extra_p_ge1"] and s2["atrial_rate_gt_1_25_ventricular"]),
        "S3_EXTRA_P_GE1_AND_RATE_GT_1_25":bool(s3["extra_p_ge1"] and s3["atrial_rate_gt_1_25_ventricular"]),
    })
    return out


def _summ(rows: list[dict[str,Any]], group: str) -> dict[str,Any]:
    z=[r for r in rows if group in r["groups"]]
    names=sorted(z[0]["flags"]) if z else []
    return {
        "n":len(z),
        "classification_counts":dict(sorted(Counter(r["classification"] for r in z).items())),
        "flag_counts":{name:sum(bool(r["flags"][name]) for r in z) for name in names},
    }


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    group_by_id: dict[int,set[str]]={}
    selected_ids:set[int]=set()
    for target in ("AVB2","AVB3"):
        spec=TARGETS[target]
        z=adult[adult["_codes"].map(lambda x,a=spec["scp"]:_target_positive(x,a))].copy()
        for ecg_id in z["ecg_id"].astype(int).tolist():
            selected_ids.add(ecg_id)
            group_by_id.setdefault(ecg_id,set()).add(target)

    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    for ecg_id in neg["ecg_id"].astype(int).tolist():
        selected_ids.add(ecg_id)
        group_by_id.setdefault(ecg_id,set()).add("CLEAN_CONTROL")

    selected=adult[adult["ecg_id"].astype(int).isin(selected_ids)].copy()
    selected=selected.sort_values(["_hash","ecg_id"]).drop_duplicates(subset=["ecg_id"])

    rows=[]; errors=[]; root=args.workdir/"records"
    for pos,(_,row) in enumerate(selected.iterrows(),1):
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            av=dict(analysis.get("av_conduction") or {})
            rows.append({
                "groups":sorted(group_by_id.get(ecg_id,set())),
                "classification":str(av.get("classification") or ""),
                "flags":_row_flags(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if pos%25==0:
            print(f"MEDCALC_AV_MECHANISM_AUDIT {pos}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_MECHANISM_COMBINED_AUDIT_V2",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "cluster_window_ms":CLUSTER_MS,
        "unique_records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "AVB2":_summ(rows,"AVB2"),
        "AVB3":_summ(rows,"AVB3"),
        "clean_control":_summ(rows,"CLEAN_CONTROL"),
        "case_level_results_emitted":False,
        "note":"Aggregate mechanism audit only. No detector, threshold, mapping or publication rule changes.",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AV mechanism audit had {len(errors)} analysis errors")


if __name__=="__main__":
    main()
