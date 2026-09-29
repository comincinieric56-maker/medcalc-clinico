from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

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


def _flags(av:dict[str,Any])->dict[str,bool]:
    cls=str(av.get("classification") or "")
    p=int(av.get("p_count") or 0)
    q=int(av.get("qrs_count") or 0)
    conducted=int(av.get("conducted_p_n") or 0)
    dropped=int(av.get("nonconducted_p_n") or 0)
    coupling=float(av.get("p_qrs_coupling_fraction") or 0.0)
    atrial_regular=bool(av.get("atrial_sequence_regular"))
    ventricular_regular=bool(av.get("ventricular_sequence_regular"))
    stable_pr=bool(av.get("stable_pr"))
    dissoc=bool(av.get("av_dissociation_phase"))
    ar=av.get("atrial_rate_bpm")
    vr=av.get("ventricular_rate_bpm")
    try:
        faster_125=bool(ar is not None and vr is not None and float(ar)>1.25*float(vr))
    except Exception:
        faster_125=False
    highgrade_input=bool(atrial_regular and dropped>=1 and conducted>=2)
    complete_core=bool(
        atrial_regular and ventricular_regular
        and p>=5 and q>=3 and faster_125
    )
    return {
        "AV_EVALUABLE":bool(av.get("evaluable")),
        "FIRST_DEGREE_CLASSIFIED":cls=="FIRST_DEGREE_AV_DELAY_COMPATIBLE",
        "NO_HIGH_GRADE_CLASSIFIED":cls=="NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED",
        "ANY_HIGH_GRADE_CLASSIFIED":cls in {
            "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
            "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
            "MOBITZ_II_COMPATIBLE",
            "MOBITZ_I_WENCKEBACH_COMPATIBLE",
            "COMPLETE_AV_BLOCK_COMPATIBLE",
        },
        "DROPPED_GE1":dropped>=1,
        "CONDUCTED_GE2":conducted>=2,
        "ATRIAL_REGULAR":atrial_regular,
        "VENTRICULAR_REGULAR":ventricular_regular,
        "STABLE_PR":stable_pr,
        "AV_DISSOCIATION_PHASE":dissoc,
        "ATRIAL_RATE_GT_1_25_VENTRICULAR":faster_125,
        "P_QRS_COUNT_DIFF_GE1":abs(p-q)>=1,
        "P_QRS_COUNT_DIFF_GE2":abs(p-q)>=2,
        "COUPLING_LT_0_80":coupling<0.80,
        "HIGH_GRADE_INPUT_PATTERN":highgrade_input,
        "HIGH_GRADE_INPUT_BUT_NO_HIGH_GRADE_CLASS":bool(
            highgrade_input and cls=="NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"
        ),
        "COMPLETE_CORE_PATTERN":complete_core,
        "COMPLETE_CORE_FAILS_PHASE":bool(complete_core and not dissoc),
        "COMPLETE_CORE_BLOCKED_BY_STABLE_PR":bool(complete_core and stable_pr),
        "FIRST_DEGREE_WITH_COUNT_DIFF":bool(
            cls=="FIRST_DEGREE_AV_DELAY_COMPATIBLE" and abs(p-q)>=1
        ),
        "FIRST_DEGREE_WITH_LOW_COUPLING":bool(
            cls=="FIRST_DEGREE_AV_DELAY_COMPATIBLE" and coupling<0.80
        ),
    }


def _summ(rows:list[dict[str,Any]], group:str)->dict[str,Any]:
    z=[r for r in rows if r["group"]==group]
    names=sorted(next(iter(z))["flags"]) if z else []
    return {
        "n":len(z),
        "classification_counts":dict(sorted(Counter(r["classification"] for r in z).items())),
        "flag_counts":{name:sum(bool(r["flags"][name]) for r in z) for name in names},
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-av-map-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AV_MAPPING_AUDIT.json"))
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
        z=z.sort_values(["_hash","ecg_id"])
        z["_group"]=target
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
            av=dict(analysis.get("av_conduction") or {})
            rows.append({
                "group":str(row["_group"]),
                "classification":str(av.get("classification") or ""),
                "flags":_flags(av),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_AV_MAPPING_AUDIT {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_AV_MAPPING_CONTRADICTIONS_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "AVB2":_summ(rows,"AVB2"),
        "AVB3":_summ(rows,"AVB3"),
        "clean_control":_summ(rows,"CLEAN_CONTROL"),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
        "note":"Descriptive anatomy only; no threshold or publication rule is changed.",
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AV mapping audit had {len(errors)} errors")


if __name__=="__main__":
    main()
