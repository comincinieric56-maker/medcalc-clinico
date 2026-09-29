from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _fast_gate_holdout_ids, _hash,
    _published_codes, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=80
CODE="SINUS_TACHYCARDIA_COMPATIBLE"


def _audit(analysis:dict[str,Any])->dict[str,Any]:
    g=dict(analysis.get("global") or {})
    hr=dict(g.get("heart_rate_bpm") or {})
    rel=dict(analysis.get("relations") or {})
    atrial=dict((analysis.get("specialist_evidence") or {}).get("atrial_activity") or {})
    mech=dict((analysis.get("specialist_evidence") or {}).get("atrial_mechanism") or {})
    cand=dict(((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE) or {})
    return {
        "heart_rate_bpm":hr.get("value"),
        "heart_rate_confidence":hr.get("confidence"),
        "p_reproducible":bool(rel.get("p_reproducible")),
        "p_qrs_coupling_fraction":rel.get("p_qrs_coupling_fraction"),
        "sinus_compatible":bool(atrial.get("sinus_compatible")),
        "atrial_mechanism":str(mech.get("mechanism") or ""),
        "atrial_mechanism_confidence":mech.get("confidence"),
        "candidate_present":bool(cand),
        "candidate_score":cand.get("score"),
        "candidate_evidence":sorted(str(x) for x in (cand.get("evidence") or [])),
    }


def _summary(rows:list[dict[str,Any]],positive:bool)->dict[str,Any]:
    z=[r for r in rows if bool(r["is_positive"])==positive]
    def n(pred): return sum(1 for r in z if pred(r["audit"]))
    hrs=[float(r["audit"]["heart_rate_bpm"]) for r in z if r["audit"].get("heart_rate_bpm") is not None]
    return {
        "n":len(z),
        "hr_gt_100_n":n(lambda a: float(a.get("heart_rate_bpm") or 0)>100.0),
        "p_reproducible_n":n(lambda a: bool(a.get("p_reproducible"))),
        "sinus_compatible_n":n(lambda a: bool(a.get("sinus_compatible"))),
        "candidate_present_n":n(lambda a: bool(a.get("candidate_present"))),
        "hr_min":min(hrs) if hrs else None,
        "hr_median":sorted(hrs)[len(hrs)//2] if hrs else None,
        "hr_max":max(hrs) if hrs else None,
        "mechanism_counts":{
            k:sum(1 for r in z if r["audit"].get("atrial_mechanism")==k)
            for k in sorted(set(r["audit"].get("atrial_mechanism") for r in z))
        },
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-sinus-tachy"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_SINUS_TACHY_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)
    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()
    spec=TARGETS["SINUS_TACHY"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy().sort_values(["_hash","ecg_id"])
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy().sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    rows=[]; errors=[]; root=args.workdir/"records"; expected=set(spec["medcalc"])
    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id))
            rows.append({
                "ecg_id":ecg_id,
                "is_positive":_target_positive(dict(row["_codes"]),spec["scp"]),
                "final_positive":bool(expected & set(_published_codes(analysis))),
                "audit":_audit(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%20==0:
            print(f"MEDCALC_SINUS_TACHY_AUDIT {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_SINUS_TACHY_ANATOMY_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_available_n":len(pos),
        "negative_selected_n":len(neg),
        "positive_summary":_summary(rows,True),
        "negative_summary":_summary(rows,False),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors: raise SystemExit(f"sinus tachy audit had {len(errors)} errors")


if __name__=="__main__":
    main()
