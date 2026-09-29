from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical, _download,
    _ensure_record, _fast_gate_holdout_ids, _hash, _published_codes,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
CODE="AF_COMPATIBLE"


def _candidate_evidence(analysis:dict[str,Any])->set[str]:
    row=dict(((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE) or {})
    return set(str(x) for x in (row.get("evidence") or []))


def _policy_hits(ev:set[str])->dict[str,bool]:
    rr="RR_IRREGULAR" in ev
    no_p="NO_REPRODUCIBLE_P" in ev
    spec="ATRIAL_SPECIALIST_AF" in ev
    afsig="AF_COMPATIBILITY_SIGNAL" in ev
    return {
        "RR_IRREGULAR": rr,
        "RR_IRREGULAR_PLUS_AF_SIGNAL": rr and afsig,
        "RR_IRREGULAR_PLUS_SPECIALIST": rr and spec,
        "RR_IRREGULAR_PLUS_NO_REPRO_P": rr and no_p,
        "RR_IRREGULAR_PLUS_SPECIALIST_PLUS_NO_REPRO_P": rr and spec and no_p,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-af-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AF_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["AF"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"])
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    neg_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    expected=set(spec["medcalc"])
    policies={}
    baseline_tp=baseline_fp=0
    pos_n=len(pos); neg_n=len(neg)
    for name in _policy_hits(set()):
        policies[name]={
            "positive_hit_n":0,
            "negative_hit_n":0,
            "positive_final_and_hit_n":0,
            "positive_final_without_hit_n":0,
            "positive_nonfinal_hit_n":0,
            "negative_final_and_hit_n":0,
            "negative_final_without_hit_n":0,
            "negative_nonfinal_hit_n":0,
        }
    errors=[]; root=args.workdir/"records"

    for i,row in selected.iterrows():
        ecg_id=int(row["ecg_id"])
        try:
            local=_ensure_record(root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(local))
            analysis=analyze_canonical_ecg(_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
            ))
            final=bool(expected & set(_published_codes(analysis)))
            ev=_candidate_evidence(analysis)
            hits=_policy_hits(ev)
            is_pos=_target_positive(dict(row["_codes"]),spec["scp"])
            if is_pos:
                baseline_tp+=int(final)
                for name,hit in hits.items():
                    p=policies[name]
                    p["positive_hit_n"]+=int(hit)
                    p["positive_final_and_hit_n"]+=int(final and hit)
                    p["positive_final_without_hit_n"]+=int(final and not hit)
                    p["positive_nonfinal_hit_n"]+=int((not final) and hit)
            elif ecg_id in neg_ids:
                baseline_fp+=int(final)
                for name,hit in hits.items():
                    p=policies[name]
                    p["negative_hit_n"]+=int(hit)
                    p["negative_final_and_hit_n"]+=int(final and hit)
                    p["negative_final_without_hit_n"]+=int(final and not hit)
                    p["negative_nonfinal_hit_n"]+=int((not final) and hit)
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_AF_AUDIT {i+1}/{len(selected)}",flush=True)

    out={}
    for name,row in policies.items():
        ph=row["positive_hit_n"]; nh=row["negative_hit_n"]
        out[name]={
            **row,
            "positive_n":pos_n,
            "sensitivity_if_required":ph/pos_n if pos_n else None,
            "negative_control_n":neg_n,
            "specificity_if_required":(neg_n-nh)/neg_n if neg_n else None,
            "projected_require_policy_tp_n":row["positive_final_and_hit_n"],
            "projected_require_policy_sensitivity":(
                row["positive_final_and_hit_n"]/pos_n if pos_n else None
            ),
            "projected_require_policy_fp_n":row["negative_final_and_hit_n"],
            "projected_require_policy_specificity":(
                (neg_n-row["negative_final_and_hit_n"])/neg_n if neg_n else None
            ),
            "projected_require_policy_plus_all_nonfinal_hits_tp_n":(
                row["positive_final_and_hit_n"]+row["positive_nonfinal_hit_n"]
            ),
            "projected_require_policy_plus_all_nonfinal_hits_sensitivity":(
                (
                    row["positive_final_and_hit_n"]+row["positive_nonfinal_hit_n"]
                )/pos_n if pos_n else None
            ),
            "projected_require_policy_plus_all_nonfinal_hits_fp_n":(
                row["negative_final_and_hit_n"]+row["negative_nonfinal_hit_n"]
            ),
            "projected_require_policy_plus_all_nonfinal_hits_specificity":(
                (
                    neg_n
                    - row["negative_final_and_hit_n"]
                    - row["negative_nonfinal_hit_n"]
                )/neg_n if neg_n else None
            ),
        }

    result={
        "version":"MEDCALC_AF_PUBLICATION_SPECIFICITY_AUDIT_V2",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_n":pos_n,
        "negative_control_n":neg_n,
        "baseline_final_positive_n":baseline_tp,
        "baseline_final_sensitivity":baseline_tp/pos_n if pos_n else None,
        "baseline_negative_fp_n":baseline_fp,
        "baseline_specificity":(neg_n-baseline_fp)/neg_n if neg_n else None,
        "policies":out,
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AF audit had {len(errors)} errors")


if __name__=="__main__":
    main()
