from __future__ import annotations

import argparse
import json
from collections import Counter
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
CODE="FLUTTER_OR_AT_COMPATIBLE"


def _audit(analysis:dict[str,Any])->dict[str,Any]:
    fused=dict(((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE) or {})
    reason=dict(analysis.get("specialist_reasoning") or {})
    primary=dict(reason.get("primary_rhythm") or {})
    final_codes=set(_published_codes(analysis))
    return {
        "fusion_present":bool(fused),
        "fusion_publishable":bool(fused.get("publishable")),
        "fusion_score":float(fused.get("score") or 0.0),
        "fusion_state":str(fused.get("fusion_state") or ""),
        "fusion_reason":str(fused.get("fusion_reason") or ""),
        "final_present":CODE in final_codes,
        "primary_rhythm":str(primary.get("code") or ""),
        "primary_confidence":float(primary.get("confidence") or 0.0),
    }


def _summary(rows:list[dict[str,Any]],group:str)->dict[str,Any]:
    z=[r for r in rows if r["group"]==group]
    fused=[r for r in z if r["audit"]["fusion_publishable"]]
    lost=[r for r in fused if not r["audit"]["final_present"]]
    return {
        "n":len(z),
        "fusion_publishable_n":len(fused),
        "final_positive_n":sum(r["audit"]["final_present"] for r in z),
        "fusion_to_final_loss_n":len(lost),
        "loss_primary_rhythm_counts":dict(sorted(Counter(
            r["audit"]["primary_rhythm"] for r in lost
        ).items())),
        "loss_fusion_state_counts":dict(sorted(Counter(
            r["audit"]["fusion_state"] for r in lost
        ).items())),
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-flutter-final"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_FLUTTER_FINAL_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    flutter=TARGETS["FLUTTER"]
    af=TARGETS["AF"]

    pos=adult[adult["_codes"].map(
        lambda x:_target_positive(x,flutter["scp"])
    )].copy().sort_values(["_hash","ecg_id"])

    af_only=adult[
        adult["_codes"].map(lambda x:_target_positive(x,af["scp"]))
        & ~adult["_codes"].map(lambda x:_target_positive(x,flutter["scp"]))
    ].copy().sort_values(["_hash","ecg_id"])

    clean=adult[~adult["_codes"].map(_any_target_positive)].copy()
    clean=clean.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)

    frames=[]
    for group,df in [("FLUTTER_POSITIVE",pos),("AF_REFERENCE",af_only),("CLEAN_CONTROL",clean)]:
        x=df.copy()
        x["_group"]=group
        frames.append(x)
    selected=pd.concat(frames,ignore_index=True).drop_duplicates(subset=["ecg_id","_group"])

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
                "group":str(row["_group"]),
                "audit":_audit(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_FLUTTER_FINAL_AUDIT {i+1}/{len(selected)}",flush=True)

    pos_s=_summary(rows,"FLUTTER_POSITIVE")
    af_s=_summary(rows,"AF_REFERENCE")
    clean_s=_summary(rows,"CLEAN_CONTROL")

    result={
        "version":"MEDCALC_FLUTTER_FUSION_TO_FINAL_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "flutter_positive":pos_s,
        "af_reference":af_s,
        "clean_control":clean_s,
        "counterfactual_preserve_any_fused_flutter_as_secondary":{
            "recoverable_flutter_positive_n":pos_s["fusion_to_final_loss_n"],
            "incremental_clean_control_n":clean_s["fusion_to_final_loss_n"],
            "incremental_af_reference_n":af_s["fusion_to_final_loss_n"],
        },
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"Flutter audit had {len(errors)} errors")


if __name__=="__main__":
    main()
