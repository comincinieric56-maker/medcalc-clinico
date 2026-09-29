from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _fast_gate_holdout_ids, _hash,
    _published_codes, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
CODE="LBBB_MORPHOLOGY_COMPATIBLE"


def _policies(analysis):
    c=dict((analysis.get("crosslead_conduction") or {}).get("criteria") or {})
    qrs=bool(c.get("multilead_qrs_ge_120_rescue"))
    global_qrs=False
    try:
        g=((analysis.get("feature_graph") or {}).get("global") or {}).get("qrs_ms") or {}
        global_qrs=float(g.get("value"))>=120.0
    except Exception:
        global_qrs=False
    qrs_wide=bool(qrs or global_qrs)
    lat_r=bool(c.get("lbbb_key_lateral_r"))
    q_abs=bool(c.get("lbbb_key_lateral_absent_q"))
    delay=bool(c.get("lbbb_delayed_or_notched_lateral"))
    v12=bool(c.get("lbbb_v1_v2_negative"))
    return {
        "QRS_PLUS_LATERAL_R_PLUS_Q_ABSENT": qrs_wide and lat_r and q_abs,
        "QRS_PLUS_LATERAL_R_PLUS_DELAY": qrs_wide and lat_r and delay,
        "QRS_PLUS_LATERAL_R_PLUS_Q_ABSENT_OR_DELAY": (
            qrs_wide and lat_r and (q_abs or delay)
        ),
        "QRS_PLUS_LATERAL_R_PLUS_V1V2_NEGATIVE": qrs_wide and lat_r and v12,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-lbbb-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_LBBB_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["LBBB"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"])
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    neg_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

    expected=set(spec["medcalc"])
    policies={k:{"recoverable_positive_n":0,"incremental_negative_trigger_n":0}
              for k in _policies({})}
    baseline_tp=baseline_fp=0
    pos_n=len(pos); neg_n=len(neg)
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
            hits=_policies(analysis)
            is_pos=_target_positive(dict(row["_codes"]),spec["scp"])
            if is_pos:
                baseline_tp+=int(final)
                for name,hit in hits.items():
                    policies[name]["recoverable_positive_n"]+=int((not final) and hit)
            elif ecg_id in neg_ids:
                baseline_fp+=int(final)
                for name,hit in hits.items():
                    policies[name]["incremental_negative_trigger_n"]+=int((not final) and hit)
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_LBBB_AUDIT {i+1}/{len(selected)}",flush=True)

    out={}
    for name,row in policies.items():
        tp=baseline_tp+row["recoverable_positive_n"]
        fp=baseline_fp+row["incremental_negative_trigger_n"]
        out[name]={
            **row,
            "projected_final_positive_n":tp,
            "projected_final_sensitivity":tp/pos_n if pos_n else None,
            "projected_negative_fp_n":fp,
            "projected_specificity":(neg_n-fp)/neg_n if neg_n else None,
        }

    result={
        "version":"MEDCALC_LBBB_MORPHOLOGY_SUPPORT_AUDIT_V1",
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
        raise SystemExit(f"LBBB audit had {len(errors)} errors")


if __name__=="__main__":
    main()
