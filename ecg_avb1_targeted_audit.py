from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _adult_rows,
    _any_target_positive,
    _av_candidate_miss_audit,
    _avb1_compound_counterfactual_audit,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _fusion_codes,
    _hash,
    _parse_codes,
    _pr_multilead_audit,
    _published_codes,
    _candidate_codes,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


POSITIVE_N = 80
NEGATIVE_N = 120
FOLDS = [1,2,3,4,5,6,7,8]


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/medcalc-avb1-targeted-audit"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/MEDCALC_AVB1_TARGETED_AUDIT.json"))
    args=ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv", meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta, FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["AVB1"]
    pos=adult[adult["_codes"].map(lambda x: _target_positive(x, spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"]).head(POSITIVE_N)

    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)

    selected=pd.concat([pos,neg],ignore_index=True)
    selected=selected.drop_duplicates(subset=["ecg_id"]).copy()
    negative_ids=set(int(x) for x in neg["ecg_id"].tolist())

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
                "ecg_id": ecg_id,
                "codes": dict(row["_codes"]),
                "candidate_codes": sorted(_candidate_codes(analysis)),
                "fusion_codes": sorted(_fusion_codes(analysis)),
                "published_codes": sorted(_published_codes(analysis)),
                "av_candidate_miss_audit": _av_candidate_miss_audit(analysis),
                "pr_multilead_audit": _pr_multilead_audit(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_AVB1_TARGETED_AUDIT {i+1}/{len(selected)}",flush=True)

    audit=_avb1_compound_counterfactual_audit(rows,negative_ids)
    result={
        "version":"MEDCALC_AVB1_TARGETED_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_selected_n":len(pos),
        "negative_selected_n":len(neg),
        "records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "counterfactual":audit,
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AVB1 targeted audit had {len(errors)} analysis errors")


if __name__=="__main__":
    main()
