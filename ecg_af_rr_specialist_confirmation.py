from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _canonical, _download, _ensure_record,
    _hash, _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg

AF_SCP=TARGETS["AF"]["scp"]


def _policy(analysis):
    specialists=analysis.get("specialist_evidence") or {}
    mech=specialists.get("atrial_mechanism") or {}
    mechanism=str(mech.get("mechanism") or "")
    rr=float(
        ((mech.get("aggregate_features") or {}).get("rr_irregularity_score"))
        or ((analysis.get("rhythm") or {}).get("rr_irregularity_score"))
        or 0.0
    )
    return bool(mechanism=="AF_COMPATIBLE" and rr>=0.45)


def _evaluate(rows):
    tp=fn=fp=tn=0
    for r in rows:
        y=bool(r["reference_af"])
        p=bool(r["policy_hit"])
        if y and p: tp+=1
        elif y and not p: fn+=1
        elif (not y) and p: fp+=1
        else: tn+=1
    sens=tp/(tp+fn) if tp+fn else None
    spec=tn/(tn+fp) if tn+fp else None
    return {
        "n":len(rows),"tp":tp,"fn":fn,"fp":fp,"tn":tn,
        "sensitivity":sens,"specificity":spec,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-af-confirm"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AF_CONFIRM.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)

    # Fold 9 confirmation: all AF positives + 400 deterministic clean controls.
    fold9=_adult_rows(meta,[9])
    f9_pos=fold9[fold9["_codes"].map(lambda x:_target_positive(x,AF_SCP))].copy()
    f9_neg=fold9[~fold9["_codes"].map(lambda x:any(_target_positive(x,TARGETS[t]["scp"]) for t in TARGETS))].copy()
    f9_neg=f9_neg.sort_values(["_hash","ecg_id"]).head(400)
    f9_sel=pd.concat([f9_pos,f9_neg],ignore_index=True).drop_duplicates(subset=["ecg_id"])

    # Frozen FAST-GATE-100: evaluate AF reference from actual PTB-XL labels across all 100.
    manifest=json.loads(Path("ecg_fast_gate_100_manifest.json").read_text())
    fast_ids={int(r["ecg_id"]) for r in manifest["cases"]}
    dev=_adult_rows(meta,[1,2,3,4,5,6,7,8])
    fg=dev[dev["ecg_id"].astype(int).isin(fast_ids)].copy()

    out_rows={"fold9":[],"fast_gate_100":[]}
    errors=[]
    root=args.workdir/"records"
    for label,frame in [("fold9",f9_sel),("fast_gate_100",fg)]:
        for i,row in frame.iterrows():
            ecg_id=int(row["ecg_id"])
            try:
                local=_ensure_record(root,str(row["filename_hr"]))
                rec=wfdb.rdrecord(str(local))
                analysis=analyze_canonical_ecg(_canonical(
                    rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id
                ))
                out_rows[label].append({
                    "reference_af":_target_positive(dict(row["_codes"]),AF_SCP),
                    "policy_hit":_policy(analysis),
                })
            except Exception as exc:
                errors.append({"set":label,"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
            if (i+1)%25==0:
                print(f"MEDCALC_AF_CONFIRM {label} {i+1}/{len(frame)}",flush=True)

    result={
        "version":"MEDCALC_AF_RR_SPECIALIST_CONFIRMATION_V1",
        "role":"POST_SELECTION_CONFIRMATION_ONLY",
        "policy_frozen_before_confirmation":True,
        "policy":"ATRIAL_MECHANISM_AF_COMPATIBLE_AND_RR_IRREGULARITY_GE_0_45",
        "tuning_data_used_in_this_run":False,
        "fold9":_evaluate(out_rows["fold9"]),
        "fast_gate_100":_evaluate(out_rows["fast_gate_100"]),
        "analysis_error_n":len(errors),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"AF confirmation had {len(errors)} errors")


if __name__=="__main__":
    main()
