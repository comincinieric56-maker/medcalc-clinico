from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    PTBXL_VERSION,
    TARGETS,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _target_positive,
    select_records,
)
from ecg_signal_measurements import analyze_canonical_ecg

VERSION = "MEDCALC_AVB2_REAL_DEV_TRANSFER_AUDIT_V1"

def _retry(fn, *args, attempts: int = 4):
    last=None
    for attempt in range(1, attempts+1):
        try:
            return fn(*args)
        except Exception as exc:
            last=exc
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)
    raise last


def _median(values):
    vals=[float(x) for x in values if x is not None and np.isfinite(float(x))]
    return float(np.median(vals)) if vals else None


def run(workdir: Path, output: Path, process_fold: int | None = None) -> dict:
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path=workdir/"ptbxl_database.csv"
    statements_path=workdir/"scp_statements.csv"
    _retry(_download, f"{BASE}/ptbxl_database.csv", metadata_path)
    _retry(_download, f"{BASE}/scp_statements.csv", statements_path)

    meta=pd.read_csv(metadata_path)
    selected, selection=select_records(
        meta,
        folds=folds,
        exclude_ecg_ids=_fast_gate_holdout_ids(),
    )
    aliases=set(TARGETS["AVB2"]["scp"])
    positives=selected[selected["_codes"].map(lambda c:_target_positive(c,aliases))].copy()
    if process_fold is not None:
        positives=positives[positives["_fold"].astype(int)==int(process_fold)].copy()

    counts=Counter()
    legacy_class=Counter()
    fusion_reason=Counter()
    fusion_state=Counter()
    domain_block=Counter()
    domain_unusable=Counter()
    pp_cv=[]
    p_count=[]
    qrs_count=[]
    errors=[]
    records_root=workdir/"records"

    for _,row in positives.iterrows():
        try:
            base=_retry(_ensure_record, records_root, str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),int(row["ecg_id"]))
            a=analyze_canonical_ecg(canonical)

            ev=dict(a.get("avb2_recovered_sequence") or {})
            counts["n"]+=1
            counts["recovered_evaluable_n"]+=int(bool(ev.get("evaluable")))
            counts["recovered_compatible_n"]+=int(bool(ev.get("compatible")))
            counts["atrial_regular_n"]+=int(bool(ev.get("atrial_sequence_regular")))
            counts["ge2_nonconducted_n"]+=int(int(ev.get("max_consecutive_nonconducted_p") or 0)>=2)
            counts["midrr_majority_n"]+=int(bool(ev.get("dropped_mid_rr_majority")))
            mask=dict(ev.get("recovery_mask") or {})
            counts["clean_r_mask_evaluable_n"]+=int(bool(mask.get("evaluable")))
            counts["clean_r_mask_changed_n"]+=int(bool(mask.get("changed")))
            if ev.get("pp_cv") is not None: pp_cv.append(float(ev["pp_cv"]))
            if ev.get("p_count") is not None: p_count.append(int(ev["p_count"]))
            if ev.get("qrs_count") is not None: qrs_count.append(int(ev["qrs_count"]))

            legacy=dict(a.get("av_conduction") or {})
            legacy_class[str(legacy.get("classification") or "NONE")]+=1

            cand=((a.get("high_recall_candidates") or {}).get("by_code") or {}).get("HIGH_GRADE_AV_BLOCK_COMPATIBLE")
            counts["candidate_high_grade_n"]+=int(bool(cand))

            fused=((a.get("evidence_fusion") or {}).get("by_code") or {}).get("HIGH_GRADE_AV_BLOCK_COMPATIBLE") or {}
            counts["fusion_high_grade_n"]+=int(bool(fused.get("publishable")))
            if cand and not fused.get("publishable"):
                fusion_reason[str(fused.get("fusion_reason") or "NONE")]+=1
                fusion_state[str(fused.get("fusion_state") or "NONE")]+=1

            gate=(((a.get("domain_gates") or {}).get("domains") or {}).get("AV_CONDUCTION") or {})
            counts["av_domain_eligible_n"]+=int(bool(gate.get("eligible")))
            for x in gate.get("blocked_by_conflicts") or []: domain_block[str(x)]+=1
            for x in gate.get("unusable_measurements") or []: domain_unusable[str(x)]+=1

            findings=(((a.get("specialist_reasoning") or {}).get("diagnostic_summary") or {}).get("findings") or [])
            final=any(
                str(x.get("code") or "")=="HIGH_GRADE_AV_BLOCK_COMPATIBLE"
                and bool(x.get("publishable"))
                for x in findings
            )
            counts["final_high_grade_n"]+=int(final)
        except Exception as exc:
            errors.append(type(exc).__name__)

    out={
        "version":VERSION,
        "dataset":"PTB-XL",
        "role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,
        "process_fold":process_fold,
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "counts":dict(counts),
        "legacy_av_classification_counts":dict(legacy_class),
        "fusion_suppression_reasons":dict(fusion_reason),
        "fusion_suppression_states":dict(fusion_state),
        "av_domain_blocked_conflicts":dict(domain_block),
        "av_domain_unusable_measurements":dict(domain_unusable),
        "medians":{
            "pp_cv":_median(pp_cv),
            "p_count":_median(p_count),
            "qrs_count":_median(qrs_count),
        },
        "analysis_error_n":len(errors),
        "analysis_error_types":dict(Counter(errors)),
        "selection_avb2_positive_n":int(len(positives)),
        "policy":"AGGREGATE_AUDIT_ONLY; FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; NO_EXTERNAL_OR_FINAL_DATA; NO_THRESHOLD_TUNING",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-avb2-transfer-targeted"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AVB2_TRANSFER_TARGETED.json"))
    ap.add_argument("--process-fold",type=int,choices=[1,2,3,4,5,6,7,8],default=None)
    args=ap.parse_args()
    run(args.workdir,args.output,process_fold=args.process_fold)
