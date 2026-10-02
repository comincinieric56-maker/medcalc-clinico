from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _pr_multilead_audit,
    _published_codes,
    _target_positive,
    select_records,
)
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB1_SPECIALIST_MULTILEAD_AUDIT_V1"
CODE="FIRST_DEGREE_AV_DELAY_COMPATIBLE"
STRONG={"1_TO_1_P_QRS","PR_MEDIAN_GT_200MS","PR_STABLE"}


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _pr_only_block(fusion:dict)->bool:
    if bool(fusion.get("publishable")):
        return False
    gate=dict(fusion.get("domain_gate") or {})
    if not bool(gate.get("eligible")) or gate.get("blocked_by_conflicts"):
        return False
    unresolved={str(x) for x in (fusion.get("unresolved_required_measurements") or [])}
    boundary=[dict(x or {}) for x in (fusion.get("boundary_failures") or [])]
    reason=str(fusion.get("fusion_reason") or "")
    if reason=="REQUIRED_MEASUREMENT_UNUSABLE":
        return unresolved=={"pr_ms"}
    if reason=="REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED":
        metrics={str(x.get("metric") or "") for x in boundary if str(x.get("metric") or "")}
        return bool(metrics) and metrics=={"pr_ms"} and not unresolved
    return False


def _apply(dst:Counter,a:dict)->None:
    cand=dict((((a.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE)) or {})
    fused=dict((((a.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE)) or {})
    published=CODE in _published_codes(a)
    ev={str(x) for x in (cand.get("evidence") or [])}
    strong=bool(cand and cand.get("specialist_confirmed") and STRONG.issubset(ev))
    blocked=bool(cand and _pr_only_block(fused))
    base_rescue=bool(strong and blocked and not published)
    ml=_pr_multilead_audit(a)
    ge2=bool(ml.get("ge2_pr_gt_200_leads"))
    median=ml.get("usable_lead_median_ms")
    median_gt200=bool(median is not None and float(median)>200.0)

    dst["n"]+=1
    dst["baseline_final_n"]+=int(published)
    dst["strong_signature_n"]+=int(strong)
    dst["strong_pr_only_block_n"]+=int(base_rescue)
    dst["ge2_long_pr_n"]+=int(ge2)
    dst["multilead_median_gt200_n"]+=int(median_gt200)
    dst["rescue_ge2_long_pr_n"]+=int(base_rescue and ge2)
    dst["rescue_ge2_and_median_gt200_n"]+=int(base_rescue and ge2 and median_gt200)


def run(workdir:Path,output:Path,process_fold:int)->dict:
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"; statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)
    meta=pd.read_csv(meta_path)
    selected,summary=select_records(meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids())
    neg_ids={int(x) for x in summary["negative_control_ecg_ids"]}
    aliases=set(TARGETS["AVB1"]["scp"])
    rows=selected[
        (selected["_fold"].astype(int)==int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(neg_ids)
            | selected["_codes"].map(lambda c:_target_positive(c,aliases))
        )
    ].copy()

    groups={"AVB1":Counter(),"CONTROL":Counter()}; errors=Counter(); root=workdir/"records"
    for _,row in rows.iterrows():
        ecg_id=int(row["ecg_id"]); group="CONTROL" if ecg_id in neg_ids else "AVB1"
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),ecg_id)
            _apply(groups[group],analyze_canonical_ecg(canonical))
        except Exception as exc:
            errors[f"{group}:{type(exc).__name__}"]+=1

    out={
        "version":VERSION,"dataset":"PTB-XL","role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,"process_fold":int(process_fold),
        "fast_gate_holdout_excluded":True,"external_validation_claim_allowed":False,
        "groups":{k:dict(v) for k,v in groups.items()},
        "analysis_error_n":sum(errors.values()),"analysis_error_types":dict(errors),
        "policy":"AUDIT_ONLY; EXISTING_GE2_PR_GT_200_LEADS_AND_CONF_GE_0_50; EXISTING_200MS_THRESHOLD; FOLDS_1_TO_8_ONLY; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL; NO_CLINICAL_OUTPUT_CHANGE",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--process-fold",type=int,choices=[1,2,3,4,5,6,7,8],required=True)
    args=ap.parse_args()
    run(args.workdir,args.output,args.process_fold)
