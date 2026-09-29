from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

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
    _fusion_codes,
    _hash,
    _published_codes,
    _candidate_codes,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FOLDS=[1,2,3,4,5,6,7,8]
NEGATIVE_N=160
CODE="VENTRICULAR_PREEXCITATION_COMPATIBLE"


def _audit(analysis:dict[str,Any])->dict[str,Any]:
    pre=dict(analysis.get("preexcitation") or {})
    crit=dict(pre.get("criteria") or {})
    cand=dict(((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE) or {})
    fused=dict(((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE) or {})
    gate=dict(fused.get("domain_gate") or {})
    return {
        "pre_classification":str(pre.get("classification") or ""),
        "global_path":bool(crit.get("global_pr_qrs_path")),
        "multilead_rescue":bool(crit.get("multilead_short_pr_delta_rescue")),
        "concordant_lead_n":len(crit.get("concordant_short_pr_delta_leads") or []),
        "short_pr_lead_n":len(crit.get("short_pr_leads") or []),
        "delta_lead_n":len(crit.get("delta_slur_leads") or []),
        "rescue_delta_wide_lead_n":len(crit.get("rescue_delta_wide_leads") or []),
        "candidate_present":bool(cand),
        "candidate_score":float(cand.get("score") or 0.0),
        "candidate_source_n":int(cand.get("independent_evidence_n") or 0),
        "specialist_confirmed":bool(cand.get("specialist_confirmed")),
        "fusion_present":bool(fused),
        "fusion_publishable":bool(fused.get("publishable")),
        "fusion_reason":str(fused.get("fusion_reason") or ""),
        "fusion_state":str(fused.get("fusion_state") or ""),
        "unresolved_required_measurements":sorted(str(x) for x in (fused.get("unresolved_required_measurements") or [])),
        "boundary_failure_metrics":sorted(
            str(x.get("metric") or "") for x in (fused.get("boundary_failures") or [])
            if str(x.get("metric") or "")
        ),
        "gate_eligible":bool(gate.get("eligible")),
    }


def _summarize(rows:list[dict[str,Any]],negative_ids:set[int])->dict[str,Any]:
    spec=TARGETS["WPW"]
    expected=set(spec["medcalc"])
    positives=[r for r in rows if _target_positive(r.get("codes") or {},spec["scp"])]
    losses=[
        r for r in positives
        if bool(expected & set(r.get("candidate_codes") or []))
        and not bool(expected & set(r.get("fusion_codes") or []))
    ]
    negatives=[r for r in rows if int(r["ecg_id"]) in negative_ids]

    def n(rows_,key): return sum(bool((r.get("wpw_audit") or {}).get(key)) for r in rows_)
    loss_reasons=Counter(str((r.get("wpw_audit") or {}).get("fusion_reason") or "") for r in losses)
    loss_states=Counter(str((r.get("wpw_audit") or {}).get("fusion_state") or "") for r in losses)
    loss_boundary=Counter()
    loss_unresolved=Counter()
    for r in losses:
        a=r.get("wpw_audit") or {}
        loss_boundary.update(a.get("boundary_failure_metrics") or [])
        loss_unresolved.update(a.get("unresolved_required_measurements") or [])

    return {
        "positive_n":len(positives),
        "candidate_positive_n":sum(bool(expected & set(r.get("candidate_codes") or [])) for r in positives),
        "fusion_positive_n":sum(bool(expected & set(r.get("fusion_codes") or [])) for r in positives),
        "final_positive_n":sum(bool(expected & set(r.get("published_codes") or [])) for r in positives),
        "candidate_to_fusion_loss_n":len(losses),
        "positive_global_path_n":n(positives,"global_path"),
        "positive_multilead_rescue_n":n(positives,"multilead_rescue"),
        "fusion_loss_global_path_n":n(losses,"global_path"),
        "fusion_loss_multilead_rescue_n":n(losses,"multilead_rescue"),
        "negative_control_n":len(negatives),
        "negative_global_path_n":n(negatives,"global_path"),
        "negative_multilead_rescue_n":n(negatives,"multilead_rescue"),
        "negative_final_fp_n":sum(CODE in set(r.get("published_codes") or []) for r in negatives),
        "fusion_loss_reasons":dict(sorted(loss_reasons.items())),
        "fusion_loss_states":dict(sorted(loss_states.items())),
        "fusion_loss_boundary_metrics":dict(sorted(loss_boundary.items())),
        "fusion_loss_unresolved_measurements":dict(sorted(loss_unresolved.items())),
    }


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-wpw-audit"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_WPW_AUDIT.json"))
    args=ap.parse_args()
    args.workdir.mkdir(parents=True,exist_ok=True)

    meta_path=args.workdir/"ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv",meta_path)
    meta=pd.read_csv(meta_path)
    adult=_adult_rows(meta,FOLDS)
    holdout=_fast_gate_holdout_ids()
    adult=adult.loc[~adult["ecg_id"].astype(int).isin(holdout)].copy()

    spec=TARGETS["WPW"]
    pos=adult[adult["_codes"].map(lambda x:_target_positive(x,spec["scp"]))].copy()
    pos=pos.sort_values(["_hash","ecg_id"])
    neg=adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg=neg.sort_values(["_hash","ecg_id"]).head(NEGATIVE_N)
    negative_ids=set(int(x) for x in neg["ecg_id"].tolist())
    selected=pd.concat([pos,neg],ignore_index=True).drop_duplicates(subset=["ecg_id"]).copy()

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
                "codes":dict(row["_codes"]),
                "candidate_codes":sorted(_candidate_codes(analysis)),
                "fusion_codes":sorted(_fusion_codes(analysis)),
                "published_codes":sorted(_published_codes(analysis)),
                "wpw_audit":_audit(analysis),
            })
        except Exception as exc:
            errors.append({"ecg_id":ecg_id,"error":f"{type(exc).__name__}:{exc}"})
        if (i+1)%25==0:
            print(f"MEDCALC_WPW_AUDIT {i+1}/{len(selected)}",flush=True)

    result={
        "version":"MEDCALC_WPW_FUSION_ANATOMY_AUDIT_V1",
        "role":"DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed":False,
        "folds":FOLDS,
        "fast_gate_100_excluded_n":len(holdout),
        "positive_selected_n":len(pos),
        "negative_selected_n":len(neg),
        "records_analyzed":len(rows),
        "analysis_error_n":len(errors),
        "metrics":_summarize(rows,negative_ids),
        "case_level_results_emitted":False,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,sort_keys=True))
    if errors:
        raise SystemExit(f"WPW audit had {len(errors)} analysis errors")


if __name__=="__main__":
    main()
