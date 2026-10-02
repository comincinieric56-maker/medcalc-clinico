from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, _canonical, _download, _ensure_record,
    _fast_gate_holdout_ids, select_records,
)
from ecg_avb2_dropped_event_source_audit import (
    _combined_events, _map, _retry,
)
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_DROPPED_EVENT_SOURCE_CONTROL_AUDIT_V1"


def run(workdir:Path,output:Path,process_fold:int):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"
    statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)

    meta=pd.read_csv(meta_path)
    selected,summary=select_records(
        meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids()
    )
    neg_ids=set(int(x) for x in summary["negative_control_ecg_ids"])
    rows=selected[
        (selected["_fold"].astype(int)==int(process_fold))
        & selected["ecg_id"].astype(int).isin(neg_ids)
    ].copy()

    counts=Counter()
    source_events=Counter()
    support_hist=Counter()
    dropped_count_hist=Counter()
    errors=Counter()
    root=workdir/"records"

    for _,row in rows.iterrows():
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(
                rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),int(row["ecg_id"])
            )
            analysis=analyze_canonical_ecg(canonical)
            evidence=recover_crosslead_atrial_candidates(
                canonical,analysis.get("leads") or {}
            )
            events=_combined_events(evidence)
            rhythm=analysis.get("rhythm") or {}
            fs=int(canonical.get("fs") or analysis.get("fs") or 500)
            mapped=_map(events,list(rhythm.get("r_peaks_samples") or []),fs)

            counts["case_n"]+=1
            per_source_drop=Counter()
            per_source_conducted=Counter()
            for e in mapped:
                src=str(e["source"])
                state="CONDUCTED" if e["conducted"] else "DROPPED"
                source_events[f"{src}:{state}"]+=1
                support=int(e.get("support_lead_n") or 0)
                support_hist[f"{src}:{state}:SUPPORT_{support}"]+=1
                if e["conducted"]:
                    per_source_conducted[src]+=1
                else:
                    per_source_drop[src]+=1

            for src in ("OBSERVED","RECOVERED_MORPHOLOGY","UNSEEDED"):
                dn=int(per_source_drop[src])
                cn=int(per_source_conducted[src])
                counts[f"case_any_dropped_{src.lower()}_n"]+=int(dn>0)
                counts[f"case_any_conducted_{src.lower()}_n"]+=int(cn>0)
                dropped_count_hist[f"{src}:DROP_COUNT_{dn}"]+=1

            recovered_drop=(
                per_source_drop["RECOVERED_MORPHOLOGY"]
                + per_source_drop["UNSEEDED"]
            )
            counts["case_any_dropped_recovered_or_unseeded_n"]+=int(recovered_drop>0)
            counts["case_ge2_dropped_recovered_or_unseeded_n"]+=int(recovered_drop>=2)
            counts["case_any_dropped_combined_n"]+=int(
                sum(per_source_drop.values())>0
            )
            counts["case_ge2_dropped_combined_n"]+=int(
                sum(per_source_drop.values())>=2
            )
        except Exception as exc:
            errors[type(exc).__name__]+=1

    out={
        "version":VERSION,
        "dataset":"PTB-XL",
        "role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,
        "process_fold":int(process_fold),
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "selection_control_n":int(len(rows)),
        "counts":dict(counts),
        "source_event_counts":dict(source_events),
        "source_support_histogram":dict(support_hist),
        "per_case_dropped_count_histogram":dict(dropped_count_hist),
        "analysis_error_n":sum(errors.values()),
        "analysis_error_types":dict(errors),
        "policy":"AUDIT_ONLY; FIXED_NEGATIVE_CONTROLS; EXACT_EXISTING_36MS_COMBINED_EVENT_DEDUP_AND_70_550MS_QRS_TO_P_MAPPING; SOURCE_TAGGING_ONLY; NO_NEW_THRESHOLDS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
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
