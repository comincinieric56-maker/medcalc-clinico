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
    _target_positive,
    select_records,
)
from ecg_avb2_evidence import build_avb2_evidence
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_REAL_ATRIAL_SURFACE_AUDIT_V1"


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _surface_row(events,r_samples,fs):
    topo=build_avb2_evidence(list(events or []),list(r_samples or []),int(fs))
    return {
        "evaluable":bool(topo.get("evaluable")),
        "compatible":bool(topo.get("compatible")),
        "atrial_regular":bool(topo.get("atrial_sequence_regular")),
        "ge2_nonconducted":int(topo.get("max_consecutive_nonconducted_p") or 0)>=2,
        "midrr_majority":bool(topo.get("dropped_mid_rr_majority")),
        "event_n":len(events or []),
    }


def run(workdir:Path,output:Path,process_fold:int|None=None):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"
    statements_path=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements_path)

    meta=pd.read_csv(meta_path)
    selected,_=select_records(meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids())
    aliases=set(TARGETS["AVB2"]["scp"])
    positives=selected[selected["_codes"].map(lambda c:_target_positive(c,aliases))].copy()
    if process_fold is not None:
        positives=positives[positives["_fold"].astype(int)==int(process_fold)].copy()

    counts=Counter()
    unseeded_reason=Counter()
    morphology_reason=Counter()
    errors=[]
    records_root=workdir/"records"

    for _,row in positives.iterrows():
        try:
            base=_retry(_ensure_record,records_root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(rec.p_signal,int(round(float(rec.fs))),list(rec.sig_name),int(row["ecg_id"]))
            a=analyze_canonical_ecg(canonical)
            per_lead=a.get("leads") or {}
            rhythm=a.get("rhythm") or {}
            r_samples=list(rhythm.get("r_peaks_samples") or [])
            fs=int(canonical.get("fs") or a.get("fs") or 500)

            ev=recover_crosslead_atrial_candidates(canonical,per_lead)
            observed=dict(ev.get("observed_consensus") or {})
            recovered=list(ev.get("recovered_events") or [])
            unseeded=list(ev.get("unseeded_events") or [])
            combined=[{"time_ms":float(t)} for t in (ev.get("combined_event_times_ms") or [])]

            counts["n"]+=1
            counts["observed_evaluable_n"]+=int(bool(observed.get("evaluable")))
            counts["observed_organized_n"]+=int(bool(observed.get("organized")))
            counts["seeded_recovered_any_n"]+=int(bool(recovered))
            counts["unseeded_any_n"]+=int(bool(unseeded))
            counts["unseeded_organized_n"]+=int(bool(ev.get("unseeded_organized")))
            counts["combined_augmented_organized_n"]+=int(bool(ev.get("organized_augmented")))
            counts["raw_recovery_evaluable_n"]+=int(bool(ev.get("evaluable")))

            ua=dict(ev.get("unseeded_audit") or {})
            reason=str(ua.get("reason") or ua.get("status") or "NONE")
            unseeded_reason[reason]+=1
            for audit in (ev.get("per_lead_audit") or {}).values():
                audit=dict(audit or {})
                morphology_reason[str(audit.get("reason") or audit.get("status") or "NONE")]+=1

            surfaces={
                "observed":list(observed.get("events") or []),
                "seeded_recovered":recovered,
                "unseeded":unseeded,
                "combined":combined,
            }
            for name,events in surfaces.items():
                sr=_surface_row(events,r_samples,fs)
                prefix=f"{name}_"
                counts[prefix+"event_ge4_n"]+=int(sr["event_n"]>=4)
                counts[prefix+"topology_evaluable_n"]+=int(sr["evaluable"])
                counts[prefix+"topology_compatible_n"]+=int(sr["compatible"])
                counts[prefix+"atrial_regular_n"]+=int(sr["atrial_regular"])
                counts[prefix+"ge2_nonconducted_n"]+=int(sr["ge2_nonconducted"])
                counts[prefix+"midrr_majority_n"]+=int(sr["midrr_majority"])
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
        "selection_avb2_positive_n":int(len(positives)),
        "analysis_error_n":len(errors),
        "analysis_error_types":dict(Counter(errors)),
        "counts":dict(counts),
        "unseeded_reason_counts":dict(unseeded_reason),
        "morphology_recovery_reason_counts":dict(morphology_reason),
        "policy":"AGGREGATE_AUDIT_ONLY; EXISTING_EVIDENCE_SURFACES_ONLY; NO_NEW_THRESHOLDS; FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; NO_EXTERNAL_OR_FINAL_DATA",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,default=Path("/tmp/medcalc-avb2-atrial-surface"))
    ap.add_argument("--output",type=Path,default=Path("/tmp/MEDCALC_AVB2_ATRIAL_SURFACE.json"))
    ap.add_argument("--process-fold",type=int,choices=[1,2,3,4,5,6,7,8],default=None)
    args=ap.parse_args()
    run(args.workdir,args.output,args.process_fold)
