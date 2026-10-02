from __future__ import annotations

import argparse, json, time
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _canonical, _download, _ensure_record,
    _fast_gate_holdout_ids, _target_positive, select_records,
)
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg

VERSION="MEDCALC_AVB2_DROPPED_EVENT_SOURCE_AUDIT_V1"
COINCIDENCE_MS=36.0


def _retry(fn,*args,attempts=4):
    for attempt in range(1,attempts+1):
        try:
            return fn(*args)
        except Exception:
            if attempt>=attempts:
                raise
            time.sleep(10*attempt)


def _combined_events(evidence:dict) -> list[dict]:
    rows=[]
    times=[]

    for row in (evidence.get("observed_consensus") or {}).get("events") or []:
        t=float(row["time_ms"])
        rows.append({
            "time_ms":t,
            "source":"OBSERVED",
            "support_lead_n":int(row.get("support_lead_n") or 0),
            "spread_ms":row.get("spread_ms"),
        })
        times.append(t)

    for source,key in (
        ("RECOVERED_MORPHOLOGY","recovered_events"),
        ("UNSEEDED","unseeded_events"),
    ):
        for row in evidence.get(key) or []:
            t=float(row["time_ms"])
            if any(abs(t-prev)<=COINCIDENCE_MS for prev in times):
                continue
            rows.append({
                "time_ms":t,
                "source":source,
                "support_lead_n":int(row.get("support_lead_n") or 0),
                "spread_ms":row.get("spread_ms"),
            })
            times.append(t)

    return sorted(rows,key=lambda x:x["time_ms"])


def _map(events:list[dict], r_samples:list[int], fs:int) -> list[dict]:
    out=[dict(e,conducted=False,pr_ms=None) for e in events]
    if fs<=0 or not out:
        return out
    p=[int(round(float(e["time_ms"])*fs/1000.0)) for e in out]
    claimed=set()
    min_pr=int((70.0*fs)//1000.0)
    max_pr=int(-(-550.0*fs//1000.0))
    for ri0 in sorted(set(int(x) for x in r_samples)):
        candidates=[
            idx for idx,pi in enumerate(p)
            if pi<ri0 and min_pr<=(ri0-pi)<=max_pr
        ]
        chosen=None
        for idx in reversed(candidates):
            if idx not in claimed:
                chosen=idx
                break
        if chosen is None:
            continue
        claimed.add(chosen)
        out[chosen]["conducted"]=True
        out[chosen]["pr_ms"]=(ri0-p[chosen])*1000.0/fs
    return out


def run(workdir:Path,output:Path):
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    meta_path=workdir/"ptbxl_database.csv"; statements=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",meta_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements)
    meta=pd.read_csv(meta_path)
    selected,_=select_records(meta,folds=folds,exclude_ecg_ids=_fast_gate_holdout_ids())
    aliases=set(TARGETS["AVB2"]["scp"])
    positives=selected[selected["_codes"].map(lambda c:_target_positive(c,aliases))].copy()

    counts=Counter()
    source_events=Counter()
    support_hist=Counter()
    dropped_count_hist=Counter()
    errors=Counter()
    root=workdir/"records"

    for _,row in positives.iterrows():
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
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "selection_avb2_positive_n":int(len(positives)),
        "counts":dict(counts),
        "source_event_counts":dict(source_events),
        "source_support_histogram":dict(support_hist),
        "per_case_dropped_count_histogram":dict(dropped_count_hist),
        "analysis_error_n":sum(errors.values()),
        "analysis_error_types":dict(errors),
        "policy":"AUDIT_ONLY; EXACT_EXISTING_36MS_COMBINED_EVENT_DEDUP_AND_70_550MS_QRS_TO_P_MAPPING; SOURCE_TAGGING_ONLY; NO_NEW_THRESHOLDS; FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL",
    }
    output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--workdir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    run(args.workdir,args.output)
