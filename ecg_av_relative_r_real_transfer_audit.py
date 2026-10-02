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
    TARGETS,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _target_positive,
    select_records,
)
from ecg_av_conduction import analyze_av_conduction
from ecg_recovered_atrial_sequence import (
    clean_r_measurement_inputs,
    clean_selected_rhythm,
)
from ecg_r_candidate_filter import RELATIVE_R_AMPLITUDE_MIN
from ecg_signal_measurements import analyze_canonical_ecg

VERSION = "MEDCALC_AV_RELATIVE_R_REAL_TRANSFER_AUDIT_V1"


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _rr_cv(samples, fs):
    x=np.asarray(sorted(set(int(v) for v in (samples or []))),dtype=float)
    if len(x)<3 or fs<=0:
        return None
    rr=np.diff(x)*1000.0/fs
    rr=rr[np.isfinite(rr)&(rr>0)]
    if rr.size<2 or float(np.mean(rr))<=0:
        return None
    return float(np.std(rr,ddof=1)/np.mean(rr))


def _median(values):
    z=[float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(z)) if z else None


def _apply(group: Counter, continuous: dict[str,list[float]], canonical: dict, analysis: dict) -> None:
    raw_rhythm=dict(analysis.get("rhythm") or {})
    clean_rhythm=clean_selected_rhythm(canonical,analysis)
    raw_r=list(raw_rhythm.get("r_peaks_samples") or [])
    clean_r=list(clean_rhythm.get("r_peaks_samples") or [])
    fs=int(analysis.get("fs") or canonical.get("fs") or 500)

    cleaned_leads, mask=clean_r_measurement_inputs(canonical,analysis)
    raw_av=dict(analysis.get("av_conduction") or {})
    clean_av=analyze_av_conduction(
        cleaned_leads,
        dict(analysis.get("atrial_activity") or {}),
        global_metrics=dict(analysis.get("global") or {}),
    )

    raw_cv=_rr_cv(raw_r,fs)
    clean_cv=_rr_cv(clean_r,fs)
    if raw_cv is not None: continuous["raw_rr_cv"].append(raw_cv)
    if clean_cv is not None: continuous["clean_rr_cv"].append(clean_cv)

    changed=bool(clean_r!=raw_r)
    removed=max(0,len(raw_r)-len(clean_r))
    raw_cls=str(raw_av.get("classification") or "NONE")
    clean_cls=str(clean_av.get("classification") or "NONE")

    group["n"]+=1
    group["filter_evaluable_n"]+=int(bool(mask.get("evaluable")))
    group["r_sequence_changed_n"]+=int(changed)
    group["r_removed_total"]+=removed
    group["r_removed_ge1_n"]+=int(removed>=1)
    group["r_removed_ge2_n"]+=int(removed>=2)
    group["raw_rr_regular_012_n"]+=int(raw_cv is not None and raw_cv<=0.12)
    group["clean_rr_regular_012_n"]+=int(clean_cv is not None and clean_cv<=0.12)
    group["raw_av_evaluable_n"]+=int(bool(raw_av.get("evaluable")))
    group["clean_av_evaluable_n"]+=int(bool(clean_av.get("evaluable")))
    group["av_classification_changed_n"]+=int(raw_cls!=clean_cls)
    group[f"raw_av:{raw_cls}"]+=1
    group[f"clean_av:{clean_cls}"]+=1

    for key in (
        "ventricular_regular","atrial_sequence_regular","one_to_one",
        "stable_pr","av_dissociation_phase",
    ):
        group[f"raw_{key}_n"]+=int(bool(raw_av.get(key)))
        group[f"clean_{key}_n"]+=int(bool(clean_av.get(key)))

    for key in ("nonconducted_p_n","max_consecutive_nonconducted_p"):
        try:
            rv=int(raw_av.get(key) or 0)
        except Exception:
            rv=0
        try:
            cv=int(clean_av.get(key) or 0)
        except Exception:
            cv=0
        group[f"raw_{key}_sum"]+=rv
        group[f"clean_{key}_sum"]+=cv


def run(workdir: Path, output: Path, process_fold: int) -> dict:
    folds=[1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True,exist_ok=True)
    metadata_path=workdir/"ptbxl_database.csv"
    statements_path=workdir/"scp_statements.csv"
    _retry(_download,f"{BASE}/ptbxl_database.csv",metadata_path)
    _retry(_download,f"{BASE}/scp_statements.csv",statements_path)

    meta=pd.read_csv(metadata_path)
    selected,selection=select_records(
        meta,
        folds=folds,
        exclude_ecg_ids=_fast_gate_holdout_ids(),
    )
    negative_ids={int(x) for x in selection["negative_control_ecg_ids"]}
    avb2_aliases=set(TARGETS["AVB2"]["scp"])
    avb3_aliases=set(TARGETS["AVB3"]["scp"])

    rows=selected[
        (selected["_fold"].astype(int)==int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c:_target_positive(c,avb2_aliases))
            | selected["_codes"].map(lambda c:_target_positive(c,avb3_aliases))
        )
    ].copy()

    groups={k:Counter() for k in ("AVB2","AVB3","CONTROL")}
    continuous={
        k:{"raw_rr_cv":[],"clean_rr_cv":[]}
        for k in groups
    }
    errors=Counter()
    root=workdir/"records"

    for _,row in rows.iterrows():
        ecg_id=int(row["ecg_id"])
        if ecg_id in negative_ids:
            group="CONTROL"
        elif _target_positive(row["_codes"],avb3_aliases):
            group="AVB3"
        else:
            group="AVB2"
        try:
            base=_retry(_ensure_record,root,str(row["filename_hr"]))
            rec=wfdb.rdrecord(str(base))
            canonical=_canonical(
                rec.p_signal,
                int(round(float(rec.fs))),
                list(rec.sig_name),
                ecg_id,
            )
            analysis=analyze_canonical_ecg(canonical)
            _apply(groups[group],continuous[group],canonical,analysis)
        except Exception as exc:
            errors[f"{group}:{type(exc).__name__}"]+=1

    out={
        "version":VERSION,
        "dataset":"PTB-XL",
        "role":"DEVELOPMENT_TUNING_ONLY",
        "folds":folds,
        "process_fold":int(process_fold),
        "fast_gate_holdout_excluded":True,
        "external_validation_claim_allowed":False,
        "relative_r_amplitude_min":RELATIVE_R_AMPLITUDE_MIN,
        "groups":{k:dict(v) for k,v in groups.items()},
        "rr_cv_medians":{
            k:{
                "raw":_median(v["raw_rr_cv"]),
                "clean":_median(v["clean_rr_cv"]),
            }
            for k,v in continuous.items()
        },
        "analysis_error_n":sum(errors.values()),
        "analysis_error_types":dict(errors),
        "clinical_output_changed":False,
        "policy":"AUDIT_ONLY; EXISTING_RELATIVE_R_AMPLITUDE_MIN_0_15; FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; NO_EXTERNAL_OR_FINAL_DATA; NO_THRESHOLD_TUNING",
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
