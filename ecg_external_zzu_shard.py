from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_signal_measurements import analyze_canonical_ecg

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
PREDICTION_CODES = {
    "atrial_fibrillation": {"AF_COMPATIBLE"},
    "atrial_flutter": {"FLUTTER_OR_AT_COMPATIBLE"},
    "sinus_bradycardia": {"SINUS_BRADYCARDIA_COMPATIBLE"},
    "sinus_tachycardia": {"SINUS_TACHYCARDIA_COMPATIBLE"},
    "right_bundle_branch_block": {"RBBB_MORPHOLOGY_COMPATIBLE"},
    "left_bundle_branch_block": {"LBBB_MORPHOLOGY_COMPATIBLE"},
    "left_anterior_fascicular_block": {"LAFB_COMPATIBLE"},
    "left_posterior_fascicular_block": {"LPFB_COMPATIBLE"},
    "first_degree_av_block": {"FIRST_DEGREE_AV_DELAY_COMPATIBLE"},
    "second_degree_av_block": {
        "MOBITZ_I_WENCKEBACH_COMPATIBLE",
        "MOBITZ_II_COMPATIBLE",
        "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
        "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
    },
    "complete_av_block": {"COMPLETE_AV_BLOCK_COMPATIBLE"},
    "ventricular_preexcitation": {"VENTRICULAR_PREEXCITATION_COMPATIBLE"},
}


def _norm_lead(name: str) -> str:
    x=str(name or "").strip().upper().replace(" ","")
    aliases={"AVR":"aVR","AVL":"aVL","AVF":"aVF"}
    return aliases.get(x, x)


def _load_record(root: Path, record_base: str) -> tuple[np.ndarray,int]:
    rec=wfdb.rdrecord(str(root / record_base))
    x=np.asarray(rec.p_signal,dtype=float)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {x.shape}")
    fs=int(round(float(rec.fs)))
    if fs != 500:
        raise ValueError(f"Expected 500 Hz, got {fs}")
    names=[_norm_lead(v) for v in (rec.sig_name or [])]
    if set(names) != set(LEADS) or x.shape[1] != 12:
        raise ValueError(f"Expected 12 standard leads, got {names}")
    order=[names.index(lead) for lead in LEADS]
    x=x[:,order]
    if x.shape[0] < 5*fs:
        raise ValueError(f"Record shorter than 5 seconds: {x.shape[0]/fs:.3f}s")
    x=x[: min(x.shape[0],10*fs), :]
    return x,fs


def _canonical(x: np.ndarray, fs: int, record_id: str, patient_id: str) -> dict[str,Any]:
    leads={}
    for idx,lead in enumerate(LEADS):
        sig=np.asarray(x[:,idx],dtype=float)
        finite=np.isfinite(sig)
        leads[lead]={
            "signal_mv":[float(v) if math.isfinite(float(v)) else None for v in sig],
            "quality_mask":np.where(finite,2,0).astype(np.uint8).tolist(),
            "fs":int(fs),
            "duration_s":float(len(sig)/fs),
            "source":"ZZU_PECG_FROZEN_EXTERNAL_PEDIATRIC_DIGITAL_SIGNAL",
            "confidence":1.0,
            "status":"MEASURED",
        }
    return {
        "contract":"MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "fs":int(fs),
        "lead_order":list(LEADS),
        "leads":leads,
        "calibration":{
            "speed_mm_per_s":25.0,
            "gain_mm_per_mv":10.0,
            "source":"DIGITAL_EXTERNAL_REFERENCE",
            "confidence":1.0,
        },
        "validation_provenance":{
            "dataset_id":"zzu_pecg",
            "record_id":record_id,
            "patient_id":patient_id,
            "frozen_external":True,
            "labels_available_to_inference":False,
            "pediatric_external":True,
        },
    }


def _predict(result: dict[str,Any]) -> dict[str,Any]:
    reasoner=result.get("specialist_reasoning") or {}
    summary=reasoner.get("diagnostic_summary") or {}
    findings=[
        row for row in summary.get("findings") or []
        if bool(row.get("publishable"))
    ]
    codes={str(row.get("code") or "") for row in findings}
    out={}
    for concept,allowed in PREDICTION_CODES.items():
        out[f"pred_{concept}"]=bool(codes & allowed)
    out["publication_allowed"]=bool(summary.get("publication_allowed"))
    out["abstention_n"]=len(summary.get("abstentions") or [])
    out["finding_codes"]="|".join(sorted(codes))
    out["remeasure_required"]=bool(
        (result.get("measurement_consensus") or {}).get("remeasure_required")
    )
    out["signal_integrity_quality"]=(result.get("signal_integrity") or {}).get("overall_quality")
    out["measurement_consensus_quality"]=(result.get("measurement_consensus") or {}).get("overall_measurement_quality")
    return out


def run(root: Path, output: Path) -> dict[str,Any]:
    manifest=pd.read_csv(root/"manifest.csv",dtype=str)
    rows=[]
    for i,row in manifest.iterrows():
        patient_id=str(row["patient_id"])
        record_id=str(row["record_id"])
        record_base=str(Path(row["header_relpath"]).with_suffix(""))
        base={"patient_id":patient_id,"record_id":record_id}
        try:
            x,fs=_load_record(root/"records",record_base)
            result=analyze_canonical_ecg(_canonical(x,fs,record_id,patient_id))
            rows.append({**base,**_predict(result),"analysis_error":""})
        except Exception as exc:
            failed={
                **base,
                "publication_allowed":False,
                "abstention_n":1,
                "remeasure_required":True,
                "finding_codes":"",
                "analysis_error":f"{type(exc).__name__}:{exc}",
            }
            for concept in PREDICTION_CODES:
                failed[f"pred_{concept}"]=False
            rows.append(failed)
        if (i+1)%25==0 or i+1==len(manifest):
            print(f"ZZU_EXTERNAL_INFERENCE {i+1}/{len(manifest)}",flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output,index=False)
    summary={
        "records":len(rows),
        "analysis_errors":sum(bool(str(x.get("analysis_error") or "")) for x in rows),
        "labels_opened":False,
        "output":str(output),
    }
    print(json.dumps(summary,indent=2,sort_keys=True))
    return summary


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    run(args.root,args.output)


if __name__=="__main__":
    main()
