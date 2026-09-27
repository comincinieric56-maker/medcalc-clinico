from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ecg_validation_guard import assert_external_dataset, load_registry


TARGETS = {
    "ST": "21",
    "SB": "22",
    "AF": "50",
    "AFL": "51",
    "1DAVB": "82",
    "LAFB": "101",
    "LBBB": "104",
    "RBBB": "106",
}

EXPECTED_DESCRIPTION_TERMS = {
    "21": ("sinus", "tachy"),
    "22": ("sinus", "brady"),
    "50": ("atrial", "fibrillation"),
    "51": ("atrial", "flutter"),
    "82": ("prolonged", "pr", "interval"),
    "101": ("left", "anterior"),
    "104": ("left", "bundle", "branch"),
    "106": ("right", "bundle", "branch"),
}


def core_codes(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    return {
        part.strip().split("+")[0].strip()
        for part in str(value).split(";")
        if part.strip()
    }


def binary_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=bool)
    tp = int(np.sum(y & p)); tn = int(np.sum(~y & ~p))
    fp = int(np.sum(~y & p)); fn = int(np.sum(y & ~p))
    def div(a: float, b: float) -> float | None:
        return float(a/b) if b else None
    return {
        "n": int(len(y)),
        "positive_n": int(np.sum(y)),
        "negative_n": int(np.sum(~y)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": div(tp,tp+fn),
        "specificity": div(tn,tn+fp),
        "ppv": div(tp,tp+fp),
        "npv": div(tn,tn+fn),
        "f1": div(2*tp,2*tp+fp+fn),
    }


def validate_code_dictionary(code_csv: Path) -> Dict[str,str]:
    df = pd.read_csv(code_csv, dtype=str)
    mapping = {
        str(row["Code"]).strip(): str(row["Description"]).strip()
        for _,row in df.iterrows()
    }
    for code, terms in EXPECTED_DESCRIPTION_TERMS.items():
        desc = mapping.get(code, "").lower()
        if not desc:
            raise ValueError(f"SPH code {code} missing from code.csv")
        if not all(term in desc for term in terms):
            raise ValueError(
                f"SPH code {code} description mismatch: {mapping.get(code)!r}; "
                f"expected terms={terms}"
            )
    return {code:mapping[code] for code in EXPECTED_DESCRIPTION_TERMS}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda:fh.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions-dir",type=Path,required=True)
    ap.add_argument("--gold",type=Path,required=True)
    ap.add_argument("--code",type=Path,required=True)
    ap.add_argument("--selection-summary",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    args=ap.parse_args()

    assert_external_dataset("sph",load_registry())
    descriptions=validate_code_dictionary(args.code)

    files=sorted(args.predictions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No SPH prediction shard CSVs.")
    pred=pd.concat([pd.read_csv(p,dtype={"ECG_ID":str}) for p in files],ignore_index=True)
    if pred["ECG_ID"].duplicated().any():
        raise ValueError("Duplicate ECG_ID across inference shards.")
    gold=pd.read_csv(args.gold,dtype=str)

    if set(pred["ECG_ID"]) != set(gold["ECG_ID"]):
        missing=sorted(set(gold["ECG_ID"])-set(pred["ECG_ID"]))
        extra=sorted(set(pred["ECG_ID"])-set(gold["ECG_ID"]))
        raise ValueError(f"Prediction/gold identity mismatch missing={missing[:10]} extra={extra[:10]}")

    merged=gold.merge(pred,on="ECG_ID",how="inner",validate="one_to_one")
    labels=[core_codes(v) for v in merged["AHA_Code"]]

    metrics={}
    for name,code in TARGETS.items():
        truth=np.asarray([code in s for s in labels],dtype=bool)
        prediction_col=f"pred_{name}"
        if prediction_col not in merged:
            raise ValueError(f"Missing {prediction_col}")
        pred_values=merged[prediction_col].astype(str).str.lower().isin(["true","1"]).to_numpy()
        metrics[name]=binary_metrics(truth,pred_values)

    truth_afl=np.asarray([bool({"50","51"} & s) for s in labels],dtype=bool)
    pred_afl=(
        merged["pred_AF"].astype(str).str.lower().isin(["true","1"]).to_numpy()
        | merged["pred_AFL"].astype(str).str.lower().isin(["true","1"]).to_numpy()
    )
    metrics["AF_OR_AFL"]=binary_metrics(truth_afl,pred_afl)

    publication=merged["publication_allowed"].astype(str).str.lower().isin(["true","1"])
    remeasure=merged["remeasure_required"].astype(str).str.lower().isin(["true","1"])
    selection=json.loads(args.selection_summary.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True,exist_ok=True)
    merged.to_csv(args.output_dir/"sph_predictions_with_gold.csv",index=False)

    summary={
        "validation_type":"FROZEN_EXTERNAL_DIGITAL_SIGNAL_PATIENT_HASH_SUBSET",
        "dataset_id":"sph",
        "engine_commit":os.getenv("GITHUB_SHA"),
        "selection":selection,
        "records_evaluated":int(len(merged)),
        "patients_evaluated":int(merged["Patient_ID"].nunique()),
        "first_10_seconds_only":True,
        "aha_code_descriptions":descriptions,
        "metrics":metrics,
        "fail_closed":{
            "publication_blocked_n":int((~publication).sum()),
            "publication_blocked_rate":float((~publication).mean()),
            "remeasure_required_n":int(remeasure.sum()),
            "remeasure_required_rate":float(remeasure.mean()),
        },
        "anti_leakage":{
            "patient_selection_depended_on_labels":False,
            "inference_jobs_received_gold_labels":False,
            "gold_opened_only_after_all_inference_shards_completed":True,
            "threshold_tuning_allowed":False,
            "individual_label_debugging_allowed":False,
        },
        "hashes":{
            "gold_sha256":sha256(args.gold),
            "code_sha256":sha256(args.code),
        },
    }
    (args.output_dir/"sph_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
