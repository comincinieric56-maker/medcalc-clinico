from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS=[
    "atrial_fibrillation","atrial_flutter","sinus_bradycardia","sinus_tachycardia",
    "right_bundle_branch_block","left_bundle_branch_block",
    "left_anterior_fascicular_block","left_posterior_fascicular_block",
    "first_degree_av_block","second_degree_av_block","complete_av_block",
    "ventricular_preexcitation",
]
BOOT_N=1000
BOOT_SEED=20260928


def _norm_text(value: object) -> str:
    x=unicodedata.normalize("NFKD",str(value or ""))
    x=x.encode("ascii","ignore").decode("ascii").lower()
    x=re.sub(r"[^a-z0-9]+"," ",x)
    return " ".join(x.split())


def _find_col(df: pd.DataFrame, candidates: list[str], contains: list[str]|None=None) -> str:
    lower={str(c).strip().lower():str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if contains:
        for col in df.columns:
            n=str(col).lower()
            if all(token in n for token in contains):
                return str(col)
    raise KeyError(f"Missing column candidates={candidates}, contains={contains}, columns={list(df.columns)}")


def _record_key(value: object) -> str:
    name=Path(str(value or "").strip()).name
    for suffix in (".hea",".dat"):
        if name.lower().endswith(suffix):
            name=name[:-len(suffix)]
    return name


def _split_codes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    out=[]
    for part in re.split(r"[;|]",str(value)):
        for sub in part.split("+"):
            token=sub.strip()
            if token:
                out.append(token)
    return out


def _code_dictionary(code: pd.DataFrame, mapping: dict) -> tuple[dict[str,set[str]],dict]:
    aha_code_col=_find_col(code,["AHA_Code","AHA code","Code"],contains=["aha","code"])
    try:
        desc_col=_find_col(
            code,
            ["AHA_Statement","AHA Statement","AHA_Description","Description","Statement","Name"],
            contains=["aha","statement"],
        )
    except KeyError:
        desc_col=_find_col(code,["Description","Statement","Name"])
    synonym_to_concept={}
    for concept in TARGETS:
        for phrase in (mapping.get("concepts") or {}).get(concept,[]):
            synonym_to_concept[_norm_text(phrase)]=concept
    concept_codes={x:set() for x in TARGETS}
    mapped_rows=0
    for _,row in code.iterrows():
        raw_code=str(row.get(aha_code_col) or "").strip()
        desc=_norm_text(row.get(desc_col))
        concept=synonym_to_concept.get(desc)
        if raw_code and concept:
            concept_codes[concept].add(raw_code)
            mapped_rows+=1
    audit={
        "aha_code_column":aha_code_col,
        "aha_description_column":desc_col,
        "dictionary_rows":int(len(code)),
        "mapped_dictionary_rows":int(mapped_rows),
        "mapped_code_counts":{k:len(v) for k,v in concept_codes.items()},
        "missing_targets":[k for k,v in concept_codes.items() if not v],
    }
    return concept_codes,audit


def _metrics(y: np.ndarray,p: np.ndarray) -> dict:
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=bool)
    tp=int(np.sum(y&p)); tn=int(np.sum(~y&~p)); fp=int(np.sum(~y&p)); fn=int(np.sum(y&~p))
    def div(a,b): return float(a/b) if b else None
    return {
        "n":int(len(y)),"positive_n":int(np.sum(y)),"negative_n":int(np.sum(~y)),
        "tp":tp,"tn":tn,"fp":fp,"fn":fn,
        "sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),
        "ppv":div(tp,tp+fp),"npv":div(tn,tn+fn),
        "f1":div(2*tp,2*tp+fp+fn),
    }


def _bootstrap(y: np.ndarray,p: np.ndarray,key: str) -> list[float]|None:
    y=np.asarray(y,dtype=bool); p=np.asarray(p,dtype=bool)
    if len(y)<30:
        return None
    rng=np.random.default_rng(BOOT_SEED)
    vals=[]
    n=len(y)
    for _ in range(BOOT_N):
        idx=rng.integers(0,n,size=n)
        v=_metrics(y[idx],p[idx]).get(key)
        if v is not None:
            vals.append(v)
    if len(vals)<100:
        return None
    return [round(float(np.percentile(vals,2.5)),6),round(float(np.percentile(vals,97.5)),6)]


def _age_stratum(days: object) -> str:
    try:
        d=float(days)
    except Exception:
        return "unknown"
    if not np.isfinite(d) or d<0:
        return "unknown"
    if d < 365:
        return "0_to_lt1"
    if d < 7*365:
        return "1_to_lt7"
    if d <= 14*365+366:
        return "7_to_14"
    return "outside_expected"


def score(predictions_dir: Path, attributes: Path, ecg_code: Path, mapping_path: Path, selection_summary: Path, output: Path) -> dict:
    files=sorted(predictions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No ZZU prediction shards found")
    pred=pd.concat([pd.read_csv(p,dtype=str) for p in files],ignore_index=True)
    if pred["record_id"].duplicated().any() or pred["patient_id"].duplicated().any():
        raise ValueError("ZZU frozen cohort must contain one unique ECG per patient")

    selection=json.loads(selection_summary.read_text(encoding="utf-8"))
    expected=int(selection["selected_records"])
    if len(pred)!=expected:
        raise ValueError(f"Inference incomplete predictions={len(pred)} expected={expected}")

    # First label access occurs here, after predictions are complete.
    attr=pd.read_csv(attributes,dtype=str)
    code=pd.read_csv(ecg_code,dtype=str)
    mapping=json.loads(mapping_path.read_text(encoding="utf-8"))
    concept_codes,dictionary_audit=_code_dictionary(code,mapping)

    file_col=_find_col(attr,["FileName","File Name","file_name","ECG_ID","ECG ID"])
    patient_col=_find_col(attr,["Patient_ID","Patient ID","patient_id"])
    aha_col=_find_col(attr,["AHA_Code","AHA Code","aha_code"],contains=["aha","code"])
    age_col=_find_col(attr,["Age","Age_days","AgeDays","age"])

    gold=pd.DataFrame({
        "record_id":attr[file_col].map(_record_key),
        "gold_patient_id":attr[patient_col].astype(str),
        "age_days":attr[age_col],
        "_aha":attr[aha_col],
    })
    if gold["record_id"].duplicated().any():
        raise ValueError("ZZU attribute dictionary has duplicate record identifiers")

    for target in TARGETS:
        codes=concept_codes[target]
        gold[f"gold_{target}"]=gold["_aha"].map(
            lambda x,codes=codes:any(token in codes for token in _split_codes(x))
        )

    joined=pred.merge(gold,on="record_id",how="left",validate="one_to_one")
    if joined["gold_patient_id"].isna().any():
        raise ValueError(f"Missing attribute rows for {int(joined['gold_patient_id'].isna().sum())} selected records")
    if not np.all(joined["patient_id"].astype(str).str.upper()==joined["gold_patient_id"].astype(str).str.upper()):
        raise ValueError("Patient identity mismatch between frozen selection and attribute dictionary")

    metrics={}
    scorable=[]
    for target in TARGETS:
        if not concept_codes[target]:
            metrics[target]={"status":"NOT_SCORABLE_NO_FROZEN_SEMANTIC_MATCH"}
            continue
        y=joined[f"gold_{target}"].astype(bool).to_numpy()
        p=joined[f"pred_{target}"].astype(str).str.lower().isin(["true","1"]).to_numpy()
        row=_metrics(y,p)
        row["sensitivity_95ci"]=_bootstrap(y,p,"sensitivity")
        row["specificity_95ci"]=_bootstrap(y,p,"specificity")
        row["f1_95ci"]=_bootstrap(y,p,"f1")
        row["status"]="SCORABLE"
        metrics[target]=row
        scorable.append(target)

    # Aggregate micro/macro F1 over scorable frozen concepts.
    macro_values=[metrics[t]["f1"] for t in scorable if metrics[t]["f1"] is not None]
    if scorable:
        ys=[]; ps=[]
        for t in scorable:
            ys.append(joined[f"gold_{t}"].astype(bool).to_numpy())
            ps.append(joined[f"pred_{t}"].astype(str).str.lower().isin(["true","1"]).to_numpy())
        yy=np.concatenate(ys); pp=np.concatenate(ps)
        micro=_metrics(yy,pp)["f1"]
    else:
        micro=None

    joined["_age_stratum"]=joined["age_days"].map(_age_stratum)
    age_strata={}
    for stratum in ["0_to_lt1","1_to_lt7","7_to_14","unknown"]:
        mask=joined["_age_stratum"]==stratum
        age_strata[stratum]={"n":int(mask.sum())}
        if not mask.any():
            continue
        vals=[]
        for t in scorable:
            y=joined.loc[mask,f"gold_{t}"].astype(bool).to_numpy()
            p=joined.loc[mask,f"pred_{t}"].astype(str).str.lower().isin(["true","1"]).to_numpy()
            f1=_metrics(y,p)["f1"]
            if f1 is not None:
                vals.append(f1)
        age_strata[stratum]["macro_f1"]=float(np.mean(vals)) if vals else None

    analysis_error=joined.get("analysis_error",pd.Series("",index=joined.index)).fillna("").astype(str).str.len()>0
    abstention=pd.to_numeric(joined.get("abstention_n",pd.Series(0,index=joined.index)),errors="coerce").fillna(0)>0
    remeasure=joined.get("remeasure_required",pd.Series(False,index=joined.index)).astype(str).str.lower().isin(["true","1"])

    summary={
        "validation_id":"ZZU_PECG_PEDIATRIC_DIAGNOSTIC_V1",
        "validation_type":"EXTERNAL_PEDIATRIC_DIAGNOSTIC_PHYSICIAN_REVIEWED",
        "clinical_engine_baseline_sha":"7b03f1ada34c93246ec9b72fa63284269d2564ab",
        "records_scored":int(len(joined)),
        "patients_scored":int(joined["patient_id"].nunique()),
        "analysis_failure_rate":float(np.mean(analysis_error)),
        "reasoner_abstention_rate":float(np.mean(abstention)),
        "remeasure_required_rate":float(np.mean(remeasure)),
        "scorable_targets":scorable,
        "dictionary_audit":dictionary_audit,
        "metrics":metrics,
        "macro_f1":float(np.mean(macro_values)) if macro_values else None,
        "micro_f1":micro,
        "age_strata_descriptive":age_strata,
        "selection":selection,
        "anti_leakage":{
            "predictions_completed_before_label_files_opened":True,
            "frozen_semantic_mapping_used":True,
            "posthoc_synonym_additions_allowed":False,
            "threshold_tuning_allowed":False,
            "individual_record_debugging_allowed":False,
            "row_level_gold_prediction_join_persisted":False,
            "disease_icd10_used_as_ecg_truth":False,
        },
        "interpretation_constraints":[
            "This is pediatric external diagnostic validation only and does not establish adult performance.",
            "Reference ECG statements were physician diagnosed and senior reviewed, then standardized by dataset authors.",
            "Only frozen semantic matches are scored; unmatched ZZU labels remain unscored.",
            "No ZZU result may be used to tune thresholds, detectors, reasoner rules, or R27.",
        ],
        "clinical_validation_claim_allowed":False,
        "pediatric_external_diagnostic_validation_claim_allowed":True,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))
    return summary


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions-dir",type=Path,required=True)
    ap.add_argument("--attributes",type=Path,required=True)
    ap.add_argument("--ecg-code",type=Path,required=True)
    ap.add_argument("--mapping",type=Path,required=True)
    ap.add_argument("--selection-summary",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    score(args.predictions_dir,args.attributes,args.ecg_code,args.mapping,args.selection_summary,args.output)


if __name__=="__main__":
    main()
