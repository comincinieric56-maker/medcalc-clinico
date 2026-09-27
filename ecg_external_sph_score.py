from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "AF","FLUTTER","SINUS_BRADY","SINUS_TACHY","RBBB_COMPLETE","LBBB",
    "LAFB","LPFB","AVB1","AVB2","AVB3","WPW",
]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    def div(a, b):
        return float(a / b) if b else None
    return {
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true)),
        "negative_n": int(np.sum(~y_true)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--selection-summary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.predictions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No SPH prediction shards found.")
    pred = pd.concat([pd.read_csv(p, dtype={"record_id": str, "patient_id": str}) for p in files], ignore_index=True)
    gold = pd.read_csv(args.gold, dtype={"ECG_ID": str, "Patient_ID": str})
    if pred["record_id"].duplicated().any():
        raise ValueError("Duplicate SPH predictions.")
    if gold["ECG_ID"].duplicated().any():
        raise ValueError("Duplicate SPH gold rows.")

    merged = gold.merge(pred, left_on="ECG_ID", right_on="record_id", how="left", validate="one_to_one")
    if merged["record_id"].isna().any():
        missing = merged.loc[merged["record_id"].isna(), "ECG_ID"].astype(str).tolist()
        raise ValueError(f"Missing predictions for {len(missing)} records: {missing[:20]}")

    metrics = {}
    general_mask = merged["cohort_general_blind"].astype(str).str.lower().isin(["true","1"])
    control_mask = merged["cohort_negative_control"].astype(str).str.lower().isin(["true","1"])

    for target in TARGETS:
        gold_col = f"gold_{target}"
        pred_col = f"pred_{target}"
        if gold_col not in merged or pred_col not in merged:
            metrics[target] = {"status": "NOT_SCORABLE_MISSING_COLUMN"}
            continue
        y = merged[gold_col].astype(str).str.lower().isin(["true","1"])
        p = merged[pred_col].astype(str).str.lower().isin(["true","1"])

        general = _metrics(y[general_mask].to_numpy(), p[general_mask].to_numpy())
        positive_mask = y
        sensitivity_all_cases = (
            float(np.mean(p[positive_mask])) if int(positive_mask.sum()) > 0 else None
        )
        negative_general = general_mask & ~y
        specificity_general_negatives = (
            float(np.mean(~p[negative_general])) if int(negative_general.sum()) > 0 else None
        )
        negative_controls = control_mask & ~y
        specificity_controls = (
            float(np.mean(~p[negative_controls])) if int(negative_controls.sum()) > 0 else None
        )
        challenge = _metrics(y.to_numpy(), p.to_numpy())

        metrics[target] = {
            "general_blind_cohort": general,
            "all_external_positive_cases": {
                "positive_n": int(positive_mask.sum()),
                "sensitivity": sensitivity_all_cases,
            },
            "specificity_general_label_negative": specificity_general_negatives,
            "specificity_predeclared_negative_controls": specificity_controls,
            "target_enriched_union_metrics_not_prevalence_calibrated": challenge,
        }

    error_mask = merged["analysis_error"].fillna("").astype(str).str.len() > 0
    abstention_mask = pd.to_numeric(merged["abstention_n"], errors="coerce").fillna(0) > 0
    remeasure_mask = merged["remeasure_required"].astype(str).str.lower().isin(["true","1"])

    selection = json.loads(args.selection_summary.read_text(encoding="utf-8"))
    summary = {
        "validation_id": "SPH_FROZEN_EXTERNAL_V1",
        "validation_type": "DIGITAL_SIGNAL_EXTERNAL_PATIENT_LEVEL",
        "source_window": "FIRST_10_SECONDS_OF_EACH_SELECTED_12_LEAD_RECORD",
        "records_scored": int(len(merged)),
        "patients_scored": int(merged["Patient_ID"].nunique()),
        "general_blind_record_n": int(general_mask.sum()),
        "negative_control_record_n": int(control_mask.sum()),
        "selection": selection,
        "analysis_failure_rate": float(np.mean(error_mask)),
        "reasoner_abstention_rate": float(np.mean(abstention_mask)),
        "remeasure_required_rate": float(np.mean(remeasure_mask)),
        "metrics": metrics,
        "interpretation_constraints": [
            "General-blind cohort was selected by patient hash without consulting AHA labels.",
            "Target-positive enrichment is used to estimate sensitivity for rare diagnoses and is not population-prevalence calibrated.",
            "PPV/NPV from the enriched union must not be interpreted as population PPV/NPV.",
            "SPH labels were unavailable to inference jobs and were joined only in this scoring step.",
            "No threshold or clinical rule may be tuned on this result if SPH is retained as an external baseline.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "sph_predictions_with_gold.csv", index=False)
    (args.output_dir / "sph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
