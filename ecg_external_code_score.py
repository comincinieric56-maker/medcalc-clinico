from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ecg_validation_guard import assert_external_dataset, load_registry


EXPECTED_RECORDS = 827
SUPPORTED = ["RBBB", "LBBB", "SB", "AF", "ST"]


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    def div(a: float, b: float) -> float | None:
        return float(a / b) if b else None

    return {
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true)),
        "negative_n": int(np.sum(~y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    dataset = assert_external_dataset("code_test", load_registry())

    files = sorted(args.predictions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No shard prediction CSV files found.")

    pred = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if pred["record_index"].duplicated().any():
        dupes = pred.loc[pred["record_index"].duplicated(), "record_index"].tolist()
        raise ValueError(f"Duplicate prediction indices: {dupes[:20]}")
    pred = pred.sort_values("record_index").reset_index(drop=True)
    expected = np.arange(EXPECTED_RECORDS, dtype=int)
    got = pred["record_index"].astype(int).to_numpy()
    if len(got) != EXPECTED_RECORDS or not np.array_equal(got, expected):
        missing = sorted(set(expected.tolist()) - set(got.tolist()))
        raise ValueError(
            f"Frozen scoring requires exactly indices 0..{EXPECTED_RECORDS - 1}; "
            f"got={len(got)} missing={missing[:20]}"
        )

    # Gold labels are first opened here, after every inference shard completed.
    gold = pd.read_csv(args.gold)
    if len(gold) != EXPECTED_RECORDS:
        raise ValueError(f"Expected {EXPECTED_RECORDS} gold rows, got {len(gold)}")

    metrics: Dict[str, Any] = {}
    for label in SUPPORTED:
        col = f"pred_{label}"
        if col not in pred.columns:
            raise ValueError(f"Missing prediction column {col}")
        y_true = gold[label].astype(int).to_numpy() == 1
        y_pred = pred[col].astype(bool).to_numpy()
        metrics[label] = _binary_metrics(y_true, y_pred)

    metrics["1dAVb"] = {
        "status": "NOT_SCORED",
        "reason": "DEDICATED_AV_BLOCK_ENGINE_NOT_YET_IMPLEMENTED",
        "positive_n": int((gold["1dAVb"].astype(int) == 1).sum()),
    }

    for label in ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]:
        pred[f"gold_{label}"] = gold[label].astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "code_test_predictions.csv", index=False)

    summary = {
        "validation_type": "FROZEN_EXTERNAL_DIGITAL_SIGNAL",
        "dataset_id": "code_test",
        "dataset_registry_status": dataset.get("status"),
        "records_evaluated": EXPECTED_RECORDS,
        "metrics": metrics,
        "unsupported_targets": ["1dAVb"],
        "source_hashes": {
            "gold_standard_sha256": _sha256(args.gold),
        },
        "anti_leakage": {
            "inference_jobs_received_gold_labels": False,
            "gold_opened_only_after_all_inference_jobs_completed": True,
            "threshold_tuning_allowed": False,
            "individual_label_debugging_allowed": False,
        },
    }
    (args.output_dir / "code_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
