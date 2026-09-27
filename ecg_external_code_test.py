from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ecg_signal_measurements import analyze_canonical_ecg
from ecg_validation_guard import assert_external_dataset, load_registry


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DATASET_ID = "code_test"
FS = 400


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _crop_zero_padding(record: np.ndarray) -> np.ndarray:
    x = np.asarray(record, dtype=float)
    if x.ndim != 2 or x.shape[1] != 12:
        raise ValueError(f"Expected (samples,12), got {x.shape}")
    support = np.nanmax(np.abs(x), axis=1) > 1e-12
    idx = np.flatnonzero(support)
    if idx.size == 0:
        return x[:0]
    return x[int(idx[0]): int(idx[-1]) + 1]


def _canonical_from_code(record: np.ndarray, *, input_scale_mv: float) -> Dict[str, Any]:
    x = _crop_zero_padding(record) * float(input_scale_mv)
    leads: Dict[str, Any] = {}
    for col, lead in enumerate(LEADS):
        signal = np.asarray(x[:, col], dtype=float)
        finite = np.isfinite(signal)
        quality = np.where(finite, 2, 0).astype(np.uint8)
        leads[lead] = {
            "signal_mv": [float(v) if math.isfinite(float(v)) else None for v in signal],
            "quality_mask": quality.tolist(),
            "fs": FS,
            "duration_s": round(len(signal) / float(FS), 6),
            "source": "CODE_TEST_FROZEN_DIGITAL_SIGNAL",
            "confidence": 1.0,
            "status": "MEASURED",
        }
    return {
        "contract": "MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "fs": FS,
        "leads": leads,
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "source": "DIGITAL_EXTERNAL_REFERENCE_NOT_IMAGE_CALIBRATION",
            "confidence": 1.0,
        },
        "validation_provenance": {
            "dataset_id": DATASET_ID,
            "frozen_external": True,
            "input_scale_mv": float(input_scale_mv),
        },
    }


def _findings(measurements: Dict[str, Any]) -> set[str]:
    rows = (measurements.get("crosslead_conduction") or {}).get("findings") or []
    return {str(row.get("code") or "") for row in rows}


def _predict(measurements: Dict[str, Any]) -> Dict[str, Any]:
    global_m = measurements.get("global") or {}
    hr = (global_m.get("heart_rate_bpm") or {}).get("value")
    try:
        hr = float(hr) if hr is not None else None
    except Exception:
        hr = None

    atrial = measurements.get("atrial_mechanism") or {}
    mechanism = str(atrial.get("mechanism") or "")
    findings = _findings(measurements)
    consistency = measurements.get("consistency") or {}
    blocked = bool(consistency.get("blocking_conflict"))

    predictions = {
        "AF": bool(mechanism == "AF_COMPATIBLE" and not blocked),
        "RBBB": bool("RBBB_MORPHOLOGY_COMPATIBLE" in findings and not blocked),
        "LBBB": bool("LBBB_MORPHOLOGY_COMPATIBLE" in findings and not blocked),
        "SB": bool(hr is not None and hr < 60.0),
        "ST": bool(hr is not None and hr > 100.0),
        # Current reasoner does not yet have a dedicated AV-block engine.
        "1dAVb": None,
    }
    return {
        "predictions": predictions,
        "heart_rate_bpm": hr,
        "atrial_mechanism": mechanism,
        "atrial_confidence": atrial.get("confidence"),
        "conduction_findings": sorted(findings),
        "consistency_status": consistency.get("status"),
        "remeasure_required": bool(
            (measurements.get("measurement_consensus") or {}).get("remeasure_required")
        ),
        "signal_integrity_quality": (
            measurements.get("signal_integrity") or {}
        ).get("overall_quality"),
        "measurement_consensus_quality": (
            measurements.get("measurement_consensus") or {}
        ).get("overall_measurement_quality"),
        "qrs_ms": (global_m.get("qrs_ms") or {}).get("value"),
        "pr_ms": (global_m.get("pr_ms") or {}).get("value"),
        "qt_ms": (global_m.get("qt_ms") or {}).get("value"),
    }


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    def div(a: float, b: float) -> float | None:
        return float(a / b) if b else None

    sensitivity = div(tp, tp + fn)
    specificity = div(tn, tn + fp)
    ppv = div(tp, tp + fp)
    npv = div(tn, tn + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)

    return {
        "n": int(len(y_true)),
        "positive_n": int(np.sum(y_true)),
        "negative_n": int(np.sum(~y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def run(
    data_dir: Path,
    output_dir: Path,
    *,
    max_records: int = 0,
    input_scale_mv: float = 1.0,
) -> Dict[str, Any]:
    registry = load_registry()
    dataset = assert_external_dataset(DATASET_ID, registry)

    h5_path = data_dir / "ecg_tracings.hdf5"
    gold_path = data_dir / "annotations" / "gold_standard.csv"
    if not h5_path.exists() or not gold_path.exists():
        raise FileNotFoundError(
            "CODE-test files not found. Expected ecg_tracings.hdf5 and "
            "annotations/gold_standard.csv inside --data-dir."
        )

    import h5py

    # Critical frozen-test invariant: perform all ECG inference before opening
    # the gold-standard label file.
    rows: list[Dict[str, Any]] = []
    with h5py.File(h5_path, "r") as fh:
        traces = fh["tracings"]
        total = int(traces.shape[0])
        n = min(total, int(max_records)) if int(max_records) > 0 else total
        for idx in range(n):
            record = np.asarray(traces[idx], dtype=float)
            canonical = _canonical_from_code(record, input_scale_mv=input_scale_mv)
            measurements = analyze_canonical_ecg(canonical)
            pred = _predict(measurements)
            rows.append({"record_index": idx, **pred})
            if (idx + 1) % 25 == 0 or idx + 1 == n:
                print(f"CODE_TEST_INFERENCE {idx + 1}/{n}", flush=True)

    gold = pd.read_csv(gold_path)
    if len(gold) < len(rows):
        raise ValueError(f"Gold labels {len(gold)} shorter than inference rows {len(rows)}")
    gold = gold.iloc[: len(rows)].reset_index(drop=True)

    supported = ["RBBB", "LBBB", "SB", "AF", "ST"]
    metrics: Dict[str, Any] = {}
    for label in supported:
        y_true = gold[label].astype(int).to_numpy() == 1
        y_pred = np.asarray(
            [bool((row["predictions"] or {}).get(label)) for row in rows],
            dtype=bool,
        )
        metrics[label] = _binary_metrics(y_true, y_pred)

    metrics["1dAVb"] = {
        "status": "NOT_SCORED",
        "reason": "DEDICATED_AV_BLOCK_ENGINE_NOT_YET_IMPLEMENTED",
        "positive_n": int((gold["1dAVb"].astype(int) == 1).sum()),
    }

    prediction_table: list[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        flat = {
            key: value for key, value in row.items() if key != "predictions"
        }
        for label, value in (row.get("predictions") or {}).items():
            flat[f"pred_{label}"] = value
            flat[f"gold_{label}"] = int(gold.loc[row_idx, label])
        prediction_table.append(flat)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prediction_table).to_csv(
        output_dir / "code_test_predictions.csv", index=False
    )

    summary = {
        "validation_type": "FROZEN_EXTERNAL_DIGITAL_SIGNAL",
        "dataset_id": DATASET_ID,
        "dataset_registry_status": dataset.get("status"),
        "records_evaluated": len(rows),
        "input_scale_mv": float(input_scale_mv),
        "lead_order": LEADS,
        "sampling_rate_hz": FS,
        "source_hashes": {
            "ecg_tracings_sha256": _sha256(h5_path),
            "gold_standard_sha256": _sha256(gold_path),
        },
        "metrics": metrics,
        "unsupported_targets": ["1dAVb"],
        "anti_leakage": {
            "gold_loaded_after_all_inference": True,
            "threshold_tuning_allowed": False,
            "individual_label_debugging_allowed": False,
        },
    }
    (output_dir / "code_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("ecg_external_results"))
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument(
        "--input-scale-mv",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to stored CODE-test samples before MEDCALC. "
            "Default 1.0 follows observed dataset amplitudes; record this value "
            "and never tune it against test labels."
        ),
    )
    args = ap.parse_args()

    summary = run(
        args.data_dir,
        args.output_dir,
        max_records=args.max_records,
        input_scale_mv=args.input_scale_mv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
