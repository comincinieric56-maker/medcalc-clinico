from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

DATASET_ID = "mimic_iv_ecg"
BOOTSTRAP_SEED = 20260928
BOOTSTRAP_N = 1000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _between(x: pd.Series, lo: float, hi: float) -> pd.Series:
    return x.where(x.between(lo, hi, inclusive="both"))


def _axis_error(engine: np.ndarray, comparator: np.ndarray) -> np.ndarray:
    return ((engine - comparator + 180.0) % 360.0) - 180.0


def _linear_error(engine: np.ndarray, comparator: np.ndarray) -> np.ndarray:
    return engine - comparator


def _bootstrap_ci(
    error: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> list[float] | None:
    x = np.asarray(error, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return None
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_boot), dtype=float)
    n = int(x.size)
    for i in range(int(n_boot)):
        sample = x[rng.integers(0, n, size=n)]
        values[i] = float(statistic(sample))
    return [
        round(float(np.percentile(values, 2.5)), 6),
        round(float(np.percentile(values, 97.5)), 6),
    ]


def _metric_summary(
    joined: pd.DataFrame,
    *,
    engine_col: str,
    comparator_col: str,
    error_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict:
    engine = _num(joined[engine_col])
    comparator = _num(joined[comparator_col])
    paired = engine.notna() & comparator.notna()
    e = engine[paired].to_numpy(dtype=float)
    c = comparator[paired].to_numpy(dtype=float)
    err = error_fn(e, c)
    abs_err = np.abs(err)

    n_total = int(len(joined))
    n_pair = int(paired.sum())
    comparator_n = int(comparator.notna().sum())
    engine_n = int(engine.notna().sum())

    if n_pair == 0:
        return {
            "selected_n": n_total,
            "comparator_available_n": comparator_n,
            "engine_available_n": engine_n,
            "paired_n": 0,
        }

    bias = float(np.mean(err))
    sd = float(np.std(err, ddof=1)) if n_pair > 1 else 0.0
    mae = float(np.mean(abs_err))
    median_ae = float(np.median(abs_err))
    p95_ae = float(np.percentile(abs_err, 95))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    return {
        "selected_n": n_total,
        "comparator_available_n": comparator_n,
        "engine_available_n": engine_n,
        "paired_n": n_pair,
        "comparator_availability_rate": round(
            comparator_n / n_total, 6
        ) if n_total else None,
        "engine_measurement_rate": round(
            engine_n / n_total, 6
        ) if n_total else None,
        "paired_rate": round(n_pair / n_total, 6) if n_total else None,
        "mean_signed_error": round(bias, 6),
        "signed_error_sd": round(sd, 6),
        "bland_altman_loa_lower": round(bias - 1.96 * sd, 6),
        "bland_altman_loa_upper": round(bias + 1.96 * sd, 6),
        "mae": round(mae, 6),
        "median_absolute_error": round(median_ae, 6),
        "p95_absolute_error": round(p95_ae, 6),
        "rmse": round(rmse, 6),
        "mae_bootstrap_95ci": _bootstrap_ci(
            err,
            lambda z: float(np.mean(np.abs(z))),
        ),
        "bias_bootstrap_95ci": _bootstrap_ci(
            err,
            lambda z: float(np.mean(z)),
            seed=BOOTSTRAP_SEED + 1,
        ),
    }


def _prepare_machine(machine: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id", "study_id", "rr_interval", "p_onset",
        "qrs_onset", "qrs_end", "t_end", "qrs_axis",
    }
    missing = required - set(machine.columns)
    if missing:
        raise ValueError(f"machine_measurements missing columns: {sorted(missing)}")

    m = machine.copy()
    m["subject_id"] = m["subject_id"].astype(str)
    m["study_id"] = m["study_id"].astype(str)

    rr = _between(_num(m["rr_interval"]), 250.0, 3000.0)
    p_on = _num(m["p_onset"]).where(_num(m["p_onset"]) > 0)
    q_on = _num(m["qrs_onset"]).where(_num(m["qrs_onset"]) > 0)
    q_end = _num(m["qrs_end"]).where(_num(m["qrs_end"]) > 0)
    t_end = _num(m["t_end"]).where(_num(m["t_end"]) > 0)

    pr_candidate = (q_on - p_on).where(p_on < q_on)
    qrs_candidate = (q_end - q_on).where(q_on < q_end)
    qt_candidate = (t_end - q_on).where((q_on < q_end) & (q_end < t_end))

    m["machine_hr_bpm"] = 60000.0 / rr
    m["machine_pr_ms"] = _between(pr_candidate, 50.0, 400.0)
    m["machine_qrs_ms"] = _between(qrs_candidate, 40.0, 250.0)
    m["machine_qt_ms"] = _between(qt_candidate, 150.0, 700.0)
    m["machine_qrs_axis_deg"] = _between(
        _num(m["qrs_axis"]),
        -180.0,
        180.0,
    )
    cols = [
        "subject_id", "study_id",
        "machine_hr_bpm", "machine_pr_ms", "machine_qrs_ms",
        "machine_qt_ms", "machine_qrs_axis_deg",
    ]
    return m[cols]


def score(
    predictions_dir: Path,
    machine_measurements: Path,
    selection_summary: Path,
    output: Path,
) -> dict:
    files = sorted(predictions_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError("No MIMIC prediction shard CSV files found.")

    pred = pd.concat(
        [pd.read_csv(p, dtype={"subject_id": str, "study_id": str}) for p in files],
        ignore_index=True,
    )
    if pred["study_id"].duplicated().any():
        raise ValueError("Duplicate MIMIC predictions by study_id")
    if pred["subject_id"].duplicated().any():
        raise ValueError("Frozen cohort must contain one ECG per subject")

    selection = json.loads(selection_summary.read_text(encoding="utf-8"))
    expected_n = int(selection["record_n"])
    if len(pred) != expected_n:
        raise ValueError(
            f"Inference is incomplete: predictions={len(pred)} expected={expected_n}"
        )

    # Anti-leakage boundary: machine comparator is opened only here, after all
    # inference shards are complete and immutable.
    usecols = [
        "subject_id", "study_id", "rr_interval", "p_onset", "qrs_onset",
        "qrs_end", "t_end", "qrs_axis",
    ]
    machine_raw = pd.read_csv(
        machine_measurements,
        dtype={"subject_id": str, "study_id": str},
        usecols=usecols,
    )
    machine = _prepare_machine(machine_raw)

    joined = pred.merge(
        machine,
        on=["subject_id", "study_id"],
        how="left",
        validate="one_to_one",
    )

    analysis_error = (
        joined.get("analysis_error", pd.Series("", index=joined.index))
        .fillna("")
        .astype(str)
        .str.len() > 0
    )
    remeasure = (
        joined.get("remeasure_required", pd.Series(False, index=joined.index))
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    metrics = {
        "heart_rate_bpm": _metric_summary(
            joined,
            engine_col="engine_hr_bpm",
            comparator_col="machine_hr_bpm",
            error_fn=_linear_error,
        ),
        "pr_ms": _metric_summary(
            joined,
            engine_col="engine_pr_ms",
            comparator_col="machine_pr_ms",
            error_fn=_linear_error,
        ),
        "qrs_ms": _metric_summary(
            joined,
            engine_col="engine_qrs_ms",
            comparator_col="machine_qrs_ms",
            error_fn=_linear_error,
        ),
        "qt_ms": _metric_summary(
            joined,
            engine_col="engine_qt_ms",
            comparator_col="machine_qt_ms",
            error_fn=_linear_error,
        ),
        "qrs_axis_deg": _metric_summary(
            joined,
            engine_col="engine_qrs_axis_deg",
            comparator_col="machine_qrs_axis_deg",
            error_fn=_axis_error,
        ),
    }

    summary = {
        "validation_id": "MIMIC_IV_ECG_FROZEN_MEASUREMENT_V1",
        "validation_type": "EXTERNAL_DIGITAL_SIGNAL_MACHINE_COMPARATOR",
        "dataset_id": DATASET_ID,
        "clinical_engine_baseline_sha": "7b03f1ada34c93246ec9b72fa63284269d2564ab",
        "records_scored": int(len(joined)),
        "patients_scored": int(joined["subject_id"].nunique()),
        "analysis_failure_rate": round(float(np.mean(analysis_error)), 6),
        "remeasure_required_rate": round(float(np.mean(remeasure)), 6),
        "metrics": metrics,
        "source_hashes": {
            "machine_measurements_sha256": _sha256(machine_measurements),
            "record_list_sha256": selection.get("record_list_sha256"),
        },
        "selection": selection,
        "anti_leakage": {
            "patient_level_one_record_each": True,
            "selection_label_blind": True,
            "machine_measurements_loaded_after_all_inference": True,
            "cardiologist_reports_used": False,
            "threshold_tuning_allowed": False,
            "individual_record_debugging_allowed": False,
            "row_level_joined_output_persisted": False,
        },
        "interpretation_constraints": [
            "This is an independent external measurement agreement study against ECG machine global measurements, not manual fiducial ground truth.",
            "Machine measurements are comparator values and may themselves contain algorithmic error.",
            "No individual record may be inspected to change MEDCALC after scoring.",
            "No threshold, detector, or diagnostic rule may be selected using this result.",
            "Diagnostic sensitivity/specificity claims are outside the scope of this measurement validation.",
        ],
        "clinical_validation_claim_allowed": False,
        "external_measurement_validation_claim_allowed": True,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--machine-measurements", type=Path, required=True)
    ap.add_argument("--selection-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    score(
        args.predictions_dir,
        args.machine_measurements,
        args.selection_summary,
        args.output,
    )


if __name__ == "__main__":
    main()
