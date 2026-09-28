from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DATASET_ID = "mimic_iv_ecg"
PATIENT_NAMESPACE = "MEDCALC_MIMIC_IV_ECG_PATIENT_V1"
RECORD_NAMESPACE = "MEDCALC_MIMIC_IV_ECG_RECORD_V1"
SHARD_NAMESPACE = "MEDCALC_MIMIC_IV_ECG_SHARD_V1"


def _hash(value: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}|{value}".encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    by_lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in by_lower:
            return by_lower[c.lower()]
    raise KeyError(f"Missing one of columns: {candidates}. Available={list(df.columns)}")


def prepare(
    record_list: Path,
    output_dir: Path,
    *,
    patient_n: int = 5000,
    shards: int = 10,
) -> dict:
    df = pd.read_csv(record_list, dtype=str)
    subject_col = _find_col(df, ["subject_id"])
    study_col = _find_col(df, ["study_id"])
    path_col = _find_col(df, ["path", "file_name"])

    work = df[[subject_col, study_col, path_col]].dropna().copy()
    work.columns = ["subject_id", "study_id", "record_path"]
    work["subject_id"] = work["subject_id"].astype(str)
    work["study_id"] = work["study_id"].astype(str)
    work["record_path"] = work["record_path"].astype(str).str.strip()

    if work.empty:
        raise ValueError("record_list produced no usable rows")
    if work["study_id"].duplicated().any():
        raise ValueError("study_id must be unique in MIMIC-IV-ECG record_list")

    patients = (
        work[["subject_id"]]
        .drop_duplicates()
        .assign(_patient_hash=lambda x: x["subject_id"].map(
            lambda v: _hash(v, PATIENT_NAMESPACE)
        ))
        .sort_values(["_patient_hash", "subject_id"])
    )
    if int(patient_n) <= 0 or int(patient_n) > len(patients):
        raise ValueError(
            f"patient_n must be in 1..{len(patients)}, got {patient_n}"
        )
    chosen_patients = set(
        patients.head(int(patient_n))["subject_id"].astype(str)
    )

    selected = work[work["subject_id"].isin(chosen_patients)].copy()
    selected["_record_hash"] = selected.apply(
        lambda r: _hash(
            f"{r['subject_id']}|{r['study_id']}",
            RECORD_NAMESPACE,
        ),
        axis=1,
    )
    selected = (
        selected
        .sort_values(["subject_id", "_record_hash", "study_id"])
        .groupby("subject_id", as_index=False)
        .first()
    )
    if len(selected) != int(patient_n):
        raise AssertionError(
            f"Expected {patient_n} one-per-patient rows, got {len(selected)}"
        )

    selected["_shard"] = selected["study_id"].map(
        lambda v: int(_hash(v, SHARD_NAMESPACE)[:8], 16) % int(shards)
    )
    selected = selected.sort_values(["_shard", "subject_id", "study_id"])
    selected = selected[
        ["subject_id", "study_id", "record_path", "_shard"]
    ].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = output_dir / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    shard_counts: dict[str, int] = {}
    for shard in range(int(shards)):
        rows = selected[selected["_shard"] == shard].drop(columns=["_shard"])
        out = manifest_root / f"mimic-shard-{shard:02d}.csv"
        rows.to_csv(out, index=False)
        shard_counts[f"{shard:02d}"] = int(len(rows))

    selected.drop(columns=["_shard"]).to_csv(
        output_dir / "mimic_selection_manifest.csv",
        index=False,
    )
    summary = {
        "version": "MEDCALC_MIMIC_IV_ECG_SELECTION_V1",
        "dataset_id": DATASET_ID,
        "selection_is_label_blind": True,
        "selection_uses_machine_measurements": False,
        "selection_uses_reports": False,
        "patient_selection": (
            "LOWEST_SHA256_SUBJECT_ID_NAMESPACE_"
            + PATIENT_NAMESPACE
        ),
        "record_selection": (
            "LOWEST_SHA256_SUBJECT_STUDY_NAMESPACE_"
            + RECORD_NAMESPACE
        ),
        "patient_n": int(patient_n),
        "record_n": int(len(selected)),
        "shards": int(shards),
        "shard_counts": shard_counts,
        "record_list_sha256": _sha256(record_list),
        "anti_leakage": {
            "machine_measurements_opened_during_selection": False,
            "cardiologist_reports_opened_during_selection": False,
            "threshold_tuning_allowed": False,
            "individual_record_debugging_allowed_after_scoring": False,
        },
    }
    (output_dir / "mimic_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-list", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patient-n", type=int, default=5000)
    ap.add_argument("--shards", type=int, default=10)
    args = ap.parse_args()
    summary = prepare(
        args.record_list,
        args.output_dir,
        patient_n=args.patient_n,
        shards=args.shards,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
