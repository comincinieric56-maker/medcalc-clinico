from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd


TARGET_PATTERNS = {
    "AF": ["atrial fibrillation"],
    "FLUTTER": ["atrial flutter"],
    "SINUS_BRADY": ["sinus bradycardia"],
    "SINUS_TACHY": ["sinus tachycardia"],
    "RBBB_COMPLETE": [
        "right bundle branch block",
        "right bundle-branch block",
    ],
    "LBBB": [
        "left bundle branch block",
        "left bundle-branch block",
    ],
    "LAFB": ["left anterior fascicular block", "left anterior hemiblock"],
    "LPFB": ["left posterior fascicular block", "left posterior hemiblock"],
    # SPH/AHA encodes first-degree AV delay as "Prolonged PR interval".
    "AVB1": [
        "prolonged pr interval",
        "first degree av block",
        "first-degree av block",
        "first degree atrioventricular",
        "first-degree atrioventricular",
    ],
    "AVB2": [
        "second-degree av block",
        "second degree av block",
        "2:1 av block",
        "av block, advanced",
    ],
    "AVB3": [
        "av block, complete",
        "third-degree av block",
        "third degree av block",
        "third degree atrioventricular",
        "third-degree atrioventricular",
        "complete atrioventricular block",
    ],
    "WPW": ["ventricular preexcitation", "ventricular pre-excitation", "wolff"],
}


def _hash(text: str, namespace: str) -> str:
    return hashlib.sha256((namespace + "|" + str(text)).encode("utf-8")).hexdigest()


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    raise KeyError(f"Missing column among {candidates}; columns={list(df.columns)}")


def _primary_codes(cell) -> list[str]:
    if pd.isna(cell):
        return []
    out = []
    for statement in str(cell).split(";"):
        statement = statement.strip()
        if not statement:
            continue
        out.append(statement.split("+")[0].strip())
    return out


def _target_code_map(code: pd.DataFrame) -> dict[str, list[str]]:
    code_col = _find_col(code, ["Code"])
    desc_col = _find_col(code, ["Description", "Statement", "Name"])
    mapping: dict[str, list[str]] = {key: [] for key in TARGET_PATTERNS}
    audit = []
    for _, row in code.iterrows():
        raw_code = str(row[code_col]).strip()
        desc = str(row[desc_col]).strip()
        norm = desc.lower()
        audit.append({"code": raw_code, "description": desc})
        for target, patterns in TARGET_PATTERNS.items():
            if not any(pattern in norm for pattern in patterns):
                continue
            if target in {"RBBB_COMPLETE", "LBBB"} and "incomplete" in norm:
                continue
            mapping[target].append(raw_code)
    missing = [k for k, values in mapping.items() if not values]
    # Some rare categories may genuinely be absent; retain the empty target for
    # transparent NOT_SCORABLE reporting rather than guessing numeric codes.
    return {"targets": mapping, "dictionary": audit, "missing_targets": missing}


def _choose_one_per_patient(df: pd.DataFrame, patient_col: str, ecg_col: str, namespace: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_record_hash"] = tmp[ecg_col].astype(str).map(lambda x: _hash(x, namespace))
    tmp = tmp.sort_values([patient_col, "_record_hash", ecg_col])
    return tmp.groupby(patient_col, as_index=False, sort=False).head(1).drop(columns=["_record_hash"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--records-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--general-patients", type=int, default=2000)
    ap.add_argument("--negative-control-patients", type=int, default=2000)
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata, dtype=str)
    code = pd.read_csv(args.code, dtype=str)
    ecg_col = _find_col(meta, ["ECG_ID"])
    patient_col = _find_col(meta, ["Patient_ID"])
    aha_col = _find_col(meta, ["AHA_Code"])
    age_col = _find_col(meta, ["Age"])
    sex_col = _find_col(meta, ["Sex"])

    target_info = _target_code_map(code)
    target_codes = {k: set(v) for k, v in target_info["targets"].items()}

    codes_per_record = meta[aha_col].map(_primary_codes)
    for target, codes in target_codes.items():
        meta[f"gold_{target}"] = codes_per_record.map(
            lambda xs, codes=codes: any(x in codes for x in xs)
        )

    meta["_patient_hash_general"] = meta[patient_col].astype(str).map(
        lambda x: _hash(x, "SPH_GENERAL_BLIND_V1")
    )
    patient_table = (
        meta[[patient_col, "_patient_hash_general"]]
        .drop_duplicates(patient_col)
        .sort_values(["_patient_hash_general", patient_col])
    )
    general_patients = set(
        patient_table.head(int(args.general_patients))[patient_col].astype(str)
    )
    general_records = _choose_one_per_patient(
        meta[meta[patient_col].astype(str).isin(general_patients)],
        patient_col,
        ecg_col,
        "SPH_GENERAL_RECORD_V1",
    )
    general_ids = set(general_records[ecg_col].astype(str))

    target_positive_ids: set[str] = set()
    target_positive_counts: dict[str, int] = {}
    for target in TARGET_PATTERNS:
        positive = meta[meta[f"gold_{target}"]].copy()
        positive = _choose_one_per_patient(
            positive, patient_col, ecg_col, f"SPH_TARGET_{target}_V1"
        )
        ids = set(positive[ecg_col].astype(str))
        target_positive_ids |= ids
        target_positive_counts[target] = len(ids)

    any_target = meta[[f"gold_{x}" for x in TARGET_PATTERNS]].any(axis=1)
    negative_pool = meta[~any_target].copy()
    negative_pool["_patient_hash_control"] = negative_pool[patient_col].astype(str).map(
        lambda x: _hash(x, "SPH_NEGATIVE_CONTROL_V1")
    )
    neg_patients = (
        negative_pool[[patient_col, "_patient_hash_control"]]
        .drop_duplicates(patient_col)
        .sort_values(["_patient_hash_control", patient_col])
        .head(int(args.negative_control_patients))
    )
    neg_patient_ids = set(neg_patients[patient_col].astype(str))
    negative_records = _choose_one_per_patient(
        negative_pool[negative_pool[patient_col].astype(str).isin(neg_patient_ids)],
        patient_col,
        ecg_col,
        "SPH_NEGATIVE_RECORD_V1",
    )
    negative_ids = set(negative_records[ecg_col].astype(str))

    selected_ids = general_ids | target_positive_ids | negative_ids
    selected = meta[meta[ecg_col].astype(str).isin(selected_ids)].copy()
    selected["cohort_general_blind"] = selected[ecg_col].astype(str).isin(general_ids)
    selected["cohort_target_positive"] = selected[ecg_col].astype(str).isin(target_positive_ids)
    selected["cohort_negative_control"] = selected[ecg_col].astype(str).isin(negative_ids)

    selected["_shard"] = selected[ecg_col].astype(str).map(
        lambda x: int(_hash(x, "SPH_SHARD_V1")[:8], 16) % int(args.shards)
    )
    selected = selected.sort_values(["_shard", ecg_col]).reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    gold_root = args.output_dir / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)

    gold_cols = [
        ecg_col, patient_col, "cohort_general_blind",
        "cohort_target_positive", "cohort_negative_control",
    ] + [f"gold_{x}" for x in TARGET_PATTERNS]
    selected[gold_cols].to_csv(gold_root / "sph_gold.csv", index=False)
    (gold_root / "sph_target_map.json").write_text(
        json.dumps(target_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shard_counts = {}
    for shard in range(int(args.shards)):
        rows = selected[selected["_shard"] == shard].copy()
        work = args.output_dir / f"work-{shard:02d}"
        rec_out = work / "records"
        rec_out.mkdir(parents=True, exist_ok=True)
        manifest = rows[[ecg_col, patient_col, age_col, sex_col]].copy()
        manifest.columns = ["record_id", "patient_id", "age", "sex"]
        manifest.to_csv(work / "manifest.csv", index=False)

        for record_id in manifest["record_id"].astype(str):
            src = args.records_dir / f"{record_id}.h5"
            if not src.is_file():
                raise FileNotFoundError(f"Missing SPH record: {src}")
            shutil.copy2(src, rec_out / src.name)

        tar_path = shard_root / f"sph-shard-{shard:02d}.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(work / "manifest.csv", arcname="manifest.csv")
            tf.add(rec_out, arcname="records")
        shutil.rmtree(work)
        shard_counts[f"{shard:02d}"] = len(rows)

    summary = {
        "version": "MEDCALC_SPH_SELECTION_V1",
        "selection_uses_labels_for_general_cohort": False,
        "general_patient_selection": "LOWEST_SHA256_PATIENT_ID_NAMESPACE_SPH_GENERAL_BLIND_V1",
        "general_patient_n": len(general_patients),
        "general_record_n": len(general_ids),
        "target_positive_selection": "ALL_TARGET_POSITIVE_PATIENTS_ONE_DETERMINISTIC_RECORD_PER_PATIENT",
        "target_positive_patient_record_counts": target_positive_counts,
        "negative_control_selection": "LABEL_NEGATIVE_PATIENTS_LOWEST_SHA256_ONE_RECORD_PER_PATIENT",
        "negative_control_record_n": len(negative_ids),
        "selected_union_record_n": len(selected),
        "shard_counts": shard_counts,
        "targets": list(TARGET_PATTERNS),
        "missing_dictionary_targets": target_info["missing_targets"],
    }
    (args.output_dir / "sph_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
