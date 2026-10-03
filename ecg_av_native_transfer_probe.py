"""Frozen-checkpoint transfer probe, native PTB-XL development signals only."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pandas as pd
import torch
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _target_positive,
)
from ecg_av_temporal_model import AVResearchModel, analyze_av_research


def probe(checkpoint: Path, root: Path) -> dict:
    torch.set_num_threads(2)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "ptbxl_database.csv"
    _download(BASE + "/ptbxl_database.csv", metadata)
    df = pd.read_csv(metadata)
    if len(df) < 20000:
        raise ValueError("Incomplete PTB-XL metadata download")
    holdout = json.loads(Path(__file__).with_name("ecg_fast_gate_100_manifest.json").read_text())
    protected_ids = {int(r["ecg_id"]) for r in holdout["cases"]}
    protected_patients = set(df.loc[df.ecg_id.isin(protected_ids), "patient_id"])
    adult = _adult_rows(df, list(range(1, 9)))
    adult = adult.loc[~adult.patient_id.isin(protected_patients)]
    adult = adult.sort_values(["_hash", "ecg_id"])
    masks = {
        "AVB2": adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB2"]["scp"]))
                 & ~adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB3"]["scp"])),
        "AVB3": adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB3"]["scp"])),
        "CONTROL": ~adult._codes.map(_any_target_positive),
    }
    positive_classes = {"AVB2": {"MOBITZ_I", "MOBITZ_II", "TWO_TO_ONE", "HIGH_GRADE"},
                        "AVB3": {"AV_DISSOCIATION"}}
    high_grade_classes = set.union(*positive_classes.values())
    model = AVResearchModel(checkpoint)
    groups = {group: adult.loc[mask].head(32 if group == "CONTROL" else 8)
              for group, mask in masks.items()}
    # Download independent public records concurrently; inference order stays fixed.
    filenames = [str(row.filename_hr) for rows in groups.values() for _, row in rows.iterrows()]
    download_errors = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {name: executor.submit(_ensure_record, root, name) for name in filenames}
        for name, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                download_errors[name] = type(exc).__name__
    report = {"role": "NATIVE_SIGNAL_DEVELOPMENT_TRANSFER_PROBE_ONLY",
              "checkpoint_sha256": model.sha256,
              "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
              "selection_policy": "Frozen hash order; up to 8 AVB2, 8 AVB3, 32 controls; folds 1-8; exclude FAST patients",
              "diagnostic_claim_allowed": False, "clinical_fusion_allowed": False,
              "image_validation_claim_allowed": False, "threshold_tuning_allowed": False,
              "limitations": ["Native signal input, not ECG images", "PTB-XL diagnostic labels, no expert P/QRS event truth",
                              "AV dissociation is not by itself proof of complete AV block",
                              "Uncalibrated top class is an experimental score, not a diagnosis"],
              "groups": {}, "analysis_error_n": 0}
    selected = []
    for group, rows in groups.items():
        counts, classes = Counter(), Counter()
        for _, row in rows.iterrows():
            counts["n"] += 1
            selected.append(int(row.ecg_id))
            try:
                if str(row.filename_hr) in download_errors:
                    raise RuntimeError(download_errors[str(row.filename_hr)])
                path = _ensure_record(root, str(row.filename_hr))
                record = wfdb.rdrecord(str(path))
                canonical = _canonical(record.p_signal, int(record.fs), record.sig_name, int(row.ecg_id))
                result = analyze_av_research(canonical, model)
                probs = result.get("probabilities") or {}
                top = max(probs, key=probs.get) if probs else "NOT_EVALUABLE"
                classes[top] += 1
                evaluable = result.get("reason") == "UNVALIDATED_RESEARCH_MODEL"
                counts["sufficient_event_candidates_n"] += int(evaluable)
                counts["top_class_match_n"] += int(evaluable and top in positive_classes.get(group, set()))
                counts["high_grade_top_class_n"] += int(evaluable and top in high_grade_classes)
                assert result["diagnostic_claim_allowed"] is False
                assert result["clinical_fusion_allowed"] is False
            except Exception as exc:
                counts["analysis_error_n"] += 1
                report["analysis_error_n"] += 1
                print(f"PROBE_ERROR {group} {type(exc).__name__}", flush=True)
            print(f"PROBE {group} {counts['n']}/{len(rows)}", flush=True)
        report["groups"][group] = {**dict(counts), "top_class_counts": dict(classes)}
    report["selected_record_ids_sha256"] = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.checkpoint, args.data_root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
