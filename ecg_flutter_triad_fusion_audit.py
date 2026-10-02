from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _published_codes,
    _target_positive,
    select_records,
)
from ecg_signal_measurements import analyze_canonical_ecg

VERSION = "MEDCALC_FLUTTER_TRIAD_FUSION_AUDIT_V1"
CODE = "FLUTTER_OR_AT_COMPATIBLE"
TRIAD_GROUPS = {
    "ATRIAL_SPECTRAL",
    "ATRIAL_PERIODICITY",
    "ATRIAL_RATE",
}
TRIAD_EVIDENCE = {
    "ORGANIZED_ATRIAL_COMPATIBILITY",
    "ATRIAL_PERIODICITY",
    "ATRIAL_RATE_BAND_SUPPORT",
}


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _apply(group: Counter, analysis: dict) -> None:
    candidate = dict(
        (((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    fusion = dict(
        (((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    published = CODE in _published_codes(analysis)
    gate = dict(fusion.get("domain_gate") or {})

    candidate_present = bool(candidate)
    evidence = {str(x) for x in (candidate.get("evidence") or [])}
    groups = {str(x) for x in (candidate.get("source_groups") or [])}
    triad = bool(
        TRIAD_GROUPS.issubset(groups)
        and TRIAD_EVIDENCE.issubset(evidence)
    )
    clean_score_only = bool(
        candidate_present
        and not bool(fusion.get("publishable"))
        and str(fusion.get("fusion_reason") or "") == "INSUFFICIENT_FUSED_EVIDENCE"
        and not (gate.get("blocked_by_conflicts") or [])
        and not (fusion.get("unresolved_required_measurements") or [])
        and not (fusion.get("boundary_failures") or [])
    )
    rescue = bool(clean_score_only and triad and not published)

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["candidate_suppressed_n"] += int(
        candidate_present and not bool(fusion.get("publishable"))
    )
    group["clean_score_only_suppression_n"] += int(clean_score_only)
    group["complete_triad_candidate_n"] += int(candidate_present and triad)
    group["complete_triad_suppressed_n"] += int(clean_score_only and triad)
    group["safe_rescue_n"] += int(rescue)
    group["specialist_confirmed_candidate_n"] += int(
        candidate_present and bool(candidate.get("specialist_confirmed"))
    )

    if candidate_present:
        try:
            score = float(candidate.get("score") or 0.0)
        except Exception:
            score = 0.0
        if score < 0.50:
            group["score_lt_050_n"] += 1
        elif score < 0.60:
            group["score_050_059_n"] += 1
        elif score < 0.68:
            group["score_060_067_n"] += 1
        elif score < 0.80:
            group["score_068_079_n"] += 1
        else:
            group["score_ge_080_n"] += 1


def run(workdir: Path, output: Path, process_fold: int) -> dict:
    folds = [1,2,3,4,5,6,7,8]
    workdir.mkdir(parents=True, exist_ok=True)

    metadata_path = workdir / "ptbxl_database.csv"
    statements_path = workdir / "scp_statements.csv"
    _retry(_download, f"{BASE}/ptbxl_database.csv", metadata_path)
    _retry(_download, f"{BASE}/scp_statements.csv", statements_path)

    meta = pd.read_csv(metadata_path)
    selected, selection = select_records(
        meta,
        folds=folds,
        exclude_ecg_ids=_fast_gate_holdout_ids(),
    )
    negative_ids = {int(x) for x in selection["negative_control_ecg_ids"]}
    aliases = set(TARGETS["FLUTTER"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"FLUTTER": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "FLUTTER"
        try:
            base = _retry(_ensure_record, records_root, str(row["filename_hr"]))
            rec = wfdb.rdrecord(str(base))
            canonical = _canonical(
                rec.p_signal,
                int(round(float(rec.fs))),
                list(rec.sig_name),
                ecg_id,
            )
            analysis = analyze_canonical_ecg(canonical)
            _apply(groups[group], analysis)
        except Exception as exc:
            errors[f"{group}:{type(exc).__name__}"] += 1

    out = {
        "version": VERSION,
        "dataset": "PTB-XL",
        "role": "DEVELOPMENT_TUNING_ONLY",
        "folds": folds,
        "process_fold": int(process_fold),
        "fast_gate_holdout_excluded": True,
        "external_validation_claim_allowed": False,
        "groups": {k: dict(v) for k,v in groups.items()},
        "analysis_error_n": sum(errors.values()),
        "analysis_error_types": dict(errors),
        "counterfactual": (
            "AUDIT_ONLY: RESCUE FLUTTER/AT CANDIDATE ONLY WHEN THE SOLE "
            "SUPPRESSION IS INSUFFICIENT_FUSED_EVIDENCE AND THE EXISTING "
            "ATRIAL_SPECTRAL + ATRIAL_PERIODICITY + ATRIAL_RATE SOURCE TRIAD "
            "AND ITS THREE EXISTING EVIDENCE FLAGS ARE ALL PRESENT"
        ),
        "clinical_output_changed": False,
        "policy": (
            "FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; "
            "NO_EXTERNAL_OR_FINAL_DATA; NO_NEW_THRESHOLDS; AGGREGATE_OUTPUT_ONLY"
        ),
    }
    output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--process-fold", type=int, choices=[1,2,3,4,5,6,7,8], required=True)
    args = ap.parse_args()
    run(args.workdir, args.output, args.process_fold)
