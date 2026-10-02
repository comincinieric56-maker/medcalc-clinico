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

VERSION = "MEDCALC_AVB1_SPECIALIST_FUSION_RESCUE_AUDIT_V1"
CODE = "FIRST_DEGREE_AV_DELAY_COMPATIBLE"
REQUIRED_SPECIALIST_EVIDENCE = {
    "1_TO_1_P_QRS",
    "PR_MEDIAN_GT_200MS",
    "PR_STABLE",
}


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _only_pr_measurement_block(fusion: dict) -> bool:
    if bool(fusion.get("publishable")):
        return False
    reason = str(fusion.get("fusion_reason") or "")
    gate = dict(fusion.get("domain_gate") or {})
    if not bool(gate.get("eligible")):
        return False
    if gate.get("blocked_by_conflicts"):
        return False

    unresolved = {
        str(x) for x in (fusion.get("unresolved_required_measurements") or [])
    }
    boundary = [
        dict(x or {}) for x in (fusion.get("boundary_failures") or [])
    ]

    if reason == "REQUIRED_MEASUREMENT_UNUSABLE":
        return unresolved == {"pr_ms"}
    if reason == "REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED":
        metrics = {
            str(x.get("metric") or "")
            for x in boundary
            if str(x.get("metric") or "")
        }
        return bool(metrics) and metrics == {"pr_ms"} and not unresolved
    return False


def _apply(group: Counter, analysis: dict) -> None:
    candidate = dict(
        (((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    fusion = dict(
        (((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    av = dict(analysis.get("av_conduction") or {})
    published = CODE in _published_codes(analysis)

    candidate_present = bool(candidate)
    specialist = bool(candidate.get("specialist_confirmed"))
    evidence = {str(x) for x in (candidate.get("evidence") or [])}
    strong_signature = bool(
        candidate_present
        and specialist
        and REQUIRED_SPECIALIST_EVIDENCE.issubset(evidence)
    )
    only_pr_block = bool(candidate_present and _only_pr_measurement_block(fusion))
    rescue = bool(strong_signature and only_pr_block and not published)

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["specialist_confirmed_candidate_n"] += int(
        candidate_present and specialist
    )
    group["strong_specialist_signature_n"] += int(strong_signature)
    group["candidate_suppressed_n"] += int(
        candidate_present and not bool(fusion.get("publishable"))
    )
    group["candidate_suppressed_only_pr_n"] += int(only_pr_block)
    group["strong_signature_only_pr_block_n"] += int(rescue)

    group["av_evaluable_n"] += int(bool(av.get("evaluable")))
    group["av_one_to_one_n"] += int(bool(av.get("one_to_one")))
    group["av_stable_pr_n"] += int(bool(av.get("stable_pr")))
    group["av_atrial_regular_n"] += int(bool(av.get("atrial_sequence_regular")))
    group["av_first_degree_classification_n"] += int(
        str(av.get("classification") or "") == CODE
    )

    if candidate_present and not bool(fusion.get("publishable")):
        group[f"fusion_reason:{str(fusion.get('fusion_reason') or 'NONE')}"] += 1
        for target in fusion.get("unresolved_required_measurements") or []:
            group[f"unresolved:{str(target)}"] += 1
        for row in fusion.get("boundary_failures") or []:
            metric = str((row or {}).get("metric") or "")
            if metric:
                group[f"boundary:{metric}"] += 1


def run(workdir: Path, output: Path, process_fold: int) -> dict:
    folds = [1, 2, 3, 4, 5, 6, 7, 8]
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
    aliases = set(TARGETS["AVB1"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"AVB1": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = (
            "CONTROL"
            if ecg_id in negative_ids
            else "AVB1"
        )
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
        "groups": {k: dict(v) for k, v in groups.items()},
        "analysis_error_n": sum(errors.values()),
        "analysis_error_types": dict(errors),
        "counterfactual": (
            "AUDIT_ONLY: ALLOW SPECIALIST-CONFIRMED AVB1 THROUGH FUSION "
            "ONLY WHEN EXISTING 1_TO_1_P_QRS + PR_MEDIAN_GT_200MS + PR_STABLE "
            "EVIDENCE IS PRESENT AND THE SOLE SUPPRESSION IS PR_MS "
            "UNUSABLE/BOUNDARY UNCERTAINTY"
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
    ap.add_argument(
        "--process-fold",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        required=True,
    )
    args = ap.parse_args()
    run(args.workdir, args.output, args.process_fold)
