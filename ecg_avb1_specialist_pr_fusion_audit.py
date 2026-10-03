from __future__ import annotations

import argparse
import json
import math
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

VERSION = "MEDCALC_AVB1_SPECIALIST_PR_FUSION_AUDIT_V2"
CODE = "FIRST_DEGREE_AV_DELAY_COMPATIBLE"


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _pr_only_block(fusion: dict) -> bool:
    if bool(fusion.get("publishable")):
        return False

    gate = dict(fusion.get("domain_gate") or {})
    if gate.get("blocked_by_conflicts"):
        return False

    unresolved = {
        str(x)
        for x in (fusion.get("unresolved_required_measurements") or [])
    }
    boundary = [
        dict(x or {})
        for x in (fusion.get("boundary_failures") or [])
    ]
    boundary_metrics = {
        str(x.get("metric") or "")
        for x in boundary
        if str(x.get("metric") or "")
    }
    reason = str(fusion.get("fusion_reason") or "")

    if unresolved and not unresolved.issubset({"pr_ms"}):
        return False
    if boundary_metrics and not boundary_metrics.issubset({"pr_ms"}):
        return False

    if reason == "REQUIRED_MEASUREMENT_UNUSABLE":
        return unresolved == {"pr_ms"}
    if reason == "REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED":
        return boundary_metrics == {"pr_ms"} and not unresolved
    return False


def _would_pass_without_pr_block(fusion: dict) -> bool:
    gate = dict(fusion.get("domain_gate") or {})
    domain_ok = bool(gate.get("eligible"))
    score = float(fusion.get("score") or 0.0)
    threshold = float(fusion.get("prospective_score_threshold") or 0.0)
    sources = int(fusion.get("independent_evidence_n") or 0)
    min_sources = int(
        fusion.get("prospective_min_independent_sources") or 0
    )
    return bool(
        domain_ok
        and score >= threshold
        and sources >= min_sources
    )


def _pr_multilead_long_n(analysis: dict) -> int:
    consensus = dict(analysis.get("measurement_consensus") or {})
    pr = dict(((consensus.get("metrics") or {}).get("pr_ms") or {}))
    values = dict(pr.get("candidate_values") or {})
    confidences = dict(pr.get("candidate_confidences") or {})
    long_n = 0
    for lead, raw_value in values.items():
        try:
            value = float(raw_value)
            confidence = float(confidences.get(lead) or 0.0)
        except Exception:
            continue
        if not (math.isfinite(value) and math.isfinite(confidence)):
            continue
        if confidence >= 0.50 and value > 200.0:
            long_n += 1
    return int(long_n)


def _apply(group: Counter, analysis: dict) -> None:
    candidate = dict(
        (
            ((analysis.get("high_recall_candidates") or {}).get("by_code") or {})
            .get(CODE)
        )
        or {}
    )
    fusion = dict(
        (
            ((analysis.get("evidence_fusion") or {}).get("by_code") or {})
            .get(CODE)
        )
        or {}
    )
    published = CODE in _published_codes(analysis)

    candidate_present = bool(candidate)
    specialist = bool(candidate.get("specialist_confirmed"))
    pr_only = bool(candidate_present and _pr_only_block(fusion))
    otherwise_pass = bool(
        candidate_present and _would_pass_without_pr_block(fusion)
    )
    rescue = bool(
        candidate_present
        and specialist
        and pr_only
        and otherwise_pass
        and not published
    )
    pr_long_lead_n = _pr_multilead_long_n(analysis)

    evidence = {str(x) for x in (candidate.get("evidence") or [])}
    source_groups = {str(x) for x in (candidate.get("source_groups") or [])}

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["specialist_confirmed_candidate_n"] += int(
        candidate_present and specialist
    )
    group["candidate_suppressed_n"] += int(
        candidate_present and not bool(fusion.get("publishable"))
    )
    group["candidate_suppressed_pr_only_n"] += int(pr_only)
    group["candidate_pr_only_otherwise_pass_n"] += int(
        pr_only and otherwise_pass
    )
    group["specialist_pr_only_rescue_n"] += int(rescue)
    if rescue:
        group[f"rescue_pr_gt_200_leads_n:{pr_long_lead_n}"] += 1
        for required_n in (2, 3, 4, 5, 6):
            group[
                f"specialist_pr_only_rescue_ge{required_n}_pr_gt_200_leads_n"
            ] += int(pr_long_lead_n >= required_n)

    group["evidence_1_to_1_p_qrs_n"] += int("1_TO_1_P_QRS" in evidence)
    group["evidence_pr_median_gt_200_n"] += int(
        "PR_MEDIAN_GT_200MS" in evidence
    )
    group["evidence_pr_stable_n"] += int("PR_STABLE" in evidence)
    group["evidence_pr_gt_200_n"] += int("PR_GT_200MS" in evidence)
    group["source_av_specialist_n"] += int("AV_SPECIALIST" in source_groups)
    group["source_p_qrs_sequence_n"] += int(
        "P_QRS_SEQUENCE" in source_groups
    )
    group["source_pr_measurement_n"] += int(
        "PR_MEASUREMENT" in source_groups
    )

    if candidate_present and not bool(fusion.get("publishable")):
        group[
            f"fusion_reason:{str(fusion.get('fusion_reason') or 'NONE')}"
        ] += 1
        for x in fusion.get("unresolved_required_measurements") or []:
            group[f"unresolved:{str(x)}"] += 1
        for x in fusion.get("boundary_failures") or []:
            metric = str((x or {}).get("metric") or "")
            if metric:
                group[f"boundary:{metric}"] += 1
        for x in (fusion.get("domain_gate") or {}).get(
            "blocked_by_conflicts"
        ) or []:
            group[f"blocking_conflict:{str(x)}"] += 1


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
    negative_ids = {
        int(x) for x in selection["negative_control_ecg_ids"]
    }
    aliases = set(TARGETS["AVB1"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(
                lambda c: _target_positive(c, aliases)
            )
        )
    ].copy()

    groups = {"AVB1": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "AVB1"
        try:
            base = _retry(
                _ensure_record,
                records_root,
                str(row["filename_hr"]),
            )
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
            "AUDIT_ONLY: RESCUE FIRST_DEGREE_AV_DELAY_COMPATIBLE ONLY "
            "WHEN EXISTING AV SPECIALIST CONFIRMS THE CANDIDATE, PR_MS "
            "IS THE SOLE FUSION ABSTENTION/BOUNDARY FAILURE, DOMAIN GATE "
            "IS ELIGIBLE, AND THE EXISTING SCORE/SOURCE REQUIREMENTS "
            "ALREADY PASS. MULTILEAD PR SUPPORT IS CHARACTERIZED AT THE "
            "EXISTING CONFIDENCE FLOOR >=0.50 WITHOUT SELECTING A NEW "
            "CLINICAL CUTOFF"
        ),
        "clinical_output_changed": False,
        "policy": (
            "FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; "
            "NO_FOLD10; NO_EXTERNAL_OR_FINAL_DATA; NO_NEW_THRESHOLDS; "
            "AGGREGATE_OUTPUT_ONLY"
        ),
    }
    output.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
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
