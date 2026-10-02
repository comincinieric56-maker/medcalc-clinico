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
    _bbb_qrs_audit,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _published_codes,
    _target_positive,
    select_records,
)
from ecg_signal_measurements import analyze_canonical_ecg

VERSION = "MEDCALC_RBBB_QRS_RESCUE_AUDIT_V1"
CODE = "RBBB_MORPHOLOGY_COMPATIBLE"


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _only_qrs_measurement_block(fusion: dict) -> bool:
    if bool(fusion.get("publishable")):
        return False

    gate = dict(fusion.get("domain_gate") or {})
    if gate.get("blocked_by_conflicts"):
        return False

    reason = str(fusion.get("fusion_reason") or "")
    unresolved = {
        str(x) for x in (fusion.get("unresolved_required_measurements") or [])
    }
    boundary = [
        dict(x or {}) for x in (fusion.get("boundary_failures") or [])
    ]
    boundary_metrics = {
        str(x.get("metric") or "")
        for x in boundary
        if str(x.get("metric") or "")
    }
    gate_unusable = {
        str(x) for x in (gate.get("unusable_measurements") or [])
    }

    if gate_unusable and not gate_unusable.issubset({"qrs_ms"}):
        return False
    if unresolved and not unresolved.issubset({"qrs_ms"}):
        return False
    if boundary_metrics and not boundary_metrics.issubset({"qrs_ms"}):
        return False

    if reason == "REQUIRED_MEASUREMENT_UNUSABLE":
        return unresolved == {"qrs_ms"}
    if reason == "REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED":
        return boundary_metrics == {"qrs_ms"} and not unresolved
    return False


def _policy_hit(b: dict) -> bool:
    return bool(
        int(b.get("wide_qrs_lead_n") or 0) >= 3
        and int(b.get("wide_qrs_limb_lead_n") or 0) >= 1
        and int(b.get("wide_qrs_precordial_lead_n") or 0) >= 2
        and int(b.get("qrs_ge_118_lead_n") or 0) >= 4
        and bool(b.get("rbbb_morphology"))
    )


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
    b = _bbb_qrs_audit(analysis)

    candidate_present = bool(candidate)
    only_qrs_block = bool(candidate_present and _only_qrs_measurement_block(fusion))
    policy_hit = _policy_hit(b)
    rescue = bool(candidate_present and not published and only_qrs_block and policy_hit)

    evidence = {str(x) for x in (candidate.get("evidence") or [])}
    full_morph_evidence = {
        "RIGHT_TERMINAL_R",
        "LATERAL_TERMINAL_S",
    }.issubset(evidence)

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["candidate_suppressed_n"] += int(
        candidate_present and not bool(fusion.get("publishable"))
    )
    group["candidate_suppressed_only_qrs_n"] += int(only_qrs_block)
    group["policy_hit_n"] += int(policy_hit)
    group["full_morph_candidate_n"] += int(
        candidate_present and full_morph_evidence
    )
    group["safe_rescue_n"] += int(rescue)

    group["wide_ge3_n"] += int(int(b.get("wide_qrs_lead_n") or 0) >= 3)
    group["distributed_wide_n"] += int(
        int(b.get("wide_qrs_limb_lead_n") or 0) >= 1
        and int(b.get("wide_qrs_precordial_lead_n") or 0) >= 2
    )
    group["ge4_qrs118_n"] += int(int(b.get("qrs_ge_118_lead_n") or 0) >= 4)
    group["rbbb_morphology_n"] += int(bool(b.get("rbbb_morphology")))

    if candidate_present and not bool(fusion.get("publishable")):
        group[f"fusion_reason:{str(fusion.get('fusion_reason') or 'NONE')}"] += 1
        for x in (fusion.get("domain_gate") or {}).get("blocked_by_conflicts") or []:
            group[f"blocking_conflict:{str(x)}"] += 1
        for x in fusion.get("unresolved_required_measurements") or []:
            group[f"unresolved:{str(x)}"] += 1
        for x in fusion.get("boundary_failures") or []:
            metric = str((x or {}).get("metric") or "")
            if metric:
                group[f"boundary:{metric}"] += 1


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
    aliases = set(TARGETS["RBBB_COMPLETE"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"RBBB_COMPLETE": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "RBBB_COMPLETE"
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
            "AUDIT_ONLY: RESCUE SUPPRESSED RBBB CANDIDATE ONLY WHEN THE SOLE "
            "SUPPRESSION IS QRS_MS, THERE IS NO BLOCKING CONFLICT, >=3 LEADS "
            "HAVE QRS>=120MS WITH LIMB+PRECORDIAL DISTRIBUTION, >=4 LEADS "
            "HAVE QRS>=118MS, AND FULL RBBB MORPHOLOGY IS PRESENT"
        ),
        "clinical_output_changed": False,
        "policy": (
            "FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; "
            "NO_EXTERNAL_OR_FINAL_DATA; AGGREGATE_OUTPUT_ONLY"
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
