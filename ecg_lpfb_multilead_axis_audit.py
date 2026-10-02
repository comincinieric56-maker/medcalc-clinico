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

VERSION = "MEDCALC_LPFB_MULTILEAD_AXIS_AUDIT_V1"
CODE = "LPFB_COMPATIBLE"


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

    fasc = dict(
        analysis.get("fascicular_conduction")
        or ((((analysis.get("feature_graph") or {}).get("specialist_evidence") or {}).get("fascicular_conduction")) or {})
    )
    crit = dict(fasc.get("criteria") or {})
    axis = dict(fasc.get("multilead_limb_qrs_axis") or {})
    gate = dict(fusion.get("domain_gate") or {})

    axis_deg = axis.get("degrees")
    try:
        axis_deg = float(axis_deg) if axis_deg is not None else None
    except Exception:
        axis_deg = None

    multilead_right_axis = bool(
        axis.get("evaluable")
        and axis_deg is not None
        and 90.0 <= axis_deg <= 180.0
    )
    primary_right_axis = bool(crit.get("axis_plus90_to_plus180"))
    superior_s = int(crit.get("superior_s_dominant_n") or 0) >= 2
    inferior_r = int(crit.get("inferior_r_dominant_n") or 0) >= 2
    small_q = bool(crit.get("small_q_inferior_support"))
    qrs_lt_120 = bool(crit.get("qrs_lt_120ms"))

    candidate_present = bool(candidate)
    suppressed_score = bool(
        candidate_present
        and not bool(fusion.get("publishable"))
        and str(fusion.get("fusion_reason") or "") == "INSUFFICIENT_FUSED_EVIDENCE"
        and not (gate.get("blocked_by_conflicts") or [])
        and not (fusion.get("unresolved_required_measurements") or [])
        and not (fusion.get("boundary_failures") or [])
    )

    exact_existing_morphology = bool(
        superior_s and inferior_r and qrs_lt_120
    )
    rescue = bool(
        suppressed_score
        and multilead_right_axis
        and exact_existing_morphology
        and not published
    )

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["suppressed_score_n"] += int(suppressed_score)

    group["primary_right_axis_n"] += int(primary_right_axis)
    group["multilead_axis_evaluable_n"] += int(bool(axis.get("evaluable")))
    group["multilead_right_axis_n"] += int(multilead_right_axis)
    group["superior_s_ge2_n"] += int(superior_s)
    group["inferior_r_ge2_n"] += int(inferior_r)
    group["small_q_inferior_n"] += int(small_q)
    group["qrs_lt_120_n"] += int(qrs_lt_120)
    group["exact_existing_morphology_n"] += int(exact_existing_morphology)
    group["safe_rescue_n"] += int(rescue)

    if suppressed_score:
        group["suppressed_multilead_right_axis_n"] += int(multilead_right_axis)
        group["suppressed_superior_s_ge2_n"] += int(superior_s)
        group["suppressed_inferior_r_ge2_n"] += int(inferior_r)
        group["suppressed_qrs_lt_120_n"] += int(qrs_lt_120)
        group["suppressed_exact_existing_morphology_n"] += int(exact_existing_morphology)


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
    aliases = set(TARGETS["LPFB"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"LPFB": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "LPFB"
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
            "AUDIT_ONLY: SUBSTITUTE EXISTING >=4-LIMB-LEAD LEAST-SQUARES AXIS "
            "+90_TO_+180 FOR THE PRIMARY AXIS COMPONENT, WHILE RETAINING THE "
            "EXISTING LPFB SUPERIOR-S, INFERIOR-R, AND QRS<120 REQUIREMENTS"
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
