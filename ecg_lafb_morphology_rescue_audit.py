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

VERSION = "MEDCALC_LAFB_MORPH_RESCUE_AUDIT_V1"
CODE = "LAFB_COMPATIBLE"


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
    gate = dict(fusion.get("domain_gate") or {})

    positive_i = bool(crit.get("positive_qrs_I"))
    positive_avl = bool(crit.get("positive_qrs_aVL"))
    inferior_s = int(crit.get("inferior_s_dominant_n") or 0) >= 2
    small_q = bool(crit.get("small_q_superior_support"))
    qrs_lt_120 = bool(crit.get("qrs_lt_120ms"))
    primary_axis = bool(crit.get("primary_axis_minus45_to_minus90"))
    multilead_axis = bool(crit.get("multilead_axis_minus45_to_minus90"))

    candidate_present = bool(candidate)
    suppressed_score = bool(
        candidate_present
        and not bool(fusion.get("publishable"))
        and str(fusion.get("fusion_reason") or "") == "INSUFFICIENT_FUSED_EVIDENCE"
        and not (gate.get("blocked_by_conflicts") or [])
        and not (fusion.get("unresolved_required_measurements") or [])
        and not (fusion.get("boundary_failures") or [])
    )

    core_morph = bool(
        positive_i and positive_avl and inferior_s and qrs_lt_120
    )
    full_morph = bool(core_morph and small_q)

    core_rescue = bool(suppressed_score and core_morph and not published)
    full_rescue = bool(suppressed_score and full_morph and not published)

    group["n"] += 1
    group["candidate_present_n"] += int(candidate_present)
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["suppressed_score_n"] += int(suppressed_score)

    group["positive_i_n"] += int(positive_i)
    group["positive_avl_n"] += int(positive_avl)
    group["inferior_s_ge2_n"] += int(inferior_s)
    group["small_q_superior_n"] += int(small_q)
    group["qrs_lt_120_n"] += int(qrs_lt_120)
    group["primary_left_axis_n"] += int(primary_axis)
    group["multilead_left_axis_n"] += int(multilead_axis)

    group["core_morph_n"] += int(core_morph)
    group["full_morph_n"] += int(full_morph)
    group["core_rescue_n"] += int(core_rescue)
    group["full_rescue_n"] += int(full_rescue)

    if suppressed_score:
        evidence = {str(x) for x in (candidate.get("evidence") or [])}
        groups = {str(x) for x in (candidate.get("source_groups") or [])}
        group["suppressed_with_superior_group_n"] += int("SUPERIOR_LIMB_MORPHOLOGY" in groups)
        group["suppressed_with_inferior_group_n"] += int("INFERIOR_LIMB_MORPHOLOGY" in groups)
        group["suppressed_with_initial_q_group_n"] += int("INITIAL_Q" in groups)
        group["suppressed_with_axis_group_n"] += int("AXIS" in groups)
        group["suppressed_with_positive_i_avl_evidence_n"] += int("POSITIVE_I_AVL" in evidence)
        group["suppressed_with_inferior_s_evidence_n"] += int("INFERIOR_S_DOMINANT" in evidence)
        group["suppressed_with_small_q_evidence_n"] += int("SMALL_Q_SUPERIOR" in evidence)
        group["suppressed_with_left_axis_evidence_n"] += int("LEFT_AXIS" in evidence)


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
    aliases = set(TARGETS["LAFB"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"LAFB": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "LAFB"
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
        "counterfactuals": {
            "CORE_MORPH_NO_AXIS": (
                "suppressed LAFB candidate + positive I + positive aVL + "
                "S-dominant in >=2 inferior leads + QRS<120; axis not required"
            ),
            "FULL_MORPH_NO_AXIS": (
                "CORE_MORPH_NO_AXIS + existing small-q superior support"
            ),
        },
        "clinical_output_changed": False,
        "policy": (
            "FOLDS_1_TO_8_ONLY; FAST_HOLDOUT_EXCLUDED; NO_FOLD9; NO_FOLD10; "
            "NO_EXTERNAL_OR_FINAL_DATA; EXISTING_MORPHOLOGY_THRESHOLDS_ONLY; "
            "AGGREGATE_OUTPUT_ONLY"
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
