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

VERSION = "MEDCALC_AF_FP_SPECIFICITY_AUDIT_V1"
CODE = "AF_COMPATIBLE"


def _retry(fn, *args, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except Exception:
            if attempt >= attempts:
                raise
            time.sleep(10 * attempt)


def _policy_flags(analysis: dict) -> dict[str, bool]:
    atrial = dict(analysis.get("atrial_activity") or {})
    mech = dict(analysis.get("atrial_mechanism") or {})
    feat = dict(mech.get("aggregate_features") or {})
    rhythm = dict(analysis.get("rhythm") or {})

    p_repro = bool(atrial.get("p_wave_reproducible"))
    coupling = float(atrial.get("rhythm_p_qrs_coupling_fraction") or 0.0)
    rr_irregularity = float(feat.get("rr_irregularity_score") or 0.0)
    ectopy_burden = float(feat.get("ectopy_burden") or 0.0)
    ectopy_driven = bool(feat.get("ectopy_driven_irregularity"))
    guideline = bool(feat.get("guideline_af_pattern"))
    multievidence = bool(feat.get("multievidence_af_pattern"))
    regular_rr = rhythm.get("regular") is True

    # Exact existing logic: this condition already gates one AF evidence path.
    ectopy_guard = bool(
        ectopy_driven
        and ectopy_burden >= 0.25
        and rr_irregularity < 0.75
    )

    # Exact existing consistency conflict, expected to already be blocked in
    # final publication; retained as an audit sanity check.
    pqrs_conflict = bool(
        str(mech.get("mechanism") or "") == "AF_COMPATIBLE"
        and p_repro
        and coupling >= 0.70
    )

    return {
        "REGULAR_RR_WARNING_BLOCK": regular_rr,
        "ECTOPY_GUARD_BLOCK": ectopy_guard,
        "REGULAR_OR_ECTOPY_BLOCK": regular_rr or ectopy_guard,
        "REPRODUCIBLE_P_BLOCK": p_repro,
        "NO_RR_IRREGULAR_SUPPORT_BLOCK": rr_irregularity < 0.45,
        "P_OR_NO_RR_SUPPORT_BLOCK": p_repro or rr_irregularity < 0.45,
        "REQUIRE_GUIDELINE_OR_MULTIEVIDENCE": not (guideline or multievidence),
        "EXISTING_PQRS_CONFLICT_SANITY": pqrs_conflict,
    }


def _apply(group: Counter, analysis: dict) -> None:
    published = CODE in _published_codes(analysis)
    candidate = dict(
        (((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    fusion = dict(
        (((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE))
        or {}
    )
    atrial = dict(analysis.get("atrial_activity") or {})
    mech = dict(analysis.get("atrial_mechanism") or {})
    feat = dict(mech.get("aggregate_features") or {})
    rhythm = dict(analysis.get("rhythm") or {})

    group["n"] += 1
    group["candidate_present_n"] += int(bool(candidate))
    group["fusion_publishable_n"] += int(bool(fusion.get("publishable")))
    group["final_published_n"] += int(published)
    group["specialist_af_n"] += int(str(mech.get("mechanism") or "") == CODE)
    group["p_reproducible_n"] += int(bool(atrial.get("p_wave_reproducible")))
    group["rhythm_regular_n"] += int(rhythm.get("regular") is True)
    group["guideline_af_pattern_n"] += int(bool(feat.get("guideline_af_pattern")))
    group["multievidence_af_pattern_n"] += int(bool(feat.get("multievidence_af_pattern")))
    group["ectopy_driven_n"] += int(bool(feat.get("ectopy_driven_irregularity")))

    if published:
        group["published_p_reproducible_n"] += int(bool(atrial.get("p_wave_reproducible")))
        group["published_regular_rr_n"] += int(rhythm.get("regular") is True)
        group["published_guideline_pattern_n"] += int(bool(feat.get("guideline_af_pattern")))
        group["published_multievidence_pattern_n"] += int(bool(feat.get("multievidence_af_pattern")))

        for name, suppress in _policy_flags(analysis).items():
            group[f"policy_suppress:{name}"] += int(bool(suppress))


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
    aliases = set(TARGETS["AF"]["scp"])

    rows = selected[
        (selected["_fold"].astype(int) == int(process_fold))
        & (
            selected["ecg_id"].astype(int).isin(negative_ids)
            | selected["_codes"].map(lambda c: _target_positive(c, aliases))
        )
    ].copy()

    groups = {"AF": Counter(), "CONTROL": Counter()}
    errors = Counter()
    records_root = workdir / "records"

    for _, row in rows.iterrows():
        ecg_id = int(row["ecg_id"])
        group = "CONTROL" if ecg_id in negative_ids else "AF"
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
        "clinical_output_changed": False,
        "policy": (
            "DESCRIPTIVE_COUNTERFACTUAL_AUDIT_ONLY; EXISTING_AF_GUARDS_AND_THRESHOLDS_ONLY; "
            "FOLDS_1_TO_8; FAST_EXCLUDED; NO_FOLD9_10_OR_EXTERNAL"
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
