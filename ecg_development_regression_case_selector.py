from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    TARGETS,
    _candidate_codes,
    _canonical,
    _download,
    _ensure_record,
    _fusion_codes,
    _published_codes,
    _target_positive,
    select_records,
)
from ecg_signal_measurements import analyze_canonical_ecg


SELECTOR_VERSION = "MEDCALC_ECG_DEVELOPMENT_REGRESSION_CASE_SELECTOR_V1"
ALLOWED_FOLDS = set(range(1, 9))
ALLOWED_STAGES = {
    "CANDIDATE_MISS",
    "CANDIDATE_TO_FUSION_LOSS",
    "FUSION_TO_FINAL_LOSS",
    "TRUE_POSITIVE",
}


def _parse_folds(raw: str) -> list[int]:
    folds = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not folds:
        raise ValueError("At least one development fold is required")
    forbidden = sorted(set(folds) - ALLOWED_FOLDS)
    if forbidden:
        raise ValueError(
            f"Known-case selection is development-only; forbidden folds: {forbidden}"
        )
    return folds


def _stage_match(
    *,
    expected_codes: set[str],
    candidate_codes: set[str],
    fusion_codes: set[str],
    final_codes: set[str],
    stage: str,
) -> bool:
    candidate = bool(expected_codes & candidate_codes)
    fusion = bool(expected_codes & fusion_codes)
    final = bool(expected_codes & final_codes)

    if stage == "CANDIDATE_MISS":
        return not candidate
    if stage == "CANDIDATE_TO_FUSION_LOSS":
        return candidate and not fusion
    if stage == "FUSION_TO_FINAL_LOSS":
        return fusion and not final
    if stage == "TRUE_POSITIVE":
        return final
    raise ValueError(f"Unsupported stage: {stage}")


def select_development_cases(
    *,
    target: str,
    stage: str,
    folds: list[int],
    max_cases: int,
) -> dict[str, Any]:
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}; choose one of {sorted(TARGETS)}")
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose one of {sorted(ALLOWED_STAGES)}")
    if not folds or not set(folds).issubset(ALLOWED_FOLDS):
        raise ValueError("Only PTB-XL development folds 1-8 are permitted")
    if max_cases < 1 or max_cases > 50:
        raise ValueError("max_cases must be between 1 and 50")

    spec = TARGETS[target]
    expected_codes = set(spec["medcalc"])

    with tempfile.TemporaryDirectory(prefix="medcalc-regression-select-") as td:
        root = Path(td)
        metadata_path = root / "ptbxl_database.csv"
        _download(f"{BASE}/ptbxl_database.csv", metadata_path)
        meta = pd.read_csv(metadata_path)
        selected, selection = select_records(meta, folds=folds)

        positives = selected[
            selected["_codes"].map(lambda codes: _target_positive(codes, spec["scp"]))
        ].copy()

        matches: list[dict[str, Any]] = []
        records_root = root / "records"

        for _, row in positives.iterrows():
            if len(matches) >= max_cases:
                break

            ecg_id = int(row["ecg_id"])
            fold = int(row["strat_fold"])
            if fold not in ALLOWED_FOLDS:
                raise RuntimeError(f"Selector encountered forbidden fold {fold}")

            filename_hr = str(row["filename_hr"])
            local_base = _ensure_record(records_root, filename_hr)
            rec = wfdb.rdrecord(str(local_base))
            analysis = analyze_canonical_ecg(
                _canonical(
                    rec.p_signal,
                    int(round(float(rec.fs))),
                    list(rec.sig_name),
                    ecg_id,
                )
            )

            candidate_codes = set(_candidate_codes(analysis))
            fusion_codes = set(_fusion_codes(analysis))
            final_codes = set(_published_codes(analysis))

            if not _stage_match(
                expected_codes=expected_codes,
                candidate_codes=candidate_codes,
                fusion_codes=fusion_codes,
                final_codes=final_codes,
                stage=stage,
            ):
                continue

            matches.append({
                "ecg_id": ecg_id,
                "fold": fold,
                "record_ref": str(ecg_id),
                "filename_hr": filename_hr,
                "target": target,
                "stage": stage,
                "reference_codes": sorted(
                    str(code)
                    for code, value in dict(row["_codes"]).items()
                    if float(value or 0.0) > 0.0
                ),
                "candidate_codes": sorted(candidate_codes),
                "fusion_codes": sorted(fusion_codes),
                "final_codes": sorted(final_codes),
                "regression_source": {
                    "dataset_id": "ptb_xl",
                    "fold": fold,
                    "record_ref": str(ecg_id),
                    "usage_role": "DEVELOPMENT_REGRESSION_ONLY",
                },
            })

    return {
        "version": SELECTOR_VERSION,
        "dataset": "PTB-XL",
        "role": "DEVELOPMENT_REGRESSION_CASE_SELECTION_ONLY",
        "external_validation_claim_allowed": False,
        "target": target,
        "stage": stage,
        "executed_folds": folds,
        "allowed_folds": sorted(ALLOWED_FOLDS),
        "fold9_accessed": False,
        "fold10_accessed": False,
        "external_dataset_accessed": False,
        "selected_case_n": len(matches),
        "cases": matches,
        "selection_summary": {
            "adult_records_selected_by_benchmark_policy": int(selection["selected_unique_records"]),
            "positive_available_by_target": selection["positive_available_by_target"],
        },
        "invariant": (
            "CASE_SELECTION_IS_DEVELOPMENT_ONLY_AND_SELECTED_RECORDS_MAY_BE_USED_"
            "FOR_REGRESSION_REPLAY_BUT_NEVER_AS_VALIDATION_EVIDENCE"
        ),
    }


def _selftest() -> None:
    assert _parse_folds("1,2,8") == [1, 2, 8]
    for raw in ("9", "10", "1,9", "8,10"):
        try:
            _parse_folds(raw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"selector accepted forbidden folds: {raw}")

    expected = {"RBBB_MORPHOLOGY_COMPATIBLE"}
    assert _stage_match(
        expected_codes=expected,
        candidate_codes=set(),
        fusion_codes=set(),
        final_codes=set(),
        stage="CANDIDATE_MISS",
    )
    assert _stage_match(
        expected_codes=expected,
        candidate_codes=expected,
        fusion_codes=set(),
        final_codes=set(),
        stage="CANDIDATE_TO_FUSION_LOSS",
    )
    assert _stage_match(
        expected_codes=expected,
        candidate_codes=expected,
        fusion_codes=expected,
        final_codes=set(),
        stage="FUSION_TO_FINAL_LOSS",
    )
    assert _stage_match(
        expected_codes=expected,
        candidate_codes=expected,
        fusion_codes=expected,
        final_codes=expected,
        stage="TRUE_POSITIVE",
    )
    print("MEDCALC_ECG_DEVELOPMENT_REGRESSION_CASE_SELECTOR_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(TARGETS))
    ap.add_argument("--stage", choices=sorted(ALLOWED_STAGES))
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--max-cases", type=int, default=5)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.target or not args.stage:
        raise ValueError("--target and --stage are required unless --selftest is used")

    result = select_development_cases(
        target=args.target,
        stage=args.stage,
        folds=_parse_folds(args.folds),
        max_cases=args.max_cases,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
