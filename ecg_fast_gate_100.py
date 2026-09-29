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
    _any_target_positive,
    _candidate_codes,
    _canonical,
    _download,
    _ensure_record,
    _fusion_codes,
    _parse_codes,
    _published_codes,
    _target_positive,
)
from ecg_signal_measurements import analyze_canonical_ecg


FAST_GATE_VERSION = "MEDCALC_ECG_FAST_GATE_100_V1"
ALL_MEDCALC_TARGET_CODES = {
    code
    for spec in TARGETS.values()
    for code in spec["medcalc"]
}


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = list(data.get("cases") or [])
    ids = [int(row["ecg_id"]) for row in cases]
    positive_n = sum(str(row.get("role")) == "POSITIVE" for row in cases)
    negative_n = sum(str(row.get("role")) == "NEGATIVE_CONTROL" for row in cases)
    targets = {
        str(row.get("reference_target"))
        for row in cases
        if str(row.get("role")) == "POSITIVE"
    }
    expected_targets = set(TARGETS) - {"SINUS_BRADY"}
    assert data.get("version") == FAST_GATE_VERSION, data
    assert data.get("role") == "DEVELOPMENT_REGRESSION_ONLY", data
    assert data.get("report_as_validation") is False, data
    assert data.get("source_dataset") == "ptb_xl", data
    assert list(data.get("allowed_folds") or []) == [1,2,3,4,5,6,7,8], data
    assert set(data.get("forbidden_folds") or []) == {9,10}, data
    assert len(cases) == 100, len(cases)
    assert len(set(ids)) == 100, "ECG IDs must be unique"
    assert positive_n == 84, positive_n
    assert negative_n == 16, negative_n
    assert targets == expected_targets, (targets, expected_targets)
    return data


def _score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [r for r in rows if r["role"] == "POSITIVE"]
    negatives = [r for r in rows if r["role"] == "NEGATIVE_CONTROL"]

    per_target: dict[str, Any] = {}
    for target in sorted({r["reference_target"] for r in positives}):
        tr = [r for r in positives if r["reference_target"] == target]
        n = len(tr)
        candidate_n = sum(bool(r["candidate_hit"]) for r in tr)
        fusion_n = sum(bool(r["fusion_hit"]) for r in tr)
        final_n = sum(bool(r["final_hit"]) for r in tr)
        per_target[target] = {
            "positive_n": n,
            "candidate_positive_n": candidate_n,
            "fusion_positive_n": fusion_n,
            "final_positive_n": final_n,
            "candidate_sensitivity": candidate_n / n if n else None,
            "fusion_sensitivity": fusion_n / n if n else None,
            "final_sensitivity": final_n / n if n else None,
            "candidate_miss_n": sum(not r["candidate_hit"] for r in tr),
            "candidate_to_fusion_loss_n": sum(
                r["candidate_hit"] and not r["fusion_hit"] for r in tr
            ),
            "fusion_to_final_loss_n": sum(
                r["fusion_hit"] and not r["final_hit"] for r in tr
            ),
        }

    pn = len(positives)
    candidate_tp = sum(bool(r["candidate_hit"]) for r in positives)
    fusion_tp = sum(bool(r["fusion_hit"]) for r in positives)
    final_tp = sum(bool(r["final_hit"]) for r in positives)

    nn = len(negatives)
    candidate_fp = sum(bool(r["candidate_any_target"]) for r in negatives)
    fusion_fp = sum(bool(r["fusion_any_target"]) for r in negatives)
    final_fp = sum(bool(r["final_any_target"]) for r in negatives)

    return {
        "positive_n": pn,
        "negative_control_n": nn,
        "overall": {
            "candidate_sensitivity": candidate_tp / pn if pn else None,
            "fusion_sensitivity": fusion_tp / pn if pn else None,
            "final_sensitivity": final_tp / pn if pn else None,
            "candidate_specificity": (nn - candidate_fp) / nn if nn else None,
            "fusion_specificity": (nn - fusion_fp) / nn if nn else None,
            "final_specificity": (nn - final_fp) / nn if nn else None,
            "candidate_tp_n": candidate_tp,
            "fusion_tp_n": fusion_tp,
            "final_tp_n": final_tp,
            "candidate_fp_n": candidate_fp,
            "fusion_fp_n": fusion_fp,
            "final_fp_n": final_fp,
        },
        "per_target": per_target,
        "analysis_error_n": sum(bool(r.get("analysis_error")) for r in rows),
    }


def run_fast_gate(manifest_path: Path, workdir: Path) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    workdir.mkdir(parents=True, exist_ok=True)
    meta_path = workdir / "ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv", meta_path)
    meta = pd.read_csv(meta_path)
    meta["_codes"] = meta["scp_codes"].map(_parse_codes)
    by_id = {int(row.ecg_id): row for _, row in meta.iterrows()}
    allowed_folds = set(int(x) for x in manifest["allowed_folds"])
    records_root = workdir / "records"

    rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        ecg_id = int(case["ecg_id"])
        role = str(case["role"])
        target = case.get("reference_target")
        if ecg_id not in by_id:
            raise RuntimeError(f"Missing PTB-XL ecg_id {ecg_id}")
        row = by_id[ecg_id]
        fold = int(row.strat_fold)
        age = float(row.age)
        codes = dict(row._codes)
        if fold not in allowed_folds:
            raise RuntimeError(f"Fast-gate ECG {ecg_id} is in forbidden fold {fold}")
        if age < 18:
            raise RuntimeError(f"Fast-gate ECG {ecg_id} is pediatric: age={age}")

        if role == "POSITIVE":
            if target not in TARGETS:
                raise RuntimeError(f"Unknown target {target!r}")
            if not _target_positive(codes, TARGETS[target]["scp"]):
                raise RuntimeError(
                    f"ECG {ecg_id} no longer satisfies reference target {target}"
                )
        elif role == "NEGATIVE_CONTROL":
            if _any_target_positive(codes):
                raise RuntimeError(
                    f"Negative control {ecg_id} now matches a benchmark target"
                )
        else:
            raise RuntimeError(f"Unsupported fast-gate role {role!r}")

        result: dict[str, Any] = {
            "ecg_id": ecg_id,
            "fold": fold,
            "age": age,
            "role": role,
            "reference_target": target,
            "reference_codes": sorted(
                str(k) for k, v in codes.items() if float(v or 0.0) >= 80.0
            ),
            "analysis_error": None,
        }
        try:
            local_base = _ensure_record(records_root, str(row.filename_hr))
            rec = wfdb.rdrecord(str(local_base))
            analysis = analyze_canonical_ecg(
                _canonical(
                    rec.p_signal,
                    int(round(float(rec.fs))),
                    list(rec.sig_name),
                    ecg_id,
                )
            )
            candidate = set(_candidate_codes(analysis))
            fusion = set(_fusion_codes(analysis))
            final = set(_published_codes(analysis))
            result["candidate_codes"] = sorted(candidate)
            result["fusion_codes"] = sorted(fusion)
            result["final_codes"] = sorted(final)

            if role == "POSITIVE":
                expected = set(TARGETS[target]["medcalc"])
                result["candidate_hit"] = bool(expected & candidate)
                result["fusion_hit"] = bool(expected & fusion)
                result["final_hit"] = bool(expected & final)
                result["candidate_any_target"] = None
                result["fusion_any_target"] = None
                result["final_any_target"] = None
            else:
                result["candidate_hit"] = None
                result["fusion_hit"] = None
                result["final_hit"] = None
                result["candidate_any_target"] = bool(
                    ALL_MEDCALC_TARGET_CODES & candidate
                )
                result["fusion_any_target"] = bool(
                    ALL_MEDCALC_TARGET_CODES & fusion
                )
                result["final_any_target"] = bool(
                    ALL_MEDCALC_TARGET_CODES & final
                )
        except Exception as exc:
            result["analysis_error"] = f"{type(exc).__name__}: {exc}"
            result.setdefault("candidate_codes", [])
            result.setdefault("fusion_codes", [])
            result.setdefault("final_codes", [])
            if role == "POSITIVE":
                result["candidate_hit"] = False
                result["fusion_hit"] = False
                result["final_hit"] = False
                result["candidate_any_target"] = None
                result["fusion_any_target"] = None
                result["final_any_target"] = None
            else:
                result["candidate_hit"] = None
                result["fusion_hit"] = None
                result["final_hit"] = None
                result["candidate_any_target"] = True
                result["fusion_any_target"] = True
                result["final_any_target"] = True
        rows.append(result)

    metrics = _score(rows)
    return {
        "version": FAST_GATE_VERSION,
        "role": "DEVELOPMENT_REGRESSION_ONLY",
        "external_validation_claim_allowed": False,
        "dataset": "PTB-XL",
        "allowed_folds": sorted(allowed_folds),
        "case_count": len(rows),
        "metrics": metrics,
        "cases": rows,
        "invariant": (
            "FAST_GATE_100_IS_A_DEVELOPMENT_REGRESSION_SCREEN_AND_MUST_NOT_BE_"
            "REPORTED_AS_EXTERNAL_OR_INTERNAL_VALIDATION"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("ecg_fast_gate_100_manifest.json"),
    )
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/medcalc-fast-gate-100"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        data = _load_manifest(args.manifest)
        print(
            "MEDCALC_ECG_FAST_GATE_100_SELFTEST_PASS",
            len(data["cases"]),
        )
        return

    result = run_fast_gate(args.manifest, args.workdir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
