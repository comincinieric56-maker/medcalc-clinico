from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS = [
    "AF","FLUTTER","SINUS_BRADY","SINUS_TACHY","RBBB_COMPLETE","LBBB",
    "LAFB","LPFB","AVB1","AVB2","AVB3","WPW",
]


def _get(d: dict, *path: str) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _delta(new: Any, old: Any) -> float | None:
    try:
        if new is None or old is None:
            return None
        return float(new) - float(old)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--engine-sha", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    current = json.loads(args.current.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    if int(current.get("records_scored") or 0) != int(baseline.get("records_scored") or 0):
        raise ValueError("SPH record count differs from frozen baseline selection.")
    if int(current.get("patients_scored") or 0) != int(baseline.get("patients_scored") or 0):
        raise ValueError("SPH patient count differs from frozen baseline selection.")
    if current.get("selection") != baseline.get("selection"):
        raise ValueError("SPH deterministic selection summary differs from frozen baseline.")

    metrics = {}
    for target in TARGETS:
        new = (current.get("metrics") or {}).get(target) or {}
        old = (baseline.get("metrics") or {}).get(target) or {}
        new_general = new.get("general_blind_cohort") or {}
        old_general = old.get("general_blind_cohort") or {}
        new_pos = new.get("all_external_positive_cases") or {}
        old_pos = old.get("all_external_positive_cases") or {}

        metrics[target] = {
            "general_blind": {
                "positive_n": new_general.get("positive_n"),
                "baseline_sensitivity": old_general.get("sensitivity"),
                "v3_sensitivity": new_general.get("sensitivity"),
                "delta_sensitivity": _delta(new_general.get("sensitivity"), old_general.get("sensitivity")),
                "baseline_specificity": old_general.get("specificity"),
                "v3_specificity": new_general.get("specificity"),
                "delta_specificity": _delta(new_general.get("specificity"), old_general.get("specificity")),
                "baseline_ppv": old_general.get("ppv"),
                "v3_ppv": new_general.get("ppv"),
                "delta_ppv": _delta(new_general.get("ppv"), old_general.get("ppv")),
                "baseline_f1": old_general.get("f1"),
                "v3_f1": new_general.get("f1"),
                "delta_f1": _delta(new_general.get("f1"), old_general.get("f1")),
            },
            "all_frozen_positive_cases": {
                "positive_n": new_pos.get("positive_n"),
                "baseline_sensitivity": old_pos.get("sensitivity"),
                "v3_sensitivity": new_pos.get("sensitivity"),
                "delta_sensitivity": _delta(new_pos.get("sensitivity"), old_pos.get("sensitivity")),
            },
            "negative_controls": {
                "baseline_specificity": old.get("specificity_predeclared_negative_controls"),
                "v3_specificity": new.get("specificity_predeclared_negative_controls"),
                "delta_specificity": _delta(
                    new.get("specificity_predeclared_negative_controls"),
                    old.get("specificity_predeclared_negative_controls"),
                ),
            },
        }

    out = {
        "rebenchmark_id": "SPH_V3_HISTORICAL_REBENCHMARK_V1",
        "status": "HISTORICAL_REBENCHMARK_NOT_INDEPENDENT_VALIDATION",
        "dataset": "SPH",
        "engine_sha": args.engine_sha,
        "baseline_engine_sha": _get(baseline, "baseline_metadata", "workflow_head_sha")
            or _get(baseline, "baseline_metadata", "baseline_commit"),
        "records_scored": current.get("records_scored"),
        "patients_scored": current.get("patients_scored"),
        "same_deterministic_selection_as_frozen_baseline": True,
        "analysis_failure_rate": {
            "baseline": baseline.get("analysis_failure_rate"),
            "v3": current.get("analysis_failure_rate"),
            "delta": _delta(current.get("analysis_failure_rate"), baseline.get("analysis_failure_rate")),
        },
        "reasoner_abstention_rate": {
            "baseline": baseline.get("reasoner_abstention_rate"),
            "v3": current.get("reasoner_abstention_rate"),
            "delta": _delta(current.get("reasoner_abstention_rate"), baseline.get("reasoner_abstention_rate")),
        },
        "remeasure_required_rate": {
            "baseline": baseline.get("remeasure_required_rate"),
            "v3": current.get("remeasure_required_rate"),
            "delta": _delta(current.get("remeasure_required_rate"), baseline.get("remeasure_required_rate")),
        },
        "metrics": metrics,
        "interpretation_constraints": [
            "SPH is a consumed external baseline; this is not a new independent validation.",
            "The same deterministic frozen selection and scoring mapping are reused only for aggregate historical comparison.",
            "Results must not be used to select or tune V3 thresholds, feature weights, or case-specific rules.",
            "No row-level gold/prediction artifact is published by this workflow.",
            "A future independent claim requires a still-untouched cohort such as the provisionally locked MIMIC-IV-ECG protocol.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
