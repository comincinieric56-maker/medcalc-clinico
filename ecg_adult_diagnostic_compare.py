from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


COMPARE_VERSION = "MEDCALC_ECG_DIAGNOSTIC_BENCHMARK_COMPARE_V1"


def _num(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None


def _delta(new: Any, old: Any) -> float | None:
    a = _num(new)
    b = _num(old)
    return (a - b) if a is not None and b is not None else None


def compare_benchmarks(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two like-for-like PTB-XL development benchmark outputs."""
    for field in ("dataset", "population", "role"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(
                f"Incompatible benchmark {field}: "
                f"{baseline.get(field)!r} != {candidate.get(field)!r}"
            )

    base_folds = list((baseline.get("fold_policy") or {}).get("executed_folds") or [])
    cand_folds = list((candidate.get("fold_policy") or {}).get("executed_folds") or [])
    if base_folds != cand_folds:
        raise ValueError(
            f"Benchmarks use different folds: {base_folds!r} != {cand_folds!r}"
        )

    base_metrics = dict(baseline.get("metrics") or {})
    cand_metrics = dict(candidate.get("metrics") or {})
    targets = sorted(set(base_metrics) & set(cand_metrics))

    rows: Dict[str, Dict[str, Any]] = {}
    improved: list[str] = []
    regressed: list[str] = []
    guardrail_breaches: list[str] = []

    for target in targets:
        old = dict(base_metrics.get(target) or {})
        new = dict(cand_metrics.get(target) or {})

        final_delta = _delta(new.get("final_sensitivity"), old.get("final_sensitivity"))
        spec_delta = _delta(
            new.get("specificity_clean_controls"),
            old.get("specificity_clean_controls"),
        )
        guardrail = _num(new.get("specificity_guardrail"))
        specificity = _num(new.get("specificity_clean_controls"))

        flags: list[str] = []
        if final_delta is not None and final_delta >= 0.02:
            improved.append(target)
            flags.append("FINAL_SENSITIVITY_IMPROVED_GE_0_02")
        if final_delta is not None and final_delta <= -0.02:
            regressed.append(target)
            flags.append("FINAL_SENSITIVITY_REGRESSED_GE_0_02")
        if (
            guardrail is not None
            and specificity is not None
            and specificity < guardrail
        ):
            guardrail_breaches.append(target)
            flags.append("SPECIFICITY_BELOW_GUARDRAIL")
        if spec_delta is not None and spec_delta <= -0.02:
            flags.append("SPECIFICITY_REGRESSED_GE_0_02")

        rows[target] = {
            "positive_n": new.get("positive_n"),
            "negative_control_n": new.get("negative_control_n"),
            "candidate_sensitivity": {
                "baseline": old.get("candidate_sensitivity"),
                "candidate": new.get("candidate_sensitivity"),
                "delta": _delta(
                    new.get("candidate_sensitivity"),
                    old.get("candidate_sensitivity"),
                ),
            },
            "fusion_sensitivity": {
                "baseline": old.get("fusion_sensitivity"),
                "candidate": new.get("fusion_sensitivity"),
                "delta": _delta(
                    new.get("fusion_sensitivity"),
                    old.get("fusion_sensitivity"),
                ),
            },
            "final_sensitivity": {
                "baseline": old.get("final_sensitivity"),
                "candidate": new.get("final_sensitivity"),
                "delta": final_delta,
            },
            "specificity_clean_controls": {
                "baseline": old.get("specificity_clean_controls"),
                "candidate": new.get("specificity_clean_controls"),
                "delta": spec_delta,
            },
            "final_published_n": {
                "baseline": old.get("final_published_n"),
                "candidate": new.get("final_published_n"),
                "delta": (
                    int(new.get("final_published_n") or 0)
                    - int(old.get("final_published_n") or 0)
                ),
            },
            "false_positive_n_on_clean_controls": {
                "baseline": old.get("false_positive_n_on_clean_controls"),
                "candidate": new.get("false_positive_n_on_clean_controls"),
                "delta": (
                    int(new.get("false_positive_n_on_clean_controls") or 0)
                    - int(old.get("false_positive_n_on_clean_controls") or 0)
                ),
            },
            "candidate_to_fusion_loss_n": {
                "baseline": old.get("candidate_to_fusion_loss_n"),
                "candidate": new.get("candidate_to_fusion_loss_n"),
                "delta": (
                    int(new.get("candidate_to_fusion_loss_n") or 0)
                    - int(old.get("candidate_to_fusion_loss_n") or 0)
                ),
            },
            "fusion_to_final_loss_n": {
                "baseline": old.get("fusion_to_final_loss_n"),
                "candidate": new.get("fusion_to_final_loss_n"),
                "delta": (
                    int(new.get("fusion_to_final_loss_n") or 0)
                    - int(old.get("fusion_to_final_loss_n") or 0)
                ),
            },
            "flags": flags,
        }

    return {
        "version": COMPARE_VERSION,
        "dataset": candidate.get("dataset"),
        "population": candidate.get("population"),
        "role": candidate.get("role"),
        "executed_folds": cand_folds,
        "baseline_benchmark_version": baseline.get("benchmark_version"),
        "candidate_benchmark_version": candidate.get("benchmark_version"),
        "targets": rows,
        "summary": {
            "improved_targets_ge_0_02_final_sensitivity": improved,
            "regressed_targets_ge_0_02_final_sensitivity": regressed,
            "specificity_guardrail_breaches": guardrail_breaches,
            "like_for_like_comparison": True,
        },
        "constraints": {
            "comparison_changes_engine_state": False,
            "safe_for_aggregate_development_artifacts": True,
        },
    }


def _selftest() -> None:
    base = {
        "benchmark_version": "A",
        "dataset": "PTB-XL",
        "population": "ADULT_AGE_GE_18",
        "role": "DEVELOPMENT_TUNING_ONLY",
        "fold_policy": {"executed_folds": [1,2,3,4,5,6,7,8]},
        "metrics": {
            "RBBB_COMPLETE": {
                "positive_n": 100,
                "negative_control_n": 400,
                "candidate_sensitivity": 1.0,
                "fusion_sensitivity": 0.34,
                "final_sensitivity": 0.27,
                "specificity_clean_controls": 0.99,
                "specificity_guardrail": 0.90,
                "final_published_n": 27,
                "false_positive_n_on_clean_controls": 4,
                "candidate_to_fusion_loss_n": 66,
                "fusion_to_final_loss_n": 7,
            }
        },
    }
    cand = json.loads(json.dumps(base))
    cand["benchmark_version"] = "B"
    row = cand["metrics"]["RBBB_COMPLETE"]
    row["fusion_sensitivity"] = 0.64
    row["final_sensitivity"] = 0.55
    row["specificity_clean_controls"] = 0.9875
    row["final_published_n"] = 55
    row["false_positive_n_on_clean_controls"] = 5
    row["candidate_to_fusion_loss_n"] = 36
    row["fusion_to_final_loss_n"] = 9

    out = compare_benchmarks(base, cand)
    rbbb = out["targets"]["RBBB_COMPLETE"]
    assert round(rbbb["final_sensitivity"]["delta"], 2) == 0.28, rbbb
    assert rbbb["final_published_n"]["delta"] == 28, rbbb
    assert rbbb["false_positive_n_on_clean_controls"]["delta"] == 1, rbbb
    assert out["summary"]["improved_targets_ge_0_02_final_sensitivity"] == [
        "RBBB_COMPLETE"
    ], out
    assert out["summary"]["specificity_guardrail_breaches"] == [], out
    print("MEDCALC_ECG_DIAGNOSTIC_BENCHMARK_COMPARE_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--candidate", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if args.baseline is None or args.candidate is None:
        raise ValueError("--baseline and --candidate are required")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = compare_benchmarks(baseline, candidate)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
