from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ecg_measurement_failure_audit import (
    aggregate_measurement_audits,
    audit_measurement_consensus,
)
from r27_model_manifest import inspect_r27_models


def main() -> None:
    consensus = {
        "remeasure_required": True,
        "remeasure_targets": ["qrs_ms", "r_peaks"],
        "unmeasurable_targets": ["pr_ms"],
        "uncertain_targets": ["qt_ms"],
        "metrics": {
            "qrs_ms": {
                "measurement_state": "REMEASURE_REQUIRED",
                "canonical_value": 102.0,
                "canonical_confidence": 0.9,
                "candidate_n": 4,
                "candidate_mad": 8.0,
                "canonical_vs_median_abs_diff": 48.0,
                "tolerance_ms": 20.0,
                "uncertainty_ms": 24.0,
            },
            "pr_ms": {
                "measurement_state": "UNMEASURABLE",
                "canonical_value": None,
                "canonical_confidence": 0.0,
                "candidate_n": 0,
                "candidate_mad": 0.0,
                "canonical_vs_median_abs_diff": None,
                "tolerance_ms": 20.0,
                "uncertainty_ms": None,
            },
            "qt_ms": {
                "measurement_state": "MEASURED_WITH_UNCERTAINTY",
                "canonical_value": 402.0,
                "canonical_confidence": 0.55,
                "candidate_n": 3,
                "candidate_mad": 12.0,
                "canonical_vs_median_abs_diff": 15.0,
                "tolerance_ms": 30.0,
                "uncertainty_ms": 18.0,
            },
            "p_duration_ms": {
                "measurement_state": "MEASURED_HIGH_CONFIDENCE",
                "canonical_value": 96.0,
                "canonical_confidence": 0.9,
                "candidate_n": 4,
                "candidate_mad": 2.0,
                "canonical_vs_median_abs_diff": 2.0,
                "tolerance_ms": 20.0,
                "uncertainty_ms": 4.0,
            },
        },
        "r_peak_verification": {
            "evaluable": True,
            "aggregate_agreement": 0.42,
            "evaluable_lead_n": 4,
        },
    }
    audit = audit_measurement_consensus(consensus)
    assert audit["metric_audit"]["qrs_ms"]["reason"] == (
        "CANONICAL_VS_CROSSLEAD_STRONG_DISAGREEMENT"
    ), audit
    assert audit["metric_audit"]["pr_ms"]["reason"] == "NO_CANONICAL_VALUE", audit
    assert audit["r_peak_reason"] == "R_PEAK_XQRS_STRONG_DISAGREEMENT", audit
    agg = aggregate_measurement_audits([audit, audit])
    assert agg["record_n"] == 2, agg
    assert agg["remeasure_n"] == 2, agg
    assert agg["remeasure_target_counts"]["qrs_ms"] == 2, agg
    assert agg["remeasure_target_counts"]["r_peaks"] == 2, agg
    print("MEDCALC_MEASUREMENT_FAILURE_AUDIT_PASS")

    # R27 model introspection must identify the frozen estimator type without
    # fitting, recalibrating, or mutating it.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        models = root / "compat" / "MEDCALC" / "fixture"
        models.mkdir(parents=True)

        et = Pipeline([
            ("classifier", ExtraTreesClassifier(
                n_estimators=7,
                max_depth=3,
                random_state=1,
            )),
        ])
        lr = Pipeline([
            ("classifier", LogisticRegression()),
        ])
        joblib.dump(et, models / "AF_EXTRATREES.joblib")
        joblib.dump(lr, models / "AVB1_LOGREG.joblib")

        manifest = inspect_r27_models(root)
        assert manifest["joblib_n"] == 2, manifest
        assert manifest["training_performed"] is False
        by_name = {row["filename"]: row for row in manifest["models"]}
        af_tree = by_name["AF_EXTRATREES.joblib"]["estimator"]
        av_tree = by_name["AVB1_LOGREG.joblib"]["estimator"]
        assert af_tree["class"] == "Pipeline", af_tree
        assert af_tree["steps"][-1]["estimator"]["class"] == "ExtraTreesClassifier", af_tree
        assert af_tree["steps"][-1]["estimator"]["n_estimators"] == 7, af_tree
        assert av_tree["steps"][-1]["estimator"]["class"] == "LogisticRegression", av_tree
        print("MEDCALC_R27_MODEL_MANIFEST_INTROSPECTION_PASS")


if __name__ == "__main__":
    main()
