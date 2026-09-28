from __future__ import annotations

from ecg_signal_reconstruction import resolve_calibration
from ecg_measurement_consensus import (
    MEASURED_HIGH_CONFIDENCE,
    MEASURED_WITH_UNCERTAINTY,
    REMEASURE_REQUIRED,
    UNMEASURABLE,
    build_measurement_consensus,
    threshold_relation,
)


def _lead_metric(value: float | None, confidence: float = 0.9) -> dict:
    return {
        "evaluable": True,
        "confidence": confidence,
        "metrics": {
            "qrs_ms": {"value": value, "confidence": confidence}
            if value is not None else {}
        },
        "r_peaks_samples": [],
    }


def main() -> None:
    calibration = resolve_calibration(
        {"x": 0.20, "y": 0.20},
        speed_mm_per_s=25.0,
        gain_mm_per_mv=10.0,
    )
    assert abs(calibration["horizontal_ms_per_pixel"] - 8.0) < 1e-9, calibration
    assert abs(calibration["vertical_mv_per_pixel"] - 0.02) < 1e-9, calibration
    assert calibration["timing_uncertainty_ms"] == calibration["horizontal_ms_per_pixel"]

    # Moderate cross-lead disagreement is uncertainty, not automatic remeasure.
    per_lead = {
        "I": _lead_metric(116.0),
        "II": _lead_metric(118.0),
    }
    c = build_measurement_consensus(
        {"fs": 500, "leads": {}},
        per_lead,
        {
            "qrs_ms": {"value": 104.0, "confidence": 0.75},
            "pr_ms": {"value": None, "confidence": 0.0},
            "qt_ms": {"value": 390.0, "confidence": 0.8},
            "p_duration_ms": {"value": 92.0, "confidence": 0.7},
        },
        {},
        {},
    )
    qrs = c["metrics"]["qrs_ms"]
    assert qrs["measurement_state"] == MEASURED_WITH_UNCERTAINTY, qrs
    assert "qrs_ms" not in c["remeasure_targets"], c
    assert c["metrics"]["pr_ms"]["measurement_state"] == UNMEASURABLE, c
    assert "pr_ms" in c["unmeasurable_targets"], c
    assert c["remeasure_required"] is False, c

    # Strong contradictory evidence still requires remeasurement.
    strong = build_measurement_consensus(
        {"fs": 500, "leads": {}},
        {
            "I": _lead_metric(140.0),
            "II": _lead_metric(146.0),
        },
        {
            "qrs_ms": {"value": 90.0, "confidence": 0.85},
            "pr_ms": {"value": 170.0, "confidence": 0.8},
            "qt_ms": {"value": 390.0, "confidence": 0.8},
            "p_duration_ms": {"value": 92.0, "confidence": 0.7},
        },
        {},
        {},
    )
    assert strong["metrics"]["qrs_ms"]["measurement_state"] == REMEASURE_REQUIRED, strong
    assert "qrs_ms" in strong["remeasure_targets"], strong
    assert strong["remeasure_required"] is True, strong

    # A precise enough finite value can be high-confidence.
    precise = build_measurement_consensus(
        {"fs": 500, "leads": {}},
        {
            "I": _lead_metric(131.0),
            "II": _lead_metric(133.0),
            "V5": _lead_metric(132.0),
        },
        {
            "qrs_ms": {"value": 132.0, "confidence": 0.92},
            "pr_ms": {"value": 170.0, "confidence": 0.85},
            "qt_ms": {"value": 390.0, "confidence": 0.85},
            "p_duration_ms": {"value": 92.0, "confidence": 0.8},
        },
        {},
        {},
    )
    assert precise["metrics"]["qrs_ms"]["measurement_state"] == MEASURED_HIGH_CONFIDENCE, precise
    assert threshold_relation(precise, "qrs_ms", 120.0) == "ABOVE", precise

    # Borderline values propagate uncertainty across the clinical boundary.
    borderline = {
        "metrics": {
            "qrs_ms": {
                "measurement_state": MEASURED_WITH_UNCERTAINTY,
                "canonical_value": 119.0,
                "uncertainty_interval": [113.0, 125.0],
            }
        }
    }
    assert threshold_relation(borderline, "qrs_ms", 120.0) == "OVERLAPS"

    print("MEDCALC_MEASUREMENT_CONSENSUS_V2_SELFTEST_PASS")


if __name__ == "__main__":
    main()
