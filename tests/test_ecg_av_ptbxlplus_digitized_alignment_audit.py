import numpy as np

from ecg_av_ptbxlplus_digitized_alignment_audit import (
    _align_r_peak,
    _event_support,
    _fit_time_map,
)


def test_r_peak_alignment_recovers_known_digitized_lag():
    native = np.zeros(5000, dtype=float)
    center = 1500
    x = np.arange(-40, 41)
    shape = np.exp(-0.5 * (x / 7.0) ** 2) - 0.35 * np.exp(-0.5 * ((x - 14) / 5.0) ** 2)
    native[center - 40:center + 41] = shape

    lag = 9
    digitized = np.zeros(5000, dtype=float)
    digitized[center + lag - 40:center + lag + 41] = shape
    quality = np.full(5000, 2, dtype=np.uint8)

    row = _align_r_peak(native, digitized, quality, center)
    assert row["status"] == "ALIGNED"
    assert row["lag_samples"] == lag
    assert row["correlation"] > 0.99


def test_time_map_recovers_scale_and_intercept():
    source = [500, 1500, 2500, 3500, 4500]
    rows = []
    for x in source:
        target = 1.002 * x + 4.0
        rows.append({
            "source_sample": x,
            "status": "ALIGNED",
            "lag_samples": int(round(target - x)),
            "correlation": 0.95,
        })
    fit = _fit_time_map(rows)
    assert fit["available"] is True
    assert abs(fit["source_to_digitized_sample_slope"] - 1.002) < 0.001
    assert abs(fit["source_to_digitized_sample_intercept"] - 4.0) < 2.0


def test_event_support_never_promotes_missing_or_interpolated_samples():
    quality = np.zeros(5000, dtype=np.uint8)
    quality[100:110] = 2
    quality[200:210] = 1
    fit = {
        "available": True,
        "source_to_digitized_sample_slope": 1.0,
        "source_to_digitized_sample_intercept": 0.0,
    }
    events = [
        {"sample": 105, "aux_note": "p-wave peak"},
        {"sample": 205, "aux_note": "R peak"},
        {"sample": 305, "aux_note": "p-wave peak"},
    ]
    result = _event_support(events, quality, fit)
    assert [row["support"] for row in result["events"]] == [
        "OBSERVED", "INTERPOLATED_ONLY", "MISSING"
    ]
    assert result["support_counts"] == {
        "INTERPOLATED_ONLY": 1,
        "MISSING": 1,
        "OBSERVED": 1,
    }
