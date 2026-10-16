#!/usr/bin/env python3
"""Synthetic contract tests for EKG V52. No real ECG data or clinical metrics."""
import numpy as np
import medcalc_ekg_unificado_v52 as u

def test_measurements():
    m = np.zeros((12, 5000), dtype=np.uint8)
    for q in range(400, 4600, 500):
        m[1, q-95:q-55] = 1
        m[1, q:q+45] = 2
        m[1, q+95:q+210] = 3
    r = u.measure_from_masks(np.zeros((5000, 12), dtype=np.float32), m)
    assert r["heart_rate_bpm"] == 60.0
    assert r["pr_ms"] == 190.0
    assert r["qrs_ms"] == 90.0
    assert r["qt_ms"] == 420.0
    assert abs(r["qtc"]["fridericia_ms"] - 420.0) < 0.1
    assert r["clinical_state"] == "CLINICAL_DEPLOYMENT_BLOCKED"

def test_insufficient_signal_no_misleading_intervals():
    x = np.zeros((12,5000), dtype=np.uint8)
    r = u.measure_from_masks(np.zeros((5000,12), dtype=np.float32), x)
    assert r["status"] == "NOT_ASSESSABLE"
    assert r["pr_ms"] is None and r["qt_ms"] is None
    assert all(v is None for v in r["qtc"].values())

def test_module_audit_does_not_load_models():
    a = u.audit()
    assert a["r27_binary_diagnoses_authorized"] is False
    assert a["image_to_12lead_calibrated_runtime_verified"] is False
    assert a["clinical_state"] == "CLINICAL_DEPLOYMENT_BLOCKED"


def test_h211_6x2_uses_real_corresponding_time_windows():
    import h211_bridge_v52 as bridge
    source = np.ones((5000, 12), dtype=np.float32)
    # Every original lead has distinct time evolution but remains in mV.
    for i in range(12):
        source[:, i] = np.linspace(-1.0, 1.0, 5000) + i / 100
    data = bridge.to_h211_6x2(source)
    assert len(data) == 13
    assert data["II_LONG"]["signal"].shape == (5000,)
    assert data["V1"]["signal"].shape == (2500,)
    np.testing.assert_allclose(data["aVF"]["signal"], source[:2500, 5])
    np.testing.assert_allclose(data["V1"]["signal"], source[2500:, 6])
    np.testing.assert_allclose(data["II_LONG"]["signal"], source[:,1])


def test_h211_v51_concordance_and_disagreement_are_research_only():
    import h211_bridge_v52 as bridge
    h = {"status": "PASS", "rhythm_regularity": "REGULAR",
         "measurements": {"HR_bpm": 61.0, "RR_representative_ms": 984.0,
                          "PR_ms": 164.0, "QRS_ms": 120.0, "QT_ms": 424.0,
                          "QTcF_ms": 422.0, "AXIS_deg": -15.0}}
    v = {"heart_rate_bpm": 60.9, "rr_median_ms": 986.0,
         "pr_ms": 162.0, "qrs_ms": 122.0, "qt_ms": 420.0,
         "qtc": {"fridericia_ms": 421.98}}
    out = bridge.combine_h211_v51(h, v)
    assert out["status"] == "DUAL_ENGINE_RESEARCH"
    assert out["unified"]["pr_ms"] == 164.0
    assert out["unified"]["axis_deg"] == -15.0
    assert out["clinical_state"] == "CLINICAL_DEPLOYMENT_BLOCKED"
    v_bad = {**v, "qrs_ms": 202.0}
    bad = bridge.combine_h211_v51(h, v_bad)
    assert bad["status"] == "DISCORDANT_REVIEW_REQUIRED"
    assert bad["unified"]["qrs_ms"] is None
    assert "qrs_ms" in bad["discrepancies"]
    irregular = bridge.combine_h211_v51({**h, "rhythm_regularity":"IRREGULAR"}, v)
    assert irregular["unified"]["qtc_fridericia_ms"] is None
    fail = bridge.combine_h211_v51({"status": "FAILED_CLOSED", "measurements": h["measurements"]}, v)
    assert fail["unified"]["pr_ms"] is None


if __name__ == "__main__":
    test_measurements()
    test_insufficient_signal_no_misleading_intervals()
    test_module_audit_does_not_load_models()
    test_h211_6x2_uses_real_corresponding_time_windows()
    test_h211_v51_concordance_and_disagreement_are_research_only()
    print("V52 synthetic contract tests passed; real-model and clinical tests pending")
