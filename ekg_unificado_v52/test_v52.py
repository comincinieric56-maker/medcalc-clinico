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

if __name__ == "__main__":
    test_measurements()
    test_insufficient_signal_no_misleading_intervals()
    test_module_audit_does_not_load_models()
    print("V52 synthetic contract tests passed; real-model and clinical tests pending")
