from __future__ import annotations

import numpy as np

from ecg_measurement_consensus import _match_peaks
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_consistency_engine import evaluate_ecg_consistency
from ecg_reasoner import reason_ecg


def _base_graph() -> dict:
    leads = {
        lead: {
            "evaluable": True,
            "confidence": 0.95,
            "qrs_polarity": "BIPHASIC",
            "rs_ratio": 1.0,
        }
        for lead in ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    }
    return {
        "global": {
            "qrs_ms": {"value": 96.0, "confidence": 0.95},
            "heart_rate_bpm": {"value": 130.0, "confidence": 0.95},
        },
        "rhythm": {"regular": False, "confidence": 0.92},
        "leads": leads,
        "specialist_evidence": {
            "atrial_activity": {
                "p_wave_reproducible": False,
                "sinus_compatible": False,
                "rhythm_p_qrs_coupling_fraction": 0.05,
            },
            "atrial_mechanism": {
                "mechanism": "AF_COMPATIBLE",
                "confidence": 0.91,
            },
            "wide_complex_tachycardia": {
                "wide_complex_tachycardia": False,
            },
            "fascicular_conduction": {
                "classification": "NO_FASCICULAR_PATTERN_ESTABLISHED",
            },
            "measurement_consensus": {
                "remeasure_required": False,
                "remeasure_targets": [],
            },
        },
    }


def main() -> None:
    matched, errors = _match_peaks(
        np.asarray([100, 200, 300], dtype=int),
        np.asarray([101, 199, 302], dtype=int),
        5,
    )
    assert matched == 3
    assert max(abs(x) for x in errors) <= 2

    graph = _base_graph()
    conduction = analyze_crosslead_conduction(graph)
    consistency = evaluate_ecg_consistency(graph, conduction)
    reasoning = reason_ecg(graph, conduction, consistency)
    assert consistency["status"] == "PASS", consistency
    assert reasoning["primary_rhythm"]["code"] == "AF_COMPATIBLE", reasoning

    contradiction = _base_graph()
    contradiction["specialist_evidence"]["atrial_activity"].update({
        "p_wave_reproducible": True,
        "sinus_compatible": True,
        "rhythm_p_qrs_coupling_fraction": 0.90,
    })
    contradiction_conduction = analyze_crosslead_conduction(contradiction)
    contradiction_consistency = evaluate_ecg_consistency(
        contradiction, contradiction_conduction
    )
    assert contradiction_consistency["blocking_conflict"], contradiction_consistency
    blocked = reason_ecg(
        contradiction,
        contradiction_conduction,
        contradiction_consistency,
    )
    assert not blocked["publication_allowed"], blocked

    lafb = _base_graph()
    lafb["rhythm"]["regular"] = True
    lafb["specialist_evidence"]["atrial_activity"] = {
        "p_wave_reproducible": True,
        "sinus_compatible": True,
        "rhythm_p_qrs_coupling_fraction": 0.92,
    }
    lafb["specialist_evidence"]["atrial_mechanism"] = {
        "mechanism": "INDETERMINATE",
        "confidence": 0.30,
    }
    lafb["specialist_evidence"]["fascicular_conduction"] = {
        "classification": "LAFB_COMPATIBLE",
        "confidence": 0.90,
        "axis_deg": -60.0,
    }
    lafb["leads"]["I"]["qrs_polarity"] = "R_DOMINANT"
    lafb["leads"]["aVL"]["qrs_polarity"] = "R_DOMINANT"
    lafb["leads"]["II"]["qrs_polarity"] = "S_DOMINANT"
    lafb["leads"]["III"]["qrs_polarity"] = "S_DOMINANT"
    lafb["leads"]["aVF"]["qrs_polarity"] = "S_DOMINANT"
    lafb_conduction = analyze_crosslead_conduction(lafb)
    assert any(
        row["code"] == "LAFB_COMPATIBLE"
        for row in lafb_conduction["findings"]
    ), lafb_conduction
    lafb_consistency = evaluate_ecg_consistency(lafb, lafb_conduction)
    assert lafb_consistency["status"] == "PASS", lafb_consistency

    lafb["specialist_evidence"]["measurement_consensus"] = {
        "remeasure_required": True,
        "remeasure_targets": ["qrs_ms"],
    }
    recheck = evaluate_ecg_consistency(lafb, lafb_conduction)
    assert any(
        row["code"] == "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT"
        for row in recheck["conflicts"]
    ), recheck

    print("MEDCALC_ECG_SPECIALIST_REASONER_SELFTEST_PASS")


if __name__ == "__main__":
    main()
