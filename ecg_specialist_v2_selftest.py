from __future__ import annotations

from ecg_av_conduction import analyze_av_conduction
from ecg_atrial_rhythm import _guideline_af_gate
from ecg_consistency_engine import evaluate_ecg_consistency
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_preexcitation import analyze_preexcitation
from ecg_reasoner import reason_ecg


LEADS = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")


def base_graph(hr: float = 75.0) -> dict:
    return {
        "global": {
            "heart_rate_bpm": {"value": hr, "confidence": 0.95},
            "pr_ms": {"value": 160.0, "confidence": 0.95},
            "qrs_ms": {"value": 96.0, "confidence": 0.95},
        },
        "rhythm": {"regular": True, "confidence": 0.94, "evaluable": True, "lead": "II"},
        "leads": {lead: {"evaluable": True} for lead in LEADS},
        "relations": {
            "p_reproducible": True,
            "p_qrs_coupling_fraction": 0.92,
        },
        "specialist_evidence": {
            "atrial_activity": {
                "p_wave_reproducible": True,
                "sinus_compatible": True,
                "rhythm_p_qrs_coupling_fraction": 0.92,
            },
            "atrial_mechanism": {
                "mechanism": "SINUS_COMPATIBLE",
                "confidence": 0.90,
                "aggregate_features": {},
            },
            "wide_complex_tachycardia": {"wide_complex_tachycardia": False},
            "fascicular_conduction": {
                "classification": "NO_FASCICULAR_PATTERN_ESTABLISHED",
            },
            "measurement_consensus": {
                "remeasure_required": False,
                "remeasure_targets": [],
            },
            "signal_integrity": {
                "per_lead": {"II": {"rhythm_eligible": True}},
            },
            "ectopy": {
                "evaluable": True,
                "premature_burden": 0.0,
                "irregularity_may_be_ectopy_driven": False,
            },
            "av_conduction": {
                "evaluable": True,
                "classification": "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED",
            },
            "qrs_morphology": {"per_lead": {}},
        },
    }


def test_guideline_af_gate() -> None:
    af, strong = _guideline_af_gate(
        p_reproducible=False,
        rr_irregularity=0.72,
        broad_entropy_score=0.70,
        periodicity_score=0.20,
        fwave_score=0.45,
        flutter_guard=False,
        ectopy_driven=False,
        ectopy_burden=0.0,
    )
    assert af and not strong

    flutter_like, _ = _guideline_af_gate(
        p_reproducible=False,
        rr_irregularity=0.72,
        broad_entropy_score=0.70,
        periodicity_score=0.80,
        fwave_score=0.45,
        flutter_guard=True,
        ectopy_driven=False,
        ectopy_burden=0.0,
    )
    assert not flutter_like

    ectopy_like, strong_ectopy = _guideline_af_gate(
        p_reproducible=False,
        rr_irregularity=0.72,
        broad_entropy_score=0.55,
        periodicity_score=0.30,
        fwave_score=0.40,
        flutter_guard=False,
        ectopy_driven=True,
        ectopy_burden=0.15,
    )
    assert not ectopy_like and not strong_ectopy


def test_sinus_rate_reasoning() -> None:
    for hr, expected in [
        (48.0, "SINUS_BRADYCARDIA_COMPATIBLE"),
        (75.0, "SINUS_COMPATIBLE"),
        (125.0, "SINUS_TACHYCARDIA_COMPATIBLE"),
    ]:
        graph = base_graph(hr)
        conduction = analyze_crosslead_conduction(graph)
        consistency = evaluate_ecg_consistency(graph, conduction)
        reasoning = reason_ecg(graph, conduction, consistency)
        assert reasoning["primary_rhythm"]["code"] == expected, reasoning


def test_rbbb_and_lbbb_crosslead() -> None:
    rbbb = base_graph()
    rbbb["global"]["qrs_ms"] = {"value": 138.0, "confidence": 0.96}
    rbbb["specialist_evidence"]["qrs_morphology"] = {"per_lead": {
        "V1": {"evaluable": True, "r_prime_present": True, "qrs_polarity": "R_DOMINANT", "terminal_positive_mv": 0.5},
        "V2": {"evaluable": True, "r_prime_present": False, "qrs_polarity": "R_DOMINANT", "terminal_positive_mv": 0.3},
        "I": {"evaluable": True, "terminal_negative_mv": -0.15, "terminal_s_duration_ms": 45.0},
        "V6": {"evaluable": True, "terminal_negative_mv": -0.12, "terminal_s_duration_ms": 42.0},
    }}
    out = analyze_crosslead_conduction(rbbb)
    assert any(x["code"] == "RBBB_MORPHOLOGY_COMPATIBLE" for x in out["findings"]), out

    lbbb = base_graph()
    lbbb["global"]["qrs_ms"] = {"value": 152.0, "confidence": 0.96}
    lbbb["specialist_evidence"]["qrs_morphology"] = {"per_lead": {
        "V1": {"evaluable": True, "qrs_polarity": "S_DOMINANT", "terminal_positive_mv": 0.02},
        "V2": {"evaluable": True, "qrs_polarity": "S_DOMINANT", "terminal_positive_mv": 0.03},
        "I": {"evaluable": True, "qrs_polarity": "R_DOMINANT", "notched_or_double_r": True, "r_peak_time_ms": 74.0, "initial_q_present": False},
        "aVL": {"evaluable": True, "qrs_polarity": "R_DOMINANT", "notched_or_double_r": True, "r_peak_time_ms": 68.0, "initial_q_present": False},
        "V5": {"evaluable": True, "qrs_polarity": "R_DOMINANT", "notched_or_double_r": False, "r_peak_time_ms": 72.0, "initial_q_present": False},
        "V6": {"evaluable": True, "qrs_polarity": "R_DOMINANT", "notched_or_double_r": False, "r_peak_time_ms": 70.0, "initial_q_present": False},
    }}
    out2 = analyze_crosslead_conduction(lbbb)
    assert any(x["code"] == "LBBB_MORPHOLOGY_COMPATIBLE" for x in out2["findings"]), out2


def test_first_degree_av_delay() -> None:
    fs = 500
    p = [100, 600, 1100, 1600, 2100]
    r = [225, 725, 1225, 1725, 2225]  # 250 ms PR
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": fs,
            "r_count": len(r),
            "r_peaks_samples": r,
            "raw_p_peaks_samples": p,
            "atrial_activity": {"p_wave_reproducible": True},
        }
    }
    atrial = {"p_wave_reproducible": True}
    out = analyze_av_conduction(per_lead, atrial)
    assert out["classification"] == "FIRST_DEGREE_AV_DELAY_COMPATIBLE", out


def test_complete_av_block_gate() -> None:
    fs = 500
    # Organized P sequence at 100 bpm and independent ventricular escape near 43 bpm.
    p = [100, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500]
    r = [250, 950, 1650, 2350]
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": fs,
            "r_count": len(r),
            "r_peaks_samples": r,
            "raw_p_peaks_samples": p,
            "atrial_activity": {"p_wave_reproducible": False, "p_candidate_n": 2},
        }
    }
    out = analyze_av_conduction(per_lead, {"p_wave_reproducible": False})
    assert out["classification"] == "COMPLETE_AV_BLOCK_COMPATIBLE", out


def test_preexcitation_gate() -> None:
    graph = base_graph()
    graph["global"]["pr_ms"] = {"value": 105.0, "confidence": 0.95}
    graph["global"]["qrs_ms"] = {"value": 118.0, "confidence": 0.95}
    morph = {"per_lead": {
        "I": {"evaluable": True, "delta_slur_compatible": True},
        "V4": {"evaluable": True, "delta_slur_compatible": True},
    }}
    out = analyze_preexcitation(graph, morph)
    assert out["classification"] == "VENTRICULAR_PREEXCITATION_COMPATIBLE", out


def main() -> None:
    test_guideline_af_gate()
    test_sinus_rate_reasoning()
    test_rbbb_and_lbbb_crosslead()
    test_first_degree_av_delay()
    test_complete_av_block_gate()
    test_preexcitation_gate()
    print("MEDCALC_ECG_SPECIALIST_V2_SELFTEST_PASS")


if __name__ == "__main__":
    main()
