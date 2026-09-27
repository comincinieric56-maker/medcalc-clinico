from __future__ import annotations

from ecg_atrial_rhythm import _guideline_af_gate
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_reasoner import reason_ecg
from ecg_rhythm_consensus import build_rhythm_consensus, rr_irregularity_score


LEADS = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")


def _lead(hr: float, *, confidence: float = 0.95, r_count: int = 7, duration: float = 5.0):
    return {
        "evaluable": True,
        "heart_rate_bpm": hr,
        "confidence": confidence,
        "r_count": r_count,
        "duration_s": duration,
    }


def _graph(qrs_ms: float = 130.0) -> dict:
    return {
        "global": {
            "qrs_ms": {"value": qrs_ms, "confidence": 0.95},
            "heart_rate_bpm": {"value": 75.0, "confidence": 0.95},
        },
        "rhythm": {"regular": True, "confidence": 0.95},
        "leads": {lead: {"evaluable": True} for lead in LEADS},
        "relations": {"p_reproducible": True},
        "specialist_evidence": {
            "atrial_activity": {
                "p_wave_reproducible": True,
                "sinus_compatible": True,
                "rhythm_p_qrs_coupling_fraction": 0.95,
            },
            "atrial_mechanism": {"mechanism": "SINUS_COMPATIBLE", "confidence": 0.9},
            "wide_complex_tachycardia": {"wide_complex_tachycardia": False},
            "fascicular_conduction": {"classification": "NO_FASCICULAR_PATTERN_ESTABLISHED"},
            "measurement_consensus": {"remeasure_required": False, "remeasure_targets": []},
            "signal_integrity": {"per_lead": {}},
            "ectopy": {},
            "av_conduction": {"evaluable": False, "classification": "AV_CONDUCTION_NOT_EVALUABLE"},
            "preexcitation": {"classification": "NO_PREEXCITATION_PATTERN_ESTABLISHED"},
            "qrs_morphology": {"per_lead": {}},
        },
    }


def main() -> None:
    # A single undercounted rhythm lead must not force false bradycardia when
    # the rest of the ECG consistently measures a normal rate.
    per_lead = {
        "II": _lead(42.0, r_count=4, duration=10.0),
        "I": _lead(72.0),
        "III": _lead(73.0),
        "aVF": _lead(71.0),
        "V1": _lead(72.0),
        "V5": _lead(74.0),
    }
    rate = build_rhythm_consensus(per_lead, "II")
    assert rate["evaluable"], rate
    assert 70.0 <= rate["heart_rate_bpm"] <= 74.0, rate
    assert rate["selected_rhythm_lead_rate_outlier"], rate

    irregular = rr_irregularity_score({
        "rr_median_ms": 760.0,
        "rr_cv": 0.24,
        "rr_mad_ms": 125.0,
        "rr_rmssd_ms": 180.0,
        "rr_pnn50": 0.70,
    })
    assert irregular["score"] >= 0.80, irregular

    af, strong = _guideline_af_gate(
        p_reproducible=False,
        rr_irregularity=0.85,
        broad_entropy_score=0.60,
        periodicity_score=0.25,
        fwave_score=0.20,
        flutter_guard=False,
        ectopy_driven=False,
        ectopy_burden=0.0,
    )
    assert af and not strong

    # RBBB: terminal right-precordial R plus broad/deep terminal S in I is
    # enough morphological support when QRS is genuinely >=120 ms.
    g = _graph(132.0)
    g["specialist_evidence"]["qrs_morphology"]["per_lead"] = {
        "V1": {
            "evaluable": True, "r_prime_present": True,
            "qrs_polarity": "R_DOMINANT", "terminal_positive_mv": 0.35,
        },
        "V2": {"evaluable": True, "qrs_polarity": "R_DOMINANT", "terminal_positive_mv": 0.20},
        "I": {
            "evaluable": True, "qrs_polarity": "BIPHASIC",
            "terminal_negative_mv": -0.18, "terminal_s_duration_ms": 22.0,
        },
        "V6": {
            "evaluable": True, "qrs_polarity": "R_DOMINANT",
            "terminal_negative_mv": -0.02, "terminal_s_duration_ms": 10.0,
        },
    }
    c = analyze_crosslead_conduction(g)
    assert any(x["code"] == "RBBB_MORPHOLOGY_COMPATIBLE" for x in c["findings"]), c

    # LBBB should not be called from a negative V1 plus one vaguely delayed
    # lateral R. It requires a coherent lateral pattern.
    l = _graph(136.0)
    l["specialist_evidence"]["qrs_morphology"]["per_lead"] = {
        "V1": {
            "evaluable": True, "qrs_polarity": "S_DOMINANT",
            "terminal_positive_mv": 0.02,
        },
        "V2": {
            "evaluable": True, "qrs_polarity": "S_DOMINANT",
            "terminal_positive_mv": 0.03,
        },
        "I": {
            "evaluable": True, "qrs_polarity": "R_DOMINANT",
            "r_peak_time_ms": 64.0, "notched_or_double_r": False,
            "initial_q_present": False,
        },
        "aVL": {
            "evaluable": True, "qrs_polarity": "BIPHASIC",
            "r_peak_time_ms": 35.0, "notched_or_double_r": False,
            "initial_q_present": False,
        },
        "V5": {
            "evaluable": True, "qrs_polarity": "R_DOMINANT",
            "r_peak_time_ms": 40.0, "notched_or_double_r": False,
            "initial_q_present": False,
        },
        "V6": {
            "evaluable": True, "qrs_polarity": "R_DOMINANT",
            "r_peak_time_ms": 42.0, "notched_or_double_r": False,
            "initial_q_present": False,
        },
    }
    lc = analyze_crosslead_conduction(l)
    assert not any(x["code"] == "LBBB_MORPHOLOGY_COMPATIBLE" for x in lc["findings"]), lc

    # AV labels with a blocking consistency conflict must never leak into the
    # final reasoner output.
    av_graph = _graph(92.0)
    av_graph["specialist_evidence"]["av_conduction"] = {
        "evaluable": True,
        "classification": "FIRST_DEGREE_AV_DELAY_COMPATIBLE",
        "confidence": 0.92,
        "basis": ["1_TO_1_P_QRS", "PR_MEDIAN_GT_200MS"],
    }
    consistency = {
        "publication_allowed": False,
        "blocking_conflict": True,
        "remeasure_targets": [],
        "conflicts": [{
            "code": "FIRST_DEGREE_AV_DELAY_WITHOUT_PR_GT_200_OR_1_TO_1",
            "severity": "BLOCKING",
        }],
        "status": "BLOCKED",
    }
    reasoned = reason_ecg(
        av_graph,
        {"findings": [], "classification": "NO_SPECIFIC_CONDUCTION_PATTERN_ESTABLISHED"},
        consistency,
    )
    assert reasoned["av_conduction_finding"] is None, reasoned
    assert reasoned["diagnostic_summary"]["authoritative"], reasoned

    print("MEDCALC_ECG_CLINICAL_HARDENING_SELFTEST_PASS")


if __name__ == "__main__":
    main()
