from __future__ import annotations

from ecg_atrial_rhythm import _guideline_af_gate
from ecg_av_conduction import analyze_av_conduction
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

    # First-degree AV delay can still be recognized from a reliable global PR
    # when no single lead has enough raw P candidates, but only with reproducible
    # P-QRS coupling and high-confidence PR >200 ms.
    first_degree = analyze_av_conduction(
        {},
        {
            "p_wave_reproducible": True,
            "pr_reportable": True,
            "rhythm_p_qrs_coupling_fraction": 0.92,
        },
        global_metrics={
            "pr_ms": {"value": 224.0, "confidence": 0.90},
        },
    )
    assert first_degree["classification"] == "FIRST_DEGREE_AV_DELAY_COMPATIBLE", first_degree
    assert first_degree["source"] == "GLOBAL_PR_CONSENSUS_FALLBACK", first_degree

    # Mechanism-level AV-block regressions. These fixtures exercise only
    # deterministic P/QRS event timing; they are engineering tests, not
    # clinical-validation cases.
    atrial_support = {"p_wave_reproducible": True}

    def av_lead(p, r):
        return {
            "II": {
                "evaluable": True,
                "fs": 500,
                "confidence": 0.95,
                "raw_p_peaks_samples": p,
                "r_peaks_samples": r,
                "atrial_activity": {"p_wave_reproducible": True},
            }
        }

    two_to_one = analyze_av_conduction(
        av_lead([100, 350, 600, 850, 1100, 1350], [180, 680, 1180]),
        atrial_support,
    )
    assert two_to_one["classification"] == "TWO_TO_ONE_AV_BLOCK_COMPATIBLE", two_to_one

    mobitz_ii = analyze_av_conduction(
        av_lead([100, 400, 700, 1000, 1300, 1600], [200, 500, 800, 1400, 1700]),
        atrial_support,
    )
    assert mobitz_ii["classification"] == "MOBITZ_II_COMPATIBLE", mobitz_ii

    high_grade = analyze_av_conduction(
        av_lead([100, 400, 700, 1000, 1300, 1600], [200, 500, 1400, 1700]),
        atrial_support,
    )
    assert high_grade["classification"] == "HIGH_GRADE_AV_BLOCK_COMPATIBLE", high_grade

    complete = analyze_av_conduction(
        av_lead(
            [100, 400, 700, 1000, 1300, 1600, 1900, 2200, 2500],
            [550, 1250, 1950, 2650],
        ),
        atrial_support,
    )
    assert complete["classification"] == "COMPLETE_AV_BLOCK_COMPATIBLE", complete
    assert complete["av_dissociation_phase"], complete

    # Negative AV controls: a normal 1:1 organized P/QRS sequence must remain
    # free of second-/third-degree AV-block labels even when P and QRS counts
    # are both high. These deterministic fixtures protect future atrial rescue
    # work from repeating the synthetic-control failure caught by the staged gate.
    normal_one_to_one = analyze_av_conduction(
        av_lead(
            [100, 400, 700, 1000, 1300, 1600],
            [180, 480, 780, 1080, 1380, 1680],
        ),
        atrial_support,
    )
    assert normal_one_to_one["classification"] == "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED", normal_one_to_one
    assert normal_one_to_one["nonconducted_p_n"] == 0, normal_one_to_one
    assert normal_one_to_one["one_to_one"], normal_one_to_one

    # Regular ventricular timing alone is not AV block evidence. With no
    # independently observed P sequence the specialist must remain unevaluable.
    no_atrial_sequence = analyze_av_conduction(
        av_lead([], [180, 480, 780, 1080, 1380, 1680]),
        {"p_wave_reproducible": False},
    )
    assert no_atrial_sequence["classification"] == "AV_CONDUCTION_NOT_EVALUABLE", no_atrial_sequence

    # Guarded cross-lead atrial recovery may support AV mapping only when
    # >=2 recovered events complete an organized train already anchored by
    # >=3 observed consensus P events. Raw P fiducials remain untouched.
    recovered_two_to_one = analyze_av_conduction(
        av_lead([100, 600, 1100], [180, 680, 1180]),
        atrial_support,
        independent_atrial_evidence={
            "organized_augmented": True,
            "recovered_event_n": 3,
            "recovered_events": [
                {"time_ms": 700.0, "support_lead_n": 2, "spread_ms": 8.0},
                {"time_ms": 1700.0, "support_lead_n": 2, "spread_ms": 6.0},
                {"time_ms": 2700.0, "support_lead_n": 2, "spread_ms": 10.0},
            ],
            "observed_consensus": {"event_n": 3},
            "combined_event_times_ms": [200.0, 700.0, 1200.0, 1700.0, 2200.0, 2700.0],
        },
    )
    assert recovered_two_to_one["classification"] == "TWO_TO_ONE_AV_BLOCK_COMPATIBLE", recovered_two_to_one
    assert recovered_two_to_one["independent_atrial_evidence_used"], recovered_two_to_one
    assert recovered_two_to_one["raw_p_count"] == 3, recovered_two_to_one
    assert recovered_two_to_one["p_count"] == 6, recovered_two_to_one
    assert "MULTILEAD_RECOVERED_ATRIAL_EVIDENCE" in recovered_two_to_one["basis"], recovered_two_to_one

    # One recovered event is insufficient even if a caller supplies a regular
    # combined sequence. The specialist must not promote it into AV-block data.
    insufficient_recovery = analyze_av_conduction(
        av_lead([], [180, 680, 1180]),
        {"p_wave_reproducible": False},
        independent_atrial_evidence={
            "organized_augmented": True,
            "recovered_event_n": 1,
            "recovered_events": [
                {"time_ms": 700.0, "support_lead_n": 2, "spread_ms": 5.0},
            ],
            "observed_consensus": {"event_n": 3},
            "combined_event_times_ms": [200.0, 700.0, 1200.0, 1700.0, 2200.0],
        },
    )
    assert insufficient_recovery["classification"] == "AV_CONDUCTION_NOT_EVALUABLE", insufficient_recovery
    assert not insufficient_recovery["independent_atrial_evidence_used"], insufficient_recovery

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
