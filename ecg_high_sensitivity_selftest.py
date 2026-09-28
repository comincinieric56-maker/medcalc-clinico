from __future__ import annotations

from ecg_candidate_detectors import build_high_recall_candidates
from ecg_domain_gating import build_domain_gates
from ecg_evidence_fusion import fuse_candidate_evidence
from ecg_fn_waterfall import classify_false_negative
from ecg_high_sensitivity_benchmark import score_labeled_development_cases
from ecg_reasoner import reason_ecg


def _base_graph() -> dict:
    return {
        "global": {
            "heart_rate_bpm": {"value": 78.0, "confidence": 0.9},
            "qrs_ms": {"value": 132.0, "confidence": 0.9},
            "pr_ms": {"value": 170.0, "confidence": 0.8},
        },
        "rhythm": {
            "evaluable": True,
            "regular": False,
            "confidence": 0.9,
            "rr_irregularity_score": 0.80,
            "rate_consensus": {
                "evaluable": True,
                "heart_rate_bpm": 78.0,
                "confidence": 0.85,
                "source_n": 6,
            },
        },
        "relations": {
            "p_reproducible": False,
            "p_qrs_coupling_fraction": 0.0,
        },
        "specialist_evidence": {
            "atrial_activity": {
                "p_wave_reproducible": False,
                "sinus_compatible": False,
                "rhythm_p_qrs_coupling_fraction": 0.0,
            },
            "atrial_mechanism": {
                "mechanism": "ATRIAL_TACHYARRHYTHMIA_UNDETERMINED",
                "confidence": 0.50,
                "compatibility_scores_not_probabilities": {
                    "AF": 0.62,
                    "FLUTTER_OR_AT": 0.20,
                    "OTHER_SVT": 0.20,
                },
                "aggregate_features": {
                    "rr_irregularity_score": 0.80,
                    "autocorrelation_periodicity": 0.20,
                    "dominant_frequency_hz": 8.0,
                },
            },
            "wide_complex_tachycardia": {"wide_complex_tachycardia": False},
            "fascicular_conduction": {
                "classification": "NO_FASCICULAR_PATTERN_ESTABLISHED",
                "criteria": {},
            },
            "measurement_consensus": {
                "remeasure_required": False,
                "remeasure_targets": [],
                "unmeasurable_targets": [],
                "unusable_targets": [],
                "uncertain_targets": [],
                "metrics": {
                    "qrs_ms": {
                        "measurement_state": "MEASURED_HIGH_CONFIDENCE",
                        "canonical_value": 132.0,
                        "uncertainty_interval": [128.0, 136.0],
                    },
                    "pr_ms": {
                        "measurement_state": "MEASURED_HIGH_CONFIDENCE",
                        "canonical_value": 170.0,
                        "uncertainty_interval": [164.0, 176.0],
                    },
                },
            },
            "signal_integrity": {"overall_quality": 0.9, "per_lead": {}},
            "ectopy": {},
            "qrs_morphology": {"per_lead": {}},
            "av_conduction": {
                "evaluable": False,
                "classification": "AV_CONDUCTION_NOT_EVALUABLE",
            },
            "preexcitation": {
                "classification": "NO_PREEXCITATION_PATTERN_ESTABLISHED",
                "criteria": {},
            },
        },
    }


def main() -> None:
    # A QRS-only problem must not globally suppress a strong AF candidate.
    graph = _base_graph()
    consistency = {
        "conflicts": [{
            "code": "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT",
            "severity": "BLOCKING",
        }],
        "blocking_conflict": True,
        "remeasure_required": True,
        "remeasure_targets": ["qrs_ms"],
    }
    gates = build_domain_gates(graph, {"findings": []}, consistency)
    assert gates["domains"]["RHYTHM"]["eligible"], gates
    assert not gates["domains"]["BUNDLE_BRANCH"]["eligible"], gates

    candidates = build_high_recall_candidates(graph, {"criteria": {}, "findings": []}, {})
    af = candidates["by_code"]["AF_COMPATIBLE"]
    assert af["independent_evidence_n"] >= 3, af
    fusion = fuse_candidate_evidence(candidates, gates)
    assert fusion["by_code"]["AF_COMPATIBLE"]["publishable"], fusion

    reasoned = reason_ecg(
        graph,
        {"criteria": {}, "findings": []},
        consistency,
        domain_gates=gates,
        evidence_fusion=fusion,
    )
    assert reasoned["primary_rhythm"]["code"] == "AF_COMPATIBLE", reasoned
    assert reasoned["primary_rhythm"]["publish_as_established"], reasoned
    assert reasoned["publication_model"] == "DOMAIN_SPECIFIC", reasoned

    # High-recall RBBB candidate: QRS width plus right-precordial morphology
    # reaches candidate/fusion stage without requiring a third serial hard gate.
    rbbb_cross = {
        "criteria": {
            "qrs_ge_120ms": True,
            "rbbb_right_terminal_r": True,
            "rbbb_lateral_terminal_s_leads": [],
            "lbbb_v1_v2_negative": False,
            "lbbb_key_lateral_r": False,
            "lbbb_key_lateral_absent_q": False,
            "lbbb_delayed_or_notched_lateral": False,
        },
        "findings": [],
    }
    clean = {
        "conflicts": [],
        "blocking_conflict": False,
        "remeasure_required": False,
        "remeasure_targets": [],
    }
    clean_gates = build_domain_gates(graph, rbbb_cross, clean)
    rbbb_candidates = build_high_recall_candidates(graph, rbbb_cross, {})
    assert rbbb_candidates["by_code"]["RBBB_MORPHOLOGY_COMPATIBLE"]["score"] >= 0.65
    rbbb_fusion = fuse_candidate_evidence(rbbb_candidates, clean_gates)
    assert rbbb_fusion["by_code"]["RBBB_MORPHOLOGY_COMPATIBLE"]["publishable"], rbbb_fusion

    # A QRS interval that overlaps 120 ms must not publish complete BBB.
    borderline_graph = _base_graph()
    borderline_graph["global"]["qrs_ms"] = {"value": 119.0, "confidence": 0.85}
    borderline_graph["specialist_evidence"]["measurement_consensus"]["metrics"]["qrs_ms"] = {
        "measurement_state": "MEASURED_WITH_UNCERTAINTY",
        "canonical_value": 119.0,
        "uncertainty_interval": [113.0, 125.0],
    }
    borderline_cross = dict(rbbb_cross)
    borderline_cross["criteria"] = dict(rbbb_cross["criteria"])
    borderline_cross["criteria"]["qrs_ge_120ms"] = False
    borderline_candidates = build_high_recall_candidates(
        borderline_graph, borderline_cross, {}
    )
    borderline_gates = build_domain_gates(borderline_graph, borderline_cross, clean)
    borderline_fusion = fuse_candidate_evidence(borderline_candidates, borderline_gates)
    assert not borderline_fusion["by_code"]["RBBB_MORPHOLOGY_COMPATIBLE"]["publishable"], borderline_fusion
    assert borderline_fusion["by_code"]["RBBB_MORPHOLOGY_COMPATIBLE"]["fusion_state"] == "MEASUREMENT_BOUNDARY_UNCERTAIN", borderline_fusion

    # Multi-lead AV rescue: a clean 2:1 P:QRS sequence must create an AV-block
    # candidate even when the legacy single-lead specialist did not classify it.
    av_graph = _base_graph()
    av_graph["global"]["qrs_ms"] = {"value": 92.0, "confidence": 0.9}
    av_graph["specialist_evidence"]["measurement_consensus"]["metrics"]["qrs_ms"] = {
        "measurement_state": "MEASURED_HIGH_CONFIDENCE",
        "canonical_value": 92.0,
        "uncertainty_interval": [88.0, 96.0],
    }
    av_graph["specialist_evidence"]["atrial_activity"] = {
        "p_wave_reproducible": True,
        "sinus_compatible": False,
        "rhythm_p_qrs_coupling_fraction": 0.5,
    }
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": 500,
            "raw_p_peaks_samples": [100, 350, 600, 850, 1100, 1350],
            "r_peaks_samples": [175, 675, 1175],
        }
    }
    av_candidates = build_high_recall_candidates(
        av_graph,
        {"criteria": {}, "findings": []},
        per_lead,
    )
    assert "TWO_TO_ONE_AV_BLOCK_COMPATIBLE" in av_candidates["by_code"], av_candidates
    av_gates = build_domain_gates(av_graph, {"findings": []}, clean)
    av_fusion = fuse_candidate_evidence(av_candidates, av_gates)
    assert av_fusion["by_code"]["TWO_TO_ONE_AV_BLOCK_COMPATIBLE"]["publishable"], av_fusion

    analysis = {
        "signal_integrity": {"overall_quality": 0.9},
        "measurement_consensus": {"remeasure_targets": []},
        "high_recall_candidates": candidates,
        "domain_gates": gates,
        "evidence_fusion": fusion,
        "specialist_reasoning": reasoned,
    }
    waterfall = classify_false_negative("AF_COMPATIBLE", analysis)
    assert waterfall["stage"] == "TRUE_POSITIVE", waterfall

    bench = score_labeled_development_cases([
        {"expected_code": "AF_COMPATIBLE", "analysis": analysis},
    ])
    af_bench = bench["diagnoses"]["AF_COMPATIBLE"]
    assert af_bench["candidate_sensitivity"] == 1.0, af_bench
    assert af_bench["final_sensitivity"] == 1.0, af_bench

    print("MEDCALC_ECG_V3_HIGH_SENSITIVITY_SELFTEST_PASS")


if __name__ == "__main__":
    main()
