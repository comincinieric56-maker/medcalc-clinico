from __future__ import annotations

from ecg_av_conduction import analyze_av_conduction
from ecg_candidate_detectors import _av_sequence_candidates, build_high_recall_candidates
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_consistency_engine import evaluate_ecg_consistency
from ecg_domain_gating import build_domain_gates
from ecg_external_engine_adapter import normalize_external_engine_result
from ecg_measurement_service import build_measurement_service
from ecg_preexcitation import analyze_preexcitation
from ecg_reasoner import reason_ecg
from ecg_signal_measurements import _fascicular_conduction_pattern


def _metric(value):
    return {"value": value, "confidence": 0.95, "status": "MEASURED"}


def _lead(*, area=None, r=None, s=None, q=None, qdur=None):
    return {
        "evaluable": True,
        "metrics": {
            "qrs_net_area_mv_ms": _metric(area),
            "r_amp_mv": _metric(r),
            "s_amp_mv": _metric(s),
            "q_amp_mv": _metric(q),
            "q_duration_ms": _metric(qdur),
        },
    }


def test_lpfb_requires_axis_morphology_and_narrow_qrs() -> None:
    per_lead = {
        "I": _lead(area=-10.0, r=0.10, s=-0.55, q=None, qdur=None),
        "aVL": _lead(area=-8.0, r=0.08, s=-0.48, q=None, qdur=None),
        "II": _lead(area=5.0, r=0.30, s=-0.10, q=-0.02, qdur=20.0),
        "III": _lead(area=12.0, r=0.60, s=-0.10, q=-0.05, qdur=24.0),
        "aVF": _lead(area=11.0, r=0.55, s=-0.10, q=-0.04, qdur=22.0),
    }
    global_metrics = {"qrs_ms": _metric(105.0)}

    positive = _fascicular_conduction_pattern(
        per_lead, {"degrees": 120.0}, global_metrics
    )
    assert positive["classification"] == "LPFB_COMPATIBLE", positive
    assert positive["criteria"]["axis_plus90_to_plus180"] is True, positive
    assert positive["criteria"]["superior_s_dominant_n"] == 2, positive
    assert positive["criteria"]["inferior_r_dominant_n"] == 2, positive

    wrong_axis = _fascicular_conduction_pattern(
        per_lead, {"degrees": 15.0}, global_metrics
    )
    assert wrong_axis["classification"] != "LPFB_COMPATIBLE", wrong_axis

    wide = _fascicular_conduction_pattern(
        per_lead, {"degrees": 120.0}, {"qrs_ms": _metric(130.0)}
    )
    assert wide["classification"] != "LPFB_COMPATIBLE", wide


def test_lpfb_crosslead_and_reasoner_propagation() -> None:
    graph = {
        "global": {"qrs_ms": _metric(105.0)},
        "specialist_evidence": {
            "qrs_morphology": {"per_lead": {}},
            "fascicular_conduction": {
                "classification": "LPFB_COMPATIBLE",
                "confidence": 0.92,
            },
        },
        "rhythm": {},
        "relations": {},
    }
    cross = analyze_crosslead_conduction(graph)
    assert any(
        row.get("code") == "LPFB_COMPATIBLE"
        for row in cross.get("findings") or []
    ), cross

    gates = {
        "domains": {
            "RHYTHM": {"eligible": False},
            "FASCICULAR": {"eligible": True},
        }
    }
    fused_lpfb = {
        "code": "LPFB_COMPATIBLE",
        "domain": "FASCICULAR",
        "publishable": True,
        "score": 0.92,
        "evidence": ["RIGHT_AXIS", "SUPERIOR_rS_PATTERN", "INFERIOR_qR_PATTERN"],
        "fusion_state": "ESTABLISHED_COMPATIBLE",
    }
    fusion = {
        "findings": [fused_lpfb],
        "by_code": {"LPFB_COMPATIBLE": fused_lpfb},
        "publishable_findings": [fused_lpfb],
    }
    reasoned = reason_ecg(
        graph,
        cross,
        {"blocking_conflict": False},
        domain_gates=gates,
        evidence_fusion=fusion,
    )
    findings = (reasoned.get("diagnostic_summary") or {}).get("findings") or []
    row = next(x for x in findings if x.get("code") == "LPFB_COMPATIBLE")
    assert row["domain"] == "FASCICULAR", row
    assert row["publishable"] is True, row


def test_multilead_prewave_rescues_only_preexcitation_domain() -> None:
    graph = {
        "global": {
            "pr_ms": _metric(None),
            "qrs_ms": _metric(None),
        },
        "relations": {"p_reproducible": False},
        "leads": {
            "I": {"evaluable": True, "confidence": 0.90, "pr_ms": 104.0},
            "II": {"evaluable": True, "confidence": 0.88, "pr_ms": 108.0},
            "V2": {"evaluable": True, "confidence": 0.90, "pr_ms": 150.0},
        },
    }
    morph = {
        "per_lead": {
            "I": {
                "evaluable": True,
                "duration_ms": 118.0,
                "delta_slur_compatible": True,
            },
            "II": {
                "evaluable": True,
                "duration_ms": 122.0,
                "delta_slur_compatible": True,
            },
            "V2": {
                "evaluable": True,
                "duration_ms": 110.0,
                "delta_slur_compatible": False,
            },
        }
    }
    pre = analyze_preexcitation(graph, morph)
    assert pre["classification"] == "VENTRICULAR_PREEXCITATION_COMPATIBLE", pre
    assert pre["criteria"]["multilead_short_pr_delta_rescue"] is True, pre
    assert pre["criteria"]["concordant_short_pr_delta_leads"] == ["I", "II"], pre

    graph["specialist_evidence"] = {"preexcitation": pre}
    consistency = {
        "conflicts": [],
        "remeasure_targets": ["pr_ms", "qrs_ms"],
        "measurement_unusable_targets": ["pr_ms", "qrs_ms"],
        "measurement_uncertain_targets": [],
        "blocking_conflict": False,
        "remeasure_required": True,
    }
    gates = build_domain_gates(graph, {}, consistency)
    assert gates["domains"]["PREEXCITATION"]["eligible"] is True, gates
    assert gates["preexcitation_multilead_rescue_active"] is True, gates
    # QRS remains unusable for bundle-branch diagnosis. Rescue is diagnosis-specific.
    assert gates["domains"]["BUNDLE_BRANCH"]["eligible"] is False, gates


def test_single_lead_short_pr_delta_does_not_rescue() -> None:
    graph = {
        "global": {"pr_ms": _metric(None), "qrs_ms": _metric(None)},
        "relations": {"p_reproducible": False},
        "leads": {
            "I": {"evaluable": True, "confidence": 0.90, "pr_ms": 105.0},
            "II": {"evaluable": True, "confidence": 0.90, "pr_ms": 145.0},
        },
    }
    morph = {
        "per_lead": {
            "I": {"evaluable": True, "duration_ms": 120.0, "delta_slur_compatible": True},
            "II": {"evaluable": True, "duration_ms": 120.0, "delta_slur_compatible": True},
        }
    }
    pre = analyze_preexcitation(graph, morph)
    assert pre["classification"] != "VENTRICULAR_PREEXCITATION_COMPATIBLE", pre
    assert pre["criteria"]["multilead_short_pr_delta_rescue"] is False, pre


def test_av_block_search_uses_nontraditional_p_rich_lead() -> None:
    per_lead = {
        "II": {"evaluable": False},
        "V1": {"evaluable": False},
        "aVF": {"evaluable": False},
        "I": {"evaluable": False},
        "V5": {
            "evaluable": True,
            "fs": 500,
            "raw_p_peaks_samples": [100, 600, 1100, 1600],
            "r_peaks_samples": [250, 750, 1250, 1750],
            "atrial_activity": {
                "p_candidate_n": 4,
                "p_wave_reproducible": True,
            },
        },
    }
    av = analyze_av_conduction(
        per_lead,
        {"p_wave_reproducible": True, "rhythm_p_qrs_coupling_fraction": 1.0},
        global_metrics={"pr_ms": _metric(300.0)},
    )
    assert av["lead"] == "V5", av
    assert av["classification"] == "FIRST_DEGREE_AV_DELAY_COMPATIBLE", av
    assert av["one_to_one"] is True, av


def test_av_lead_quality_beats_static_priority() -> None:
    # Lead II is technically evaluable but has a disorganized/nonreproducible
    # P sequence. V5 has a clean organized atrial sequence and must be chosen.
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": 500,
            "confidence": 0.70,
            "raw_p_peaks_samples": [100, 310, 900, 1600],
            "r_peaks_samples": [250, 750, 1250, 1750],
            "atrial_activity": {
                "p_candidate_n": 4,
                "p_wave_reproducible": False,
                "p_qrs_coupling_fraction": 0.25,
            },
        },
        "V5": {
            "evaluable": True,
            "fs": 500,
            "confidence": 0.90,
            "raw_p_peaks_samples": [100, 600, 1100, 1600],
            "r_peaks_samples": [250, 750, 1250, 1750],
            "atrial_activity": {
                "p_candidate_n": 4,
                "p_wave_reproducible": True,
                "p_qrs_coupling_fraction": 1.0,
            },
        },
    }
    av = analyze_av_conduction(
        per_lead,
        {"p_wave_reproducible": True, "rhythm_p_qrs_coupling_fraction": 1.0},
        global_metrics={"pr_ms": _metric(300.0)},
    )
    assert av["lead"] == "V5", av
    assert av["classification"] == "FIRST_DEGREE_AV_DELAY_COMPATIBLE", av



def test_fast_two_to_one_uses_nearest_preceding_p() -> None:
    # At PP=300 ms, a blocked P remains within the broad 500-ms PR search
    # window of the next QRS. P->next-QRS matching therefore steals the QRS
    # from the actually conducted P. QRS->nearest-preceding-P must preserve the
    # alternating 2:1 pattern.
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": 500,
            "confidence": 0.95,
            "raw_p_peaks_samples": [50, 200, 350, 500, 650, 800],
            "r_peaks_samples": [100, 400, 700],
            "atrial_activity": {
                "p_candidate_n": 6,
                "p_wave_reproducible": True,
                "p_qrs_coupling_fraction": 0.50,
            },
        },
    }
    av = analyze_av_conduction(
        per_lead,
        {"p_wave_reproducible": True, "rhythm_p_qrs_coupling_fraction": 0.50},
        global_metrics={"pr_ms": _metric(None)},
    )
    assert av["classification"] == "TWO_TO_ONE_AV_BLOCK_COMPATIBLE", av
    assert av["conducted_p_n"] == 3, av
    assert av["nonconducted_p_n"] == 3, av
    assert av["p_qrs_coupling_fraction"] == 0.5, av



def test_candidate_layer_fast_two_to_one_uses_nearest_preceding_p() -> None:
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": 500,
            "confidence": 0.95,
            "raw_p_peaks_samples": [50, 200, 350, 500, 650, 800],
            "r_peaks_samples": [100, 400, 700],
            "atrial_activity": {
                "p_candidate_n": 6,
                "p_wave_reproducible": True,
                "p_qrs_coupling_fraction": 0.50,
            },
        },
    }
    rows = _av_sequence_candidates(per_lead)
    codes = {str(row.get("code") or "") for row in rows}
    assert "TWO_TO_ONE_AV_BLOCK_COMPATIBLE" in codes, rows
    row = next(
        row for row in rows
        if row.get("code") == "TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
    )
    audit = row.get("sequence_audit") or {}
    assert audit.get("conducted_p_n") == 3, row
    assert audit.get("dropped_p_n") == 3, row


def test_multilead_qrs_rescue_requires_strict_wide_consensus() -> None:
    graph = {
        "global": {"qrs_ms": {"value": 116.0, "confidence": 0.30, "status": "REMEASURE"}},
        "specialist_evidence": {
            "measurement_consensus": {
                "remeasure_targets": ["qrs_ms"],
                "unusable_targets": ["qrs_ms"],
                "unmeasurable_targets": [],
                "uncertain_targets": ["qrs_ms"],
            },
            "qrs_morphology": {
                "per_lead": {
                    "V1": {
                        "evaluable": True, "duration_ms": 130.0,
                        "r_prime_present": True, "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.15, "terminal_negative_mv": -0.02,
                    },
                    "V2": {
                        "evaluable": True, "duration_ms": 128.0,
                        "r_prime_present": True, "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.12, "terminal_negative_mv": -0.02,
                    },
                    "I": {
                        "evaluable": True, "duration_ms": 126.0,
                        "qrs_polarity": "BIPHASIC", "terminal_negative_mv": -0.12,
                        "terminal_s_duration_ms": 40.0, "terminal_positive_mv": 0.04,
                    },
                    "V6": {
                        "evaluable": True, "duration_ms": 124.0,
                        "qrs_polarity": "BIPHASIC", "terminal_negative_mv": -0.11,
                        "terminal_s_duration_ms": 38.0, "terminal_positive_mv": 0.04,
                    },
                }
            },
            "fascicular_conduction": {},
        },
        "rhythm": {},
        "relations": {},
    }
    cross = analyze_crosslead_conduction(graph)
    criteria = cross.get("criteria") or {}
    assert criteria.get("multilead_qrs_ge_120_rescue") is True, cross
    assert criteria.get("wide_qrs_lead_n") == 4, cross
    assert any(
        row.get("code") == "RBBB_MORPHOLOGY_COMPATIBLE"
        for row in cross.get("findings") or []
    ), cross

    consistency = evaluate_ecg_consistency(graph, cross)
    blocking = {
        str(row.get("code") or "")
        for row in consistency.get("conflicts") or []
        if str(row.get("severity") or "") == "BLOCKING"
    }
    assert "COMPLETE_BBB_WITH_QRS_LT_120_CONFLICT" not in blocking, consistency
    assert "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT" not in blocking, consistency

    gates = build_domain_gates(graph, cross, consistency)
    assert gates["domains"]["BUNDLE_BRANCH"]["eligible"] is True, gates
    assert gates["bundle_branch_multilead_qrs_rescue_active"] is True, gates

    candidates = build_high_recall_candidates(graph, cross, {})
    rbbb = (candidates.get("by_code") or {}).get("RBBB_MORPHOLOGY_COMPATIBLE") or {}
    assert rbbb, candidates
    assert rbbb.get("required_measurements") == [], rbbb
    assert rbbb.get("boundary_requirements") == [], rbbb
    assert "GE_4_MULTILEAD_QRS_GE_120MS" in (rbbb.get("evidence") or []), rbbb

    # Three wide leads are insufficient: the rescue must stay off.
    graph["specialist_evidence"]["qrs_morphology"]["per_lead"]["V6"]["duration_ms"] = 118.0
    cross_3 = analyze_crosslead_conduction(graph)
    assert (cross_3.get("criteria") or {}).get("multilead_qrs_ge_120_rescue") is False, cross_3


def test_external_engine_adapter_is_advisory_only() -> None:
    normalized = normalize_external_engine_result(
        {
            "findings": [
                {
                    "code": "VENDOR_AF",
                    "label": "Atrial fibrillation",
                    "confidence": 0.91,
                    "evidence": ["IRREGULAR_RR", "NO_P"],
                    "publishable": True,
                },
                {
                    "code": "BAD_CONFIDENCE",
                    "confidence": 91.0,
                },
            ],
            "measurements": {
                "heart_rate_bpm": {
                    "value": 72.0,
                    "unit": "bpm",
                    "confidence": 0.94,
                },
                "qrs_ms": {
                    "value": 118.0,
                    "unit": "ms",
                    "confidence": 0.88,
                },
            },
        },
        engine_id="TEST_VENDOR",
        engine_version="1.2.3",
        input_kind="CANONICAL_12_LEAD_DIGITAL",
        code_map={"VENDOR_AF": "AF_COMPATIBLE"},
        source_digest="sha256:test",
    )
    findings = normalized["findings"]
    assert len(findings) == 1, normalized
    af = findings[0]
    assert af["vendor_code"] == "VENDOR_AF", af
    assert af["canonical_code"] == "AF_COMPATIBLE", af
    assert af["publishable"] is False, af
    assert af["fusion_eligible"] is False, af
    assert af["advisory_only"] is True, af
    assert normalized["policy"]["direct_publication_allowed"] is False, normalized
    assert normalized["policy"]["fusion_allowed"] is False, normalized
    assert normalized["policy"]["measurement_override_allowed"] is False, normalized
    assert normalized["measurements"]["qrs_ms"]["measurement_override_allowed"] is False, normalized
    assert normalized["measurements"]["qrs_ms"]["usable_for_medcalc_measurement_consensus"] is False, normalized
    rejected = normalized["rejected_items"]
    assert any(
        row.get("reason") == "INVALID_CONFIDENCE_SCALE_EXPECTED_0_TO_1"
        for row in rejected
    ), rejected


def test_measurement_service_preserves_canonical_values_and_provenance() -> None:
    global_metrics = {
        "qrs_ms": {
            "value": 126.0,
            "unit": "ms",
            "confidence": 0.91,
            "status": "MEASURED",
            "reason": None,
        },
        "pr_ms": {
            "value": 184.0,
            "unit": "ms",
            "confidence": 0.82,
            "status": "MEASURED",
            "reason": None,
        },
    }
    per_lead = {
        "I": {
            "metrics": {
                "qrs_ms": {"value": 124.0, "confidence": 0.88, "beat_n": 7},
            }
        },
        "V2": {
            "metrics": {
                "qrs_ms": {"value": 128.0, "confidence": 0.90, "beat_n": 6},
            }
        },
    }
    consensus = {
        "metrics": {
            "qrs_ms": {
                "measurement_state": "MEASURED_WITH_UNCERTAINTY",
                "source_leads": ["I", "V2"],
                "candidate_values": {"I": 124.0, "V2": 128.0},
                "candidate_confidences": {"I": 0.88, "V2": 0.90},
                "candidate_median": 126.0,
                "candidate_mad": 2.0,
                "candidate_iqr": None,
                "canonical_vs_median_abs_diff": 0.0,
                "uncertainty_ms": 4.0,
                "uncertainty_interval": [122.0, 130.0],
                "uncertainty_sources": ["RECONSTRUCTED_SIGNAL_SAMPLING_FLOOR"],
                "remeasure": False,
                "unusable": False,
                "usable_with_uncertainty": True,
            },
            "pr_ms": {
                "measurement_state": "MEASURED_HIGH_CONFIDENCE",
                "source_leads": [],
                "candidate_values": {},
                "candidate_confidences": {},
                "candidate_median": None,
                "candidate_mad": 0.0,
                "candidate_iqr": None,
                "canonical_vs_median_abs_diff": None,
                "uncertainty_ms": 4.0,
                "uncertainty_interval": [180.0, 188.0],
                "uncertainty_sources": ["RECONSTRUCTED_SIGNAL_SAMPLING_FLOOR"],
                "remeasure": False,
                "unusable": False,
                "usable_with_uncertainty": False,
            },
        },
        "overall_measurement_quality": 0.86,
        "remeasure_required": False,
        "remeasure_targets": [],
        "unmeasurable_targets": [],
        "uncertain_targets": ["qrs_ms"],
    }
    service = build_measurement_service(
        global_metrics,
        per_lead,
        consensus,
    )
    qrs = service["metrics"]["qrs_ms"]
    assert qrs["value"] == 126.0, qrs
    assert qrs["confidence"] == 0.91, qrs
    assert qrs["measurement_state"] == "MEASURED_WITH_UNCERTAINTY", qrs
    assert qrs["provenance"]["source_leads"] == ["I", "V2"], qrs
    assert qrs["provenance"]["lead_beat_n"] == {"I": 7, "V2": 6}, qrs
    assert qrs["provenance"]["beat_n_total"] == 13, qrs
    assert qrs["uncertainty"]["interval"] == [122.0, 130.0], qrs
    assert qrs["crosslead_dispersion"]["mad"] == 2.0, qrs
    assert "override" not in qrs, qrs


def test_preexcitation_warning_preserves_independently_fused_bbb() -> None:
    graph = {
        "global": {},
        "specialist_evidence": {
            "atrial_activity": {},
            "atrial_mechanism": {},
            "wide_complex_tachycardia": {},
            "ectopy": {},
        },
        "rhythm": {},
        "relations": {},
    }
    rbbb = {
        "code": "RBBB_MORPHOLOGY_COMPATIBLE",
        "domain": "BUNDLE_BRANCH",
        "publishable": True,
        "score": 0.88,
        "evidence": [
            "QRS_GE_120MS",
            "V1_R_PRIME_OR_TERMINAL_POSITIVE",
            "LATERAL_TERMINAL_S",
        ],
        "fusion_state": "ESTABLISHED_COMPATIBLE",
    }
    pre = {
        "code": "VENTRICULAR_PREEXCITATION_COMPATIBLE",
        "domain": "PREEXCITATION",
        "publishable": True,
        "score": 0.82,
        "evidence": ["SHORT_PR", "MULTILEAD_DELTA_SLUR", "QRS_GE_110MS"],
        "fusion_state": "ESTABLISHED_COMPATIBLE",
    }
    fusion = {
        "findings": [rbbb, pre],
        "by_code": {
            "RBBB_MORPHOLOGY_COMPATIBLE": rbbb,
            "VENTRICULAR_PREEXCITATION_COMPATIBLE": pre,
        },
        "publishable_findings": [rbbb, pre],
    }
    gates = {
        "domains": {
            "RHYTHM": {"eligible": False},
            "BUNDLE_BRANCH": {"eligible": True},
            "PREEXCITATION": {"eligible": True},
            "ECTOPY": {"eligible": False},
        }
    }
    consistency = {
        "status": "PASS_WITH_WARNINGS",
        "blocking_conflict": False,
        "conflicts": [{
            "code": "PREEXCITATION_CONFOUNDS_BUNDLE_BRANCH_PATTERN",
            "severity": "WARNING",
            "action": "REPORT_COEXISTING_PATTERNS_WITH_CONFOUNDING_REVIEW",
        }],
    }
    reasoned = reason_ecg(
        graph,
        {"findings": []},
        consistency,
        domain_gates=gates,
        evidence_fusion=fusion,
    )
    findings = (reasoned.get("diagnostic_summary") or {}).get("findings") or []
    codes = {str(row.get("code") or "") for row in findings}
    assert "RBBB_MORPHOLOGY_COMPATIBLE" in codes, findings
    assert "VENTRICULAR_PREEXCITATION_COMPATIBLE" in codes, findings
    bbb = next(
        row for row in findings
        if row.get("code") == "RBBB_MORPHOLOGY_COMPATIBLE"
    )
    assert bbb["publishable"] is True, bbb
    assert bbb["confounded_by_preexcitation"] is True, bbb
    assert "PREEXCITATION_CONFOUNDING_WARNING" in (bbb.get("basis") or []), bbb


def main() -> None:
    test_preexcitation_warning_preserves_independently_fused_bbb()
    test_multilead_qrs_rescue_requires_strict_wide_consensus()
    test_external_engine_adapter_is_advisory_only()
    test_measurement_service_preserves_canonical_values_and_provenance()
    test_lpfb_requires_axis_morphology_and_narrow_qrs()
    test_lpfb_crosslead_and_reasoner_propagation()
    test_multilead_prewave_rescues_only_preexcitation_domain()
    test_single_lead_short_pr_delta_does_not_rescue()
    test_av_block_search_uses_nontraditional_p_rich_lead()
    test_av_lead_quality_beats_static_priority()
    test_fast_two_to_one_uses_nearest_preceding_p()
    test_candidate_layer_fast_two_to_one_uses_nearest_preceding_p()
    print("MEDCALC_ADULT_DIAGNOSTIC_V2_HARDENING_SELFTEST_PASS")


if __name__ == "__main__":
    main()
