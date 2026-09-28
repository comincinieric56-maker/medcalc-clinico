from __future__ import annotations

from ecg_av_conduction import analyze_av_conduction
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_domain_gating import build_domain_gates
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


def main() -> None:
    test_lpfb_requires_axis_morphology_and_narrow_qrs()
    test_lpfb_crosslead_and_reasoner_propagation()
    test_multilead_prewave_rescues_only_preexcitation_domain()
    test_single_lead_short_pr_delta_does_not_rescue()
    test_av_block_search_uses_nontraditional_p_rich_lead()
    test_av_lead_quality_beats_static_priority()
    print("MEDCALC_ADULT_DIAGNOSTIC_V2_HARDENING_SELFTEST_PASS")


if __name__ == "__main__":
    main()
