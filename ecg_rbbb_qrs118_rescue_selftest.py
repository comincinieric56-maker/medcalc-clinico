from __future__ import annotations

from ecg_candidate_detectors import build_high_recall_candidates
from ecg_evidence_fusion import fuse_candidate_evidence

CODE = "RBBB_MORPHOLOGY_COMPATIBLE"
MARKER = "RBBB_MULTILEAD_QRS_GE3_120_GE4_118_DISTRIBUTED"


def _feature_graph() -> dict:
    return {
        "global": {
            "qrs_ms": {"value": 119.0, "confidence": 0.70},
        },
        "rhythm": {},
        "specialist_evidence": {
            "atrial_activity": {},
            "atrial_mechanism": {},
            "fascicular_conduction": {},
            "preexcitation": {},
            "measurement_consensus": {
                "metrics": {
                    "qrs_ms": {
                        "measurement_state": "REMEASURE_REQUIRED",
                        "canonical_value": 119.0,
                        "uncertainty_interval": [116.0, 122.0],
                    },
                    "pr_ms": {
                        "measurement_state": "REMEASURE_REQUIRED",
                    },
                },
            },
            "qrs_morphology": {
                "per_lead": {
                    "I": {"evaluable": True, "duration_ms": 122.0},
                    "V1": {"evaluable": True, "duration_ms": 121.0},
                    "V2": {"evaluable": True, "duration_ms": 120.0},
                    "V6": {"evaluable": True, "duration_ms": 118.0},
                },
            },
        },
    }


def _crosslead() -> dict:
    return {
        "criteria": {
            "multilead_qrs_ge_120_rescue": False,
            "wide_qrs_lead_n": 3,
            "wide_qrs_limb_lead_n": 1,
            "wide_qrs_precordial_lead_n": 2,
            "rbbb_right_terminal_r": True,
            "rbbb_lateral_terminal_s_leads": ["I"],
            "rbbb_morphology": True,
        },
        "findings": [],
    }


def _gates(conflict: bool = False) -> dict:
    blocked = (
        ["CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT"]
        if conflict else []
    )
    return {
        "domains": {
            "BUNDLE_BRANCH": {
                "eligible": False,
                "blocked_by_conflicts": blocked,
                "unusable_measurements": ["qrs_ms"],
            },
        },
        "global_unusable_targets": ["qrs_ms"],
    }


def main() -> None:
    layer = build_high_recall_candidates(
        _feature_graph(),
        _crosslead(),
        {},
    )
    candidate = dict((layer.get("by_code") or {}).get(CODE) or {})
    assert candidate, layer
    assert MARKER in set(candidate.get("evidence") or []), candidate
    assert candidate.get("required_measurements") == ["qrs_ms"], candidate

    rescued = fuse_candidate_evidence(
        {"candidates": [candidate]},
        _gates(False),
    )["by_code"][CODE]
    assert rescued["publishable"] is True, rescued
    assert rescued["rbbb_qrs118_rescue_active"] is True, rescued
    assert rescued["unresolved_required_measurements"] == [], rescued
    assert rescued["boundary_failures"] == [], rescued

    blocked_conflict = fuse_candidate_evidence(
        {"candidates": [candidate]},
        _gates(True),
    )["by_code"][CODE]
    assert blocked_conflict["publishable"] is False, blocked_conflict
    assert blocked_conflict["rbbb_qrs118_rescue_active"] is False, blocked_conflict

    without_marker = dict(candidate)
    without_marker["evidence"] = [
        x for x in (candidate.get("evidence") or []) if x != MARKER
    ]
    blocked_missing_marker = fuse_candidate_evidence(
        {"candidates": [without_marker]},
        _gates(False),
    )["by_code"][CODE]
    assert blocked_missing_marker["publishable"] is False, blocked_missing_marker
    assert blocked_missing_marker["rbbb_qrs118_rescue_active"] is False, blocked_missing_marker

    print("MEDCALC_RBBB_QRS118_RESCUE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
