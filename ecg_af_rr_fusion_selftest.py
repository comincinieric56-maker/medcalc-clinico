from __future__ import annotations

from ecg_evidence_fusion import fuse_candidate_evidence


def _gates() -> dict:
    return {
        "domains": {
            "RHYTHM": {
                "eligible": True,
                "blocked_by_conflicts": [],
                "unusable_measurements": [],
            },
            "BUNDLE_BRANCH": {
                "eligible": True,
                "blocked_by_conflicts": [],
                "unusable_measurements": [],
            },
        },
        "global_unusable_targets": [],
    }


def main() -> None:
    af_without_rr = {
        "domain": "RHYTHM",
        "code": "AF_COMPATIBLE",
        "score": 0.90,
        "evidence": ["AF_COMPATIBILITY_SIGNAL", "ATRIAL_SPECIALIST_AF"],
        "source_groups": ["ATRIAL_SPECTRAL", "ATRIAL_SPECIALIST"],
        "independent_evidence_n": 2,
        "required_measurements": [],
        "boundary_requirements": [],
        "specialist_confirmed": True,
    }
    blocked = fuse_candidate_evidence(
        {"candidates": [af_without_rr]},
        _gates(),
    )["by_code"]["AF_COMPATIBLE"]
    assert blocked["publishable"] is False, blocked
    assert blocked["fusion_reason"] == "AF_REQUIRES_RR_IRREGULAR_SUPPORT", blocked
    assert blocked["af_rr_irregular_support"] is False, blocked

    af_with_rr = {
        **af_without_rr,
        "evidence": [
            "AF_COMPATIBILITY_SIGNAL",
            "ATRIAL_SPECIALIST_AF",
            "RR_IRREGULAR",
        ],
        "source_groups": ["ATRIAL_SPECTRAL", "ATRIAL_SPECIALIST", "RR"],
        "independent_evidence_n": 3,
    }
    published = fuse_candidate_evidence(
        {"candidates": [af_with_rr]},
        _gates(),
    )["by_code"]["AF_COMPATIBLE"]
    assert published["publishable"] is True, published
    assert published["af_rr_irregular_support"] is True, published

    rbbb = {
        "domain": "BUNDLE_BRANCH",
        "code": "RBBB_MORPHOLOGY_COMPATIBLE",
        "score": 0.80,
        "evidence": ["QRS_GE_120MS", "RIGHT_TERMINAL_R"],
        "source_groups": ["QRS_DURATION", "RIGHT_PRECORDIAL_MORPHOLOGY"],
        "independent_evidence_n": 2,
        "required_measurements": [],
        "boundary_requirements": [],
        "specialist_confirmed": True,
    }
    unaffected = fuse_candidate_evidence(
        {"candidates": [rbbb]},
        _gates(),
    )["by_code"]["RBBB_MORPHOLOGY_COMPATIBLE"]
    assert unaffected["publishable"] is True, unaffected
    assert unaffected["af_rr_irregular_support"] is True, unaffected

    print("MEDCALC_AF_RR_FUSION_SELFTEST_PASS")


if __name__ == "__main__":
    main()
