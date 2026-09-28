from __future__ import annotations

import copy

from ecg_validation_guard import (
    ProvenanceError,
    assert_consumed_baseline,
    assert_development_dataset,
    assert_external_dataset,
    load_registry,
    validate_registry,
)


def must_fail(fn, *args) -> None:
    try:
        fn(*args)
    except ProvenanceError:
        return
    raise AssertionError("Expected ProvenanceError")


def main() -> None:
    registry = load_registry()
    summary = validate_registry(registry)
    assert summary["development_contaminated_n"] >= 10, summary
    assert summary["external_locked_n"] >= 0, summary
    assert summary["external_consumed_n"] >= 3, summary

    assert_development_dataset("ludb", registry)
    must_fail(assert_external_dataset, "ludb", registry)

    must_fail(assert_external_dataset, "code_test", registry)
    must_fail(assert_external_dataset, "sph", registry)
    assert_consumed_baseline("code_test", registry)
    assert_consumed_baseline("sph", registry)
    assert_consumed_baseline("mimic_iv_ecg", registry)
    must_fail(assert_external_dataset, "mimic_iv_ecg", registry)

    bad = copy.deepcopy(registry)
    bad.setdefault("provisional_external_locked", []).append({
        "id": "synthetic_external_guard_case",
        "status": "PROVISIONAL_EXTERNAL_LOCKED",
        "frozen": True,
        "allow_tuning": True,
        "allow_threshold_selection": False,
        "external_validation_allowed": True,
    })
    must_fail(validate_registry, bad)

    bad2 = copy.deepcopy(registry)
    bad2["development_contaminated"][0]["external_validation_allowed"] = True
    must_fail(validate_registry, bad2)

    print("MEDCALC_ECG_DATASET_PROVENANCE_GUARD_PASS")


if __name__ == "__main__":
    main()
