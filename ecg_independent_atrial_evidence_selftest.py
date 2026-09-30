from __future__ import annotations

from ecg_independent_atrial_evidence import (
    build_independent_atrial_consensus,
    filter_atrial_candidates_outside_ventricular_repolarization,
)


def _lead(p, *, fs=500):
    return {
        "evaluable": True,
        "fs": fs,
        "raw_p_peaks_samples": list(p),
    }


def main() -> None:
    # Same atrial events in two leads survive small delineation jitter.
    per_lead = {
        "II": _lead([100, 400, 700, 1000, 1300]),
        "V1": _lead([106, 405, 694, 1007, 1295]),
    }
    result = build_independent_atrial_consensus(per_lead)
    assert result["event_n"] == 5, result
    assert result["organized"], result
    assert result["diagnostic_claim_allowed"] is False, result

    # A periodic sequence seen in only one lead is explicitly insufficient.
    single = build_independent_atrial_consensus({
        "II": _lead([100, 400, 700, 1000, 1300]),
    })
    assert single["event_n"] == 0, single
    assert not single["organized"], single

    # Coincident noise at only one time point is not an organized atrial train.
    sparse = build_independent_atrial_consensus({
        "II": _lead([100, 400, 700, 1000]),
        "V1": _lead([106, 900, 1400, 1800]),
    })
    assert sparse["event_n"] == 1, sparse
    assert not sparse["organized"], sparse

    # T/QRS guard: a candidate near ventricular repolarization is rejected,
    # while a later candidate outside measured T-end remains observable.
    kept, audit = filter_atrial_candidates_outside_ventricular_repolarization(
        [140, 260, 360, 500],
        r_peaks=[100],
        t_offsets=[300],
        fs=500,
    )
    assert kept == [360, 500], (kept, audit)
    assert audit["rejected_n"] == 2, audit

    # If T-end is unavailable, use the conservative fixed post-R guard rather
    # than treating an unmeasured repolarization interval as atrial evidence.
    kept_fallback, audit_fallback = filter_atrial_candidates_outside_ventricular_repolarization(
        [140, 260, 320, 360],
        r_peaks=[100],
        t_offsets=[],
        fs=500,
    )
    assert kept_fallback == [320, 360], (kept_fallback, audit_fallback)

    print("MEDCALC_INDEPENDENT_ATRIAL_EVIDENCE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
