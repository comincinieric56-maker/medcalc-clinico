from __future__ import annotations

from ecg_independent_atrial_evidence import build_independent_atrial_consensus


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

    print("MEDCALC_INDEPENDENT_ATRIAL_EVIDENCE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
