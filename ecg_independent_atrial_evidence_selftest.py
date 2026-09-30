from __future__ import annotations

import numpy as np

from ecg_independent_atrial_evidence import (
    build_independent_atrial_consensus,
    filter_atrial_candidates_outside_ventricular_repolarization,
    recover_crosslead_atrial_candidates,
    recover_morphology_matched_atrial_candidates,
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

    # Cross-lead consensus must not turn a lead-specific repolarization
    # candidate into an organized atrial train after the ventricular guard.
    ii_kept, _ = filter_atrial_candidates_outside_ventricular_repolarization(
        [140, 360, 640, 860], r_peaks=[100, 600], t_offsets=[300, 800], fs=500
    )
    v1_kept, _ = filter_atrial_candidates_outside_ventricular_repolarization(
        [145, 365, 645, 865], r_peaks=[100, 600], t_offsets=[305, 805], fs=500
    )
    guarded = build_independent_atrial_consensus({
        "II": _lead(ii_kept),
        "V1": _lead(v1_kept),
    })
    assert guarded["event_n"] == 2, guarded
    assert not guarded["organized"], guarded
    assert guarded["diagnostic_claim_allowed"] is False, guarded


    # Deterministic morphology recovery: conducted P seeds are spaced at twice
    # the true atrial period; homologous nonconducted P waves lie after T-end.
    fs = 500
    n = 3000
    grid = np.arange(n, dtype=float)
    seeds = [300, 900, 1500, 2100, 2700]
    hidden = [600, 1200, 1800, 2400]
    r_peaks = [380, 980, 1580, 2180, 2780]
    t_offsets = [550, 1150, 1750, 2350, 2950]

    def gaussian(center, sigma):
        return np.exp(-0.5 * ((grid - float(center)) / float(sigma)) ** 2)

    def make_signal(offset=0, include_hidden=True, prominent_t=False):
        y = np.zeros(n, dtype=float)
        for p in seeds:
            y += 0.12 * gaussian(p + offset, 13.5)
        if include_hidden:
            for p in hidden:
                y += 0.114 * gaussian(p + offset, 13.5)
        for r in r_peaks:
            y += 1.00 * gaussian(r + offset, 6.0)
            y += (0.42 if prominent_t else 0.24) * gaussian(r + offset + 110, 30.0)
        y += 0.0015 * np.sin(2.0 * np.pi * grid / 173.0)
        return y

    recovered, recovery_audit = recover_morphology_matched_atrial_candidates(
        make_signal(),
        seed_p_peaks=seeds,
        r_peaks=r_peaks,
        t_offsets=t_offsets,
        fs=fs,
    )
    assert len(recovered) == len(hidden), (recovered, recovery_audit)
    assert all(min(abs(v - h) for v in recovered) <= 6 for h in hidden), recovered

    canonical = {
        "fs": fs,
        "leads": {
            "II": {"fs": fs, "signal_mv": make_signal(0).tolist()},
            "V1": {"fs": fs, "signal_mv": make_signal(4).tolist()},
        },
    }
    measured = {
        "II": {
            "evaluable": True, "fs": fs,
            "raw_p_peaks_samples": seeds,
            "r_peaks_samples": r_peaks,
            "beats": [{"t_offset_sample": v} for v in t_offsets],
        },
        "V1": {
            "evaluable": True, "fs": fs,
            "raw_p_peaks_samples": [v + 4 for v in seeds],
            "r_peaks_samples": [v + 4 for v in r_peaks],
            "beats": [{"t_offset_sample": v + 4} for v in t_offsets],
        },
    }
    crosslead = recover_crosslead_atrial_candidates(canonical, measured)
    assert crosslead["recovered_event_n"] == len(hidden), crosslead
    assert crosslead["organized_augmented"], crosslead
    assert crosslead["diagnostic_claim_allowed"] is False, crosslead

    # Prominent T waves without hidden atrial events must not be recovered.
    no_hidden, no_hidden_audit = recover_morphology_matched_atrial_candidates(
        make_signal(include_hidden=False, prominent_t=True),
        seed_p_peaks=seeds,
        r_peaks=r_peaks,
        t_offsets=t_offsets,
        fs=fs,
    )
    assert no_hidden == [], (no_hidden, no_hidden_audit)

    print("MEDCALC_INDEPENDENT_ATRIAL_EVIDENCE_SELFTEST_PASS")


if __name__ == "__main__":
    main()
