import pytest

from ecg_av_ludb_event_probe import match_events, score_events


def test_matching_maximizes_pairs_instead_of_nearest_pair_greed():
    counts, unmatched = match_events([1, 1.1], [1.06, 1.16], .061)
    assert counts == {"tp": 2, "fp": 0, "fn": 0}
    assert unmatched == []


def test_duplicates_cannot_reuse_one_reference():
    counts, unmatched = match_events([1], [1, 1])
    assert counts == {"tp": 1, "fp": 1, "fn": 0}
    assert unmatched == [1]


def test_unannotated_edges_excluded_and_t_coincidence_counted():
    events = [{"kind": "P", "onset": 100, "peak": 120, "offset": 140},
              {"kind": "QRS", "onset": 160, "peak": 180, "offset": 200},
              {"kind": "T", "onset": 230, "peak": 250, "offset": 280}]
    nodes = [{"kind": "P", "time_s": t} for t in (.1, 1.2, 2.5, 3)]
    counts = score_events(events, nodes, 100)
    assert counts["P"] == {"tp": 1, "fp": 1, "fn": 0,
                           "unmatched_near_t_peak": 1, "unmatched_near_qrs_peak": 0}
    assert counts["QRS"]["fn"] == 1


def test_empty_reference_is_error_not_perfect_score():
    with pytest.raises(ValueError):
        score_events([], [], 500)
