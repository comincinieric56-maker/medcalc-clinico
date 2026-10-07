from __future__ import annotations

import pytest

from ecg_av_event_graph import build_av_event_graph
from ecg_av_graph_topology_probe import (
    graph_topology_features,
    maximum_non_crossing_candidate_matches,
)


def events(times, confidence=.9):
    return [{"time_s": float(t), "confidence": confidence} for t in times]


def test_maximum_non_crossing_matching_respects_one_to_one_candidate_edges():
    p = [0.10, 0.40, 0.70, 1.00]
    qrs = [0.26, 0.86, 1.16]
    assert maximum_non_crossing_candidate_matches(p, qrs) == 3


def test_graph_features_expose_excess_atrial_candidates_without_calling_conduction():
    p = [0.10, 0.50, 0.90, 1.30, 1.70, 2.10]
    qrs = [0.26, 1.06, 1.86]
    graph = build_av_event_graph(events(p), events(qrs), 2.5)
    features = graph_topology_features(graph)
    assert features["p_qrs_count_ratio"] == pytest.approx(2.0)
    assert features["minimum_unmatched_p_n"] >= 3
    assert features["minimum_unmatched_p_fraction"] >= 0.5
    assert graph["diagnostic_claim_allowed"] is False


def test_coupled_sequence_has_high_phase_concentration():
    p = [0.10, 0.90, 1.70, 2.50, 3.30]
    qrs = [x + 0.16 for x in p]
    graph = build_av_event_graph(events(p), events(qrs), 4.0)
    features = graph_topology_features(graph)
    assert features["phase_concentration"] > 0.95
    assert features["minimum_unmatched_p_n"] == 0
    assert features["p_qrs_rate_ratio"] == pytest.approx(1.0, rel=0.01)


def test_matching_rejects_nonfinite_times():
    with pytest.raises(ValueError):
        maximum_non_crossing_candidate_matches([0.1, float("nan")], [0.2])
