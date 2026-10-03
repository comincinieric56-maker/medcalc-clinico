from __future__ import annotations

import copy
import json
import numpy as np
import pytest

from ecg_av_event_graph import build_av_event_graph
from ecg_av_temporal_model import analyze_av_research, events_from_probabilities
from ecg_av_training_data import FS, synthetic_case
from ecg_av_train import load_real_training


def events(times):
    return [{"time_s": t, "confidence": .9} for t in times]


def test_graph_preserves_ambiguous_edges_without_claiming_conduction():
    graph = build_av_event_graph(events([.1, .4]), events([.58]), 1.)
    assert len(graph["candidate_edges"]) == 2
    assert graph["ambiguous_node_n"] == 1
    assert graph["diagnostic_claim_allowed"] is False
    assert "conducted_p_n" not in graph
    assert "pr_median_ms" not in graph


def test_boundary_p_not_counted_as_blocked():
    graph = build_av_event_graph(events([.1, .85]), [], 1.)
    assert graph["p_without_candidate_qrs_n"] == 1


@pytest.mark.parametrize("time,confidence", [(float("nan"), .9), (1.5, .9), (.2, 2)])
def test_graph_rejects_invalid_events(time, confidence):
    with pytest.raises(ValueError):
        build_av_event_graph([{"time_s": time, "confidence": confidence}], [], 1.)


def test_independent_detector_keeps_p_between_qrs():
    prob = np.zeros((2, FS * 8))
    prob[0, [100, 250, 400, 550, 700, 850]] = .9
    prob[1, [150, 450, 750]] = .95
    p, r = events_from_probabilities(prob)
    assert len(p) == 6 and len(r) == 3
    assert build_av_event_graph(p, r, 8.)["p_without_candidate_qrs_n"] >= 2


class SpyModel:
    def __init__(self):
        self.called = []

    def analyze_signal(self, signal, fs):
        self.called.append((signal.copy(), fs))
        return {"diagnostic_claim_allowed": False, "clinical_fusion_allowed": False}


def canonical():
    return {"fs": FS, "leads": {"II": {"signal_mv": np.ones(FS * 8).tolist(),
                                          "quality_mask": [2] * (FS * 8), "fs": FS}}}


def test_research_does_not_mutate_native_input():
    ecg, model = canonical(), SpyModel()
    before = copy.deepcopy(ecg)
    result = analyze_av_research(ecg, model)
    assert ecg == before and len(model.called) == 1
    assert result["observed_window"]["crosslead_alignment_assumed"] is False


def test_repeated_segments_and_missing_data_never_joined():
    ecg, model = canonical(), SpyModel()
    ecg["leads"]["II"]["repeated"] = True
    assert analyze_av_research(ecg, model)["status"] == "NOT_EVALUABLE"
    ecg["leads"]["II"].pop("repeated")
    ecg["leads"]["II"]["quality_mask"][FS * 4] = 0
    assert analyze_av_research(ecg, model)["status"] == "NOT_EVALUABLE"
    assert not model.called


def test_independent_training_namespaces_and_wenckebach_progression():
    row = synthetic_case(2)
    assert np.all(np.diff((row["r_s"][:3] - row["p_s"][:3])) > 0)
    assert not np.array_equal(row["signal"], synthetic_case(2, "heldout")["signal"])
    assert np.array_equal(row["signal"], synthetic_case(2)["signal"])


@pytest.mark.parametrize("override", [{"fold": 9}, {"fold": 10}, {"ecg_id": 351},
                                     {"dataset_id": "mimic"}, {"waveform_origin": "NATIVE"}])
def test_real_training_cannot_use_protected_data(tmp_path, override):
    row = {"dataset_id": "ptbxl", "usage_role": "DEVELOPMENT_TRAIN",
           "waveform_origin": "DIGITIZED_IMAGE", "patient_id": "123", "ecg_id": 99999,
           "fold": 1, "annotation_source": "EXPERT_P_QRS_AND_RHYTHM"}
    row.update(override)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="provenance"):
        load_real_training(path)


def test_optional_branch_cannot_change_clinical_output():
    from ecg_signal_measurements import analyze_canonical_ecg
    from ecg_synthetic_signal_cohort import all_specs, canonical as make_canonical, make_signal
    spec = next(s for s in all_specs() if s.get("target") == "AVB2")
    ecg = make_canonical(spec, make_signal(spec))
    baseline = analyze_canonical_ecg(ecg)
    research = analyze_canonical_ecg(ecg, av_research_model=SpyModel())
    assert "av_research" not in baseline
    assert research.pop("av_research")["clinical_fusion_allowed"] is False
    assert research == baseline


def test_checkpoint_roundtrip_and_research_failure_isolation(tmp_path):
    import torch
    from ecg_av_event_graph import TOKEN_FEATURES
    from ecg_av_temporal_model import AVResearchModel, VERSION, create_networks
    from ecg_av_training_data import CLASSES
    detector, classifier = create_networks()
    path = tmp_path / "research.pt"
    metadata = {"version": VERSION, "classes": list(CLASSES), "fs": FS,
                "token_features": list(TOKEN_FEATURES), "diagnostic_claim_allowed": False,
                "training_source": "UNIT_TEST_RANDOM_WEIGHTS"}
    torch.save({"metadata": metadata, "detector": detector.state_dict(),
                "classifier": classifier.state_dict()}, path)
    model = AVResearchModel(path)
    result = model.analyze_signal(synthetic_case(2)["signal"][0], FS)
    assert result["diagnostic_claim_allowed"] is False
    assert result["clinical_fusion_allowed"] is False and result["abstain"]
    assert len(result["probabilities"]) == len(CLASSES)
    from ecg_signal_measurements import analyze_canonical_ecg
    class BrokenModel:
        def analyze_signal(self, *args):
            raise ValueError("Intentional checkpoint test failure")
    ecg = canonical()
    baseline = analyze_canonical_ecg(ecg)
    result = analyze_canonical_ecg(ecg, av_research_model=BrokenModel())
    assert result.pop("av_research")["status"] == "RESEARCH_ERROR"
    assert result == baseline
