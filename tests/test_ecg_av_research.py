from __future__ import annotations

import copy
import hashlib
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


def test_t_auxiliary_channel_never_vetoes_overlapping_p():
    prob = np.zeros((3, FS * 4))
    prob[0, 100] = .95
    prob[2, 100] = .95
    prob[1, 500] = .95
    p, r, t = events_from_probabilities(prob, include_t=True)
    assert [round(e["time_s"], 3) for e in p] == [round(100 / FS, 3)]
    assert [round(e["time_s"], 3) for e in r] == [round(500 / FS, 3)]
    assert [round(e["time_s"], 3) for e in t] == [round(100 / FS, 3)]


def test_synthetic_training_exposes_lead_specific_t_supervision():
    row = synthetic_case(0)
    assert row["targets"].shape == (2, 3, FS * 10)
    assert len(row["t_s_by_lead"]) == 2
    assert row["targets"][0, 2].sum() > 0
    assert row["targets"][1, 2].sum() > 0


@pytest.mark.parametrize("namespace,index,expected", [
    ("train", 0, "1e03aa3acfedbb0d0e9fb6e47db0dafcf54f0ac5333305dde9e7ca8d5b867bd2"),
    ("train", 2, "c9830866e0bf9bda76c11a4f2bb1698329191525f9057750592cc21eec2eaf1f"),
    ("heldout", 0, "8fcac0bee89d1c2cd46211a20aa3b7f7d7d966f45617d7cb6e2ed965fc8737a8"),
])
def test_t_auxiliary_experiment_preserves_v1_waveform_bytes(namespace, index, expected):
    actual = hashlib.sha256(synthetic_case(index, namespace)["signal"].tobytes()).hexdigest()
    assert actual == expected


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
           "fold": 1, "annotation_source": "EXPERT_P_QRS_T_AND_RHYTHM", "t_s": []}
    row.update(override)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="provenance"):
        load_real_training(path)


def test_real_training_rejects_p_qrs_only_annotation_contract(tmp_path):
    row = {"dataset_id": "ptbxl", "usage_role": "DEVELOPMENT_TRAIN",
           "waveform_origin": "DIGITIZED_IMAGE", "patient_id": "123", "ecg_id": 99999,
           "fold": 1, "annotation_source": "EXPERT_P_QRS_AND_RHYTHM"}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="provenance"):
        load_real_training(path)


def test_optional_branch_cannot_change_clinical_output():
    from ecg_signal_measurements import analyze_canonical_ecg
    from ecg_av_research_adapter import analyze_ecg_with_av_research
    from ecg_synthetic_signal_cohort import all_specs, canonical as make_canonical, make_signal
    spec = next(s for s in all_specs() if s.get("target") == "AVB2")
    ecg = make_canonical(spec, make_signal(spec))
    baseline = analyze_canonical_ecg(ecg)
    research = analyze_ecg_with_av_research(ecg, av_research_model=SpyModel())
    assert "av_research" not in baseline
    assert research.pop("av_research")["clinical_fusion_allowed"] is False
    assert research == baseline


@pytest.mark.parametrize("event_channels", [("P", "QRS"), ("P", "QRS", "T")])
def test_checkpoint_roundtrip_and_research_failure_isolation(tmp_path, event_channels):
    import torch
    from ecg_av_event_graph import TOKEN_FEATURES
    from ecg_av_temporal_model import AVResearchModel, VERSION, create_networks
    from ecg_av_training_data import CLASSES
    detector, classifier = create_networks(len(event_channels))
    path = tmp_path / ("research-" + str(len(event_channels)) + ".pt")
    metadata = {"version": VERSION, "classes": list(CLASSES), "fs": FS,
                "token_features": list(TOKEN_FEATURES), "diagnostic_claim_allowed": False,
                "training_source": "UNIT_TEST_RANDOM_WEIGHTS"}
    if len(event_channels) == 3:
        metadata["event_channels"] = list(event_channels)
    torch.save({"metadata": metadata, "detector": detector.state_dict(),
                "classifier": classifier.state_dict()}, path)
    model = AVResearchModel(path)
    result = model.analyze_signal(synthetic_case(2)["signal"][0], FS)
    assert result["diagnostic_claim_allowed"] is False
    assert result["clinical_fusion_allowed"] is False and result["abstain"]
    assert result["event_channels"] == list(event_channels)
    assert len(result["probabilities"]) == len(CLASSES)
    from ecg_signal_measurements import analyze_canonical_ecg
    from ecg_av_research_adapter import analyze_ecg_with_av_research
    class BrokenModel:
        def analyze_signal(self, *args):
            raise ValueError("Intentional checkpoint test failure")
    ecg = canonical()
    baseline = analyze_canonical_ecg(ecg)
    result = analyze_ecg_with_av_research(ecg, av_research_model=BrokenModel())
    assert result.pop("av_research")["status"] == "RESEARCH_ERROR"
    assert result == baseline
