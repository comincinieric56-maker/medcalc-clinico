import copy

from ecg_av_image_preparation_audit import inspect_digitized


def fixture():
    manifest = {"lead_display_windows": [{"lead": "II", "start_s": 0, "end_s": 10},
                                          {"lead": "aVF", "start_s": 2.5, "end_s": 5}]}
    meta = {"status": "DIGITIZED_ONLY", "signal": {
        "observed_seconds_by_lead": {"II": 8, "aVF": 2.5},
        "calibrated_digital_signal": {"fs": 100, "leads": {"II": {
            "signal_mv": [1.] * 800, "quality_mask": [2] * 800}}}}}
    return meta, manifest


def test_available_window_still_cannot_train_without_verified_annotations():
    meta, manifest = fixture(); before = copy.deepcopy(meta)
    result = inspect_digitized(meta, manifest)
    assert result["r28_input_window_available"]
    assert not result["training_allowed"] and not result["annotation_alignment_verified"]
    assert meta == before


def test_long_strip_mapped_to_short_avf_is_rejected():
    meta, manifest = fixture(); meta["signal"]["observed_seconds_by_lead"]["aVF"] = 8
    result = inspect_digitized(meta, manifest)
    assert result["unexpected_duration_leads"] == ["aVF"]
    assert not result["r28_input_window_available"]


def test_interpolated_gap_does_not_join_observed_segments():
    meta, manifest = fixture()
    meta["signal"]["calibrated_digital_signal"]["leads"]["II"]["quality_mask"][400] = 1
    result = inspect_digitized(meta, manifest)
    assert result["longest_contiguous_observed_ii_s"] == 4
    assert not result["r28_input_window_available"]


def test_tiled_input_rejected_even_when_it_has_a_long_window():
    meta, manifest = fixture(); meta["signal"]["r27_tiled"] = True
    result = inspect_digitized(meta, manifest)
    assert "REPEATED_SIGNAL_REJECTED" in result["blockers"]
    assert not result["r28_input_window_available"]


def test_long_time_axis_cannot_pass_by_hiding_edges_as_missing_samples():
    meta, manifest = fixture()
    ii = meta["signal"]["calibrated_digital_signal"]["leads"]["II"]
    ii["signal_mv"] += [None] * 250
    ii["quality_mask"] += [0] * 250
    result = inspect_digitized(meta, manifest)
    assert result["canonical_ii_time_extent_s"] == 10.5
    assert "CANONICAL_II_TIME_EXTENT_EXCEEDS_KNOWN_DISPLAY_WINDOW" in result["blockers"]
    assert not result["r28_input_window_available"]
