from ecg_av_ptbxlplus_weak_supervision_audit import (
    _folder,
    _parse_csv_ints,
    _parse_csv_strings,
    annotation_url,
    _summarize_annotation,
)


def test_ptbxlplus_folder_contract():
    assert _folder(70) == "00000"
    assert _folder(484) == "00000"
    assert _folder(2188) == "02000"
    assert _folder(10000) == "10000"


def test_ptbxlplus_annotation_url_is_versioned_and_lead_specific():
    url = annotation_url(2188, "II")
    assert "/ptb-xl-plus/1.0.1/" in url
    assert "/02000/02188_points_lead_II.atr" in url


def test_csv_argument_contracts():
    assert _parse_csv_ints("70,2188,484") == (70, 2188, 484)
    assert _parse_csv_strings("II,V1") == ("II", "V1")



def test_capture_endpoint_is_separate_from_trainable_peak_domain():
    summary = _summarize_annotation(
        [10, 20, 30, 5000],
        ['"','"','"','"'],
        ["p-wave peak", "R peak", "t-wave peak", "t-wave offset"],
    )
    assert summary["all_annotations_within_closed_0_5000_domain"] is True
    assert summary["all_peak_targets_within_signal_sample_domain"] is True
    assert summary["capture_endpoint_event_n"] == 1
    assert summary["capture_endpoint_events"][0]["aux_note"] == "t-wave offset"
    assert summary["peak_targets_outside_signal_sample_domain"] == []


def test_peak_at_capture_endpoint_is_rejected_for_weak_supervision():
    summary = _summarize_annotation(
        [10, 5000],
        ['"','"'],
        ["p-wave peak", "R peak"],
    )
    assert summary["all_peak_targets_within_signal_sample_domain"] is False
    assert summary["peak_targets_outside_signal_sample_domain"][0]["aux_note"] == "R peak"
