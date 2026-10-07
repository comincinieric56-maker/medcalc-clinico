from ecg_av_ptbxlplus_weak_supervision_audit import (
    _folder,
    _parse_csv_ints,
    _parse_csv_strings,
    annotation_url,
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
