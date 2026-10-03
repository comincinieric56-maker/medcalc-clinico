import pytest

from ecg_av_prepare_development_images import display_windows, select_records


def rows():
    return {"1": {"ecg_id": "1", "patient_id": "p1", "strat_fold": "1", "scp_codes": "{}"},
            "2": {"ecg_id": "2", "patient_id": "p1", "strat_fold": "2", "scp_codes": "{}"},
            "3": {"ecg_id": "3", "patient_id": "p2", "strat_fold": "1", "scp_codes": "{}"}}


def inventory(ids):
    return {"ptbxl_delineation": {"selection": {"eligible_ecg_ids": ids}}}


def test_image_selection_is_deterministic_and_one_record_per_patient():
    a = select_records(inventory([1, 2, 3]), rows(), set(), 2)
    b = select_records(inventory([3, 2, 1]), rows(), set(), 2)
    assert a == b
    assert len({r["patient_id"] for r in a}) == 2


def test_stale_inventory_cannot_include_a_newly_protected_patient():
    with pytest.raises(ValueError, match="protected"):
        select_records(inventory([2, 3]), rows(), {"1"}, 1)


def test_heldout_record_cannot_enter_image_preparation():
    metadata = rows(); metadata["3"]["strat_fold"] = "9"
    with pytest.raises(ValueError, match="heldout"):
        select_records(inventory([1, 3]), metadata, set(), 1)


def test_short_leads_and_rhythm_strip_have_distinct_time_windows():
    windows = display_windows()
    assert len(windows) == 13
    assert next(w for w in windows if w["lead"] == "V4")["start_s"] == 7.5
    ii = [w for w in windows if w["lead"] == "II"]
    assert [w["end_s"] for w in ii] == [2.5, 10.]
    assert ii[1]["role"] == "CONTIGUOUS_RHYTHM_STRIP"
