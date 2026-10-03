import zipfile

import numpy as np
import pytest

from ecg_av_annotation_source_audit import eligible_records, inspect_isp, inspect_mask


def metadata(patient, fold):
    return {"patient_id": patient, "strat_fold": str(fold), "scp_codes": "{'1AVB': 100}"}


def test_protected_patient_is_excluded_even_for_another_record():
    rows = {"1": metadata("p1", 1), "2": metadata("p1", 2),
            "3": metadata("p2", 9), "4": metadata("p3", 1)}
    selection = eligible_records(["00002_hr", "00003_hr", "00004_hr"], rows, {"1"})
    assert selection["eligible_ecg_ids"] == [4]
    assert selection["counts"] == {"protected_patient": 1, "heldout_fold": 1, "eligible_development": 1}
    assert selection["eligibility_is_not_annotation_approval"]


def test_duplicate_source_identifier_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        eligible_records(["00001_hr", "00001_hr"], {"1": metadata("p1", 1)}, set())


def test_missing_protected_metadata_rejected():
    with pytest.raises(ValueError, match="protected"):
        eligible_records([], {}, {"1"})


@pytest.mark.parametrize("value", [np.nan, .5, 4])
def test_invalid_mask_values_rejected(tmp_path, value):
    mask = np.zeros((12, 20)); mask[0, 0] = value
    path = tmp_path / "mask.npy"; np.save(path, mask)
    with pytest.raises(ValueError):
        inspect_mask(path, 1)


def test_valid_shape_does_not_imply_verified_time_or_classes(tmp_path):
    path = tmp_path / "mask.npy"; np.save(path, np.zeros((12, 1200)))
    result = inspect_mask(path, 1)
    assert not result["time_mapping_verified"] and not result["class_mapping_verified"]
    with pytest.raises(ValueError, match="rows"):
        inspect_mask(path, 2)


def test_split_record_names_are_scoped_and_duplicate_waveforms_reported(tmp_path):
    path = tmp_path / "isp.zip"
    with zipfile.ZipFile(path, "w") as z:
        for split in ("train", "test"):
            base = f"isp_delineation_dataset/{split}_data/1"
            z.writestr(base + ".hea", "1 12 1000 10000\n")
            z.writestr(base + ".dat", b"same-waveform")
            z.writestr(f"isp_delineation_dataset/{split}_isp_delineation_data.csv",
                       'file_name,age,sex,target\n1,40,0,"[(0, 10, 20), (1, 30, 40), (2, 90, 11000), (2, 50, 50)]"\n')
    result = inspect_isp(path)
    assert result["records_by_publisher_split"] == {"train": 1, "test": 1}
    assert result["cross_split_identical_dat_files"] == 1
    assert not result["training_allowed"] and not result["expert_peak_annotations_present"]
    assert result["invalid_annotation_counts"] == {
        "out_of_capture": 2, "empty_or_reversed": 2, "records_with_invalid_spans": 2}
    assert not result["invalid_spans_clipped_or_repaired"]
