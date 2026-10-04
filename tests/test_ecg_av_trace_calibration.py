import pytest
import torch

from ecg_av_trace_calibration import restore_extractor_coordinates


def test_crop_restoration_preserves_time_distances_gaps_and_amplitude():
    row = torch.zeros(1732)
    row[100] = 8.; row[300] = 9.; row[400:450] = float('nan')
    restored = restore_extractor_coordinates([row], 150, 1881, 2000)[0]
    assert restored[250] == 8. and restored[450] == 9.
    assert torch.isnan(restored[:150]).all()
    assert torch.isnan(restored[1882:]).all()
    assert torch.isnan(restored[550:600]).all()
    assert torch.allclose(restored[150:1882], row, equal_nan=True)
    # A 200-pixel interval retains its duration at the measured pixel scale.
    assert (450 - 250) * .14449428021907806 / 25 == pytest.approx(1.15595424175)


@pytest.mark.parametrize('left,right,width', [(-1, 2, 5), (2, 1, 5), (0, 5, 5)])
def test_invalid_crop_bounds_fail(left, right, width):
    with pytest.raises(ValueError):
        restore_extractor_coordinates([torch.ones(3)], left, right, width)


def test_unknown_crop_size_is_never_stretched_to_fit():
    with pytest.raises(ValueError, match='WIDTH_MISMATCH'):
        restore_extractor_coordinates([torch.ones(1732)], 71, 1925, 2000)
