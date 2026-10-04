import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_layout_detector import recover_signal_geometry_from_preflight


def load_contract():
    path = Path(__file__).resolve().parents[1] / 'ecg_digitizer_vendor/src/model/coordinate_contract.py'
    spec = importlib.util.spec_from_file_location('aligned_coordinate_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restore_aligned_lines


@pytest.mark.parametrize('rows', [3, 4, 6, 7, 12])
def test_all_layout_row_counts_keep_exact_samples_and_missing_margins(rows):
    source = torch.arange(rows * 80, dtype=torch.float32).reshape(rows, 80)
    source[:, 20:25] = float('nan')
    restored, bounds = load_contract()(source, 120, (15, 94))
    assert bounds == [15, 94]
    assert restored.shape == (rows, 120)
    assert torch.allclose(restored[:, 15:95], source, equal_nan=True)
    assert torch.isnan(restored[:, :15]).all()
    assert torch.isnan(restored[:, 95:]).all()


def test_absent_or_inconsistent_provenance_never_stretches():
    restore = load_contract()
    with pytest.raises(ValueError, match='UNAVAILABLE'):
        restore(torch.ones(4, 80), 120, None)
    with pytest.raises(ValueError, match='WIDTH_MISMATCH'):
        restore(torch.ones(4, 80), 120, (15, 95))
    result, bounds = restore(torch.empty(0, 120), 120, None)
    assert result.shape == (0, 120) and bounds is None


@pytest.mark.parametrize('layout,rows', [('3x4', 3), ('6x2', 6)])
def test_guided_geometry_uses_aligned_x_before_measuring_support(layout, rows):
    centers = [40 + 50 * i for i in range(rows)]
    h = 50 * rows + 40
    probability = np.zeros((h, 240), dtype=np.float32)
    for y in centers:
        probability[y-2:y+3, 30:211] = 1
    preflight = {'layout': layout, 'primary_centers_y': centers,
                 'detection_image_size': [240, h], 'active_x': [5, 230]}
    geometry = recover_signal_geometry_from_preflight(
        probability, preflight, aligned_active_x=[30, 210])
    assert geometry['active_x'] == [30, 210]
    assert geometry['primary_row_support'] == [1.] * rows


def test_invalid_aligned_geometry_rejected():
    preflight = {'layout': '3x4', 'primary_centers_y': [20, 50, 80],
                 'detection_image_size': [100, 100]}
    with pytest.raises(ValueError, match='INVALID_ALIGNED'):
        recover_signal_geometry_from_preflight(np.ones((100, 100)), preflight,
                                               aligned_active_x=[0, 100])
