import importlib.util
from pathlib import Path

import pytest
import torch

from ecg_av_grid_calibration_research import install_research_correction, zero_lag_grid_search


def test_search_receives_zero_lag_at_index_zero():
    class Recorder:
        max_number_of_grid_lines = 120
        min_number_of_grid_lines = 15
        samples = 10
        max_zoom = 0

        def _grid_search_min_distance(self, correlation, *args):
            self.correlation = correlation
            return 20

    finder = Recorder()
    source = torch.tensor([1., 2., 3., 9., 3., 2., 1.])
    before = source.clone()
    assert zero_lag_grid_search(finder, source, 1) == 20
    assert torch.equal(source, before)
    assert torch.equal(finder.correlation, torch.tensor([9., 3., 2., 1.]) - 3.75)


@pytest.mark.parametrize('spacing', [20, 30, 40])
def test_direct_correction_is_not_safe_to_promote(spacing):
    path = Path(__file__).resolve().parents[1] / 'ecg_digitizer_vendor/src/model/pixel_size_finder.py'
    spec = importlib.util.spec_from_file_location('research_grid_vendor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = torch.zeros(1600)
    x = torch.arange(1600)
    profile[x % spacing == 0] = 1
    profile[(x % (spacing // 5) == 0) & (x % spacing != 0)] = .5
    image = profile[:, None].repeat(1, 2)
    finder = module.PixelSizeFinder()
    baseline = float(finder._find_pxls_between_horizontal_grid_lines(image, 1))
    install_research_correction(finder)
    candidate = float(finder._find_pxls_between_horizontal_grid_lines(image, 1))
    assert abs(baseline / spacing - 1) < .003
    if spacing in (20, 40):
        assert abs(candidate / spacing - 1) > .45
    else:
        assert abs(candidate / spacing - 1) < .003
