import numpy as np
import pytest
from pathlib import Path

from ecg_av_red_grid_audit import inspect_profile, inspect_image


@pytest.mark.parametrize('spacing,offset', [(4, 0), (6, 3), (8, 11)])
def test_known_scale_and_crop_phase(spacing, offset):
    x = np.arange(1600) + offset
    profile = np.zeros(1600)
    profile[x % spacing == 0] = 20
    profile[x % (5 * spacing) == 0] = 45
    result = inspect_profile(profile)
    assert result['accepted']
    assert result['mm_per_pixel'] == 1 / spacing


def test_uniform_lines_abstain_instead_of_guessing_major_period():
    profile = np.zeros(1600); profile[::4] = 30
    assert inspect_profile(profile)['reason'] == 'MAJOR_MINOR_AMBIGUOUS'


def test_missing_minor_lines_cannot_be_mistaken_for_major_scale():
    profile = np.zeros(1600); profile[::20] = 45
    assert not inspect_profile(profile)['accepted']


def test_invalid_and_blank_profiles_abstain():
    assert not inspect_profile(np.zeros(100))['accepted']
    assert not inspect_profile([float('nan')] * 100)['accepted']


def test_wider_major_lines_and_fixed_prepared_images():
    profile = np.zeros(1600); profile[::4] = 45; profile[1::20] = 45
    assert inspect_profile(profile)['mm_per_pixel'] == .25
    root = Path(__file__).resolve().parents[1] / 'models/r28_av_research/prepared_images'
    paths = sorted(root.glob('*.png'))
    assert len(paths) == 3
    for path in paths:
        result = inspect_image(path)
        assert result['accepted']
        assert not result['clinical_scale_override_allowed']
        assert not result['training_allowed']
        assert all(axis['mm_per_pixel'] == .25 for axis in result['axes'].values())
