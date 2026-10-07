import importlib.util
from pathlib import Path

import torch


def extractor():
    path = Path(__file__).resolve().parents[1] / 'ecg_digitizer_vendor/src/model/signal_extractor.py'
    spec = importlib.util.spec_from_file_location('fragment_test_extractor', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SignalExtractor()


def test_short_waveform_piece_survives_real_graph_merge_without_filling_gaps():
    engine = extractor()
    pieces = []
    for start, end in [(10, 100), (105, 125), (130, 240)]:
        row = torch.full((300,), float('nan'))
        row[start:end] = 50.
        pieces.append(row)
    pieces[1][105:125] += torch.sin(torch.linspace(0, torch.pi, 20)) * 8
    engine._iterative_extraction = lambda _: pieces
    engine._autodetect_num_peaks = lambda _: 1
    result = engine(torch.zeros(100, 300))
    assert result.shape == (1, 230)
    assert engine.last_crop_bounds == (10, 239)
    assert torch.equal(result[0, 95:115], pieces[1][105:125])
    assert torch.isnan(result[0, 90:95]).all()
    assert torch.isnan(result[0, 115:120]).all()
    assert torch.isfinite(result).sum() == 220


def test_isolated_short_piece_still_rejected_after_merge():
    engine = extractor()
    piece = torch.full((300,), float('nan')); piece[80:100] = 50.
    engine._iterative_extraction = lambda _: [piece]
    engine._autodetect_num_peaks = lambda _: 1
    assert engine(torch.zeros(100, 300)).shape == (0, 300)
    assert engine.last_crop_bounds is None


def test_empty_next_input_does_not_reuse_previous_crop():
    engine = extractor()
    engine.last_crop_bounds = (10, 239)
    engine._iterative_extraction = lambda _: []
    engine._autodetect_num_peaks = lambda _: 0
    assert engine(torch.zeros(100, 300)).shape == (0, 300)
    assert engine.last_crop_bounds is None
