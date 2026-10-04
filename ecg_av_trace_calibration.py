"""Research-only instrumentation: run with ecg_unet_worker arguments.

ECG_CALIBRATION_TRACE points to a directory for intermediate probability maps.
By default hooks observe original computations. ECG_RESEARCH_PRESERVE_X=1
restores extractor crop coordinates and uses its aligned bounds for the
high-fidelity route. This opt-in experiment never replaces grid calibration.
"""
import json
import os
from pathlib import Path

import numpy as np


def restore_extractor_coordinates(rows, left, right, canvas_width):
    """Undo the extractor's horizontal crop by padding, without resampling."""
    if not 0 <= left <= right < canvas_width:
        raise ValueError('INVALID_EXTRACTOR_CROP_BOUNDS')
    restored = []
    for row in rows:
        if row.ndim != 1 or row.numel() != right-left+1:
            raise ValueError('EXTRACTOR_CROP_WIDTH_MISMATCH')
        padded = row.new_full((canvas_width,), float('nan'))
        padded[left:right+1] = row
        restored.append(padded)
    return restored


def main():
    import ecg_unet_worker as worker
    destination = Path(os.environ['ECG_CALIBRATION_TRACE'])
    destination.mkdir(parents=True, exist_ok=True)
    events = []
    loader = worker._load_digitizer
    preserve_x = os.environ.get('ECG_RESEARCH_PRESERVE_X') == '1'
    aligned_bounds = {}
    recover_geometry = worker.recover_signal_geometry_from_preflight

    def recover_with_aligned_bounds(probability, *args, **kwargs):
        geometry = recover_geometry(probability, *args, **kwargs)
        bounds = aligned_bounds.get(tuple(probability.shape))
        if preserve_x and bounds is not None:
            geometry['active_x'] = list(bounds)
            geometry['active_x_debug'] = {'source': 'RESEARCH_EXTRACTOR_ALIGNED_COORDINATES'}
        return geometry

    worker.recover_signal_geometry_from_preflight = recover_with_aligned_bounds

    def record(event):
        events.append(event)
        (destination / 'trace.json').write_text(json.dumps(events, indent=2) + '\n')

    def traced_loader(*args, **kwargs):
        model = loader(*args, **kwargs)
        extractor = model.signal_extractor
        merge = extractor.match_and_merge_lines

        def traced_merge(lines):
            import torch
            valid = torch.isfinite(lines) & (lines != 0)
            columns = torch.where(valid.any(dim=0))[0]
            result, overlaps = merge(lines)
            if columns.numel():
                left, right = int(columns[0]), int(columns[-1])
                record({'stage': 'extractor_crop', 'canvas_width': int(lines.shape[1]),
                        'left': left, 'right': right, 'cropped_width': right-left+1,
                        'preserve_x_experiment': bool(preserve_x and hasattr(model, '_research_aligned_shape'))})
                if preserve_x and hasattr(model, '_research_aligned_shape'):
                    result = restore_extractor_coordinates(result, left, right, int(lines.shape[1]))
                    # Shape of the same aligned signal map used by the extractor.
                    aligned_bounds[tuple(model._research_aligned_shape)] = (left, right)
            return result, overlaps

        extractor.match_and_merge_lines = traced_merge
        resample = model._resample_image

        def traced_resample(image):
            result = resample(image)
            record({'stage': 'resample', 'input_shape': list(image.shape),
                    'output_shape': list(result.shape)})
            return result

        model._resample_image = traced_resample
        align = model._align_signal_grid_only

        def traced_align(signal, grid, points):
            result = align(signal, grid, points)
            model._research_aligned_shape = tuple(result[0].squeeze().shape)
            record({'stage': 'perspective', 'input_shape': list(grid.shape),
                    'output_shape': list(result[1].shape),
                    'source_points': points.detach().cpu().tolist()})
            np.save(destination / 'grid_before_perspective.npy', grid.squeeze().detach().cpu().numpy())
            return result

        model._align_signal_grid_only = traced_align

        def calibration_hook(module, inputs, result):
            number = sum(e['stage'] == 'calibration' for e in events)
            grid = inputs[0].squeeze().detach().cpu().numpy()
            np.save(destination / f'grid_calibration_{number}.npy', grid)
            record({'stage': 'calibration', 'index': number, 'shape': list(grid.shape),
                    'mm_per_pixel_x': float(result[0]), 'mm_per_pixel_y': float(result[1])})

        model.pixel_size_finder.register_forward_hook(calibration_hook)
        return model

    worker._load_digitizer = traced_loader
    worker.main()


if __name__ == '__main__':
    main()
