"""Observe row extraction without modifying it; accepts normal worker arguments.

Set ECG_GAP_PROBE_DIR to save the first guided high-resolution map and rows.
"""
import json
import os
from pathlib import Path

import numpy as np


def main():
    import ecg_unet_worker as worker
    destination = Path(os.environ['ECG_GAP_PROBE_DIR'])
    destination.mkdir(parents=True, exist_ok=True)
    original = worker.build_rows_from_signal_probability
    captured = False

    def observe(probability, raw_lines, geometry):
        nonlocal captured
        result = original(probability, raw_lines, geometry)
        if not captured:
            raw = raw_lines.detach().cpu().numpy() if hasattr(raw_lines, 'detach') else np.asarray(raw_lines)
            np.savez_compressed(destination / 'rows.npz', probability=probability,
                                raw_lines=raw, selected_rows=result[0])
            (destination / 'geometry.json').write_text(json.dumps(
                {'geometry': geometry, 'sources': result[1], 'debug': result[2]}, indent=2) + '\n')
            captured = True
        return result

    worker.build_rows_from_signal_probability = observe
    worker.main()


if __name__ == '__main__':
    main()
