"""Replay the three fixed development images through the normal worker.

Writes a compact checkpoint after each image; never treats this as clinical
validation or creates training annotations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from ecg_av_image_preparation_audit import inspect_digitized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    images = root / 'models/r28_av_research/prepared_images'
    manifest = json.loads((images / 'image_preparation_manifest.json').read_text())
    report = {'role': 'DEVELOPMENT_IMAGE_ENGINEERING_REPLAY', 'clinical_ready': False,
              'training_allowed': False, 'worker': 'ecg_unet_worker.py', 'outputs': []}
    sources = ['ecg_unet_worker.py', 'ecg_layout_detector.py',
               'ecg_digitizer_vendor/src/model/inference_wrapper.py',
               'ecg_digitizer_vendor/src/model/signal_extractor.py',
               'ecg_digitizer_vendor/src/model/coordinate_contract.py']
    report['source_sha256'] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                               for path in sources}
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for ecg_id in [70, 2188, 484]:
        directory = args.work_dir / str(ecg_id)
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / 'meta.json'
        image = images / f'ptbxl-{ecg_id:05d}-3x4-strip.png'
        expected = next(r['image_sha256'] for r in manifest['records'] if r['ecg_id'] == ecg_id)
        if hashlib.sha256(image.read_bytes()).hexdigest() != expected:
            raise ValueError('FIXED_IMAGE_HASH_MISMATCH')
        command = [sys.executable, str(root / 'ecg_unet_worker.py'),
                   '--vendor-root', str(root / 'ecg_digitizer_vendor'),
                   '--segmentation-model', str(root / 'ecg_digitizer_assets/unet_weights_07072025.pt'),
                   '--lead-model', str(root / 'ecg_digitizer_assets/lead_name_unet_weights_07072025.pt'),
                   '--source', str(image), '--output-root', str(directory), '--meta', str(meta_path)]
        with (directory / 'worker.log').open('w') as log:
            completed = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        if completed.returncode:
            raise RuntimeError(f'WORKER_FAILED:{ecg_id}:see {directory / "worker.log"}')
        raw = meta_path.read_bytes()
        meta = json.loads(raw)
        audit = inspect_digitized(meta, manifest)
        ii = meta['signal']['calibrated_digital_signal']['leads']['II']
        quality = np.asarray(ii['quality_mask'])
        edges = np.diff(np.r_[False, quality != 2, False].astype(int))
        gaps = [{'start_s': float(start / ii['fs']), 'end_s': float(end / ii['fs']),
                 'duration_ms': float((end-start) * 1000 / ii['fs']),
                 'quality_codes': sorted(set(int(v) for v in quality[start:end]))}
                for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]
        report['outputs'].append({'ecg_id': ecg_id, 'image_sha256': expected,
            'worker_meta_sha256': hashlib.sha256(raw).hexdigest(),
            'segmentation_model_sha256': meta['segmentation_model_sha256'],
            'lead_model_sha256': meta['lead_model_sha256'],
            'geometry': meta['signal']['signal_geometry'],
            'ii_not_observed_intervals': gaps, **audit})
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(ecg_id, audit['canonical_ii_time_extent_s'], audit['blockers'], flush=True)


if __name__ == '__main__':
    main()
