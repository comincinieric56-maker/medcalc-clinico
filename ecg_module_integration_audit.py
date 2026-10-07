"""Exercise measurements and guarded AV research on fixed digitized images.

This reports execution and missing measurements, never diagnostic accuracy.
Input metadata must match the image replay report to avoid stale evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

from ecg_av_research_adapter import analyze_ecg_with_av_research
from ecg_av_temporal_model import AVResearchModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--replay-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    replay = json.loads(args.replay_report.read_text())
    if (len(replay['outputs']) != 3 or
            {r['ecg_id'] for r in replay['outputs']} != {70, 2188, 484}):
        raise ValueError('INCOMPLETE_FIXED_IMAGE_REPLAY')
    for path, expected in replay['source_sha256'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError('REPLAY_SOURCE_HASH_MISMATCH')
    model = AVResearchModel(root / 'models/r28_av_research/r28_av_research.pt')
    report = {'role': 'DEVELOPMENT_MODULE_EXECUTION_AUDIT',
              'clinical_ready': False, 'accuracy_validated': False,
              'training_allowed': False, 'av_model_sha256': model.sha256,
              'replay_sha256': hashlib.sha256(args.replay_report.read_bytes()).hexdigest(),
              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(root.glob('ecg_*.py'))},
              'outputs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for row in replay['outputs']:
        raw = (args.work_dir / str(row['ecg_id']) / 'meta.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['worker_meta_sha256']:
            raise ValueError('WORKER_METADATA_HASH_MISMATCH')
        meta = json.loads(raw)
        result = analyze_ecg_with_av_research(
            meta['signal']['calibrated_digital_signal'], av_research_model=model)
        av = result['av_research']
        if (av.get('abstain') is not True or
                av.get('clinical_fusion_allowed') is not False or
                av.get('diagnostic_claim_allowed') is not False):
            raise ValueError('AV_RESEARCH_GUARD_VIOLATION')
        if av['status'] == 'RESEARCH_ERROR':
            raise RuntimeError(f"AV_EXECUTION_FAILED:{av.get('reason')}")
        consensus = result.get('measurement_consensus') or {}
        report['outputs'].append({
            'ecg_id': row['ecg_id'], 'execution_completed': True,
            'measurement_version': result['version'],
            'global_measurements': result['global'],
            'measurement_states': consensus.get('measurement_states'),
            'remeasure_targets': consensus.get('remeasure_targets'),
            'unmeasurable_targets': consensus.get('unmeasurable_targets'),
            'av_research': {k: v for k, v in av.items() if k not in ('graph', 'probabilities')},
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(row['ecg_id'], av['status'], consensus.get('measurement_states'), flush=True)


if __name__ == '__main__':
    main()
