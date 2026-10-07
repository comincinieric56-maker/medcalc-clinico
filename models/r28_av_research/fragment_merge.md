# Preserve short waveform fragments before graph matching

The extractor rejected every candidate with 30 or fewer observed columns before matching. Steep QRS trajectories can produce short disconnected segmentation components, so this removed genuine observed pixels before they could join the rest of their row. The minimum-width filter now runs on merged rows. The existing graph cost and component coverage filters are unchanged. Isolated short components remain rejected; missing columns remain NaN. Empty results clear their coordinate bounds.

The final source hashes in `fragment_merge_report.json` match the integrated extractor and worker. All three fixed rendered PTB-XL development images completed the normal high-fidelity worker. These are development engineering cases, not independent clinical validation.

| ECG | Previous longest observed II (s) | After fragment fix (s) | Time extent (s) |
|---|---:|---:|---:|
| 70 | 5.648 | 8.016 | 10.004 |
| 2188 | 9.988 | 9.988 | 10.006 |
| 484 | 7.112 | 9.994 | 10.008 |

The 106 ms internal extraction loss in ECG 70 is recovered without interpolation. Its 146 ms loss at 1.826–1.972 s remains missing. An offline cached-map experiment recovered additional fragments, but the normal worker did not reproduce that result; the table reports only the normal worker. Passing the six-second input-window guard means an observed research input exists, not that the waveform or diagnosis is validated.

The previous interrupted attempt at ECG 484 failed reading an empty image before inference. The final complete replay supersedes that attempt. Reported model hashes are unchanged. A later workspace inspection found a truncated local lead-name checkpoint; the exact tracked model was restored after verifying its hash against all three completed worker outputs.

Validation: 65 research/coordinate/fragment contract tests and five existing synthetic known-case scenarios passed. The fragment tests exercise real graph matching, preservation of exact short-fragment values, retained gaps, rejection of isolated short pieces, and cleared crop state. The measurement CI path filter now includes vendor extractor and layout changes.

Reproduce:

```bash
python ecg_coordinate_integration_replay.py --work-dir /tmp/ecg-fragment --output models/r28_av_research/fragment_merge_report.json
python ecg_module_integration_audit.py --work-dir /tmp/ecg-fragment --replay-report models/r28_av_research/fragment_merge_report.json --output models/r28_av_research/module_integration_report.json
python ecg_measurement_v2_development_benchmark.py native --output models/r28_av_research/measurement_native_status.json
```

## Remaining blockers

`module_integration_report.json` verifies execution of image-derived measurement and guarded AV research on all three replay outputs, checking their metadata hashes and extractor source hashes first. All three complete without an AV execution error. PR, QRS, QT and P duration are measured with uncertainty for ECGs 70 and 2188. ECG 484 has unmeasurable PR and P duration; QRS and QT retain uncertainty. All AV results remain `RESEARCH_ONLY`, abstaining, and excluded from clinical fusion. This is not an accuracy assessment.

- Extraction: the 146 ms loss remains; broader photographed-image and layout regressions are needed.
- Measurements: the native synthetic reference run has QRS absolute errors of 26, 14 and 16 ms at 50, 75 and 120 bpm. PR errors are 21 and 14 ms at 50 and 75 bpm; PR and QT are unmeasurable at 120 bpm. Execution success is not interval accuracy.
- P/QRS and AV: the existing frozen real-signal probes still show false P detections and failed AVB2/AVB3 transfer. No model weights or clinical thresholds were changed here. R28 remains abstaining and excluded from clinical fusion.
- Data: expert image annotations with verified time/lead alignment remain unavailable for the guarded real-image training loader. No annotations were inferred or manufactured.
- Release: the correction is in the development branch; this work does not merge or deploy it.
