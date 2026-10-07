# Integrated aligned-coordinate contract

This records the coordinate-only baseline. The subsequent short-fragment correction and complete replay are documented in `fragment_merge.md` and `fragment_merge_report.json`.

The segmentation-only worker previously stretched cropped centerlines onto an ROI derived from preflight canvas dimensions. The crop offset and aligned-image bounds are now carried explicitly from the extractor to the geometry adapter. The wrapper pads cropped lines back into their original columns; it never resamples these trajectories. The neural layout identifier continues to receive its original cropped input.

`SignalExtractor.last_crop_bounds` resets for every input and records the inclusive bounds actually used by `preprocess_lines`. `restore_aligned_lines` checks those bounds and the exact cropped width, preserving values and NaN gaps. The segmentation-only result exposes `ALIGNED_CANVAS_PIXELS` and `aligned_active_x`. Guided layout recovery uses these bounds before calculating support, and canonical reconstruction uses the same coordinate system.

Validation consists of the research/coordinate test suite, five existing synthetic known-case engineering regressions, and three fixed rendered PTB-XL development images passed directly through `ecg_unet_worker.py` without research monkeypatches. The regression scenarios are not diagnostic validation. See `coordinate_integration_report.json` for end-to-end outcomes and hashes. No held-out dataset was used and no weights or diagnostic thresholds were tuned.

The normal worker in this branch now uses the correction. The PR has not been merged or deployed. R28 remains research-only and abstaining. Broader photographed-image and layout validation, plus verification of diagnostic behavior after the corrected time scale, remain required before a clinical release. Vertical row priors still use local recentering; this change addresses the proven horizontal stretch defect.

Reproduce the fixed-image replay and save a checkpoint after each case:

```bash
python ecg_coordinate_integration_replay.py --work-dir /tmp/ekg-integration --output models/r28_av_research/coordinate_integration_report.json
```

The old `ECG_RESEARCH_PRESERVE_X` hook is unnecessary with this integrated vendor contract; the tracer avoids applying restoration twice.

## Normal-worker replay results

| ECG | Extent (s) | Longest observed II (s) | Input-window guard |
|---|---:|---:|---|
| 70 | 10.004 | 5.648 | FAIL: continuity |
| 2188 | 10.006 | 9.988 | PASS |
| 484 | 10.008 | 7.112 | PASS |

The normal worker reproduces the experimental duration correction on all three fixed images. ECG 70 has internal unobserved spans 1.826–1.972 s (146 ms) and 7.620–7.726 s (106 ms), plus short edge gaps. The source image visibly contains QRS complexes in these regions: these are extraction losses, not demonstrated physiological pauses. They remain missing and are not interpolated into an accepted R28 window. 62 tests and all five synthetic known-case scenarios passed.
