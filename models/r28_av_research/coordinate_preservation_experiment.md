# Coordinate-preservation experiment

A traced end-to-end replay of fixed development image 2188 isolates the time-axis error. Turning dewarping off changes the last-sample duration only from 10.716 s to 10.692 s. Independent major-grid spacing on the aligned probability map agrees with the reported x calibration within approximately 0.12%.

The extractor crops its merged centerlines to observed columns, discarding the original horizontal offset. The current geometry adapter stretches these cropped lines onto `active_x`, which was obtained by scaling preflight coordinates by canvas dimensions without accounting for perspective/dewarping. On image 2188, 1732 extractor columns are stretched into a 1855-column ROI. This is a coordinate-system mismatch, not primarily a pixel-size calibration error.

`ecg_av_trace_calibration.py` records resampling, perspective points, grid calibration, and extractor crop bounds. With `ECG_RESEARCH_PRESERVE_X=1`, it restores the exact original column offsets using missing-value padding and supplies the aligned extractor bounds to the geometry adapter. It does not force a 10-second target, resample centerlines, fill gaps, change grid calibration, retrain weights, or change thresholds. This is an explicit research process; the normal worker is unchanged.

Reproduce (from the repository root):

```bash
ECG_CALIBRATION_TRACE=/tmp/ekg-baseline python ecg_av_trace_calibration.py --vendor-root ecg_digitizer_vendor --segmentation-model ecg_digitizer_assets/unet_weights_07072025.pt --lead-model ecg_digitizer_assets/lead_name_unet_weights_07072025.pt --source models/r28_av_research/prepared_images/ptbxl-02188-3x4-strip.png --output-root /tmp/ekg-baseline --meta /tmp/ekg-baseline/meta.json
```

Repeat with `MEDCALC_ECG_ENABLE_DEWARP=0` for the dewarping ablation, or `ECG_RESEARCH_PRESERVE_X=1` for the coordinate-restoration experiment, using separate output/trace directories. The fixed additional images are 00070 and 00484. Compact evidence and output hashes are recorded in `coordinate_preservation_report.json`.

These are rendered real development signals, not clinical photographs. Passing the duration check does not establish P/QRS accuracy, AV block performance, clinical readiness, or verified annotation alignment. Full perspective-aware transport of all lead/row boundaries and broader layout/photo regression remain necessary before promoting the candidate into the clinical path. Segmentation gaps remain observable and can independently block R28 input.

## Fixed image results

| ECG | Original extent (s) | Corrected extent (s) | Longest observed II (s) | Input window guard |
|---|---:|---:|---:|---|
| 70 | 10.322 | 10.004 | 5.648 | FAIL: continuity |
| 2188 | 10.718 | 10.006 | 9.988 | PASS |
| 484 | 10.348 | 10.008 | 7.112 | PASS |

Extent uses sample count / sampling frequency consistently; last-sample timestamps are 2 ms shorter at 500 Hz. The primary full replay and dewarping ablation were rerun for ECG 2188. Baselines for 70 and 484 are the previously committed frozen digitization report, and both corrected runs were executed in this experiment. All three corrected outputs have no lead exceeding its known display-duration tolerance. All 53 research tests pass locally.
