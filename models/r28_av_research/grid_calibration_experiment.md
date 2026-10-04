# Grid calibration experiment — rejected candidate

The vendor reverses the negative autocorrelation prefix without zero lag. Its first index therefore represents lag one, while the periodic score assumes lag zero.

A research-only experiment supplies the nonnegative half, including zero lag, to the unchanged score and search. It does not rescale ECG samples or use known duration as a target.

Controlled profiles: 1600 pixels, major line amplitude 1, minor line amplitude 0.5, five minor intervals per major interval; default vendor search settings.

| Known major interval (px) | Original estimate (px) | Candidate estimate (px) |
|---|---|---|
| 20 | 19.964117 | 39.987438 |
| 30 | 29.944031 | 29.990601 |
| 40 | 39.927887 | 19.993719 |

The indexing defect is real, but the direct correction exposes harmonic ambiguity in the scoring function. It cannot safely fix the prepared-image duration failures. Candidate rejected before image inference or clinical deployment. Tests preserve this failure as a regression case; these idealized profiles do not validate real-image calibration.

Clinical worker, vendor implementation, weights, and thresholds remain unchanged. The next calibration candidate must distinguish major/minor grid harmonics and pass known-scale grids before testing the three fixed development images. No training input or annotation alignment is approved by this experiment.

## Independent raw red-grid audit

`ecg_av_red_grid_audit.py` finds minor lines in median red-color profiles and verifies five-line major modulation using integrated line width and intensity. It abstains for uniform lines, absent minors, insufficient lines, and irregular spacing. It supports axis-aligned red grids only; it is not a general photographed-ECG calibrator.

All three fixed prepared images recover 4 pixels/mm (0.25 mm/pixel) on both original-image axes. This agrees with rendering metadata without using it as input. The major/minor integrated contrast is 2.0. Results are in `red_grid_audit.json`. No original-image scalar is applied after perspective correction or dewarping: those transformations must be measured in their own coordinates. The digitized duration failure remains unresolved.
