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
