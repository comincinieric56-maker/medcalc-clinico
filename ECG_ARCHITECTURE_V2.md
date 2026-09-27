# MEDCALC ECG Architecture V2 — calibrated digital signal first

## Invariant

The clinical ECG analyzer consumes a calibrated reconstructed digital signal.
The raster/photo/PDF is an acquisition source and an audit reference; it is not
the primary source for interval, rhythm, amplitude, ST or morphology
measurements once reconstruction passes quality control.

The priority rule is:

```
reliable numeric measurement on calibrated digital signal
    > image/model classification
```

R27 remains frozen, probability-only and secondary. A model score never
overrides a reliable signed numeric measurement such as ST displacement.

## Pipeline

```
source photo/PDF
  -> source ROI / orientation
  -> perspective correction
  -> grid scale
  -> speed + gain
  -> U-Net segmentation
  -> centerline extraction
  -> layout/lead ROI assignment
  -> x/y pixels -> mm
  -> mm -> seconds / mV
  -> small-gap repair only
  -> fixed-rate resampling (500 Hz)
  -> per-sample quality mask
  -> canonical per-lead digital signal
  -> fiducials / beats
  -> numeric measurements
  -> rhythm + morphology interpretation
  -> report / audit
  -> optional frozen R27 probability layer
```

## Canonical signal contract

Every supported layout becomes the same object after ROI assignment:

```python
ecg["II"] = {
    "signal_mv": [...],
    "quality_mask": [...],  # 0 missing, 1 short-gap interpolation, 2 observed
    "fs": 500,
    "duration_s": 5.0,
    "source": "digitized",
    "confidence": 0.97,
    "status": "MEASURED",
}
```

The same keys exist for I, II, III, aVR, aVL, aVF and V1-V6. Layout is not a
clinical-analysis input after reconstruction.

## Modules

### `ecg_unet_worker.py`
Orchestrates the source raster, remote high-fidelity U-Net, dewarping,
post-U-Net layout routing, calibrated reconstruction, clinical measurement and
the optional R27 compatibility adapter.

### `ecg_layout_detector.py`
Geometry/layout and post-segmentation row routing. Supports 3x4, 6x2 and 12x1.
Its output assigns physical row/column ROIs only; it does not define clinical
time/amplitude measurements.

### `ecg_signal_reconstruction.py`
Owns the physical signal contract. It uses independent x/y grid scale, speed
and gain, converts centerline pixels to mV and seconds, repairs only short
internal gaps, rejects isolated impossible centerline spikes, resamples to
500 Hz and preserves a per-sample quality mask.

### `ecg_signal_measurements.py`
Owns all primary clinical numeric measurements from the digital signal:
R peaks/RR, HR, RR variability, P/QRS/T fiducials, P duration, PR, QRS, QT,
QTc, J/ST J+40/J+60/J+80, R/S/Q/T/P amplitudes, R/S ratio, Q duration,
T polarity, axis, R progression and descriptive voltage metrics.

### `ecg_signal_report_adapter.py`
Maps V2 numeric measurements to the existing report/UI contract while keeping
the reconstructed-signal evidence panels. It prevents legacy image/model labels
from replacing signal measurements.

### `ecg_machine_header.py`
Reads printed machine values for comparison and, when valid, speed/gain for
physical calibration. OCR calibration is snapped only to plausible standard
paper values; invalid/zero OCR values are rejected.

### `ecg_report_pdf.py`
Presentation/audit layer. It receives structured signal measurements and the
source image; it must not recalculate clinical values from the rendered report.

### `ecg_unet_r27_bridge.py`
Remote-job transport. It forwards detected speed/gain to the private runner.

### `ecg_validation_harness.py`
Creates digital ECG ground truth, renders standard paper layouts, applies
controlled degradation and scores recovered measurements.

## Calibration

For paper speed `v` in mm/s and gain `g` in mm/mV:

```
time_s = x_pixels * mm_per_pixel_x / v
amplitude_mV = -(y_pixels - baseline_y_pixels) * mm_per_pixel_y / g
```

At 25 mm/s, 1 mm = 40 ms. At 10 mm/mV, 1 mm = 0.1 mV.

Grid scale x and y are retained independently so residual anisotropy after
perspective correction is visible instead of silently averaged away.

If printed speed/gain cannot be recovered, compatibility defaults
25 mm/s / 10 mm/mV are explicitly marked as assumptions and reduce confidence.

## Missing data

Large gaps stay missing. No clinical component may fill an unobserved interval
merely to satisfy a model input. Only short bounded gaps are interpolated and
are marked separately in the quality mask.

R27-TILED is a research compatibility path only and never becomes the source of
rhythm or morphology measurements.

## Validation targets

The validation harness reports, where ground truth is available:

- MAE_QRS
- MAE_PR
- MAE_QT
- MAE_ST
- ERROR_RR
- ERROR_AMPLITUDE
- ERROR_FC
- percentage of leads recovered

The synthetic matrix spans 6x2, 3x4, 3x4 with rhythm strip and 12x1, plus
rotation, perspective, blur/noise, JPEG/low resolution, thick traces, red,
green, gray/no grid and text overlays.

The long-term acceptance criterion is dataset-level performance across the
matrix, not correction of individual ECG examples.