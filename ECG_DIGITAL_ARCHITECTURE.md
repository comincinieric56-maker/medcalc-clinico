# MEDCALC ECG Digital-Signal Architecture

## Contract

The clinical ECG analyzer consumes a canonical digital signal, not the source raster.

```
source ECG
  -> page/ROI and geometric correction
  -> grid / pixel scale
  -> speed + gain evidence
  -> post-U-Net layout / lead mapping
  -> U-Net segmentation
  -> digitizer centerline
  -> pixel -> mm
  -> mm -> ms / mV
  -> gap-aware resampling at 500 Hz
  -> MEDCALC_DIGITAL_ECG_V2
  -> fiducials and numerical measurements
  -> rhythm and morphology
  -> interpretation/report
```

The source image, reconstructed-paper image and overlay are audit surfaces only. They are not fed back into the clinical measurement engine.

## Canonical per-lead representation

Every standard lead is represented independently of the original page layout:

```python
digital_ecg["leads"]["II"] = {
    "signal_mv": [...],
    "time_ms": [...],
    "fs": 500,
    "duration_s": 10.0,
    "source": "U_NET_DIGITIZER_CENTERLINE_LONG_RHYTHM",
    "confidence": 0.97,
    "observed_mask": [...],
    "interpolated_mask": [...],
    "low_confidence_mask": [...],
    "confidence_mask": [...],
}
```

Layouts 3x4, 6x2 and 12x1 are reconstruction adapters. After lead mapping they do not participate in clinical measurement logic. The neural/Open-ECG fallback is wrapped in the same schema for compatibility with additional layouts.

## Physical calibration

Pixel spacing comes from the digitizer grid detector. Speed and gain are resolved from independent scale evidence:

- printed machine setting when OCR can read it;
- calibration pulse when detected;
- standard 25 mm/s and 10 mm/mV only as an explicitly low-confidence compatibility fallback.

Quantitative ms/mV values fail closed below the calibration-confidence gate. An assumed standard scale therefore cannot silently become a high-confidence clinical measurement.

## Signal reconstruction

The reconstruction layer:

1. takes U-Net/digitizer physical centerlines;
2. rejects isolated impossible centerline spikes without flattening broad QRS excursions;
3. interpolates only short internal gaps;
4. leaves unrecoverable regions as NaN;
5. converts x-pixel to time from pixel spacing and paper speed;
6. converts y-pixel to mV with the image-y-down -> ECG-mV-up sign convention;
7. resamples each lead to 500 Hz;
8. preserves observed, interpolated, low-confidence and continuous confidence masks.

## Measurement precedence

`NUMERIC_DIGITAL_SIGNAL_GT_CLASSIFIER`

The deterministic digital measurement layer is authoritative for measurable quantities. Classifier scores may support interpretation but cannot reverse the sign or value of a reliable numerical measurement. Example: a positive R27 ST_ELEVATION score does not make a measured -0.14 mV J+60 value into ST elevation.

R27 remains frozen and probability-only. R27-TILED remains research-only.

## Measurements

The digital engine calculates, with confidence and source traceability:

- R peaks, RR series, mean/SD/CV/MAD/RMSSD/pNN50, heart rate and regularity;
- P duration/amplitude and reproducibility before QRS;
- PR;
- QRS onset/offset/duration;
- QT and QTc (Bazett and Fridericia);
- J, J+40, J+60 and J+80 relative to PR/TP baseline when available;
- signed ST deviation and elevation/depression direction;
- R/S/Q amplitudes, R/S ratio, Q duration;
- T amplitude/polarity;
- digital QRS net area / axis;
- R progression, voltage summaries and Q-wave candidates.

Unreliable measurements remain unavailable instead of being inferred from an image label.

## Audit

The result includes:

- source ECG preview;
- standardized reconstructed ECG drawn from digital arrays;
- U-Net/digitizer centerline overlay in rectified segmentation coordinates;
- calibration provenance;
- per-lead masks/confidence;
- fiducial positions and baseline source.

The current overlay is explicitly a rectified-segmentation-space audit. It is not labeled as a pixel-perfect overlay on the unrectified source raster.

## Validation

`ecg_validation_harness.py` creates digital ground-truth ECGs, renders them on paper and applies layout/scan distortions. `ecg_validation_runner.py` passes those rendered cases through the same production U-Net worker.

The validation matrix includes 6x2, 3x4, 3x4+rhythm, 12x1, red/green/gray/no grid, rotation, perspective, blur, noise, JPEG compression, reduced resolution, thick trace, grayscale scan and text overlay.

Tracked metrics include QRS/PR/QT/ST MAE, RR/heart-rate error, waveform-amplitude MAE and recovered-lead percentage.

## Compatibility boundary

The historical 5000x12 NaN-masked matrix is generated only by `pack_legacy_10s_uv()` for WFDB/R27 and legacy surfaces. New clinical measurements must not depend on that layout-shaped compatibility matrix.
