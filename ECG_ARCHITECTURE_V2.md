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

A delineator P fiducial is only a candidate. PR/P duration are publishable only
after a separate atrial reproducibility gate confirms discrete P activity
coupled consistently to QRS complexes. Long native rhythm strips with many QRS
and little/no P-QRS coupling are explicit negative evidence against publishing
PR.

### `ecg_signal_integrity.py`
Audits the reconstructed canonical signal before interpretation. It quantifies
finite/observed/interpolated coverage, longest contiguous usable support,
near-flat signals and per-lead eligibility for interval, morphology and rhythm
analysis. It never fills missing signal and is fail-closed.

### `ecg_measurement_consensus.py`
Adds an independent measurement-verification layer around the existing MEDCALC
measurement engine. Native R peaks are cross-checked with WFDB XQRS, while PR,
QRS, QT and P-duration values receive cross-lead dispersion/discordance audits.
The existing engine remains canonical; disagreement requests remeasurement
rather than silently replacing a value.

### `ecg_feature_graph.py`
Builds the shared evidence representation consumed by specialist reasoning.
Global measurements, per-lead morphology, atrial evidence, rhythm, conduction
and measurement QA are represented together without allowing downstream modules
to mutate the canonical numeric measurements.

### `ecg_crosslead_conduction.py`
Synthesizes conduction evidence across leads. Complete bundle-branch patterns
require QRS duration plus compatible cross-lead morphology; QRS width alone is
never sufficient. The existing LAFB/HBAI morphology gate is carried forward as
a specialist finding.

### `ecg_consistency_engine.py`
Checks mutually incompatible findings before publication, including AF versus
stable P-QRS coupling, sinus labels without reproducible P waves, contradictory
bundle morphology, conduction labels dependent on a discordant QRS measurement,
and rhythm interpretation sourced from a lead that fails signal-integrity QA.
Blocking contradictions suppress the corresponding interpretation.

### `ecg_reasoner.py`
Evidence-constrained specialist reasoner. It selects only among hypotheses
already produced by the atrial, rhythm, WCT and conduction layers. It cannot
change measured values and does not use an LLM for clinical arbitration. An LLM,
if used later, is limited to wording after structured reasoning is complete.

### `ecg_atrial_rhythm.py`
Analyzes atrial activity directly from native calibrated digital leads, with
DII and V1 preferred. It performs narrow ventricular-template cancellation and
uses 3-10 Hz spectral organization, dominant frequency, autocorrelation
periodicity, cross-lead frequency consistency, RR behavior and P-QRS evidence.
Its conservative research outputs are `AF_COMPATIBLE`,
`FLUTTER_OR_AT_COMPATIBLE`, `OTHER_SVT_COMPATIBLE` or an indeterminate
atrial mechanism. Compatibility scores are not calibrated probabilities.

### `ecg_wide_complex_tachycardia.py`
Activates only for measured tachycardia with QRS >=120 ms. It evaluates
ventricular vs supraventricular wide-complex mechanisms using direct digital
morphology: precordial concordance, RS presence/interval, aVR and limb-lead
morphology, II/aVR time to first major deflection, AV dissociation support,
capture/fusion candidates, QRS stability and coarse BBB morphology. It returns
`VT_COMPATIBLE`, `SVT_ABERRANCY_OR_PREEXCITATION_COMPATIBLE` or
`WIDE_COMPLEX_TACHYCARDIA_UNDETERMINED`; no single criterion is treated as a
definitive diagnosis.

### `ecg_signal_report_adapter.py`
Maps V2 numeric measurements to the existing report/UI contract while keeping
the reconstructed-signal evidence panels. RR regularity is a ventricular timing
descriptor, not a rhythm mechanism diagnosis. A sinus-compatible label requires
reproducible atrial evidence; wide-complex tachycardia analysis takes priority
over atrial labels when its activation gate is met. Legacy image/model labels
cannot replace signal measurements.

### `ecg_machine_header.py`
Reads printed machine values for comparison/audit. Printed speed/gain may be
reported as discordance evidence, but they do not control clinical calibration:
MEDCALC's acquisition protocol is fixed at 25 mm/s and 10 mm/mV.

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

MEDCALC acquisition invariant: every uploaded ECG is treated as 25 mm/s and
10 mm/mV. These are protocol-known values, not assumptions, and therefore do
not reduce calibration confidence. OCR speed/gain remain audit-only.

## Layout-aware expected duration and coverage

Clinical coverage is the longest contiguous usable supported duration divided
by the duration physically expected for each lead in the selected layout, not
the total scattered sample count and not the 10 s legacy/R27 compatibility
matrix.

- 6x2: 5 s expected per lead; with a native rhythm strip, lead II expects 10 s.
- 3x4: 2.5 s expected per lead; with a native rhythm strip, lead II expects 10 s.
- 12x1: 10 s expected per lead.

Therefore, a fully recovered 5 s lead in a 6x2 ECG is 100% clinically covered,
not 50%. The historical 10 s matrix remains a compatibility adapter only.

## Missing data

Large gaps stay missing. No clinical component may fill an unobserved interval
merely to satisfy a model input. Only short bounded gaps are interpolated and
are marked separately in the quality mask.

R27-TILED is a research compatibility path only and never becomes the source of
rhythm or morphology measurements.

## Validation status and targets

MEDCALC currently has **internal engineering validation / regression testing**, not
external clinical validation of the complete ECG interpretation pipeline.

The synthetic harness creates deterministic digital ECGs with known fiducials so
paper rendering, reconstruction, calibration, layout handling and numeric recovery
can be tested end-to-end. It is intentionally **not a physiological simulator** and
must not be described as proof of clinical diagnostic accuracy on heterogeneous
real-world ECGs.

Real adjudicated ECGs are a separate validation layer. ECG_05 (clinician-adjudicated
AF) and ECG_10 (clinician-adjudicated sinus rhythm with isolated LAFB/HBAI) are being
used as initial real-case regression evidence, but two cases do not constitute
external validation.

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

## Rhythm semantics

RR regularity and rhythm mechanism are separate outputs:

1. RR statistics describe ventricular timing only.
2. P-wave reproducibility determines whether PR/P duration can be reported and
   whether sinus compatibility can be considered.
3. Native atrial analysis characterizes AF-compatible vs organized flutter/AT
   vs other SVT patterns.
4. If HR >=100 bpm and measured QRS >=120 ms, the wide-complex tachycardia
   analyzer evaluates VT vs wide SVT before a supraventricular mechanism is
   promoted in the report.
5. R27 remains an independent probability-only research profile.

No rule equates "regular RR" with sinus rhythm, "no P" with AF, or "wide QRS"
with VT.

## High-fidelity performance policy

The stable primary route remains 2000 px high-fidelity segmentation with
dewarping and probability-weighted subpixel centerlines. The row fallback
already computes a probability-weighted vertical centroid rather than rounding
the trace to integer pixels.

The 1200 px route is a rescue/reference pass, not a mandatory second inference.
It is skipped when the primary route already passes strong layout, continuity,
rhythm-strip and R27-readiness gates.

CPU inference uses four Torch intra-op threads by default
(`MEDCALC_ECG_TORCH_THREADS=4`, bounded to 1-4). On the same real-photo smoke
case, the unchanged 2000 px route reduced primary model load+inference from
131.716 s at one thread to 76.436 s at two threads and then to 33.695 s at four
threads. Worker total time fell from 135.628 s to 81.349 s and then to 36.574 s.
At four threads the selected layout remained 6x2 with post-U-Net score 0.996666,
and the adaptive 1200 px reference remained safely skipped. No model weights,
resolution, dewarping, calibration or clinical algorithms changed.

A direct full-page increase from 2000 to 3000 px is not approved. In the
2026-09-27 synthetic benchmark, the 3000 px route lost the correct 6x2 layout
on one of two benchmark cases; on the other it recovered 16.7% of leads versus
25.0% at 2000 px, with no meaningful QRS advantage and worse ST/amplitude
error. The corresponding clean 6x2 case at 2000 px retained the correct layout
and recovered 58.3% of leads.

Any future 3000 px work must therefore be selective ROI/uncertainty refinement
after a trusted 2000 px solution. It must preserve the 2000 px layout and
calibration as the authoritative geometry, demonstrate improved per-lead QC,
and fail back to the 2000 px signal if refinement does not improve evidence.