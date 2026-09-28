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
QTc (Bazett, Fridericia, Framingham and Hodges), JT/JTc, cross-lead QRS/QT
dispersion, J/ST J+40/J+60/J+80, R/S/Q/T/P amplitudes, R/S ratio, Q duration,
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
The existing engine remains canonical and is never silently overwritten.

V2 separates four measurement states: `MEASURED_HIGH_CONFIDENCE`,
`MEASURED_WITH_UNCERTAINTY`, `REMEASURE_REQUIRED` and `UNMEASURABLE`.
Moderate detector/cross-lead disagreement propagates an uncertainty interval
instead of automatically turning the entire ECG into a remeasurement case.
Only strong contradictory evidence requests remeasurement. Threshold-sensitive
diagnoses consume the uncertainty interval, so a QRS or PR interval that
overlaps a clinical boundary cannot be promoted as definitively above/below it.

The reconstruction layer also exports the physical one-pixel resolution implied
by grid spacing, 25 mm/s and 10 mm/mV. This supplies a minimum timing/amplitude
engineering uncertainty that measurement consensus combines with cross-lead
dispersion. It is an engineering bound, not a statistical confidence interval.

### `ecg_feature_graph.py`
Builds the shared evidence representation consumed by specialist reasoning.
Global measurements, per-lead morphology, atrial evidence, rhythm, conduction
and measurement QA are represented together without allowing downstream modules
to mutate the canonical numeric measurements.

### `ecg_qrs_morphology.py`
Builds a median digital QRS morphology per lead from aligned high-quality beats.
It extracts R-prime/notching, terminal R/S support, lateral initial q waves,
R-peak time and an initial-slur descriptor. These features are descriptive and
feed the conduction/preexcitation specialists; they never replace the canonical
QRS duration.

### `ecg_ectopy.py`
Classifies premature-beat patterns using RR prematurity, compensatory pauses and
QRS width/amplitude outliers on the same accepted beats. Its main architectural
role is to prevent ventricular/supraventricular ectopy from being mistaken for
AF solely because RR intervals are irregular.

### `ecg_av_conduction.py`
Tracks reproducible P sequences against QRS complexes. It implements
conservative evidence gates for first-degree AV delay, Mobitz I, Mobitz II, 2:1
and high-grade AV block. First-degree AV delay requires 1:1 conduction plus a
stable median PR >200 ms. Higher-grade labels require observed nonconducted P
waves and a regular atrial sequence.

### `ecg_preexcitation.py`
Requires reproducible P waves, PR <120 ms, QRS prolongation and an initial
delta/slur-compatible morphology in at least two leads before reporting a
preexcitation-compatible pattern. When preexcitation is present, BBB labels are
not promoted because ventricular activation is confounded.

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
Evidence-constrained specialist reasoner and the authoritative structured
interpretation layer. V3 uses domain-specific publication: a contradiction in
QRS conduction cannot silence an otherwise well-supported atrial rhythm, and a
PR problem cannot suppress unrelated rate or bundle evidence. High-recall
candidates may be promoted only after multisource evidence fusion and the
relevant domain gate passes. A ventricular rate below 60 or above 100 bpm is
called sinus bradycardia/tachycardia only when atrial evidence also supports
sinus origin. Numeric measurements remain immutable and an LLM is never used
for clinical arbitration.

### `ecg_candidate_detectors.py`
Prospective high-recall candidate layer. It uses OR-shaped evidence to avoid
serial hard-gate sensitivity collapse for AF/flutter, sinus rate phenotypes,
bundle/fascicular conduction, AV conduction and preexcitation. Candidates are
not diagnoses and cannot bypass downstream fusion.

### `ecg_domain_gating.py`
Maps conflicts and remeasurement requests to the diagnosis domains that actually
depend on them. Abstention is domain-specific rather than global. Measurement
dependencies are code-specific where appropriate: for example, a missing global
PR may block first-degree AV delay but must not automatically block high-grade
AV-block sequence analysis.

### `ecg_evidence_fusion.py`
Combines independent evidence groups using prospective engineering thresholds
declared without SPH tuning. A candidate becomes publishable only when its domain
is eligible, all measurements required by that diagnosis are usable, and the
minimum score/source-count criteria are satisfied.

### `ecg_fn_waterfall.py`
Development-only error-localization utility. For labelled development/regression
data it classifies a false negative as signal, measurement/detection, candidate,
domain-gate, evidence-fusion, reasoner/reporting failure, or true positive.
Frozen external records must never be debugged case by case with this utility.

### `ecg_rhythm_consensus.py`
Separates absolute-rate estimation from RR-mechanism analysis. Heart rate is
robustly aggregated across independent lead windows so a single underdetected
lead cannot create false bradycardia/tachycardia. RR irregularity is never
stitched across non-simultaneous layout windows; it remains a single native
rhythm-lead measurement and is summarized from CV, MAD/median, RMSSD/median and
pNN50.

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

### `ecg_r27_consensus.py`
Treats frozen R27 as an independent probability-only QA specialist. It compares
already-publishable MEDCALC findings with the homologous R27 module and emits
cross-engine support, neutral evidence, discordance-for-review or R27-only
review signals. R27 never originates a clinical diagnosis, never changes a
numeric measurement and never overrides the specialist reasoner. Temporal R27
modules remain non-comparable when the input is R27-TILED.

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

## Post-CODE-test hardening policy

CODE-test and SPH are consumed historical external baselines and are not
eligible for future threshold selection, individual-record debugging or claims
of independent validation for new clinical changes. The V3 high-sensitivity
architecture was designed after observing aggregate SPH failure modes, but its
candidate/fusion thresholds are prospective defaults and are not fitted to SPH.
Development and regression work must use datasets already classified as
development-contaminated. MIMIC-IV-ECG remains a future provisionally locked
external cohort subject to a prespecified label-mapping protocol.

## V3 high-sensitivity invariants

The target is not achieved by repeatedly moving one diagnostic cutoff. V3
separates sensitivity into stages:

```
usable signal
→ measurable features
→ high-recall candidate
→ domain gate
→ evidence fusion
→ authoritative reasoner
→ report
```

Engineering targets are tracked separately as feature coverage, candidate
detector sensitivity and final diagnostic sensitivity. Serial AND-gates are
avoided at the candidate stage; final publication still requires multiple
independent evidence groups. Global abstention is prohibited when only an
unrelated domain is uncertain.

Measurement uncertainty is also diagnosis-specific: an uncertain QT does not
block rhythm/conduction, an unmeasurable PR does not block high-grade AV sequence
analysis, and complete bundle-branch labels require QRS uncertainty to remain
entirely above the 120 ms boundary. First-degree AV delay and short-PR
preexcitation use the same boundary-aware rule for 200 ms and 120 ms,
respectively.

Key hardening invariants:
- heart-rate labels use multilead rate consensus, not one lead alone;
- AF requires absent reproducible P activity plus robust irregularity and/or
  disorganized atrial evidence, with flutter/ectopy exclusions;
- P reproducibility now includes beat-to-beat morphology consistency;
- complete RBBB/LBBB require QRS duration plus characteristic cross-lead
  morphology;
- the specialist reasoner is the authoritative diagnostic output and may
  abstain rather than allowing legacy adapter logic to publish a low-confidence
  label.

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
## External validation lock after CODE-test baseline

CODE-test was consumed on 2026-09-27 as the first frozen external baseline. Its
aggregate results identified capability gaps, therefore it is no longer eligible
to validate future changes. It is locked as historical baseline only and cannot
be used for threshold selection, individual-case debugging or future improvement
claims. SPH remains reserved as the next untouched external cohort.
