# MEDCALC ECG — External validation provenance policy

## Why this exists

The measurement engine has already been developed using several public ECG
databases. Those databases remain useful for regression testing, but they are
not external validation sets anymore. Performance measured on them must never be
reported as independent generalization performance.

## Development-contaminated datasets

The following are explicitly locked to development/regression because prior
exposure was confirmed:

- LUDB
- QT Database
- PTB-XL
- MIT-BIH Arrhythmia Database
- MIT-BIH AF Database
- Chapman-Shaoxing
- Ningbo / LSAS-ECG
- CPSC 2018
- Georgia Challenge ECG data
- INCART
- PhysioNet/CinC Challenge 2020/2021 aggregate

Any images, crops, augmentations or rendered paper ECGs derived from these
sources inherit the same status.

## Provisional external locked cohorts

### CODE-test

827 12-lead ECGs from different patients, 400 Hz, with adjudicated gold-standard
labels for first-degree AV block, RBBB, LBBB, sinus bradycardia, atrial
fibrillation and sinus tachycardia.

Use: first independent diagnostic benchmark for the current signal-first engine.

Restriction: labels are test-only. They may be scored after inference but may
not be used to choose thresholds or patch individual cases.

Source: https://zenodo.org/records/3765780

### SPH

25,770 ECGs from 24,666 patients, 12 leads, 500 Hz, 10-60 seconds, with
standardized diagnostic statements reviewed by cardiologists.

Frozen evaluation design for the post-CODE-test engine:

- general cohort: 2,000 patients selected by SHA-256 of Patient_ID without
  consulting AHA_Code, one deterministic ECG per patient;
- target-positive cohort: one deterministic ECG per positive patient for each
  predeclared target (AF, flutter, sinus bradycardia/tachycardia, complete
  RBBB, LBBB, LAFB, LPFB, AVB1/2/3 and ventricular preexcitation);
- negative controls: 2,000 patients carrying none of those target statements,
  selected by SHA-256, one ECG per patient;
- inference jobs receive only waveforms/ID/age/sex; AHA_Code and code.csv remain
  isolated until all inference shards finish;
- every selected record is evaluated on the first 10 seconds, the minimum SPH
  duration, so record length cannot influence target selection.

General-cohort performance estimates generalization without label-based
selection. Target-positive enrichment is used to estimate sensitivity for rare
diagnoses; PPV/NPV from the enriched union are not prevalence-calibrated.

Use: external generalization benchmark across rhythm and conduction.

Source: https://www.nature.com/articles/s41597-022-01403-5

### MIMIC-IV-ECG

Approximately 800,000 10-second diagnostic 12-lead ECGs from nearly 160,000
patients at 500 Hz. Access is credentialed. Machine measurements and links to
cardiologist reports are available where applicable.

Use: large-scale clinical robustness and measurement-distribution audit after
access is available.

Source: https://physionet.org/content/mimic-iv-ecg/1.0/

## Frozen-test rules

1. External test labels are never used to tune thresholds, feature weights,
   fiducial windows, model selection or report wording rules.
2. A failure discovered in external validation may be analyzed, but the same
   external cohort cannot then be reused as an unbiased final test after tuning.
   The modified engine must be evaluated on a new frozen cohort or predeclared
   untouched partition.
3. Patient-level separation is mandatory.
4. Derived images inherit the provenance of their source waveform.
5. If prior exposure to CODE-test, SPH or MIMIC-IV-ECG is later discovered,
   change its status to DEVELOPMENT_CONTAMINATED before any further claim.
6. CI runs `ecg_validation_guard_selftest.py` to prevent accidental provenance
   relabeling.

## Validation layers

### Layer A — development regression

Use contaminated datasets and synthetic cases freely to catch regressions. Do
not call this external validation.

### Layer B — digital-signal external validation

Feed untouched digital 12-lead signals directly into the canonical MEDCALC
measurement/reasoning engine. This isolates measurement and interpretation from
image reconstruction.

### Layer C — end-to-end external validation

Render the same untouched external signals into paper/PDF/photo variants, then
run:

source -> U-Net -> digitizer -> calibrated signal -> measurement -> reasoner.

Compare both the reconstructed signal/measurements and the final interpretation
against the frozen source truth.

## Required reporting

For measurements, report MAE, median absolute error, 95th percentile absolute
error, bias and failure-to-measure rate where reference annotations exist.

For categorical diagnoses, report sensitivity, specificity, PPV, NPV, F1 and
abstention/fail-closed rate with denominators.

Results must identify the exact engine commit SHA and exact dataset version.
