# MEDCALC ECG known-case regression fixtures

This directory is reserved for **development-only ECG regression fixtures**.

A fixture may be added only when all of the following are true:

1. The ECG has already been designated development/regression material.
2. Its dataset is registered as `DEVELOPMENT_CONTAMINATED` in
   `ecg_dataset_provenance.json`.
3. If the source is PTB-XL, the record belongs to folds 1-8 only.
4. The fixture is referenced by a stable `record_ref` in
   `ecg_regression_corpus.json`.
5. The fixture contains no unnecessary direct identifiers or PHI.
6. Its expected behavior is defined before using future validation data.
7. It is never reported as internal validation, external validation, test-set
   performance, or evidence of clinical readiness.

Forbidden sources include, but are not limited to:

- PTB-XL fold 9.
- PTB-XL fold 10.
- CODE-test after its consumed baseline.
- SPH after its consumed baseline.
- MIMIC-IV-ECG after its consumed baseline.
- ZZU pECG after its consumed pediatric baseline.
- HEEDB while locked for future external validation.
- Any future dataset designated external/frozen before first evaluation.

The regression runner fails closed when a real fixture does not have an allowed
development provenance entry.

## Intended workflow

A previously understood failure is frozen as a regression fixture. After every
relevant ECG-engine change:

```
unit/synthetic tests
  -> known-case replay
  -> runtime / pipeline CI
  -> PTB-XL folds 1-8 aggregate development benchmark
  -> optional fold 9 internal generalization measurement
  -> later reserved confirmation / external validation
```

Known-case replay answers: **"Did a previously understood bug stay fixed?"**

It does not answer: **"How well does the model generalize?"**
