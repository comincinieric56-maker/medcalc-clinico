# PR127 — independent AV P-sequence regression note

Purpose: preserve the engineering contract for the AVB2/AVB3 structural fix.

Observed development failure before this change:
- QRS-centered P fiducials frequently contained no nonconducted P waves.
- AVB2/AVB3 therefore failed before the candidate layer.

Required behavior after this change:
- Supplemental P candidates are used only by AV-conduction analysis.
- Supplemental P candidates must be morphologically template-compatible.
- A supplemental P is accepted only with temporal support in at least two leads.
- Canonical/raw P fiducials and PR/P measurements are not overwritten.
- A true synthetic 2:1 sequence is recovered.
- An otherwise identical single-lead-only supplemental signal is rejected.

This note is DEVELOPMENT_REGRESSION_ONLY. It is not validation evidence.

After a real development ECG from an allowed source is frozen for this failure
mode, add it to ecg_regression_corpus.json as a known-case replay. If PTB-XL is
used, only folds 1-8 are permitted.
