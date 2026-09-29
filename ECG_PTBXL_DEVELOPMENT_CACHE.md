# PTB-XL development cache

This composite action is infrastructure only. It does not alter ECG analysis,
record selection, labels, thresholds, or validation policy.

Development-only workflows may use a common workdir:

`/tmp/medcalc-ptbxl-development-cache`

The cache uses unique save keys per workflow run and a stable restore prefix,
so each successful audit can restore the most recent cache snapshot, add any
missing PTB-XL records, and save a newer snapshot. This avoids repeatedly
downloading the same immutable PTB-XL signals.

The cache is only a byte-level performance optimization. Selection guards still
enforce:

- folds 1-8 for tuning/audits,
- FAST-GATE-100 exclusion where required,
- folds 9/10 separation,
- no external-dataset tuning.
