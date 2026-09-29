# MEDCALC ECG development decision log

This log records development-only engineering decisions. It is not a validation
report and must not be used to claim external performance.

## Data separation

- PTB-XL folds 1-8: development/tuning only.
- FAST-GATE-100: fixed post-change evaluation panel; its IDs are excluded from
  future folds 1-8 tuning selection.
- PTB-XL fold 9: internal development evaluation, not tuning.
- PTB-XL fold 10: later confirmation when appropriate.
- Frozen external datasets must not be used for tuning.

## Rejected mechanisms

### Independent supplemental P sequence (reverted)

The independent supplemental-P implementation was rejected after the isolated
folds 1-8 benchmark. AVB2 final sensitivity improved only from 0/12 to 2/12,
specificity fell to 56.25% with 175 false positives, AVB3 remained 0/11, and
AVB1 candidate sensitivity regressed. Do not reintroduce this mechanism in the
same form.

### RBBB three-wide-lead morphology rescue

A pre-specified RBBB-only rescue using >=3 QRS>=120 ms leads distributed across
>=1 limb and >=2 precordial leads with full RBBB morphology passed synthetic
regressions but produced no change on FAST-GATE-100: final TP remained 45/84,
FP remained 5/16, and RBBB remained 6/9. The change was closed without merge.

### AVB1 compound multilead PR rescue

A counterfactual requiring >=2 PR>200 ms leads plus AV evaluability, 1:1
conduction, stable PR, reproducible P, and P-QRS coupling >=0.55 was rejected.
On a development sample excluding FAST-GATE-100, it would recover 46/80
positives but trigger 47/120 additional negatives, for projected specificity
of 58.3%.

### AVB2/AVB3 same-class multilead replay

Re-running the existing AV classifier independently per eligible lead did not
recover any development AVB2 or AVB3 cases outside FAST-GATE-100. The failure
is therefore not solved by lead selection alone.

## Established architectural finding: AV block

The current P representation is QRS-centric. Lead analysis obtains P fiducials
by searching before each QRS, and the regular-rhythm fallback also uses
pre-R-aligned templates. Development AVB2/AVB3 cases are AV-evaluable in
multiple leads, but nonconducted P counts remain zero and classifications
collapse to first-degree AV delay or no high-grade block. Future AV work should
therefore investigate QRS-independent atrial observability with strict
cross-lead constraints, rather than loosening fusion thresholds.

## WPW / ventricular preexcitation

The existing same-lead multilead rescue is useful and should not be removed
without contrary evidence. In the targeted development audit excluding
FAST-GATE-100, 23/46 WPW positives were final-positive and 23/46 had the
existing multilead rescue; 2/160 clean controls were final false positives and
both had the rescue. The 22 candidate-to-fusion losses all failed PR-related
measurement/boundary gates and none had the existing rescue.

A stricter distributed-support counterfactual is being audited separately; no
clinical behavior is changed until it passes development specificity checks
and then FAST-GATE-100.

## Current work queue

1. QRS-independent residual atrial periodicity audit for AVB2/AVB3.
2. Distributed multilead WPW support audit.
3. LAFB morphology-triad audit.
4. After those decisions, address remaining fusion losses (LAFB, LBBB, LPFB)
   and AF false-positive safety.
