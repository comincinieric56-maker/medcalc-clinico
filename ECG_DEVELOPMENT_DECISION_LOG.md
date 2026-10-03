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

### QRS-independent residual atrial periodicity

A QRS-independent residual periodicity audit was rejected. After subtracting an
R-locked ventricular template and requiring concordant periodicity across
multiple P-rich leads, the trigger was present in 0/5 development AVB2 cases
and 1/4 AVB3 cases, but also in 51/120 clean controls. Projected specificity
was 57.5%. Do not use this mechanism as a high-grade AV-block rescue.

### AF RR-irregularity plus atrial-specialist safety gate

The frozen policy ATRIAL_MECHANISM_AF_COMPATIBLE plus RR irregularity >=0.45
was tested after selection. On FAST-GATE-100, current AF was 6/8 true positives
with 15 false positives among the non-AF records in that confirmation set.
Requiring the policy reduced false positives to 11 but also reduced AF true
positives to 5/8. On fold 9, it preserved the same 1/3 AF true positives while
reducing false positives 20 -> 9. Because the frozen panel lost a true AF case,
the safety gate is rejected under the no-per-target-loss rule.

### LAFB morphology-triad rescue

A morphology-only LAFB rescue using existing axis criteria, positive QRS in I
and aVL, >=2 inferior S-dominant leads, small superior q support, and QRS<120 ms
was rejected. On folds 1-8 excluding FAST-GATE-100 it projected 43/80 -> 60/80
positives, but false positives increased 2 -> 16 in 80 clean controls, for
projected specificity 80%.

### LBBB lateral-delay rescue

The tuning-selected policy requiring current wide-QRS support plus dominant
lateral R and lateral delay/notching did not generalize safely. On fold 9,
baseline-or-policy improved true positives 43 -> 46 but false positives 2 -> 9.
On FAST-GATE-100 it did not improve LBBB sensitivity at all (8/9 remained 8/9)
and false positives increased 1 -> 6. A direct clinical implementation also
produced zero aggregate benefit on FAST-GATE-100 and was rejected by the
benefit gate.

### LPFB specialist-only safety gate

Requiring the existing fascicular specialist to independently classify
LPFB_COMPATIBLE was rejected. In folds 1-8 development data, current final LPFB
was 64/80 with 3/80 false positives. The safety gate reduced false positives to
1/80 but reduced true positives to 24/80, dropping sensitivity from 80% to 30%.

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

### R28 learned AV research branch

R28 adds a QRS-independent dilated CNN event detector, ambiguous P/QRS candidate
graph and temporal Transformer. It is explicitly opt-in and returns a separate
`av_research` result; it cannot alter clinical candidates, fusion, R27 scores or
report interpretation. The relative-R filter and the rejected supplemental-P
rescue are not used. All research scores remain uncalibrated and abstain from
clinical diagnosis.

The first predefined training run used 1,024 new synthetic training records and
256 separate synthetic evaluation records. Eight-class accuracy was 95.3125%;
P-event precision/recall within 60 ms was 89.6132%/98.8644%. False P candidates
remain a material concern. These are waveform-simulation results, not clinical
or image-digitalization validation. Expert-annotated real digitalized training
and end-to-end acceptance gates remain pending. See `ECG_R28_AV_RESEARCH.md`
and `ECG_R28_AV_STATE.json`; do not promote this checkpoint to clinical fusion.

The frozen checkpoint subsequently failed a native PTB-XL development transfer
probe: AVB2 top-class match 2/5, AVB3 top-class match 0/4, and second/third-degree AV-block top
scores in 10/32 negative controls. There were no analysis errors. These are
uncalibrated research scores, not emitted clinical findings or image validation.
No model/threshold tuning followed this probe. The integration now uses a
separate adapter; the clinical engine file and its original entry point are
byte-identical to the baseline. Architecture work may continue, but this first
checkpoint is rejected for clinical activation.

1. Frozen confirmation of the six-lead least-squares frontal QRS-axis LAFB
   hypothesis on fold 9 and FAST-GATE-100. The tuning audit recovered 8
   additional LAFB positives (43 -> 51/81) with 0 incremental triggers in 80
   clean controls; no clinical change is allowed before confirmation.
2. Flutter fusion-to-final anatomy. FAST-GATE-100 has 8/8 flutter candidates
   and 8/8 fused findings but only 6/8 final findings, indicating a reasoner
   hierarchy loss rather than detector failure. Quantify the cost of preserving
   fused flutter as a secondary rhythm finding before any change.
3. AVB2/AVB3 remain architectural research items only. Existing same-class
   multilead replay, independent supplemental-P, and residual-periodicity
   approaches have all failed specificity or recovery requirements.
