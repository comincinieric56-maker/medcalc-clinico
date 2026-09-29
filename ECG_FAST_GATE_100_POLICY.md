# MEDCALC ECG FAST-GATE-100 policy

## Purpose

FAST-GATE-100 is a fixed **post-change evaluation panel** for ECG engineering.
It is not a training set, tuning set, threshold-search set, or validation claim.

The panel contains 100 unique adult PTB-XL ECGs from folds 1-8 only.
Its V1 membership is frozen.

## Anti-overfitting rules

1. A clinical change must be defined before FAST-GATE-100 is run.
2. FAST-GATE-100 must not be used to choose thresholds, features, morphology
   rules, or case-specific exceptions.
3. Individual ECG failures from the panel must not be chased iteratively.
   Routine CI exposes aggregate metrics only.
4. Clinical logic and FAST-GATE-100 membership/baseline must not change in
   the same pull request.
5. A clinical change is a regression if it increases final false positives,
   introduces analysis errors, or reduces final positives for any covered
   diagnosis relative to the frozen aggregate baseline.
6. Improvement on FAST-GATE-100 is engineering evidence only. It does not
   replace PTB-XL full-fold development benchmarking and does not constitute
   internal or external validation.
7. Folds 9 and 10 remain outside this panel. Frozen external datasets remain
   forbidden for tuning.
8. Changing panel membership requires a new version and the new version may
   not be used to claim improvement over V1 without an independently defined
   comparison plan.

## Diagnostic coverage

Real cases in FAST-GATE-100 cover the adult diagnostic priorities available
in PTB-XL folds 1-8: atrial fibrillation, atrial flutter, sinus tachycardia,
RBBB, LBBB, LAFB, LPFB, first-degree AV block, second-degree/high-grade AV
block, complete AV block, and ventricular preexcitation.

At the configured PTB-XL reference-label likelihood threshold, folds 1-8
contain no sinus-bradycardia positives. Sinus bradycardia therefore remains
covered by synthetic/unit regression and must not be filled by using folds
9/10.

## Workflow

Predefined engineering change
→ unit/synthetic regression
→ FAST-GATE-100 aggregate comparison
→ if no regression and evidence supports benefit, full PTB-XL folds 1-8
→ fold 9 internal development evaluation
→ fold 10 confirmation when appropriate
→ frozen external validation only after development is locked.
