# R28 AV graph and temporal research branch

The current AV representation searches for P waves before each QRS. A blocked
P can therefore be absent from the input even when downstream conduction rules
are correct. This branch adds a separate learned detector scanning every sample,
a bipartite P/QRS timing graph, and a small Transformer consuming graph event
tokens. Existing R27 scores and the AV V2 clinical branch remain the comparison
baseline. No claim of improved clinical sensitivity follows from this code.

## Components

- `ecg_av_training_data.py`: a new, deterministic synthetic corpus with sinus,
  first-degree delay, Wenckebach, Mobitz II, 2:1, high-grade, independent AV
  clocks, and negative examples including AF-like rhythm, premature atrial
  events and junctional rhythm. Noise, P amplitude, polarity, T morphology,
  baseline and quantization vary. The frozen Synthetic-1000 fixture is untouched.
- `ecg_av_temporal_model.py`: dilated CNN with independent P/QRS channels;
  two-layer, four-head, 32-dimensional Transformer over detected event tokens.
  Candidate-edge degree and event timing enter the Transformer. This is a
  graph-informed temporal model, not a trained GNN or an LLM.
- `ecg_av_event_graph.py`: keeps every timing-compatible P/QRS edge instead of
  forcing a nearest-P assignment. Multiple edges remain ambiguous. A P without
  an edge is not automatically a blocked P. End-censored P events are excluded
  from this count. P-peak/R-peak delay is explicitly not clinical PR.
- `ecg_av_train.py`: trains the detector first, then the Transformer on its
  **detected events**, not oracle annotations. Patient groups and waveform
  duplicates cannot cross train/evaluation partitions. Held-out synthetic data
  is evaluated after the predefined training schedule, once.

## Explicit engine integration

The model is loaded only when explicitly requested through a separate adapter.
The clinical engine source and entry point are unchanged. Its default path does
not import Torch or load a checkpoint for this branch.

```python
from ecg_av_temporal_model import AVResearchModel
from ecg_av_research_adapter import analyze_ecg_with_av_research

model = AVResearchModel("models/r28_av_research/r28_av_research.pt")
analysis = analyze_ecg_with_av_research(canonical_ecg, av_research_model=model)
research = analysis["av_research"]
```

`av_research` is separate from the clinical feature graph, candidates, fusion,
reasoner and report interpretation. All research results abstain from clinical
diagnosis and identify their checkpoint hash. The softmax scores are uncalibrated
research outputs. A failed optional inference is recorded as `RESEARCH_ERROR`
and does not erase the baseline clinical result.

Inference uses the longest eligible uninterrupted native observed segment from
one lead, 6-60 seconds. It does not concatenate paper columns, bridge missing
samples, copy segments, infer multi-lead synchrony, or invoke the rejected
relative-R filter. Event timestamps are relative to the selected segment.
The current conservative gate requires quality-mask value 2 throughout; images
with frequent interpolation can therefore remain unevaluable. This limitation
must be measured before relaxing the gate.

## Reproducible synthetic development run

```bash
python -m pip install 'torch>=2.6,<3' numpy scipy neurokit2 pytest
python ecg_av_train.py --output models/r28_av_research
python -m pytest tests/test_ecg_av_research.py -q
```

The default schedule is 1,024 training records, 256 held-out records, six CNN
epochs and 20 Transformer epochs; the two lead variants remain within the same
synthetic patient partition. The report includes event precision/recall within
60 ms, an eight-class confusion matrix, source hashes, losses and the checkpoint
hash. These are synthetic development metrics, not performance on ECG images.

## Expert-annotated real training input

`--real-manifest records.jsonl --ptbxl-metadata ptbxl_database.csv` adds eligible
real examples. Each JSONL record requires `dataset_id="ptbxl"`,
`usage_role="DEVELOPMENT_TRAIN"`, `waveform_origin="DIGITIZED_IMAGE"`,
`annotation_source="EXPERT_P_QRS_AND_RHYTHM"`, `ecg_id`, `patient_id`, `fold`,
`label` from the eight declared classes, `fs=250`, one 10-second `signal_mv`
array and expert `p_s`/`r_s` arrays. Empty P arrays are permitted for relevant
negative examples. The native PTB-XL CSV must verify patient and fold.

Folds 9/10, FAST-GATE-100 records **and their patients**, native-signal-only
examples and other datasets are rejected for this training route. The loader
does not infer Mobitz subtype from a generic AVB2 label. Real patient partitions
are assigned deterministically; the real and synthetic confusion matrices are
reported separately. No real ECG annotations were supplied for the first run.

## Remaining acceptance work

### First real-signal transfer probe: failed

The frozen synthetic-only checkpoint was tested on 41 native PTB-XL development
records selected by the existing hash order, excluding FAST-GATE records and
their patients and all folds 9/10. The top experimental class matched AVB2 in
2/5 records and AVB3 in 0/4. A second- or third-degree AV-block class was the top score in 10/32
negative controls. All 41 runs completed without analysis errors, but the
checkpoint **failed transfer and must not be activated clinically**.

The probe uses native signals, not images, and no expert event-level P/QRS
annotations. These small counts are development observations, not validated
sensitivity/specificity estimates. No weights or thresholds were adjusted to
these results. The high synthetic accuracy does not generalize to this input.
See `models/r28_av_research/native_transfer_probe.json` and the reproducible
`ecg_av_native_transfer_probe.py` runner. Future development must not reuse this
probe as independent validation or tune case-specific exceptions against it.

```bash
python ecg_av_native_transfer_probe.py \
  --checkpoint models/r28_av_research/r28_av_research.pt \
  --data-root /tmp/r28-native-probe --output /tmp/r28-native-probe.json
```

1. Assemble expert P/QRS annotations on real digitalized images, with blocked
   P, P/T overlap, broad QRS, pacemakers, concealed extrasystoles, flutter,
   isorhythmic dissociation, dropped/extra detections and multiple paper layouts.
2. Train on eligible development data and freeze a checkpoint, calibration and
   abstention policy before any evaluation on protected panels.
3. Compare event precision/recall and AV outcomes against the current engine
   end to end, including actual rasterization/digitalization errors. The first
   corpus simulates trace quantization; it does not exercise the image digitizer.
4. Require the repository's Synthetic-1000, FAST-GATE-100, development and
   reserved-fold gates in order before proposing clinical fusion. Never tune
   against a protected panel or use an AV-dissociation score alone as proof of
   complete heart block. The eight classes are mutually exclusive training
   labels; coexisting conditions and mixed rhythms need dedicated evaluation.

This is the implemented experimental architecture and its first trained
checkpoint. The first checkpoint failed real native-signal transfer. It is not
a validated replacement for AV V2 and does not establish sensitivity above 90%.
