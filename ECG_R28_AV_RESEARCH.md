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

## Frozen event-level transfer audit (LUDB)

The unchanged checkpoint was evaluated on the pre-existing fixed LUDB 1.0.1
record sample 1,21,...,181, using lead II only. No weights or thresholds were
changed. LUDB remains evaluation-only; its annotations must not enter training.
Reference: Kalyakulina et al., DOI 10.13026/eegm-h675, ODC Attribution v1.0.

Within the annotated envelope, excluding a 60 ms edge margin, P matching yielded
71 TP, 43 FP, 1 FN (precision 62.28%, recall 98.61%). Twenty unmatched P events
were within 60 ms of annotated T peaks, and one near a QRS peak; these are
temporal coincidences and do not establish a causal error mechanism. QRS yielded
75 TP, 0 FP, 2 FN (precision 100%, recall 97.40%). Maximum-cardinality ordered
matching prevents duplicate reuse and nearest-pair greedy undercounting.

This small native-signal sample is an event transfer audit, not independent
clinical validation, image validation, or AV subtype validation. The annotated
envelope is only a proxy for complete coverage. The finding supports prioritizing
expert P/T/QRS hard negatives on eligible digitized PTB-XL development images;
it does not authorize tuning against LUDB or the previous transfer probe.

Reproduce with:
```sh
python ecg_av_ludb_event_probe.py \
  --checkpoint models/r28_av_research/r28_av_research.pt \
  --output /tmp/r28-ludb-event-probe.json
```

Aggregate counts and source hashes are recorded in
`models/r28_av_research/ludb_event_transfer_probe.json`. All 20 local research
and event-audit contract tests pass. No expert-annotated digitized training
manifest is available in the current workspace; new real-data training remains
blocked on that input.

## Acquired annotation sources and eligibility audit

Public annotation bytes have now been acquired and SHA-256 verified. The prior
statement that no ready digitized training manifest is available still applies;
source acquisition does not imply training readiness. See
`models/r28_av_research/annotation_source_audit.json`.

- PTB-XL annotation source: https://huggingface.co/datasets/figureli/ptb-xl-ecg-delineation
  pinned revision `c9d0577dc2bb82a6738e6d5dce1451aae4b09904`.
  AF masks are `(2604,1200)` for 217 records; other masks `(10728,1200)`
  for 894 records. These row counts are consistent with twelve leads per record,
  but class mapping, lead-row order and the 1200-position time axis are not
  documented. The original PTB-XL ECGs contain 10 seconds at 500 Hz; do not
  silently stretch masks to match them or infer class meaning from morphology.
  The source provides boundary segmentation rather than expert peak targets.
  Of 1111 IDs, 809 satisfy development fold/patient exclusions, 296 are in
  heldout folds, and six belong to protected FAST-GATE patients. The eligible
  list has 14 first-degree AV diagnostic labels and no AVB2/AVB3 labels.
  Eligibility does not certify annotation quality, completeness, or AV subtype.
- ISP v2: https://zenodo.org/records/14679837 (DOI 10.5281/zenodo.14679837).
  Published tables have 403 train and 72 test records; all headers specify
  1000 Hz. CSV targets are numeric class/onset/offset tuples, not expert peaks.
  Three records have invalid annotations: two spans beyond the capture and one
  zero-length span. No spans were clipped, repaired or used in training. No
  exact raw DAT file is shared across published splits, but patient identity
  is absent, so patient independence cannot be certified. ISP is outside the
  existing PTB-XL training allowlist and its inputs are native signals.

The source audit does not execute the digitizer or generate a training manifest.
No waveform was relabelled as DIGITIZED_IMAGE and no existing weights/thresholds
were changed. LUDB and other frozen evaluation datasets remain excluded from
training. Native annotation intervals must not be reported as expert peak truth.
A future boundary-supervised detector requires a separate, explicit target
contract rather than pretending these intervals satisfy the current peak loader.

Reproduce acquisition and auditing (requires NumPy and requests):
```sh
python ecg_av_annotation_source_audit.py --download \
  --source-root /tmp/r28-annotation-sources \
  --ptbxl-metadata /tmp/r28-native-probe/ptbxl_database.csv \
  --output /tmp/r28-annotation-source-audit.json
```

Downloads use pinned versions and verified hashes; existing mismatched files
raise an error rather than being overwritten. Without `--download`, the audit
is offline. The report contains the eligible IDs for later image acquisition.
The next concrete dependency is a published schema establishing PTB-XL mask
class IDs, row/lead order, temporal sampling and coverage. Only after that can
eligible images be digitized with verified event/interval alignment.

## Prepared-image digitizer smoke (three distinct development patients)

`ecg_av_prepare_development_images.py` rechecks authoritative PTB-XL folds and
protected patients before choosing three records by fixed SHA-256 order, one
record per patient. IDs 70,2188,484 were rendered as calibrated 3x4 pages with
a full II rhythm strip. These are rendered images of real waveforms, not
clinical photographs. The three PNGs and their source/image hashes are in
`models/r28_av_research/prepared_images/`. Each short lead shows its actual
2.5-second display window; the II strip shows 0-10 seconds. No annotations
were fabricated, stretched or attached, and no training manifest was created.

All three pages were run through the existing worker in both low-memory and
default high-fidelity modes, with verified original digitizer weights and no
R27 tiling. Worker status DIGITIZED_ONLY is expected for 3x4 pages, because
they do not contain 10 observed seconds in every lead. That status alone is
not a reconstruction-quality claim or a worker failure.

The original preparation smoke exposed a horizontal-coordinate provenance bug:
cropped centerlines were stretched over preflight bounds, producing implausible
10.322-10.718 s time extents. The branch now preserves extractor crop bounds,
restores the segmentation-only output to the original canvas with missing columns
represented as missing data, and refuses inconsistent coordinate provenance.

A subsequent short-fragment fix moved the minimum-width filter after graph
matching so valid waveform fragments survive without filling gaps. The corrected
normal-worker replay is:

| ID | Longest observed contiguous II | Canonical II time extent | R28 input |
|---|---:|---:|---|
|70|8.016 s|10.004 s|available|
|2188|9.988 s|10.006 s|available|
|484|9.994 s|10.008 s|available|

ECG 70 still contains a real 146 ms missing interval; it is not interpolated.
Its separate 106 ms extraction loss was recovered by fragment preservation.
All three cases now satisfy the research input-window guard. These are engineering
replays on rendered development images, not clinical-photograph validation and
not measurement-accuracy validation. See
`models/r28_av_research/coordinate_integration_report.json` and
`models/r28_av_research/fragment_merge_report.json`.

Reproduce preparation:
```sh
python ecg_av_prepare_development_images.py \
  --annotation-audit models/r28_av_research/annotation_source_audit.json \
  --ptbxl-metadata /tmp/r28-native-probe/ptbxl_database.csv \
  --data-root /tmp/r28-native-probe --output /tmp/r28-development-images --limit 3
```

Run the existing `ecg_unet_worker.py` on each PNG with its normal documented
weight/vendor arguments, once with default options and once with
`--force-low-memory`; do not enable `--allow-r27-tiled`. Then audit the worker
JSON using `ecg_av_image_preparation_audit.py --manifest ... --worker-output
ECG_ID=PATH --output ...`, repeating `--worker-output` for each result.
See `prepared_image_digitization_audit.json` for worker/weight hashes and guards.
All37 local research tests pass, including patient separation, exact display
windows, rejection of misplaced long strips, missing/interpolated gaps and
time-axis overruns. Model weights and thresholds remain unchanged.


## Synthetic/native graph-topology alignment

The frozen V1 checkpoint was also used for a development-only comparison of
graph topology between the predefined held-out synthetic corpus and the already
inspected native PTB-XL development groups. The audit does not tune thresholds,
change weights, or authorize clinical fusion.

Of 256 synthetic held-out records, 254 produced an event graph. Two `OTHER`
records abstained with `INSUFFICIENT_OBSERVED_EVENT_CANDIDATES`; abstention is
now retained as a missing topology observation rather than converted into an
execution failure. The native topology probe completed with 5 AVB2, 4 AVB3 and
32 control records and no analysis errors.

Several AVB3 structural medians were directionally close across domains:
minimum unmatched-P fraction was 0.6471 native versus 0.6429 synthetic,
zero-degree-P fraction 0.6360 versus 0.6339, and candidate edges per P 0.3824
versus 0.3661. Phase concentration remained low in both domains (0.2853 versus
0.3604). The ventricular timing variability did not align: QRS interval CV was
0.5790 native versus 0.00228 synthetic.

AVB2 showed a larger domain gap. P:QRS count ratio was 0.8261 native versus
1.7619 synthetic, P:QRS rate ratio approximately 1.00 versus 1.7448, phase
concentration 0.5494 versus 0.9996, and P-interval CV 0.5087 versus 0.0108.
Native controls were also substantially more variable than synthetic sinus
controls (P-interval CV 0.2873 versus 0.00864).

Interpretation: the ambiguous graph representation preserves potentially useful
AV-dissociation structure, but the current synthetic timing distribution is not
a faithful model of the native AVB2/control domain. The PTB-XL records in this
audit have diagnostic labels but no expert event-level P/QRS truth, and they have
already been inspected; they must not be tuned against and then presented as
independent validation. Before changing the generator or classifier, the next
dependency is a verified annotation contract for real digitized development
data. Summary evidence is frozen in
`models/r28_av_research/synthetic_native_graph_alignment_summary.json`.

## Rejected T-aware detector experiments

Two development-only experiments tested whether explicit T-wave supervision could
reduce the false P candidates observed in the frozen LUDB transfer audit. Both
experiments were rejected and are not part of the retained R28 detector.

The frozen V1 comparison point used 1,024 synthetic training records and 256
held-out records. It achieved 95.3125% eight-class accuracy. P detection within
60 ms was 2873 TP / 333 FP / 33 FN (precision 89.6132%, recall 98.8644%);
QRS was 2343 TP / 0 FP / 1 FN (precision 100%, recall 99.9573%).

The first T-aware experiment added a third P/QRS/T detector channel, changed the
synthetic T-wave construction, and multiplied P probability by one minus T
probability before peak finding. The full predefined synthetic run completed in
GitHub Actions run 37686018731. Accuracy fell to 83.59375%; P detection was
2857 TP / 343 FP / 148 FN (precision 89.28125%, recall 95.0749%), and QRS was
2405 TP / 30 FP / 0 FN (precision 98.7680%, recall 100%). Because the experiment
changed both supervision and waveform generation, it was not a clean causal
comparison. More importantly, a T-derived hard veto is physiologically unsafe
for AV-block research because true atrial activity can overlap the T wave.

A second controlled experiment preserved the V1 synthetic waveforms byte for
byte using fixed SHA-256 regression cases, used lead-specific T labels only as
an auxiliary task, and never allowed T probability to suppress a P candidate.
All 72 research/coordinate/fragment contracts passed. The full predefined run,
GitHub Actions run 37686943064, still underperformed V1: accuracy 89.0625%; P
2837 TP / 376 FP / 69 FN (precision 88.2975%, recall 97.6256%); QRS 2344 TP /
25 FP / 0 FN (precision 98.9447%, recall 100%). T itself was detected well
(precision 97.1002%, recall 99.5627%), but the shared detector degraded the AV
events that matter.

Decision: reject both T-aware variants, restore the V1 P/QRS detector and
synthetic generator exactly, and retain the frozen V1 checkpoint. No clinical
fusion, diagnostic threshold, deployed weight, or protected evaluation set was
changed. LUDB was not used to tune a threshold; it only motivated the rejected
research hypothesis. Future P/T disambiguation should be isolated from the P/QRS
event detector (for example as a separate representation or post-hoc research
feature) and must not hard-veto atrial candidates.
