"""Reproducible detector + graph-token Transformer training, research only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from ecg_av_event_graph import TOKEN_FEATURES, build_av_event_graph
from ecg_av_temporal_model import (VERSION, create_networks, events_from_probabilities,
                                   normalize_signal, token_batch)
from ecg_av_training_data import CLASSES, DATA_VERSION, DURATION_S, FS, synthetic_case

EVENT_CHANNELS = ("P", "QRS", "T")


def load_real_training(path: Path, metadata_path: Path | None = None):
    """Accept expert P/QRS/T-annotated digitized PTB-XL development records."""
    frozen = json.loads(Path(__file__).with_name("ecg_fast_gate_100_manifest.json").read_text())
    def ids(value):
        out = set()
        if isinstance(value, dict):
            if "ecg_id" in value:
                out.add(str(value["ecg_id"]))
            for child in value.values():
                out |= ids(child)
        elif isinstance(value, list):
            for child in value:
                out |= ids(child)
        return out
    protected, seen, rows = ids(frozen), set(), []
    metadata = {}
    if metadata_path is not None:
        with metadata_path.open(newline="") as stream:
            metadata = {str(row["ecg_id"]): row for row in csv.DictReader(stream)}
        if not protected.issubset(metadata):
            raise ValueError("Metadata does not resolve every protected FAST-GATE record")
    protected_patients = {str(metadata[i]["patient_id"]) for i in protected} if metadata else set()
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if (record.get("dataset_id") != "ptbxl" or record.get("usage_role") != "DEVELOPMENT_TRAIN"
                or record.get("waveform_origin") != "DIGITIZED_IMAGE"
                or not record.get("patient_id") or not record.get("ecg_id")
                or int(record.get("fold", 0)) not in range(1, 9)
                or str(record["ecg_id"]) in protected
                or record.get("annotation_source") != "EXPERT_P_QRS_T_AND_RHYTHM"
                or "t_s" not in record):
            raise ValueError("Real training record lacks eligible provenance or enters protected data")
        native = metadata.get(str(record["ecg_id"]))
        if (not native or int(native["strat_fold"]) != int(record["fold"])
                or str(native["patient_id"]) != str(record["patient_id"])
                or str(native["patient_id"]) in protected_patients):
            raise ValueError("Provenance disagrees with PTB-XL metadata or protected patient membership")
        x = np.asarray(record["signal_mv"], dtype=np.float32)
        if int(record["fs"]) != FS or x.ndim != 1 or len(x) != FS * DURATION_S or not np.isfinite(x).all():
            raise ValueError("Real training requires one finite 10s digitized lead at 250 Hz")
        digest = hashlib.sha256(x.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate real waveform")
        seen.add(digest)
        label = CLASSES.index(record["label"])
        y = np.zeros((1, 3, len(x)), dtype=np.float32)
        for channel, key, half_width in ((0, "p_s", .024), (1, "r_s", .016), (2, "t_s", .040)):
            for event_time in record[key]:
                if not np.isfinite(event_time) or not 0 <= event_time < DURATION_S:
                    raise ValueError("Invalid expert event timestamp")
                y[0, channel, abs(np.arange(len(x)) / FS - event_time) <= half_width] = 1
        patient = "ptbxl-" + str(record["patient_id"])
        partition = int(hashlib.sha256(patient.encode()).hexdigest()[:8], 16) % 5
        rows.append({"case_id": digest, "patient_id": patient, "label": label,
                     "signal": x[None], "targets": y, "p_s": record["p_s"],
                     "r_s": record["r_s"], "t_s": record["t_s"],
                     "fs": FS, "duration_s": DURATION_S, "partition": "eval" if partition == 0 else "train"})
    return rows


def train(args):
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(2801)
    np.random.seed(2801)
    torch.use_deterministic_algorithms(True)
    detector, classifier = create_networks(detector_channels=len(EVENT_CHANNELS))
    train_cases = [synthetic_case(i, "train") for i in range(args.train_cases)]
    eval_cases = [synthetic_case(i, "heldout") for i in range(args.eval_cases)]
    real = load_real_training(Path(args.real_manifest), Path(args.ptbxl_metadata)) if args.real_manifest else []
    train_cases += [r for r in real if r["partition"] == "train"]
    eval_cases += [r for r in real if r["partition"] == "eval"]
    if {r["patient_id"] for r in train_cases} & {r["patient_id"] for r in eval_cases}:
        raise ValueError("Patient overlap across partitions")
    train_hashes = {hashlib.sha256(r["signal"].tobytes()).hexdigest() for r in train_cases}
    if train_hashes & {hashlib.sha256(r["signal"].tobytes()).hexdigest() for r in eval_cases}:
        raise ValueError("Waveform overlap across partitions")

    xs, ys = [], []
    for row in train_cases:
        if row["targets"].shape != (len(row["signal"]), len(EVENT_CHANNELS), int(FS * DURATION_S)):
            raise ValueError("Detector targets must be lead-specific P/QRS/T arrays")
        for lead_index, lead in enumerate(row["signal"]):
            xs.append(normalize_signal(lead)[None])
            ys.append(row["targets"][lead_index])
    xs, ys = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
    optimizer = torch.optim.AdamW(detector.parameters(), lr=.002, weight_decay=.001)
    positives = ys.sum((0, 2)).clamp(min=1)
    positive_weight = ((ys.shape[0] * ys.shape[2] - positives) / positives).clamp(max=20)[None, :, None]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    history = []
    for epoch in range(args.detector_epochs):
        detector.train()
        order = torch.randperm(len(xs))
        total = 0.
        for batch in order.split(args.batch_size):
            optimizer.zero_grad()
            loss = criterion(detector(xs[batch]), ys[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(total / len(xs))
        print(f"DETECTOR epoch={epoch + 1} loss={history[-1]:.4f}", flush=True)
    detector.eval()

    def graphs_and_detections(cases):
        graphs, detections = [], []
        with torch.inference_mode():
            for row in cases:
                x = torch.from_numpy(normalize_signal(row["signal"][0]))[None, None]
                prob = detector(x).sigmoid()[0].numpy()
                p, r, t = events_from_probabilities(prob, include_t=True)
                graphs.append(build_av_event_graph(p, r, row["duration_s"]))
                detections.append({"P": p, "QRS": r, "T": t})
        return graphs, detections

    train_graphs, _ = graphs_and_detections(train_cases)
    labels = torch.tensor([r["label"] for r in train_cases])
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.002, weight_decay=.001)
    sequence_history = []
    for epoch in range(args.sequence_epochs):
        classifier.train()
        total = 0.
        for batch in torch.randperm(len(train_graphs)).split(args.batch_size):
            tokens, mask = token_batch([train_graphs[i] for i in batch.tolist()])
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(classifier(tokens, mask), labels[batch])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        sequence_history.append(total / len(train_graphs))
        if (epoch + 1) % 5 == 0:
            print(f"TEMPORAL epoch={epoch + 1} loss={sequence_history[-1]:.4f}", flush=True)

    classifier.eval()
    eval_graphs, eval_detections = graphs_and_detections(eval_cases)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in EVENT_CHANNELS}
    source_confusion = {}
    with torch.inference_mode():
        for row, graph, detected in zip(eval_cases, eval_graphs, eval_detections):
            tokens, mask = token_batch([graph])
            pred = int(classifier(tokens, mask).argmax(-1)[0])
            confusion[row["label"], pred] += 1
            source = "real_digitized_development" if "partition" in row else "synthetic"
            source_confusion.setdefault(source, np.zeros_like(confusion))[row["label"], pred] += 1
            for kind, key in (("P", "p_s"), ("QRS", "r_s"), ("T", "t_s")):
                candidates = [n["time_s"] for n in detected[kind]]
                truth = list(row[key])
                pairs = sorted((abs(t - c), i, j) for i, t in enumerate(truth) for j, c in enumerate(candidates)
                               if abs(t - c) <= .060)
                used_t, used_c = set(), set()
                for _, i, j in pairs:
                    if i not in used_t and j not in used_c:
                        used_t.add(i)
                        used_c.add(j)
                counts[kind]["tp"] += len(used_t)
                counts[kind]["fn"] += len(truth) - len(used_t)
                counts[kind]["fp"] += len(candidates) - len(used_c)
    for count in counts.values():
        count["precision"] = count["tp"] / max(1, count["tp"] + count["fp"])
        count["recall"] = count["tp"] / max(1, count["tp"] + count["fn"])
    source = DATA_VERSION if not real else DATA_VERSION + "+EXPERT_DIGITIZED_PTBXL_DEVELOPMENT"
    metadata = {"version": VERSION, "classes": list(CLASSES), "fs": FS,
                "token_features": list(TOKEN_FEATURES), "event_channels": list(EVENT_CHANNELS),
                "p_candidate_policy": "RAW_P_PROBABILITY_T_AUXILIARY_ONLY_V1",
                "training_source": source,
                "diagnostic_claim_allowed": False, "clinical_fusion_allowed": False,
                "seed": 2801, "torch_version": str(torch.__version__)}
    metadata["training_source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("ecg_av_train.py", "ecg_av_training_data.py",
                     "ecg_av_event_graph.py", "ecg_av_temporal_model.py")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "r28_av_research.pt"
    torch.save({"metadata": metadata, "detector": detector.state_dict(),
                "classifier": classifier.state_dict()}, checkpoint)
    report = {"metadata": metadata, "train_cases": len(train_cases), "eval_cases": len(eval_cases),
              "detector_loss": history, "sequence_loss": sequence_history,
              "confusion_rows_truth_columns_predicted": confusion.tolist(),
              "source_confusion": {k: v.tolist() for k, v in source_confusion.items()},
              "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
              "event_detection_60ms": counts,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "clinical_ready": False, "external_validation_claim_allowed": False,
              "limitation": "T-auxiliary synthetic waveform evaluation is not digitized-image or clinical validation"}
    (args.output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("accuracy", "event_detection_60ms", "clinical_ready")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cases", type=int, default=1024)
    parser.add_argument("--eval-cases", type=int, default=256)
    parser.add_argument("--detector-epochs", type=int, default=6)
    parser.add_argument("--sequence-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--real-manifest")
    parser.add_argument("--ptbxl-metadata")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_cases, args.eval_cases, args.detector_epochs, args.sequence_epochs,
           args.batch_size, args.threads) <= 0:
        parser.error("Training sizes and epochs must be positive")
    if args.real_manifest and not args.ptbxl_metadata:
        parser.error("Real training requires --ptbxl-metadata for authoritative fold/patient checks")
    train(args)
