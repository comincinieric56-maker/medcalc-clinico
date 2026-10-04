"""Frozen R28 event transfer audit. LUDB is evaluation only, never training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ecg_av_temporal_model import AVResearchModel

RECORDS = tuple(str(i) for i in range(1, 201, 20))
TOLERANCE_S = .060


def match_events(truth, candidates, tolerance=TOLERANCE_S):
    """Maximum cardinality matching of ordered timestamps within tolerance."""
    truth, candidates = sorted(truth), sorted(candidates)
    if tolerance < 0 or not np.isfinite([*truth, *candidates, tolerance]).all():
        raise ValueError("Finite timestamps and nonnegative tolerance required")
    i = j = tp = 0
    unmatched = []
    while i < len(truth) and j < len(candidates):
        if candidates[j] < truth[i] - tolerance:
            unmatched.append(candidates[j])
            j += 1
        elif truth[i] < candidates[j] - tolerance:
            i += 1
        else:
            tp += 1
            i += 1
            j += 1
    unmatched.extend(candidates[j:])
    return {"tp": tp, "fp": len(candidates) - tp, "fn": len(truth) - tp}, unmatched


def score_events(events, nodes, fs):
    """Score only the interior of the annotated envelope; exclude edge censoring.

    The envelope is a proxy for annotation coverage, not independently reviewed
    completeness. No false positives are counted outside that envelope.
    """
    if not events:
        raise ValueError("No reference annotations")
    start = min(e["onset"] for e in events) / fs + TOLERANCE_S
    end = max(e["offset"] for e in events) / fs - TOLERANCE_S
    if end <= start:
        raise ValueError("Empty annotated interior")
    truth = {kind: [e["peak"] / fs for e in events
                    if e["kind"] == kind and start <= e["peak"] / fs <= end]
             for kind in ("P", "QRS", "T")}
    result = {}
    for kind in ("P", "QRS"):
        predicted = [n["time_s"] for n in nodes
                     if n["kind"] == kind and start <= n["time_s"] <= end]
        result[kind], unmatched = match_events(truth[kind], predicted)
        if kind == "P":
            # Categories may overlap; these are temporal coincidences, not causes.
            result[kind]["unmatched_near_t_peak"] = sum(
                any(abs(t - c) <= TOLERANCE_S for t in truth["T"]) for c in unmatched)
            result[kind]["unmatched_near_qrs_peak"] = sum(
                any(abs(t - c) <= TOLERANCE_S for t in truth["QRS"]) for c in unmatched)
    return result


def run(checkpoint, output):
    import torch
    from ecg_ludb_delineation_dev_benchmark import _canonical_from_record, _parse_events
    torch.set_num_threads(2)
    model = AVResearchModel(checkpoint)
    totals = {"P": {}, "QRS": {}}
    source_hashes = []
    for record in RECORDS:
        canonical, fs = _canonical_from_record(record)
        signal = np.asarray(canonical["leads"]["II"]["signal_mv"], dtype=np.float32)
        events = _parse_events(record, "II")
        source_hashes.append({"record": record,
                              "signal_sha256": hashlib.sha256(signal.tobytes()).hexdigest(),
                              "events_sha256": hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()})
        result = model.analyze_signal(signal, int(fs))
        counts = score_events(events, result["graph"]["nodes"], fs)
        for kind, values in counts.items():
            for key, value in values.items():
                totals[kind][key] = totals[kind].get(key, 0) + value
        print(f"Scored frozen LUDB record {record}", flush=True)
    for count in totals.values():
        count["precision"] = count["tp"] / max(1, count["tp"] + count["fp"])
        count["recall"] = count["tp"] / max(1, count["tp"] + count["fn"])
    report = {"role": "FROZEN_NATIVE_EVENT_TRANSFER_AUDIT_ONLY", "dataset": "LUDB_1.0.1",
              "dataset_doi": "10.13026/eegm-h675", "lead": "II", "records": list(RECORDS),
              "checkpoint_sha256": model.sha256, "matching_tolerance_s": TOLERANCE_S,
              "source_hashes": source_hashes, "event_counts": totals,
              "weights_or_thresholds_changed": False, "clinical_ready": False,
              "training_allowed": False, "image_validation_claim_allowed": False,
              "limitations": ["Small fixed native-signal sample, not image validation",
                              "Annotated envelope is a coverage proxy, excludes edges",
                              "Temporal P/T or P/QRS coincidence does not establish error cause",
                              "No AV subtype or clinical performance claim"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.output)["event_counts"], indent=2))
