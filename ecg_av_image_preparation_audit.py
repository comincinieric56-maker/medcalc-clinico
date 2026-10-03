"""Audit prepared-image digitizer outputs against known display windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect_digitized(meta, manifest):
    signal = meta.get("signal") or {}
    expected = {}
    for window in manifest["lead_display_windows"]:
        expected[window["lead"]] = max(expected.get(window["lead"], 0.),
                                      window["end_s"] - window["start_s"])
    blockers = []
    unexpected = [lead for lead, seconds in (signal.get("observed_seconds_by_lead") or {}).items()
                  if lead not in expected or float(seconds) > expected[lead] + .1]
    if unexpected:
        blockers.append("OBSERVED_DURATION_EXCEEDS_KNOWN_DISPLAY_WINDOW")
    if meta.get("status") == "FAIL":
        blockers.append("DIGITIZER_FAILED")
    if signal.get("r27_tiled"):
        blockers.append("REPEATED_SIGNAL_REJECTED")
    canonical = signal.get("calibrated_digital_signal") or {}
    item = (canonical.get("leads") or {}).get("II") or {}
    longest, extent = 0., None
    if not item:
        blockers.append("NO_SERIALIZED_CANONICAL_II_SIGNAL")
    elif item.get("repeated"):
        blockers.append("REPEATED_SIGNAL_REJECTED")
    else:
        try:
            fs = float(item.get("fs") or canonical.get("fs") or 0)
            x = np.asarray(item["signal_mv"], dtype=float)
            mask = np.asarray(item["quality_mask"])
            if fs <= 0 or not np.isfinite(fs) or x.ndim != 1 or x.shape != mask.shape:
                raise ValueError("Invalid canonical signal")
            extent = float(len(x) / fs)
            if extent > expected["II"] + .1:
                blockers.append("CANONICAL_II_TIME_EXTENT_EXCEEDS_KNOWN_DISPLAY_WINDOW")
            usable = np.isfinite(x) & (mask == 2)
            edges = np.diff(np.r_[False, usable, False].astype(int))
            lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
            longest = float(max(lengths, default=0) / fs)
            if longest > expected["II"] + .1:
                blockers.append("CANONICAL_II_EXCEEDS_KNOWN_DISPLAY_WINDOW")
        except (KeyError, TypeError, ValueError):
            blockers.append("INVALID_SERIALIZED_CANONICAL_II_SIGNAL")
    if longest < 6.:
        blockers.append("INSUFFICIENT_CONTIGUOUS_OBSERVED_II_FOR_R28")
    return {"digitizer_status": meta.get("status"),
            "fidelity_mode": signal.get("fidelity_mode"),
            "observed_seconds_by_lead": signal.get("observed_seconds_by_lead") or {},
            "unexpected_duration_leads": unexpected,
            "longest_contiguous_observed_ii_s": longest,
            "canonical_ii_time_extent_s": extent,
            "known_display_duration_tolerance_s": .1,
            "r28_input_window_available": not blockers,
            "blockers": sorted(set(blockers)), "training_allowed": False,
            "annotation_alignment_verified": False}


def audit(manifest_path, outputs, report_path):
    manifest = json.loads(manifest_path.read_text())
    report = {"role": "PREPARED_IMAGE_DIGITIZATION_SMOKE_ONLY",
              "image_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "clinical_ready": False, "training_manifest_created": False,
              "weights_or_thresholds_changed": False, "outputs": [],
              "limitations": ["Small fixed sample of rendered real signals, not clinical photographs",
                              "Image labels remain unverified; no annotation performance measured",
                              "Low-memory and default routes must be reported separately"]}
    ids = {record["ecg_id"] for record in manifest["records"]}
    for ecg_id, path in outputs:
        if ecg_id not in ids:
            raise ValueError("Output record not present in prepared manifest")
        meta = json.loads(path.read_text())
        report["outputs"].append({"ecg_id": ecg_id,
                                  "worker_meta_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "segmentation_model_sha256": meta.get("segmentation_model_sha256"),
                                  "lead_model_sha256": meta.get("lead_model_sha256"),
                                  **inspect_digitized(meta, manifest)})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--worker-output", action="append", required=True, metavar="ECG_ID=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outputs = [(int(value.split("=", 1)[0]), Path(value.split("=", 1)[1]))
               for value in args.worker_output]
    print(json.dumps(audit(args.manifest, outputs, args.output), indent=2))
