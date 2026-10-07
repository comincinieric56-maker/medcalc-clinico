"""Prepare real-signal rendered development images without inventing annotations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ecg_av_annotation_source_audit import eligible_records, protected_ids


def select_records(audit, metadata, protected, limit):
    if limit < 1:
        raise ValueError("Positive image count required")
    proposed = audit["ptbxl_delineation"]["selection"]["eligible_ecg_ids"]
    # Independently recheck patient/fold exclusions, even for an existing inventory.
    checked = eligible_records([f"{int(i):05d}_hr" for i in proposed], metadata, protected)
    if checked["counts"].get("eligible_development", 0) != len(proposed):
        raise ValueError("Image inventory contains a protected or heldout record")
    ordered = sorted(proposed, key=lambda i: hashlib.sha256(f"R28_IMAGE_PREPARATION_V1:{i}".encode()).hexdigest())
    selected, patients = [], set()
    for ecg_id in ordered:
        row = metadata[str(ecg_id)]
        patient = row["patient_id"]
        if patient not in patients:
            patients.add(patient)
            selected.append(row)
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError("Insufficient distinct eligible patients")
    return selected


def display_windows():
    from ecg_validation_harness import LAYOUTS
    windows = [{"lead": lead, "row": row, "column": col,
                "start_s": col * 2.5, "end_s": (col + 1) * 2.5}
               for row, leads in enumerate(LAYOUTS["3x4"]) for col, lead in enumerate(leads)]
    windows.append({"lead": "II", "row": 3, "column": 0, "start_s": 0., "end_s": 10.,
                    "role": "CONTIGUOUS_RHYTHM_STRIP"})
    return windows


def prepare(audit_path, metadata_path, data_root, output, limit):
    import numpy as np
    import wfdb
    from ecg_adult_diagnostic_dev_benchmark import _ensure_record
    from ecg_validation_harness import LEADS, render_ecg_paper
    audit = json.loads(audit_path.read_text())
    if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != audit["metadata_sha256"]:
        raise ValueError("Metadata differs from annotation source audit")
    fast_path = Path(__file__).with_name("ecg_fast_gate_100_manifest.json")
    if hashlib.sha256(fast_path.read_bytes()).hexdigest() != audit["fast_manifest_sha256"]:
        raise ValueError("Protected panel differs from source audit")
    with metadata_path.open(newline="") as stream:
        metadata = {str(int(row["ecg_id"])): row for row in csv.DictReader(stream)}
    records = select_records(audit, metadata, protected_ids(json.loads(fast_path.read_text())), limit)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"role": "RENDERED_REAL_SIGNAL_IMAGE_PREPARATION_ONLY",
                "selection": "Fixed SHA-256 order, one record per patient, no outcome ranking",
                "annotation_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "metadata_sha256": audit["metadata_sha256"], "fast_manifest_sha256": audit["fast_manifest_sha256"],
                "paper_speed_mm_per_s": 25., "gain_mm_per_mv": 10., "pixels_per_mm": 4.,
                "layout": "3x4_WITH_II_RHYTHM_STRIP", "lead_display_windows": display_windows(),
                "annotation_alignment_verified": False, "expert_annotations_attached": False,
                "training_allowed": False, "clinical_ready": False, "records": [],
                "limitations": ["Rendered images of real waveforms, not clinical photographs/scans",
                                "Source mask time/class/lead schema remains unverified",
                                "Digitizer output must preserve missing samples and observed windows",
                                "No native waveform may be substituted for digitized training input"]}
    for row in records:
        filename = row["filename_hr"]
        parts = Path(filename).parts
        if len(parts) != 3 or parts[0] != "records500" or not parts[1].isdigit() or not parts[2].endswith("_hr"):
            raise ValueError("Unexpected authoritative record path")
        base = _ensure_record(data_root, filename)
        rec = wfdb.rdrecord(str(base), physical=True)
        if (rec.fs != 500 or rec.sig_len != 5000 or rec.n_sig != 12
                or {name.upper() for name in rec.sig_name} != {name.upper() for name in LEADS}
                or any(u != "mV" for u in rec.units)
                or not np.isfinite(rec.p_signal).all()):
            raise ValueError("Expected finite calibrated 10s twelve-lead PTB-XL waveform")
        lead_index = {name.upper(): i for i, name in enumerate(rec.sig_name)}
        signals = {lead: rec.p_signal[:, lead_index[lead.upper()]] for lead in LEADS}
        image = render_ecg_paper(signals, fs=500, layout="3x4", rhythm_strip=True,
                                 px_per_mm=4., speed_mm_per_s=25., gain_mm_per_mv=10.)
        name = f"ptbxl-{int(row['ecg_id']):05d}-3x4-strip.png"
        path = output / name
        image.save(path)
        manifest["records"].append({"ecg_id": int(row["ecg_id"]), "patient_id": row["patient_id"],
                                    "fold": int(row["strat_fold"]), "image": name,
                                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                    "native_dat_sha256": hashlib.sha256(base.with_suffix(".dat").read_bytes()).hexdigest(),
                                    "native_hea_sha256": hashlib.sha256(base.with_suffix(".hea").read_bytes()).hexdigest(),
                                    "image_width_px": image.width, "image_height_px": image.height})
        print(f"Prepared {name}", flush=True)
    (output / "image_preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-audit", type=Path, required=True)
    parser.add_argument("--ptbxl-metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    prepare(args.annotation_audit, args.ptbxl_metadata, args.data_root, args.output, args.limit)
