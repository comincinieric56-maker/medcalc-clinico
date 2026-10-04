"""Offline audit of acquired annotation sources; never produces training labels."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

SOURCE_HASHES = {
    "AF.txt": "3cd570bdfd819097b49e5dce3e14cc4a5c04c35d01c38bf597a2c0f212fedac4",
    "others.txt": "71c8562fd929f90b24e7504d54f2ae17b2ec7e7ac96c4f0ffcfdf2d45064b496",
    "AF.npy": "ed4e6ae8e4a1cbbe2c5b7d2ebefb61785c97812c72a8e6b07201de248a784146",
    "others.npy": "d9d4e92328b54a6fc3f10f2e03034625d026e3c776b1ff872a7d159bfa635d58",
    "isp.zip": "a3ccade908b1cd90d1db63655daf43e025975f580dee2fae1fa8f178cfc18fdf",
}
HF_REVISION = "c9d0577dc2bb82a6738e6d5dce1451aae4b09904"


def acquire(root):
    """Download pinned public bytes, verify hashes before making them available."""
    import requests
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_HASHES.items():
        path = root / name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Existing source hash mismatch: {name}")
            continue
        url = ("https://zenodo.org/records/14679837/files/isp_delineation_dataset.zip?download=1"
               if name == "isp.zip" else
               f"https://huggingface.co/datasets/figureli/ptb-xl-ecg-delineation/resolve/{HF_REVISION}/{name}")
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            digest = hashlib.sha256()
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        digest.update(chunk)
                        stream.write(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"Downloaded source hash mismatch: {name}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def protected_ids(value):
    if isinstance(value, dict):
        result = {str(value["ecg_id"])} if "ecg_id" in value else set()
        for child in value.values():
            result |= protected_ids(child)
        return result
    if isinstance(value, list):
        return set().union(*(protected_ids(child) for child in value))
    return set()


def eligible_records(record_names, metadata, protected):
    if not protected.issubset(metadata):
        raise ValueError("Metadata does not resolve protected records")
    patients = {metadata[key]["patient_id"] for key in protected}
    counts, eligible, labels = Counter(), [], Counter()
    seen = set()
    for name in record_names:
        if not re.fullmatch(r"\d{5}_hr", name):
            raise ValueError("Invalid source record identifier")
        key = str(int(name.split("_")[0]))
        if key in seen or key not in metadata:
            raise ValueError("Duplicate or unresolved source record")
        seen.add(key)
        row = metadata[key]
        fold = int(row["strat_fold"])
        if fold not in range(1, 11) or not row["patient_id"]:
            raise ValueError("Invalid fold or missing patient identity")
        reason = ("heldout_fold" if fold >= 9 else
                  "protected_patient" if row["patient_id"] in patients else
                  "eligible_development")
        counts[reason] += 1
        if reason == "eligible_development":
            eligible.append(int(key))
            codes = ast.literal_eval(row["scp_codes"])
            labels.update(k for k in codes if k in ("1AVB", "AVB2", "AVB3"))
    return {"counts": dict(counts), "eligible_ecg_ids": sorted(eligible),
            "eligible_av_diagnostic_labels": dict(labels),
            "eligibility_is_not_annotation_approval": True}


def inspect_mask(path, records_n):
    mask = np.load(path, allow_pickle=False, mmap_mode="r")
    if mask.ndim != 2 or mask.shape[0] != records_n * 12:
        raise ValueError("Mask rows do not match twelve leads per source record")
    if not np.isfinite(mask).all() or not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Mask contains nonfinite or noninteger labels")
    values, counts = np.unique(mask, return_counts=True)
    if not set(values).issubset({0, 1, 2, 3}):
        raise ValueError("Unexpected mask class")
    return {"shape": list(mask.shape), "dtype": str(mask.dtype),
            "label_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            "time_mapping_verified": False, "class_mapping_verified": False,
            "lead_order_verified": False}


def inspect_isp(path):
    counts, fs_counts, event_classes, invalid = Counter(), Counter(), Counter(), Counter()
    fingerprints = {"train": set(), "test": set()}
    with zipfile.ZipFile(path) as archive:
        for split in ("train", "test"):
            csv_name = f"isp_delineation_dataset/{split}_isp_delineation_data.csv"
            rows = csv.DictReader(io.StringIO(archive.read(csv_name).decode()))
            seen = set()
            for row in rows:
                name = row["file_name"]
                if not name.isdecimal() or name in seen:
                    raise ValueError("Duplicate or invalid ISP record within split")
                seen.add(name)
                base = f"isp_delineation_dataset/{split}_data/{name}"
                head = archive.read(base + ".hea").decode().splitlines()[0].split()
                if len(head) != 4 or int(head[1]) != 12:
                    raise ValueError("Invalid ISP header")
                fs, samples = float(head[2]), int(head[3])
                if not np.isfinite(fs) or fs <= 0 or samples <= 0:
                    raise ValueError("Invalid sampling frequency or length")
                fs_counts[str(fs)] += 1
                spans = ast.literal_eval(row["target"])
                invalid_record = False
                for span in spans:
                    if (not isinstance(span, tuple) or len(span) != 3
                            or any(type(v) is not int for v in span)
                            or span[0] not in (0, 1, 2)):
                        raise ValueError("Malformed ISP boundary annotation")
                    if not 0 <= span[1] < span[2] <= samples:
                        invalid["out_of_capture" if span[1] < 0 or span[2] > samples
                                else "empty_or_reversed"] += 1
                        invalid_record = True
                        continue
                    event_classes[str(span[0])] += 1
                invalid["records_with_invalid_spans"] += int(invalid_record)
                fingerprints[split].add(hashlib.sha256(archive.read(base + ".dat")).hexdigest())
                counts[split] += 1
    return {"records_by_publisher_split": dict(counts), "sampling_frequencies_hz": dict(fs_counts),
            "boundary_class_counts": dict(event_classes),
            "invalid_annotation_counts": dict(invalid), "invalid_spans_clipped_or_repaired": False,
            "cross_split_identical_dat_files": len(fingerprints["train"] & fingerprints["test"]),
            "patient_separation_verified": False, "expert_peak_annotations_present": False,
            "waveform_origin": "NATIVE", "training_allowed": False,
            "blockers": ["No patient identifiers in published annotation tables",
                         "Invalid source spans require provenance review; never silently clip",
                         "Onset/offset spans do not provide expert P/R peak targets",
                         "Native input has not passed through the image digitizer",
                         "Dataset is outside the current PTB-XL training allowlist"]}


def audit(root, metadata_path, fast_path):
    hashes = {}
    for name, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source hash mismatch: {name}")
        hashes[name] = actual
    with metadata_path.open(newline="") as stream:
        metadata = {str(int(row["ecg_id"])): row for row in csv.DictReader(stream)}
    names, masks = [], {}
    for group in ("AF", "others"):
        group_names = (root / f"{group}.txt").read_text().splitlines()
        names.extend(group_names)
        masks[group] = inspect_mask(root / f"{group}.npy", len(group_names))
    selection = eligible_records(names, metadata, protected_ids(json.loads(fast_path.read_text())))
    return {"role": "ANNOTATION_SOURCE_ACQUISITION_AUDIT_ONLY", "source_sha256": hashes,
            "ptbxl_annotation_revision": HF_REVISION,
            "isp_doi": "10.5281/zenodo.14679837",
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "fast_manifest_sha256": hashlib.sha256(fast_path.read_bytes()).hexdigest(),
            "ptbxl_delineation": {"url": "https://huggingface.co/datasets/figureli/ptb-xl-ecg-delineation",
                                  "masks": masks, "selection": selection, "training_allowed": False,
                                  "blockers": ["Unspecified class, time and lead-row mappings",
                                               "No expert peak or AV-subtype annotations",
                                               "Image alignment/digitization not performed"]},
            "isp": inspect_isp(root / "isp.zip"), "clinical_ready": False,
            "training_manifest_created": False, "weights_or_thresholds_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Acquire pinned public source files")
    parser.add_argument("--ptbxl-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.download:
        acquire(args.source_root)
    report = audit(args.source_root, args.ptbxl_metadata,
                   Path(__file__).with_name("ecg_fast_gate_100_manifest.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"ptbxl": report["ptbxl_delineation"]["selection"]["counts"],
                      "isp": report["isp"], "training_allowed": False}, indent=2))
