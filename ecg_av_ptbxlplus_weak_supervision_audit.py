"""Audit PTB-XL+ ECGDeli fiducials as a possible weak-supervision source.

This tool is deliberately non-training. ECGDeli annotations are algorithmic
fiducial points, not expert ground truth. The audit verifies development
eligibility, downloads only requested per-lead annotation files, decodes the
WFDB annotation payload, and records the raw symbols/auxiliary labels without
inventing semantic mappings.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests
import wfdb


VERSION = "MEDCALC_R28_PTBXLPLUS_WEAK_SUPERVISION_AUDIT_V1"
PTBXL_PLUS_VERSION = "1.0.1"
BASE = (
    "https://physionet.org/files/ptb-xl-plus/"
    f"{PTBXL_PLUS_VERSION}/fiducial_points/ecgdeli"
)
DEFAULT_IDS = (70, 2188, 484)
DEFAULT_LEADS = ("II",)


def _protected_patient_ids(metadata: pd.DataFrame) -> set:
    manifest = json.loads(
        Path(__file__).with_name("ecg_fast_gate_100_manifest.json").read_text()
    )
    protected_ids = {int(row["ecg_id"]) for row in manifest["cases"]}
    resolved = metadata.loc[metadata.ecg_id.isin(protected_ids)]
    if len(resolved) != len(protected_ids):
        raise ValueError("Metadata does not resolve every FAST-GATE record")
    return set(resolved.patient_id.tolist())


def _folder(ecg_id: int) -> str:
    if ecg_id <= 0:
        raise ValueError("ECG id must be positive")
    return f"{(ecg_id // 1000) * 1000:05d}"


def annotation_url(ecg_id: int, lead: str) -> str:
    if not lead or "/" in lead or "\\" in lead:
        raise ValueError("Invalid lead")
    return (
        f"{BASE}/{_folder(ecg_id)}/"
        f"{ecg_id:05d}_points_lead_{lead}.atr"
    )


def _download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_annotation(path: Path) -> dict:
    record_name = str(path.with_suffix(""))
    ann = wfdb.rdann(record_name, extension="atr")
    samples = [int(x) for x in ann.sample.tolist()]
    symbols = [str(x) for x in ann.symbol]
    aux = [str(x) for x in ann.aux_note]
    if not (len(samples) == len(symbols) == len(aux)):
        raise ValueError("WFDB annotation arrays have inconsistent lengths")
    if samples != sorted(samples):
        raise ValueError("WFDB annotations are not monotone in sample index")

    events = [
        {"sample": sample, "symbol": symbol, "aux_note": note}
        for sample, symbol, note in zip(samples, symbols, aux)
    ]
    duplicate_sample_n = sum(
        count - 1 for count in Counter(samples).values() if count > 1
    )
    return {
        "annotation_n": len(events),
        "sample_min": min(samples) if samples else None,
        "sample_max": max(samples) if samples else None,
        "within_ptbxl_hr_5000_sample_capture": bool(
            samples and min(samples) >= 0 and max(samples) < 5000
        ),
        "symbol_counts": dict(sorted(Counter(symbols).items())),
        "aux_note_counts": dict(sorted(Counter(aux).items())),
        "duplicate_sample_n": duplicate_sample_n,
        "events_preview": events[:80],
    }


def audit(
    metadata_path: Path,
    output_root: Path,
    record_ids: tuple[int, ...],
    leads: tuple[str, ...],
) -> dict:
    metadata = pd.read_csv(metadata_path)
    required = {"ecg_id", "patient_id", "strat_fold"}
    if not required.issubset(metadata.columns):
        raise ValueError("PTB-XL metadata missing required columns")
    protected_patients = _protected_patient_ids(metadata)

    rows = []
    for ecg_id in record_ids:
        match = metadata.loc[metadata.ecg_id == ecg_id]
        if len(match) != 1:
            raise ValueError(f"Unresolved or duplicate PTB-XL ecg_id {ecg_id}")
        row = match.iloc[0]
        fold = int(row.strat_fold)
        if fold not in range(1, 9):
            raise ValueError(f"Record {ecg_id} is not in development folds 1-8")
        if row.patient_id in protected_patients:
            raise ValueError(f"Record {ecg_id} belongs to a protected patient")

        per_lead = {}
        for lead in leads:
            url = annotation_url(ecg_id, lead)
            target = output_root / _folder(ecg_id) / Path(url).name
            digest = _download(url, target)
            per_lead[lead] = {
                "url": url,
                "sha256": digest,
                **_read_annotation(target),
            }
        rows.append({
            "ecg_id": int(ecg_id),
            "patient_id_sha256": hashlib.sha256(
                str(row.patient_id).encode()
            ).hexdigest(),
            "strat_fold": fold,
            "leads": per_lead,
        })

    raw_aux = Counter()
    raw_symbols = Counter()
    for row in rows:
        for item in row["leads"].values():
            raw_aux.update(item["aux_note_counts"])
            raw_symbols.update(item["symbol_counts"])

    return {
        "version": VERSION,
        "role": "PTBXLPLUS_ECGDELI_WEAK_SUPERVISION_SOURCE_AUDIT_ONLY",
        "source": {
            "dataset": "PTB-XL+",
            "version": PTBXL_PLUS_VERSION,
            "provider": "PhysioNet",
            "annotation_method": "ECGDeli",
            "annotation_truth_status": "ALGORITHMIC_FIDUCIALS_NOT_EXPERT_GROUND_TRUTH",
        },
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "record_ids": list(record_ids),
        "leads": list(leads),
        "records": rows,
        "aggregate_raw_symbol_counts": dict(sorted(raw_symbols.items())),
        "aggregate_raw_aux_note_counts": dict(sorted(raw_aux.items())),
        "published_context": {
            "ptbxl_hr_sampling_hz": 500,
            "ptbxl_hr_capture_samples": 5000,
            "fiducial_storage": "WFDB-compatible annotation files",
            "lead_specific_files": True,
        },
        "guards": {
            "training_allowed": False,
            "clinical_validation_allowed": False,
            "expert_ground_truth_claim_allowed": False,
            "clinical_fusion_allowed": False,
            "protected_patient_use_allowed": False,
        },
        "next_gate": (
            "Review decoded raw aux_note semantics and temporal alignment. "
            "Only after an explicit mapping contract may this source be proposed "
            "for pseudo-label pretraining; never use it as expert validation truth."
        ),
    }


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("Record ids must be a non-empty unique list")
    return result


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(x.strip() for x in value.split(",") if x.strip())
    if not result or len(set(result)) != len(result):
        raise ValueError("Leads must be a non-empty unique list")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ptbxl-metadata", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--record-ids",
        default=",".join(str(x) for x in DEFAULT_IDS),
    )
    ap.add_argument("--leads", default=",".join(DEFAULT_LEADS))
    args = ap.parse_args()
    report = audit(
        args.ptbxl_metadata,
        args.source_root,
        _parse_csv_ints(args.record_ids),
        _parse_csv_strings(args.leads),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "role": report["role"],
        "record_ids": report["record_ids"],
        "leads": report["leads"],
        "raw_symbols": report["aggregate_raw_symbol_counts"],
        "raw_aux_notes": report["aggregate_raw_aux_note_counts"],
        "training_allowed": report["guards"]["training_allowed"],
    }, indent=2))
