"""Audit temporal alignment from PTB-XL native samples to digitized lead II.

Development-only engineering audit. ECGDeli fiducials are algorithmic weak
labels, never expert truth. The audit estimates the native->digitized time map
from local native/digitized QRS morphology around ECGDeli R peaks, then measures
whether candidate P/R labels land on actually observed reconstructed samples.
It never trains a model or changes a clinical threshold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import _ensure_record
from ecg_av_ptbxlplus_weak_supervision_audit import _protected_patient_ids
from ecg_signal_reconstruction import (
    QUALITY_INTERPOLATED,
    QUALITY_MISSING,
    QUALITY_OBSERVED,
)


VERSION = "MEDCALC_R28_PTBXLPLUS_DIGITIZED_ALIGNMENT_AUDIT_V1"
NATIVE_FS = 500
QRS_HALF_WINDOW_S = 0.080
SEARCH_LAG_S = 0.120


def _parse_worker_outputs(items: list[str]) -> dict[int, Path]:
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("Worker output must be ECG_ID=PATH")
        key, value = item.split("=", 1)
        ecg_id = int(key)
        path = Path(value)
        if ecg_id in result or not value:
            raise ValueError("Duplicate or empty worker-output mapping")
        result[ecg_id] = path
    if not result:
        raise ValueError("At least one worker output is required")
    return result


def _worker_lead_ii(path: Path) -> tuple[np.ndarray, np.ndarray, int, dict]:
    payload = json.loads(path.read_text())
    signal_meta = payload.get("signal") or {}
    canonical = signal_meta.get("calibrated_digital_signal") or {}
    lead = (canonical.get("leads") or {}).get("II") or {}
    fs = int(lead.get("fs") or canonical.get("fs") or 0)
    if fs != NATIVE_FS:
        raise ValueError(f"Expected reconstructed lead II at 500 Hz, got {fs}")
    raw_signal = lead.get("signal_mv") or []
    signal = np.asarray(
        [np.nan if value is None else float(value) for value in raw_signal],
        dtype=np.float64,
    )
    quality = np.asarray(lead.get("quality_mask") or [], dtype=np.uint8)
    if signal.ndim != 1 or quality.shape != signal.shape or signal.size < 1000:
        raise ValueError("Invalid reconstructed lead II signal/quality contract")
    if not set(np.unique(quality)).issubset(
        {QUALITY_MISSING, QUALITY_INTERPOLATED, QUALITY_OBSERVED}
    ):
        raise ValueError("Unexpected quality-mask code")
    return signal, quality, fs, payload


def _native_lead_ii(metadata_row: dict, data_root: Path) -> tuple[np.ndarray, Path]:
    base = _ensure_record(data_root, str(metadata_row["filename_hr"]))
    record = wfdb.rdrecord(str(base), physical=True)
    if int(record.fs) != NATIVE_FS or int(record.sig_len) != 5000:
        raise ValueError("Expected PTB-XL high-resolution 500 Hz / 5000 samples")
    names = [str(name).upper() for name in record.sig_name]
    if "II" not in names:
        raise ValueError("Native PTB-XL record has no lead II")
    signal = np.asarray(record.p_signal[:, names.index("II")], dtype=np.float64)
    if signal.shape != (5000,) or not np.isfinite(signal).all():
        raise ValueError("Native lead II is not a finite 5000-sample waveform")
    return signal, base


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 8 or len(a) != len(b):
        return None
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if not np.isfinite(aa).all() or not np.isfinite(bb).all():
        return None
    aa = aa - np.mean(aa)
    bb = bb - np.mean(bb)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-12:
        return None
    return float(np.dot(aa, bb) / denom)


def _align_r_peak(
    native: np.ndarray,
    digitized: np.ndarray,
    quality: np.ndarray,
    source_sample: int,
) -> dict:
    half = int(round(QRS_HALF_WINDOW_S * NATIVE_FS))
    radius = int(round(SEARCH_LAG_S * NATIVE_FS))
    source_sample = int(source_sample)
    lo = source_sample - half
    hi = source_sample + half + 1
    if lo < 0 or hi > len(native):
        return {
            "source_sample": source_sample,
            "status": "NATIVE_EDGE_CENSORED",
            "lag_samples": None,
            "correlation": None,
        }

    reference = native[lo:hi]
    candidates = []
    for lag in range(-radius, radius + 1):
        dlo, dhi = lo + lag, hi + lag
        if dlo < 0 or dhi > len(digitized):
            continue
        observed = quality[dlo:dhi] == QUALITY_OBSERVED
        if int(observed.sum()) < max(8, int(np.ceil(0.70 * len(reference)))):
            continue
        corr = _corr(reference[observed], digitized[dlo:dhi][observed])
        if corr is not None:
            candidates.append((corr, lag, int(observed.sum())))
    if not candidates:
        return {
            "source_sample": source_sample,
            "status": "NO_OBSERVED_QRS_ALIGNMENT_WINDOW",
            "lag_samples": None,
            "correlation": None,
        }
    corr, lag, observed_n = max(candidates, key=lambda row: row[0])
    return {
        "source_sample": source_sample,
        "status": "ALIGNED",
        "lag_samples": int(lag),
        "lag_ms": round(1000.0 * lag / NATIVE_FS, 3),
        "correlation": round(float(corr), 6),
        "observed_samples_in_window": observed_n,
    }


def _fit_time_map(alignments: list[dict]) -> dict:
    rows = [
        row for row in alignments
        if row.get("status") == "ALIGNED"
        and row.get("lag_samples") is not None
        and np.isfinite(float(row.get("correlation")))
    ]
    if len(rows) < 2:
        return {
            "available": False,
            "reason": "INSUFFICIENT_ALIGNED_R_PEAKS",
            "aligned_r_peak_n": len(rows),
        }
    source = np.asarray([row["source_sample"] for row in rows], dtype=float)
    target = np.asarray(
        [row["source_sample"] + row["lag_samples"] for row in rows],
        dtype=float,
    )
    if float(np.ptp(source)) <= 1.0:
        return {
            "available": False,
            "reason": "INSUFFICIENT_TEMPORAL_SPAN",
            "aligned_r_peak_n": len(rows),
        }
    slope, intercept = np.polyfit(source, target, deg=1)
    predicted = slope * source + intercept
    residual = target - predicted
    return {
        "available": True,
        "aligned_r_peak_n": len(rows),
        "source_to_digitized_sample_slope": float(slope),
        "source_to_digitized_sample_intercept": float(intercept),
        "scale_error_ppm": float((slope - 1.0) * 1e6),
        "intercept_ms": float(intercept * 1000.0 / NATIVE_FS),
        "residual_median_abs_ms": float(
            np.median(np.abs(residual)) * 1000.0 / NATIVE_FS
        ),
        "residual_max_abs_ms": float(
            np.max(np.abs(residual)) * 1000.0 / NATIVE_FS
        ),
        "r_peak_correlation_median": float(
            np.median([row["correlation"] for row in rows])
        ),
        "r_peak_correlation_min": float(
            np.min([row["correlation"] for row in rows])
        ),
    }


def _event_support(
    events: list[dict],
    quality: np.ndarray,
    time_map: dict,
) -> dict:
    if not time_map.get("available"):
        return {
            "mapping_available": False,
            "event_n": len(events),
            "events": [],
        }
    slope = float(time_map["source_to_digitized_sample_slope"])
    intercept = float(time_map["source_to_digitized_sample_intercept"])
    rows = []
    for event in events:
        source_sample = int(event["sample"])
        mapped = int(round(slope * source_sample + intercept))
        if not 0 <= mapped < len(quality):
            state = "OUT_OF_RANGE"
        else:
            local = quality[max(0, mapped - 2): min(len(quality), mapped + 3)]
            if np.any(local == QUALITY_OBSERVED):
                state = "OBSERVED"
            elif np.any(local == QUALITY_INTERPOLATED):
                state = "INTERPOLATED_ONLY"
            else:
                state = "MISSING"
        rows.append({
            "kind": "P" if event["aux_note"] == "p-wave peak" else "R",
            "source_sample": source_sample,
            "mapped_digitized_sample": mapped,
            "mapped_time_s": round(mapped / NATIVE_FS, 6),
            "support": state,
        })
    counts = {}
    for row in rows:
        counts[row["support"]] = counts.get(row["support"], 0) + 1
    return {
        "mapping_available": True,
        "event_n": len(rows),
        "support_counts": dict(sorted(counts.items())),
        "observed_fraction": (
            sum(row["support"] == "OBSERVED" for row in rows) / len(rows)
            if rows else None
        ),
        "events": rows,
    }


def audit(
    source_report_path: Path,
    metadata_path: Path,
    data_root: Path,
    worker_outputs: dict[int, Path],
) -> dict:
    source_report = json.loads(source_report_path.read_text())
    if source_report.get("role") != "PTBXLPLUS_ECGDELI_WEAK_SUPERVISION_SOURCE_AUDIT_ONLY":
        raise ValueError("Unexpected weak-supervision source report")
    if source_report.get("guards", {}).get("training_allowed") is not False:
        raise ValueError("Source report unexpectedly permits training")

    with metadata_path.open(newline="") as stream:
        metadata = {int(row["ecg_id"]): row for row in csv.DictReader(stream)}
    protected_patients = {
        str(value)
        for value in _protected_patient_ids(pd.read_csv(metadata_path))
    }

    source_records = {int(row["ecg_id"]): row for row in source_report["records"]}
    if set(worker_outputs) != set(source_records):
        raise ValueError("Worker-output ids must exactly match source-report ids")

    records = []
    for ecg_id in sorted(source_records):
        meta = metadata.get(ecg_id)
        if not meta:
            raise ValueError(f"Metadata missing ecg_id {ecg_id}")
        if int(meta["strat_fold"]) not in range(1, 9):
            raise ValueError("Alignment audit accepts development folds 1-8 only")
        if str(meta["patient_id"]) in protected_patients:
            raise ValueError("Protected FAST-GATE patient entered alignment audit")

        native, native_base = _native_lead_ii(meta, data_root)
        digitized, quality, fs, worker_payload = _worker_lead_ii(
            worker_outputs[ecg_id]
        )
        lead_source = source_records[ecg_id]["leads"].get("II") or {}
        events = list(lead_source.get("candidate_peak_events") or [])
        p_events = [row for row in events if row.get("aux_note") == "p-wave peak"]
        r_events = [row for row in events if row.get("aux_note") == "R peak"]
        unexpected = [
            row for row in events
            if row.get("aux_note") not in {"p-wave peak", "R peak"}
        ]
        if unexpected or not r_events:
            raise ValueError("Candidate peak mapping is incomplete or malformed")

        r_alignment = [
            _align_r_peak(native, digitized, quality, int(row["sample"]))
            for row in r_events
        ]
        time_map = _fit_time_map(r_alignment)
        support = _event_support(events, quality, time_map)

        records.append({
            "ecg_id": ecg_id,
            "fold": int(meta["strat_fold"]),
            "native_record_sha256": hashlib.sha256(
                native.astype("<f8").tobytes()
            ).hexdigest(),
            "worker_output_sha256": hashlib.sha256(
                worker_outputs[ecg_id].read_bytes()
            ).hexdigest(),
            "weak_source_annotation_sha256": lead_source.get("sha256"),
            "digitized_signal_n": int(len(digitized)),
            "digitized_observed_fraction": float(
                np.mean(quality == QUALITY_OBSERVED)
            ),
            "candidate_p_peak_n": len(p_events),
            "candidate_r_peak_n": len(r_events),
            "r_peak_alignment": r_alignment,
            "time_map": time_map,
            "candidate_peak_support": support,
            "worker_status": worker_payload.get("status"),
        })

    maps = [row["time_map"] for row in records if row["time_map"].get("available")]
    support_rows = [
        event
        for row in records
        for event in row["candidate_peak_support"].get("events", [])
    ]
    return {
        "version": VERSION,
        "role": "PTBXLPLUS_ECGDELI_TO_DIGITIZED_TIME_ALIGNMENT_DEVELOPMENT_AUDIT_ONLY",
        "source_report_sha256": hashlib.sha256(
            source_report_path.read_bytes()
        ).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "records": records,
        "aggregate": {
            "record_n": len(records),
            "time_map_available_n": len(maps),
            "median_scale_error_ppm": (
                float(np.median([m["scale_error_ppm"] for m in maps]))
                if maps else None
            ),
            "median_intercept_ms": (
                float(np.median([m["intercept_ms"] for m in maps]))
                if maps else None
            ),
            "candidate_peak_n": len(support_rows),
            "candidate_peak_observed_fraction": (
                sum(row["support"] == "OBSERVED" for row in support_rows)
                / len(support_rows)
                if support_rows else None
            ),
        },
        "guards": {
            "training_allowed": False,
            "clinical_validation_allowed": False,
            "expert_ground_truth_claim_allowed": False,
            "clinical_fusion_allowed": False,
            "threshold_tuning_allowed": False,
        },
        "interpretation_guard": (
            "Alignment is an engineering property of rendered PTB-XL development images. "
            "ECGDeli points are algorithmic weak labels. These records cannot establish "
            "clinical event sensitivity, AV-block subtype performance or independent validation."
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-report", type=Path, required=True)
    ap.add_argument("--ptbxl-metadata", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--worker-output", action="append", default=[], help="ECG_ID=PATH")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = audit(
        args.source_report,
        args.ptbxl_metadata,
        args.data_root,
        _parse_worker_outputs(args.worker_output),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "role": report["role"],
        "aggregate": report["aggregate"],
        "records": [
            {
                "ecg_id": row["ecg_id"],
                "time_map": row["time_map"],
                "candidate_peak_support": {
                    k: v for k, v in row["candidate_peak_support"].items()
                    if k != "events"
                },
            }
            for row in report["records"]
        ],
        "training_allowed": False,
    }, indent=2))
