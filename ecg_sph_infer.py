from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd

from ecg_signal_measurements import analyze_canonical_ecg
from ecg_validation_guard import assert_external_dataset, load_registry


LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
FS = 500
WINDOW_SAMPLES = 5000


def canonical(record_path: Path) -> Dict[str, Any]:
    with h5py.File(record_path, "r") as fh:
        data = np.asarray(fh["ecg"][:, :WINDOW_SAMPLES], dtype=float)
    if data.ndim != 2 or data.shape[0] != 12:
        raise ValueError(f"{record_path}: expected 12xL, got {data.shape}")
    if data.shape[1] < WINDOW_SAMPLES:
        raise ValueError(f"{record_path}: shorter than required 10 s: {data.shape}")

    leads = {}
    for i, lead in enumerate(LEADS):
        x = np.asarray(data[i], dtype=float)
        finite = np.isfinite(x)
        leads[lead] = {
            "signal_mv": [float(v) if math.isfinite(float(v)) else None for v in x],
            "quality_mask": np.where(finite, 2, 0).astype(np.uint8).tolist(),
            "fs": FS,
            "duration_s": 10.0,
            "source": "SPH_FROZEN_DIGITAL_SIGNAL_FIRST_10S",
            "confidence": 1.0,
            "status": "MEASURED",
        }
    return {
        "contract": "MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "fs": FS,
        "leads": leads,
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "source": "DIGITAL_EXTERNAL_REFERENCE_MV",
            "confidence": 1.0,
        },
        "validation_provenance": {
            "dataset_id": "sph",
            "frozen_external": True,
            "window": "FIRST_10_SECONDS_LABEL_BLIND",
        },
    }


def prediction(measurements: Dict[str, Any]) -> Dict[str, Any]:
    reasoner = measurements.get("specialist_reasoning") or {}
    primary = reasoner.get("primary_rhythm") or {}
    primary_code = str(primary.get("code") or "")
    conduction = {
        str(row.get("code") or "")
        for row in (reasoner.get("conduction_findings") or [])
    }
    av = reasoner.get("av_conduction_finding") or {}
    av_code = str(av.get("code") or "")

    return {
        "pred_AF": primary_code == "AF_COMPATIBLE",
        "pred_AFL": primary_code == "FLUTTER_OR_AT_COMPATIBLE",
        "pred_SB": primary_code == "SINUS_BRADYCARDIA_COMPATIBLE",
        "pred_ST": primary_code == "SINUS_TACHYCARDIA_COMPATIBLE",
        "pred_RBBB": "RBBB_MORPHOLOGY_COMPATIBLE" in conduction,
        "pred_LBBB": "LBBB_MORPHOLOGY_COMPATIBLE" in conduction,
        "pred_LAFB": "LAFB_COMPATIBLE" in conduction,
        "pred_1DAVB": av_code == "FIRST_DEGREE_AV_DELAY_COMPATIBLE",
        "primary_rhythm_code": primary_code,
        "primary_rhythm_confidence": primary.get("confidence"),
        "publication_allowed": bool(reasoner.get("publication_allowed")),
        "consistency_status": reasoner.get("consistency_status"),
        "remeasure_required": bool(
            (measurements.get("measurement_consensus") or {}).get("remeasure_required")
        ),
        "heart_rate_bpm": ((measurements.get("global") or {}).get("heart_rate_bpm") or {}).get("value"),
        "pr_ms": ((measurements.get("global") or {}).get("pr_ms") or {}).get("value"),
        "qrs_ms": ((measurements.get("global") or {}).get("qrs_ms") or {}).get("value"),
        "qt_ms": ((measurements.get("global") or {}).get("qt_ms") or {}).get("value"),
        "qtc_fridericia_ms": ((measurements.get("global") or {}).get("qtc_fridericia_ms") or {}).get("value"),
        "axis_deg": (measurements.get("axis") or {}).get("degrees"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    assert_external_dataset("sph", load_registry())

    manifest = args.artifact_dir / "manifest.csv"
    records_tar = args.artifact_dir / "records.tar"
    extract_dir = args.artifact_dir / "records"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(records_tar, "r:*") as tf:
        tf.extractall(extract_dir)

    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    out = []
    for i, row in enumerate(rows):
        ecg_id = str(row["ECG_ID"])
        fp = extract_dir / f"{ecg_id}.h5"
        measurements = analyze_canonical_ecg(canonical(fp))
        out.append({"ECG_ID": ecg_id, **prediction(measurements)})
        if (i + 1) % 20 == 0 or i + 1 == len(rows):
            print(f"SPH_INFERENCE {i+1}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(args.output, index=False)
    print(json.dumps({
        "records_inferred": len(out),
        "gold_labels_accessed": False,
        "dataset_id": "sph",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
