from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ecg_signal_measurements import analyze_canonical_ecg


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 500
WINDOW_SAMPLES = 5000


def _load_signal(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as fh:
        if "ecg" in fh:
            x = np.asarray(fh["ecg"], dtype=float)
        else:
            keys = list(fh.keys())
            if not keys:
                raise ValueError(f"No datasets in {path}")
            x = np.asarray(fh[keys[0]], dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D ECG, got {x.shape}")
    if x.shape[0] == 12:
        pass
    elif x.shape[1] == 12:
        x = x.T
    else:
        raise ValueError(f"{path.name}: expected 12 leads, got {x.shape}")
    if x.shape[1] < WINDOW_SAMPLES:
        raise ValueError(f"{path.name}: fewer than 10 seconds ({x.shape[1]} samples)")
    return x[:, :WINDOW_SAMPLES]


def _canonical(x: np.ndarray, record_id: str) -> dict:
    leads = {}
    for idx, lead in enumerate(LEADS):
        sig = np.asarray(x[idx], dtype=float)
        finite = np.isfinite(sig)
        leads[lead] = {
            "signal_mv": [float(v) if math.isfinite(float(v)) else None for v in sig],
            "quality_mask": np.where(finite, 2, 0).astype(np.uint8).tolist(),
            "fs": FS,
            "duration_s": WINDOW_SAMPLES / FS,
            "source": "SPH_FROZEN_EXTERNAL_DIGITAL_SIGNAL",
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
            "source": "DIGITAL_EXTERNAL_REFERENCE",
            "confidence": 1.0,
        },
        "validation_provenance": {
            "dataset_id": "sph",
            "record_id": record_id,
            "frozen_external": True,
            "window_seconds": 10.0,
            "units": "mV",
        },
    }


def _predict(measurements: dict) -> dict:
    reasoner = measurements.get("specialist_reasoning") or {}
    summary = reasoner.get("diagnostic_summary") or {}
    findings = [
        row for row in summary.get("findings") or []
        if bool(row.get("publishable"))
    ]
    codes = {str(row.get("code") or "") for row in findings}

    av2_codes = {
        "MOBITZ_I_WENCKEBACH_COMPATIBLE",
        "MOBITZ_II_COMPATIBLE",
        "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
        "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
    }
    predictions = {
        "AF": "AF_COMPATIBLE" in codes,
        "FLUTTER": "FLUTTER_OR_AT_COMPATIBLE" in codes,
        "SINUS_BRADY": "SINUS_BRADYCARDIA_COMPATIBLE" in codes,
        "SINUS_TACHY": "SINUS_TACHYCARDIA_COMPATIBLE" in codes,
        "RBBB_COMPLETE": "RBBB_MORPHOLOGY_COMPATIBLE" in codes,
        "LBBB": "LBBB_MORPHOLOGY_COMPATIBLE" in codes,
        "LAFB": "LAFB_COMPATIBLE" in codes,
        "LPFB": "LPFB_COMPATIBLE" in codes,
        "AVB1": "FIRST_DEGREE_AV_DELAY_COMPATIBLE" in codes,
        "AVB2": bool(codes & av2_codes),
        "AVB3": "COMPLETE_AV_BLOCK_COMPATIBLE" in codes,
        "WPW": "VENTRICULAR_PREEXCITATION_COMPATIBLE" in codes,
    }
    global_m = measurements.get("global") or {}
    return {
        "predictions": predictions,
        "finding_codes": sorted(codes),
        "publication_allowed": bool(summary.get("publication_allowed")),
        "abstention_n": len(summary.get("abstentions") or []),
        "remeasure_required": bool(
            (measurements.get("measurement_consensus") or {}).get("remeasure_required")
        ),
        "signal_integrity_quality": (
            measurements.get("signal_integrity") or {}
        ).get("overall_quality"),
        "measurement_consensus_quality": (
            measurements.get("measurement_consensus") or {}
        ).get("overall_measurement_quality"),
        "heart_rate_bpm": (global_m.get("heart_rate_bpm") or {}).get("value"),
        "pr_ms": (global_m.get("pr_ms") or {}).get("value"),
        "qrs_ms": (global_m.get("qrs_ms") or {}).get("value"),
        "qt_ms": (global_m.get("qt_ms") or {}).get("value"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = pd.read_csv(args.root / "manifest.csv", dtype=str)
    rows = []
    for i, row in manifest.iterrows():
        record_id = str(row["record_id"])
        base = {
            "record_id": record_id,
            "patient_id": str(row["patient_id"]),
        }
        try:
            signal = _load_signal(args.root / "records" / f"{record_id}.h5")
            result = analyze_canonical_ecg(_canonical(signal, record_id))
            pred = _predict(result)
            flat = {k: v for k, v in pred.items() if k not in {"predictions", "finding_codes"}}
            flat["finding_codes"] = "|".join(pred["finding_codes"])
            for target, value in pred["predictions"].items():
                flat[f"pred_{target}"] = bool(value)
            rows.append({**base, **flat, "analysis_error": ""})
        except Exception as exc:
            failed = {
                **base,
                "publication_allowed": False,
                "abstention_n": 1,
                "remeasure_required": True,
                "analysis_error": f"{type(exc).__name__}:{exc}",
            }
            for target in (
                "AF","FLUTTER","SINUS_BRADY","SINUS_TACHY","RBBB_COMPLETE",
                "LBBB","LAFB","LPFB","AVB1","AVB2","AVB3","WPW"
            ):
                failed[f"pred_{target}"] = False
            rows.append(failed)
        if (i + 1) % 25 == 0 or i + 1 == len(manifest):
            print(f"SPH_INFERENCE {i + 1}/{len(manifest)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(json.dumps({
        "records": len(rows),
        "analysis_errors": sum(bool(str(row.get("analysis_error") or "")) for row in rows),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
