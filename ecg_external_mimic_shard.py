from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_signal_measurements import analyze_canonical_ecg

DATASET_ID = "mimic_iv_ecg"
BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _norm_lead(value: str) -> str:
    x = str(value or "").strip().upper().replace(" ", "")
    aliases = {
        "I": "I", "II": "II", "III": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3",
        "V4": "V4", "V5": "V5", "V6": "V6",
    }
    return aliases.get(x, str(value or "").strip())


def _remote_base(record_path: str) -> str:
    p = str(record_path or "").strip().replace("\\", "/").lstrip("/")
    marker = "files/"
    if marker in p:
        p = p[p.index(marker):]
    for suffix in (".hea", ".dat"):
        if p.endswith(suffix):
            p = p[: -len(suffix)]
    if not p.startswith("files/"):
        raise ValueError(f"Unexpected MIMIC record path: {record_path}")
    return f"{BASE_URL}/{p}"


def _download(url: str, target: Path, attempts: int = 4) -> None:
    last: Exception | None = None
    for attempt in range(int(attempts)):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MEDCALC-ECG-external-validation/1.0"},
            )
            with urllib.request.urlopen(req, timeout=90) as src:
                target.write_bytes(src.read())
            if target.stat().st_size <= 0:
                raise IOError(f"Empty download: {url}")
            return
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}: {last}")


def _load_record(record_path: str) -> tuple[np.ndarray, int, list[str]]:
    remote = _remote_base(record_path)
    with tempfile.TemporaryDirectory(prefix="medcalc_mimic_ecg_") as td:
        root = Path(td)
        name = Path(remote).name
        base = root / name
        _download(remote + ".hea", Path(str(base) + ".hea"))
        _download(remote + ".dat", Path(str(base) + ".dat"))
        record = wfdb.rdrecord(str(base))
        x = np.asarray(record.p_signal, dtype=float)
        fs = int(round(float(record.fs)))
        names = [_norm_lead(v) for v in (record.sig_name or [])]

    if x.ndim != 2:
        raise ValueError(f"Expected 2D WFDB signal, got {x.shape}")
    if x.shape[0] != 5000:
        raise ValueError(f"Expected 5000 samples, got {x.shape[0]}")
    if x.shape[1] != 12:
        raise ValueError(f"Expected 12 leads, got {x.shape[1]}")
    if fs != 500:
        raise ValueError(f"Expected 500 Hz, got {fs}")
    if set(names) != set(LEADS):
        raise ValueError(f"Unexpected lead set: {names}")

    order = [names.index(lead) for lead in LEADS]
    return x[:, order], fs, LEADS


def _canonical(
    matrix_mv: np.ndarray,
    *,
    fs: int,
    subject_id: str,
    study_id: str,
) -> dict[str, Any]:
    leads: dict[str, Any] = {}
    for idx, lead in enumerate(LEADS):
        sig = np.asarray(matrix_mv[:, idx], dtype=float)
        finite = np.isfinite(sig)
        leads[lead] = {
            "signal_mv": [
                float(v) if math.isfinite(float(v)) else None
                for v in sig
            ],
            "quality_mask": np.where(finite, 2, 0).astype(np.uint8).tolist(),
            "fs": int(fs),
            "duration_s": float(len(sig) / fs),
            "source": "MIMIC_IV_ECG_FROZEN_EXTERNAL_DIGITAL_SIGNAL",
            "confidence": 1.0,
            "status": "MEASURED",
        }
    return {
        "contract": "MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "fs": int(fs),
        "leads": leads,
        "lead_order": list(LEADS),
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "source": "DIGITAL_EXTERNAL_REFERENCE",
            "confidence": 1.0,
        },
        "validation_provenance": {
            "dataset_id": DATASET_ID,
            "subject_id": str(subject_id),
            "study_id": str(study_id),
            "frozen_external": True,
            "machine_measurements_available_to_inference": False,
            "cardiologist_reports_available_to_inference": False,
        },
    }


def _value(metric: Any) -> float | None:
    if isinstance(metric, dict):
        metric = metric.get("value")
    try:
        x = float(metric)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _extract(result: dict[str, Any]) -> dict[str, Any]:
    g = result.get("global") or {}
    axis = result.get("axis") or {}
    consensus = result.get("measurement_consensus") or {}
    states = consensus.get("measurement_states") or {}
    return {
        "engine_hr_bpm": _value(g.get("heart_rate_bpm")),
        "engine_pr_ms": _value(g.get("pr_ms")),
        "engine_qrs_ms": _value(g.get("qrs_ms")),
        "engine_qt_ms": _value(g.get("qt_ms")),
        "engine_qtc_fridericia_ms": _value(g.get("qtc_fridericia_ms")),
        "engine_qrs_axis_deg": _value(axis.get("degrees")),
        "state_hr": str(states.get("heart_rate_bpm") or ""),
        "state_pr": str(states.get("pr_ms") or ""),
        "state_qrs": str(states.get("qrs_ms") or ""),
        "state_qt": str(states.get("qt_ms") or ""),
        "remeasure_required": bool(consensus.get("remeasure_required")),
        "overall_measurement_quality": consensus.get(
            "overall_measurement_quality"
        ),
    }


def run(manifest: Path, output: Path) -> dict[str, Any]:
    rows = pd.read_csv(manifest, dtype=str)
    required = {"subject_id", "study_id", "record_path"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Manifest missing columns: {required - set(rows.columns)}")

    out_rows: list[dict[str, Any]] = []
    for idx, row in rows.iterrows():
        subject_id = str(row["subject_id"])
        study_id = str(row["study_id"])
        record_path = str(row["record_path"])
        base = {
            "subject_id": subject_id,
            "study_id": study_id,
        }
        try:
            x, fs, _ = _load_record(record_path)
            result = analyze_canonical_ecg(
                _canonical(
                    x,
                    fs=fs,
                    subject_id=subject_id,
                    study_id=study_id,
                )
            )
            out_rows.append({
                **base,
                **_extract(result),
                "analysis_error": "",
            })
        except Exception as exc:
            out_rows.append({
                **base,
                "analysis_error": f"{type(exc).__name__}:{exc}",
            })

        if (idx + 1) % 25 == 0 or idx + 1 == len(rows):
            print(
                f"MIMIC_EXTERNAL_INFERENCE {idx + 1}/{len(rows)}",
                flush=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(output, index=False)
    summary = {
        "dataset_id": DATASET_ID,
        "records": int(len(out_rows)),
        "analysis_errors": int(
            sum(bool(str(r.get("analysis_error") or "")) for r in out_rows)
        ),
        "machine_measurements_opened": False,
        "cardiologist_reports_opened": False,
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run(args.manifest, args.output)


if __name__ == "__main__":
    main()
