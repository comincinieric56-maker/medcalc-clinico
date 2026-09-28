from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_signal_measurements import analyze_canonical_ecg


STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _legacy_v1_would_remeasure(consensus: dict[str, Any]) -> tuple[bool, list[str]]:
    targets: list[str] = []
    for name, item in (consensus.get("metrics") or {}).items():
        if str(item.get("status") or "") in {"DISCORDANT", "NO_CANONICAL_VALUE"}:
            targets.append(str(name))
    r = consensus.get("r_peak_verification") or {}
    try:
        score = float(r.get("aggregate_agreement"))
        n = int(r.get("evaluable_lead_n") or 0)
        if math.isfinite(score) and score < 0.65 and n >= 2:
            targets.append("r_peaks")
    except Exception:
        pass
    return bool(targets), sorted(set(targets))


def _to_mv(values: np.ndarray, unit: str | None) -> np.ndarray:
    unit_norm = str(unit or "mV").strip().lower().replace("μ", "u").replace("µ", "u")
    if unit_norm in {"mv", "millivolt", "millivolts"}:
        return values.astype(float)
    if unit_norm in {"uv", "microvolt", "microvolts"}:
        return values.astype(float) / 1000.0
    if unit_norm in {"v", "volt", "volts"}:
        return values.astype(float) * 1000.0
    return values.astype(float)


def canonical_from_wfdb(record: Any) -> dict[str, Any]:
    fs = int(round(float(record.fs)))
    matrix = np.asarray(record.p_signal, dtype=float)
    names = [str(x) for x in record.sig_name]
    units = list(record.units or ["mV"] * len(names))

    leads: dict[str, Any] = {}
    for lead in STANDARD_LEADS:
        if lead not in names:
            leads[lead] = {
                "lead": lead,
                "signal_mv": [],
                "quality_mask": [],
                "fs": fs,
                "duration_s": 0.0,
                "source": "PTB_XL_DEVELOPMENT_NATIVE",
                "status": "UNAVAILABLE",
                "confidence": 0.0,
            }
            continue
        idx = names.index(lead)
        x = _to_mv(matrix[:, idx], units[idx] if idx < len(units) else "mV")
        q = np.where(np.isfinite(x), 2, 0).astype(np.uint8)
        leads[lead] = {
            "lead": lead,
            "signal_mv": [None if not np.isfinite(v) else float(v) for v in x],
            "quality_mask": [int(v) for v in q],
            "fs": fs,
            "duration_s": float(len(x) / fs),
            "source": "PTB_XL_DEVELOPMENT_NATIVE",
            "status": "MEASURABLE" if np.isfinite(x).mean() >= 0.95 else "PARTIAL",
            "confidence": float(np.isfinite(x).mean()),
        }

    return {
        "version": "PTB_XL_DEVELOPMENT_CANONICAL_V1",
        "source": "PTB_XL_DEVELOPMENT_NATIVE",
        "fs": fs,
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "confidence": 1.0,
        },
        "uncertainty": {
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "source": "NATIVE_DIGITAL_DEVELOPMENT_SIGNAL",
        },
        "leads": leads,
        "lead_order": list(STANDARD_LEADS),
    }


def _finite_value(d: dict[str, Any], *path: str) -> float | None:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    try:
        x = float(cur)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def run_benchmark(manifest: Path, records_root: Path, output: Path) -> None:
    df = pd.read_csv(manifest)
    rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    metric_state_counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("pr_ms", "qrs_ms", "qt_ms", "p_duration_ms")
    }
    conversion_counts: Counter[str] = Counter()

    for _, row in df.iterrows():
        ecg_id = int(row["ecg_id"])
        rel = str(row["filename_hr"])
        base = records_root / rel
        result_row: dict[str, Any] = {
            "ecg_id": ecg_id,
            "patient_id": int(row["patient_id"]) if "patient_id" in row and pd.notna(row["patient_id"]) else None,
            "filename_hr": rel,
        }
        try:
            record = wfdb.rdrecord(str(base))
            canonical = canonical_from_wfdb(record)
            analysis = analyze_canonical_ecg(canonical)
            consensus = analysis.get("measurement_consensus") or {}
            legacy_remeasure, legacy_targets = _legacy_v1_would_remeasure(consensus)

            states = dict(consensus.get("measurement_states") or {})
            for metric, state in states.items():
                metric_state_counts.setdefault(metric, Counter())[str(state or "UNKNOWN")] += 1
                state_counts[str(state or "UNKNOWN")] += 1

            v2_remeasure = bool(consensus.get("remeasure_required"))
            v2_remeasure_targets = list(consensus.get("remeasure_targets") or [])
            unusable = list(consensus.get("unusable_targets") or [])
            uncertain = list(consensus.get("uncertain_targets") or [])
            unmeasurable = list(consensus.get("unmeasurable_targets") or [])

            if legacy_remeasure and not v2_remeasure:
                if uncertain:
                    conversion_counts["V1_REMEASURE_TO_V2_UNCERTAINTY"] += 1
                elif unmeasurable:
                    conversion_counts["V1_REMEASURE_TO_V2_UNMEASURABLE"] += 1
                else:
                    conversion_counts["V1_REMEASURE_TO_V2_USABLE"] += 1
            elif legacy_remeasure and v2_remeasure:
                conversion_counts["V1_REMEASURE_REMAINS_V2_REMEASURE"] += 1
            elif not legacy_remeasure and v2_remeasure:
                conversion_counts["NEW_V2_REMEASURE"] += 1
            else:
                conversion_counts["NO_REMEASURE_EITHER"] += 1

            result_row.update({
                "analysis_success": True,
                "legacy_v1_would_remeasure": legacy_remeasure,
                "legacy_v1_remeasure_targets": legacy_targets,
                "v2_remeasure_required": v2_remeasure,
                "v2_remeasure_targets": v2_remeasure_targets,
                "v2_unusable_targets": unusable,
                "v2_unmeasurable_targets": unmeasurable,
                "v2_uncertain_targets": uncertain,
                "measurement_states": states,
                "overall_measurement_quality": consensus.get("overall_measurement_quality"),
                "heart_rate_bpm": _finite_value(analysis, "global", "heart_rate_bpm", "value"),
                "pr_ms": _finite_value(analysis, "global", "pr_ms", "value"),
                "qrs_ms": _finite_value(analysis, "global", "qrs_ms", "value"),
                "qt_ms": _finite_value(analysis, "global", "qt_ms", "value"),
                "reasoner_publication_allowed": bool(
                    (analysis.get("specialist_reasoning") or {}).get("publication_allowed")
                ),
                "error": None,
            })
        except Exception as exc:
            result_row.update({
                "analysis_success": False,
                "error": f"{type(exc).__name__}:{exc}",
            })
        rows.append(result_row)

    successful = [r for r in rows if r.get("analysis_success")]
    n = len(rows)
    success_n = len(successful)

    def rate(predicate) -> float | None:
        if not success_n:
            return None
        return round(sum(1 for r in successful if predicate(r)) / success_n, 6)

    availability = {}
    for metric in ("heart_rate_bpm", "pr_ms", "qrs_ms", "qt_ms"):
        availability[metric] = {
            "available_n": sum(r.get(metric) is not None for r in successful),
            "availability_rate": (
                round(sum(r.get(metric) is not None for r in successful) / success_n, 6)
                if success_n else None
            ),
        }

    summary = {
        "benchmark_version": "MEDCALC_MEASUREMENT_CONSENSUS_V2_PTBXL_DEV_V1",
        "scope": "PTB_XL_DEVELOPMENT_CONTAMINATED_NATIVE_SIGNAL_NO_LABEL_SCORING",
        "clinical_validation_claim_allowed": False,
        "selection_n": n,
        "analysis_success_n": success_n,
        "analysis_failure_n": n - success_n,
        "analysis_success_rate": round(success_n / n, 6) if n else None,
        "legacy_v1_counterfactual_remeasure_rate": rate(
            lambda r: bool(r.get("legacy_v1_would_remeasure"))
        ),
        "v2_remeasure_rate": rate(lambda r: bool(r.get("v2_remeasure_required"))),
        "v2_any_unusable_rate": rate(lambda r: bool(r.get("v2_unusable_targets"))),
        "v2_any_unmeasurable_rate": rate(lambda r: bool(r.get("v2_unmeasurable_targets"))),
        "v2_any_uncertain_rate": rate(lambda r: bool(r.get("v2_uncertain_targets"))),
        "reasoner_publication_allowed_rate": rate(
            lambda r: bool(r.get("reasoner_publication_allowed"))
        ),
        "measurement_availability": availability,
        "measurement_state_counts": dict(sorted(state_counts.items())),
        "per_metric_state_counts": {
            metric: dict(sorted(counts.items()))
            for metric, counts in metric_state_counts.items()
        },
        "v1_to_v2_conversion_counts": dict(sorted(conversion_counts.items())),
        "rows": rows,
        "interpretation_constraints": [
            "PTB-XL is development-contaminated and is intentionally used only for engineering iteration.",
            "No diagnostic labels are scored in this benchmark.",
            "The V1 comparator is a counterfactual reconstructed from the same consensus audit fields, not a separate old-engine run.",
            "MIMIC-IV-ECG remains locked for future independent evaluation.",
            "SPH and CODE-test are not used.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        k: v for k, v in summary.items()
        if k not in {"rows"}
    }, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--records-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run_benchmark(args.manifest, args.records_root, args.output)


if __name__ == "__main__":
    main()
