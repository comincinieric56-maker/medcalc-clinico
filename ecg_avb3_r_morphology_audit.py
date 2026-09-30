from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    FS,
    ROLE,
    _event_times,
    _seed,
    all_specs,
    canonical,
    make_signal,
)

VERSION = "MEDCALC_AVB3_R_MORPHOLOGY_AUDIT_V1"
FEATURES = (
    "qrs_ms",
    "r_amp_mv",
    "qrs_net_area_mv_ms",
    "beat_quality",
    "fiducial_confidence",
    "baseline_confidence",
    "q_duration_ms",
)


def _near(value_ms: float, centers_ms: list[float], tol_ms: float) -> bool:
    return any(abs(float(value_ms) - float(c)) <= tol_ms for c in centers_ms)


def _quantiles(values: list[float]) -> dict[str, float | None]:
    z = np.asarray([float(v) for v in values if np.isfinite(v)], dtype=float)
    if not z.size:
        return {"n": 0, "median": None, "p10": None, "p25": None, "p75": None, "p90": None}
    return {
        "n": int(z.size),
        "median": round(float(np.median(z)), 6),
        "p10": round(float(np.percentile(z, 10)), 6),
        "p25": round(float(np.percentile(z, 25)), 6),
        "p75": round(float(np.percentile(z, 75)), 6),
        "p90": round(float(np.percentile(z, 90)), 6),
    }


def _blank_class() -> dict[str, Any]:
    return {
        "n": 0,
        "features": {name: [] for name in FEATURES},
        "fiducial_source": Counter(),
        "baseline_source": Counter(),
    }


def run() -> dict[str, Any]:
    specs = [s for s in all_specs() if s.get("target") == "AVB3"]
    classes = {
        "TRUE_QRS": _blank_class(),
        "P_AS_R": _blank_class(),
        "OTHER": _blank_class(),
    }
    errors: list[str] = []

    for pos, spec in enumerate(specs, 1):
        try:
            rng = np.random.default_rng(_seed(str(spec["case_id"])))
            p_s, r_s, _ = _event_times(spec, rng)
            true_p_ms = [1000.0 * float(x) for x in p_s]
            true_r_ms = [1000.0 * float(x) for x in r_s]

            analysis = analyze_canonical_ecg(canonical(spec, make_signal(spec)))
            rhythm = analysis.get("rhythm") or {}
            lead = rhythm.get("lead")
            item = (analysis.get("leads") or {}).get(lead) or {}
            fs = int(item.get("fs") or analysis.get("fs") or FS)

            for beat in item.get("beats") or []:
                sample = beat.get("r_sample")
                if sample is None:
                    continue
                t_ms = 1000.0 * int(sample) / max(fs, 1)
                if _near(t_ms, true_r_ms, 80.0):
                    label = "TRUE_QRS"
                elif _near(t_ms, true_p_ms, 80.0):
                    label = "P_AS_R"
                else:
                    label = "OTHER"

                dst = classes[label]
                dst["n"] += 1
                for name in FEATURES:
                    value = beat.get(name)
                    if value is None:
                        continue
                    try:
                        value = float(value)
                    except Exception:
                        continue
                    if not np.isfinite(value):
                        continue
                    if name in {"r_amp_mv", "qrs_net_area_mv_ms"}:
                        value = abs(value)
                    dst["features"][name].append(value)
                dst["fiducial_source"][str(beat.get("fiducial_source") or "NONE")] += 1
                dst["baseline_source"][str(beat.get("baseline_source") or "NONE")] += 1

        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")

        if pos % 10 == 0:
            print(f"MEDCALC_AVB3_R_MORPHOLOGY_AUDIT {pos}/{len(specs)}", flush=True)

    out_classes = {}
    for name, row in classes.items():
        out_classes[name] = {
            "n": int(row["n"]),
            "features": {
                feature: _quantiles(values)
                for feature, values in row["features"].items()
            },
            "fiducial_source_counts": dict(sorted(row["fiducial_source"].items())),
            "baseline_source_counts": dict(sorted(row["baseline_source"].items())),
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_AVB3_R_MORPHOLOGY_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "case_count": len(specs),
        "analysis_error_n": len(errors),
        "classes": out_classes,
        "case_level_outputs_emitted": False,
        "errors": errors,
        "interpretation": [
            "Engineering audit only. Classes use synthetic generator timing only to compare existing beat morphology; no cutoff is selected here.",
            "No clinical threshold, baseline, tolerance, FAST-GATE-100, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    assert _near(100.0, [95.0], 10.0)
    assert not _near(100.0, [80.0], 10.0)
    q = _quantiles([1.0, 2.0, 3.0])
    assert q["n"] == 3 and q["median"] == 2.0, q
    print("MEDCALC_AVB3_R_MORPHOLOGY_AUDIT_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
