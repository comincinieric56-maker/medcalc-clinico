from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ecg_avb3_r_morphology_audit import _near, _quantiles
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

VERSION = "MEDCALC_AVBLOCK_R_RELATIVE_AMPLITUDE_AUDIT_V1"
TARGETS = ("AVB2", "AVB3")


def _upper_half_reference(amplitudes: list[float]) -> float | None:
    z = sorted(float(v) for v in amplitudes if np.isfinite(float(v)) and float(v) > 0)
    if len(z) < 3:
        return None
    start = len(z) // 2
    upper = z[start:]
    if not upper:
        return None
    ref = float(np.median(np.asarray(upper, dtype=float)))
    return ref if ref > 0 else None


def _blank_group() -> dict[str, Any]:
    return {
        "n_cases": 0,
        "reference_evaluable_n": 0,
        "TRUE_QRS": {"n": 0, "ratio": [], "abs_amp_mv": []},
        "P_AS_R": {"n": 0, "ratio": [], "abs_amp_mv": []},
        "OTHER": {"n": 0, "ratio": [], "abs_amp_mv": []},
    }


def run() -> dict[str, Any]:
    groups = {target: _blank_group() for target in TARGETS}
    errors: list[str] = []

    specs = [s for s in all_specs() if s.get("target") in TARGETS]
    for pos, spec in enumerate(specs, 1):
        target = str(spec["target"])
        dst = groups[target]
        dst["n_cases"] += 1
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
            beats = list(item.get("beats") or [])

            amps = []
            for beat in beats:
                value = beat.get("r_amp_mv")
                if value is None:
                    continue
                try:
                    value = abs(float(value))
                except Exception:
                    continue
                if np.isfinite(value):
                    amps.append(value)
            reference = _upper_half_reference(amps)
            if reference is None:
                continue
            dst["reference_evaluable_n"] += 1

            for beat in beats:
                sample = beat.get("r_sample")
                amp = beat.get("r_amp_mv")
                if sample is None or amp is None:
                    continue
                try:
                    amp = abs(float(amp))
                except Exception:
                    continue
                if not np.isfinite(amp):
                    continue

                time_ms = 1000.0 * int(sample) / max(fs, 1)
                if _near(time_ms, true_r_ms, 80.0):
                    label = "TRUE_QRS"
                elif _near(time_ms, true_p_ms, 80.0):
                    label = "P_AS_R"
                else:
                    label = "OTHER"

                row = dst[label]
                row["n"] += 1
                row["ratio"].append(float(amp / reference))
                row["abs_amp_mv"].append(float(amp))

        except Exception as exc:
            errors.append(f"{target}:{type(exc).__name__}:{exc}")

        if pos % 10 == 0:
            print(f"MEDCALC_AVBLOCK_R_RELATIVE_AMPLITUDE_AUDIT {pos}/{len(specs)}", flush=True)

    out = {}
    for target, group in groups.items():
        out[target] = {
            "n_cases": int(group["n_cases"]),
            "reference_evaluable_n": int(group["reference_evaluable_n"]),
        }
        for label in ("TRUE_QRS", "P_AS_R", "OTHER"):
            out[target][label] = {
                "n": int(group[label]["n"]),
                "relative_r_amp": _quantiles(group[label]["ratio"]),
                "absolute_r_amp_mv": _quantiles(group[label]["abs_amp_mv"]),
            }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_RELATIVE_R_AMPLITUDE_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "case_count": len(specs),
        "analysis_error_n": len(errors),
        "per_target": out,
        "case_level_outputs_emitted": False,
        "errors": errors,
        "normalization": (
            "ABS_R_AMP / MEDIAN_OF_UPPER_HALF_ABS_R_AMPLITUDES_ON_SELECTED_RHYTHM_LEAD"
        ),
        "interpretation": [
            "Engineering audit only. Synthetic generator timing labels existing beats; no relative-amplitude cutoff is selected.",
            "No clinical threshold, frozen baseline/tolerance, FAST-GATE-100, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    ref = _upper_half_reference([0.1, 0.12, 0.9, 1.0, 1.1, 1.05])
    assert ref is not None and 0.9 < ref < 1.2, ref
    assert _upper_half_reference([0.1, 0.2]) is None
    print("MEDCALC_AVBLOCK_R_RELATIVE_AMPLITUDE_AUDIT_SELFTEST_PASS")


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
