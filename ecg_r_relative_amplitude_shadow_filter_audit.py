from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_avb3_ventricular_detection_audit import _greedy_match
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    CONTROL_N,
    DIAGNOSTIC_CASES_EACH,
    DIAGNOSTIC_GROUPS,
    FS,
    ROLE,
    TOTAL_CASES,
    _event_times,
    _seed,
    all_specs,
    canonical,
    make_signal,
)

VERSION = "MEDCALC_R_RELATIVE_AMPLITUDE_SHADOW_FILTER_AUDIT_V1"
RELATIVE_AMP_MIN = 0.25
REGULAR_CV_MAX = 0.12


def _rr_cv(samples: list[int], fs: int) -> float | None:
    if fs <= 0 or len(samples) < 3:
        return None
    rr = np.diff(np.asarray(sorted(set(samples)), dtype=float)) * 1000.0 / float(fs)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 2 or float(np.mean(rr)) <= 0:
        return None
    return float(np.std(rr, ddof=1) / np.mean(rr))


def _candidate_amplitude_mv(signal_mv: np.ndarray, sample: int, fs: int) -> float | None:
    """Local-baseline amplitude for a raw R candidate.

    The baseline window mirrors the existing conservative pre-QRS fallback
    territory (roughly 200-120 ms before the candidate) without modifying the
    clinical measurement path.
    """
    x = np.asarray(signal_mv, dtype=float).reshape(-1)
    sample = int(sample)
    if fs <= 0 or sample < 0 or sample >= x.size or not np.isfinite(x[sample]):
        return None
    a = max(0, sample - int(round(0.200 * fs)))
    b = max(a, sample - int(round(0.120 * fs)))
    if b - a < max(5, int(round(0.020 * fs))):
        a = max(0, sample - int(round(0.250 * fs)))
        b = max(a, sample - int(round(0.060 * fs)))
    window = x[a:b]
    window = window[np.isfinite(window)]
    if window.size < 5:
        return None
    baseline = float(np.median(window))
    return abs(float(x[sample]) - baseline)


def _upper_half_reference(values: list[float]) -> float | None:
    z = sorted(float(v) for v in values if np.isfinite(float(v)) and float(v) > 0)
    if len(z) < 3:
        return None
    upper = z[len(z) // 2 :]
    if not upper:
        return None
    ref = float(np.median(np.asarray(upper, dtype=float)))
    return ref if np.isfinite(ref) and ref > 0 else None


def _shadow_filter(
    raw_r_samples: list[int],
    signal_mv: np.ndarray,
    fs: int,
) -> dict[str, Any]:
    raw = sorted(set(int(v) for v in raw_r_samples))
    amplitudes: list[tuple[int, float]] = []
    for sample in raw:
        amp = _candidate_amplitude_mv(signal_mv, sample, fs)
        if amp is not None:
            amplitudes.append((sample, float(amp)))

    reference = _upper_half_reference([amp for _, amp in amplitudes])
    if reference is None:
        return {
            "evaluable": False,
            "reference_mv": None,
            "raw_samples": raw,
            "kept_samples": raw,
            "removed_samples": [],
        }

    amp_by_sample = {sample: amp for sample, amp in amplitudes}
    kept: list[int] = []
    removed: list[int] = []
    for sample in raw:
        amp = amp_by_sample.get(sample)
        if amp is None:
            kept.append(sample)
            continue
        ratio = float(amp / reference)
        if ratio >= RELATIVE_AMP_MIN:
            kept.append(sample)
        else:
            removed.append(sample)

    return {
        "evaluable": True,
        "reference_mv": round(reference, 6),
        "raw_samples": raw,
        "kept_samples": kept,
        "removed_samples": removed,
    }


def _near_count(values_ms: list[float], centers_ms: list[float], tolerance_ms: float) -> int:
    return sum(
        any(abs(float(value) - float(center)) <= tolerance_ms for center in centers_ms)
        for value in values_ms
    )


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "reference_evaluable_n": 0,
        "cases_any_removed_n": 0,
        "raw_r_total": 0,
        "shadow_r_total": 0,
        "removed_r_total": 0,
        "raw_regular_n": 0,
        "shadow_regular_n": 0,
        "truth_evaluable_n": 0,
        "truth_r_total": 0,
        "raw_matched_total": 0,
        "shadow_matched_total": 0,
        "raw_missed_total": 0,
        "shadow_missed_total": 0,
        "raw_extra_total": 0,
        "shadow_extra_total": 0,
        "raw_extra_near_p_total": 0,
        "shadow_extra_near_p_total": 0,
        "raw_extra_near_t_total": 0,
        "shadow_extra_near_t_total": 0,
        "cases_new_truth_miss_n": 0,
        "cases_extra_reduced_n": 0,
    }


def _apply_case(
    dst: dict[str, int],
    *,
    raw: list[int],
    shadow: list[int],
    fs: int,
    reference_evaluable: bool,
    truth_p_ms: list[float] | None,
    truth_r_ms: list[float] | None,
) -> None:
    dst["n"] += 1
    dst["reference_evaluable_n"] += int(reference_evaluable)
    dst["raw_r_total"] += len(raw)
    dst["shadow_r_total"] += len(shadow)
    dst["removed_r_total"] += max(0, len(raw) - len(shadow))
    dst["cases_any_removed_n"] += int(len(shadow) < len(raw))

    raw_cv = _rr_cv(raw, fs)
    shadow_cv = _rr_cv(shadow, fs)
    dst["raw_regular_n"] += int(raw_cv is not None and raw_cv <= REGULAR_CV_MAX)
    dst["shadow_regular_n"] += int(
        shadow_cv is not None and shadow_cv <= REGULAR_CV_MAX
    )

    if truth_r_ms is None or truth_p_ms is None:
        return

    dst["truth_evaluable_n"] += 1
    dst["truth_r_total"] += len(truth_r_ms)
    raw_ms = [1000.0 * v / float(fs) for v in raw]
    shadow_ms = [1000.0 * v / float(fs) for v in shadow]
    raw_match = _greedy_match(truth_r_ms, raw_ms, 80.0)
    shadow_match = _greedy_match(truth_r_ms, shadow_ms, 80.0)

    dst["raw_matched_total"] += int(raw_match["matched_n"])
    dst["shadow_matched_total"] += int(shadow_match["matched_n"])
    dst["raw_missed_total"] += int(raw_match["missed_n"])
    dst["shadow_missed_total"] += int(shadow_match["missed_n"])
    dst["raw_extra_total"] += int(raw_match["extra_n"])
    dst["shadow_extra_total"] += int(shadow_match["extra_n"])
    dst["cases_new_truth_miss_n"] += int(
        int(shadow_match["missed_n"]) > int(raw_match["missed_n"])
    )
    dst["cases_extra_reduced_n"] += int(
        int(shadow_match["extra_n"]) < int(raw_match["extra_n"])
    )

    true_t_ms = [float(x) + 280.0 for x in truth_r_ms]
    raw_extras = [float(x) for x in raw_match["extras_ms"]]
    shadow_extras = [float(x) for x in shadow_match["extras_ms"]]
    dst["raw_extra_near_p_total"] += _near_count(raw_extras, truth_p_ms, 80.0)
    dst["shadow_extra_near_p_total"] += _near_count(
        shadow_extras, truth_p_ms, 80.0
    )
    dst["raw_extra_near_t_total"] += _near_count(raw_extras, true_t_ms, 100.0)
    dst["shadow_extra_near_t_total"] += _near_count(
        shadow_extras, true_t_ms, 100.0
    )


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    specs = all_specs()
    selected = [spec for i, spec in enumerate(specs) if i % shard_count == shard_index]
    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, int]] = {}
    errors = Counter()

    for pos, spec in enumerate(selected, 1):
        try:
            signal = make_signal(spec)
            canonical_ecg = canonical(spec, signal)
            analysis = analyze_canonical_ecg(canonical_ecg)
            rhythm = analysis.get("rhythm") or {}
            lead = rhythm.get("lead")
            item = (analysis.get("leads") or {}).get(lead) or {}
            source = (canonical_ecg.get("leads") or {}).get(lead) or {}
            fs = int(item.get("fs") or source.get("fs") or analysis.get("fs") or FS)
            raw = sorted(
                set(int(v) for v in (rhythm.get("r_peaks_samples") or []))
            )
            signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)
            shadow_result = _shadow_filter(raw, signal_mv, fs)
            shadow = list(shadow_result["kept_samples"])

            truth_p_ms = None
            truth_r_ms = None
            if not bool(spec.get("use_neurokit_multilead")):
                rng = np.random.default_rng(_seed(str(spec["case_id"])))
                p_s, r_s, _ = _event_times(spec, rng)
                truth_p_ms = [1000.0 * float(x) for x in p_s]
                truth_r_ms = [1000.0 * float(x) for x in r_s]

            if spec.get("kind") == "TARGET":
                dst = per_target[str(spec["target"])]
                _apply_case(
                    dst,
                    raw=raw,
                    shadow=shadow,
                    fs=fs,
                    reference_evaluable=bool(shadow_result["evaluable"]),
                    truth_p_ms=truth_p_ms,
                    truth_r_ms=truth_r_ms,
                )
            else:
                _apply_case(
                    controls,
                    raw=raw,
                    shadow=shadow,
                    fs=fs,
                    reference_evaluable=bool(shadow_result["evaluable"]),
                    truth_p_ms=None,
                    truth_r_ms=None,
                )
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply_case(
                    control_types.setdefault(ctype, _blank()),
                    raw=raw,
                    shadow=shadow,
                    fs=fs,
                    reference_evaluable=bool(shadow_result["evaluable"]),
                    truth_p_ms=None,
                    truth_r_ms=None,
                )
        except Exception as exc:
            errors[type(exc).__name__] += 1

        if pos % 10 == 0:
            print(
                f"MEDCALC_R_RELATIVE_AMP_SHADOW {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_R_SHADOW_FILTER_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "relative_amp_min": RELATIVE_AMP_MIN,
        "case_count": len(selected),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "metrics": {
            "per_target": per_target,
            "controls": controls,
            "control_types": control_types,
            "analysis_error_n": int(sum(errors.values())),
            "analysis_error_types": dict(sorted(errors.items())),
        },
        "case_level_outputs_emitted": False,
    }


def _merge(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key in dst:
        dst[key] += int(src.get(key) or 0)


def aggregate_dir(path: Path) -> dict[str, Any]:
    shards = []
    for file in sorted(path.rglob("*.json")):
        try:
            row = json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("version") == VERSION and "shard_index" in row:
            shards.append(row)
    if not shards:
        raise SystemExit("No shadow-filter shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing shadow-filter shards: {indices}")

    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, int]] = {}
    total = 0
    error_n = 0
    error_types = Counter()

    for shard in shards:
        total += int(shard.get("case_count") or 0)
        metrics = shard.get("metrics") or {}
        error_n += int(metrics.get("analysis_error_n") or 0)
        error_types.update(metrics.get("analysis_error_types") or {})
        for target, row in (metrics.get("per_target") or {}).items():
            _merge(per_target[target], row)
        _merge(controls, metrics.get("controls") or {})
        for ctype, row in (metrics.get("control_types") or {}).items():
            _merge(control_types.setdefault(ctype, _blank()), row)

    if total != TOTAL_CASES:
        raise SystemExit(f"cohort size mismatch: {total} != {TOTAL_CASES}")
    for target, row in per_target.items():
        if row["n"] != DIAGNOSTIC_CASES_EACH:
            raise SystemExit(f"{target} count mismatch: {row['n']}")
    if controls["n"] != CONTROL_N:
        raise SystemExit(f"control count mismatch: {controls['n']}")

    def enrich(row: dict[str, int]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        truth_n = max(int(row["truth_evaluable_n"]), 1)
        return {
            **row,
            "reference_evaluable_fraction": row["reference_evaluable_n"] / n,
            "cases_any_removed_fraction": row["cases_any_removed_n"] / n,
            "raw_regular_fraction": row["raw_regular_n"] / n,
            "shadow_regular_fraction": row["shadow_regular_n"] / n,
            "cases_new_truth_miss_fraction_among_truth_evaluable": (
                row["cases_new_truth_miss_n"] / truth_n
            ),
            "cases_extra_reduced_fraction_among_truth_evaluable": (
                row["cases_extra_reduced_n"] / truth_n
            ),
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_R_SHADOW_FILTER_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "relative_amp_min": RELATIVE_AMP_MIN,
        "case_count": total,
        "analysis_error_n": error_n,
        "analysis_error_types": dict(sorted(error_types.items())),
        "metrics": {
            "per_target": {k: enrich(v) for k, v in per_target.items()},
            "controls": enrich(controls),
            "control_types": {k: enrich(v) for k, v in control_types.items()},
        },
        "case_level_outputs_emitted": False,
        "interpretation": [
            "Shadow-only engineering audit; the 0.25 candidate was pre-specified from the broad synthetic-development separation observed before this run.",
            "The shadow sequence is not written back into r_peaks_samples or any diagnostic output.",
            "No frozen baseline/tolerance, FAST-GATE-100, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    fs = 500
    x = np.zeros(2000, dtype=float)
    for s, amp in [(300, 1.0), (600, 0.1), (900, 0.95), (1200, 0.12), (1500, 1.05)]:
        x[s] = amp
    out = _shadow_filter([300, 600, 900, 1200, 1500], x, fs)
    assert out["evaluable"], out
    assert out["kept_samples"] == [300, 900, 1500], out
    assert out["removed_samples"] == [600, 1200], out
    print("MEDCALC_R_RELATIVE_AMPLITUDE_SHADOW_FILTER_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--shard-index", type=int)
    ap.add_argument("--shard-count", type=int, default=10)
    ap.add_argument("--aggregate-dir", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.aggregate_dir is not None:
        result = aggregate_dir(args.aggregate_dir)
    else:
        if args.shard_index is None:
            raise SystemExit("--shard-index is required")
        result = run_shard(args.shard_index, args.shard_count)

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
