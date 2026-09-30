from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_avb3_ventricular_detection_audit import _greedy_match
from ecg_r_relative_amplitude_shadow_filter_audit import (
    _candidate_amplitude_mv,
    _near_count,
    _rr_cv,
    _upper_half_reference,
)
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

VERSION = "MEDCALC_R_RELATIVE_AMPLITUDE_MARGIN_AUDIT_V1"
THRESHOLDS = (0.15, 0.20, 0.25)
REGULAR_CV_MAX = 0.12


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "reference_evaluable_n": 0,
        "cases_any_removed_n": 0,
        "raw_r_total": 0,
        "kept_r_total": 0,
        "removed_r_total": 0,
        "raw_regular_n": 0,
        "kept_regular_n": 0,
        "truth_evaluable_n": 0,
        "truth_r_total": 0,
        "raw_matched_total": 0,
        "kept_matched_total": 0,
        "raw_missed_total": 0,
        "kept_missed_total": 0,
        "raw_extra_total": 0,
        "kept_extra_total": 0,
        "raw_extra_near_p_total": 0,
        "kept_extra_near_p_total": 0,
        "cases_new_truth_miss_n": 0,
        "cases_extra_reduced_n": 0,
    }


def _filter_for_threshold(
    raw: list[int],
    signal_mv: np.ndarray,
    fs: int,
    threshold: float,
) -> tuple[bool, list[int]]:
    amplitudes: list[tuple[int, float]] = []
    for sample in raw:
        amp = _candidate_amplitude_mv(signal_mv, sample, fs)
        if amp is not None:
            amplitudes.append((int(sample), float(amp)))
    reference = _upper_half_reference([amp for _, amp in amplitudes])
    if reference is None:
        return False, list(raw)
    amp_by_sample = {sample: amp for sample, amp in amplitudes}
    kept = []
    for sample in raw:
        amp = amp_by_sample.get(sample)
        if amp is None or float(amp / reference) >= float(threshold):
            kept.append(int(sample))
    return True, kept


def _apply(
    dst: dict[str, int],
    *,
    raw: list[int],
    kept: list[int],
    fs: int,
    ref_ok: bool,
    truth_p_ms: list[float] | None,
    truth_r_ms: list[float] | None,
) -> None:
    dst["n"] += 1
    dst["reference_evaluable_n"] += int(ref_ok)
    dst["raw_r_total"] += len(raw)
    dst["kept_r_total"] += len(kept)
    dst["removed_r_total"] += max(0, len(raw) - len(kept))
    dst["cases_any_removed_n"] += int(len(kept) < len(raw))

    raw_cv = _rr_cv(raw, fs)
    kept_cv = _rr_cv(kept, fs)
    dst["raw_regular_n"] += int(raw_cv is not None and raw_cv <= REGULAR_CV_MAX)
    dst["kept_regular_n"] += int(
        kept_cv is not None and kept_cv <= REGULAR_CV_MAX
    )

    if truth_p_ms is None or truth_r_ms is None:
        return

    dst["truth_evaluable_n"] += 1
    dst["truth_r_total"] += len(truth_r_ms)
    raw_ms = [1000.0 * v / float(fs) for v in raw]
    kept_ms = [1000.0 * v / float(fs) for v in kept]
    raw_match = _greedy_match(truth_r_ms, raw_ms, 80.0)
    kept_match = _greedy_match(truth_r_ms, kept_ms, 80.0)

    dst["raw_matched_total"] += int(raw_match["matched_n"])
    dst["kept_matched_total"] += int(kept_match["matched_n"])
    dst["raw_missed_total"] += int(raw_match["missed_n"])
    dst["kept_missed_total"] += int(kept_match["missed_n"])
    dst["raw_extra_total"] += int(raw_match["extra_n"])
    dst["kept_extra_total"] += int(kept_match["extra_n"])
    dst["cases_new_truth_miss_n"] += int(
        int(kept_match["missed_n"]) > int(raw_match["missed_n"])
    )
    dst["cases_extra_reduced_n"] += int(
        int(kept_match["extra_n"]) < int(raw_match["extra_n"])
    )

    raw_extras = [float(x) for x in raw_match["extras_ms"]]
    kept_extras = [float(x) for x in kept_match["extras_ms"]]
    dst["raw_extra_near_p_total"] += _near_count(raw_extras, truth_p_ms, 80.0)
    dst["kept_extra_near_p_total"] += _near_count(kept_extras, truth_p_ms, 80.0)


def _threshold_key(value: float) -> str:
    return f"{value:.2f}"


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    specs = all_specs()
    selected = [spec for i, spec in enumerate(specs) if i % shard_count == shard_index]
    per_target = {
        target: {_threshold_key(t): _blank() for t in THRESHOLDS}
        for target in DIAGNOSTIC_GROUPS
    }
    controls = {_threshold_key(t): _blank() for t in THRESHOLDS}
    control_types: dict[str, dict[str, dict[str, int]]] = {}
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
            raw = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
            signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)

            truth_p_ms = None
            truth_r_ms = None
            if not bool(spec.get("use_neurokit_multilead")):
                rng = np.random.default_rng(_seed(str(spec["case_id"])))
                p_s, r_s, _ = _event_times(spec, rng)
                truth_p_ms = [1000.0 * float(x) for x in p_s]
                truth_r_ms = [1000.0 * float(x) for x in r_s]

            for threshold in THRESHOLDS:
                key = _threshold_key(threshold)
                ref_ok, kept = _filter_for_threshold(
                    raw, signal_mv, fs, threshold
                )
                if spec.get("kind") == "TARGET":
                    dst = per_target[str(spec["target"])][key]
                else:
                    dst = controls[key]
                _apply(
                    dst,
                    raw=raw,
                    kept=kept,
                    fs=fs,
                    ref_ok=ref_ok,
                    truth_p_ms=truth_p_ms,
                    truth_r_ms=truth_r_ms,
                )
                if spec.get("kind") != "TARGET":
                    ctype = str(spec.get("control_type") or "UNKNOWN")
                    crows = control_types.setdefault(
                        ctype,
                        {_threshold_key(t): _blank() for t in THRESHOLDS},
                    )
                    _apply(
                        crows[key],
                        raw=raw,
                        kept=kept,
                        fs=fs,
                        ref_ok=ref_ok,
                        truth_p_ms=None,
                        truth_r_ms=None,
                    )
        except Exception as exc:
            errors[type(exc).__name__] += 1

        if pos % 10 == 0:
            print(
                f"MEDCALC_R_RELATIVE_MARGIN {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_R_MARGIN_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "thresholds": list(THRESHOLDS),
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
        raise SystemExit("No margin-audit shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing margin-audit shards: {indices}")

    per_target = {
        target: {_threshold_key(t): _blank() for t in THRESHOLDS}
        for target in DIAGNOSTIC_GROUPS
    }
    controls = {_threshold_key(t): _blank() for t in THRESHOLDS}
    control_types: dict[str, dict[str, dict[str, int]]] = {}
    total = 0
    error_n = 0
    error_types = Counter()

    for shard in shards:
        total += int(shard.get("case_count") or 0)
        metrics = shard.get("metrics") or {}
        error_n += int(metrics.get("analysis_error_n") or 0)
        error_types.update(metrics.get("analysis_error_types") or {})
        for target, threshold_rows in (metrics.get("per_target") or {}).items():
            for key, row in threshold_rows.items():
                _merge(per_target[target][key], row)
        for key, row in (metrics.get("controls") or {}).items():
            _merge(controls[key], row)
        for ctype, threshold_rows in (metrics.get("control_types") or {}).items():
            dst_rows = control_types.setdefault(
                ctype,
                {_threshold_key(t): _blank() for t in THRESHOLDS},
            )
            for key, row in threshold_rows.items():
                _merge(dst_rows[key], row)

    if total != TOTAL_CASES:
        raise SystemExit(f"cohort size mismatch: {total} != {TOTAL_CASES}")
    for target, threshold_rows in per_target.items():
        for key, row in threshold_rows.items():
            if row["n"] != DIAGNOSTIC_CASES_EACH:
                raise SystemExit(f"{target}/{key} count mismatch: {row['n']}")
    for key, row in controls.items():
        if row["n"] != CONTROL_N:
            raise SystemExit(f"controls/{key} count mismatch: {row['n']}")

    def enrich(row: dict[str, int]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        truth_n = max(int(row["truth_evaluable_n"]), 1)
        return {
            **row,
            "cases_any_removed_fraction": row["cases_any_removed_n"] / n,
            "raw_regular_fraction": row["raw_regular_n"] / n,
            "kept_regular_fraction": row["kept_regular_n"] / n,
            "new_truth_miss_fraction": row["cases_new_truth_miss_n"] / truth_n,
            "extra_reduced_fraction": row["cases_extra_reduced_n"] / truth_n,
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_R_MARGIN_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "thresholds": list(THRESHOLDS),
        "case_count": total,
        "analysis_error_n": error_n,
        "analysis_error_types": dict(sorted(error_types.items())),
        "metrics": {
            "per_target": {
                target: {key: enrich(row) for key, row in rows.items()}
                for target, rows in per_target.items()
            },
            "controls": {key: enrich(row) for key, row in controls.items()},
            "control_types": {
                ctype: {key: enrich(row) for key, row in rows.items()}
                for ctype, rows in control_types.items()
            },
        },
        "case_level_outputs_emitted": False,
        "interpretation": [
            "Synthetic-development margin audit only; 0.15/0.20/0.25 are evaluated together before any clinical implementation.",
            "No threshold is applied to r_peaks_samples or diagnostic outputs in this PR.",
            "No frozen baseline/tolerance, FAST-GATE-100, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    fs = 500
    x = np.zeros(2000, dtype=float)
    for s, amp in [(300, 1.0), (600, 0.18), (900, 0.95), (1200, 0.24), (1500, 1.05)]:
        x[s] = amp
    raw = [300, 600, 900, 1200, 1500]
    _, kept15 = _filter_for_threshold(raw, x, fs, 0.15)
    _, kept20 = _filter_for_threshold(raw, x, fs, 0.20)
    _, kept25 = _filter_for_threshold(raw, x, fs, 0.25)
    assert len(kept15) >= len(kept20) >= len(kept25)
    assert 300 in kept25 and 900 in kept25 and 1500 in kept25
    print("MEDCALC_R_RELATIVE_AMPLITUDE_MARGIN_AUDIT_SELFTEST_PASS")


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
