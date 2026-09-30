from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_av_conduction import PREFERRED, analyze_av_conduction
from ecg_avb3_ventricular_detection_audit import _greedy_match
from ecg_r_relative_amplitude_margin_audit import _filter_for_threshold
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

VERSION = "MEDCALC_AV_GLOBAL_R_TIMELINE_SHADOW_AUDIT_V1"
RELATIVE_AMP_MIN = 0.15
MATCH_MS = 80.0

AVB2_CODES = {
    "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
    "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
    "MOBITZ_II_COMPATIBLE",
    "MOBITZ_I_WENCKEBACH_COMPATIBLE",
}
AVB3_CODES = {"COMPLETE_AV_BLOCK_COMPATIBLE"}
ANY_BLOCK_CODES = AVB2_CODES | AVB3_CODES | {"FIRST_DEGREE_AV_DELAY_COMPATIBLE"}


def _blank() -> dict[str, Any]:
    return {
        "n": 0,
        "raw_expected_mechanism_n": 0,
        "shadow_expected_mechanism_n": 0,
        "raw_any_block_n": 0,
        "shadow_any_block_n": 0,
        "classification_changed_n": 0,
        "timeline_evaluable_n": 0,
        "timeline_removed_case_n": 0,
        "timeline_raw_r_total": 0,
        "timeline_kept_r_total": 0,
        "truth_evaluable_n": 0,
        "truth_r_total": 0,
        "raw_matched_total": 0,
        "shadow_matched_total": 0,
        "raw_extra_total": 0,
        "shadow_extra_total": 0,
        "new_truth_miss_n": 0,
        "raw_classification_counts": {},
        "shadow_classification_counts": {},
    }


def _inc_code(dst: dict[str, Any], field: str, code: str) -> None:
    counts = dst[field]
    counts[code] = int(counts.get(code) or 0) + 1


def _expected_hit(target: str, code: str) -> bool:
    if target == "AVB2":
        return code in AVB2_CODES
    if target == "AVB3":
        return code in AVB3_CODES
    return False


def _filtered_global_r_timeline(
    canonical_ecg: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[bool, list[float], list[float]]:
    rhythm = analysis.get("rhythm") or {}
    rhythm_lead = rhythm.get("lead")
    if rhythm_lead is None:
        return False, [], []
    item = (analysis.get("leads") or {}).get(rhythm_lead) or {}
    source = (canonical_ecg.get("leads") or {}).get(rhythm_lead) or {}
    fs = int(item.get("fs") or source.get("fs") or analysis.get("fs") or FS)
    raw = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)
    if fs <= 0 or len(raw) < 3 or signal_mv.size == 0:
        raw_ms = [1000.0 * v / max(fs, 1) for v in raw]
        return False, raw_ms, list(raw_ms)
    ok, kept = _filter_for_threshold(raw, signal_mv, fs, RELATIVE_AMP_MIN)
    raw_ms = [1000.0 * float(v) / float(fs) for v in raw]
    kept_ms = [1000.0 * float(v) / float(fs) for v in kept]
    return bool(ok), raw_ms, kept_ms


def _project_timeline_to_preferred(
    analysis: dict[str, Any],
    kept_ms: list[float],
) -> dict[str, dict[str, Any]]:
    shadow = copy.deepcopy(analysis.get("leads") or {})
    global_fs = int(analysis.get("fs") or FS)
    for lead in PREFERRED:
        item = shadow.get(lead) or {}
        if not item.get("evaluable"):
            continue
        fs = int(item.get("fs") or global_fs)
        if fs <= 0:
            continue
        projected = sorted(set(int(round(float(t) * fs / 1000.0)) for t in kept_ms))
        if len(projected) < 3:
            continue
        item["r_peaks_samples"] = projected
        item["r_count"] = len(projected)
        shadow[lead] = item
    return shadow


def _apply(
    dst: dict[str, Any],
    *,
    target: str,
    raw_av: dict[str, Any],
    shadow_av: dict[str, Any],
    timeline_ok: bool,
    raw_ms: list[float],
    kept_ms: list[float],
    truth_r_ms: list[float] | None,
) -> None:
    dst["n"] += 1
    raw_code = str(raw_av.get("classification") or "NONE")
    shadow_code = str(shadow_av.get("classification") or "NONE")
    dst["raw_expected_mechanism_n"] += int(_expected_hit(target, raw_code))
    dst["shadow_expected_mechanism_n"] += int(_expected_hit(target, shadow_code))
    dst["raw_any_block_n"] += int(raw_code in ANY_BLOCK_CODES)
    dst["shadow_any_block_n"] += int(shadow_code in ANY_BLOCK_CODES)
    dst["classification_changed_n"] += int(raw_code != shadow_code)
    dst["timeline_evaluable_n"] += int(timeline_ok)
    dst["timeline_removed_case_n"] += int(len(kept_ms) < len(raw_ms))
    dst["timeline_raw_r_total"] += len(raw_ms)
    dst["timeline_kept_r_total"] += len(kept_ms)
    _inc_code(dst, "raw_classification_counts", raw_code)
    _inc_code(dst, "shadow_classification_counts", shadow_code)

    if truth_r_ms is None:
        return
    dst["truth_evaluable_n"] += 1
    dst["truth_r_total"] += len(truth_r_ms)
    raw_match = _greedy_match(truth_r_ms, raw_ms, MATCH_MS)
    kept_match = _greedy_match(truth_r_ms, kept_ms, MATCH_MS)
    dst["raw_matched_total"] += int(raw_match["matched_n"])
    dst["shadow_matched_total"] += int(kept_match["matched_n"])
    dst["raw_extra_total"] += int(raw_match["extra_n"])
    dst["shadow_extra_total"] += int(kept_match["extra_n"])
    dst["new_truth_miss_n"] += int(
        int(kept_match["missed_n"]) > int(raw_match["missed_n"])
    )


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    specs = all_specs()
    selected = [spec for i, spec in enumerate(specs) if i % shard_count == shard_index]
    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, Any]] = {}
    errors = Counter()

    for pos, spec in enumerate(selected, 1):
        try:
            signal = make_signal(spec)
            canonical_ecg = canonical(spec, signal)
            analysis = analyze_canonical_ecg(canonical_ecg)
            raw_av = analysis.get("av_conduction") or {}

            timeline_ok, raw_ms, kept_ms = _filtered_global_r_timeline(
                canonical_ecg,
                analysis,
            )
            shadow_leads = _project_timeline_to_preferred(analysis, kept_ms)
            shadow_av = analyze_av_conduction(
                shadow_leads,
                analysis.get("atrial_activity") or {},
                global_metrics=analysis.get("global") or {},
            )

            truth_r_ms = None
            if not bool(spec.get("use_neurokit_multilead")):
                rng = np.random.default_rng(_seed(str(spec["case_id"])))
                _, r_s, _ = _event_times(spec, rng)
                truth_r_ms = [1000.0 * float(x) for x in r_s]

            if spec.get("kind") == "TARGET":
                target = str(spec["target"])
                _apply(
                    per_target[target],
                    target=target,
                    raw_av=raw_av,
                    shadow_av=shadow_av,
                    timeline_ok=timeline_ok,
                    raw_ms=raw_ms,
                    kept_ms=kept_ms,
                    truth_r_ms=truth_r_ms,
                )
            else:
                _apply(
                    controls,
                    target="CONTROL",
                    raw_av=raw_av,
                    shadow_av=shadow_av,
                    timeline_ok=timeline_ok,
                    raw_ms=raw_ms,
                    kept_ms=kept_ms,
                    truth_r_ms=truth_r_ms,
                )
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply(
                    control_types.setdefault(ctype, _blank()),
                    target="CONTROL",
                    raw_av=raw_av,
                    shadow_av=shadow_av,
                    timeline_ok=timeline_ok,
                    raw_ms=raw_ms,
                    kept_ms=kept_ms,
                    truth_r_ms=None,
                )
        except Exception as exc:
            errors[type(exc).__name__] += 1

        if pos % 10 == 0:
            print(
                f"MEDCALC_AV_GLOBAL_R_TIMELINE {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_AV_GLOBAL_R_TIMELINE_SHADOW_AUDIT",
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


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if key in {"raw_classification_counts", "shadow_classification_counts"}:
            for code, count in (value or {}).items():
                dst[key][code] = int(dst[key].get(code) or 0) + int(count)
        else:
            dst[key] += int(value or 0)


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
        raise SystemExit("No global-R timeline audit shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing shards: got {indices}, expected 0..{expected-1}")

    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, Any]] = {}
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
        raise SystemExit(f"controls count mismatch: {controls['n']}")

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        truth_n = max(int(row["truth_evaluable_n"]), 1)
        return {
            **row,
            "raw_expected_mechanism_fraction": row["raw_expected_mechanism_n"] / n,
            "shadow_expected_mechanism_fraction": row["shadow_expected_mechanism_n"] / n,
            "raw_any_block_fraction": row["raw_any_block_n"] / n,
            "shadow_any_block_fraction": row["shadow_any_block_n"] / n,
            "classification_changed_fraction": row["classification_changed_n"] / n,
            "timeline_removed_case_fraction": row["timeline_removed_case_n"] / n,
            "new_truth_miss_fraction": row["new_truth_miss_n"] / truth_n,
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_AV_GLOBAL_R_TIMELINE_SHADOW_AUDIT",
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
            "Synthetic engineering audit only; no clinical output is changed.",
            "A fixed 0.15-filtered selected-rhythm ventricular timeline is projected in shadow to AV preferred leads while local P fiducials are preserved.",
            "No FAST-GATE-100, fold 9/10, external/final validation set, frozen baseline, or tolerance is modified or consumed.",
        ],
    }


def selftest() -> None:
    assert _expected_hit("AVB3", "COMPLETE_AV_BLOCK_COMPATIBLE")
    assert _expected_hit("AVB2", "MOBITZ_II_COMPATIBLE")
    assert not _expected_hit("CONTROL", "MOBITZ_II_COMPATIBLE")
    print("MEDCALC_AV_GLOBAL_R_TIMELINE_SHADOW_AUDIT_SELFTEST_PASS")


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
