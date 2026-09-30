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
from ecg_independent_atrial_evidence import build_independent_atrial_consensus
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

VERSION = "MEDCALC_AV_P_CONSENSUS_R_SHADOW_AUDIT_V1"
RELATIVE_AMP_MIN = 0.15
P_PROXIMITY_MS = 80.0
POLICIES = {"P2": 2, "P3": 3}
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
        "preferred_lead_instances_n": 0,
        "reference_evaluable_lead_n": 0,
        "low_amp_candidate_total": 0,
        "consensus_p_removed_total": 0,
        "leads_any_removed_n": 0,
        "truth_lead_instances_n": 0,
        "raw_matched_total": 0,
        "shadow_matched_total": 0,
        "raw_extra_total": 0,
        "shadow_extra_total": 0,
        "new_truth_miss_lead_n": 0,
        "raw_classification_counts": {},
        "shadow_classification_counts": {},
    }


def _inc_code(dst: dict[str, Any], field: str, code: str) -> None:
    counts = dst[field]
    counts[code] = int(counts.get(code) or 0) + 1


def _expected_hit(target: str, classification: str) -> bool:
    if target == "AVB2":
        return classification in AVB2_CODES
    if target == "AVB3":
        return classification in AVB3_CODES
    return False


def _near_any(time_ms: float, events_ms: list[float], window_ms: float) -> bool:
    return any(abs(float(time_ms) - float(t)) <= float(window_ms) for t in events_ms)


def _shadow_for_support(
    canonical_ecg: dict[str, Any],
    analysis: dict[str, Any],
    *,
    support_min: int,
    truth_r_ms: list[float] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    original = analysis.get("leads") or {}
    shadow = copy.deepcopy(original)
    source_leads = canonical_ecg.get("leads") or {}
    consensus = build_independent_atrial_consensus(original)
    p_times = [
        float(row["time_ms"])
        for row in (consensus.get("events") or [])
        if row.get("time_ms") is not None
        and int(row.get("support_lead_n") or 0) >= int(support_min)
    ]

    stats = {
        "preferred_lead_instances_n": 0,
        "reference_evaluable_lead_n": 0,
        "low_amp_candidate_total": 0,
        "consensus_p_removed_total": 0,
        "leads_any_removed_n": 0,
        "truth_lead_instances_n": 0,
        "raw_matched_total": 0,
        "shadow_matched_total": 0,
        "raw_extra_total": 0,
        "shadow_extra_total": 0,
        "new_truth_miss_lead_n": 0,
    }

    for lead in PREFERRED:
        item = original.get(lead) or {}
        source = source_leads.get(lead) or {}
        if not item.get("evaluable"):
            continue
        fs = int(item.get("fs") or source.get("fs") or analysis.get("fs") or FS)
        raw = sorted(set(int(v) for v in (item.get("r_peaks_samples") or [])))
        if len(raw) < 3:
            continue
        signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)
        ref_ok, amp_kept = _filter_for_threshold(
            raw,
            signal_mv,
            fs,
            RELATIVE_AMP_MIN,
        )
        amp_kept_set = set(int(v) for v in amp_kept)
        low_amp = {sample for sample in raw if sample not in amp_kept_set}

        removed = {
            sample
            for sample in low_amp
            if _near_any(
                1000.0 * float(sample) / float(max(fs, 1)),
                p_times,
                P_PROXIMITY_MS,
            )
        }
        kept = [sample for sample in raw if sample not in removed]

        stats["preferred_lead_instances_n"] += 1
        stats["reference_evaluable_lead_n"] += int(ref_ok)
        stats["low_amp_candidate_total"] += len(low_amp)
        stats["consensus_p_removed_total"] += len(removed)
        stats["leads_any_removed_n"] += int(bool(removed))

        shadow_item = shadow.get(lead) or {}
        shadow_item["r_peaks_samples"] = list(kept)
        shadow_item["r_count"] = len(kept)
        shadow[lead] = shadow_item

        if truth_r_ms is not None:
            stats["truth_lead_instances_n"] += 1
            raw_ms = [1000.0 * v / float(fs) for v in raw]
            kept_ms = [1000.0 * v / float(fs) for v in kept]
            raw_match = _greedy_match(truth_r_ms, raw_ms, MATCH_MS)
            shadow_match = _greedy_match(truth_r_ms, kept_ms, MATCH_MS)
            stats["raw_matched_total"] += int(raw_match["matched_n"])
            stats["shadow_matched_total"] += int(shadow_match["matched_n"])
            stats["raw_extra_total"] += int(raw_match["extra_n"])
            stats["shadow_extra_total"] += int(shadow_match["extra_n"])
            stats["new_truth_miss_lead_n"] += int(
                int(shadow_match["missed_n"]) > int(raw_match["missed_n"])
            )

    return shadow, stats


def _apply_case(
    dst: dict[str, Any],
    *,
    target: str,
    raw_av: dict[str, Any],
    shadow_av: dict[str, Any],
    stats: dict[str, int],
) -> None:
    dst["n"] += 1
    raw_code = str(raw_av.get("classification") or "NONE")
    shadow_code = str(shadow_av.get("classification") or "NONE")
    dst["raw_expected_mechanism_n"] += int(_expected_hit(target, raw_code))
    dst["shadow_expected_mechanism_n"] += int(_expected_hit(target, shadow_code))
    dst["raw_any_block_n"] += int(raw_code in ANY_BLOCK_CODES)
    dst["shadow_any_block_n"] += int(shadow_code in ANY_BLOCK_CODES)
    dst["classification_changed_n"] += int(raw_code != shadow_code)
    _inc_code(dst, "raw_classification_counts", raw_code)
    _inc_code(dst, "shadow_classification_counts", shadow_code)
    for key, value in stats.items():
        dst[key] += int(value)


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    specs = all_specs()
    selected = [spec for i, spec in enumerate(specs) if i % shard_count == shard_index]
    per_target = {
        target: {policy: _blank() for policy in POLICIES}
        for target in DIAGNOSTIC_GROUPS
    }
    controls = {policy: _blank() for policy in POLICIES}
    control_types: dict[str, dict[str, dict[str, Any]]] = {}
    errors = Counter()

    for pos, spec in enumerate(selected, 1):
        try:
            signal = make_signal(spec)
            canonical_ecg = canonical(spec, signal)
            analysis = analyze_canonical_ecg(canonical_ecg)
            raw_av = analysis.get("av_conduction") or {}

            truth_r_ms = None
            if not bool(spec.get("use_neurokit_multilead")):
                rng = np.random.default_rng(_seed(str(spec["case_id"])))
                _, r_s, _ = _event_times(spec, rng)
                truth_r_ms = [1000.0 * float(x) for x in r_s]

            for policy, support_min in POLICIES.items():
                shadow_leads, stats = _shadow_for_support(
                    canonical_ecg,
                    analysis,
                    support_min=support_min,
                    truth_r_ms=truth_r_ms,
                )
                shadow_av = analyze_av_conduction(
                    shadow_leads,
                    analysis.get("atrial_activity") or {},
                    global_metrics=analysis.get("global") or {},
                )

                if spec.get("kind") == "TARGET":
                    target = str(spec["target"])
                    _apply_case(
                        per_target[target][policy],
                        target=target,
                        raw_av=raw_av,
                        shadow_av=shadow_av,
                        stats=stats,
                    )
                else:
                    _apply_case(
                        controls[policy],
                        target="CONTROL",
                        raw_av=raw_av,
                        shadow_av=shadow_av,
                        stats=stats,
                    )
                    ctype = str(spec.get("control_type") or "UNKNOWN")
                    rows = control_types.setdefault(
                        ctype,
                        {name: _blank() for name in POLICIES},
                    )
                    _apply_case(
                        rows[policy],
                        target="CONTROL",
                        raw_av=raw_av,
                        shadow_av=shadow_av,
                        stats=stats,
                    )
        except Exception as exc:
            errors[type(exc).__name__] += 1

        if pos % 10 == 0:
            print(
                f"MEDCALC_AV_P_CONSENSUS_R_SHADOW {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_P_CONSENSUS_R_SHADOW_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "relative_amp_min": RELATIVE_AMP_MIN,
        "p_proximity_ms": P_PROXIMITY_MS,
        "policies": POLICIES,
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
        raise SystemExit("No P-consensus R shadow shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing shards: got {indices}, expected 0..{expected-1}")

    per_target = {
        target: {policy: _blank() for policy in POLICIES}
        for target in DIAGNOSTIC_GROUPS
    }
    controls = {policy: _blank() for policy in POLICIES}
    control_types: dict[str, dict[str, dict[str, Any]]] = {}
    total = 0
    error_n = 0
    error_types = Counter()

    for shard in shards:
        total += int(shard.get("case_count") or 0)
        metrics = shard.get("metrics") or {}
        error_n += int(metrics.get("analysis_error_n") or 0)
        error_types.update(metrics.get("analysis_error_types") or {})
        for target, policy_rows in (metrics.get("per_target") or {}).items():
            for policy, row in policy_rows.items():
                _merge(per_target[target][policy], row)
        for policy, row in (metrics.get("controls") or {}).items():
            _merge(controls[policy], row)
        for ctype, policy_rows in (metrics.get("control_types") or {}).items():
            dst_rows = control_types.setdefault(
                ctype,
                {name: _blank() for name in POLICIES},
            )
            for policy, row in policy_rows.items():
                _merge(dst_rows[policy], row)

    if total != TOTAL_CASES:
        raise SystemExit(f"cohort size mismatch: {total} != {TOTAL_CASES}")
    for target, policy_rows in per_target.items():
        for policy, row in policy_rows.items():
            if row["n"] != DIAGNOSTIC_CASES_EACH:
                raise SystemExit(f"{target}/{policy} count mismatch: {row['n']}")
    for policy, row in controls.items():
        if row["n"] != CONTROL_N:
            raise SystemExit(f"controls/{policy} count mismatch: {row['n']}")

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        lead_n = max(int(row["preferred_lead_instances_n"]), 1)
        truth_lead_n = max(int(row["truth_lead_instances_n"]), 1)
        return {
            **row,
            "raw_expected_mechanism_fraction": row["raw_expected_mechanism_n"] / n,
            "shadow_expected_mechanism_fraction": row["shadow_expected_mechanism_n"] / n,
            "raw_any_block_fraction": row["raw_any_block_n"] / n,
            "shadow_any_block_fraction": row["shadow_any_block_n"] / n,
            "classification_changed_fraction": row["classification_changed_n"] / n,
            "consensus_p_removed_per_lead": row["consensus_p_removed_total"] / lead_n,
            "new_truth_miss_lead_fraction": row["new_truth_miss_lead_n"] / truth_lead_n,
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_P_CONSENSUS_R_SHADOW_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "relative_amp_min": RELATIVE_AMP_MIN,
        "p_proximity_ms": P_PROXIMITY_MS,
        "policies": POLICIES,
        "case_count": total,
        "analysis_error_n": error_n,
        "analysis_error_types": dict(sorted(error_types.items())),
        "metrics": {
            "per_target": {
                target: {policy: enrich(row) for policy, row in rows.items()}
                for target, rows in per_target.items()
            },
            "controls": {policy: enrich(row) for policy, row in controls.items()},
            "control_types": {
                ctype: {policy: enrich(row) for policy, row in rows.items()}
                for ctype, rows in control_types.items()
            },
        },
        "case_level_outputs_emitted": False,
        "interpretation": [
            "Synthetic development evidence only; no clinical output is changed.",
            "An R candidate is removed in shadow only when it is below the fixed 0.15 relative-amplitude rule and near an observed cross-lead P consensus event.",
            "P2 and P3 map existing atrial support requirements >=2 and >=3 leads; neither is promoted clinically in this audit.",
            "No FAST-GATE-100, fold 9/10, external/final validation set, frozen baseline, or tolerance is modified or consumed.",
        ],
    }


def selftest() -> None:
    assert POLICIES == {"P2": 2, "P3": 3}
    assert _near_any(1000.0, [930.0], P_PROXIMITY_MS)
    assert not _near_any(1000.0, [900.0], P_PROXIMITY_MS)
    assert _expected_hit("AVB3", "COMPLETE_AV_BLOCK_COMPATIBLE")
    assert _expected_hit("AVB2", "MOBITZ_II_COMPATIBLE")
    assert not _expected_hit("CONTROL", "MOBITZ_II_COMPATIBLE")
    print("MEDCALC_AV_P_CONSENSUS_R_SHADOW_AUDIT_SELFTEST_PASS")


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
