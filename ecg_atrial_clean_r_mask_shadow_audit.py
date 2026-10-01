from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_independent_atrial_evidence import (
    PREFERRED_LEADS,
    recover_crosslead_atrial_candidates,
)
from ecg_r_relative_amplitude_margin_audit import _filter_for_threshold
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    CONTROL_N,
    DIAGNOSTIC_CASES_EACH,
    DIAGNOSTIC_GROUPS,
    FS,
    ROLE,
    TOTAL_CASES,
    all_specs,
    canonical,
    make_signal,
)

VERSION = "MEDCALC_ATRIAL_CLEAN_R_MASK_SHADOW_AUDIT_V1"
RELATIVE_AMP_MIN = 0.15


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "mask_evaluable_n": 0,
        "mask_changed_n": 0,
        "baseline_recovered_any_n": 0,
        "shadow_recovered_any_n": 0,
        "baseline_unseeded_any_n": 0,
        "shadow_unseeded_any_n": 0,
        "baseline_unseeded_event_total": 0,
        "shadow_unseeded_event_total": 0,
        "baseline_unseeded_organized_n": 0,
        "shadow_unseeded_organized_n": 0,
        "baseline_organized_augmented_n": 0,
        "shadow_organized_augmented_n": 0,
        "evidence_changed_n": 0,
    }


def _shadow_mask_inputs(
    canonical_ecg: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    original = analysis.get("leads") or {}
    shadow = copy.deepcopy(original)
    rhythm = analysis.get("rhythm") or {}
    rhythm_lead = rhythm.get("lead")
    audit = {
        "evaluable": False,
        "changed": False,
        "raw_r_n": 0,
        "kept_r_n": 0,
    }
    if not rhythm_lead:
        return shadow, audit

    item = original.get(rhythm_lead) or {}
    source = (canonical_ecg.get("leads") or {}).get(rhythm_lead) or {}
    fs = int(item.get("fs") or source.get("fs") or analysis.get("fs") or FS)
    raw = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)
    audit["raw_r_n"] = len(raw)
    audit["kept_r_n"] = len(raw)
    if fs <= 0 or len(raw) < 3 or signal_mv.size == 0:
        return shadow, audit

    ok, kept = _filter_for_threshold(raw, signal_mv, fs, RELATIVE_AMP_MIN)
    audit["evaluable"] = bool(ok)
    audit["kept_r_n"] = len(kept)
    if not ok or len(kept) < 3 or len(kept) >= len(raw):
        return shadow, audit

    kept_ms = [1000.0 * float(v) / float(fs) for v in kept]
    for lead in PREFERRED_LEADS:
        measured = shadow.get(lead) or {}
        source_lead = (canonical_ecg.get("leads") or {}).get(lead) or {}
        if not measured.get("evaluable"):
            continue
        lead_fs = int(
            measured.get("fs")
            or source_lead.get("fs")
            or analysis.get("fs")
            or FS
        )
        if lead_fs <= 0:
            continue
        projected = sorted(
            set(int(round(t_ms * lead_fs / 1000.0)) for t_ms in kept_ms)
        )
        if len(projected) < 3:
            continue
        measured["r_peaks_samples"] = projected
        measured["r_count"] = len(projected)
        shadow[lead] = measured

    audit["changed"] = True
    return shadow, audit


def _apply(
    dst: dict[str, int],
    baseline: dict[str, Any],
    shadow: dict[str, Any],
    mask_audit: dict[str, Any],
) -> None:
    dst["n"] += 1
    dst["mask_evaluable_n"] += int(bool(mask_audit.get("evaluable")))
    dst["mask_changed_n"] += int(bool(mask_audit.get("changed")))

    b_rec = int(baseline.get("recovered_event_n") or 0)
    b_uns = int(baseline.get("unseeded_event_n") or 0)
    s_rec = int(shadow.get("recovered_event_n") or 0)
    s_uns = int(shadow.get("unseeded_event_n") or 0)

    dst["baseline_recovered_any_n"] += int((b_rec + b_uns) > 0)
    dst["shadow_recovered_any_n"] += int((s_rec + s_uns) > 0)
    dst["baseline_unseeded_any_n"] += int(b_uns > 0)
    dst["shadow_unseeded_any_n"] += int(s_uns > 0)
    dst["baseline_unseeded_event_total"] += b_uns
    dst["shadow_unseeded_event_total"] += s_uns
    dst["baseline_unseeded_organized_n"] += int(bool(baseline.get("unseeded_organized")))
    dst["shadow_unseeded_organized_n"] += int(bool(shadow.get("unseeded_organized")))
    dst["baseline_organized_augmented_n"] += int(bool(baseline.get("organized_augmented")))
    dst["shadow_organized_augmented_n"] += int(bool(shadow.get("organized_augmented")))

    before = (
        b_rec,
        b_uns,
        bool(baseline.get("unseeded_organized")),
        bool(baseline.get("organized_augmented")),
    )
    after = (
        s_rec,
        s_uns,
        bool(shadow.get("unseeded_organized")),
        bool(shadow.get("organized_augmented")),
    )
    dst["evidence_changed_n"] += int(before != after)


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
            baseline = recover_crosslead_atrial_candidates(
                canonical_ecg,
                analysis.get("leads") or {},
            )
            shadow_leads, mask_audit = _shadow_mask_inputs(canonical_ecg, analysis)
            shadow = (
                recover_crosslead_atrial_candidates(canonical_ecg, shadow_leads)
                if mask_audit.get("changed")
                else baseline
            )

            if spec.get("kind") == "TARGET":
                _apply(
                    per_target[str(spec["target"])],
                    baseline,
                    shadow,
                    mask_audit,
                )
            else:
                _apply(controls, baseline, shadow, mask_audit)
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply(
                    control_types.setdefault(ctype, _blank()),
                    baseline,
                    shadow,
                    mask_audit,
                )
        except Exception as exc:
            errors[type(exc).__name__] += 1

        if pos % 10 == 0:
            print(
                f"MEDCALC_ATRIAL_CLEAN_R_MASK {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_ATRIAL_CLEAN_R_MASK_AUDIT",
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
        raise SystemExit("No clean-R-mask audit shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing shards: {indices}")

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
        raise SystemExit(f"controls count mismatch: {controls['n']}")

    def enrich(row: dict[str, int]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        return {
            **row,
            "mask_changed_fraction": row["mask_changed_n"] / n,
            "baseline_recovered_any_fraction": row["baseline_recovered_any_n"] / n,
            "shadow_recovered_any_fraction": row["shadow_recovered_any_n"] / n,
            "baseline_unseeded_any_fraction": row["baseline_unseeded_any_n"] / n,
            "shadow_unseeded_any_fraction": row["shadow_unseeded_any_n"] / n,
            "baseline_unseeded_event_mean": row["baseline_unseeded_event_total"] / n,
            "shadow_unseeded_event_mean": row["shadow_unseeded_event_total"] / n,
            "baseline_unseeded_organized_fraction": row["baseline_unseeded_organized_n"] / n,
            "shadow_unseeded_organized_fraction": row["shadow_unseeded_organized_n"] / n,
            "baseline_organized_augmented_fraction": row["baseline_organized_augmented_n"] / n,
            "shadow_organized_augmented_fraction": row["shadow_organized_augmented_n"] / n,
            "evidence_changed_fraction": row["evidence_changed_n"] / n,
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_ATRIAL_CLEAN_R_MASK_AUDIT",
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
            "Synthetic engineering evidence only; no clinical output is changed.",
            "The fixed 0.15 cleaned selected-rhythm R timeline is used only as the ventricular mask for atrial evidence, and only when it actually removes candidates.",
            "Canonical/per-lead R outputs and AV conduction remain unchanged.",
            "No FAST-GATE-100, fold 9/10, external/final validation set, frozen baseline, or tolerance is modified or consumed.",
        ],
    }


def selftest() -> None:
    row = _blank()
    _apply(
        row,
        {"recovered_event_n": 0, "unseeded_event_n": 1, "unseeded_organized": False, "organized_augmented": False},
        {"recovered_event_n": 0, "unseeded_event_n": 4, "unseeded_organized": True, "organized_augmented": True},
        {"evaluable": True, "changed": True},
    )
    assert row["mask_changed_n"] == 1
    assert row["evidence_changed_n"] == 1
    assert row["shadow_unseeded_organized_n"] == 1
    print("MEDCALC_ATRIAL_CLEAN_R_MASK_SHADOW_AUDIT_SELFTEST_PASS")


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
