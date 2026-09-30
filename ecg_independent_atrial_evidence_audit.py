from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    CONTROL_N,
    DIAGNOSTIC_CASES_EACH,
    DIAGNOSTIC_GROUPS,
    ROLE,
    TOTAL_CASES,
    all_specs,
    canonical,
    make_signal,
)

VERSION = "MEDCALC_ATRIAL_EVIDENCE_SYNTHETIC_AUDIT_V5_HARMONIC_INTERSECTIONS"


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "recovered_any_n": 0,
        "unseeded_any_n": 0,
        "unseeded_ge4_n": 0,
        "unseeded_event_total": 0,
        "unseeded_ge4_stable_pr_n": 0,
        "unseeded_ge4_unstable_pr_n": 0,
        "harmonic_evaluable_n": 0,
        "harmonic_consistent_n": 0,
        "harmonic_faster_than_ventricular_n": 0,
        "harmonic_consistent_and_faster_n": 0,
        "harmonic_phase_dissociation_n": 0,
        "harmonic_ventricular_regular_n": 0,
        "harmonic_consistent_and_phase_n": 0,
        "harmonic_faster_and_phase_n": 0,
        "harmonic_consistent_faster_phase_n": 0,
        "harmonic_consistent_faster_regular_n": 0,
        "harmonic_complete_block_mechanism_n": 0,
        "unseeded_organized_n": 0,
        "organized_augmented_n": 0,
        "observed_organized_n": 0,
    }


def _apply(
    row: dict[str, int],
    evidence: dict[str, Any],
    pr_relation: dict[str, Any] | None = None,
    harmonic_relation: dict[str, Any] | None = None,
) -> None:
    row["n"] += 1
    seeded_n = int(evidence.get("recovered_event_n") or 0)
    unseeded_n = int(evidence.get("unseeded_event_n") or 0)
    row["recovered_any_n"] += int((seeded_n + unseeded_n) > 0)
    row["unseeded_any_n"] += int(unseeded_n > 0)
    row["unseeded_ge4_n"] += int(unseeded_n >= 4)
    row["unseeded_event_total"] += unseeded_n
    relation = pr_relation or {}
    if unseeded_n >= 4 and relation.get("evaluable"):
        stable = bool(relation.get("stable_pr_like"))
        row["unseeded_ge4_stable_pr_n"] += int(stable)
        row["unseeded_ge4_unstable_pr_n"] += int(not stable)
    harmonic = harmonic_relation or {}
    row["harmonic_evaluable_n"] += int(bool(harmonic.get("evaluable")))
    row["harmonic_consistent_n"] += int(bool(harmonic.get("harmonic_consistent")))
    row["harmonic_faster_than_ventricular_n"] += int(bool(harmonic.get("faster_than_ventricular")))
    row["harmonic_consistent_and_faster_n"] += int(bool(harmonic.get("harmonic_consistent_and_faster")))
    h_consistent = bool(harmonic.get("harmonic_consistent"))
    h_faster = bool(harmonic.get("faster_than_ventricular"))
    h_phase = bool(harmonic.get("phase_dissociation"))
    h_regular = bool(harmonic.get("ventricular_regular"))
    row["harmonic_phase_dissociation_n"] += int(h_phase)
    row["harmonic_ventricular_regular_n"] += int(h_regular)
    row["harmonic_consistent_and_phase_n"] += int(h_consistent and h_phase)
    row["harmonic_faster_and_phase_n"] += int(h_faster and h_phase)
    row["harmonic_consistent_faster_phase_n"] += int(h_consistent and h_faster and h_phase)
    row["harmonic_consistent_faster_regular_n"] += int(h_consistent and h_faster and h_regular)
    row["harmonic_complete_block_mechanism_n"] += int(bool(harmonic.get("complete_block_mechanism")))
    row["unseeded_organized_n"] += int(bool(evidence.get("unseeded_organized")))
    row["organized_augmented_n"] += int(bool(evidence.get("organized_augmented")))
    observed = evidence.get("observed_consensus") or {}
    row["observed_organized_n"] += int(bool(observed.get("organized")))



def _unseeded_pr_relation(
    evidence: dict[str, Any],
    rhythm: dict[str, Any],
    fs: int,
) -> dict[str, Any]:
    """Audit whether recovered atrial events retain a stable conducted PR.

    Uses the same physiological PR window and PR-stability MAD already used by
    the AV specialist. This is descriptive development evidence only.
    """
    events = sorted(
        float(row["time_ms"])
        for row in (evidence.get("unseeded_events") or [])
        if row.get("time_ms") is not None
    )
    r_samples = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    if fs <= 0 or len(events) < 4 or len(r_samples) < 3:
        return {"evaluable": False, "stable_pr_like": False}

    r_ms = [1000.0 * sample / fs for sample in r_samples]
    used: set[int] = set()
    prs: list[float] = []
    for p_ms in events:
        candidates = [
            (idx, r - p_ms)
            for idx, r in enumerate(r_ms)
            if idx not in used and 80.0 <= r - p_ms <= 500.0
        ]
        if not candidates:
            continue
        idx, pr = min(candidates, key=lambda row: row[1])
        used.add(idx)
        prs.append(float(pr))

    coupling = len(prs) / max(len(events), 1)
    pr_mad = (
        float(np.median(np.abs(np.asarray(prs) - np.median(prs))))
        if len(prs) >= 3 else None
    )
    stable = bool(
        len(prs) >= 3
        and coupling >= 0.70
        and pr_mad is not None
        and pr_mad <= 30.0
    )
    return {
        "evaluable": True,
        "event_n": len(events),
        "r_n": len(r_ms),
        "coupled_n": len(prs),
        "coupling_fraction": round(float(coupling), 6),
        "pr_mad_ms": round(float(pr_mad), 6) if pr_mad is not None else None,
        "stable_pr_like": stable,
        "rule": "PR_80_500MS; COUPLING_GE_0.70; PR_MAD_LE_30MS",
    }

def _shadow_harmonic_relation(
    evidence: dict[str, Any],
    rhythm: dict[str, Any],
    fs: int,
) -> dict[str, Any]:
    """Describe cadence in observed unseeded events without creating events.

    Candidate periods come only from integer divisors of observed consecutive
    event intervals. The 0.12 consistency cutoff and 1.25 atrial/ventricular
    rate ratio reuse existing AV-specialist constants; they are not widened.
    """
    events = sorted(
        float(row["time_ms"])
        for row in (evidence.get("unseeded_events") or [])
        if row.get("time_ms") is not None
    )
    r_samples = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    if fs <= 0 or len(events) < 3 or len(r_samples) < 3:
        return {
            "evaluable": False,
            "harmonic_consistent": False,
            "faster_than_ventricular": False,
            "harmonic_consistent_and_faster": False,
        }

    diffs = np.diff(np.asarray(events, dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size < 2:
        return {
            "evaluable": False,
            "harmonic_consistent": False,
            "faster_than_ventricular": False,
            "harmonic_consistent_and_faster": False,
        }

    candidates: list[float] = []
    for delta in diffs.tolist():
        for divisor in (1, 2, 3, 4):
            period = float(delta) / float(divisor)
            if 300.0 <= period <= 1500.0:
                candidates.append(period)
    if not candidates:
        return {
            "evaluable": True,
            "harmonic_consistent": False,
            "faster_than_ventricular": False,
            "harmonic_consistent_and_faster": False,
            "reason": "NO_PERIOD_CANDIDATE_300_1500MS",
        }

    best_period = None
    best_residual = None
    for period in candidates:
        cycles = np.maximum(1.0, np.rint(diffs / period))
        residuals = np.abs(diffs - cycles * period) / period
        score = float(np.median(residuals))
        if (
            best_residual is None
            or score < best_residual - 1e-12
            or (abs(score - best_residual) <= 1e-12 and period > float(best_period))
        ):
            best_residual = score
            best_period = float(period)

    r_ms = np.asarray([1000.0 * sample / fs for sample in r_samples], dtype=float)
    rr = np.diff(r_ms)
    rr = rr[np.isfinite(rr) & (rr > 0)]
    rr_median = float(np.median(rr)) if rr.size else None
    rr_cv = (
        float(np.std(rr, ddof=1) / np.mean(rr))
        if rr.size >= 2 and float(np.mean(rr)) > 0 else None
    )
    ventricular_regular = bool(rr_cv is not None and rr_cv <= 0.12)
    atrial_rate = 60000.0 / best_period if best_period and best_period > 0 else None
    ventricular_rate = 60000.0 / rr_median if rr_median and rr_median > 0 else None

    harmonic_consistent = bool(
        best_residual is not None and best_residual <= 0.12
    )
    faster = bool(
        atrial_rate is not None
        and ventricular_rate is not None
        and atrial_rate > 1.25 * ventricular_rate
    )

    phase_mad = None
    phase_range = None
    phase_dissociation = False
    if best_period is not None and best_period > 0 and r_ms.size >= 3:
        phases = np.sort(np.mod(r_ms - float(events[0]), best_period))
        if phases.size >= 3:
            circular_gaps = np.diff(np.r_[phases, phases[0] + best_period])
            cut = int(np.argmax(circular_gaps))
            unwrapped = np.r_[
                phases[cut + 1:],
                phases[:cut + 1] + best_period,
            ]
            phase_median = float(np.median(unwrapped))
            phase_mad = float(np.median(np.abs(unwrapped - phase_median)))
            phase_range = float(np.max(unwrapped) - np.min(unwrapped))
            phase_dissociation = bool(
                phase_mad >= max(50.0, 0.15 * best_period)
                and phase_range >= 0.30 * best_period
            )

    complete_block_mechanism = bool(
        harmonic_consistent
        and faster
        and ventricular_regular
        and phase_dissociation
    )
    return {
        "evaluable": True,
        "event_n": len(events),
        "candidate_period_ms": round(float(best_period), 6) if best_period is not None else None,
        "harmonic_residual_fraction": round(float(best_residual), 6) if best_residual is not None else None,
        "harmonic_consistent": harmonic_consistent,
        "atrial_rate_bpm": round(float(atrial_rate), 6) if atrial_rate is not None else None,
        "ventricular_rate_bpm": round(float(ventricular_rate), 6) if ventricular_rate is not None else None,
        "rr_cv": round(float(rr_cv), 6) if rr_cv is not None else None,
        "ventricular_regular": ventricular_regular,
        "faster_than_ventricular": faster,
        "harmonic_consistent_and_faster": bool(harmonic_consistent and faster),
        "phase_mad_ms": round(float(phase_mad), 6) if phase_mad is not None else None,
        "phase_range_ms": round(float(phase_range), 6) if phase_range is not None else None,
        "phase_dissociation": phase_dissociation,
        "complete_block_mechanism": complete_block_mechanism,
        "policy": (
            "SHADOW_ONLY; OBSERVED_UNSEEDED_INTERVALS_ONLY; "
            "NO_EVENT_SYNTHESIS; RESIDUAL_LE_0_12; ATRIAL_RATE_GT_1_25X_VENTRICULAR; "
            "RR_CV_LE_0_12; PHASE_MAD_AND_RANGE_USE_EXISTING_AV_THRESHOLDS"
        ),
    }


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    specs = all_specs()
    selected = [spec for i, spec in enumerate(specs) if i % shard_count == shard_index]
    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for pos, spec in enumerate(selected, 1):
        try:
            signal = make_signal(spec)
            canonical_ecg = canonical(spec, signal)
            analysis = analyze_canonical_ecg(canonical_ecg)
            evidence = recover_crosslead_atrial_candidates(
                canonical_ecg,
                analysis.get("leads") or {},
            )
            analysis_fs = int(analysis.get("fs") or canonical_ecg.get("fs") or 0)
            pr_relation = _unseeded_pr_relation(
                evidence,
                analysis.get("rhythm") or {},
                analysis_fs,
            )
            harmonic_relation = _shadow_harmonic_relation(
                evidence,
                analysis.get("rhythm") or {},
                analysis_fs,
            )
            if spec.get("kind") == "TARGET":
                _apply(per_target[str(spec["target"])], evidence, pr_relation, harmonic_relation)
            else:
                _apply(controls, evidence, pr_relation, harmonic_relation)
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply(control_types.setdefault(ctype, _blank()), evidence, pr_relation, harmonic_relation)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")

        if pos % 10 == 0:
            print(
                f"MEDCALC_ATRIAL_EVIDENCE_AUDIT {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_ATRIAL_EVIDENCE_AUDIT",
        "external_validation_claim_allowed": False,
        "diagnostic_claim_allowed": False,
        "clinical_output_changed": False,
        "case_count": len(selected),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "metrics": {
            "analysis_error_n": len(errors),
            "per_target": per_target,
            "controls": controls,
            "control_types": control_types,
        },
        "errors": errors,
        "case_level_outputs_emitted": False,
    }


def _merge_counts(dst: dict[str, int], src: dict[str, Any]) -> None:
    for key in ("n", "recovered_any_n", "unseeded_any_n", "unseeded_ge4_n", "unseeded_event_total", "unseeded_ge4_stable_pr_n", "unseeded_ge4_unstable_pr_n", "harmonic_evaluable_n", "harmonic_consistent_n", "harmonic_faster_than_ventricular_n", "harmonic_consistent_and_faster_n", "harmonic_phase_dissociation_n", "harmonic_ventricular_regular_n", "harmonic_consistent_and_phase_n", "harmonic_faster_and_phase_n", "harmonic_consistent_faster_phase_n", "harmonic_consistent_faster_regular_n", "harmonic_complete_block_mechanism_n", "unseeded_organized_n", "organized_augmented_n", "observed_organized_n"):
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
        raise SystemExit("No atrial-evidence audit shards found")

    expected = max(int(row["shard_count"]) for row in shards)
    indices = sorted(int(row["shard_index"]) for row in shards)
    if indices != list(range(expected)):
        raise SystemExit(f"Missing audit shards: got {indices}, expected 0..{expected-1}")

    per_target = {target: _blank() for target in DIAGNOSTIC_GROUPS}
    controls = _blank()
    control_types: dict[str, dict[str, int]] = {}
    total = 0
    error_n = 0

    for shard in shards:
        total += int(shard.get("case_count") or 0)
        metrics = shard.get("metrics") or {}
        error_n += int(metrics.get("analysis_error_n") or 0)
        for target, row in (metrics.get("per_target") or {}).items():
            _merge_counts(per_target[target], row)
        _merge_counts(controls, metrics.get("controls") or {})
        for ctype, row in (metrics.get("control_types") or {}).items():
            _merge_counts(control_types.setdefault(ctype, _blank()), row)

    if total != TOTAL_CASES:
        raise SystemExit(f"Audit cohort size mismatch: {total} != {TOTAL_CASES}")
    for target, row in per_target.items():
        if row["n"] != DIAGNOSTIC_CASES_EACH:
            raise SystemExit(f"{target} audit count mismatch: {row['n']}")
    if controls["n"] != CONTROL_N:
        raise SystemExit(f"Control audit count mismatch: {controls['n']}")

    def with_fraction(row: dict[str, int]) -> dict[str, Any]:
        n = max(int(row["n"]), 1)
        return {
            **row,
            "recovered_any_fraction": row["recovered_any_n"] / n,
            "unseeded_any_fraction": row["unseeded_any_n"] / n,
            "unseeded_ge4_fraction": row["unseeded_ge4_n"] / n,
            "unseeded_event_mean": row["unseeded_event_total"] / n,
            "unseeded_ge4_stable_pr_fraction": row["unseeded_ge4_stable_pr_n"] / n,
            "unseeded_ge4_unstable_pr_fraction": row["unseeded_ge4_unstable_pr_n"] / n,
            "harmonic_evaluable_fraction": row["harmonic_evaluable_n"] / n,
            "harmonic_consistent_fraction": row["harmonic_consistent_n"] / n,
            "harmonic_faster_than_ventricular_fraction": row["harmonic_faster_than_ventricular_n"] / n,
            "harmonic_consistent_and_faster_fraction": row["harmonic_consistent_and_faster_n"] / n,
            "harmonic_phase_dissociation_fraction": row["harmonic_phase_dissociation_n"] / n,
            "harmonic_ventricular_regular_fraction": row["harmonic_ventricular_regular_n"] / n,
            "harmonic_consistent_and_phase_fraction": row["harmonic_consistent_and_phase_n"] / n,
            "harmonic_faster_and_phase_fraction": row["harmonic_faster_and_phase_n"] / n,
            "harmonic_consistent_faster_phase_fraction": row["harmonic_consistent_faster_phase_n"] / n,
            "harmonic_consistent_faster_regular_fraction": row["harmonic_consistent_faster_regular_n"] / n,
            "harmonic_complete_block_mechanism_fraction": row["harmonic_complete_block_mechanism_n"] / n,
            "unseeded_organized_fraction": row["unseeded_organized_n"] / n,
            "organized_augmented_fraction": row["organized_augmented_n"] / n,
            "observed_organized_fraction": row["observed_organized_n"] / n,
        }

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_ATRIAL_EVIDENCE_AUDIT",
        "external_validation_claim_allowed": False,
        "diagnostic_claim_allowed": False,
        "clinical_output_changed": False,
        "case_count": total,
        "metrics": {
            "analysis_error_n": error_n,
            "per_target": {k: with_fraction(v) for k, v in per_target.items()},
            "controls": with_fraction(controls),
            "control_types": {k: with_fraction(v) for k, v in control_types.items()},
        },
        "case_level_outputs_emitted": False,
        "interpretation": [
            "This audit measures observability of nonpublishing atrial evidence only.",
            "It is engineering-development evidence, not clinical accuracy or validation.",
            "No diagnostic threshold, frozen baseline, FAST-GATE-100 panel, fold 9, or external/final validation set is used for tuning.",
        ],
    }


def selftest() -> None:
    specs = all_specs()
    assert len(specs) == TOTAL_CASES
    assert sum(spec.get("target") == "AVB2" for spec in specs) == DIAGNOSTIC_CASES_EACH
    assert sum(spec.get("target") == "AVB3" for spec in specs) == DIAGNOSTIC_CASES_EACH
    assert sum(spec.get("kind") == "CONTROL" for spec in specs) == CONTROL_N
    row = _blank()
    _apply(row, {
        "recovered_event_n": 2,
        "organized_augmented": True,
        "observed_consensus": {"organized": False},
    })
    assert row == {
        "n": 1,
        "recovered_any_n": 1,
        "unseeded_any_n": 0,
        "unseeded_ge4_n": 0,
        "unseeded_event_total": 0,
        "unseeded_ge4_stable_pr_n": 0,
        "unseeded_ge4_unstable_pr_n": 0,
        "harmonic_evaluable_n": 0,
        "harmonic_consistent_n": 0,
        "harmonic_faster_than_ventricular_n": 0,
        "harmonic_consistent_and_faster_n": 0,
        "harmonic_phase_dissociation_n": 0,
        "harmonic_ventricular_regular_n": 0,
        "harmonic_consistent_and_phase_n": 0,
        "harmonic_faster_and_phase_n": 0,
        "harmonic_consistent_faster_phase_n": 0,
        "harmonic_consistent_faster_regular_n": 0,
        "harmonic_complete_block_mechanism_n": 0,
        "unseeded_organized_n": 0,
        "organized_augmented_n": 1,
        "observed_organized_n": 0,
    }, row

    unseeded = _blank()
    _apply(unseeded, {
        "recovered_event_n": 0,
        "unseeded_event_n": 4,
        "unseeded_organized": True,
        "organized_augmented": True,
        "observed_consensus": {"organized": False},
    })
    assert unseeded == {
        "n": 1,
        "recovered_any_n": 1,
        "unseeded_any_n": 1,
        "unseeded_ge4_n": 1,
        "unseeded_event_total": 4,
        "unseeded_ge4_stable_pr_n": 0,
        "unseeded_ge4_unstable_pr_n": 0,
        "harmonic_evaluable_n": 0,
        "harmonic_consistent_n": 0,
        "harmonic_faster_than_ventricular_n": 0,
        "harmonic_consistent_and_faster_n": 0,
        "harmonic_phase_dissociation_n": 0,
        "harmonic_ventricular_regular_n": 0,
        "harmonic_consistent_and_phase_n": 0,
        "harmonic_faster_and_phase_n": 0,
        "harmonic_consistent_faster_phase_n": 0,
        "harmonic_consistent_faster_regular_n": 0,
        "harmonic_complete_block_mechanism_n": 0,
        "unseeded_organized_n": 1,
        "organized_augmented_n": 1,
        "observed_organized_n": 0,
    }, unseeded
    harmonic = _shadow_harmonic_relation(
        {
            "unseeded_events": [
                {"time_ms": 500.0},
                {"time_ms": 1150.0},
                {"time_ms": 1800.0},
                {"time_ms": 2450.0},
            ]
        },
        {"r_peaks_samples": [450, 1150, 1850, 2550]},
        500,
    )
    assert harmonic["evaluable"], harmonic
    assert harmonic["harmonic_consistent"], harmonic
    assert harmonic["faster_than_ventricular"], harmonic
    assert harmonic["harmonic_consistent_and_faster"], harmonic
    assert harmonic["ventricular_regular"], harmonic
    assert harmonic["phase_dissociation"], harmonic
    assert harmonic["complete_block_mechanism"], harmonic
    print("MEDCALC_ATRIAL_EVIDENCE_SYNTHETIC_AUDIT_SELFTEST_PASS")


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
            raise SystemExit("--shard-index is required unless --aggregate-dir is used")
        if not (0 <= args.shard_index < args.shard_count):
            raise SystemExit("invalid shard index")
        result = run_shard(args.shard_index, args.shard_count)

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
