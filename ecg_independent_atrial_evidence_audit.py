from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_independent_atrial_evidence import (
    PREFERRED_LEADS,
    recover_crosslead_atrial_candidates,
    recover_morphology_matched_atrial_candidates,
)
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

VERSION = "MEDCALC_ATRIAL_EVIDENCE_SYNTHETIC_AUDIT_V3_SHADOW_BOOTSTRAP"


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "recovered_any_n": 0,
        "unseeded_any_n": 0,
        "unseeded_ge4_n": 0,
        "unseeded_event_total": 0,
        "unseeded_ge4_stable_pr_n": 0,
        "unseeded_ge4_unstable_pr_n": 0,
        "shadow_bootstrap_evaluable_n": 0,
        "shadow_bootstrap_added_any_n": 0,
        "shadow_bootstrap_event_total": 0,
        "shadow_bootstrap_organized_n": 0,
        "unseeded_organized_n": 0,
        "organized_augmented_n": 0,
        "observed_organized_n": 0,
    }


def _apply(
    row: dict[str, int],
    evidence: dict[str, Any],
    pr_relation: dict[str, Any] | None = None,
    shadow_bootstrap: dict[str, Any] | None = None,
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
    shadow = shadow_bootstrap or {}
    row["shadow_bootstrap_evaluable_n"] += int(bool(shadow.get("evaluable")))
    row["shadow_bootstrap_added_any_n"] += int(int(shadow.get("added_event_n") or 0) > 0)
    row["shadow_bootstrap_event_total"] += int(shadow.get("combined_event_n") or 0)
    row["shadow_bootstrap_organized_n"] += int(bool(shadow.get("organized")))
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

def _shadow_bootstrap_from_unseeded(
    canonical_ecg: dict[str, Any],
    per_lead: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    *,
    coincidence_ms: float = 36.0,
) -> dict[str, Any]:
    """Shadow-only morphology expansion from observed unseeded events.

    This never mutates canonical P fiducials and is not returned to the clinical
    ECG pipeline. It asks whether cross-lead unseeded events can act as observed
    morphology seeds and recover additional homologous atrial deflections.
    """
    seed_events = [
        row for row in (evidence.get("unseeded_events") or [])
        if row.get("time_ms") is not None
    ]
    if len(seed_events) < 3:
        return {
            "evaluable": False,
            "seed_event_n": len(seed_events),
            "added_event_n": 0,
            "combined_event_n": len(seed_events),
            "organized": False,
            "reason": "LT_3_UNSEEDED_CROSSLEAD_SEEDS",
        }

    lead_items = canonical_ecg.get("leads") or {}
    observations: list[tuple[float, str, int]] = []
    seeded_lead_n = 0
    for lead in PREFERRED_LEADS:
        measured = per_lead.get(lead) or {}
        source = lead_items.get(lead) or {}
        if not measured.get("evaluable"):
            continue
        fs = int(measured.get("fs") or source.get("fs") or canonical_ecg.get("fs") or 0)
        signal = source.get("signal_mv")
        if fs <= 0 or signal is None:
            continue
        lead_seed_samples = sorted(set(
            int(round(float(row["time_ms"]) * fs / 1000.0))
            for row in seed_events
            if lead in (row.get("support_leads") or [])
        ))
        if len(lead_seed_samples) < 3:
            continue
        seeded_lead_n += 1
        t_offsets = [
            int(beat["t_offset_sample"])
            for beat in (measured.get("beats") or [])
            if beat.get("t_offset_sample") is not None
        ]
        candidates, _ = recover_morphology_matched_atrial_candidates(
            signal,
            seed_p_peaks=lead_seed_samples,
            r_peaks=list(measured.get("r_peaks_samples") or []),
            t_offsets=t_offsets,
            fs=fs,
        )
        for sample in candidates:
            observations.append((1000.0 * sample / fs, lead, int(sample)))

    observations.sort(key=lambda row: (row[0], row[1], row[2]))
    clusters: list[list[tuple[float, str, int]]] = []
    for obs in observations:
        if not clusters:
            clusters.append([obs])
            continue
        center = float(np.median([row[0] for row in clusters[-1]]))
        if abs(obs[0] - center) <= coincidence_ms:
            clusters[-1].append(obs)
        else:
            clusters.append([obs])

    added_events = []
    for cluster in clusters:
        leads = sorted(set(row[1] for row in cluster))
        if len(leads) < 3:
            continue
        times = [row[0] for row in cluster]
        added_events.append({
            "time_ms": round(float(np.median(times)), 3),
            "support_lead_n": len(leads),
        })

    combined_times = sorted(float(row["time_ms"]) for row in seed_events)
    for row in added_events:
        t_ms = float(row["time_ms"])
        if not any(abs(t_ms - prev) <= coincidence_ms for prev in combined_times):
            combined_times.append(t_ms)
    combined_times.sort()

    pp = (
        np.diff(np.asarray(combined_times, dtype=float))
        if len(combined_times) >= 2 else np.asarray([], dtype=float)
    )
    pp_median = float(np.median(pp)) if pp.size else None
    pp_cv = (
        float(np.std(pp, ddof=1) / np.mean(pp))
        if pp.size >= 2 and float(np.mean(pp)) > 0 else None
    )
    organized = bool(
        len(combined_times) >= 4
        and pp_median is not None
        and 300.0 <= pp_median <= 1500.0
        and pp_cv is not None
        and pp_cv <= 0.12
    )
    return {
        "evaluable": True,
        "seed_event_n": len(seed_events),
        "seeded_lead_n": seeded_lead_n,
        "added_event_n": len(added_events),
        "combined_event_n": len(combined_times),
        "pp_median_ms": round(pp_median, 3) if pp_median is not None else None,
        "pp_cv": round(pp_cv, 6) if pp_cv is not None else None,
        "organized": organized,
        "policy": (
            "SHADOW_ONLY; GE_3_UNSEEDED_CROSSLEAD_SEEDS; "
            "GE_3_LEAD_MORPHOLOGY_RECOVERY; PP_CV_LE_0_12; "
            "NO_CANONICAL_P_MUTATION; NO_CLINICAL_OUTPUT"
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
            pr_relation = _unseeded_pr_relation(
                evidence,
                analysis.get("rhythm") or {},
                int(analysis.get("fs") or canonical_ecg.get("fs") or 0),
            )
            shadow_bootstrap = _shadow_bootstrap_from_unseeded(
                canonical_ecg,
                analysis.get("leads") or {},
                evidence,
            )
            if spec.get("kind") == "TARGET":
                _apply(per_target[str(spec["target"])], evidence, pr_relation, shadow_bootstrap)
            else:
                _apply(controls, evidence, pr_relation, shadow_bootstrap)
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply(control_types.setdefault(ctype, _blank()), evidence, pr_relation, shadow_bootstrap)
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
    for key in ("n", "recovered_any_n", "unseeded_any_n", "unseeded_ge4_n", "unseeded_event_total", "unseeded_ge4_stable_pr_n", "unseeded_ge4_unstable_pr_n", "shadow_bootstrap_evaluable_n", "shadow_bootstrap_added_any_n", "shadow_bootstrap_event_total", "shadow_bootstrap_organized_n", "unseeded_organized_n", "organized_augmented_n", "observed_organized_n"):
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
            "shadow_bootstrap_evaluable_fraction": row["shadow_bootstrap_evaluable_n"] / n,
            "shadow_bootstrap_added_any_fraction": row["shadow_bootstrap_added_any_n"] / n,
            "shadow_bootstrap_event_mean": row["shadow_bootstrap_event_total"] / n,
            "shadow_bootstrap_organized_fraction": row["shadow_bootstrap_organized_n"] / n,
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
        "shadow_bootstrap_evaluable_n": 0,
        "shadow_bootstrap_added_any_n": 0,
        "shadow_bootstrap_event_total": 0,
        "shadow_bootstrap_organized_n": 0,
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
        "shadow_bootstrap_evaluable_n": 0,
        "shadow_bootstrap_added_any_n": 0,
        "shadow_bootstrap_event_total": 0,
        "shadow_bootstrap_organized_n": 0,
        "unseeded_organized_n": 1,
        "organized_augmented_n": 1,
        "observed_organized_n": 0,
    }, unseeded
    shadow_empty = _shadow_bootstrap_from_unseeded({}, {}, {"unseeded_events": []})
    assert not shadow_empty["evaluable"], shadow_empty
    assert shadow_empty["reason"] == "LT_3_UNSEEDED_CROSSLEAD_SEEDS", shadow_empty
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

    text = json.dumps(result, indent=2, sort_keys=True) + "\\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
