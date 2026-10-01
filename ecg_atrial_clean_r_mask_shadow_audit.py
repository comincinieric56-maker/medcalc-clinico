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
from ecg_independent_atrial_evidence_audit import (
    _shadow_harmonic_relation,
    _unseeded_pr_relation,
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

VERSION = "MEDCALC_ATRIAL_CLEAN_R_MASK_SHADOW_AUDIT_V5_OBSERVED_UNSEEDED_TRAIN"
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
        "shadow_pr_evaluable_n": 0,
        "shadow_stable_pr_like_n": 0,
        "shadow_harmonic_evaluable_n": 0,
        "shadow_harmonic_consistent_n": 0,
        "shadow_faster_than_ventricular_n": 0,
        "shadow_ventricular_regular_n": 0,
        "shadow_phase_dissociation_n": 0,
        "shadow_harmonic_consistent_faster_n": 0,
        "shadow_harmonic_faster_not_phase_n": 0,
        "shadow_complete_block_mechanism_n": 0,
        "shadow_organized_harmonic_consistent_n": 0,
        "shadow_organized_faster_n": 0,
        "shadow_organized_ventricular_regular_n": 0,
        "shadow_organized_phase_dissociation_n": 0,
        "shadow_organized_complete_block_mechanism_n": 0,
        "shadow_organized_harmonic_not_faster_n": 0,
        "shadow_organized_phase_not_faster_n": 0,
        "shadow_organized_not_phase_n": 0,
        "shadow_organized_stable_pr_like_n": 0,
        "shadow_combined_event_total": 0,
        "shadow_combined_organized_n": 0,
        "shadow_combined_pr_evaluable_n": 0,
        "shadow_combined_stable_pr_like_n": 0,
        "shadow_combined_harmonic_evaluable_n": 0,
        "shadow_combined_harmonic_consistent_n": 0,
        "shadow_combined_faster_than_ventricular_n": 0,
        "shadow_combined_ventricular_regular_n": 0,
        "shadow_combined_phase_dissociation_n": 0,
        "shadow_combined_harmonic_consistent_faster_n": 0,
        "shadow_combined_complete_block_mechanism_n": 0,
        "shadow_combined_organized_complete_block_mechanism_n": 0,
        "shadow_ou_event_total": 0,
        "shadow_ou_organized_n": 0,
        "shadow_ou_pr_evaluable_n": 0,
        "shadow_ou_stable_pr_like_n": 0,
        "shadow_ou_harmonic_evaluable_n": 0,
        "shadow_ou_harmonic_consistent_n": 0,
        "shadow_ou_faster_than_ventricular_n": 0,
        "shadow_ou_ventricular_regular_n": 0,
        "shadow_ou_phase_dissociation_n": 0,
        "shadow_ou_harmonic_consistent_faster_n": 0,
        "shadow_ou_complete_block_mechanism_n": 0,
        "shadow_ou_organized_complete_block_mechanism_n": 0,
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


def _clean_selected_rhythm(
    canonical_ecg: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Return a shadow rhythm using the same fixed clean-R candidate filter."""
    rhythm = copy.deepcopy(analysis.get("rhythm") or {})
    rhythm_lead = rhythm.get("lead")
    if not rhythm_lead:
        return rhythm
    item = (analysis.get("leads") or {}).get(rhythm_lead) or {}
    source = (canonical_ecg.get("leads") or {}).get(rhythm_lead) or {}
    fs = int(item.get("fs") or source.get("fs") or analysis.get("fs") or FS)
    raw = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    signal_mv = np.asarray(source.get("signal_mv") or [], dtype=float)
    if fs <= 0 or len(raw) < 3 or signal_mv.size == 0:
        return rhythm
    ok, kept = _filter_for_threshold(raw, signal_mv, fs, RELATIVE_AMP_MIN)
    if not ok or len(kept) < 3 or len(kept) >= len(raw):
        return rhythm
    rhythm["r_peaks_samples"] = list(kept)
    rhythm["r_count"] = len(kept)
    return rhythm


def _combined_atrial_evidence(
    evidence: dict[str, Any],
    *,
    coincidence_ms: float = 36.0,
    include_seeded_recovered: bool = True,
) -> dict[str, Any]:
    """Combine observed/recovered atrial events without synthesizing timing."""
    times: list[float] = []
    observed = evidence.get("observed_consensus") or {}
    for row in observed.get("events") or []:
        if row.get("time_ms") is not None:
            times.append(float(row["time_ms"]))
    fields = ["unseeded_events"]
    if include_seeded_recovered:
        fields.insert(0, "recovered_events")
    for field in fields:
        for row in evidence.get(field) or []:
            if row.get("time_ms") is not None:
                times.append(float(row["time_ms"]))
    times.sort()

    deduped: list[float] = []
    for value in times:
        if deduped and abs(value - deduped[-1]) <= coincidence_ms:
            deduped[-1] = float(np.median([deduped[-1], value]))
        else:
            deduped.append(value)

    arr = np.asarray(deduped, dtype=float)
    pp = np.diff(arr) if arr.size >= 2 else np.asarray([], dtype=float)
    pp_median = float(np.median(pp)) if pp.size else None
    pp_cv = (
        float(np.std(pp, ddof=1) / np.mean(pp))
        if pp.size >= 2 and float(np.mean(pp)) > 0
        else None
    )
    organized = bool(
        len(deduped) >= 4
        and pp_median is not None
        and 300.0 <= pp_median <= 1500.0
        and pp_cv is not None
        and pp_cv <= 0.12
    )
    return {
        "unseeded_events": [{"time_ms": round(v, 6)} for v in deduped],
        "unseeded_organized": organized,
        "event_n": len(deduped),
        "pp_median_ms": round(pp_median, 6) if pp_median is not None else None,
        "pp_cv": round(pp_cv, 6) if pp_cv is not None else None,
        "policy": (
            ("AUDIT_ONLY; OBSERVED_PLUS_RECOVERED_EVENTS; " if include_seeded_recovered else "AUDIT_ONLY; OBSERVED_PLUS_UNSEEDED_ONLY; ")
            "NO_EVENT_SYNTHESIS; ORGANIZED_USES_EXISTING_300_1500MS_AND_CV_LE_0_12"
        ),
    }


def _apply(
    dst: dict[str, int],
    baseline: dict[str, Any],
    shadow: dict[str, Any],
    mask_audit: dict[str, Any],
    pr_relation: dict[str, Any] | None = None,
    harmonic_relation: dict[str, Any] | None = None,
    combined_evidence: dict[str, Any] | None = None,
    combined_pr_relation: dict[str, Any] | None = None,
    combined_harmonic_relation: dict[str, Any] | None = None,
    ou_evidence: dict[str, Any] | None = None,
    ou_pr_relation: dict[str, Any] | None = None,
    ou_harmonic_relation: dict[str, Any] | None = None,
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

    pr = pr_relation or {}
    harmonic = harmonic_relation or {}
    dst["shadow_pr_evaluable_n"] += int(bool(pr.get("evaluable")))
    dst["shadow_stable_pr_like_n"] += int(bool(pr.get("stable_pr_like")))
    h_eval = bool(harmonic.get("evaluable"))
    h_consistent = bool(harmonic.get("harmonic_consistent"))
    h_faster = bool(harmonic.get("faster_than_ventricular"))
    h_regular = bool(harmonic.get("ventricular_regular"))
    h_phase = bool(harmonic.get("phase_dissociation"))
    dst["shadow_harmonic_evaluable_n"] += int(h_eval)
    dst["shadow_harmonic_consistent_n"] += int(h_consistent)
    dst["shadow_faster_than_ventricular_n"] += int(h_faster)
    dst["shadow_ventricular_regular_n"] += int(h_regular)
    dst["shadow_phase_dissociation_n"] += int(h_phase)
    dst["shadow_harmonic_consistent_faster_n"] += int(h_consistent and h_faster)
    dst["shadow_harmonic_faster_not_phase_n"] += int(
        h_consistent and h_faster and not h_phase
    )
    h_complete = bool(harmonic.get("complete_block_mechanism"))
    dst["shadow_complete_block_mechanism_n"] += int(h_complete)

    organized = bool(shadow.get("unseeded_organized"))
    stable = bool(pr.get("stable_pr_like"))
    dst["shadow_organized_harmonic_consistent_n"] += int(
        organized and h_consistent
    )
    dst["shadow_organized_faster_n"] += int(organized and h_faster)
    dst["shadow_organized_ventricular_regular_n"] += int(
        organized and h_regular
    )
    dst["shadow_organized_phase_dissociation_n"] += int(
        organized and h_phase
    )
    dst["shadow_organized_complete_block_mechanism_n"] += int(
        organized and h_complete
    )
    dst["shadow_organized_harmonic_not_faster_n"] += int(
        organized and h_consistent and not h_faster
    )
    dst["shadow_organized_phase_not_faster_n"] += int(
        organized and h_phase and not h_faster
    )
    dst["shadow_organized_not_phase_n"] += int(
        organized and not h_phase
    )
    dst["shadow_organized_stable_pr_like_n"] += int(
        organized and stable
    )

    combined = combined_evidence or {}
    combined_pr = combined_pr_relation or {}
    combined_h = combined_harmonic_relation or {}
    dst["shadow_combined_event_total"] += int(combined.get("event_n") or 0)
    combined_organized = bool(combined.get("unseeded_organized"))
    dst["shadow_combined_organized_n"] += int(combined_organized)
    dst["shadow_combined_pr_evaluable_n"] += int(bool(combined_pr.get("evaluable")))
    dst["shadow_combined_stable_pr_like_n"] += int(bool(combined_pr.get("stable_pr_like")))
    ch_eval = bool(combined_h.get("evaluable"))
    ch_consistent = bool(combined_h.get("harmonic_consistent"))
    ch_faster = bool(combined_h.get("faster_than_ventricular"))
    ch_regular = bool(combined_h.get("ventricular_regular"))
    ch_phase = bool(combined_h.get("phase_dissociation"))
    ch_complete = bool(combined_h.get("complete_block_mechanism"))
    dst["shadow_combined_harmonic_evaluable_n"] += int(ch_eval)
    dst["shadow_combined_harmonic_consistent_n"] += int(ch_consistent)
    dst["shadow_combined_faster_than_ventricular_n"] += int(ch_faster)
    dst["shadow_combined_ventricular_regular_n"] += int(ch_regular)
    dst["shadow_combined_phase_dissociation_n"] += int(ch_phase)
    dst["shadow_combined_harmonic_consistent_faster_n"] += int(
        ch_consistent and ch_faster
    )
    dst["shadow_combined_complete_block_mechanism_n"] += int(ch_complete)
    dst["shadow_combined_organized_complete_block_mechanism_n"] += int(
        combined_organized and ch_complete
    )

    ou = ou_evidence or {}
    ou_pr = ou_pr_relation or {}
    ou_h = ou_harmonic_relation or {}
    dst["shadow_ou_event_total"] += int(ou.get("event_n") or 0)
    ou_organized = bool(ou.get("unseeded_organized"))
    dst["shadow_ou_organized_n"] += int(ou_organized)
    dst["shadow_ou_pr_evaluable_n"] += int(bool(ou_pr.get("evaluable")))
    dst["shadow_ou_stable_pr_like_n"] += int(bool(ou_pr.get("stable_pr_like")))
    oh_eval = bool(ou_h.get("evaluable"))
    oh_consistent = bool(ou_h.get("harmonic_consistent"))
    oh_faster = bool(ou_h.get("faster_than_ventricular"))
    oh_regular = bool(ou_h.get("ventricular_regular"))
    oh_phase = bool(ou_h.get("phase_dissociation"))
    oh_complete = bool(ou_h.get("complete_block_mechanism"))
    dst["shadow_ou_harmonic_evaluable_n"] += int(oh_eval)
    dst["shadow_ou_harmonic_consistent_n"] += int(oh_consistent)
    dst["shadow_ou_faster_than_ventricular_n"] += int(oh_faster)
    dst["shadow_ou_ventricular_regular_n"] += int(oh_regular)
    dst["shadow_ou_phase_dissociation_n"] += int(oh_phase)
    dst["shadow_ou_harmonic_consistent_faster_n"] += int(
        oh_consistent and oh_faster
    )
    dst["shadow_ou_complete_block_mechanism_n"] += int(oh_complete)
    dst["shadow_ou_organized_complete_block_mechanism_n"] += int(
        ou_organized and oh_complete
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
            analysis_fs = int(analysis.get("fs") or canonical_ecg.get("fs") or 0)
            shadow_rhythm = _clean_selected_rhythm(canonical_ecg, analysis)
            pr_relation = _unseeded_pr_relation(
                shadow,
                shadow_rhythm,
                analysis_fs,
            )
            harmonic_relation = _shadow_harmonic_relation(
                shadow,
                shadow_rhythm,
                analysis_fs,
            )
            combined_evidence = _combined_atrial_evidence(shadow)
            combined_pr_relation = _unseeded_pr_relation(
                combined_evidence,
                shadow_rhythm,
                analysis_fs,
            )
            combined_harmonic_relation = _shadow_harmonic_relation(
                combined_evidence,
                shadow_rhythm,
                analysis_fs,
            )
            ou_evidence = _combined_atrial_evidence(
                shadow,
                include_seeded_recovered=False,
            )
            ou_pr_relation = _unseeded_pr_relation(
                ou_evidence,
                shadow_rhythm,
                analysis_fs,
            )
            ou_harmonic_relation = _shadow_harmonic_relation(
                ou_evidence,
                shadow_rhythm,
                analysis_fs,
            )

            if spec.get("kind") == "TARGET":
                _apply(
                    per_target[str(spec["target"])],
                    baseline,
                    shadow,
                    mask_audit,
                    pr_relation,
                    harmonic_relation,
                    combined_evidence,
                    combined_pr_relation,
                    combined_harmonic_relation,
                    ou_evidence,
                    ou_pr_relation,
                    ou_harmonic_relation,
                )
            else:
                _apply(
                    controls,
                    baseline,
                    shadow,
                    mask_audit,
                    pr_relation,
                    harmonic_relation,
                    combined_evidence,
                    combined_pr_relation,
                    combined_harmonic_relation,
                    ou_evidence,
                    ou_pr_relation,
                    ou_harmonic_relation,
                )
                ctype = str(spec.get("control_type") or "UNKNOWN")
                _apply(
                    control_types.setdefault(ctype, _blank()),
                    baseline,
                    shadow,
                    mask_audit,
                    pr_relation,
                    harmonic_relation,
                    combined_evidence,
                    combined_pr_relation,
                    combined_harmonic_relation,
                    ou_evidence,
                    ou_pr_relation,
                    ou_harmonic_relation,
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
            "shadow_pr_evaluable_fraction": row["shadow_pr_evaluable_n"] / n,
            "shadow_stable_pr_like_fraction": row["shadow_stable_pr_like_n"] / n,
            "shadow_harmonic_evaluable_fraction": row["shadow_harmonic_evaluable_n"] / n,
            "shadow_harmonic_consistent_fraction": row["shadow_harmonic_consistent_n"] / n,
            "shadow_faster_than_ventricular_fraction": row["shadow_faster_than_ventricular_n"] / n,
            "shadow_ventricular_regular_fraction": row["shadow_ventricular_regular_n"] / n,
            "shadow_phase_dissociation_fraction": row["shadow_phase_dissociation_n"] / n,
            "shadow_harmonic_consistent_faster_fraction": row["shadow_harmonic_consistent_faster_n"] / n,
            "shadow_harmonic_faster_not_phase_fraction": row["shadow_harmonic_faster_not_phase_n"] / n,
            "shadow_complete_block_mechanism_fraction": row["shadow_complete_block_mechanism_n"] / n,
            "shadow_organized_harmonic_consistent_fraction": row["shadow_organized_harmonic_consistent_n"] / n,
            "shadow_organized_faster_fraction": row["shadow_organized_faster_n"] / n,
            "shadow_organized_ventricular_regular_fraction": row["shadow_organized_ventricular_regular_n"] / n,
            "shadow_organized_phase_dissociation_fraction": row["shadow_organized_phase_dissociation_n"] / n,
            "shadow_organized_complete_block_mechanism_fraction": row["shadow_organized_complete_block_mechanism_n"] / n,
            "shadow_organized_harmonic_not_faster_fraction": row["shadow_organized_harmonic_not_faster_n"] / n,
            "shadow_organized_phase_not_faster_fraction": row["shadow_organized_phase_not_faster_n"] / n,
            "shadow_organized_not_phase_fraction": row["shadow_organized_not_phase_n"] / n,
            "shadow_organized_stable_pr_like_fraction": row["shadow_organized_stable_pr_like_n"] / n,
            "shadow_combined_event_mean": row["shadow_combined_event_total"] / n,
            "shadow_combined_organized_fraction": row["shadow_combined_organized_n"] / n,
            "shadow_combined_pr_evaluable_fraction": row["shadow_combined_pr_evaluable_n"] / n,
            "shadow_combined_stable_pr_like_fraction": row["shadow_combined_stable_pr_like_n"] / n,
            "shadow_combined_harmonic_evaluable_fraction": row["shadow_combined_harmonic_evaluable_n"] / n,
            "shadow_combined_harmonic_consistent_fraction": row["shadow_combined_harmonic_consistent_n"] / n,
            "shadow_combined_faster_than_ventricular_fraction": row["shadow_combined_faster_than_ventricular_n"] / n,
            "shadow_combined_ventricular_regular_fraction": row["shadow_combined_ventricular_regular_n"] / n,
            "shadow_combined_phase_dissociation_fraction": row["shadow_combined_phase_dissociation_n"] / n,
            "shadow_combined_harmonic_consistent_faster_fraction": row["shadow_combined_harmonic_consistent_faster_n"] / n,
            "shadow_combined_complete_block_mechanism_fraction": row["shadow_combined_complete_block_mechanism_n"] / n,
            "shadow_combined_organized_complete_block_mechanism_fraction": row["shadow_combined_organized_complete_block_mechanism_n"] / n,
            "shadow_ou_event_mean": row["shadow_ou_event_total"] / n,
            "shadow_ou_organized_fraction": row["shadow_ou_organized_n"] / n,
            "shadow_ou_pr_evaluable_fraction": row["shadow_ou_pr_evaluable_n"] / n,
            "shadow_ou_stable_pr_like_fraction": row["shadow_ou_stable_pr_like_n"] / n,
            "shadow_ou_harmonic_evaluable_fraction": row["shadow_ou_harmonic_evaluable_n"] / n,
            "shadow_ou_harmonic_consistent_fraction": row["shadow_ou_harmonic_consistent_n"] / n,
            "shadow_ou_faster_than_ventricular_fraction": row["shadow_ou_faster_than_ventricular_n"] / n,
            "shadow_ou_ventricular_regular_fraction": row["shadow_ou_ventricular_regular_n"] / n,
            "shadow_ou_phase_dissociation_fraction": row["shadow_ou_phase_dissociation_n"] / n,
            "shadow_ou_harmonic_consistent_faster_fraction": row["shadow_ou_harmonic_consistent_faster_n"] / n,
            "shadow_ou_complete_block_mechanism_fraction": row["shadow_ou_complete_block_mechanism_n"] / n,
            "shadow_ou_organized_complete_block_mechanism_fraction": row["shadow_ou_organized_complete_block_mechanism_n"] / n,
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
    combined = _combined_atrial_evidence({
        "observed_consensus": {"events": [{"time_ms": 1000.0}, {"time_ms": 2000.0}]},
        "recovered_events": [],
        "unseeded_events": [{"time_ms": 1500.0}, {"time_ms": 2500.0}],
    })
    assert combined["event_n"] == 4, combined
    assert combined["unseeded_organized"], combined
    ou = _combined_atrial_evidence({
        "observed_consensus": {"events": [{"time_ms": 1000.0}, {"time_ms": 2000.0}]},
        "recovered_events": [{"time_ms": 1250.0}],
        "unseeded_events": [{"time_ms": 1500.0}, {"time_ms": 2500.0}],
    }, include_seeded_recovered=False)
    assert ou["event_n"] == 4, ou
    assert all(abs(float(x["time_ms"]) - 1250.0) > 1e-6 for x in ou["unseeded_events"]), ou
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
