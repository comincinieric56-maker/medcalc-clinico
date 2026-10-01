from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from ecg_av_conduction import analyze_av_conduction
from ecg_atrial_clean_r_mask_shadow_audit import _clean_selected_rhythm, _shadow_mask_inputs
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    CONTROL_N, DIAGNOSTIC_CASES_EACH, DIAGNOSTIC_GROUPS, FS,
    all_specs, canonical, make_signal,
)

VERSION = "MEDCALC_UNSEEDED_AV_TOPOLOGY_SHADOW_AUDIT_V1"


def _shadow_av(evidence: dict[str, Any], rhythm: dict[str, Any], fs: int) -> dict[str, Any]:
    events = sorted(
        float(row["time_ms"]) for row in (evidence.get("unseeded_events") or [])
        if row.get("time_ms") is not None
    )
    r = sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    if fs <= 0 or len(events) < 4 or len(r) < 3:
        return {"evaluable": False, "classification": "AV_CONDUCTION_NOT_EVALUABLE"}
    p = sorted(set(int(round(v * fs / 1000.0)) for v in events))
    pseudo = {
        "II": {
            "evaluable": True,
            "fs": fs,
            "raw_p_peaks_samples": p,
            "r_peaks_samples": r,
            "confidence": 1.0,
            "atrial_activity": {
                "p_wave_reproducible": False,
                "p_qrs_coupling_fraction": 0.0,
            },
        }
    }
    result = analyze_av_conduction(pseudo, {}, {})
    result["shadow_policy"] = (
        "AUDIT_ONLY; CLEAN_R_UNSEEDED_P_EVENTS_ONLY; REUSES_AV_CONDUCTION_V2; "
        "NO_CANONICAL_P_MUTATION; NO_DIAGNOSTIC_OUTPUT"
    )
    return result


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    selected = [s for i, s in enumerate(all_specs()) if i % shard_count == shard_index]
    groups: dict[str, Counter] = {target: Counter() for target in DIAGNOSTIC_GROUPS}
    controls = Counter()
    control_types: dict[str, Counter] = {}
    errors: list[str] = []
    for spec in selected:
        try:
            signal = make_signal(spec)
            ecg = canonical(spec, signal)
            analysis = analyze_canonical_ecg(ecg)
            shadow_leads, _ = _shadow_mask_inputs(ecg, analysis)
            evidence = recover_crosslead_atrial_candidates(ecg, shadow_leads)
            rhythm = _clean_selected_rhythm(ecg, analysis)
            fs = int(analysis.get("fs") or FS)
            av = _shadow_av(evidence, rhythm, fs)
            if spec.get("kind") == "TARGET":
                dst = groups[str(spec["target"])]
            else:
                dst = controls
            dst["n"] += 1
            dst["unseeded_organized_n"] += int(bool(evidence.get("unseeded_organized")))
            dst["evaluable_n"] += int(bool(av.get("evaluable")))
            cls = str(av.get("classification") or "UNKNOWN")
            dst["classification__" + cls] += 1
            if spec.get("kind") != "TARGET":
                ctype = str(spec.get("control_type") or "UNKNOWN")
                c = control_types.setdefault(ctype, Counter())
                c["n"] += 1
                c["unseeded_organized_n"] += int(bool(evidence.get("unseeded_organized")))
                c["evaluable_n"] += int(bool(av.get("evaluable")))
                c["classification__" + cls] += 1
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
    total_target_n = sum(int(row.get("n", 0)) for row in groups.values())
    expected_target_n = DIAGNOSTIC_CASES_EACH * len(DIAGNOSTIC_GROUPS)
    expected_control_n = CONTROL_N
    shard_is_full = shard_count == 1
    if shard_is_full and (total_target_n != expected_target_n or int(controls.get("n", 0)) != expected_control_n):
        errors.append(
            f"COHORT_COUNT_MISMATCH:targets={total_target_n}/{expected_target_n};"
            f"controls={int(controls.get('n', 0))}/{expected_control_n}"
        )
    return {
        "version": VERSION,
        "role": "DEVELOPMENT_REGRESSION_ONLY",
        "purpose": "AUDIT_ONLY_UNSEEDED_CLEAN_R_AV_TOPOLOGY",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "groups": {k: dict(v) for k, v in sorted(groups.items())},
        "controls": dict(controls),
        "control_types": {k: dict(v) for k, v in sorted(control_types.items())},
        "errors": errors,
        "policy": (
            "NO_CLINICAL_CHANGE; NO_THRESHOLD_TUNING; SYNTHETIC_ONLY; "
            "NO_FAST_GATE; NO_FOLD9_OR_FOLD10; NO_EXTERNAL_OR_FINAL_DATA"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--output")
    args = ap.parse_args()
    result = run_shard(args.shard_index, args.shard_count)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
