from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    PTBXL_VERSION,
    TARGETS,
    _adult_rows,
    _any_target_positive,
    _canonical,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _target_positive,
)
from ecg_analysis_cache import analyze_with_cache
from ecg_signal_measurements import analyze_canonical_ecg

FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
NEGATIVE_N = 160
CLUSTER_MS = 50.0
MIN_PR_MS = 80.0
MAX_PR_MS = 500.0
REGULAR_CV_MAX = 0.12
STABLE_PR_MAD_MAX_MS = 30.0


def _consensus_p_ms(analysis: dict[str, Any], support_n: int) -> list[float]:
    events: list[tuple[float, str]] = []
    leads = analysis.get("leads") or {}
    global_fs = int(analysis.get("fs") or 500)

    for lead, item_raw in leads.items():
        item = dict(item_raw or {})
        fs = int(item.get("fs") or global_fs)
        for p in item.get("raw_p_peaks_samples") or []:
            events.append((float(p) * 1000.0 / max(fs, 1), str(lead)))

    events.sort(key=lambda row: (row[0], row[1]))
    out: list[float] = []
    i = 0
    while i < len(events):
        seed_t = events[i][0]
        group: list[tuple[float, str]] = []
        j = i
        while j < len(events) and events[j][0] - seed_t <= CLUSTER_MS:
            group.append(events[j])
            j += 1

        by_lead: dict[str, list[float]] = {}
        for t, lead in group:
            by_lead.setdefault(lead, []).append(t)
        if len(by_lead) >= support_n:
            per_lead = [float(np.median(v)) for v in by_lead.values()]
            out.append(float(np.median(per_lead)))
        i = j

    # Adjacent accepted clusters must remain separated; this protects against
    # duplicated detections from neighboring leads creating two P events.
    dedup: list[float] = []
    for t in out:
        if not dedup or t - dedup[-1] > CLUSTER_MS:
            dedup.append(t)
    return dedup


def _qrs_ms(analysis: dict[str, Any]) -> list[float]:
    rhythm = analysis.get("rhythm") or {}
    fs = int(analysis.get("fs") or 500)
    peaks = np.unique(np.asarray(rhythm.get("r_peaks_samples") or [], dtype=int))
    return [float(x) * 1000.0 / max(fs, 1) for x in peaks.tolist()]


def _cv(intervals: np.ndarray) -> float | None:
    if intervals.size < 2:
        return None
    mean = float(np.mean(intervals))
    if mean <= 0:
        return None
    return float(np.std(intervals, ddof=1) / mean)


def _map_p_to_qrs(p_ms: list[float], qrs_ms: list[float]) -> list[dict[str, Any]]:
    p = np.asarray(p_ms, dtype=float)
    mappings = [
        {"p_ms": float(pi), "conducted": False, "qrs_ms": None, "pr_ms": None}
        for pi in p
    ]
    claimed: set[int] = set()

    for ri in np.asarray(qrs_ms, dtype=float):
        candidate_idx = np.where(
            (p < ri)
            & ((ri - p) >= MIN_PR_MS)
            & ((ri - p) <= MAX_PR_MS)
        )[0]
        chosen = None
        for idx in candidate_idx[::-1]:
            j = int(idx)
            if j not in claimed:
                chosen = j
                break
        if chosen is None:
            continue
        pr = float(ri - p[chosen])
        mappings[chosen] = {
            "p_ms": float(p[chosen]),
            "conducted": True,
            "qrs_ms": float(ri),
            "pr_ms": pr,
        }
        claimed.add(chosen)
    return mappings


def _physiology(analysis: dict[str, Any], support_n: int) -> dict[str, Any]:
    p = _consensus_p_ms(analysis, support_n)
    qrs = _qrs_ms(analysis)
    pp = np.diff(np.asarray(p, dtype=float)) if len(p) >= 2 else np.asarray([], dtype=float)
    rr = np.diff(np.asarray(qrs, dtype=float)) if len(qrs) >= 2 else np.asarray([], dtype=float)

    pp_med = float(np.median(pp)) if pp.size else None
    rr_med = float(np.median(rr)) if rr.size else None
    pp_cv = _cv(pp)
    rr_cv = _cv(rr)
    atrial_regular = bool(pp_cv is not None and pp_cv <= REGULAR_CV_MAX)
    ventricular_regular = bool(rr_cv is not None and rr_cv <= REGULAR_CV_MAX)

    mappings = _map_p_to_qrs(p, qrs)
    conducted = [m for m in mappings if m["conducted"]]
    dropped = [m for m in mappings if not m["conducted"]]
    pr = np.asarray([float(m["pr_ms"]) for m in conducted], dtype=float)
    pr_med = float(np.median(pr)) if pr.size else None
    pr_mad = (
        float(np.median(np.abs(pr - np.median(pr))))
        if pr.size else None
    )
    stable_pr = bool(
        pr.size >= 3
        and pr_mad is not None
        and pr_mad <= STABLE_PR_MAD_MAX_MS
    )
    coupling = float(len(conducted) / max(len(p), 1))

    atrial_rate = 60000.0 / pp_med if pp_med and pp_med > 0 else None
    ventricular_rate = 60000.0 / rr_med if rr_med and rr_med > 0 else None
    atrial_faster_125 = bool(
        atrial_rate is not None
        and ventricular_rate is not None
        and atrial_rate > 1.25 * ventricular_rate
    )

    phase = []
    p_arr = np.asarray(p, dtype=float)
    for ri in np.asarray(qrs, dtype=float):
        prior = p_arr[p_arr < ri]
        if prior.size == 0:
            continue
        delta = float(ri - prior[-1])
        if pp_med is None or delta <= 1.05 * pp_med:
            phase.append(delta)
    phase_arr = np.asarray(phase, dtype=float)
    phase_mad = (
        float(np.median(np.abs(phase_arr - np.median(phase_arr))))
        if phase_arr.size >= 3 else None
    )
    phase_range = (
        float(np.max(phase_arr) - np.min(phase_arr))
        if phase_arr.size >= 3 else None
    )
    dissociation = bool(
        pp_med is not None
        and phase_mad is not None
        and phase_range is not None
        and phase_mad >= max(50.0, 0.15 * pp_med)
        and phase_range >= 0.30 * pp_med
    )

    flags = [bool(m["conducted"]) for m in mappings]
    run = 0
    max_consecutive_drop = 0
    for flag in flags:
        if flag:
            run = 0
        else:
            run += 1
            max_consecutive_drop = max(max_consecutive_drop, run)

    p_qrs_ratio = float(len(p) / max(len(qrs), 1))
    even_conducted = sum(flags[::2])
    odd_conducted = sum(flags[1::2])
    alternating = bool(
        flags
        and abs(even_conducted - odd_conducted) >= max(1, len(flags) // 3)
    )

    progressive = False
    for j, mapping in enumerate(mappings):
        if mapping["conducted"] or j < 3:
            continue
        prior_pr = [
            float(x["pr_ms"])
            for x in mappings[max(0, j - 3):j]
            if x["conducted"] and x["pr_ms"] is not None
        ]
        if (
            len(prior_pr) >= 3
            and all((b - a) > 8.0 for a, b in zip(prior_pr[:-1], prior_pr[1:]))
        ):
            progressive = True
            break

    high_grade_input = bool(
        atrial_regular and len(dropped) >= 1 and len(conducted) >= 2
    )
    complete_pre_phase = bool(
        atrial_regular
        and ventricular_regular
        and len(p) >= 5
        and len(qrs) >= 3
        and atrial_faster_125
    )

    return {
        "evaluable": bool(len(p) >= 4 and len(qrs) >= 3),
        "p_count": int(len(p)),
        "qrs_count": int(len(qrs)),
        "pp_cv": pp_cv,
        "rr_cv": rr_cv,
        "conducted_p_n": int(len(conducted)),
        "nonconducted_p_n": int(len(dropped)),
        "coupling_fraction": coupling,
        "pr_median_ms": pr_med,
        "pr_mad_ms": pr_mad,
        "phase_mad_ms": phase_mad,
        "phase_range_ms": phase_range,
        "atrial_rate_bpm": atrial_rate,
        "ventricular_rate_bpm": ventricular_rate,
        "p_qrs_count_ratio": p_qrs_ratio,
        "flags": {
            "ATRIAL_REGULAR": atrial_regular,
            "VENTRICULAR_REGULAR": ventricular_regular,
            "DROPPED_GE1": len(dropped) >= 1,
            "CONDUCTED_GE2": len(conducted) >= 2,
            "STABLE_PR": stable_pr,
            "ATRIAL_RATE_GT_1_25_VENTRICULAR": atrial_faster_125,
            "AV_DISSOCIATION_PHASE": dissociation,
            "HIGH_GRADE_INPUT_PATTERN": high_grade_input,
            "TWO_TO_ONE_CORE": bool(
                high_grade_input
                and 1.75 <= p_qrs_ratio <= 2.25
                and alternating
            ),
            "GE2_CONSECUTIVE_NONCONDUCTED_P": bool(
                high_grade_input and max_consecutive_drop >= 2
            ),
            "MOBITZ_II_CORE": bool(high_grade_input and stable_pr),
            "WENCKEBACH_CORE": bool(high_grade_input and progressive),
            "COMPLETE_PRE_PHASE_CORE": complete_pre_phase,
            "COMPLETE_AV_BLOCK_CORE": bool(
                complete_pre_phase and dissociation and not stable_pr
            ),
        },
    }


def _quantiles(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not arr.size:
        return {"median": None, "p25": None, "p75": None}
    return {
        "median": round(float(np.median(arr)), 6),
        "p25": round(float(np.percentile(arr, 25)), 6),
        "p75": round(float(np.percentile(arr, 75)), 6),
    }


def _summ(rows: list[dict[str, Any]], group: str, support_n: int) -> dict[str, Any]:
    z = [r[f"s{support_n}"] for r in rows if r["group"] == group]
    flag_names = sorted(z[0]["flags"]) if z else []
    continuous = [
        "p_count",
        "qrs_count",
        "pp_cv",
        "rr_cv",
        "conducted_p_n",
        "nonconducted_p_n",
        "coupling_fraction",
        "pr_median_ms",
        "pr_mad_ms",
        "phase_mad_ms",
        "phase_range_ms",
        "atrial_rate_bpm",
        "ventricular_rate_bpm",
        "p_qrs_count_ratio",
    ]
    return {
        "n": len(z),
        "evaluable_n": sum(bool(r["evaluable"]) for r in z),
        "flag_counts": {
            name: sum(bool(r["flags"][name]) for r in z)
            for name in flag_names
        },
        "distributions": {
            name: _quantiles([
                float(r[name])
                for r in z
                if r.get(name) is not None
            ])
            for name in continuous
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workdir",
        type=Path,
        default=Path("/tmp/medcalc-av-pqrs-physiology"),
    )
    ap.add_argument(
        "--analysis-cache-dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/MEDCALC_AV_PQRS_PHYSIOLOGY_AUDIT.json"),
    )
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.workdir / "ptbxl_database.csv"
    _download(f"{BASE}/ptbxl_database.csv", metadata_path)
    meta = pd.read_csv(metadata_path)
    adult = _adult_rows(meta, FOLDS)
    holdout = _fast_gate_holdout_ids()
    adult = adult.loc[
        ~adult["ecg_id"].astype(int).isin(holdout)
    ].copy()

    parts = []
    for target in ("AVB2", "AVB3"):
        spec = TARGETS[target]
        z = adult[
            adult["_codes"].map(
                lambda x, a=spec["scp"]: _target_positive(x, a)
            )
        ].copy()
        z = z.sort_values(["_hash", "ecg_id"])
        z["_group"] = target
        parts.append(z)

    neg = adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg = neg.sort_values(["_hash", "ecg_id"]).head(NEGATIVE_N)
    neg["_group"] = "CLEAN_CONTROL"

    selected = (
        pd.concat(parts + [neg], ignore_index=True)
        .drop_duplicates(subset=["ecg_id", "_group"])
    )
    if selected["ecg_id"].astype(int).isin(holdout).any():
        raise RuntimeError("FAST_GATE_100 contamination detected")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    root = args.workdir / "records"

    for i, row in selected.iterrows():
        ecg_id = int(row["ecg_id"])
        group = str(row["_group"])
        try:
            local = _ensure_record(root, str(row["filename_hr"]))
            rec = wfdb.rdrecord(str(local))
            canonical = _canonical(
                rec.p_signal,
                int(round(float(rec.fs))),
                list(rec.sig_name),
                ecg_id,
            )
            if args.analysis_cache_dir is not None:
                analysis = analyze_with_cache(
                    canonical,
                    cache_dir=args.analysis_cache_dir,
                    record_key=f"ptbxl-{PTBXL_VERSION}-{ecg_id}",
                )
            else:
                analysis = analyze_canonical_ecg(canonical)
            rows.append({
                "group": group,
                "s2": _physiology(analysis, 2),
                "s3": _physiology(analysis, 3),
            })
        except Exception as exc:
            errors.append({
                "group": group,
                "error": f"{type(exc).__name__}:{exc}",
            })
        if (i + 1) % 25 == 0:
            print(
                f"MEDCALC_AV_PQRS_PHYSIOLOGY {i+1}/{len(selected)}",
                flush=True,
            )

    error_counts = Counter(e["group"] for e in errors)
    result = {
        "version": "MEDCALC_AV_PQRS_PHYSIOLOGY_AUDIT_V1",
        "role": "DEVELOPMENT_TUNING_AUDIT_ONLY",
        "external_validation_claim_allowed": False,
        "clinical_rule_changed": False,
        "dataset": "PTB-XL",
        "dataset_version": PTBXL_VERSION,
        "folds": FOLDS,
        "fast_gate_100_excluded_n": len(holdout),
        "selection": {
            "AVB2_selected_n": int(sum(selected["_group"] == "AVB2")),
            "AVB3_selected_n": int(sum(selected["_group"] == "AVB3")),
            "clean_control_selected_n": int(
                sum(selected["_group"] == "CLEAN_CONTROL")
            ),
        },
        "predefined_physiology": {
            "cluster_window_ms": CLUSTER_MS,
            "support_levels": [2, 3],
            "pr_window_ms": [MIN_PR_MS, MAX_PR_MS],
            "regularity_cv_max": REGULAR_CV_MAX,
            "stable_pr_mad_max_ms": STABLE_PR_MAD_MAX_MS,
            "complete_block_rate_ratio_min": 1.25,
            "phase_dissociation_rule": (
                "MAD>=max(50ms,0.15*PP_median) and range>=0.30*PP_median"
            ),
            "source": (
                "Existing MEDCALC AV_CONDUCTION_V2 physiology applied to "
                "cross-lead consensus P events; no threshold optimized here."
            ),
        },
        "support_2": {
            "AVB2": _summ(rows, "AVB2", 2),
            "AVB3": _summ(rows, "AVB3", 2),
            "clean_control": _summ(rows, "CLEAN_CONTROL", 2),
        },
        "support_3": {
            "AVB2": _summ(rows, "AVB2", 3),
            "AVB3": _summ(rows, "AVB3", 3),
            "clean_control": _summ(rows, "CLEAN_CONTROL", 3),
        },
        "analysis_error_n": len(errors),
        "analysis_error_by_group": dict(sorted(error_counts.items())),
        "case_level_results_emitted": False,
        "note": (
            "Aggregate developmental anatomy only. FAST-GATE-100 and folds "
            "9/10 are excluded; results must not be reported as clinical "
            "validation or used for case-specific rules."
        ),
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
