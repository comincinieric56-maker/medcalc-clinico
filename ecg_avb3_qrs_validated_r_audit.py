from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_avb3_ventricular_detection_audit import _cv, _greedy_match, _near_any
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    FS,
    ROLE,
    _event_times,
    _seed,
    all_specs,
    canonical,
    make_signal,
)

VERSION = "MEDCALC_AVB3_QRS_VALIDATED_R_AUDIT_V1"
REGULAR_CV_MAX = 0.12


def _cluster_validated_r(
    per_lead: dict[str, dict[str, Any]],
    *,
    coincidence_ms: float = 50.0,
    min_leads: int = 3,
) -> list[float]:
    observations: list[tuple[float, str]] = []
    for lead, item in (per_lead or {}).items():
        fs = int(item.get("fs") or FS)
        if not item.get("evaluable") or fs <= 0:
            continue
        for beat in item.get("beats") or []:
            sample = beat.get("r_sample")
            if sample is None:
                continue
            observations.append((1000.0 * int(sample) / fs, str(lead)))

    observations.sort(key=lambda row: (row[0], row[1]))
    clusters: list[list[tuple[float, str]]] = []
    for obs in observations:
        if not clusters:
            clusters.append([obs])
            continue
        center = float(np.median([row[0] for row in clusters[-1]]))
        if abs(float(obs[0]) - center) <= coincidence_ms:
            clusters[-1].append(obs)
        else:
            clusters.append([obs])

    out: list[float] = []
    for cluster in clusters:
        leads = set(lead for _, lead in cluster)
        if len(leads) < min_leads:
            continue
        out.append(float(np.median([t for t, _ in cluster])))
    return out


def _source_summary(
    truth_r_ms: list[float],
    truth_p_ms: list[float],
    truth_t_ms: list[float],
    detected_ms: list[float],
) -> dict[str, Any]:
    m = _greedy_match(truth_r_ms, detected_ms, 80.0)
    extras = [float(x) for x in m["extras_ms"]]
    cv = _cv(detected_ms)
    return {
        "detected_n": len(detected_ms),
        "matched_n": int(m["matched_n"]),
        "missed_n": int(m["missed_n"]),
        "extra_n": int(m["extra_n"]),
        "extra_near_p_n": sum(_near_any(x, truth_p_ms, 80.0) for x in extras),
        "extra_near_t_n": sum(_near_any(x, truth_t_ms, 100.0) for x in extras),
        "rr_cv": cv,
        "regular": bool(cv is not None and cv <= REGULAR_CV_MAX),
    }


def run() -> dict[str, Any]:
    specs = [s for s in all_specs() if s.get("target") == "AVB3"]
    counts = {
        "n": 0,
        "analysis_error_n": 0,
        "truth_r_total": 0,
        "raw": {"matched": 0, "missed": 0, "extra": 0, "extra_p": 0, "extra_t": 0, "regular_n": 0},
        "selected_qrs_validated": {"matched": 0, "missed": 0, "extra": 0, "extra_p": 0, "extra_t": 0, "regular_n": 0},
        "crosslead_qrs_validated": {"matched": 0, "missed": 0, "extra": 0, "extra_p": 0, "extra_t": 0, "regular_n": 0},
    }
    raw_cv: list[float] = []
    selected_cv: list[float] = []
    cross_cv: list[float] = []
    errors: list[str] = []

    for pos, spec in enumerate(specs, 1):
        counts["n"] += 1
        try:
            rng = np.random.default_rng(_seed(str(spec["case_id"])))
            p_s, r_s, _ = _event_times(spec, rng)
            p_ms = [1000.0 * float(x) for x in p_s]
            r_ms = [1000.0 * float(x) for x in r_s]
            t_ms = [x + 280.0 for x in r_ms]
            counts["truth_r_total"] += len(r_ms)

            analysis = analyze_canonical_ecg(canonical(spec, make_signal(spec)))
            rhythm = analysis.get("rhythm") or {}
            fs = int(analysis.get("fs") or FS)
            raw_ms = [
                1000.0 * int(v) / fs
                for v in sorted(set(int(x) for x in (rhythm.get("r_peaks_samples") or [])))
            ]

            selected_lead = rhythm.get("lead")
            selected_item = (analysis.get("leads") or {}).get(selected_lead) or {}
            selected_fs = int(selected_item.get("fs") or fs)
            selected_validated_ms = [
                1000.0 * int(beat["r_sample"]) / selected_fs
                for beat in (selected_item.get("beats") or [])
                if beat.get("r_sample") is not None
            ]
            selected_validated_ms = sorted(set(selected_validated_ms))

            cross_validated_ms = _cluster_validated_r(analysis.get("leads") or {})

            for key, values, cvs in [
                ("raw", raw_ms, raw_cv),
                ("selected_qrs_validated", selected_validated_ms, selected_cv),
                ("crosslead_qrs_validated", cross_validated_ms, cross_cv),
            ]:
                row = _source_summary(r_ms, p_ms, t_ms, values)
                dst = counts[key]
                dst["matched"] += row["matched_n"]
                dst["missed"] += row["missed_n"]
                dst["extra"] += row["extra_n"]
                dst["extra_p"] += row["extra_near_p_n"]
                dst["extra_t"] += row["extra_near_t_n"]
                dst["regular_n"] += int(row["regular"])
                if row["rr_cv"] is not None:
                    cvs.append(float(row["rr_cv"]))
        except Exception as exc:
            counts["analysis_error_n"] += 1
            errors.append(f"{type(exc).__name__}:{exc}")

        if pos % 10 == 0:
            print(f"MEDCALC_AVB3_QRS_VALIDATED_R_AUDIT {pos}/{len(specs)}", flush=True)

    def dist(z: list[float]) -> dict[str, float | None]:
        if not z:
            return {"median": None, "p25": None, "p75": None}
        a = np.asarray(z, dtype=float)
        return {
            "median": round(float(np.median(a)), 6),
            "p25": round(float(np.percentile(a, 25)), 6),
            "p75": round(float(np.percentile(a, 75)), 6),
        }

    n = max(counts["n"], 1)
    for key, cvs in [
        ("raw", raw_cv),
        ("selected_qrs_validated", selected_cv),
        ("crosslead_qrs_validated", cross_cv),
    ]:
        counts[key]["regular_fraction"] = counts[key]["regular_n"] / n
        counts[key]["rr_cv"] = dist(cvs)

    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_QRS_VALIDATED_R_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "case_count": counts["n"],
        "analysis_error_n": counts["analysis_error_n"],
        "aggregate": counts,
        "case_level_outputs_emitted": False,
        "errors": errors,
        "interpretation": [
            "Engineering audit only. QRS-validated R means existing beats[].r_sample; it does not change published r_peaks_samples.",
            "No threshold, baseline, tolerance, FAST-GATE-100, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    leads = {
        "II": {"evaluable": True, "fs": 500, "beats": [{"r_sample": 100}, {"r_sample": 600}, {"r_sample": 1100}]},
        "V1": {"evaluable": True, "fs": 500, "beats": [{"r_sample": 102}, {"r_sample": 602}, {"r_sample": 1102}]},
        "aVF": {"evaluable": True, "fs": 500, "beats": [{"r_sample": 98}, {"r_sample": 598}, {"r_sample": 1098}]},
    }
    events = _cluster_validated_r(leads)
    assert len(events) == 3, events
    assert all(abs(a - b) <= 5.0 for a, b in zip(events, [200.0, 1200.0, 2200.0])), events
    print("MEDCALC_AVB3_QRS_VALIDATED_R_AUDIT_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    result = run()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
