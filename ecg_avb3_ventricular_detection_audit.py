from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

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

VERSION = "MEDCALC_AVB3_VENTRICULAR_DETECTION_AUDIT_V1"
MATCH_WINDOWS_MS = (40.0, 80.0, 120.0)
REGULAR_CV_MAX = 0.12


def _cv(times_ms: list[float]) -> float | None:
    if len(times_ms) < 3:
        return None
    rr = np.diff(np.asarray(times_ms, dtype=float))
    rr = rr[np.isfinite(rr) & (rr > 0)]
    if rr.size < 2 or float(np.mean(rr)) <= 0:
        return None
    return float(np.std(rr, ddof=1) / np.mean(rr))


def _greedy_match(
    truth_ms: list[float],
    detected_ms: list[float],
    tolerance_ms: float,
) -> dict[str, Any]:
    truth = list(enumerate(float(x) for x in truth_ms))
    det = list(enumerate(float(x) for x in detected_ms))
    candidates: list[tuple[float, int, int]] = []
    for ti, tv in truth:
        for di, dv in det:
            err = abs(dv - tv)
            if err <= tolerance_ms:
                candidates.append((err, ti, di))
    candidates.sort()

    used_t: set[int] = set()
    used_d: set[int] = set()
    errors: list[float] = []
    for err, ti, di in candidates:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        errors.append(float(err))

    extras = [dv for di, dv in det if di not in used_d]
    misses = [tv for ti, tv in truth if ti not in used_t]
    return {
        "matched_n": len(errors),
        "missed_n": len(misses),
        "extra_n": len(extras),
        "median_abs_error_ms": (
            round(float(np.median(errors)), 6) if errors else None
        ),
        "extras_ms": extras,
        "misses_ms": misses,
    }


def _near_any(value_ms: float, centers_ms: list[float], tolerance_ms: float) -> bool:
    return any(abs(float(value_ms) - float(c)) <= tolerance_ms for c in centers_ms)


def _case_audit(spec: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(_seed(str(spec["case_id"])))
    true_p_s, true_r_s, _ = _event_times(spec, rng)
    true_p_ms = [1000.0 * float(x) for x in true_p_s]
    true_r_ms = [1000.0 * float(x) for x in true_r_s]
    true_t_ms = [x + 280.0 for x in true_r_ms]

    analysis = analyze_canonical_ecg(canonical(spec, make_signal(spec)))
    rhythm = analysis.get("rhythm") or {}
    fs = int(analysis.get("fs") or FS)
    detected_samples = sorted(
        set(int(v) for v in (rhythm.get("r_peaks_samples") or []))
    )
    detected_ms = [1000.0 * v / max(fs, 1) for v in detected_samples]

    matches = {
        str(int(window)): _greedy_match(true_r_ms, detected_ms, window)
        for window in MATCH_WINDOWS_MS
    }
    primary = matches["80"]
    extras = [float(x) for x in primary["extras_ms"]]
    extra_near_p = sum(_near_any(x, true_p_ms, 80.0) for x in extras)
    extra_near_t = sum(_near_any(x, true_t_ms, 100.0) for x in extras)
    extra_other = len(extras) - sum(
        _near_any(x, true_p_ms, 80.0) or _near_any(x, true_t_ms, 100.0)
        for x in extras
    )

    per_lead: dict[str, Any] = {}
    for lead, item in (analysis.get("leads") or {}).items():
        if not item.get("evaluable"):
            continue
        lead_fs = int(item.get("fs") or fs)
        lead_samples = sorted(
            set(int(v) for v in (item.get("r_peaks_samples") or []))
        )
        lead_ms = [1000.0 * v / max(lead_fs, 1) for v in lead_samples]
        m = _greedy_match(true_r_ms, lead_ms, 80.0)
        per_lead[str(lead)] = {
            "detected_r_n": len(lead_ms),
            "rr_cv": _cv(lead_ms),
            "matched_n_80ms": m["matched_n"],
            "missed_n_80ms": m["missed_n"],
            "extra_n_80ms": m["extra_n"],
        }

    return {
        "true_r_n": len(true_r_ms),
        "detected_r_n": len(detected_ms),
        "true_rr_cv": _cv(true_r_ms),
        "detected_rr_cv": _cv(detected_ms),
        "detected_regular": bool(
            _cv(detected_ms) is not None and float(_cv(detected_ms)) <= REGULAR_CV_MAX
        ),
        "matches": matches,
        "extra_near_true_p_n_80ms": int(extra_near_p),
        "extra_near_true_t_n_100ms": int(extra_near_t),
        "extra_other_n": int(extra_other),
        "per_lead": per_lead,
    }


def _blank() -> dict[str, Any]:
    return {
        "n": 0,
        "analysis_error_n": 0,
        "true_r_total": 0,
        "detected_r_total": 0,
        "detected_regular_n": 0,
        "cases_with_extra_r_n": 0,
        "cases_with_missed_r_n": 0,
        "extra_r_total_80ms": 0,
        "missed_r_total_80ms": 0,
        "matched_r_total_80ms": 0,
        "extra_near_true_p_total_80ms": 0,
        "extra_near_true_t_total_100ms": 0,
        "extra_other_total": 0,
        "detected_rr_cv_values": [],
        "true_rr_cv_values": [],
        "per_lead": {},
    }


def run() -> dict[str, Any]:
    specs = [s for s in all_specs() if s.get("target") == "AVB3"]
    out = _blank()
    errors: list[str] = []

    for pos, spec in enumerate(specs, 1):
        out["n"] += 1
        try:
            row = _case_audit(spec)
            out["true_r_total"] += int(row["true_r_n"])
            out["detected_r_total"] += int(row["detected_r_n"])
            out["detected_regular_n"] += int(bool(row["detected_regular"]))
            m80 = row["matches"]["80"]
            out["matched_r_total_80ms"] += int(m80["matched_n"])
            out["missed_r_total_80ms"] += int(m80["missed_n"])
            out["extra_r_total_80ms"] += int(m80["extra_n"])
            out["cases_with_extra_r_n"] += int(int(m80["extra_n"]) > 0)
            out["cases_with_missed_r_n"] += int(int(m80["missed_n"]) > 0)
            out["extra_near_true_p_total_80ms"] += int(row["extra_near_true_p_n_80ms"])
            out["extra_near_true_t_total_100ms"] += int(row["extra_near_true_t_n_100ms"])
            out["extra_other_total"] += int(row["extra_other_n"])
            if row["detected_rr_cv"] is not None:
                out["detected_rr_cv_values"].append(float(row["detected_rr_cv"]))
            if row["true_rr_cv"] is not None:
                out["true_rr_cv_values"].append(float(row["true_rr_cv"]))

            for lead, lrow in row["per_lead"].items():
                dst = out["per_lead"].setdefault(lead, {
                    "evaluable_n": 0,
                    "matched_r_total_80ms": 0,
                    "missed_r_total_80ms": 0,
                    "extra_r_total_80ms": 0,
                    "regular_n": 0,
                    "rr_cv_values": [],
                })
                dst["evaluable_n"] += 1
                dst["matched_r_total_80ms"] += int(lrow["matched_n_80ms"])
                dst["missed_r_total_80ms"] += int(lrow["missed_n_80ms"])
                dst["extra_r_total_80ms"] += int(lrow["extra_n_80ms"])
                if lrow["rr_cv"] is not None:
                    dst["rr_cv_values"].append(float(lrow["rr_cv"]))
                    dst["regular_n"] += int(float(lrow["rr_cv"]) <= REGULAR_CV_MAX)
        except Exception as exc:
            out["analysis_error_n"] += 1
            errors.append(f"{type(exc).__name__}:{exc}")

        if pos % 10 == 0:
            print(f"MEDCALC_AVB3_VENTRICULAR_DETECTION_AUDIT {pos}/{len(specs)}", flush=True)

    def summarize(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p25": None, "p75": None}
        z = np.asarray(values, dtype=float)
        return {
            "median": round(float(np.median(z)), 6),
            "p25": round(float(np.percentile(z, 25)), 6),
            "p75": round(float(np.percentile(z, 75)), 6),
        }

    per_lead = {}
    for lead, row in sorted(out["per_lead"].items()):
        n = max(int(row["evaluable_n"]), 1)
        per_lead[lead] = {
            **{k: v for k, v in row.items() if k != "rr_cv_values"},
            "regular_fraction": row["regular_n"] / n,
            "rr_cv": summarize(row["rr_cv_values"]),
        }

    n = max(int(out["n"]), 1)
    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_AVB3_R_DETECTION_AUDIT",
        "clinical_output_changed": False,
        "diagnostic_claim_allowed": False,
        "external_validation_claim_allowed": False,
        "case_count": out["n"],
        "analysis_error_n": out["analysis_error_n"],
        "aggregate": {
            "true_r_total": out["true_r_total"],
            "detected_r_total": out["detected_r_total"],
            "detected_regular_n": out["detected_regular_n"],
            "detected_regular_fraction": out["detected_regular_n"] / n,
            "cases_with_extra_r_n": out["cases_with_extra_r_n"],
            "cases_with_missed_r_n": out["cases_with_missed_r_n"],
            "matched_r_total_80ms": out["matched_r_total_80ms"],
            "missed_r_total_80ms": out["missed_r_total_80ms"],
            "extra_r_total_80ms": out["extra_r_total_80ms"],
            "extra_near_true_p_total_80ms": out["extra_near_true_p_total_80ms"],
            "extra_near_true_t_total_100ms": out["extra_near_true_t_total_100ms"],
            "extra_other_total": out["extra_other_total"],
            "true_rr_cv": summarize(out["true_rr_cv_values"]),
            "detected_rr_cv": summarize(out["detected_rr_cv_values"]),
            "per_lead": per_lead,
        },
        "case_level_outputs_emitted": False,
        "errors": errors,
        "interpretation": [
            "Engineering audit only; synthetic generator truth is used only to localize R-detection failure modes.",
            "No diagnostic threshold, baseline, tolerance, FAST-GATE-100 panel, fold 9/10, or external/final validation set is modified or consumed.",
        ],
    }


def selftest() -> None:
    m = _greedy_match([100.0, 500.0, 900.0], [102.0, 498.0, 700.0, 903.0], 10.0)
    assert m["matched_n"] == 3, m
    assert m["missed_n"] == 0, m
    assert m["extra_n"] == 1, m
    assert _near_any(700.0, [690.0], 20.0)
    assert not _near_any(700.0, [600.0], 20.0)
    print("MEDCALC_AVB3_VENTRICULAR_DETECTION_AUDIT_SELFTEST_PASS")


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
