from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_validation_harness import (
    LEADS,
    degrade_ecg_image,
    generate_ground_truth_ecg,
    render_ecg_paper,
    score_recovered_measurements,
)
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_measurement_failure_audit import aggregate_measurement_audits


CASES = [
    {
        "case_id": "6x2_clean_red",
        "layout": "6x2",
        "rhythm_strip": False,
        "grid_color": "red",
        "trace_width": 2,
        "degradation": {},
    },
    {
        "case_id": "6x2_rotated_red",
        "layout": "6x2",
        "rhythm_strip": False,
        "grid_color": "red",
        "trace_width": 2,
        "degradation": {"rotation_deg": 3.0},
    },
    {
        "case_id": "6x2_oblique_red",
        "layout": "6x2",
        "rhythm_strip": False,
        "grid_color": "red",
        "trace_width": 2,
        "degradation": {"perspective": 0.05},
    },
    {
        "case_id": "6x2_lowres_jpeg",
        "layout": "6x2",
        "rhythm_strip": False,
        "grid_color": "red",
        "trace_width": 2,
        "degradation": {"scale": 0.65, "jpeg_quality": 70},
    },
    {
        "case_id": "6x2_green_thick_blur",
        "layout": "6x2",
        "rhythm_strip": False,
        "grid_color": "green",
        "trace_width": 4,
        "degradation": {"blur_sigma": 0.8, "noise_sd": 2.0},
    },
    {
        "case_id": "3x4_strip_clean",
        "layout": "3x4",
        "rhythm_strip": True,
        "grid_color": "red",
        "trace_width": 2,
        "degradation": {},
    },
]


def _canonical_from_native(signals: dict[str, np.ndarray], fs: int) -> dict[str, Any]:
    leads = {}
    for lead in LEADS:
        x = np.asarray(signals[lead], dtype=float)
        leads[lead] = {
            "lead": lead,
            "signal_mv": [float(v) for v in x],
            "quality_mask": [2] * int(x.size),
            "fs": int(fs),
            "duration_s": float(x.size / fs),
            "source": "DEVELOPMENT_NATIVE_SYNTHETIC",
            "confidence": 1.0,
            "status": "MEASURABLE",
        }
    return {
        "version": "DEVELOPMENT_NATIVE_CANONICAL_V1",
        "source": "DEVELOPMENT_NATIVE_SYNTHETIC",
        "fs": int(fs),
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "confidence": 1.0,
        },
        "uncertainty": {
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "source": "NATIVE_DIGITAL_DEVELOPMENT_SIGNAL",
        },
        "leads": leads,
        "lead_order": list(LEADS),
    }


def benchmark_native(output: Path) -> None:
    rows = []
    for hr in (50.0, 75.0, 120.0):
        signals, truth = generate_ground_truth_ecg(heart_rate_bpm=hr)
        canonical = _canonical_from_native(signals, int(truth["fs"]))
        recovered = analyze_canonical_ecg(canonical)
        consensus = recovered.get("measurement_consensus") or {}
        legacy_remeasure, legacy_targets = _legacy_v1_would_remeasure(consensus)
        rows.append({
            "case_id": f"native_hr_{int(hr)}",
            "heart_rate_bpm_truth": hr,
            "measurement_failure_audit": recovered.get("measurement_failure_audit") or {},
            "legacy_v1_would_remeasure": legacy_remeasure,
            "legacy_v1_remeasure_targets": legacy_targets,
            "v2_remeasure_required": bool(consensus.get("remeasure_required")),
            "v2_remeasure_targets": list(consensus.get("remeasure_targets") or []),
            "v2_unmeasurable_targets": list(consensus.get("unmeasurable_targets") or []),
            "v2_uncertain_targets": list(consensus.get("uncertain_targets") or []),
            "v2_unusable_targets": list(consensus.get("unusable_targets") or []),
            "measurement_states": dict(consensus.get("measurement_states") or {}),
            "overall_measurement_quality": consensus.get("overall_measurement_quality"),
            "errors": score_recovered_measurements(truth, recovered),
        })

    n = len(rows)
    legacy_n = sum(bool(x["legacy_v1_would_remeasure"]) for x in rows)
    v2_n = sum(bool(x["v2_remeasure_required"]) for x in rows)
    result = {
        "benchmark_version": "MEDCALC_MEASUREMENT_CONSENSUS_V2_NATIVE_DEV_V1",
        "scope": "DEVELOPMENT_NATIVE_SYNTHETIC_MEASUREMENT_LAYER",
        "clinical_validation_claim_allowed": False,
        "case_n": n,
        "legacy_v1_counterfactual_remeasure_rate": round(legacy_n / n, 6),
        "v2_remeasure_rate": round(v2_n / n, 6),
        "measurement_failure_breakdown": aggregate_measurement_audits(
            [row["measurement_failure_audit"] for row in rows if row.get("measurement_failure_audit")]
        ),
        "cases": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


def generate_cases(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    signals, truth = generate_ground_truth_ecg()
    manifest = []

    for spec in CASES:
        base = render_ecg_paper(
            signals,
            fs=int(truth["fs"]),
            layout=str(spec["layout"]),
            rhythm_strip=bool(spec["rhythm_strip"]),
            grid_color=str(spec["grid_color"]),
            trace_width=int(spec["trace_width"]),
        )
        image = degrade_ecg_image(base, **dict(spec["degradation"]))
        path = output_dir / f'{spec["case_id"]}.png'
        image.save(path)
        manifest.append({
            **spec,
            "path": str(path),
            "ground_truth": truth,
        })

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"generated_n": len(manifest), "output_dir": str(output_dir)}, indent=2))


def _legacy_v1_would_remeasure(consensus: dict[str, Any]) -> tuple[bool, list[str]]:
    targets = []
    for name, item in (consensus.get("metrics") or {}).items():
        if str(item.get("status") or "") in {"DISCORDANT", "NO_CANONICAL_VALUE"}:
            targets.append(str(name))
    r = consensus.get("r_peak_verification") or {}
    try:
        score = float(r.get("aggregate_agreement"))
        n = int(r.get("evaluable_lead_n") or 0)
        if np.isfinite(score) and score < 0.65 and n >= 2:
            targets.append("r_peaks")
    except Exception:
        pass
    return bool(targets), sorted(set(targets))


def _analysis_metric_value(analysis: dict[str, Any], name: str) -> float | None:
    item = ((analysis.get("global") or {}).get(name) or {})
    try:
        value = item.get("value")
        return float(value) if value is not None else None
    except Exception:
        return None


def _qrs_boundary_profile(analysis: dict[str, Any]) -> dict[str, float | int | None]:
    """Summarize QRS onset/offset position relative to R across usable beats."""
    pre_r_ms: list[float] = []
    post_r_ms: list[float] = []
    for lead_result in (analysis.get("leads") or {}).values():
        fs = float((lead_result or {}).get("fs") or 500.0)
        if fs <= 0:
            continue
        for beat in (lead_result or {}).get("beats") or []:
            r = beat.get("r_sample")
            q_on = beat.get("qrs_onset_sample")
            q_off = beat.get("qrs_offset_sample")
            if r is None or q_on is None or q_off is None:
                continue
            try:
                r_i, on_i, off_i = int(r), int(q_on), int(q_off)
            except Exception:
                continue
            if not (on_i < r_i < off_i):
                continue
            pre_r_ms.append((r_i - on_i) * 1000.0 / fs)
            post_r_ms.append((off_i - r_i) * 1000.0 / fs)

    return {
        "beat_n": int(min(len(pre_r_ms), len(post_r_ms))),
        "pre_r_ms_median": (
            round(float(np.median(pre_r_ms)), 6) if pre_r_ms else None
        ),
        "pre_r_ms_mean": (
            round(float(np.mean(pre_r_ms)), 6) if pre_r_ms else None
        ),
        "post_r_ms_median": (
            round(float(np.median(post_r_ms)), 6) if post_r_ms else None
        ),
        "post_r_ms_mean": (
            round(float(np.mean(post_r_ms)), 6) if post_r_ms else None
        ),
    }


def _qrs_boundary_digitization_delta(
    native: dict[str, Any],
    reconstructed: dict[str, Any],
) -> dict[str, float | int | None]:
    """Decompose digitization-induced QRS widening into onset and offset components."""
    a = _qrs_boundary_profile(native)
    b = _qrs_boundary_profile(reconstructed)
    out: dict[str, float | int | None] = {
        "native_beat_n": a.get("beat_n"),
        "reconstructed_beat_n": b.get("beat_n"),
        "native_pre_r_ms_median": a.get("pre_r_ms_median"),
        "reconstructed_pre_r_ms_median": b.get("pre_r_ms_median"),
        "native_post_r_ms_median": a.get("post_r_ms_median"),
        "reconstructed_post_r_ms_median": b.get("post_r_ms_median"),
    }
    pre_a, pre_b = a.get("pre_r_ms_median"), b.get("pre_r_ms_median")
    post_a, post_b = a.get("post_r_ms_median"), b.get("post_r_ms_median")
    out["signed_onset_extension_ms"] = (
        round(float(pre_b) - float(pre_a), 6)
        if pre_a is not None and pre_b is not None else None
    )
    out["signed_offset_extension_ms"] = (
        round(float(post_b) - float(post_a), 6)
        if post_a is not None and post_b is not None else None
    )
    if (
        out["signed_onset_extension_ms"] is not None
        and out["signed_offset_extension_ms"] is not None
    ):
        out["signed_width_extension_ms"] = round(
            float(out["signed_onset_extension_ms"])
            + float(out["signed_offset_extension_ms"]),
            6,
        )
    else:
        out["signed_width_extension_ms"] = None
    return out


def _digitization_induced_delta(
    native: dict[str, Any],
    reconstructed: dict[str, Any],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in ("qrs_ms", "pr_ms", "qt_ms", "heart_rate_bpm"):
        a = _analysis_metric_value(native, name)
        b = _analysis_metric_value(reconstructed, name)
        out[f"ABS_DELTA_{name.upper()}"] = (
            abs(float(b) - float(a))
            if a is not None and b is not None
            else None
        )
    return out


def _summarize_error_key(
    rows: list[dict[str, Any]],
    container_key: str,
    metric_key: str,
) -> dict[str, Any]:
    values = [
        float((row.get(container_key) or {}).get(metric_key))
        for row in rows
        if (row.get(container_key) or {}).get(metric_key) is not None
    ]
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 6) if values else None,
        "median": round(float(np.median(values)), 6) if values else None,
        "p95": round(float(np.percentile(values, 95)), 6) if values else None,
    }


def _calibration_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    errors = []
    covered = []
    for row in rows:
        item = (row.get("metrics") or {}).get(metric) or {}
        value, truth = item.get("value"), item.get("truth")
        if value is not None and truth is not None:
            errors.append(float(value) - float(truth))
        inside = item.get("truth_inside_uncertainty_interval")
        if inside is not None:
            covered.append(bool(inside))
    return {
        "evaluable_n": len(errors),
        "signed_bias_ms_mean": round(float(np.mean(errors)), 6) if errors else None,
        "signed_bias_ms_median": round(float(np.median(errors)), 6) if errors else None,
        "mae_ms": round(float(np.mean(np.abs(errors))), 6) if errors else None,
        "coverage_rate": round(float(sum(covered)) / len(covered), 6) if covered else None,
    }


def _truth_covered(metric: dict[str, Any], truth_value: float | None) -> bool | None:
    if truth_value is None:
        return None
    interval = metric.get("uncertainty_interval")
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None
    try:
        lo, hi = float(interval[0]), float(interval[1])
        return bool(lo <= float(truth_value) <= hi)
    except Exception:
        return None


def score_cases(manifest_path: Path, meta_dir: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    native_cache: dict[float, dict[str, Any]] = {}

    for case in manifest:
        case_id = str(case["case_id"])
        meta_path = meta_dir / f"{case_id}.json"
        if not meta_path.exists():
            rows.append({
                "case_id": case_id,
                "worker_success": False,
                "error": "META_NOT_FOUND",
            })
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        signal_meta = meta.get("signal") or {}
        recovered = signal_meta.get("digital_measurements_v2") or {}
        consensus = recovered.get("measurement_consensus") or {}
        truth = case.get("ground_truth") or {}
        legacy_remeasure, legacy_targets = _legacy_v1_would_remeasure(consensus)
        errors = score_recovered_measurements(truth, recovered)

        hr = float(truth.get("heart_rate_bpm") or 75.0)
        if hr not in native_cache:
            native_signals, native_truth = generate_ground_truth_ecg(
                heart_rate_bpm=hr
            )
            native_cache[hr] = analyze_canonical_ecg(
                _canonical_from_native(native_signals, int(native_truth["fs"]))
            )
        native_recovered = native_cache[hr]
        native_errors = score_recovered_measurements(truth, native_recovered)
        digitization_delta = _digitization_induced_delta(
            native_recovered,
            recovered,
        )
        qrs_boundary_digitization = _qrs_boundary_digitization_delta(
            native_recovered,
            recovered,
        )
        measurement_failure_audit = recovered.get("measurement_failure_audit") or {}

        # Development-only provenance: expose which digital fiducial sources
        # produced the interval candidates. This does not alter measurement
        # logic or clinical gates; it localizes residual delineation bias.
        fiducial_provenance: dict[str, dict[str, int]] = {
            "qrs": {},
            "t": {},
            "p": {},
        }
        fiducial_measurements: dict[str, dict[str, list[float]]] = {
            "qrs": {},
            "t": {},
            "p": {},
        }
        source_metric = {
            "qrs": "qrs_ms",
            "t": "qt_ms",
            "p": "pr_ms",
        }
        for lead_result in (recovered.get("leads") or {}).values():
            for beat in (lead_result or {}).get("beats") or []:
                for group, key in (
                    ("qrs", "fiducial_source"),
                    ("t", "t_fiducial_source"),
                    ("p", "p_fiducial_source"),
                ):
                    source = str(beat.get(key) or "UNKNOWN")
                    fiducial_provenance[group][source] = (
                        fiducial_provenance[group].get(source, 0) + 1
                    )
                    value = beat.get(source_metric[group])
                    if value is not None and np.isfinite(value):
                        fiducial_measurements[group].setdefault(source, []).append(
                            float(value)
                        )

        fiducial_source_measurement_audit: dict[str, dict[str, dict[str, float | int]]] = {
            "qrs": {},
            "t": {},
            "p": {},
        }
        for group, by_source in fiducial_measurements.items():
            for source, values in by_source.items():
                arr = np.asarray(values, dtype=float)
                fiducial_source_measurement_audit[group][source] = {
                    "n": int(arr.size),
                    "mean_ms": round(float(np.mean(arr)), 6),
                    "median_ms": round(float(np.median(arr)), 6),
                    "p10_ms": round(float(np.percentile(arr, 10)), 6),
                    "p90_ms": round(float(np.percentile(arr, 90)), 6),
                }

        t_consensus_audit = {
            str(lead_name): dict((lead_result or {}).get("t_consensus_audit") or {})
            for lead_name, lead_result in (recovered.get("leads") or {}).items()
            if (lead_result or {}).get("t_consensus_audit")
        }

        coverage = {}
        for metric_name, truth_key in (
            ("qrs_ms", "qrs_ms"),
            ("pr_ms", "pr_ms"),
            ("qt_ms", "qt_ms"),
        ):
            metric = (consensus.get("metrics") or {}).get(metric_name) or {}
            coverage[metric_name] = {
                "state": metric.get("measurement_state"),
                "value": metric.get("canonical_value"),
                "uncertainty_ms": metric.get("uncertainty_ms"),
                "uncertainty_interval": metric.get("uncertainty_interval"),
                "truth": truth.get(truth_key),
                "truth_inside_uncertainty_interval": _truth_covered(
                    metric, truth.get(truth_key)
                ),
            }

        rows.append({
            "case_id": case_id,
            "worker_success": recovered != {},
            "layout": case.get("layout"),
            "degradation": case.get("degradation"),
            "legacy_v1_would_remeasure": legacy_remeasure,
            "legacy_v1_remeasure_targets": legacy_targets,
            "v2_remeasure_required": bool(consensus.get("remeasure_required")),
            "v2_remeasure_targets": list(consensus.get("remeasure_targets") or []),
            "v2_unmeasurable_targets": list(consensus.get("unmeasurable_targets") or []),
            "v2_uncertain_targets": list(consensus.get("uncertain_targets") or []),
            "v2_unusable_targets": list(consensus.get("unusable_targets") or []),
            "measurement_states": dict(consensus.get("measurement_states") or {}),
            "overall_measurement_quality": consensus.get("overall_measurement_quality"),
            "physical_time_uncertainty_ms": consensus.get(
                "explicit_digitization_time_uncertainty_ms"
            ),
            "measurement_failure_audit": measurement_failure_audit,
            "fiducial_provenance": fiducial_provenance,
            "fiducial_source_measurement_audit": fiducial_source_measurement_audit,
            "t_consensus_audit": t_consensus_audit,
            "native_measurement_errors": native_errors,
            "digitization_induced_measurement_delta": digitization_delta,
            "qrs_boundary_digitization_audit": qrs_boundary_digitization,
            "errors": errors,
            "metrics": coverage,
            "measurement_error": signal_meta.get("signal_primary_measurement_error"),
            "worker_status": meta.get("status"),
            "worker_reason": meta.get("reason"),
            "pipeline_failure_stage": (
                None
                if recovered
                else (
                    "WORKER_FAILED"
                    if str(meta.get("status") or "") == "FAIL"
                    else "MEASUREMENT_NOT_PRODUCED_AFTER_DIGITIZATION"
                )
            ),
            "trusted_preflight_layout": meta.get("trusted_preflight_layout"),
            "layout_preflight": meta.get("layout_preflight"),
            "layout_router": meta.get("layout_router"),
            "signal_layout_name": signal_meta.get("layout_name"),
            "signal_primary_measurement_error": signal_meta.get(
                "signal_primary_measurement_error"
            ),
            "development_row_counterfactual": signal_meta.get(
                "development_row_counterfactual"
            ),
        })

    successful = [r for r in rows if r.get("worker_success")]
    n = len(rows)
    success_n = len(successful)
    legacy_n = sum(bool(r.get("legacy_v1_would_remeasure")) for r in successful)
    v2_remeasure_n = sum(bool(r.get("v2_remeasure_required")) for r in successful)
    v2_unusable_n = sum(bool(r.get("v2_unusable_targets")) for r in successful)
    v2_uncertain_n = sum(bool(r.get("v2_uncertain_targets")) for r in successful)

    def rate(x: int, denom: int) -> float | None:
        return round(float(x) / float(denom), 6) if denom else None

    mae_summary = {}
    for key in ("MAE_QRS_MS", "MAE_PR_MS", "MAE_QT_MS", "MAE_HEART_RATE_BPM"):
        values = [
            float((r.get("errors") or {}).get(key))
            for r in successful
            if (r.get("errors") or {}).get(key) is not None
        ]
        mae_summary[key] = {
            "n": len(values),
            "mean": round(float(np.mean(values)), 6) if values else None,
            "median": round(float(np.median(values)), 6) if values else None,
            "p95": round(float(np.percentile(values, 95)), 6) if values else None,
        }

    native_measurement_error_summary = {
        key: _summarize_error_key(successful, "native_measurement_errors", key)
        for key in ("MAE_QRS_MS", "MAE_PR_MS", "MAE_QT_MS", "MAE_HEART_RATE_BPM")
    }
    digitization_induced_delta_summary = {
        key: _summarize_error_key(
            successful,
            "digitization_induced_measurement_delta",
            key,
        )
        for key in (
            "ABS_DELTA_QRS_MS",
            "ABS_DELTA_PR_MS",
            "ABS_DELTA_QT_MS",
            "ABS_DELTA_HEART_RATE_BPM",
        )
    }
    failure_audits = [
        row.get("measurement_failure_audit") or {}
        for row in successful
        if row.get("measurement_failure_audit")
    ]

    interval_coverage = {}
    for metric in ("qrs_ms", "pr_ms", "qt_ms"):
        values = [
            (r.get("metrics") or {}).get(metric, {}).get(
                "truth_inside_uncertainty_interval"
            )
            for r in successful
        ]
        evaluable = [bool(v) for v in values if v is not None]
        interval_coverage[metric] = {
            "evaluable_n": len(evaluable),
            "coverage_rate": (
                round(float(sum(evaluable)) / len(evaluable), 6)
                if evaluable else None
            ),
        }

    summary = {
        "benchmark_version": "MEDCALC_MEASUREMENT_CONSENSUS_V2_END_TO_END_DEV_V1",
        "scope": "DEVELOPMENT_SYNTHETIC_PAPER_DIGITIZATION_ONLY",
        "clinical_validation_claim_allowed": False,
        "case_n": n,
        "worker_success_n": success_n,
        "worker_success_rate": rate(success_n, n),
        "legacy_v1_counterfactual_remeasure_n": legacy_n,
        "legacy_v1_counterfactual_remeasure_rate": rate(legacy_n, success_n),
        "v2_remeasure_n": v2_remeasure_n,
        "v2_remeasure_rate": rate(v2_remeasure_n, success_n),
        "v2_any_unusable_n": v2_unusable_n,
        "v2_any_unusable_rate": rate(v2_unusable_n, success_n),
        "v2_any_uncertain_n": v2_uncertain_n,
        "v2_any_uncertain_rate": rate(v2_uncertain_n, success_n),
        "mae_summary": mae_summary,
        "error_decomposition": {
            "native_measurement_or_delineation_error": native_measurement_error_summary,
            "digitization_induced_measurement_delta": digitization_induced_delta_summary,
            "total_reconstructed_measurement_error": mae_summary,
        },
        "measurement_failure_breakdown": (
            aggregate_measurement_audits(failure_audits)
            if failure_audits
            else None
        ),
        "uncertainty_interval_truth_coverage": interval_coverage,
        "measurement_calibration_audit": {
            metric: _calibration_summary(successful, metric)
            for metric in ("qrs_ms", "pr_ms", "qt_ms")
        },
        "qrs_boundary_digitization_summary": {
            key: _summarize_error_key(
                successful,
                "qrs_boundary_digitization_audit",
                key,
            )
            for key in (
                "signed_onset_extension_ms",
                "signed_offset_extension_ms",
                "signed_width_extension_ms",
            )
        },
        "cases": rows,
        "interpretation": (
            "This suite tests reconstruction/measurement behavior on deterministic "
            "paper-rendered development signals with known numeric truth. It does "
            "not establish diagnostic sensitivity or clinical external validity."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--output-dir", type=Path, required=True)

    native = sub.add_parser("native")
    native.add_argument("--output", type=Path, required=True)

    score = sub.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--meta-dir", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "generate":
        generate_cases(args.output_dir)
    elif args.cmd == "native":
        benchmark_native(args.output)
    else:
        score_cases(args.manifest, args.meta_dir, args.output)


if __name__ == "__main__":
    main()
