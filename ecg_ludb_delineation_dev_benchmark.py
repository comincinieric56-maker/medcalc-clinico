from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_signal_measurements import analyze_canonical_ecg


LUDB_VERSION = "1.0.1"
LUDB_PN_DIR = f"ludb/{LUDB_VERSION}/data"
LUDB_DOI = "10.13026/eegm-h675"
LUDB_LICENSE = "Open Data Commons Attribution License v1.0"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ANNOTATOR_BY_LEAD = {
    "I": "i",
    "II": "ii",
    "III": "iii",
    "aVR": "avr",
    "aVL": "avl",
    "aVF": "avf",
    "V1": "v1",
    "V2": "v2",
    "V3": "v3",
    "V4": "v4",
    "V5": "v5",
    "V6": "v6",
}
QRS_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F",
    "e", "j", "n", "E", "/", "f", "Q",
}
DEFAULT_RECORDS = [str(x) for x in range(1, 201, 20)]  # fixed, outcome-independent index sample


def _norm_lead(name: str) -> str:
    key = str(name).strip().replace(" ", "").lower()
    mapping = {
        "i": "I",
        "ii": "II",
        "iii": "III",
        "avr": "aVR",
        "avl": "aVL",
        "avf": "aVF",
        "v1": "V1",
        "v2": "V2",
        "v3": "V3",
        "v4": "V4",
        "v5": "V5",
        "v6": "V6",
    }
    return mapping.get(key, str(name))


def _canonical_from_record(record_id: str) -> tuple[dict[str, Any], float]:
    rec = wfdb.rdrecord(str(record_id), pn_dir=LUDB_PN_DIR, physical=True)
    fs = float(rec.fs)
    if not math.isfinite(fs) or abs(fs - 500.0) > 1e-6:
        raise RuntimeError(f"LUDB record {record_id}: expected 500 Hz, got {fs}")

    signals = np.asarray(rec.p_signal, dtype=float)
    names = [_norm_lead(x) for x in rec.sig_name]
    index = {lead: i for i, lead in enumerate(names)}

    leads: dict[str, Any] = {}
    for lead in LEADS:
        if lead not in index:
            raise RuntimeError(f"LUDB record {record_id}: missing lead {lead}")
        x = np.asarray(signals[:, index[lead]], dtype=float)
        finite = np.isfinite(x)
        leads[lead] = {
            "lead": lead,
            "signal_mv": [float(v) if math.isfinite(float(v)) else None for v in x],
            "quality_mask": [2 if bool(v) else 0 for v in finite],
            "fs": int(round(fs)),
            "duration_s": float(len(x) / fs),
            "source": "LUDB_DEVELOPMENT_REFERENCE",
            "confidence": 1.0,
            "status": "MEASURABLE" if int(finite.sum()) >= int(1.5 * fs) else "UNMEASURABLE",
        }

    return {
        "version": "MEDCALC_LUDB_NATIVE_CANONICAL_V1",
        "source": "LUDB_DEVELOPMENT_REFERENCE",
        "fs": int(round(fs)),
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
            "source": "NATIVE_DIGITAL_REFERENCE",
        },
        "leads": leads,
        "lead_order": list(LEADS),
    }, fs


def _parse_events(record_id: str, lead: str) -> list[dict[str, Any]]:
    ann = wfdb.rdann(
        str(record_id),
        extension=ANNOTATOR_BY_LEAD[lead],
        pn_dir=LUDB_PN_DIR,
    )
    events: list[dict[str, Any]] = []
    onset: int | None = None
    peak: int | None = None
    kind: str | None = None
    peak_symbol: str | None = None

    for sample, symbol in zip(ann.sample, ann.symbol):
        sample_i = int(sample)
        symbol_s = str(symbol)
        if symbol_s == "(":
            onset = sample_i
            peak = None
            kind = None
            peak_symbol = None
            continue

        if symbol_s == ")":
            if onset is not None and peak is not None and kind is not None and sample_i > onset:
                events.append({
                    "kind": kind,
                    "onset": int(onset),
                    "peak": int(peak),
                    "offset": int(sample_i),
                    "peak_symbol": peak_symbol,
                })
            onset = None
            peak = None
            kind = None
            peak_symbol = None
            continue

        if onset is None:
            continue
        if symbol_s == "p":
            kind, peak, peak_symbol = "P", sample_i, symbol_s
        elif symbol_s == "t":
            kind, peak, peak_symbol = "T", sample_i, symbol_s
        elif symbol_s in QRS_SYMBOLS:
            kind, peak, peak_symbol = "QRS", sample_i, symbol_s

    return events


def _nearest_recovered_beat(
    beats: list[dict[str, Any]],
    target_peak: int,
    fs: float,
    used: set[int],
) -> tuple[int, dict[str, Any]] | None:
    candidates: list[tuple[int, int, dict[str, Any]]] = []
    max_delta = int(round(0.12 * fs))
    for idx, beat in enumerate(beats):
        if idx in used:
            continue
        r = beat.get("r_sample")
        if r is None:
            continue
        delta = abs(int(r) - int(target_peak))
        if delta <= max_delta:
            candidates.append((delta, idx, beat))
    if not candidates:
        return None
    _, idx, beat = min(candidates, key=lambda row: row[0])
    return idx, beat


def _previous_event(events: list[dict[str, Any]], kind: str, qrs_onset: int, fs: float) -> dict[str, Any] | None:
    candidates = [
        e for e in events
        if e["kind"] == kind
        and e["offset"] < qrs_onset
        and 0 < qrs_onset - e["onset"] <= int(round(0.40 * fs))
    ]
    return max(candidates, key=lambda e: e["offset"]) if candidates else None


def _next_event(events: list[dict[str, Any]], kind: str, qrs_offset: int, fs: float) -> dict[str, Any] | None:
    candidates = [
        e for e in events
        if e["kind"] == kind
        and e["onset"] > qrs_offset
        and 0 < e["offset"] - qrs_offset <= int(round(0.65 * fs))
    ]
    return min(candidates, key=lambda e: e["onset"]) if candidates else None


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "signed_bias_ms_mean": None,
            "signed_bias_ms_median": None,
            "mae_ms": None,
            "median_abs_error_ms": None,
            "p95_abs_error_ms": None,
            "within_20ms_rate": None,
            "within_40ms_rate": None,
        }
    abs_arr = np.abs(arr)
    return {
        "n": int(arr.size),
        "signed_bias_ms_mean": round(float(np.mean(arr)), 6),
        "signed_bias_ms_median": round(float(np.median(arr)), 6),
        "mae_ms": round(float(np.mean(abs_arr)), 6),
        "median_abs_error_ms": round(float(np.median(abs_arr)), 6),
        "p95_abs_error_ms": round(float(np.percentile(abs_arr, 95)), 6),
        "within_20ms_rate": round(float(np.mean(abs_arr <= 20.0)), 6),
        "within_40ms_rate": round(float(np.mean(abs_arr <= 40.0)), 6),
    }


def benchmark(records: list[str]) -> dict[str, Any]:
    errors: dict[str, list[float]] = {
        "qrs_onset": [],
        "qrs_offset": [],
        "qrs_duration": [],
        "p_onset": [],
        "p_offset": [],
        "pr_interval": [],
        "t_offset": [],
        "qt_interval": [],
        "t_offset_dwt_counterfactual": [],
        "qt_interval_dwt_counterfactual": [],
        "t_offset_candidate_counterfactual": [],
        "qt_interval_candidate_counterfactual": [],
        "t_offset_later_of_dwt_candidate_counterfactual": [],
        "qt_interval_later_of_dwt_candidate_counterfactual": [],
        "t_offset_candidate_conf_ge_048_else_dwt": [],
        "qt_interval_candidate_conf_ge_048_else_dwt": [],
        "t_offset_candidate_conf_ge_052_else_dwt": [],
        "qt_interval_candidate_conf_ge_052_else_dwt": [],
        "t_offset_candidate_conf_ge_058_else_dwt": [],
        "qt_interval_candidate_conf_ge_058_else_dwt": [],
        "t_offset_later_if_candidate_conf_ge_048_else_dwt": [],
        "qt_interval_later_if_candidate_conf_ge_048_else_dwt": [],
        "t_offset_later_if_candidate_conf_ge_052_else_dwt": [],
        "qt_interval_later_if_candidate_conf_ge_052_else_dwt": [],
        "t_offset_later_if_candidate_conf_ge_058_else_dwt": [],
        "qt_interval_later_if_candidate_conf_ge_058_else_dwt": [],
    }
    qrs_source_errors: dict[str, list[float]] = {}
    t_source_errors: dict[str, list[float]] = {}
    t_error_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    lead_n = 0
    qrs_reference_n = 0
    qrs_matched_n = 0
    p_reference_n = 0
    p_matched_n = 0
    t_reference_n = 0
    t_matched_n = 0

    for record_id in records:
        canonical, fs = _canonical_from_record(record_id)
        recovered = analyze_canonical_ecg(canonical)
        row_counts = {
            "record_id": str(record_id),
            "lead_n": 0,
            "qrs_reference_n": 0,
            "qrs_matched_n": 0,
            "p_reference_n": 0,
            "p_matched_n": 0,
            "t_reference_n": 0,
            "t_matched_n": 0,
        }

        for lead in LEADS:
            lead_result = (recovered.get("leads") or {}).get(lead) or {}
            if not lead_result.get("evaluable"):
                continue
            row_counts["lead_n"] += 1
            lead_n += 1
            beats = list(lead_result.get("beats") or [])
            events = _parse_events(record_id, lead)
            qrs_events = [e for e in events if e["kind"] == "QRS"]
            row_counts["qrs_reference_n"] += len(qrs_events)
            qrs_reference_n += len(qrs_events)
            used: set[int] = set()

            for qev in qrs_events:
                matched = _nearest_recovered_beat(beats, qev["peak"], fs, used)
                if matched is None:
                    continue
                beat_idx, beat = matched
                used.add(beat_idx)
                row_counts["qrs_matched_n"] += 1
                qrs_matched_n += 1

                q_on = beat.get("qrs_onset_sample")
                q_off = beat.get("qrs_offset_sample")
                if q_on is not None:
                    errors["qrs_onset"].append((int(q_on) - qev["onset"]) * 1000.0 / fs)
                if q_off is not None:
                    errors["qrs_offset"].append((int(q_off) - qev["offset"]) * 1000.0 / fs)
                if q_on is not None and q_off is not None:
                    got = (int(q_off) - int(q_on)) * 1000.0 / fs
                    ref = (qev["offset"] - qev["onset"]) * 1000.0 / fs
                    qrs_error = float(got - ref)
                    errors["qrs_duration"].append(qrs_error)
                    qrs_source = str(beat.get("fiducial_source") or "UNKNOWN")
                    qrs_source_errors.setdefault(qrs_source, []).append(qrs_error)

                pev = _previous_event(events, "P", qev["onset"], fs)
                if pev is not None:
                    row_counts["p_reference_n"] += 1
                    p_reference_n += 1
                    p_on = beat.get("p_onset_sample")
                    p_off = beat.get("p_offset_sample")
                    if p_on is not None:
                        errors["p_onset"].append((int(p_on) - pev["onset"]) * 1000.0 / fs)
                    if p_off is not None:
                        errors["p_offset"].append((int(p_off) - pev["offset"]) * 1000.0 / fs)
                    if p_on is not None and q_on is not None:
                        row_counts["p_matched_n"] += 1
                        p_matched_n += 1
                        got_pr = (int(q_on) - int(p_on)) * 1000.0 / fs
                        ref_pr = (qev["onset"] - pev["onset"]) * 1000.0 / fs
                        errors["pr_interval"].append(got_pr - ref_pr)

                tev = _next_event(events, "T", qev["offset"], fs)
                if tev is not None:
                    row_counts["t_reference_n"] += 1
                    t_reference_n += 1
                    t_off = beat.get("t_offset_sample")
                    if t_off is not None:
                        row_counts["t_matched_n"] += 1
                        t_matched_n += 1
                        t_error = (int(t_off) - tev["offset"]) * 1000.0 / fs
                        errors["t_offset"].append(t_error)
                        t_source = str(beat.get("t_fiducial_source") or "UNKNOWN")
                        t_source_errors.setdefault(t_source, []).append(float(t_error))
                        t_error_rows.append({
                            "record_id": str(record_id),
                            "lead": lead,
                            "r_sample": int(beat.get("r_sample") or qev["peak"]),
                            "source": t_source,
                            "error_ms": round(float(t_error), 6),
                        })
                        if q_on is not None:
                            got_qt = (int(t_off) - int(q_on)) * 1000.0 / fs
                            ref_qt = (tev["offset"] - qev["onset"]) * 1000.0 / fs
                            errors["qt_interval"].append(got_qt - ref_qt)

                    dwt_t_off = beat.get("t_dwt_offset_sample")
                    if dwt_t_off is not None:
                        dwt_error = (int(dwt_t_off) - tev["offset"]) * 1000.0 / fs
                        errors["t_offset_dwt_counterfactual"].append(dwt_error)
                        if q_on is not None:
                            dwt_qt = (int(dwt_t_off) - int(q_on)) * 1000.0 / fs
                            ref_qt = (tev["offset"] - qev["onset"]) * 1000.0 / fs
                            errors["qt_interval_dwt_counterfactual"].append(dwt_qt - ref_qt)

                    candidate_t_off = beat.get("t_candidate_offset_sample")
                    candidate_conf = float(beat.get("t_candidate_confidence") or 0.0)
                    if candidate_t_off is not None:
                        candidate_error = (int(candidate_t_off) - tev["offset"]) * 1000.0 / fs
                        errors["t_offset_candidate_counterfactual"].append(candidate_error)
                        if q_on is not None:
                            candidate_qt = (int(candidate_t_off) - int(q_on)) * 1000.0 / fs
                            ref_qt = (tev["offset"] - qev["onset"]) * 1000.0 / fs
                            errors["qt_interval_candidate_counterfactual"].append(
                                candidate_qt - ref_qt
                            )

                    if dwt_t_off is not None and candidate_t_off is not None:
                        later_t_off = max(int(dwt_t_off), int(candidate_t_off))
                        errors["t_offset_later_of_dwt_candidate_counterfactual"].append(
                            (later_t_off - tev["offset"]) * 1000.0 / fs
                        )
                        if q_on is not None:
                            later_qt = (later_t_off - int(q_on)) * 1000.0 / fs
                            ref_qt = (tev["offset"] - qev["onset"]) * 1000.0 / fs
                            errors["qt_interval_later_of_dwt_candidate_counterfactual"].append(
                                later_qt - ref_qt
                            )

                        for threshold, suffix in (
                            (0.48, "048"),
                            (0.52, "052"),
                            (0.58, "058"),
                        ):
                            selected_t_off = (
                                int(candidate_t_off)
                                if candidate_conf >= threshold
                                else int(dwt_t_off)
                            )
                            errors[f"t_offset_candidate_conf_ge_{suffix}_else_dwt"].append(
                                (selected_t_off - tev["offset"]) * 1000.0 / fs
                            )
                            later_if_confident = (
                                max(int(dwt_t_off), int(candidate_t_off))
                                if candidate_conf >= threshold
                                else int(dwt_t_off)
                            )
                            errors[
                                f"t_offset_later_if_candidate_conf_ge_{suffix}_else_dwt"
                            ].append(
                                (later_if_confident - tev["offset"]) * 1000.0 / fs
                            )
                            if q_on is not None:
                                selected_qt = (selected_t_off - int(q_on)) * 1000.0 / fs
                                later_qt = (
                                    later_if_confident - int(q_on)
                                ) * 1000.0 / fs
                                ref_qt = (tev["offset"] - qev["onset"]) * 1000.0 / fs
                                errors[f"qt_interval_candidate_conf_ge_{suffix}_else_dwt"].append(
                                    selected_qt - ref_qt
                                )
                                errors[
                                    f"qt_interval_later_if_candidate_conf_ge_{suffix}_else_dwt"
                                ].append(later_qt - ref_qt)

        record_rows.append(row_counts)

    metrics = {name: _stats(vals) for name, vals in errors.items()}
    source_metrics = {
        "qrs_duration_by_source": {
            source: _stats(vals) for source, vals in sorted(qrs_source_errors.items())
        },
        "t_offset_by_source": {
            source: _stats(vals) for source, vals in sorted(t_source_errors.items())
        },
    }
    worst_t_offset_errors = sorted(
        t_error_rows,
        key=lambda row: abs(float(row["error_ms"])),
        reverse=True,
    )[:25]
    return {
        "benchmark_version": "MEDCALC_LUDB_DELINEATION_DEV_V1",
        "scope": "DEVELOPMENT_NATIVE_MANUAL_FIDUCIAL_REFERENCE",
        "clinical_validation_claim_allowed": False,
        "tuning_role": "DEVELOPMENT_ONLY",
        "dataset": {
            "name": "Lobachevsky University Electrocardiography Database",
            "version": LUDB_VERSION,
            "doi": LUDB_DOI,
            "license": LUDB_LICENSE,
            "sampling_hz": 500,
            "selection": "FIXED_INDEX_STRATIFIED_NO_OUTCOME_SELECTION",
            "records": list(records),
        },
        "record_n": len(records),
        "evaluable_lead_n": int(lead_n),
        "qrs_reference_n": int(qrs_reference_n),
        "qrs_matched_n": int(qrs_matched_n),
        "qrs_match_rate": round(float(qrs_matched_n / qrs_reference_n), 6) if qrs_reference_n else None,
        "p_reference_n": int(p_reference_n),
        "p_matched_n": int(p_matched_n),
        "p_match_rate": round(float(p_matched_n / p_reference_n), 6) if p_reference_n else None,
        "t_reference_n": int(t_reference_n),
        "t_matched_n": int(t_matched_n),
        "t_match_rate": round(float(t_matched_n / t_reference_n), 6) if t_reference_n else None,
        "metrics": metrics,
        "source_metrics": source_metrics,
        "worst_t_offset_errors": worst_t_offset_errors,
        "records": record_rows,
        "interpretation": (
            "Development-only comparison against cardiologist-marked native LUDB fiducials. "
            "This benchmark is used to diagnose delineation error and must not be presented "
            "as independent clinical validation after it informs algorithm changes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=",".join(DEFAULT_RECORDS))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = [x.strip() for x in str(args.records).split(",") if x.strip()]
    if not records:
        raise SystemExit("No LUDB records selected.")
    result = benchmark(records)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
