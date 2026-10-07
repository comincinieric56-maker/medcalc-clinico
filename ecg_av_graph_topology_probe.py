"""Development-only graph-topology audit on frozen PTB-XL native records.

This does not classify or tune thresholds. It asks whether structural P/QRS graph
features carry label-associated signal independently of the temporal Transformer.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE, TARGETS, _adult_rows, _any_target_positive, _canonical,
    _download, _ensure_record, _target_positive,
)
from ecg_av_temporal_model import AVResearchModel, analyze_av_research

VERSION = "MEDCALC_R28_AV_GRAPH_TOPOLOGY_PROBE_V1"
NUMERIC_FEATURES = (
    "p_n", "qrs_n", "p_qrs_count_ratio", "p_interval_cv", "qrs_interval_cv",
    "p_rate_bpm", "qrs_rate_bpm", "p_qrs_rate_ratio", "phase_concentration",
    "candidate_edge_n", "candidate_edges_per_p", "max_non_crossing_match_n",
    "minimum_unmatched_p_n", "minimum_unmatched_p_fraction",
    "zero_degree_p_fraction", "ambiguous_node_fraction",
    "p_confidence_median", "qrs_confidence_median",
)


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def maximum_non_crossing_candidate_matches(
    p_times: list[float], qrs_times: list[float],
    *, minimum_s: float = 0.08, maximum_s: float = 0.50,
) -> int:
    """Maximum cardinality monotone matching under the graph timing contract.

    The result is structural compatibility only; a matched edge is not asserted
    to be physiological conduction.
    """
    p = sorted(float(x) for x in p_times)
    r = sorted(float(x) for x in qrs_times)
    if minimum_s < 0 or maximum_s <= minimum_s:
        raise ValueError("Invalid candidate timing window")
    if not np.isfinite([*p, *r]).all():
        raise ValueError("Event times must be finite")
    dp = np.zeros((len(p) + 1, len(r) + 1), dtype=np.int16)
    for i in range(1, len(p) + 1):
        for j in range(1, len(r) + 1):
            best = max(int(dp[i - 1, j]), int(dp[i, j - 1]))
            dt = r[j - 1] - p[i - 1]
            if minimum_s <= dt <= maximum_s:
                best = max(best, int(dp[i - 1, j - 1]) + 1)
            dp[i, j] = best
    return int(dp[-1, -1])


def graph_topology_features(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = list(graph.get("nodes") or [])
    p_nodes = [n for n in nodes if n.get("kind") == "P"]
    qrs_nodes = [n for n in nodes if n.get("kind") == "QRS"]
    p_times = [float(n["time_s"]) for n in p_nodes]
    qrs_times = [float(n["time_s"]) for n in qrs_nodes]
    p_n, qrs_n = len(p_nodes), len(qrs_nodes)

    degree = {str(n["id"]): 0 for n in nodes}
    for edge in graph.get("candidate_edges") or []:
        for key in ("p_id", "qrs_id"):
            node_id = str(edge[key])
            if node_id in degree:
                degree[node_id] += 1
    zero_p = sum(degree.get(str(n["id"]), 0) == 0 for n in p_nodes)
    matched = maximum_non_crossing_candidate_matches(p_times, qrs_times)
    atrial = graph.get("atrial_timing") or {}
    ventricular = graph.get("ventricular_timing") or {}
    p_rate = _finite(atrial.get("rate_bpm"))
    qrs_rate = _finite(ventricular.get("rate_bpm"))
    edge_n = len(graph.get("candidate_edges") or [])

    return {
        "p_n": p_n,
        "qrs_n": qrs_n,
        "p_qrs_count_ratio": (p_n / qrs_n) if qrs_n else None,
        "p_interval_cv": _finite(atrial.get("cv")),
        "qrs_interval_cv": _finite(ventricular.get("cv")),
        "p_rate_bpm": p_rate,
        "qrs_rate_bpm": qrs_rate,
        "p_qrs_rate_ratio": (p_rate / qrs_rate) if p_rate is not None and qrs_rate else None,
        "phase_concentration": _finite(graph.get("phase_concentration")),
        "candidate_edge_n": edge_n,
        "candidate_edges_per_p": (edge_n / p_n) if p_n else None,
        "max_non_crossing_match_n": matched,
        "minimum_unmatched_p_n": max(0, p_n - matched),
        "minimum_unmatched_p_fraction": (max(0, p_n - matched) / p_n) if p_n else None,
        "zero_degree_p_fraction": (zero_p / p_n) if p_n else None,
        "ambiguous_node_fraction": (int(graph.get("ambiguous_node_n") or 0) / len(nodes)) if nodes else None,
        "p_confidence_median": float(np.median([float(n["confidence"]) for n in p_nodes])) if p_nodes else None,
        "qrs_confidence_median": float(np.median([float(n["confidence"]) for n in qrs_nodes])) if qrs_nodes else None,
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"n": len(records)}
    for feature in NUMERIC_FEATURES:
        values = [_finite(row.get(feature)) for row in records]
        values = [v for v in values if v is not None]
        result[feature] = {
            "n": len(values),
            "median": float(np.median(values)) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return result


def _cliffs_delta(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    greater = less = 0
    for x in a:
        for y in b:
            greater += x > y
            less += x < y
    return float((greater - less) / (len(a) * len(b)))


def probe(checkpoint: Path, root: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "ptbxl_database.csv"
    _download(BASE + "/ptbxl_database.csv", metadata)
    df = pd.read_csv(metadata)
    if len(df) < 20000:
        raise ValueError("Incomplete PTB-XL metadata download")

    holdout = json.loads(Path(__file__).with_name("ecg_fast_gate_100_manifest.json").read_text())
    protected_ids = {int(r["ecg_id"]) for r in holdout["cases"]}
    protected_patients = set(df.loc[df.ecg_id.isin(protected_ids), "patient_id"])
    adult = _adult_rows(df, list(range(1, 9)))
    adult = adult.loc[~adult.patient_id.isin(protected_patients)].sort_values(["_hash", "ecg_id"])

    masks = {
        "AVB2": adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB2"]["scp"]))
                & ~adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB3"]["scp"])),
        "AVB3": adult._codes.map(lambda c: _target_positive(c, TARGETS["AVB3"]["scp"])),
        "CONTROL": ~adult._codes.map(_any_target_positive),
    }
    groups = {group: adult.loc[mask].head(32 if group == "CONTROL" else 8)
              for group, mask in masks.items()}

    filenames = [str(row.filename_hr) for rows in groups.values() for _, row in rows.iterrows()]
    download_errors = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {name: executor.submit(_ensure_record, root, name) for name in filenames}
        for name, future in futures.items():
            try:
                future.result()
            except Exception as exc:
                download_errors[name] = type(exc).__name__

    model = AVResearchModel(checkpoint)
    records_by_group: dict[str, list[dict[str, Any]]] = {k: [] for k in groups}
    selected_ids = []
    errors = []
    for group, rows in groups.items():
        for _, row in rows.iterrows():
            selected_ids.append(int(row.ecg_id))
            try:
                if str(row.filename_hr) in download_errors:
                    raise RuntimeError(download_errors[str(row.filename_hr)])
                path = _ensure_record(root, str(row.filename_hr))
                record = wfdb.rdrecord(str(path))
                canonical = _canonical(record.p_signal, int(record.fs), record.sig_name, int(row.ecg_id))
                result = analyze_av_research(canonical, model)
                if result.get("reason") != "UNVALIDATED_RESEARCH_MODEL":
                    raise RuntimeError(f"R28_NOT_EVALUABLE:{result.get('reason')}")
                features = graph_topology_features(result["graph"])
                probs = result.get("probabilities") or {}
                top = max(probs, key=probs.get) if probs else None
                records_by_group[group].append({
                    "ecg_id": int(row.ecg_id),
                    "top_class": top,
                    **features,
                })
                assert result["diagnostic_claim_allowed"] is False
                assert result["clinical_fusion_allowed"] is False
            except Exception as exc:
                errors.append({"group": group, "ecg_id": int(row.ecg_id),
                               "error": f"{type(exc).__name__}:{exc}"})
            print(f"GRAPH_PROBE {group} {len(records_by_group[group])}/{len(rows)}", flush=True)

    summaries = {group: _summary(rows) for group, rows in records_by_group.items()}
    control = records_by_group["CONTROL"]
    effect_vs_control = {}
    for group in ("AVB2", "AVB3"):
        effect_vs_control[group] = {}
        for feature in NUMERIC_FEATURES:
            a = [_finite(r.get(feature)) for r in records_by_group[group]]
            b = [_finite(r.get(feature)) for r in control]
            effect_vs_control[group][feature] = _cliffs_delta(
                [x for x in a if x is not None],
                [x for x in b if x is not None],
            )

    return {
        "version": VERSION,
        "role": "NATIVE_SIGNAL_GRAPH_TOPOLOGY_DEVELOPMENT_AUDIT_ONLY",
        "checkpoint_sha256": model.sha256,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "selection_policy": (
            "Same frozen-hash PTB-XL folds1-8 selection as native transfer probe; "
            "exclude FAST patients; up to 8 AVB2, 8 AVB3, 32 controls"
        ),
        "selected_record_ids_sha256": hashlib.sha256(json.dumps(selected_ids).encode()).hexdigest(),
        "groups": summaries,
        "records": records_by_group,
        "cliffs_delta_vs_control": effect_vs_control,
        "analysis_errors": errors,
        "threshold_tuning_allowed": False,
        "diagnostic_claim_allowed": False,
        "clinical_fusion_allowed": False,
        "limitations": [
            "Small development sample with diagnostic labels but no expert P/QRS event truth",
            "Native signals, not digitized images",
            "Graph matches encode temporal compatibility, never proven conduction",
            "Effect sizes are descriptive architecture evidence, not validation",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = probe(args.checkpoint, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"groups": report["groups"],
                      "cliffs_delta_vs_control": report["cliffs_delta_vs_control"],
                      "analysis_error_n": len(report["analysis_errors"])}, indent=2))
