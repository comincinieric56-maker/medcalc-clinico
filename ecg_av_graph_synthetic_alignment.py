"""Compare frozen synthetic graph topology with the native PTB-XL audit.

Development-only architecture check. No thresholds, weights, or clinical outputs
are modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecg_av_graph_topology_probe import NUMERIC_FEATURES, _finite, _summary, graph_topology_features
from ecg_av_temporal_model import AVResearchModel
from ecg_av_training_data import CLASSES, FS, synthetic_case


AVB2_SYNTHETIC = {"MOBITZ_I", "MOBITZ_II", "TWO_TO_ONE", "HIGH_GRADE"}


def run(checkpoint: Path, native_report: Path) -> dict:
    model = AVResearchModel(checkpoint)
    by_class = {name: [] for name in CLASSES}
    for i in range(256):
        row = synthetic_case(i, "heldout")
        result = model.analyze_signal(row["signal"][0], FS)
        if result.get("reason") != "UNVALIDATED_RESEARCH_MODEL":
            raise RuntimeError(f"SYNTHETIC_NOT_EVALUABLE:{i}:{result.get('reason')}")
        by_class[CLASSES[row["label"]]].append(graph_topology_features(result["graph"]))

    native = json.loads(native_report.read_text())
    synthetic_avb2 = [row for name in AVB2_SYNTHETIC for row in by_class[name]]
    mappings = {
        "AVB2": synthetic_avb2,
        "AVB3": by_class["AV_DISSOCIATION"],
        "CONTROL_REFERENCE": by_class["SINUS"],
    }
    summaries = {name: _summary(rows) for name, rows in by_class.items()}
    aggregate = {name: _summary(rows) for name, rows in mappings.items()}

    cross_domain = {}
    for native_group, synth_group in (("AVB2", "AVB2"), ("AVB3", "AVB3"), ("CONTROL", "CONTROL_REFERENCE")):
        cross_domain[native_group] = {}
        for feature in NUMERIC_FEATURES:
            nmed = _finite(((native["groups"].get(native_group) or {}).get(feature) or {}).get("median"))
            smed = _finite(((aggregate.get(synth_group) or {}).get(feature) or {}).get("median"))
            if nmed is None or smed is None:
                cross_domain[native_group][feature] = {"native_median": nmed, "synthetic_median": smed,
                                                       "ratio": None, "absolute_difference": None}
            else:
                cross_domain[native_group][feature] = {
                    "native_median": nmed,
                    "synthetic_median": smed,
                    "ratio": (nmed / smed) if abs(smed) > 1e-12 else None,
                    "absolute_difference": nmed - smed,
                }

    return {
        "role": "R28_SYNTHETIC_NATIVE_GRAPH_TOPOLOGY_ALIGNMENT_AUDIT_ONLY",
        "clinical_ready": False,
        "threshold_tuning_allowed": False,
        "weights_changed": False,
        "checkpoint_sha256": model.sha256,
        "synthetic_cases": 256,
        "synthetic_by_class": summaries,
        "synthetic_aggregate": aggregate,
        "native_vs_synthetic_medians": cross_domain,
        "interpretation_guard": (
            "Direction/agreement is architecture evidence only; PTB-XL groups are development data "
            "and are not independent validation after this inspection."
        ),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--native-report", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = run(args.checkpoint, args.native_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    keep = {
        group: {
            feature: report["native_vs_synthetic_medians"][group][feature]
            for feature in ("p_qrs_count_ratio", "phase_concentration",
                            "minimum_unmatched_p_fraction", "candidate_edges_per_p",
                            "p_qrs_rate_ratio")
        }
        for group in ("AVB2", "AVB3", "CONTROL")
    }
    print(json.dumps(keep, indent=2))
