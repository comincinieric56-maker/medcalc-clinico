"""Research P/QRS graph. Candidate edges express timing, never proven conduction."""
from __future__ import annotations

from typing import Any
import numpy as np

VERSION = "MEDCALC_R28_AV_EVENT_GRAPH_V1"
TOKEN_FEATURES = ("is_p", "is_qrs", "delta_s", "since_p_s", "since_qrs_s",
                  "confidence", "candidate_degree", "time_fraction")


def build_av_event_graph(p_events: list[dict], qrs_events: list[dict],
                         duration_s: float) -> dict[str, Any]:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    nodes = []
    for kind, events in (("P", p_events), ("QRS", qrs_events)):
        last = -1.0
        for event in sorted(events, key=lambda e: float(e["time_s"])):
            t, confidence = float(event["time_s"]), float(event["confidence"])
            if not np.isfinite(t) or not 0 <= t < duration_s:
                raise ValueError("Event outside observed signal")
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid event confidence")
            if t == last:
                raise ValueError("Duplicate event timestamp")
            last = t
            nodes.append({"id": f"{kind}{len(nodes)}", "kind": kind,
                          "time_s": t, "confidence": confidence})
    nodes.sort(key=lambda n: (n["time_s"], n["kind"]))
    p = [n for n in nodes if n["kind"] == "P"]
    r = [n for n in nodes if n["kind"] == "QRS"]
    edges, degree = [], {n["id"]: 0 for n in nodes}
    for pn in p:
        for rn in r:
            dt = rn["time_s"] - pn["time_s"]
            if 0.08 <= dt <= 0.50:
                edges.append({"p_id": pn["id"], "qrs_id": rn["id"],
                              "p_peak_to_r_peak_ms": round(dt * 1000, 3),
                              "confidence": min(pn["confidence"], rn["confidence"]),
                              "semantics": "CANDIDATE_ONLY"})
                degree[pn["id"]] += 1
                degree[rn["id"]] += 1
    tokens, last_p, last_r, last_event = [], None, None, 0.0
    for n in nodes:
        t = n["time_s"]
        tokens.append([float(n["kind"] == "P"), float(n["kind"] == "QRS"),
                       min(3.0, t - last_event),
                       min(3.0, t - last_p) if last_p is not None else 3.0,
                       min(3.0, t - last_r) if last_r is not None else 3.0,
                       n["confidence"], min(degree[n["id"]], 4) / 4.0,
                       t / duration_s])
        if n["kind"] == "P":
            last_p = t
        else:
            last_r = t
        last_event = t
    def timing(seq):
        intervals = np.diff([n["time_s"] for n in seq])
        if len(intervals) < 2:
            return {"median_ms": None, "cv": None, "rate_bpm": None}
        median = float(np.median(intervals))
        return {"median_ms": median * 1000,
                "cv": float(np.std(intervals) / np.mean(intervals)),
                "rate_bpm": 60 / median}
    atrial, ventricular = timing(p), timing(r)
    phase = []
    pp_s = (atrial["median_ms"] or 0) / 1000
    if pp_s:
        for rn in r:
            prior = [pn["time_s"] for pn in p if pn["time_s"] < rn["time_s"]]
            if prior and rn["time_s"] - prior[-1] <= 1.1 * pp_s:
                phase.append((rn["time_s"] - prior[-1]) / pp_s)
    # Low concentration suggests variable relative phase; it does not prove AV block.
    concentration = float(abs(np.mean(np.exp(2j * np.pi * np.asarray(phase))))) if len(phase) >= 3 else None
    uncoupled = [n for n in p if degree[n["id"]] == 0 and n["time_s"] + .5 < duration_s]
    return {"version": VERSION, "nodes": nodes, "candidate_edges": edges,
            "token_features": list(TOKEN_FEATURES), "tokens": tokens,
            "atrial_timing": atrial, "ventricular_timing": ventricular,
            "phase_concentration": concentration,
            "p_without_candidate_qrs_n": len(uncoupled),
            "ambiguous_node_n": sum(v > 1 for v in degree.values()),
            "edge_semantics": "TIMING_COMPATIBILITY_NOT_OBSERVED_CONDUCTION",
            "interval_semantics": "P_PEAK_TO_R_PEAK_NOT_CLINICAL_PR",
            "diagnostic_claim_allowed": False}
