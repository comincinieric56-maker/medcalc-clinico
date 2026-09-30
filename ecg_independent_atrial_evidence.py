from __future__ import annotations

from typing import Any, Dict

import numpy as np


VERSION = "MEDCALC_INDEPENDENT_ATRIAL_EVIDENCE_V1"
PREFERRED_LEADS = ("II", "V1", "aVF", "I", "III", "aVL", "V5", "V6")


def build_independent_atrial_consensus(
    per_lead: Dict[str, Dict[str, Any]],
    *,
    coincidence_ms: float = 36.0,
) -> Dict[str, Any]:
    """Audit cross-lead atrial events without changing canonical P fiducials.

    This layer is deliberately evidence-only. It clusters already observed raw
    P fiducials across independent leads and reports only events supported by at
    least two leads. It never creates P waves from timing, never writes back to
    raw_p_peaks_samples, and therefore cannot by itself create an AV-block
    diagnosis. Signal-level recovery can be added later behind this contract.
    """
    observations: list[tuple[float, str]] = []
    lead_fs: dict[str, int] = {}
    for lead in PREFERRED_LEADS:
        item = per_lead.get(lead) or {}
        if not item.get("evaluable"):
            continue
        fs = int(item.get("fs") or 0)
        if fs <= 0:
            continue
        lead_fs[lead] = fs
        for sample in sorted(set(item.get("raw_p_peaks_samples") or [])):
            try:
                t_ms = 1000.0 * int(sample) / fs
            except Exception:
                continue
            observations.append((t_ms, lead))

    observations.sort(key=lambda row: (row[0], row[1]))
    clusters: list[list[tuple[float, str]]] = []
    for obs in observations:
        if not clusters:
            clusters.append([obs])
            continue
        center = float(np.median([x[0] for x in clusters[-1]]))
        if abs(obs[0] - center) <= coincidence_ms:
            clusters[-1].append(obs)
        else:
            clusters.append([obs])

    events = []
    for cluster in clusters:
        leads = sorted(set(lead for _, lead in cluster))
        if len(leads) < 2:
            continue
        times = [t for t, _ in cluster]
        events.append({
            "time_ms": round(float(np.median(times)), 3),
            "support_leads": leads,
            "support_lead_n": len(leads),
            "spread_ms": round(float(max(times) - min(times)), 3),
        })

    times = np.asarray([row["time_ms"] for row in events], dtype=float)
    pp = np.diff(times) if times.size >= 2 else np.asarray([], dtype=float)
    pp_median = float(np.median(pp)) if pp.size else None
    pp_cv = (
        float(np.std(pp, ddof=1) / np.mean(pp))
        if pp.size >= 2 and float(np.mean(pp)) > 0
        else None
    )
    organized = bool(
        len(events) >= 4
        and pp_cv is not None
        and pp_cv <= 0.12
    )

    return {
        "version": VERSION,
        "evaluable": bool(events),
        "event_n": len(events),
        "events": events,
        "organized": organized,
        "pp_median_ms": round(pp_median, 3) if pp_median is not None else None,
        "pp_cv": round(pp_cv, 6) if pp_cv is not None else None,
        "source_leads": sorted({lead for row in events for lead in row["support_leads"]}),
        "policy": (
            "EVIDENCE_ONLY; GE_2_LEAD_TEMPORAL_CONSENSUS; "
            "NO_TIMING_SYNTHESIS; NO_CANONICAL_P_MUTATION; "
            "NO_DIAGNOSTIC_CLAIM"
        ),
        "diagnostic_claim_allowed": False,
    }
