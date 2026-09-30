from __future__ import annotations

from typing import Any, Dict

import numpy as np


VERSION = "MEDCALC_INDEPENDENT_ATRIAL_EVIDENCE_V1"
PREFERRED_LEADS = ("II", "V1", "aVF", "I", "III", "aVL", "V5", "V6")



def filter_atrial_candidates_outside_ventricular_repolarization(
    candidates: list[int],
    *,
    r_peaks: list[int],
    t_offsets: list[int] | None,
    fs: int,
) -> tuple[list[int], Dict[str, Any]]:
    """Remove candidates inside QRS/early-repolarization territory.

    Conservative evidence gate only. When a measured T offset is available,
    the protected interval extends from 80 ms before R through 40 ms after
    T-end. Without a T offset, a 420 ms post-R exclusion is used. This helper
    never creates atrial events.
    """
    if fs <= 0:
        return [], {"status": "REJECTED", "reason": "INVALID_FS"}
    r = sorted(set(int(v) for v in r_peaks))
    t = sorted(set(int(v) for v in (t_offsets or [])))
    qrs_pre = int(round(0.080 * fs))
    t_guard = int(round(0.040 * fs))
    fallback_post = int(round(0.420 * fs))

    protected: list[tuple[int, int]] = []
    for rp in r:
        after = [v for v in t if v > rp and v - rp <= int(round(0.700 * fs))]
        end = (after[0] + t_guard) if after else (rp + fallback_post)
        protected.append((rp - qrs_pre, end))

    kept = []
    rejected = []
    for sample in sorted(set(int(v) for v in candidates)):
        if any(lo <= sample <= hi for lo, hi in protected):
            rejected.append(sample)
        else:
            kept.append(sample)
    return kept, {
        "status": "APPLIED",
        "input_n": len(set(candidates)),
        "kept_n": len(kept),
        "rejected_n": len(rejected),
        "rejected_samples": rejected,
        "rule": "R_MINUS_80MS_THROUGH_TEND_PLUS_40MS_OR_R_PLUS_420MS",
    }


def recover_morphology_matched_atrial_candidates(
    signal_mv: list[float] | np.ndarray,
    *,
    seed_p_peaks: list[int],
    r_peaks: list[int],
    t_offsets: list[int] | None,
    fs: int,
) -> tuple[list[int], Dict[str, Any]]:
    """Recover morphology-matched atrial candidates without publishing them.

    Recovery requires >=3 observed P seeds, a reproducible median P template,
    amplitude above a robust noise floor, high positive morphology correlation,
    and location outside QRS/repolarization protection. Candidate timing is
    never synthesized from an expected P-P interval.
    """
    if fs <= 0:
        return [], {"status": "REJECTED", "reason": "INVALID_FS"}
    x = np.asarray(signal_mv, dtype=float).reshape(-1)
    if x.size < int(round(2.0 * fs)):
        return [], {"status": "REJECTED", "reason": "SIGNAL_LT_2S"}

    seeds = sorted(set(int(v) for v in seed_p_peaks if 0 <= int(v) < x.size))
    if len(seeds) < 3:
        return [], {
            "status": "REJECTED",
            "reason": "LT_3_OBSERVED_P_SEEDS",
            "seed_n": len(seeds),
        }

    half = max(8, int(round(0.055 * fs)))
    edge = max(2, int(round(0.012 * fs)))
    templates = []
    seed_amplitudes = []
    usable_seeds = []
    for center in seeds:
        a, b = center - half, center + half + 1
        if a < 0 or b > x.size:
            continue
        seg = np.asarray(x[a:b], dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        baseline = float(np.median(np.r_[seg[:edge], seg[-edge:]]))
        z = seg - baseline
        amp = float(np.ptp(z))
        norm = float(np.linalg.norm(z - np.mean(z)))
        if amp < 0.010 or norm <= 1e-10:
            continue
        templates.append((z - np.mean(z)) / norm)
        seed_amplitudes.append(amp)
        usable_seeds.append(center)

    if len(templates) < 3:
        return [], {
            "status": "REJECTED",
            "reason": "P_TEMPLATE_NOT_REPRODUCIBLE",
            "seed_n": len(seeds),
            "usable_seed_n": len(templates),
        }

    template = np.median(np.vstack(templates), axis=0)
    template_norm = float(np.linalg.norm(template))
    if template_norm <= 1e-10:
        return [], {"status": "REJECTED", "reason": "DEGENERATE_P_TEMPLATE"}
    template = template / template_norm

    dx = np.diff(x[np.isfinite(x)])
    if dx.size:
        med_dx = float(np.median(dx))
        mad_dx = float(np.median(np.abs(dx - med_dx)))
        noise = 1.4826 * mad_dx / np.sqrt(2.0)
    else:
        noise = 0.0
    median_seed_amp = float(np.median(seed_amplitudes))
    amplitude_gate = max(0.015, 0.35 * median_seed_amp, 4.5 * noise)

    raw_candidates: list[tuple[float, int, float]] = []
    seed_guard = int(round(0.100 * fs))
    step = max(1, int(round(0.004 * fs)))
    for center in range(half, int(x.size) - half, step):
        if any(abs(center - seed) <= seed_guard for seed in usable_seeds):
            continue
        seg = np.asarray(x[center-half:center+half+1], dtype=float)
        if not np.all(np.isfinite(seg)):
            continue
        baseline = float(np.median(np.r_[seg[:edge], seg[-edge:]]))
        z = seg - baseline
        amp = float(np.ptp(z))
        if amp < amplitude_gate:
            continue
        z0 = z - np.mean(z)
        norm = float(np.linalg.norm(z0))
        if norm <= 1e-10:
            continue
        corr = float(np.dot(z0 / norm, template))
        if corr < 0.94:
            continue
        raw_candidates.append((corr, center, amp))

    protected_kept, protection_audit = (
        filter_atrial_candidates_outside_ventricular_repolarization(
            [center for _, center, _ in raw_candidates],
            r_peaks=r_peaks,
            t_offsets=t_offsets,
            fs=fs,
        )
    )
    protected_set = set(protected_kept)
    eligible = [row for row in raw_candidates if row[1] in protected_set]

    refractory = int(round(0.180 * fs))
    accepted: list[tuple[float, int, float]] = []
    for row in sorted(eligible, key=lambda z: (-z[0], z[1])):
        if any(abs(row[1] - prev[1]) < refractory for prev in accepted):
            continue
        accepted.append(row)
    accepted.sort(key=lambda z: z[1])

    return [center for _, center, _ in accepted], {
        "status": "APPLIED",
        "source": "OBSERVED_P_TEMPLATE_MORPHOLOGY_SEARCH",
        "seed_n": len(seeds),
        "usable_seed_n": len(templates),
        "median_seed_amplitude_mv": round(median_seed_amp, 6),
        "noise_mv": round(float(noise), 6),
        "amplitude_gate_mv": round(float(amplitude_gate), 6),
        "correlation_gate": 0.94,
        "raw_candidate_n": len(raw_candidates),
        "accepted_n": len(accepted),
        "accepted": [
            {
                "sample": int(center),
                "correlation": round(float(corr), 6),
                "amplitude_mv": round(float(amp), 6),
            }
            for corr, center, amp in accepted
        ],
        "ventricular_protection": protection_audit,
        "policy": "EVIDENCE_ONLY; NO_TIMING_SYNTHESIS; NO_DIAGNOSTIC_CLAIM",
    }


def recover_crosslead_atrial_candidates(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
    *,
    coincidence_ms: float = 36.0,
) -> Dict[str, Any]:
    """Require independently recovered morphology evidence in >=2 leads."""
    lead_items = canonical_ecg.get("leads") or {}
    observations: list[tuple[float, str, int]] = []
    per_lead_audit: Dict[str, Any] = {}

    for lead in PREFERRED_LEADS:
        measured = per_lead.get(lead) or {}
        source = lead_items.get(lead) or {}
        if not measured.get("evaluable"):
            continue
        fs = int(measured.get("fs") or source.get("fs") or canonical_ecg.get("fs") or 0)
        signal = source.get("signal_mv")
        if signal is None:
            signal = []
        t_offsets = [
            int(beat["t_offset_sample"])
            for beat in (measured.get("beats") or [])
            if beat.get("t_offset_sample") is not None
        ]
        candidates, audit = recover_morphology_matched_atrial_candidates(
            signal,
            seed_p_peaks=list(measured.get("raw_p_peaks_samples") or []),
            r_peaks=list(measured.get("r_peaks_samples") or []),
            t_offsets=t_offsets,
            fs=fs,
        )
        per_lead_audit[lead] = audit
        for sample in candidates:
            observations.append((1000.0 * sample / fs, lead, sample))

    observations.sort(key=lambda row: (row[0], row[1], row[2]))
    clusters: list[list[tuple[float, str, int]]] = []
    for obs in observations:
        if not clusters:
            clusters.append([obs])
            continue
        center = float(np.median([x[0] for x in clusters[-1]]))
        if abs(obs[0] - center) <= coincidence_ms:
            clusters[-1].append(obs)
        else:
            clusters.append([obs])

    recovered_events = []
    for cluster in clusters:
        leads = sorted(set(row[1] for row in cluster))
        if len(leads) < 2:
            continue
        times = [row[0] for row in cluster]
        recovered_events.append({
            "time_ms": round(float(np.median(times)), 3),
            "support_leads": leads,
            "support_lead_n": len(leads),
            "spread_ms": round(float(max(times) - min(times)), 3),
        })

    observed = build_independent_atrial_consensus(
        per_lead,
        coincidence_ms=coincidence_ms,
    )
    combined_times = [float(row["time_ms"]) for row in observed.get("events") or []]
    for row in recovered_events:
        t_ms = float(row["time_ms"])
        if not any(abs(t_ms - prev) <= coincidence_ms for prev in combined_times):
            combined_times.append(t_ms)
    combined_times.sort()

    pp = np.diff(np.asarray(combined_times, dtype=float)) if len(combined_times) >= 2 else np.asarray([], dtype=float)
    pp_cv = (
        float(np.std(pp, ddof=1) / np.mean(pp))
        if pp.size >= 2 and float(np.mean(pp)) > 0
        else None
    )
    organized_augmented = bool(
        recovered_events
        and len(combined_times) >= 4
        and pp_cv is not None
        and pp_cv <= 0.12
    )

    return {
        "version": VERSION,
        "evaluable": bool(recovered_events),
        "recovered_event_n": len(recovered_events),
        "recovered_events": recovered_events,
        "observed_consensus": observed,
        "combined_event_times_ms": [round(v, 3) for v in combined_times],
        "combined_pp_cv": round(pp_cv, 6) if pp_cv is not None else None,
        "organized_augmented": organized_augmented,
        "per_lead_audit": per_lead_audit,
        "policy": (
            "EVIDENCE_ONLY; RECOVERED_EVENT_REQUIRES_GE_2_LEADS; "
            "QRS_T_PROTECTED; NO_TIMING_SYNTHESIS; "
            "NO_CANONICAL_P_MUTATION; NO_DIAGNOSTIC_CLAIM"
        ),
        "diagnostic_claim_allowed": False,
    }

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
