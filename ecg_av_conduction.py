from __future__ import annotations

from typing import Any, Dict

import numpy as np


AV_VERSION = "MEDCALC_AV_CONDUCTION_V2"
PREFERRED = ("II","V1","aVF","I","III","aVL","V5","V6","V2","V4")


def _choose_lead(per_lead: Dict[str, Dict[str, Any]]) -> str | None:
    """Choose the strongest atrial-evidence lead, with conventional leads as tie-breakers.

    A fixed lead-order bonus previously overwhelmed signal quality and could
    force AV analysis onto a marginal lead II even when another lead showed a
    cleaner organized P sequence. The new score never relaxes P/QRS-count
    requirements; it ranks only already-evaluable candidates.
    """
    candidates = []
    for priority, lead in enumerate(PREFERRED):
        item = per_lead.get(lead) or {}
        p = item.get("raw_p_peaks_samples") or []
        r = item.get("r_peaks_samples") or []
        if not item.get("evaluable") or len(p) < 4 or len(r) < 3:
            continue

        atrial = item.get("atrial_activity") or {}
        fs = int(item.get("fs") or 500)
        p_arr = np.unique(np.asarray(p, dtype=int))
        pp = np.diff(p_arr) * 1000.0 / max(fs, 1)
        pp_cv = (
            float(np.std(pp, ddof=1) / np.mean(pp))
            if pp.size >= 2 and float(np.mean(pp)) > 0
            else None
        )
        organized_p = bool(pp_cv is not None and pp_cv <= 0.12)
        reproducible_p = bool(atrial.get("p_wave_reproducible"))
        coupling = float(atrial.get("p_qrs_coupling_fraction") or 0.0)
        confidence = float(item.get("confidence") or 0.0)

        quality_score = (
            40.0 * int(reproducible_p)
            + 25.0 * int(organized_p)
            + 10.0 * min(max(confidence, 0.0), 1.0)
            + 8.0 * min(max(coupling, 0.0), 1.0)
            + min(len(p_arr), 12)
            + 0.5 * min(len(r), 12)
        )
        # Conventional P-rich leads remain a small tie-breaker only.
        tie_break = 2.0 * (len(PREFERRED) - priority) / len(PREFERRED)
        candidates.append((quality_score + tie_break, lead))

    return max(candidates)[1] if candidates else None


def _validated_independent_atrial_times_ms(
    evidence: Dict[str, Any] | None,
) -> list[float]:
    """Return a conservative recovered atrial train or an empty list.

    Recovered evidence is eligible for AV mapping only when it is organized,
    adds at least two cross-lead events to >=3 already observed consensus P
    events, and every recovered event is supported by >=2 leads with tight
    temporal agreement. This gate does not itself classify AV block.
    """
    if not isinstance(evidence, dict) or not evidence.get("organized_augmented"):
        return []
    recovered = list(evidence.get("recovered_events") or [])
    observed = evidence.get("observed_consensus") or {}
    observed_n = int(observed.get("event_n") or 0)
    if observed_n < 3 or len(recovered) < 2:
        return []
    for event in recovered:
        if int(event.get("support_lead_n") or 0) < 2:
            return []
        try:
            spread = float(event.get("spread_ms") or 0.0)
        except Exception:
            return []
        if spread > 40.0:
            return []

    values = []
    for value in evidence.get("combined_event_times_ms") or []:
        try:
            value = float(value)
        except Exception:
            continue
        if np.isfinite(value):
            values.append(value)
    times = sorted(set(values))
    if len(times) < max(5, observed_n + 2):
        return []

    pp = np.diff(np.asarray(times, dtype=float))
    if pp.size < 2 or float(np.mean(pp)) <= 0:
        return []
    pp_cv = float(np.std(pp, ddof=1) / np.mean(pp))
    if pp_cv > 0.10:
        return []
    return times


def _choose_ventricular_lead(
    per_lead: Dict[str, Dict[str, Any]],
) -> str | None:
    candidates = []
    for priority, lead in enumerate(PREFERRED):
        item = per_lead.get(lead) or {}
        r = item.get("r_peaks_samples") or []
        if not item.get("evaluable") or len(r) < 3:
            continue
        confidence = float(item.get("confidence") or 0.0)
        tie_break = 0.01 * (len(PREFERRED) - priority) / len(PREFERRED)
        candidates.append((confidence + tie_break, len(r), lead))
    return max(candidates)[2] if candidates else None


def analyze_av_conduction(
    per_lead: Dict[str, Dict[str, Any]],
    global_atrial: Dict[str, Any],
    global_metrics: Dict[str, Dict[str, Any]] | None = None,
    independent_atrial_evidence: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Analyze P-to-QRS conduction without assuming every P must conduct.

    Organized P-P timing is evaluated independently from P-QRS coupling so 2:1,
    high-grade and complete AV block remain detectable. First-degree AV delay,
    in contrast, requires reproducible 1:1 P-QRS conduction.
    """
    lead = _choose_lead(per_lead)
    independent_times_ms = _validated_independent_atrial_times_ms(
        independent_atrial_evidence
    )
    independent_used = False

    if lead is None and independent_times_ms:
        lead = _choose_ventricular_lead(per_lead)

    if lead is None:
        global_metrics = global_metrics or {}
        pr_metric = dict(global_metrics.get("pr_ms") or {})
        pr_value = pr_metric.get("value")
        pr_conf = float(pr_metric.get("confidence") or 0.0)
        try:
            pr_value = float(pr_value) if pr_value is not None else None
        except Exception:
            pr_value = None
        coupling = float(global_atrial.get("rhythm_p_qrs_coupling_fraction") or 0.0)
        reproducible = bool(global_atrial.get("p_wave_reproducible"))
        if (
            reproducible
            and bool(global_atrial.get("pr_reportable"))
            and coupling >= 0.70
            and pr_value is not None
            and pr_value > 200.0
            and pr_conf >= 0.55
        ):
            return {
                "version": AV_VERSION,
                "evaluable": True,
                "lead": None,
                "classification": "FIRST_DEGREE_AV_DELAY_COMPATIBLE",
                "confidence": round(min(0.88, pr_conf), 6),
                "pr_median_ms": round(pr_value, 3),
                "one_to_one": True,
                "stable_pr": True,
                "nonconducted_p_n": 0,
                "p_qrs_coupling_fraction": round(coupling, 6),
                "basis": [
                    "GLOBAL_REPRODUCIBLE_P_QRS_COUPLING",
                    "GLOBAL_PR_GT_200MS",
                    "PR_MEASUREMENT_CONFIDENCE_GE_0_55",
                ],
                "diagnostic_claim_allowed": False,
                "source": "GLOBAL_PR_CONSENSUS_FALLBACK",
                "independent_atrial_evidence_used": False,
            }
        return {
            "version": AV_VERSION,
            "evaluable": False,
            "classification": "AV_CONDUCTION_NOT_EVALUABLE",
            "reason": "NO_LEAD_WITH_SUFFICIENT_P_AND_QRS_CANDIDATES",
            "diagnostic_claim_allowed": False,
            "independent_atrial_evidence_used": False,
        }

    item = per_lead[lead]
    fs = int(item.get("fs") or 500)
    raw_p = np.asarray(item.get("raw_p_peaks_samples") or [], dtype=int)
    r = np.unique(np.asarray(item.get("r_peaks_samples") or [], dtype=int))
    p = np.unique(raw_p)

    if independent_times_ms and len(r) >= 3:
        independent_p = np.unique(np.asarray([
            int(round(float(t_ms) * fs / 1000.0))
            for t_ms in independent_times_ms
        ], dtype=int))
        ratio = len(independent_p) / max(len(r), 1)
        if len(independent_p) >= 5 and ratio >= 1.35:
            p = independent_p
            independent_used = True

    pp = np.diff(p) * 1000.0/fs
    rr = np.diff(r) * 1000.0/fs
    pp_med = float(np.median(pp)) if pp.size else None
    rr_med = float(np.median(rr)) if rr.size else None
    pp_cv = float(np.std(pp,ddof=1)/np.mean(pp)) if pp.size >= 2 and np.mean(pp)>0 else None
    rr_cv = float(np.std(rr,ddof=1)/np.mean(rr)) if rr.size >= 2 and np.mean(rr)>0 else None
    atrial_regular = bool(pp_cv is not None and pp_cv <= 0.12)
    ventricular_regular = bool(rr_cv is not None and rr_cv <= 0.12)

    # Assign each QRS to the nearest preceding P within the physiological
    # PR window. Mapping P -> next QRS can steal a QRS from the truly conducted
    # P during fast 2:1 conduction (the earlier blocked P can still lie <500 ms
    # before that QRS), which corrupts dropped-beat pattern recognition.
    mappings = [
        {"p": int(pi), "conducted": False, "pr_ms": None}
        for pi in p
    ]
    claimed_p_idx: set[int] = set()
    min_pr_samples = int(np.floor(80.0 * fs / 1000.0))
    max_pr_samples = int(np.ceil(500.0 * fs / 1000.0))
    for ri_raw in r:
        ri = int(ri_raw)
        candidate_idx = np.where(
            (p < ri)
            & ((ri - p) >= min_pr_samples)
            & ((ri - p) <= max_pr_samples)
        )[0]
        if candidate_idx.size == 0:
            continue
        chosen = None
        for idx in candidate_idx[::-1]:
            j = int(idx)
            if j not in claimed_p_idx:
                chosen = j
                break
        if chosen is None:
            continue
        pi = int(p[chosen])
        pr_ms = (ri - pi) * 1000.0 / fs
        mappings[chosen] = {
            "p": pi,
            "conducted": True,
            "r": ri,
            "pr_ms": float(pr_ms),
        }
        claimed_p_idx.add(chosen)

    seen = {
        int(m["r"])
        for m in mappings
        if m.get("conducted") and m.get("r") is not None
    }

    conducted = [m for m in mappings if m.get("conducted")]
    dropped = [m for m in mappings if not m.get("conducted")]
    pr = np.asarray([m["pr_ms"] for m in conducted],dtype=float)
    pr_med = float(np.median(pr)) if pr.size else None
    pr_mad = float(np.median(np.abs(pr-np.median(pr)))) if pr.size else None
    stable_pr = bool(pr.size >= 3 and pr_mad is not None and pr_mad <= 30.0)
    coupling_fraction = len(conducted)/max(len(p),1)

    per_lead_atrial = item.get("atrial_activity") or {}
    coupled_p_reproducible = bool(per_lead_atrial.get("p_wave_reproducible"))
    one_to_one = bool(
        coupled_p_reproducible
        and len(dropped)==0
        and len(conducted)>=3
        and len(seen)==len(conducted)
        and abs(len(p)-len(r)) <= 1
    )

    atrial_rate = 60000.0/pp_med if pp_med and pp_med > 0 else None
    ventricular_rate = 60000.0/rr_med if rr_med and rr_med > 0 else None

    # For AV dissociation, evaluate the phase of every QRS relative to the
    # immediately preceding atrial depolarization. Randomly varying phase is
    # stronger evidence than naive P->next-QRS time-window matches.
    phase_pr = []
    for ri in r:
        prior = p[p < ri]
        if prior.size == 0:
            continue
        delta = (int(ri)-int(prior[-1]))*1000.0/fs
        if pp_med is None or delta <= 1.05*pp_med:
            phase_pr.append(float(delta))
    phase_arr = np.asarray(phase_pr,dtype=float)
    phase_mad = (
        float(np.median(np.abs(phase_arr-np.median(phase_arr))))
        if phase_arr.size >= 3 else None
    )
    phase_range = (
        float(np.max(phase_arr)-np.min(phase_arr))
        if phase_arr.size >= 3 else None
    )
    av_dissociation_phase = bool(
        pp_med is not None
        and phase_mad is not None
        and phase_range is not None
        and phase_mad >= max(50.0,0.15*pp_med)
        and phase_range >= 0.30*pp_med
    )

    classification = "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"
    confidence = 0.65 if atrial_regular else 0.35
    basis = []

    if one_to_one and stable_pr and pr_med is not None and pr_med > 200.0:
        classification = "FIRST_DEGREE_AV_DELAY_COMPATIBLE"
        confidence = 0.90
        basis = ["1_TO_1_P_QRS","PR_MEDIAN_GT_200MS","PR_STABLE"]
    elif (
        atrial_regular
        and ventricular_regular
        and len(p) >= 5
        and len(r) >= 3
        and atrial_rate is not None
        and ventricular_rate is not None
        and atrial_rate > 1.25 * ventricular_rate
        and av_dissociation_phase
        and not stable_pr
    ):
        classification = "COMPLETE_AV_BLOCK_COMPATIBLE"
        confidence = 0.86
        basis = [
            "REGULAR_ATRIAL_SEQUENCE",
            "REGULAR_SLOWER_VENTRICULAR_SEQUENCE",
            "NO_STABLE_AV_COUPLING",
        ]
    elif atrial_regular and len(dropped) >= 1 and len(conducted) >= 2:
        flags = [bool(m.get("conducted")) for m in mappings]
        max_consecutive_drop = 0
        run = 0
        for flag in flags:
            if not flag:
                run += 1
                max_consecutive_drop=max(max_consecutive_drop,run)
            else:
                run=0

        p_to_r_ratio = len(p)/max(len(r),1)
        even_conducted = sum(flags[::2])
        odd_conducted = sum(flags[1::2])
        alternating = abs(even_conducted-odd_conducted) >= max(1,len(flags)//3)

        if 1.75 <= p_to_r_ratio <= 2.25 and alternating:
            classification = "TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
            confidence = 0.84
            basis = ["REGULAR_P_SEQUENCE","APPROX_2_TO_1_P_QRS_RATIO","ALTERNATING_CONDUCTION"]
        elif max_consecutive_drop >= 2:
            classification = "HIGH_GRADE_AV_BLOCK_COMPATIBLE"
            confidence = 0.88
            basis = ["REGULAR_P_SEQUENCE","GE_2_CONSECUTIVE_NONCONDUCTED_P"]
        elif stable_pr:
            classification = "MOBITZ_II_COMPATIBLE"
            confidence = 0.80
            basis = ["REGULAR_P_SEQUENCE","DROPPED_P","STABLE_CONDUCTED_PR"]
        else:
            # Wenckebach requires progressive PR prolongation before a dropped P.
            progressive = False
            for j,m in enumerate(mappings):
                if m.get("conducted") or j < 3:
                    continue
                prior = [
                    float(x["pr_ms"]) for x in mappings[max(0,j-3):j]
                    if x.get("conducted") and x.get("pr_ms") is not None
                ]
                if len(prior) >= 3 and all((b-a) > 8.0 for a,b in zip(prior[:-1],prior[1:])):
                    progressive = True
                    break
            if progressive:
                classification = "MOBITZ_I_WENCKEBACH_COMPATIBLE"
                confidence = 0.78
                basis = ["REGULAR_P_SEQUENCE","PROGRESSIVE_PR_PROLONGATION","DROPPED_P"]

    return {
        "version": AV_VERSION,
        "evaluable": True,
        "lead": lead,
        "classification": classification,
        "confidence": round(confidence,6),
        "p_count": int(len(p)),
        "qrs_count": int(len(r)),
        "conducted_p_n": int(len(conducted)),
        "nonconducted_p_n": int(len(dropped)),
        "p_qrs_coupling_fraction": round(float(coupling_fraction),6),
        "pp_median_ms": round(pp_med,3) if pp_med is not None else None,
        "pp_cv": round(pp_cv,6) if pp_cv is not None else None,
        "rr_median_ms": round(rr_med,3) if rr_med is not None else None,
        "rr_cv": round(rr_cv,6) if rr_cv is not None else None,
        "atrial_rate_bpm": round(atrial_rate,3) if atrial_rate is not None else None,
        "ventricular_rate_bpm": round(ventricular_rate,3) if ventricular_rate is not None else None,
        "pr_median_ms": round(pr_med,3) if pr_med is not None else None,
        "pr_mad_ms": round(pr_mad,3) if pr_mad is not None else None,
        "qrs_to_preceding_p_phase_mad_ms": round(phase_mad,3) if phase_mad is not None else None,
        "qrs_to_preceding_p_phase_range_ms": round(phase_range,3) if phase_range is not None else None,
        "av_dissociation_phase": av_dissociation_phase,
        "atrial_sequence_regular": atrial_regular,
        "ventricular_sequence_regular": ventricular_regular,
        "one_to_one": one_to_one,
        "stable_pr": stable_pr,
        "basis": basis,
        "diagnostic_claim_allowed": False,
        "source": "ORGANIZED_P_SEQUENCE_TO_QRS_MAPPING",
    }
