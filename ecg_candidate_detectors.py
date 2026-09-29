from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from ecg_measurement_consensus import threshold_relation


CANDIDATE_VERSION = "MEDCALC_ECG_HIGH_RECALL_CANDIDATES_V2"
PREFERRED_AV_LEADS = ("II", "V1", "aVF", "I", "III", "aVL", "V5", "V6", "V2", "V4")


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _append(
    rows: list[Dict[str, Any]],
    *,
    domain: str,
    code: str,
    score: float,
    evidence: list[str],
    source_groups: list[str],
    required_measurements: list[str] | None = None,
    boundary_requirements: list[Dict[str, Any]] | None = None,
    specialist_confirmed: bool = False,
) -> None:
    groups = sorted(set(source_groups))
    rows.append({
        "domain": domain,
        "code": code,
        "score": round(_clip(score), 6),
        "evidence": list(dict.fromkeys(evidence)),
        "source_groups": groups,
        "independent_evidence_n": len(groups),
        "required_measurements": sorted(set(required_measurements or [])),
        "boundary_requirements": list(boundary_requirements or []),
        "specialist_confirmed": bool(specialist_confirmed),
        "candidate_only": True,
    })


def _global(feature_graph: Dict[str, Any], key: str) -> tuple[float | None, float]:
    row = ((feature_graph.get("global") or {}).get(key) or {})
    return _finite(row.get("value")), float(_finite(row.get("confidence")) or 0.0)


def _av_sequence_candidates(per_lead: Dict[str, Dict[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for lead in PREFERRED_AV_LEADS:
        item = per_lead.get(lead) or {}
        if not item.get("evaluable"):
            continue
        fs = int(item.get("fs") or 500)
        p = np.unique(np.asarray(item.get("raw_p_peaks_samples") or [], dtype=int))
        r = np.unique(np.asarray(item.get("r_peaks_samples") or [], dtype=int))
        if len(p) < 4 or len(r) < 3 or fs <= 0:
            continue

        pp = np.diff(p) * 1000.0 / fs
        rr = np.diff(r) * 1000.0 / fs
        if pp.size < 2 or np.mean(pp) <= 0:
            continue
        pp_cv = float(np.std(pp, ddof=1) / np.mean(pp)) if pp.size >= 2 else 1.0
        rr_cv = float(np.std(rr, ddof=1) / np.mean(rr)) if rr.size >= 2 and np.mean(rr) > 0 else 1.0
        atrial_regular = pp_cv <= 0.18
        ventricular_regular = rr_cv <= 0.18
        pp_med = float(np.median(pp))
        rr_med = float(np.median(rr)) if rr.size else None

        # High-recall AV candidate mapping must mirror the specialist engine:
        # assign each QRS to the nearest preceding P in the physiological PR
        # window. P->next-QRS can misassign fast 2:1 conduction by letting a
        # blocked P claim the QRS belonging to the next P.
        mappings = [
            {"p": int(pi), "r": None, "conducted": False, "pr_ms": None}
            for pi in p
        ]
        claimed_p_idx: set[int] = set()
        min_pr_samples = int(np.floor(70.0 * fs / 1000.0))
        max_pr_samples = int(np.ceil(550.0 * fs / 1000.0))
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
                "r": ri,
                "conducted": True,
                "pr_ms": float(pr_ms),
            }
            claimed_p_idx.add(chosen)

        conducted = [m for m in mappings if m["conducted"]]
        dropped = [m for m in mappings if not m["conducted"]]
        pr = np.asarray([m["pr_ms"] for m in conducted], dtype=float)
        pr_mad = float(np.median(np.abs(pr - np.median(pr)))) if pr.size >= 2 else None
        stable_pr = bool(pr.size >= 3 and pr_mad is not None and pr_mad <= 40.0)
        flags = [bool(m["conducted"]) for m in mappings]
        max_drop = 0
        run = 0
        for flag in flags:
            if flag:
                run = 0
            else:
                run += 1
                max_drop = max(max_drop, run)

        ratio = len(p) / max(len(r), 1)
        progressive = False
        for j, m in enumerate(mappings):
            if m["conducted"] or j < 3:
                continue
            prior = [
                float(x["pr_ms"]) for x in mappings[max(0, j - 3):j]
                if x["conducted"] and x["pr_ms"] is not None
            ]
            if len(prior) >= 3 and all((b - a) > 5.0 for a, b in zip(prior[:-1], prior[1:])):
                progressive = True
                break

        phase = []
        for ri in r:
            prior = p[p < ri]
            if not prior.size:
                continue
            delta = (int(ri) - int(prior[-1])) * 1000.0 / fs
            if delta <= 1.05 * pp_med:
                phase.append(float(delta))
        phase_arr = np.asarray(phase, dtype=float)
        phase_range = float(np.ptp(phase_arr)) if phase_arr.size >= 3 else 0.0
        phase_mad = (
            float(np.median(np.abs(phase_arr - np.median(phase_arr))))
            if phase_arr.size >= 3 else 0.0
        )
        atrial_rate = 60000.0 / pp_med if pp_med > 0 else None
        ventricular_rate = 60000.0 / rr_med if rr_med and rr_med > 0 else None

        evidence = {
            "lead": lead,
            "atrial_regular": atrial_regular,
            "ventricular_regular": ventricular_regular,
            "dropped_p_n": len(dropped),
            "conducted_p_n": len(conducted),
            "stable_pr": stable_pr,
            "progressive_pr": progressive,
            "max_consecutive_dropped_p": max_drop,
            "p_to_qrs_ratio": round(ratio, 4),
            "phase_range_ms": round(phase_range, 3),
            "phase_mad_ms": round(phase_mad, 3),
            "atrial_rate_bpm": round(atrial_rate, 3) if atrial_rate else None,
            "ventricular_rate_bpm": round(ventricular_rate, 3) if ventricular_rate else None,
        }

        if atrial_regular and len(dropped) >= 1 and len(conducted) >= 2:
            if 1.60 <= ratio <= 2.40:
                code = "TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
                score = 0.76 + 0.04 * min(len(dropped), 2)
                ev = ["REGULAR_P_SEQUENCE", "P_TO_QRS_RATIO_NEAR_2_TO_1", "NONCONDUCTED_P_PRESENT"]
            elif max_drop >= 2:
                code = "HIGH_GRADE_AV_BLOCK_COMPATIBLE"
                score = 0.82
                ev = ["REGULAR_P_SEQUENCE", "GE_2_CONSECUTIVE_NONCONDUCTED_P"]
            elif stable_pr:
                code = "MOBITZ_II_COMPATIBLE"
                score = 0.76
                ev = ["REGULAR_P_SEQUENCE", "NONCONDUCTED_P_PRESENT", "STABLE_CONDUCTED_PR"]
            elif progressive:
                code = "MOBITZ_I_WENCKEBACH_COMPATIBLE"
                score = 0.74
                ev = ["REGULAR_P_SEQUENCE", "PROGRESSIVE_PR_BEFORE_DROP", "NONCONDUCTED_P_PRESENT"]
            else:
                code = "SECOND_DEGREE_AV_BLOCK_CANDIDATE"
                score = 0.62
                ev = ["REGULAR_P_SEQUENCE", "NONCONDUCTED_P_PRESENT"]
            rows.append({
                "domain": "AV_CONDUCTION",
                "code": code,
                "score": round(_clip(score), 6),
                "evidence": ev,
                "source_groups": ["ATRIAL_SEQUENCE", "P_QRS_MAPPING"],
                "independent_evidence_n": 2,
                "required_measurements": ["r_peaks"],
                "specialist_confirmed": False,
                "candidate_only": True,
                "lead": lead,
                "sequence_audit": evidence,
            })

        dissociation = bool(
            atrial_regular
            and ventricular_regular
            and atrial_rate is not None
            and ventricular_rate is not None
            and atrial_rate > 1.15 * ventricular_rate
            and phase_range >= 0.20 * pp_med
            and phase_mad >= max(35.0, 0.10 * pp_med)
        )
        if dissociation:
            rows.append({
                "domain": "AV_CONDUCTION",
                "code": "COMPLETE_AV_BLOCK_COMPATIBLE",
                "score": 0.84,
                "evidence": [
                    "REGULAR_ATRIAL_SEQUENCE",
                    "REGULAR_SLOWER_VENTRICULAR_SEQUENCE",
                    "VARIABLE_AV_PHASE",
                ],
                "source_groups": ["ATRIAL_SEQUENCE", "VENTRICULAR_SEQUENCE", "AV_PHASE"],
                "independent_evidence_n": 3,
                "required_measurements": ["r_peaks"],
                "specialist_confirmed": False,
                "candidate_only": True,
                "lead": lead,
                "sequence_audit": evidence,
            })
    return rows


def build_high_recall_candidates(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate permissive candidates; this function never publishes diagnoses.

    Candidate gates intentionally use OR-shaped evidence so downstream fusion,
    rather than the first hard gate, determines whether a diagnosis is
    publishable. Thresholds are prospective engineering defaults and were not
    fitted to SPH.
    """
    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    fascicular = specialists.get("fascicular_conduction") or {}
    preexcitation = specialists.get("preexcitation") or {}
    qrs_morph = specialists.get("qrs_morphology") or {}
    rhythm = feature_graph.get("rhythm") or {}

    candidates: list[Dict[str, Any]] = []

    hr, hr_conf = _global(feature_graph, "heart_rate_bpm")
    qrs_ms, qrs_conf = _global(feature_graph, "qrs_ms")
    pr_ms, pr_conf = _global(feature_graph, "pr_ms")
    p_repro = bool(atrial.get("p_wave_reproducible"))
    coupling = float(_finite(atrial.get("rhythm_p_qrs_coupling_fraction")) or 0.0)

    scores = atrial_mech.get("compatibility_scores_not_probabilities") or {}
    af_compat = float(_finite(scores.get("AF")) or 0.0)
    flutter_compat = float(_finite(scores.get("FLUTTER_OR_AT")) or 0.0)
    rr_irregular = float(
        _finite((atrial_mech.get("aggregate_features") or {}).get("rr_irregularity_score"))
        or _finite(rhythm.get("rr_irregularity_score"))
        or 0.0
    )
    mechanism = str(atrial_mech.get("mechanism") or "")
    atrial_conf = float(_finite(atrial_mech.get("confidence")) or 0.0)

    af_evidence = []
    af_groups = []
    if mechanism == "AF_COMPATIBLE":
        af_evidence.append("ATRIAL_SPECIALIST_AF")
        af_groups.append("ATRIAL_SPECIALIST")
    if not p_repro:
        af_evidence.append("NO_REPRODUCIBLE_P")
        af_groups.append("P_WAVE")
    if rr_irregular >= 0.45:
        af_evidence.append("RR_IRREGULAR")
        af_groups.append("RR")
    if af_compat >= 0.40:
        af_evidence.append("AF_COMPATIBILITY_SIGNAL")
        af_groups.append("ATRIAL_SPECTRAL")
    if len(set(af_groups)) >= 2:
        af_score = max(
            atrial_conf if mechanism == "AF_COMPATIBLE" else 0.0,
            0.35 * rr_irregular + 0.30 * af_compat + 0.35 * (0.0 if p_repro else 1.0),
        )
        _append(candidates, domain="RHYTHM", code="AF_COMPATIBLE", score=af_score,
                evidence=af_evidence, source_groups=af_groups, required_measurements=["r_peaks"],
                specialist_confirmed=mechanism == "AF_COMPATIBLE")

    flutter_features = atrial_mech.get("aggregate_features") or {}
    periodicity = float(_finite(flutter_features.get("autocorrelation_periodicity")) or 0.0)
    df = _finite(flutter_features.get("dominant_frequency_hz"))
    flutter_evidence = []
    flutter_groups = []
    if mechanism == "FLUTTER_OR_AT_COMPATIBLE":
        flutter_evidence.append("ATRIAL_SPECIALIST_FLUTTER_AT")
        flutter_groups.append("ATRIAL_SPECIALIST")
    if flutter_compat >= 0.45:
        flutter_evidence.append("ORGANIZED_ATRIAL_COMPATIBILITY")
        flutter_groups.append("ATRIAL_SPECTRAL")
    if periodicity >= 0.40:
        flutter_evidence.append("ATRIAL_PERIODICITY")
        flutter_groups.append("ATRIAL_PERIODICITY")
    if df is not None and 3.0 <= df <= 7.0:
        flutter_evidence.append("ATRIAL_RATE_BAND_SUPPORT")
        flutter_groups.append("ATRIAL_RATE")
    if len(set(flutter_groups)) >= 2:
        flutter_score = max(
            atrial_conf if mechanism == "FLUTTER_OR_AT_COMPATIBLE" else 0.0,
            0.45 * flutter_compat + 0.35 * periodicity + 0.20 * (1.0 if df is not None and 3.0 <= df <= 7.0 else 0.0),
        )
        _append(candidates, domain="RHYTHM", code="FLUTTER_OR_AT_COMPATIBLE", score=flutter_score,
                evidence=flutter_evidence, source_groups=flutter_groups, required_measurements=["r_peaks"],
                specialist_confirmed=mechanism == "FLUTTER_OR_AT_COMPATIBLE")

    sinus_support = bool(atrial.get("sinus_compatible")) or bool(
        p_repro and coupling >= 0.55 and mechanism not in {"AF_COMPATIBLE", "FLUTTER_OR_AT_COMPATIBLE"}
    )
    if hr is not None and hr_conf >= 0.35 and sinus_support:
        if hr < 60.0:
            _append(candidates, domain="RATE", code="SINUS_BRADYCARDIA_COMPATIBLE",
                    score=min(0.95, 0.55 + 0.25 * hr_conf + 0.20 * min(coupling, 1.0)),
                    evidence=["VENTRICULAR_RATE_LT_60", "ORGANIZED_P_QRS_SUPPORT"],
                    source_groups=["RATE", "ATRIAL_ACTIVITY"], required_measurements=["r_peaks"],
                    specialist_confirmed=mechanism == "SINUS_COMPATIBLE")
        elif hr > 100.0:
            _append(candidates, domain="RATE", code="SINUS_TACHYCARDIA_COMPATIBLE",
                    score=min(0.95, 0.55 + 0.25 * hr_conf + 0.20 * min(coupling, 1.0)),
                    evidence=["VENTRICULAR_RATE_GT_100", "ORGANIZED_P_QRS_SUPPORT"],
                    source_groups=["RATE", "ATRIAL_ACTIVITY"], required_measurements=["r_peaks"],
                    specialist_confirmed=mechanism == "SINUS_COMPATIBLE")

    criteria = crosslead_conduction.get("criteria") or {}
    measurement_consensus = specialists.get("measurement_consensus") or {}
    qrs_120_relation = threshold_relation(measurement_consensus, "qrs_ms", 120.0)
    multilead_qrs_ge_120_rescue = bool(criteria.get("multilead_qrs_ge_120_rescue"))
    qrs_120_effective_relation = "ABOVE" if multilead_qrs_ge_120_rescue else qrs_120_relation
    pr_200_relation = threshold_relation(measurement_consensus, "pr_ms", 200.0)
    pr_120_relation = threshold_relation(measurement_consensus, "pr_ms", 120.0)
    rbbb_components = [
        ("QRS_GE_120MS", qrs_120_effective_relation == "ABOVE", "QRS_DURATION", 0.35),
        ("RIGHT_TERMINAL_R", bool(criteria.get("rbbb_right_terminal_r")), "RIGHT_PRECORDIAL_MORPHOLOGY", 0.35),
        ("LATERAL_TERMINAL_S", bool(criteria.get("rbbb_lateral_terminal_s_leads")), "LATERAL_MORPHOLOGY", 0.30),
    ]
    score = sum(w for _, yes, _, w in rbbb_components if yes)
    groups = [g for _, yes, g, _ in rbbb_components if yes]
    evidence = [e for e, yes, _, _ in rbbb_components if yes]
    if groups:
        if multilead_qrs_ge_120_rescue:
            evidence = sorted(set(evidence) | {"GE_4_MULTILEAD_QRS_GE_120MS"})
            groups = sorted(set(groups) | {"MULTILEAD_QRS_DURATION"})
        _append(candidates, domain="BUNDLE_BRANCH", code="RBBB_MORPHOLOGY_COMPATIBLE",
                score=score, evidence=evidence, source_groups=groups,
                required_measurements=[] if multilead_qrs_ge_120_rescue else ["qrs_ms"],
                boundary_requirements=[] if multilead_qrs_ge_120_rescue else [{"metric":"qrs_ms","threshold":120.0,"required_relation":"ABOVE","actual_relation":qrs_120_relation}],
                specialist_confirmed=any(str(x.get("code") or "") == "RBBB_MORPHOLOGY_COMPATIBLE" for x in crosslead_conduction.get("findings") or []))

    lbbb_components = [
        ("QRS_GE_120MS", qrs_120_effective_relation == "ABOVE", "QRS_DURATION", 0.30),
        ("V1_V2_NEGATIVE", bool(criteria.get("lbbb_v1_v2_negative")), "RIGHT_PRECORDIAL_MORPHOLOGY", 0.25),
        ("LATERAL_R_DOMINANT", bool(criteria.get("lbbb_key_lateral_r")), "LATERAL_POLARITY", 0.15),
        ("LATERAL_Q_ABSENT", bool(criteria.get("lbbb_key_lateral_absent_q")), "LATERAL_INITIAL_Q", 0.15),
        ("LATERAL_DELAY_OR_NOTCH", bool(criteria.get("lbbb_delayed_or_notched_lateral")), "LATERAL_ACTIVATION", 0.15),
    ]
    score = sum(w for _, yes, _, w in lbbb_components if yes)
    groups = [g for _, yes, g, _ in lbbb_components if yes]
    evidence = [e for e, yes, _, _ in lbbb_components if yes]
    if groups:
        if multilead_qrs_ge_120_rescue:
            evidence = sorted(set(evidence) | {"GE_4_MULTILEAD_QRS_GE_120MS"})
            groups = sorted(set(groups) | {"MULTILEAD_QRS_DURATION"})
        _append(candidates, domain="BUNDLE_BRANCH", code="LBBB_MORPHOLOGY_COMPATIBLE",
                score=score, evidence=evidence, source_groups=groups,
                required_measurements=[] if multilead_qrs_ge_120_rescue else ["qrs_ms"],
                boundary_requirements=[] if multilead_qrs_ge_120_rescue else [{"metric":"qrs_ms","threshold":120.0,"required_relation":"ABOVE","actual_relation":qrs_120_relation}],
                specialist_confirmed=any(str(x.get("code") or "") == "LBBB_MORPHOLOGY_COMPATIBLE" for x in crosslead_conduction.get("findings") or []))

    fcriteria = fascicular.get("criteria") or {}
    lafb_components = [
        ("LEFT_AXIS", bool(fcriteria.get("axis_minus45_to_minus90")), "AXIS", 0.40),
        ("POSITIVE_I_AVL", bool(fcriteria.get("positive_qrs_I")) and bool(fcriteria.get("positive_qrs_aVL")), "SUPERIOR_LIMB_MORPHOLOGY", 0.25),
        ("INFERIOR_S_DOMINANT", int(fcriteria.get("inferior_s_dominant_n") or 0) >= 2, "INFERIOR_LIMB_MORPHOLOGY", 0.25),
        ("SMALL_Q_SUPERIOR", bool(fcriteria.get("small_q_superior_support")), "INITIAL_Q", 0.10),
    ]
    score = sum(w for _, yes, _, w in lafb_components if yes)
    groups = [g for _, yes, g, _ in lafb_components if yes]
    evidence = [e for e, yes, _, _ in lafb_components if yes]
    if groups:
        _append(candidates, domain="FASCICULAR", code="LAFB_COMPATIBLE",
                score=score, evidence=evidence, source_groups=groups, required_measurements=[],
                specialist_confirmed=str(fascicular.get("classification") or "") == "LAFB_COMPATIBLE")

    lpfb_components = [
        ("RIGHT_AXIS", bool(fcriteria.get("axis_plus90_to_plus180")), "AXIS", 0.40),
        ("SUPERIOR_S_DOMINANT", int(fcriteria.get("superior_s_dominant_n") or 0) >= 2, "SUPERIOR_LIMB_MORPHOLOGY", 0.25),
        ("INFERIOR_R_DOMINANT", int(fcriteria.get("inferior_r_dominant_n") or 0) >= 2, "INFERIOR_LIMB_MORPHOLOGY", 0.25),
        ("SMALL_Q_INFERIOR", bool(fcriteria.get("small_q_inferior_support")), "INITIAL_Q", 0.05),
        ("QRS_LT_120MS", bool(fcriteria.get("qrs_lt_120ms")), "QRS_DURATION", 0.05),
    ]
    lpfb_score = sum(w for _, yes, _, w in lpfb_components if yes)
    lpfb_groups = [g for _, yes, g, _ in lpfb_components if yes]
    lpfb_evidence = [e for e, yes, _, _ in lpfb_components if yes]
    if lpfb_groups:
        _append(candidates, domain="FASCICULAR", code="LPFB_COMPATIBLE",
                score=lpfb_score, evidence=lpfb_evidence, source_groups=lpfb_groups,
                required_measurements=[],
                specialist_confirmed=str(fascicular.get("classification") or "") == "LPFB_COMPATIBLE")

    av = specialists.get("av_conduction") or {}
    av_code = str(av.get("classification") or "")
    if av_code not in {"", "AV_CONDUCTION_NOT_EVALUABLE", "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"}:
        _append(candidates, domain="AV_CONDUCTION", code=av_code,
                score=float(_finite(av.get("confidence")) or 0.0),
                evidence=list(av.get("basis") or ["AV_SPECIALIST"]),
                source_groups=["AV_SPECIALIST", "P_QRS_SEQUENCE"],
                required_measurements=["pr_ms"] if av_code == "FIRST_DEGREE_AV_DELAY_COMPATIBLE" else [],
                boundary_requirements=(
                    [{"metric":"pr_ms","threshold":200.0,"required_relation":"ABOVE","actual_relation":pr_200_relation}]
                    if av_code == "FIRST_DEGREE_AV_DELAY_COMPATIBLE"
                    else []
                ),
                specialist_confirmed=True)

    if (
        pr_ms is not None and pr_200_relation == "ABOVE" and pr_conf >= 0.40
        and p_repro and coupling >= 0.55
    ):
        _append(candidates, domain="AV_CONDUCTION", code="FIRST_DEGREE_AV_DELAY_COMPATIBLE",
                score=min(0.90, 0.50 + 0.25 * pr_conf + 0.25 * min(coupling, 1.0)),
                evidence=["PR_GT_200MS", "REPRODUCIBLE_P", "P_QRS_COUPLING"],
                source_groups=["PR_MEASUREMENT", "P_WAVE", "P_QRS_COUPLING"],
                required_measurements=["pr_ms"],
                boundary_requirements=[{"metric":"pr_ms","threshold":200.0,"required_relation":"ABOVE","actual_relation":pr_200_relation}],
                specialist_confirmed=av_code == "FIRST_DEGREE_AV_DELAY_COMPATIBLE")

    candidates.extend(_av_sequence_candidates(per_lead))

    pcrit = preexcitation.get("criteria") or {}
    delta_leads = list(pcrit.get("delta_slur_leads") or [])
    pre_components = [
        ("SHORT_PR", pr_120_relation == "BELOW", "PR_MEASUREMENT", 0.35),
        ("QRS_GE_100MS", bool(qrs_ms is not None and qrs_ms >= 100.0), "QRS_DURATION", 0.15),
        ("DELTA_SLUR_PRESENT", bool(delta_leads), "INITIAL_QRS_MORPHOLOGY", 0.35),
        ("REPRODUCIBLE_P", p_repro, "P_WAVE", 0.15),
    ]
    pre_score = sum(w for _, yes, _, w in pre_components if yes)
    pre_groups = [g for _, yes, g, _ in pre_components if yes]
    pre_ev = [e for e, yes, _, _ in pre_components if yes]
    multilead_preexcitation_rescue = bool(
        pcrit.get("multilead_short_pr_delta_rescue")
    )
    if multilead_preexcitation_rescue:
        # Concordant short PR + delta morphology in >=2 same leads is an
        # independent multilead substitute only when global PR/QRS consensus
        # is unavailable. It does not relax the adult PR/QRS thresholds.
        pre_score = max(pre_score, 0.80)
        pre_groups = sorted(set(pre_groups) | {
            "MULTILEAD_PR_MEASUREMENT",
            "MULTILEAD_INITIAL_QRS_MORPHOLOGY",
        })
        pre_ev = sorted(set(pre_ev) | {
            "GE_2_CONCORDANT_SHORT_PR_DELTA_LEADS",
        })
    if pre_groups:
        _append(candidates, domain="PREEXCITATION", code="VENTRICULAR_PREEXCITATION_COMPATIBLE",
                score=pre_score, evidence=pre_ev, source_groups=pre_groups,
                required_measurements=(
                    [] if multilead_preexcitation_rescue else ["pr_ms", "qrs_ms"]
                ),
                boundary_requirements=(
                    [] if multilead_preexcitation_rescue else
                    [{"metric":"pr_ms","threshold":120.0,"required_relation":"BELOW","actual_relation":pr_120_relation}]
                ),
                specialist_confirmed=str(preexcitation.get("classification") or "") == "VENTRICULAR_PREEXCITATION_COMPATIBLE")

    # De-duplicate by keeping the strongest candidate while preserving all
    # distinct evidence sources from repeated AV-lead analyses.
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in candidates:
        key = (str(row["domain"]), str(row["code"]))
        if key not in merged or float(row["score"]) > float(merged[key]["score"]):
            merged[key] = dict(row)
        else:
            current = merged[key]
            current["evidence"] = sorted(set(current.get("evidence") or []) | set(row.get("evidence") or []))
            current["source_groups"] = sorted(set(current.get("source_groups") or []) | set(row.get("source_groups") or []))
            current["independent_evidence_n"] = len(current["source_groups"])
            current["specialist_confirmed"] = bool(current.get("specialist_confirmed") or row.get("specialist_confirmed"))

    final = sorted(merged.values(), key=lambda x: (x["domain"], -float(x["score"]), x["code"]))
    return {
        "version": CANDIDATE_VERSION,
        "policy": "HIGH_RECALL_OR_GATE_ONLY; CANDIDATES_NEVER_BYPASS_DOMAIN_GATING_OR_EVIDENCE_FUSION",
        "threshold_note": "PROSPECTIVE_ENGINEERING_DEFAULTS_NOT_FITTED_TO_SPH",
        "candidates": final,
        "by_code": {row["code"]: row for row in final},
    }
