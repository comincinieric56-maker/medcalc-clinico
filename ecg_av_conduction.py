from __future__ import annotations

from typing import Any, Dict

import numpy as np


AV_VERSION = "MEDCALC_AV_CONDUCTION_V1"
PREFERRED = ("II","V1","aVF","I")


def _choose_lead(per_lead: Dict[str, Dict[str, Any]]) -> str | None:
    for lead in PREFERRED:
        item = per_lead.get(lead) or {}
        atrial = item.get("atrial_activity") or {}
        p = item.get("raw_p_peaks_samples") or []
        if (
            item.get("evaluable")
            and bool(atrial.get("p_wave_reproducible"))
            and len(p) >= 4
            and int(item.get("r_count") or 0) >= 3
        ):
            return lead
    return None


def analyze_av_conduction(
    per_lead: Dict[str, Dict[str, Any]],
    global_atrial: Dict[str, Any],
) -> Dict[str, Any]:
    lead = _choose_lead(per_lead)
    if lead is None or not bool(global_atrial.get("p_wave_reproducible")):
        return {
            "version": AV_VERSION,
            "evaluable": False,
            "classification": "AV_CONDUCTION_NOT_EVALUABLE",
            "reason": "NO_REPRODUCIBLE_P_SEQUENCE",
            "diagnostic_claim_allowed": False,
        }

    item = per_lead[lead]
    fs = int(item.get("fs") or 500)
    p = np.asarray(item.get("raw_p_peaks_samples") or [], dtype=int)
    r = np.asarray(item.get("r_peaks_samples") or [], dtype=int)
    if p.size < 4 or r.size < 3:
        return {
            "version": AV_VERSION,
            "evaluable": False,
            "classification": "AV_CONDUCTION_NOT_EVALUABLE",
            "reason": "INSUFFICIENT_P_OR_QRS",
            "diagnostic_claim_allowed": False,
        }

    pp = np.diff(p) * 1000.0/fs
    pp_med = float(np.median(pp)) if pp.size else None
    pp_cv = float(np.std(pp,ddof=1)/np.mean(pp)) if pp.size >= 2 and np.mean(pp)>0 else None
    atrial_regular = bool(pp_cv is None or pp_cv <= 0.12)

    mappings = []
    for pi in p:
        candidates = r[r > pi]
        if candidates.size == 0:
            mappings.append({"p":int(pi),"conducted":False,"pr_ms":None})
            continue
        ri = int(candidates[0])
        pr = (ri-int(pi))*1000.0/fs
        if 80.0 <= pr <= 500.0:
            mappings.append({"p":int(pi),"conducted":True,"r":ri,"pr_ms":float(pr)})
        else:
            mappings.append({"p":int(pi),"conducted":False,"pr_ms":None})

    # Prevent the same QRS from being assigned to multiple P waves.
    seen = set()
    for m in mappings:
        if not m.get("conducted"):
            continue
        rr = int(m["r"])
        if rr in seen:
            m["conducted"] = False
            m["pr_ms"] = None
        else:
            seen.add(rr)

    conducted = [m for m in mappings if m.get("conducted")]
    dropped = [m for m in mappings if not m.get("conducted")]
    pr = np.asarray([m["pr_ms"] for m in conducted],dtype=float)
    pr_med = float(np.median(pr)) if pr.size else None
    pr_mad = float(np.median(np.abs(pr-np.median(pr)))) if pr.size else None
    stable_pr = bool(pr.size >= 3 and pr_mad is not None and pr_mad <= 30.0)
    one_to_one = bool(len(dropped)==0 and len(conducted)>=3 and len(seen)==len(conducted))

    classification = "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"
    confidence = 0.70 if one_to_one else 0.45
    basis = []

    if one_to_one and stable_pr and pr_med is not None and pr_med > 200.0:
        classification = "FIRST_DEGREE_AV_DELAY_COMPATIBLE"
        confidence = 0.90
        basis = ["1_TO_1_P_QRS","PR_MEDIAN_GT_200MS","PR_STABLE"]
    elif atrial_regular and len(dropped) >= 1 and len(conducted) >= 3:
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
        if 1.75 <= p_to_r_ratio <= 2.25 and abs(sum(flags[::2]) - sum(flags[1::2])) >= max(1,len(flags)//3):
            classification = "TWO_TO_ONE_AV_BLOCK_COMPATIBLE"
            confidence = 0.82
            basis = ["REGULAR_P_SEQUENCE","APPROX_2_TO_1_P_QRS_RATIO"]
        elif max_consecutive_drop >= 2:
            classification = "HIGH_GRADE_AV_BLOCK_COMPATIBLE"
            confidence = 0.88
            basis = ["REGULAR_P_SEQUENCE","GE_2_CONSECUTIVE_NONCONDUCTED_P"]
        elif stable_pr:
            classification = "MOBITZ_II_COMPATIBLE"
            confidence = 0.78
            basis = ["REGULAR_P_SEQUENCE","DROPPED_P","STABLE_CONDUCTED_PR"]
        elif pr.size >= 3 and np.all(np.diff(pr[-3:]) > 8.0):
            classification = "MOBITZ_I_WENCKEBACH_COMPATIBLE"
            confidence = 0.76
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
        "pp_median_ms": round(pp_med,3) if pp_med is not None else None,
        "pp_cv": round(pp_cv,6) if pp_cv is not None else None,
        "pr_median_ms": round(pr_med,3) if pr_med is not None else None,
        "pr_mad_ms": round(pr_mad,3) if pr_mad is not None else None,
        "one_to_one": one_to_one,
        "stable_pr": stable_pr,
        "basis": basis,
        "diagnostic_claim_allowed": False,
        "source": "REPRODUCIBLE_P_SEQUENCE_TO_QRS_MAPPING",
    }
