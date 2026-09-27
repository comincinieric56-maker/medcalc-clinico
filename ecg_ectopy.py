from __future__ import annotations

from typing import Any, Dict

import numpy as np


ECTOPY_VERSION = "MEDCALC_ECTOPY_ENGINE_V1"


def analyze_ectopy(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
    rhythm: Dict[str, Any],
) -> Dict[str, Any]:
    lead = str(rhythm.get("lead") or "")
    item = per_lead.get(lead) or {}
    beats = item.get("beats") or []
    rr = np.asarray(item.get("rr_ms") or [], dtype=float)
    if len(beats) < 5 or rr.size < 4:
        return {
            "version": ECTOPY_VERSION,
            "evaluable": False,
            "reason": "LT_5_BEATS",
            "lead": lead,
        }

    widths = np.asarray([
        float(b.get("qrs_ms")) if b.get("qrs_ms") is not None else np.nan
        for b in beats
    ], dtype=float)
    r_amps = np.asarray([
        float(b.get("r_amp_mv")) if b.get("r_amp_mv") is not None else np.nan
        for b in beats
    ], dtype=float)
    med_rr = float(np.nanmedian(rr))
    med_w = float(np.nanmedian(widths)) if np.isfinite(widths).any() else None
    med_r = float(np.nanmedian(r_amps)) if np.isfinite(r_amps).any() else None

    pvc = 0
    pac = 0
    premature = 0
    rows = []
    usable_n = min(len(beats)-1, len(rr))
    for i in range(1, usable_n):
        prev_rr = float(rr[i-1])
        next_rr = float(rr[i]) if i < len(rr) else None
        if prev_rr >= 0.82 * med_rr:
            continue
        premature += 1
        width = widths[i] if i < len(widths) and np.isfinite(widths[i]) else np.nan
        amp = r_amps[i] if i < len(r_amps) and np.isfinite(r_amps[i]) else np.nan
        wide_outlier = bool(
            np.isfinite(width)
            and med_w is not None
            and (width >= 120.0 or width >= med_w + 25.0)
        )
        amp_outlier = bool(
            np.isfinite(amp)
            and med_r is not None
            and abs(amp - med_r) >= max(0.30, 0.60*abs(med_r))
        )
        compensatory = bool(next_rr is not None and (prev_rr + next_rr) >= 1.75 * med_rr)
        if wide_outlier and (amp_outlier or compensatory):
            pvc += 1
            kind = "PVC_COMPATIBLE"
        elif not wide_outlier:
            pac += 1
            kind = "PAC_OR_NARROW_PREMATURE_BEAT"
        else:
            kind = "PREMATURE_BEAT_UNCLASSIFIED"
        rows.append({
            "beat_index": i,
            "kind": kind,
            "premature_rr_ms": round(prev_rr,3),
            "next_rr_ms": round(next_rr,3) if next_rr is not None else None,
            "qrs_ms": round(float(width),3) if np.isfinite(width) else None,
            "wide_outlier": wide_outlier,
            "amplitude_outlier": amp_outlier,
            "compensatory_pause": compensatory,
        })

    beat_count = max(len(beats),1)
    burden = premature / beat_count
    irregularity_explained = bool(
        premature >= 2 and burden >= 0.10
    )
    return {
        "version": ECTOPY_VERSION,
        "evaluable": True,
        "lead": lead,
        "premature_beat_n": int(premature),
        "pvc_compatible_n": int(pvc),
        "pac_or_narrow_premature_n": int(pac),
        "premature_burden": round(float(burden),6),
        "irregularity_may_be_ectopy_driven": irregularity_explained,
        "events": rows,
        "diagnostic_claim_allowed": False,
        "source": "RR_TIMING_PLUS_BEAT_QRS_MORPHOLOGY",
    }
