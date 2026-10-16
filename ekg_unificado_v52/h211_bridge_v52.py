#!/usr/bin/env python3
"""Research-only bridge: original 500-Hz WFDB signals -> frozen H211 (6x2).

H211 expects a sequential 6x2 paper-layout representation, not
12 simultaneous 10-second signals. The bridge supplies original
samples: limb leads from seconds 0..5, precordial leads from
seconds 5..10 and a 10-second continuous II_LONG rhythm strip.
This does NOT re-train or modify the H211 model or V51 checkpoint.
The domain adaptation has not been externally validated.
"""
from __future__ import annotations

import math
import traceback
import numpy as np

LEADS = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")
CLINICAL_STATE = "CLINICAL_DEPLOYMENT_BLOCKED"


def to_h211_6x2(ecg500):
    a = np.asarray(ecg500, dtype=np.float64)
    if a.shape != (5000,12) or not np.isfinite(a).all():
        raise ValueError("Exactly 5000 x 12 finite 500-Hz mV samples required")
    if np.max(np.abs(a)) > 30:
        raise ValueError("H211 input outside plausible mV range")
    result = {}
    for i, lead in enumerate(LEADS):
        panel = a[:2500,i] if i < 6 else a[2500:,i]
        result[lead] = {"signal":panel.copy(),"fs":500.0}
    result["II_LONG"] = {"signal":a[:,1].copy(),"fs":500.0}
    return result


def run_frozen_h211(ecg500, source_path, importer):
    if ecg500 is None:
        return {"status":"NOT_ASSESSABLE","reason":"500-Hz original required"}
    if not source_path.is_file():
        return {"status":"NOT_AVAILABLE","reason":"Missing original H211 source"}
    try:
        signals = to_h211_6x2(ecg500)
        h211 = importer(source_path, "_medcalc_v52_h211_original")
        normalized = h211.normalize_signals(signals)
        if len(normalized) != 13:
            raise RuntimeError("H211 normalized fewer than 13 expected lead channels")
        full = h211.measure_dynamic(signals)
        if not isinstance(full, dict):
            raise TypeError("Original H211 did not return a dict")
        if full.get("reference_leakage_guard") != "PASS_NO_12SL_UNIG_INPUT":
            raise RuntimeError("H211 provenance/leakage guard failed")
        qc = str(full.get("measurement_qc", "FAIL"))
        if qc not in ("PASS","PARTIAL","FAIL"):
            raise RuntimeError("Unrecognized H211 QC state")
        raw = h211.compact_result(full)
        fields = ("HR_bpm","RR_representative_ms","PR_ms","QRS_ms",
                  "QT_ms","QTcF_ms","JT_ms","AXIS_deg")
        numerical = {}
        for field in fields:
            try:
                v=raw.get(field)
                x=float(v) if v is not None else None
                numerical[field]=x if x is None or math.isfinite(x) else None
            except (TypeError,ValueError):
                numerical[field]=None
        if qc == "FAIL":
            numerical = {k:None for k in fields}
        if full.get("rhythm_regularity") == "IRREGULAR":
            numerical["QTcF_ms"]=None
        return {"status":qc,
                "engine_version":full.get("engine_version"),
                "measurements":numerical,
                "rhythm_regularity":full.get("rhythm_regularity"),
                "support":full.get("support"),
                "reference_leakage_guard":"PASS_NO_12SL_UNIG_INPUT",
                "input_adapter":"6X2_PAPER_LAYOUT_LIMB_0_5S_CHEST_5_10S_II_LONG_0_10S",
                "input_is_real_500hz_simultaneous_ecg":True,
                "layout_adapter_externally_validated":False,
                "original_engine_unchanged":True,
                "clinical_state":CLINICAL_STATE}
    except Exception as exc:
        return {"status":"FAILED_CLOSED",
                "reason":f"{type(exc).__name__}: {exc}",
                "traceback_tail":traceback.format_exc()[-1600:],
                "clinical_state":CLINICAL_STATE}


def combine_h211_v51(h211,v51):
    """Do not average two interval estimates or hide disagreement.

    Difference tolerances are engineering QA triggers, never diagnostic cutoffs.
    Individual source measurements remain in the research report.
    """
    checks={}
    discrepancies=[]
    primary={}
    mapping={
        "heart_rate_bpm":("HR_bpm",v51.get("heart_rate_bpm"),8.0),
        "rr_median_ms":("RR_representative_ms",v51.get("rr_median_ms"),100.0),
        "pr_ms":("PR_ms",v51.get("pr_ms"),40.0),
        "qrs_ms":("QRS_ms",v51.get("qrs_ms"),30.0),
        "qt_ms":("QT_ms",v51.get("qt_ms"),50.0),
        "qtc_fridericia_ms":("QTcF_ms",(v51.get("qtc") or {}).get("fridericia_ms"),50.0),
    }
    source=(h211.get("measurements") or {})
    accepted=h211.get("status") in ("PASS","PARTIAL")
    for name,(oldkey,b,tol) in mapping.items():
        a=source.get(oldkey)
        comparable=a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
        difference=round(abs(float(a)-float(b)),2) if comparable else None
        mismatch=bool(comparable and difference > tol)
        if mismatch:
            discrepancies.append(name)
        state="DISCORDANT" if mismatch else "NO_FLAGGED_DISCORDANCE" if comparable else "NOT_COMPARABLE"
        primary[name]=float(a) if accepted and a is not None and not mismatch else None
        checks[name]={"h211":a,"v51":b,"difference":difference,
                      "qa_tolerance":tol,"comparison":state,
                      "used_h211_value":primary[name] is not None}
    if h211.get("rhythm_regularity") == "IRREGULAR":
        primary["qtc_fridericia_ms"]=None
        checks["qtc_fridericia_ms"]["comparison"]="SUPPRESSED_IRREGULAR_RHYTHM"
    primary["axis_deg"]=source.get("AXIS_deg") if accepted else None
    primary["jt_ms"]=source.get("JT_ms") if accepted else None
    if not accepted:
        status="H211_FAILED_OR_UNAVAILABLE"
    elif discrepancies:
        status="DISCORDANT_REVIEW_REQUIRED"
    else:
        status="DUAL_ENGINE_RESEARCH"
    return {"status":status,"primary_source":"H211_WITH_QC_AND_CROSS_CHECK",
            "unified":primary,"comparisons":checks,
            "discrepancies":discrepancies,
            "h211":h211,"v51_mask_estimates":v51,
            "notes":["Original H211 6x2 layout adapter requires independent validation",
                     "Intervals are research-only, not clinical measurements",
                     "Missing comparison is not evidence of agreement"],
            "clinical_state":CLINICAL_STATE}


if __name__=="__main__":
    x=(np.arange(5000,dtype=np.float64)/1000.0)[:,None]*np.ones((1,12))
    y=to_h211_6x2(x)
    assert len(y)==13 and y["II_LONG"]["signal"].shape==(5000,)
    assert y["V6"]["signal"][0]==2.5
    print("H211 adapter synthetic structure PASS; real clinical validation pending")
