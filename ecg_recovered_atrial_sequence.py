from __future__ import annotations
import copy
from typing import Any
import numpy as np
from ecg_independent_atrial_evidence import PREFERRED_LEADS, recover_crosslead_atrial_candidates
from ecg_r_candidate_filter import filter_relative_r_amplitude, RELATIVE_R_AMPLITUDE_MIN

VERSION="MEDCALC_RECOVERED_ATRIAL_SEQUENCE_V1"

def clean_r_measurement_inputs(canonical_ecg: dict[str,Any], analysis: dict[str,Any]) -> tuple[dict[str,dict[str,Any]],dict[str,Any]]:
    original=analysis.get("leads") or {}; cleaned=copy.deepcopy(original)
    rhythm=analysis.get("rhythm") or {}; lead=rhythm.get("lead")
    audit={"evaluable":False,"changed":False,"raw_r_n":0,"kept_r_n":0}
    if not lead: return cleaned,audit
    item=original.get(lead) or {}; source=(canonical_ecg.get("leads") or {}).get(lead) or {}
    fs=int(item.get("fs") or source.get("fs") or analysis.get("fs") or 500)
    raw=sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    signal=np.asarray(source.get("signal_mv") or [],dtype=float)
    audit.update(raw_r_n=len(raw),kept_r_n=len(raw))
    if fs<=0 or len(raw)<3 or signal.size==0: return cleaned,audit
    ok,kept=filter_relative_r_amplitude(raw,signal,fs,RELATIVE_R_AMPLITUDE_MIN)
    audit.update(evaluable=bool(ok),kept_r_n=len(kept))
    if not ok or len(kept)<3 or len(kept)>=len(raw): return cleaned,audit
    kept_ms=[1000.0*v/fs for v in kept]
    for name in PREFERRED_LEADS:
        measured=cleaned.get(name) or {}; src=(canonical_ecg.get("leads") or {}).get(name) or {}
        if not measured.get("evaluable"): continue
        lfs=int(measured.get("fs") or src.get("fs") or analysis.get("fs") or 500)
        if lfs<=0: continue
        projected=sorted(set(int(round(t*lfs/1000.0)) for t in kept_ms))
        if len(projected)<3: continue
        measured["r_peaks_samples"]=projected; measured["r_count"]=len(projected); cleaned[name]=measured
    audit["changed"]=True
    return cleaned,audit

def clean_selected_rhythm(canonical_ecg: dict[str,Any], analysis: dict[str,Any]) -> dict[str,Any]:
    rhythm=copy.deepcopy(analysis.get("rhythm") or {}); lead=rhythm.get("lead")
    if not lead: return rhythm
    item=(analysis.get("leads") or {}).get(lead) or {}; source=(canonical_ecg.get("leads") or {}).get(lead) or {}
    fs=int(item.get("fs") or source.get("fs") or analysis.get("fs") or 500)
    raw=sorted(set(int(v) for v in (rhythm.get("r_peaks_samples") or [])))
    signal=np.asarray(source.get("signal_mv") or [],dtype=float)
    if fs<=0 or len(raw)<3 or signal.size==0: return rhythm
    ok,kept=filter_relative_r_amplitude(raw,signal,fs,RELATIVE_R_AMPLITUDE_MIN)
    if not ok or len(kept)<3 or len(kept)>=len(raw): return rhythm
    rhythm["r_peaks_samples"]=kept; rhythm["r_count"]=len(kept)
    return rhythm

def recover_unseeded_atrial_sequence(canonical_ecg: dict[str,Any], analysis: dict[str,Any]) -> dict[str,Any]:
    cleaned,audit=clean_r_measurement_inputs(canonical_ecg,analysis)
    evidence=recover_crosslead_atrial_candidates(canonical_ecg,cleaned)
    return {"version":VERSION,"mask":audit,"unseeded_events":list(evidence.get("unseeded_events") or []),
            "unseeded_event_n":int(evidence.get("unseeded_event_n") or 0),
            "unseeded_organized":bool(evidence.get("unseeded_organized"))}
