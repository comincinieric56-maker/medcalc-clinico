from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Any
import numpy as np

from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_recovered_atrial_sequence import clean_r_measurement_inputs, clean_selected_rhythm
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import FS, all_specs, canonical, make_signal

VERSION="MEDCALC_AVB2_FULL_SEQUENCE_AUDIT_V1"

def combine_observed_unseeded(evidence:dict[str,Any], coincidence_ms:float=36.0)->list[dict[str,float]]:
    times=[]
    for row in (evidence.get("observed_consensus") or {}).get("events") or []:
        if row.get("time_ms") is not None: times.append(float(row["time_ms"]))
    for row in evidence.get("unseeded_events") or []:
        if row.get("time_ms") is not None: times.append(float(row["time_ms"]))
    times.sort()
    out=[]
    for t in times:
        if out and abs(t-out[-1])<=coincidence_ms:
            out[-1]=float(np.median([out[-1],t]))
        else:
            out.append(t)
    return [{"time_ms":round(t,6)} for t in out]

def topology(events:list[dict[str,float]], r_samples:list[int], fs:int)->dict[str,Any]:
    p=np.asarray(sorted({int(round(float(e["time_ms"])*fs/1000.0)) for e in events}),dtype=int)
    r=np.asarray(sorted({int(x) for x in r_samples}),dtype=int)
    if fs<=0 or len(p)<4 or len(r)<3: return {"evaluable":False,"p_n":len(p),"r_n":len(r)}
    pp=np.diff(p)*1000.0/fs; rr=np.diff(r)*1000.0/fs
    pp_med=float(np.median(pp)); rr_med=float(np.median(rr))
    pp_cv=float(np.std(pp,ddof=1)/np.mean(pp)) if len(pp)>=2 and np.mean(pp)>0 else None
    rr_cv=float(np.std(rr,ddof=1)/np.mean(rr)) if len(rr)>=2 and np.mean(rr)>0 else None
    mappings=[{"p":int(x),"conducted":False,"pr_ms":None} for x in p]
    used=set()
    lo=int(np.floor(.080*fs)); hi=int(np.ceil(.500*fs))
    for ri in r:
        idx=np.where((p<int(ri)) & ((int(ri)-p)>=lo) & ((int(ri)-p)<=hi))[0]
        chosen=None
        for q in idx[::-1]:
            if int(q) not in used:
                chosen=int(q); break
        if chosen is not None:
            used.add(chosen); mappings[chosen]["conducted"]=True
            mappings[chosen]["pr_ms"]=float((int(ri)-int(p[chosen]))*1000.0/fs)
    flags=[bool(x["conducted"]) for x in mappings]
    prs=[float(x["pr_ms"]) for x in mappings if x["conducted"] and x["pr_ms"] is not None]
    pr_mad=float(np.median(np.abs(np.asarray(prs)-np.median(prs)))) if len(prs)>=3 else None
    maxdrop=run=0
    for flag in flags:
        if flag: run=0
        else: run+=1; maxdrop=max(maxdrop,run)
    phase=[]
    for ri in r:
        prior=p[p<int(ri)]
        if len(prior):
            d=float((int(ri)-int(prior[-1]))*1000.0/fs)
            if d<=1.05*pp_med: phase.append(d)
    phase_mad=float(np.median(np.abs(np.asarray(phase)-np.median(phase)))) if len(phase)>=3 else None
    phase_range=float(max(phase)-min(phase)) if len(phase)>=3 else None
    diss=bool(phase_mad is not None and phase_range is not None and phase_mad>=max(50.0,.15*pp_med) and phase_range>=.30*pp_med)
    return {
      "evaluable":True,"p_n":len(p),"r_n":len(r),"p_r_ratio":float(len(p)/max(len(r),1)),
      "pp_median_ms":pp_med,"pp_cv":pp_cv,"rr_median_ms":rr_med,"rr_cv":rr_cv,
      "conducted_n":sum(flags),"dropped_n":len(flags)-sum(flags),"max_consecutive_drop":maxdrop,
      "pr_median_ms":float(np.median(prs)) if prs else None,"pr_mad_ms":pr_mad,
      "stable_pr":bool(pr_mad is not None and pr_mad<=30.0),
      "phase_mad_ms":phase_mad,"phase_range_ms":phase_range,"phase_dissociation":diss,
    }

def run(target:str)->dict[str,Any]:
    rows=[]
    for spec in all_specs():
        if spec.get("target")!=target: continue
        ecg=canonical(spec,make_signal(spec)); a=analyze_canonical_ecg(ecg)
        shadow=dict(a.get("avb2_evidence_shadow") or {})
        cleaned,_=clean_r_measurement_inputs(ecg,a)
        ev=recover_crosslead_atrial_candidates(ecg,cleaned)
        rhythm=clean_selected_rhythm(ecg,a); fs=int(a.get("fs") or FS)
        full=combine_observed_unseeded(ev)
        t=topology(full,rhythm.get("r_peaks_samples") or [],fs)
        rows.append({"case_id":spec["case_id"],"shadow_compatible":bool(shadow.get("compatible")),
                     "observed_n":int((ev.get("observed_consensus") or {}).get("event_n") or 0),
                     "unseeded_n":int(ev.get("unseeded_event_n") or 0),"full_event_n":len(full),**t})
    return {"version":VERSION,"target":target,"n":len(rows),"rows":rows,
            "policy":"SYNTHETIC_AVB2_AVB3_ONLY; CONDITION_ON_EXISTING_SHADOW; OBSERVED_PLUS_UNSEEDED_ONLY; EXISTING_36MS_DEDUP; EXISTING_80_500MS_PR; NO_FAST; NO_FOLD9; NO_THRESHOLD_PROMOTION"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--target",choices=["AVB2","AVB3"],required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); out=run(a.target); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__": main()
