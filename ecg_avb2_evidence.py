from __future__ import annotations
from typing import Any
import numpy as np

VERSION="MEDCALC_AVB2_EVIDENCE_V1"

def build_avb2_evidence(atrial_events: list[dict[str,Any]], r_samples: list[int], fs: int) -> dict[str,Any]:
    """Mechanistic AVB2 evidence only. Does not make a diagnostic claim."""
    p=np.asarray(sorted({int(round(float(e["time_ms"])*fs/1000.0)) for e in atrial_events if e.get("time_ms") is not None}),dtype=int)
    r=np.asarray(sorted({int(x) for x in r_samples}),dtype=int)
    out={"version":VERSION,"evaluable":False,"compatible":False,"diagnostic_claim_allowed":False,
         "basis":[],"source":"RECOVERED_UNSEEDED_ATRIAL_SEQUENCE"}
    if fs<=0 or len(p)<4 or len(r)<3: return out
    pp=np.diff(p)*1000.0/fs
    pp_cv=float(np.std(pp,ddof=1)/np.mean(pp)) if len(pp)>=2 and np.mean(pp)>0 else None
    organized=bool(pp_cv is not None and pp_cv<=.12)
    dropped=[]; conducted=[]; phases=[]
    for pi in p:
        future=r[r>pi]
        pr=(future[0]-pi)*1000.0/fs if len(future) else None
        is_conducted=bool(pr is not None and 80.0<=pr<=500.0)
        (conducted if is_conducted else dropped).append(int(pi))
        left=r[r<pi]; right=r[r>pi]
        if not is_conducted and len(left) and len(right) and right[0]>left[-1]:
            phases.append(float((pi-left[-1])/(right[0]-left[-1])))
    flags=[int(x) not in set(dropped) for x in p]
    run=maxrun=0
    for flag in flags:
        if flag: run=0
        else: run+=1; maxrun=max(maxrun,run)
    mid_majority=bool(len(phases)>=2 and sum(.25<=x<=.75 for x in phases)/len(phases)>=.5)
    compatible=bool(organized and maxrun>=2 and mid_majority)
    out.update({"evaluable":True,"compatible":compatible,"atrial_sequence_regular":organized,
                "p_count":len(p),"qrs_count":len(r),"nonconducted_p_n":len(dropped),
                "max_consecutive_nonconducted_p":maxrun,"dropped_mid_rr_majority":mid_majority,
                "pp_cv":round(pp_cv,6) if pp_cv is not None else None,
                "basis":["REGULAR_RECOVERED_P_SEQUENCE","GE_2_CONSECUTIVE_NONCONDUCTED_P","DROPPED_P_MID_RR_GEOMETRY"] if compatible else []})
    return out
