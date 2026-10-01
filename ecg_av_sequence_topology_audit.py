from __future__ import annotations

import argparse, json
from collections import Counter
from typing import Any
import numpy as np

from ecg_atrial_clean_r_mask_shadow_audit import _clean_selected_rhythm, _shadow_mask_inputs
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import CONTROL_N, DIAGNOSTIC_CASES_EACH, DIAGNOSTIC_GROUPS, FS, all_specs, canonical, make_signal

VERSION="MEDCALC_AV_SEQUENCE_TOPOLOGY_AUDIT_V1"

def topology(events:list[dict[str,Any]], r_samples:list[int], fs:int)->dict[str,Any]:
    p=np.asarray(sorted(set(int(round(float(e["time_ms"])*fs/1000.0)) for e in events if e.get("time_ms") is not None)),dtype=int)
    r=np.asarray(sorted(set(int(x) for x in r_samples)),dtype=int)
    if len(p)<4 or len(r)<3:
        return {"evaluable":False}
    pp=np.diff(p)*1000.0/fs; rr=np.diff(r)*1000.0/fs
    pp_med=float(np.median(pp)); rr_med=float(np.median(rr))
    pp_cv=float(np.std(pp,ddof=1)/np.mean(pp)) if len(pp)>=2 and np.mean(pp)>0 else None
    rr_cv=float(np.std(rr,ddof=1)/np.mean(rr)) if len(rr)>=2 and np.mean(rr)>0 else None
    mappings=[]
    for pi in p:
        future=r[r>pi]
        pr=float((future[0]-pi)*1000.0/fs) if len(future) else None
        conducted=bool(pr is not None and 80.0<=pr<=500.0)
        mappings.append((int(pi),conducted,pr if conducted else None))
    flags=[x[1] for x in mappings]
    prs=[float(x[2]) for x in mappings if x[1] and x[2] is not None]
    pr_mad=float(np.median(np.abs(np.asarray(prs)-np.median(prs)))) if len(prs)>=3 else None
    stable=bool(pr_mad is not None and pr_mad<=30.0)
    max_drop=run=0
    for f in flags:
        run=0 if f else run+1; max_drop=max(max_drop,run)
    ratio=float(len(p)/max(len(r),1))
    alternating=abs(sum(flags[::2])-sum(flags[1::2]))>=max(1,len(flags)//3)
    progressive=False
    for j,f in enumerate(flags):
        if f or j<3: continue
        prior=[x[2] for x in mappings[max(0,j-3):j] if x[1] and x[2] is not None]
        if len(prior)>=3 and all((b-a)>8.0 for a,b in zip(prior[:-1],prior[1:])): progressive=True
    phase=[]
    for ri in r:
        prior=p[p<ri]
        if len(prior):
            d=float((ri-prior[-1])*1000.0/fs)
            if d<=1.05*pp_med: phase.append(d)
    phase_mad=float(np.median(np.abs(np.asarray(phase)-np.median(phase)))) if len(phase)>=3 else None
    phase_range=float(max(phase)-min(phase)) if len(phase)>=3 else None
    diss=bool(phase_mad is not None and phase_range is not None and phase_mad>=max(50.0,.15*pp_med) and phase_range>=.30*pp_med)
    return {"evaluable":True,"p_n":len(p),"r_n":len(r),"pp_cv":pp_cv,"rr_cv":rr_cv,
      "conducted_n":sum(flags),"dropped_n":len(flags)-sum(flags),"max_consecutive_drop":max_drop,
      "p_r_ratio":ratio,"alternating":alternating,"stable_pr":stable,"progressive_pr":progressive,
      "phase_dissociation":diss}

def run_shard(si:int,sc:int)->dict[str,Any]:
    groups={k:Counter() for k in DIAGNOSTIC_GROUPS}; controls=Counter(); errors=[]
    for i,spec in enumerate(all_specs()):
        if i%sc!=si: continue
        try:
            ecg=canonical(spec,make_signal(spec)); a=analyze_canonical_ecg(ecg)
            shadow,_=_shadow_mask_inputs(ecg,a); ev=recover_crosslead_atrial_candidates(ecg,shadow)
            rhythm=_clean_selected_rhythm(ecg,a); fs=int(a.get("fs") or FS)
            t=topology(ev.get("unseeded_events") or [],rhythm.get("r_peaks_samples") or [],fs)
            dst=groups[str(spec["target"])] if spec.get("kind")=="TARGET" else controls
            dst["n"]+=1; dst["evaluable_n"]+=int(t.get("evaluable",False))
            if t.get("evaluable"):
                dst["organized_n"]+=int((t["pp_cv"] is not None and t["pp_cv"]<=.12))
                dst["dropped_any_n"]+=int(t["dropped_n"]>=1); dst["drop_ge2_n"]+=int(t["max_consecutive_drop"]>=2)
                dst["ratio_2to1_n"]+=int(1.75<=t["p_r_ratio"]<=2.25); dst["alternating_n"]+=int(t["alternating"])
                dst["stable_pr_n"]+=int(t["stable_pr"]); dst["progressive_pr_n"]+=int(t["progressive_pr"])
                dst["phase_dissociation_n"]+=int(t["phase_dissociation"])
                dst["organized_drop_ge2_n"]+=int(t["pp_cv"] is not None and t["pp_cv"]<=.12 and t["max_consecutive_drop"]>=2)
                dst["organized_2to1_alternating_n"]+=int(t["pp_cv"] is not None and t["pp_cv"]<=.12 and 1.75<=t["p_r_ratio"]<=2.25 and t["alternating"])
        except Exception as e: errors.append(f"{type(e).__name__}:{e}")
    return {"version":VERSION,"groups":{k:dict(v) for k,v in groups.items()},"controls":dict(controls),"errors":errors,
      "policy":"AUDIT_ONLY; SYNTHETIC_1000_ONLY; NO_CLINICAL_OUTPUT; NO_THRESHOLD_TUNING; NO_FAST_GATE; NO_FOLD9_OR_FINAL_DATA"}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--shard-index",type=int,default=0); ap.add_argument("--shard-count",type=int,default=1); ap.add_argument("--output")
    x=ap.parse_args(); out=run_shard(x.shard_index,x.shard_count); txt=json.dumps(out,indent=2,sort_keys=True)
    if x.output: open(x.output,"w").write(txt+"\n")
    print(txt)
