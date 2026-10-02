from __future__ import annotations

import argparse, json
from collections import Counter
from typing import Any
import numpy as np

from ecg_recovered_atrial_sequence import clean_selected_rhythm, recover_unseeded_atrial_sequence
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
    rr_phase=[]
    dropped_rr_phase=[]
    for pi,conducted,_ in mappings:
        left=r[r<pi]; right=r[r>pi]
        if len(left) and len(right) and right[0]>left[-1]:
            phase=float((pi-left[-1])/(right[0]-left[-1]))
            rr_phase.append(phase)
            if not conducted: dropped_rr_phase.append(phase)
    return {"evaluable":True,"p_n":len(p),"r_n":len(r),"pp_cv":pp_cv,"rr_cv":rr_cv,
      "conducted_n":sum(flags),"dropped_n":len(flags)-sum(flags),"max_consecutive_drop":max_drop,
      "p_r_ratio":ratio,"alternating":alternating,"stable_pr":stable,"progressive_pr":progressive,
      "phase_dissociation":diss,
      "dropped_rr_phase_mid_n":sum(.25<=x<=.75 for x in dropped_rr_phase),
      "dropped_rr_phase_n":len(dropped_rr_phase),
      "dropped_rr_phase_median":float(np.median(dropped_rr_phase)) if dropped_rr_phase else None,
      "pr_median_ms":float(np.median(prs)) if prs else None,
      "pr_mad_ms":pr_mad}

def run_shard(si:int,sc:int)->dict[str,Any]:
    groups={k:Counter() for k in DIAGNOSTIC_GROUPS}; controls=Counter(); errors=[]
    for i,spec in enumerate(all_specs()):
        if i%sc!=si: continue
        try:
            ecg=canonical(spec,make_signal(spec)); a=analyze_canonical_ecg(ecg)
            seq=recover_unseeded_atrial_sequence(ecg,a)
            rhythm=clean_selected_rhythm(ecg,a); fs=int(a.get("fs") or FS)
            t=topology(seq.get("unseeded_events") or [],rhythm.get("r_peaks_samples") or [],fs)
            dst=groups[str(spec["target"])] if spec.get("kind")=="TARGET" else controls
            dst["n"]+=1; dst["evaluable_n"]+=int(t.get("evaluable",False))
            shadow=a.get("avb2_evidence_shadow") or {}
            dst["pipeline_shadow_evaluable_n"]+=int(bool(shadow.get("evaluable")))
            dst["pipeline_shadow_compatible_n"]+=int(bool(shadow.get("compatible")))
            dst["pipeline_shadow_error_n"]+=int(shadow.get("shadow_status")=="SHADOW_ERROR")
            dst["pipeline_shadow_claim_allowed_n"]+=int(bool(shadow.get("diagnostic_claim_allowed")))
            if t.get("evaluable"):
                dst["organized_n"]+=int((t["pp_cv"] is not None and t["pp_cv"]<=.12))
                dst["dropped_any_n"]+=int(t["dropped_n"]>=1); dst["drop_ge2_n"]+=int(t["max_consecutive_drop"]>=2)
                dst["ratio_2to1_n"]+=int(1.75<=t["p_r_ratio"]<=2.25); dst["alternating_n"]+=int(t["alternating"])
                dst["stable_pr_n"]+=int(t["stable_pr"]); dst["progressive_pr_n"]+=int(t["progressive_pr"])
                dst["phase_dissociation_n"]+=int(t["phase_dissociation"])
                midfrac=(t["dropped_rr_phase_mid_n"]/t["dropped_rr_phase_n"]) if t["dropped_rr_phase_n"] else 0.0
                dst["dropped_midrr_majority_n"]+=int(t["dropped_rr_phase_n"]>=2 and midfrac>=.5)
                dst["organized_dropped_midrr_majority_n"]+=int(t["pp_cv"] is not None and t["pp_cv"]<=.12 and t["dropped_rr_phase_n"]>=2 and midfrac>=.5)
                dst["pr_measurable_n"]+=int(t["pr_median_ms"] is not None)
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
