from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from ecg_recovered_atrial_sequence import clean_selected_rhythm, recover_unseeded_atrial_sequence
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import FS, all_specs, canonical, make_signal

VERSION="MEDCALC_AVB2_AVB3_OVERLAP_AUDIT_V1"

def direct_pr_features(events, r_samples, fs):
    p=np.asarray(sorted({int(round(float(e["time_ms"])*fs/1000.0)) for e in events if e.get("time_ms") is not None}),dtype=int)
    r=np.asarray(sorted({int(x) for x in r_samples}),dtype=int)
    prs=[]
    for pi in p:
        future=r[r>pi]
        if not len(future):
            continue
        pr=float((future[0]-pi)*1000.0/fs)
        if 80.0 <= pr <= 500.0:
            prs.append(pr)
    pr_mad=float(np.median(np.abs(np.asarray(prs)-np.median(prs)))) if len(prs)>=3 else None
    return {
        "p_qrs_ratio": float(len(p)/max(len(r),1)),
        "direct_pr_n": len(prs),
        "direct_pr_median_ms": float(np.median(prs)) if prs else None,
        "direct_pr_mad_ms": pr_mad,
        "direct_stable_pr": bool(pr_mad is not None and pr_mad<=30.0),
    }

def run(target):
    rows=[]
    for spec in all_specs():
        if spec.get("target")!=target:
            continue
        ecg=canonical(spec,make_signal(spec))
        analysis=analyze_canonical_ecg(ecg)
        shadow=dict(analysis.get("avb2_evidence_shadow") or {})
        seq=recover_unseeded_atrial_sequence(ecg,analysis)
        rhythm=clean_selected_rhythm(ecg,analysis)
        fs=int(analysis.get("fs") or FS)
        direct=direct_pr_features(seq.get("unseeded_events") or [],rhythm.get("r_peaks_samples") or [],fs)
        av=dict(analysis.get("av_conduction") or {})
        ar=av.get("atrial_rate_bpm"); vr=av.get("ventricular_rate_bpm")
        rate_ratio=(float(ar)/float(vr)) if ar not in (None,0) and vr not in (None,0) else None
        rows.append({
            "case_id":spec.get("case_id"),
            "target":target,
            "shadow_compatible":bool(shadow.get("compatible")),
            "shadow_evaluable":bool(shadow.get("evaluable")),
            "shadow_p_count":shadow.get("p_count"),
            "shadow_qrs_count":shadow.get("qrs_count"),
            "shadow_nonconducted_p_n":shadow.get("nonconducted_p_n"),
            "shadow_max_consecutive_nonconducted_p":shadow.get("max_consecutive_nonconducted_p"),
            "shadow_pp_cv":shadow.get("pp_cv"),
            **direct,
            "av_classification":av.get("classification"),
            "av_stable_pr":av.get("stable_pr"),
            "av_pr_mad_ms":av.get("pr_mad_ms"),
            "av_pp_cv":av.get("pp_cv"),
            "av_rr_cv":av.get("rr_cv"),
            "av_atrial_rate_bpm":ar,
            "av_ventricular_rate_bpm":vr,
            "av_rate_ratio":rate_ratio,
            "av_phase_mad_ms":av.get("qrs_to_preceding_p_phase_mad_ms"),
            "av_phase_range_ms":av.get("qrs_to_preceding_p_phase_range_ms"),
            "av_dissociation_phase":av.get("av_dissociation_phase"),
        })
    return {"version":VERSION,"target":target,"n":len(rows),"rows":rows,
            "policy":"SYNTHETIC_AVB2_AVB3_ONLY; CHARACTERIZATION_ONLY; NO_THRESHOLD_PROMOTION; NO_FAST; NO_FOLD9"}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--target",choices=["AVB2","AVB3"],required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(); out=run(a.target); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__": main()
