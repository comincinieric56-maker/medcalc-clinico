from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

VERSION = "MEDCALC_ECG_SYNTHETIC_SIGNAL_COHORT_V3"
SEED_NAMESPACE = "MEDCALC_ECG_SYNTHETIC_SIGNAL_COHORT_V3"
ROLE = "DEVELOPMENT_REGRESSION_ONLY"
FS = 500
DURATION_S = 10.0
DIAGNOSTIC_CASES_EACH = 45
CONTROL_N = 145
TOTAL_CASES = 1000
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LIMB_ANGLES = {
    "I": 0.0, "II": 60.0, "III": 120.0,
    "aVR": -150.0, "aVL": -30.0, "aVF": 90.0,
}
PRECORDIAL_SCALE = {
    "V1": -0.55, "V2": -0.28, "V3": 0.12,
    "V4": 0.55, "V5": 0.82, "V6": 0.74,
}
P_SCALE = {
    "I": 0.72, "II": 1.00, "III": 0.55,
    "aVR": -0.55, "aVL": 0.44, "aVF": 0.82,
    "V1": 0.48, "V2": 0.38, "V3": 0.32,
    "V4": 0.28, "V5": 0.25, "V6": 0.22,
}
DIAGNOSTIC_GROUPS: dict[str, set[str]] = {
    "AF": {"AF_COMPATIBLE"},
    "FLUTTER": {"FLUTTER_OR_AT_COMPATIBLE"},
    "SINUS_BRADY": {"SINUS_BRADYCARDIA_COMPATIBLE"},
    "SINUS_TACHY": {"SINUS_TACHYCARDIA_COMPATIBLE"},
    "SINUS_NORMAL": {"SINUS_COMPATIBLE"},
    "RBBB_COMPLETE": {"RBBB_MORPHOLOGY_COMPATIBLE"},
    "RBBB_INCOMPLETE": {"INCOMPLETE_RBBB_MORPHOLOGY_COMPATIBLE"},
    "LBBB": {"LBBB_MORPHOLOGY_COMPATIBLE"},
    "LBBB_INCOMPLETE": {"INCOMPLETE_LBBB_MORPHOLOGY_COMPATIBLE"},
    "LAFB": {"LAFB_COMPATIBLE"},
    "LPFB": {"LPFB_COMPATIBLE"},
    "AVB1": {"FIRST_DEGREE_AV_DELAY_COMPATIBLE"},
    "AVB2": {
        "MOBITZ_I_WENCKEBACH_COMPATIBLE",
        "MOBITZ_II_COMPATIBLE",
        "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
        "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
    },
    "AVB3": {"COMPLETE_AV_BLOCK_COMPATIBLE"},
    "WPW": {"VENTRICULAR_PREEXCITATION_COMPATIBLE"},
    "PVC": {"PVC_COMPATIBLE"},
    "PAC": {"PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE"},
    "VT": {"VT_COMPATIBLE"},
    "OTHER_SVT": {"OTHER_SVT_COMPATIBLE"},
}
ALL_TARGET_CODES = sorted({
    code for codes in DIAGNOSTIC_GROUPS.values() for code in codes
})
ABNORMAL_CONTROL_CODES = sorted(
    set(ALL_TARGET_CODES) - {"SINUS_COMPATIBLE"}
)


def _seed(case_id: str) -> int:
    raw = hashlib.sha256(f"{SEED_NAMESPACE}|{case_id}".encode()).digest()
    return int.from_bytes(raw[:8], "big") % (2**32)


def _gauss(t: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / max(float(sigma), 1e-5)) ** 2)


def _triangle(t: np.ndarray, center: float, half_width: float) -> np.ndarray:
    return np.clip(1.0 - np.abs(t-center)/max(float(half_width),1e-5), 0.0, 1.0)


def _lead_scale(lead: str, axis_deg: float) -> float:
    if lead in LIMB_ANGLES:
        return float(math.cos(math.radians(LIMB_ANGLES[lead] - axis_deg)))
    return float(PRECORDIAL_SCALE[lead])


def _regular_times(rate_bpm: float, start: float = 0.90) -> list[float]:
    interval = 60.0 / max(rate_bpm, 1.0)
    out: list[float] = []
    x = start
    while x < DURATION_S - 0.55:
        out.append(float(x))
        x += interval
    return out


def _irregular_times(rng: np.random.Generator) -> list[float]:
    out: list[float] = []
    x = 0.75
    while x < DURATION_S - 0.55:
        out.append(float(x))
        x += float(np.clip(rng.normal(0.78, 0.19), 0.44, 1.24))
    return out


def _normal_qrs(
    t: np.ndarray,
    center: float,
    lead: str,
    axis_deg: float,
    qrs_ms: float,
    gain: float,
) -> np.ndarray:
    scale = _lead_scale(lead, axis_deg) * gain
    stretch = max(qrs_ms / 92.0, 0.75)
    return (
        -0.12 * scale * _gauss(t, center-0.024*stretch, 0.008*stretch)
        +1.02 * scale * _gauss(t, center, 0.012*stretch)
        -0.28 * scale * _gauss(t, center+0.026*stretch, 0.010*stretch)
    )


def _rbbb_qrs(t, center, lead, axis_deg, qrs_ms, gain):
    base = _normal_qrs(t, center, lead, axis_deg, max(qrs_ms,128.0), gain)
    if lead in {"V1","V2"}:
        base += 0.78*gain*_gauss(t,center+0.065,0.018)
        base -= 0.18*gain*_gauss(t,center+0.030,0.012)
    if lead in {"I","aVL","V5","V6"}:
        base -= 0.52*gain*_gauss(t,center+0.060,0.022)
    return base


def _lbbb_qrs(t, center, lead, axis_deg, qrs_ms, gain):
    if lead in {"V1","V2"}:
        return (
            -0.86*gain*_gauss(t,center+0.005,0.025)
            -0.38*gain*_gauss(t,center+0.060,0.022)
        )
    if lead in {"I","aVL","V5","V6"}:
        return (
            0.78*gain*_gauss(t,center-0.018,0.026)
            +0.62*gain*_gauss(t,center+0.050,0.024)
        )
    return _normal_qrs(t, center, lead, axis_deg, max(qrs_ms,138.0), gain)


def _irbbb_qrs(t, center, lead, axis_deg, qrs_ms, gain):
    base = _normal_qrs(t, center, lead, axis_deg, max(qrs_ms,108.0), gain)
    if lead in {"V1","V2"}:
        base += 0.50*gain*_gauss(t,center+0.045,0.014)
        base -= 0.10*gain*_gauss(t,center+0.022,0.010)
    if lead in {"I","aVL","V5","V6"}:
        base -= 0.30*gain*_gauss(t,center+0.042,0.016)
    return base


def _ilbbb_qrs(t, center, lead, axis_deg, qrs_ms, gain):
    if lead in {"V1","V2"}:
        return (
            -0.72*gain*_gauss(t,center+0.004,0.018)
            -0.26*gain*_gauss(t,center+0.040,0.016)
        )
    if lead in {"I","aVL","V5","V6"}:
        return (
            0.72*gain*_gauss(t,center-0.012,0.019)
            +0.42*gain*_gauss(t,center+0.032,0.017)
        )
    return _normal_qrs(t, center, lead, axis_deg, max(qrs_ms,108.0), gain)


def _wpw_qrs(t, center, lead, axis_deg, qrs_ms, gain):
    scale = _lead_scale(lead, axis_deg)
    delta = 0.30*scale*gain*_triangle(t,center-0.030,0.040)
    return delta + _normal_qrs(
        t, center, lead, axis_deg, max(qrs_ms,116.0), gain
    )


def _base_target_spec(target: str, v: int) -> dict[str, Any]:
    noise = 0.004 + 0.002*(v % 5)
    gain = 0.88 + 0.04*(v % 6)
    jitter = (v % 7) - 3
    common = {
        "kind":"TARGET", "target":target, "variant":v,
        "noise_sd":noise, "gain":gain, "axis_deg":55.0,
        "hr":70.0, "pr_ms":160.0, "qrs_ms":92.0, "mode":"SINUS",
    }
    if target in {"SINUS_NORMAL","SINUS_BRADY","SINUS_TACHY"}:
        common["use_neurokit_multilead"] = True
    if target == "AF":
        common.update(mode="AF")
    elif target == "FLUTTER":
        common.update(mode="FLUTTER",hr=70.0+5.0*(v%4))
    elif target == "SINUS_BRADY":
        common.update(hr=40.0+2.0*(v%6))
    elif target == "SINUS_TACHY":
        common.update(hr=106.0+5.0*(v%6))
    elif target == "SINUS_NORMAL":
        common.update(hr=64.0+4.0*(v%7),pr_ms=145.0+5.0*(v%5),qrs_ms=86.0+2.0*(v%5))
    elif target == "RBBB_COMPLETE":
        common.update(mode="RBBB",hr=68.0+2.0*(v%5),qrs_ms=130.0+4.0*(v%5))
    elif target == "RBBB_INCOMPLETE":
        common.update(mode="IRBBB",hr=68.0+2.0*(v%5),qrs_ms=106.0+2.0*(v%5))
    elif target == "LBBB":
        common.update(mode="LBBB",hr=66.0+2.0*(v%5),qrs_ms=138.0+4.0*(v%5))
    elif target == "LBBB_INCOMPLETE":
        common.update(mode="ILBBB",hr=66.0+2.0*(v%5),qrs_ms=106.0+2.0*(v%5))
    elif target == "LAFB":
        common.update(axis_deg=-58.0-4.0*(v%4),qrs_ms=90.0+2.0*(v%3))
    elif target == "LPFB":
        common.update(axis_deg=112.0+5.0*(v%4),qrs_ms=90.0+2.0*(v%3))
    elif target == "AVB1":
        common.update(hr=62.0+2.0*(v%5),pr_ms=220.0+8.0*(v%5))
    elif target == "AVB2":
        common.update(mode="AVB2")
    elif target == "AVB3":
        common.update(mode="AVB3")
    elif target == "WPW":
        common.update(mode="WPW",hr=68.0+2.0*(v%5),pr_ms=88.0+4.0*(v%4),qrs_ms=118.0+4.0*(v%4))
    elif target == "PVC":
        common.update(mode="PVC",hr=68.0+2.0*(v%5))
    elif target == "PAC":
        common.update(mode="PAC",hr=70.0+2.0*(v%5))
    elif target == "VT":
        common.update(mode="VT",hr=145.0+5.0*(v%6),axis_deg=-55.0+5.0*(v%4),qrs_ms=145.0+5.0*(v%4))
    elif target == "OTHER_SVT":
        common.update(mode="OTHER_SVT",hr=145.0+5.0*(v%7),qrs_ms=88.0+2.0*(v%4))
    common["axis_deg"] = float(common["axis_deg"] + 0.5*jitter)
    return common


def _control_spec(v: int) -> dict[str, Any]:
    mode = v % 5
    spec = {
        "kind":"CONTROL", "target":"CONTROL", "variant":v,
        "mode":"SINUS", "axis_deg":55.0, "hr":72.0,
        "pr_ms":160.0, "qrs_ms":92.0,
        "noise_sd":0.004, "gain":1.0,
        "control_type":"NORMAL_MIDRATE",
        "use_neurokit_multilead":True,
    }
    if mode == 0:
        spec.update(control_type="NORMAL_MIDRATE",hr=68.0+2.0*(v%5),noise_sd=0.002)
    elif mode == 1:
        spec.update(control_type="SLOW_NORMAL",hr=56.0+2.0*(v%4),noise_sd=0.003)
    elif mode == 2:
        spec.update(control_type="FAST_NORMAL",hr=88.0+2.0*(v%4),noise_sd=0.003)
    elif mode == 3:
        spec.update(control_type="NOISY_NORMAL",hr=68.0+2.0*(v%5),noise_sd=0.012+0.002*(v%3))
    else:
        spec.update(control_type="LOW_AMPLITUDE_NORMAL",hr=68.0+2.0*(v%5),noise_sd=0.004,gain=0.58+0.04*(v%4))
    return spec


def all_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for target in DIAGNOSTIC_GROUPS:
        for v in range(DIAGNOSTIC_CASES_EACH):
            s = _base_target_spec(target,v)
            s["case_id"] = f"SYNTH_{target}_{v:03d}"
            specs.append(s)
    for v in range(CONTROL_N):
        s = _control_spec(v)
        s["case_id"] = f"SYNTH_CONTROL_{v:03d}"
        specs.append(s)
    assert len(specs) == TOTAL_CASES, len(specs)
    return specs


def _event_times(spec: dict[str, Any], rng: np.random.Generator):
    mode = spec["mode"]
    if mode == "AF":
        return [], _irregular_times(rng), set()
    if mode == "FLUTTER":
        r = _regular_times(float(spec["hr"]))
        p = []
        x = 0.26
        while x < DURATION_S-0.30:
            p.append(float(x))
            x += 0.20
        return p, r, set()
    if mode == "AVB2":
        atrial_rate = 88.0 + 2.0*(int(spec["variant"])%5)
        pp = 60.0/atrial_rate
        p=[]; x=0.55
        while x < DURATION_S-0.45:
            p.append(float(x)); x += pp
        r=[]; conducted=set()
        conduct = (
            (lambda idx: idx%2 == 0)
            if int(spec["variant"])%2 == 0
            else (lambda idx: idx%3 != 2)
        )
        for idx,pt in enumerate(p):
            if conduct(idx):
                rt=pt+0.160+0.004*(int(spec["variant"])%3)
                if rt < DURATION_S-0.35:
                    r.append(rt); conducted.add(round(pt,6))
        return p,r,conducted
    if mode == "AVB3":
        p = _regular_times(88.0+2.0*(int(spec["variant"])%4),0.42)
        r = _regular_times(42.0+2.0*(int(spec["variant"])%5),0.88+0.025*(int(spec["variant"])%4))
        return p,r,set()
    if mode == "VT":
        return [], _regular_times(float(spec["hr"]),0.62), set()
    if mode == "OTHER_SVT":
        return [], _regular_times(float(spec["hr"]),0.62), set()
    if mode in {"PVC","PAC"}:
        base=_regular_times(float(spec["hr"]))
        if len(base)<6:
            return [],base,set()
        idx=len(base)//2
        rr=60.0/float(spec["hr"])
        ect=base[idx-1]+(0.54 if mode=="PVC" else 0.62)*rr
        r=base[:idx]+[ect]+base[idx+1:]
        pr=float(spec["pr_ms"])/1000.0
        p=[]
        for rt in r:
            if mode=="PVC" and abs(rt-ect)<1e-6:
                continue
            p.append(rt-pr)
        return p,r,{round(ect,6)}
    r = _regular_times(float(spec["hr"]))
    pr = float(spec["pr_ms"])/1000.0
    p = [rt-pr for rt in r if rt-pr>0.15]
    return p,r,{round(pt,6) for pt in p}


def make_signal(spec: dict[str, Any]) -> np.ndarray:
    rng=np.random.default_rng(_seed(str(spec["case_id"])))

    if bool(spec.get("use_neurokit_multilead")):
        import neurokit2 as nk
        simulated=nk.ecg_simulate(
            duration=DURATION_S,
            sampling_rate=FS,
            heart_rate=float(spec["hr"]),
            heart_rate_std=1.0,
            noise=float(spec["noise_sd"]),
            method="multileads",
            random_state=_seed(str(spec["case_id"])),
            random_state_distort="spawn",
        )
        arr=np.asarray(simulated[LEADS],dtype=float)
        if arr.shape[0] != int(round(DURATION_S*FS)):
            arr=arr[:int(round(DURATION_S*FS)),:]
        if arr.shape != (int(round(DURATION_S*FS)),len(LEADS)):
            raise ValueError(f"unexpected NeuroKit multilead shape {arr.shape}")
        return arr * float(spec.get("gain") or 1.0)

    n=int(round(DURATION_S*FS))
    t=np.arange(n,dtype=float)/FS
    x=np.zeros((n,len(LEADS)),dtype=float)
    p_times,r_times,conducted=_event_times(spec,rng)
    for j,lead in enumerate(LEADS):
        y=x[:,j]
        gain=float(spec["gain"])*(0.96+0.08*rng.random())
        y += 0.014*np.sin(2*math.pi*0.22*t+rng.uniform(0,2*math.pi))
        if spec["mode"]=="AF":
            y += 0.032*P_SCALE[lead]*np.sin(2*math.pi*6.1*t+rng.uniform(0,2*math.pi))
            y += 0.018*P_SCALE[lead]*np.sin(2*math.pi*8.0*t+rng.uniform(0,2*math.pi))
        elif spec["mode"]=="FLUTTER":
            for pt in p_times:
                y += 0.060*P_SCALE[lead]*_triangle(t,pt,0.042)
                y -= 0.032*P_SCALE[lead]*_triangle(t,pt+0.070,0.050)
        else:
            for pt in p_times:
                amp=1.0
                if spec["mode"]=="AVB2" and round(pt,6) not in conducted:
                    amp=0.95
                y += 0.12*P_SCALE[lead]*gain*amp*_gauss(t,pt,0.027)
        for rt in r_times:
            if spec["mode"]=="RBBB":
                y += _rbbb_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            elif spec["mode"]=="IRBBB":
                y += _irbbb_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            elif spec["mode"]=="LBBB":
                y += _lbbb_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            elif spec["mode"]=="ILBBB":
                y += _ilbbb_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            elif spec["mode"]=="WPW":
                y += _wpw_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            elif spec["mode"]=="VT":
                y += _normal_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),1.10*gain)
            elif spec["mode"]=="PVC" and round(rt,6) in conducted:
                y += _normal_qrs(t,rt,lead,-35.0,155.0,1.15*gain)
            else:
                y += _normal_qrs(t,rt,lead,float(spec["axis_deg"]),float(spec["qrs_ms"]),gain)
            y += 0.24*_lead_scale(lead,float(spec["axis_deg"]))*gain*_gauss(t,rt+0.28,0.072)
        y += rng.normal(0.0,float(spec["noise_sd"]),size=n)
    return x


def canonical(spec: dict[str, Any], signal: np.ndarray) -> dict[str, Any]:
    leads={}
    for j,lead in enumerate(LEADS):
        y=np.asarray(signal[:,j],dtype=float)
        leads[lead]={
            "lead":lead,
            "signal_mv":y.tolist(),
            "quality_mask":np.full(len(y),2,dtype=np.uint8).tolist(),
            "fs":FS,
            "duration_s":DURATION_S,
            "source":"MEDCALC_SYNTHETIC_SIGNAL_COHORT",
            "confidence":1.0,
            "status":"MEASURABLE",
        }
    return {
        "version":"MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "source":"MEDCALC_SYNTHETIC_SIGNAL_COHORT",
        "fs":FS,
        "lead_order":list(LEADS),
        "leads":leads,
        "calibration":{
            "speed_mm_per_s":25.0,
            "gain_mm_per_mv":10.0,
            "confidence":1.0,
            "source":"SYNTHETIC_GENERATOR",
        },
        "validation_provenance":{
            "dataset_id":"synthetic",
            "record_ref":str(spec["case_id"]),
            "development_contaminated":True,
            "usage_role":ROLE,
            "external_validation_claim_allowed":False,
        },
    }


def _blank_counts() -> dict[str, int]:
    return {
        "n":0,
        "candidate_positive_n":0,
        "fusion_positive_n":0,
        "final_positive_n":0,
    }


def run_shard(shard_index: int, shard_count: int) -> dict[str, Any]:
    from ecg_adult_diagnostic_dev_benchmark import (
        _candidate_codes,
        _fusion_codes,
        _published_codes,
    )
    from ecg_signal_measurements import analyze_canonical_ecg
    specs=all_specs()
    selected=[s for i,s in enumerate(specs) if i%shard_count==shard_index]
    per_target={target:_blank_counts() for target in DIAGNOSTIC_GROUPS}
    control_types: dict[str,dict[str,int]]={}
    control_any={
        "n":0,
        "candidate_fp_n":0,
        "fusion_fp_n":0,
        "final_fp_n":0,
        "final_code_counts":{},
    }
    errors=[]
    target_code_union=set(ABNORMAL_CONTROL_CODES)

    for pos,spec in enumerate(selected,1):
        try:
            analysis=analyze_canonical_ecg(canonical(spec,make_signal(spec)))
            candidate=_candidate_codes(analysis)
            fusion=_fusion_codes(analysis)
            final=_published_codes(analysis)
            if spec["kind"]=="TARGET":
                expected=set(DIAGNOSTIC_GROUPS[str(spec["target"])])
                row=per_target[str(spec["target"])]
                row["n"]+=1
                row["candidate_positive_n"]+=int(bool(candidate & expected))
                row["fusion_positive_n"]+=int(bool(fusion & expected))
                row["final_positive_n"]+=int(bool(final & expected))
            else:
                ctype=str(spec["control_type"])
                row=control_types.setdefault(ctype,{"n":0,"candidate_fp_n":0,"fusion_fp_n":0,"final_fp_n":0})
                row["n"]+=1; control_any["n"]+=1
                for key,codes in [
                    ("candidate_fp_n",candidate),
                    ("fusion_fp_n",fusion),
                    ("final_fp_n",final),
                ]:
                    hit=int(bool(codes & target_code_union))
                    row[key]+=hit; control_any[key]+=hit
                for code in sorted(final & target_code_union):
                    counts=control_any["final_code_counts"]
                    counts[code]=int(counts.get(code) or 0)+1
        except Exception as exc:
            errors.append({
                "case_id":str(spec["case_id"]),
                "error":f"{type(exc).__name__}:{exc}",
            })
        if pos%10==0:
            print(
                f"MEDCALC_SYNTHETIC_SHARD {shard_index}/{shard_count} "
                f"{pos}/{len(selected)}",
                flush=True,
            )

    return {
        "version":VERSION,
        "role":ROLE,
        "external_validation_claim_allowed":False,
        "shard_index":shard_index,
        "shard_count":shard_count,
        "case_count":len(selected),
        "metrics":{
            "analysis_error_n":len(errors),
            "per_target":per_target,
            "controls":control_any,
            "control_types":control_types,
        },
        "errors":errors,
        "case_level_predictions_emitted":False,
    }


def aggregate_dir(path: Path) -> dict[str, Any]:
    files=sorted(path.rglob("*.json"))
    shards=[]
    for file in files:
        try:
            row=json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("version")==VERSION and "shard_index" in row:
            shards.append(row)
    if not shards:
        raise SystemExit("No synthetic shard JSON files found")

    expected_count=max(int(x["shard_count"]) for x in shards)
    indices=sorted(int(x["shard_index"]) for x in shards)
    if indices != list(range(expected_count)):
        raise SystemExit(f"Missing synthetic shards: got {indices}, expected 0..{expected_count-1}")

    per_target={target:_blank_counts() for target in DIAGNOSTIC_GROUPS}
    controls={
        "n":0,
        "candidate_fp_n":0,
        "fusion_fp_n":0,
        "final_fp_n":0,
        "final_code_counts":{},
    }
    control_types: dict[str,dict[str,int]]={}
    errors=[]
    total=0
    for shard in shards:
        total += int(shard["case_count"])
        errors.extend(shard.get("errors") or [])
        m=shard["metrics"]
        for target,row in m["per_target"].items():
            for key,val in row.items():
                per_target[target][key]+=int(val)
        for key,val in m["controls"].items():
            if key == "final_code_counts":
                for code,count in val.items():
                    controls[key][code]=int(controls[key].get(code) or 0)+int(count)
            else:
                controls[key]+=int(val)
        for ctype,row in m["control_types"].items():
            dst=control_types.setdefault(ctype,{"n":0,"candidate_fp_n":0,"fusion_fp_n":0,"final_fp_n":0})
            for key,val in row.items():
                dst[key]+=int(val)

    if total != TOTAL_CASES:
        raise SystemExit(f"Synthetic cohort size mismatch: {total} != {TOTAL_CASES}")
    for target,row in per_target.items():
        if row["n"] != DIAGNOSTIC_CASES_EACH:
            raise SystemExit(f"{target} count mismatch: {row['n']}")
    if controls["n"] != CONTROL_N:
        raise SystemExit(f"control count mismatch: {controls['n']}")

    for target,row in per_target.items():
        n=max(row["n"],1)
        row["candidate_sensitivity"]=row["candidate_positive_n"]/n
        row["fusion_sensitivity"]=row["fusion_positive_n"]/n
        row["final_sensitivity"]=row["final_positive_n"]/n
    controls["candidate_specificity"]=(controls["n"]-controls["candidate_fp_n"])/max(controls["n"],1)
    controls["fusion_specificity"]=(controls["n"]-controls["fusion_fp_n"])/max(controls["n"],1)
    controls["final_specificity"]=(controls["n"]-controls["final_fp_n"])/max(controls["n"],1)

    return {
        "version":VERSION,
        "role":ROLE,
        "purpose":"FAST_ENGINEERING_REGRESSION_AND_STRESS_GATE",
        "external_validation_claim_allowed":False,
        "case_count":total,
        "diagnostic_case_n_each":DIAGNOSTIC_CASES_EACH,
        "diagnostic_group_n":len(DIAGNOSTIC_GROUPS),
        "control_n":CONTROL_N,
        "shard_count":expected_count,
        "metrics":{
            "analysis_error_n":len(errors),
            "per_target":per_target,
            "controls":controls,
            "control_types":control_types,
        },
        "case_level_predictions_emitted":False,
        "interpretation":[
            "Synthetic results are engineering-regression evidence only.",
            "They are not clinical sensitivity/specificity estimates.",
            "Real FAST-GATE-100 and PTB-XL fold confirmation remain required.",
        ],
    }


def selftest() -> None:
    specs=all_specs()
    assert len(specs)==1000
    counts=Counter(s["target"] for s in specs if s["kind"]=="TARGET")
    assert counts==Counter({target:45 for target in DIAGNOSTIC_GROUPS}),counts
    assert sum(s["kind"]=="CONTROL" for s in specs)==145
    a=make_signal(specs[0]); b=make_signal(specs[0])
    assert a.shape==(int(FS*DURATION_S),12),a.shape
    assert np.array_equal(a,b)
    assert np.isfinite(a).all()
    c=canonical(specs[0],a)
    assert c["lead_order"]==LEADS
    assert set(c["leads"])==set(LEADS)

    sinus_spec=next(s for s in specs if s.get("target")=="SINUS_NORMAL")
    sinus_a=make_signal(sinus_spec); sinus_b=make_signal(sinus_spec)
    assert sinus_a.shape==(int(FS*DURATION_S),12),sinus_a.shape
    assert np.array_equal(sinus_a,sinus_b)
    assert np.isfinite(sinus_a).all()

    control_spec=next(s for s in specs if s.get("kind")=="CONTROL")
    control_a=make_signal(control_spec); control_b=make_signal(control_spec)
    assert control_a.shape==(int(FS*DURATION_S),12),control_a.shape
    assert np.array_equal(control_a,control_b)
    assert np.isfinite(control_a).all()

    print("MEDCALC_ECG_SYNTHETIC_1000_SELFTEST_PASS")
    print(json.dumps({
        "version":VERSION,
        "case_count":len(specs),
        "target_counts":dict(sorted(counts.items())),
        "control_n":145,
        "first_case_sha256":hashlib.sha256(a.tobytes()).hexdigest(),
        "sinus_multilead_sha256":hashlib.sha256(sinus_a.tobytes()).hexdigest(),
        "control_multilead_sha256":hashlib.sha256(control_a.tobytes()).hexdigest(),
    },indent=2,sort_keys=True))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--shard-index",type=int)
    ap.add_argument("--shard-count",type=int,default=10)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--aggregate-dir",type=Path)
    args=ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.aggregate_dir is not None:
        result=aggregate_dir(args.aggregate_dir)
    else:
        if args.shard_index is None:
            raise SystemExit("--shard-index is required unless --aggregate-dir is used")
        if not (0 <= args.shard_index < args.shard_count):
            raise SystemExit("invalid shard index")
        result=run_shard(args.shard_index,args.shard_count)
    text=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding="utf-8")
    print(text)


if __name__=="__main__":
    main()
