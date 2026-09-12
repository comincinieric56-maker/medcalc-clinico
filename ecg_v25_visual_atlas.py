#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC ECG V25 · DIGITAL VISUAL MORPHOLOGY ATLAS
==================================================

This module creates ORIGINAL synthetic ECG reference patterns from explicit
electrocardiographic morphology rules. It does NOT copy or rasterize figures
from textbooks. The atlas is intended as a visual/morphologic reference layer
for MEDCALC, on top of V23.1 measurements and V24 literature criteria.

Design principles
-----------------
1. The atlas is generated from parametric waveforms (P, Q, R, S, R', ST, T).
2. Patterns are lead-specific.
3. The same registry drives:
   - the synthetic reference image,
   - signal-level visual matching,
   - the explanation shown to the user.
4. Visual similarity can SUPPORT a diagnosis only when it is concordant with
   V24 rule evidence. It is never a standalone disease diagnosis.
5. Synthetic templates are pedagogical/algorithmic prototypes, not a claim
   that every real ECG with the condition must look exactly like the template.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math

import numpy as np

VERSION = "V25.0-DIGITAL-VISUAL-ATLAS"
FS = 500.0
BEAT_START = -0.28
BEAT_END = 0.56

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# Baseline synthetic morphology in mV. Parameters are deliberately simple
# enough to remain interpretable and editable.
BASE = {
    "I":   dict(P=.09,Q=-.04,R=.70,S=-.12,T=.22),
    "II":  dict(P=.12,Q=-.04,R=.90,S=-.16,T=.30),
    "III": dict(P=.07,Q=-.03,R=.48,S=-.20,T=.18),
    "aVR": dict(P=-.07,Q=.02,R=-.45,S=.10,T=-.18),
    "aVL": dict(P=.06,Q=-.03,R=.42,S=-.10,T=.15),
    "aVF": dict(P=.09,Q=-.03,R=.65,S=-.15,T=.22),
    "V1":  dict(P=.05,Q=-.02,R=.23,S=-.72,T=-.08),
    "V2":  dict(P=.06,Q=-.02,R=.38,S=-.78,T=.12),
    "V3":  dict(P=.07,Q=-.02,R=.65,S=-.58,T=.25),
    "V4":  dict(P=.08,Q=-.03,R=1.00,S=-.30,T=.36),
    "V5":  dict(P=.08,Q=-.03,R=1.10,S=-.20,T=.34),
    "V6":  dict(P=.07,Q=-.03,R=.92,S=-.14,T=.28),
}

def _g(t, mu, sigma, amp):
    return amp * np.exp(-0.5*((t-mu)/sigma)**2)

def _smooth_window(t, a, b, edge=0.008):
    # Smooth rectangular function 0..1.
    return 0.5*(np.tanh((t-a)/edge)-np.tanh((t-b)/edge))

def _lead_base(t, lead, pr_ms=160.0, qrs_scale=1.0, p_scale=1.0,
               q_scale=1.0, r_scale=1.0, s_scale=1.0, t_scale=1.0,
               st_mv=0.0, st_slope=0.0, t_invert=False,
               t_biphasic=False, delta_mv=0.0, rprime_mv=0.0,
               qrs_shift=0.0, p_override=None, t_override=None):
    p = dict(BASE[lead])
    # QRS landmarks. qrs_scale expands depolarization around R.
    q_mu = -0.028*qrs_scale + qrs_shift
    r_mu = 0.000 + qrs_shift
    s_mu = 0.032*qrs_scale + qrs_shift

    # PR determines P center. Longer PR moves P earlier, shorter PR closer.
    p_mu = qrs_shift - max(0.105, (pr_ms/1000.0) - 0.045)
    p_amp = p["P"] * p_scale if p_override is None else p_override

    y = np.zeros_like(t)
    y += _g(t, p_mu, 0.028, p_amp)
    y += _g(t, q_mu, 0.010*qrs_scale, p["Q"]*q_scale)
    y += _g(t, r_mu, 0.012*qrs_scale, p["R"]*r_scale)
    y += _g(t, s_mu, 0.014*qrs_scale, p["S"]*s_scale)

    if rprime_mv:
        y += _g(t, 0.067*qrs_scale + qrs_shift, 0.018*qrs_scale, rprime_mv)

    if delta_mv:
        # Broad pre-QRS slur.
        y += delta_mv * _smooth_window(t, -0.055 + qrs_shift, -0.004 + qrs_shift, 0.012)

    # ST starts near the end of QRS and ends before T.
    st_start = 0.070*qrs_scale + qrs_shift
    st_end = 0.190 + qrs_shift
    stw = _smooth_window(t, st_start, st_end, 0.012)
    y += st_mv * stw
    if st_slope:
        frac = np.clip((t-st_start)/max(st_end-st_start,1e-6), 0.0, 1.0)
        y += st_slope*frac*stw

    t_mu = 0.285 + qrs_shift
    tamp = p["T"] * t_scale if t_override is None else t_override
    if t_invert:
        tamp = -abs(tamp)
    if t_biphasic:
        y += _g(t, t_mu-0.025, 0.045, abs(tamp)*0.70)
        y += _g(t, t_mu+0.035, 0.050, -abs(tamp))
    else:
        y += _g(t, t_mu, 0.055, tamp)
    return y

def _normal(t, lead):
    return _lead_base(t, lead)

def _bav1(t, lead):
    return _lead_base(t, lead, pr_ms=240.0)

def _rbbb(t, lead):
    if lead in {"V1","V2"}:
        return _lead_base(t, lead, qrs_scale=1.42, r_scale=.55, s_scale=.80,
                          rprime_mv=0.75 if lead=="V1" else 0.55, t_invert=True)
    if lead in {"I","V5","V6"}:
        y = _lead_base(t, lead, qrs_scale=1.32, r_scale=1.0, s_scale=1.7)
        y += _g(t, 0.070, 0.025, -0.28 if lead=="I" else -0.22)
        return y
    return _lead_base(t, lead, qrs_scale=1.30)

def _lbbb(t, lead):
    if lead in {"V1","V2"}:
        # Broad QS/rS and discordant positive T/ST.
        y = _lead_base(t, lead, qrs_scale=1.65, r_scale=.15, s_scale=1.45,
                       st_mv=.09, t_override=.25)
        return y
    if lead in {"I","aVL","V5","V6"}:
        y = _lead_base(t, lead, qrs_scale=1.65, q_scale=.05, r_scale=1.28,
                       s_scale=.15, st_mv=-.08, t_invert=True)
        y += _g(t, 0.050, 0.025, 0.42)
        y += _g(t, 0.085, 0.020, 0.20)
        return y
    return _lead_base(t, lead, qrs_scale=1.55)

def _lafb(t, lead):
    if lead in {"I","aVL"}:
        return _lead_base(t, lead, q_scale=1.8, r_scale=1.25, s_scale=.35)
    if lead in {"II","III","aVF"}:
        return _lead_base(t, lead, q_scale=.15, r_scale=.55, s_scale=1.70)
    return _lead_base(t, lead)

def _wpw(t, lead):
    # Short PR + delta + wider QRS. Polarity varies; this is a generic prototype.
    delta_sign = 1.0 if lead not in {"aVR","V1"} else -1.0
    return _lead_base(t, lead, pr_ms=95.0, qrs_scale=1.30,
                      delta_mv=delta_sign*0.16, r_scale=1.05, s_scale=1.05)

def _lvh(t, lead):
    if lead in {"V1","V2"}:
        return _lead_base(t, lead, s_scale=1.65, r_scale=.75)
    if lead in {"aVL","V5","V6"}:
        y = _lead_base(t, lead, r_scale=1.65, st_mv=-.09, t_invert=True, t_scale=1.2)
        return y
    return _lead_base(t, lead, r_scale=1.15)

def _low_voltage(t, lead):
    return 0.34*_lead_base(t, lead)

def _stemi_inferior(t, lead):
    if lead in {"II","III","aVF"}:
        return _lead_base(t, lead, st_mv=.24 if lead=="III" else .18, t_scale=1.15)
    if lead in {"I","aVL"}:
        return _lead_base(t, lead, st_mv=-.10, t_invert=True)
    return _lead_base(t, lead)

def _stemi_anterior(t, lead):
    if lead in {"V2","V3","V4"}:
        amp = {"V2":.26,"V3":.30,"V4":.22}[lead]
        return _lead_base(t, lead, st_mv=amp, t_scale=1.25)
    if lead in {"II","III","aVF"}:
        return _lead_base(t, lead, st_mv=-.05)
    return _lead_base(t, lead)

def _std_ischemia(t, lead):
    if lead in {"I","aVL","V4","V5","V6"}:
        return _lead_base(t, lead, st_mv=-.12, t_invert=True)
    return _lead_base(t, lead)

def _wellens(t, lead):
    if lead in {"V2","V3"}:
        return _lead_base(t, lead, st_mv=.015, t_override=-.45 if lead=="V3" else -.38)
    if lead=="V4":
        return _lead_base(t, lead, t_override=-.22)
    return _lead_base(t, lead)

def _dewinter(t, lead):
    if lead in {"V2","V3","V4","V5"}:
        return _lead_base(t, lead, st_mv=-.16, st_slope=.14, t_override=.70)
    if lead=="aVR":
        return _lead_base(t, lead, st_mv=.08)
    return _lead_base(t, lead)

def _pericarditis(t, lead):
    if lead=="aVR":
        return _lead_base(t, lead, st_mv=-.10, p_scale=1.0)
    if lead=="V1":
        return _lead_base(t, lead, st_mv=-.04)
    return _lead_base(t, lead, st_mv=.12, t_scale=1.0)

def _brugada1(t, lead):
    if lead in {"V1","V2"}:
        # rSr-like depolarization + coved high takeoff and negative T.
        y = _lead_base(t, lead, qrs_scale=1.20, r_scale=.65, s_scale=.70,
                       rprime_mv=.50 if lead=="V1" else .36, st_mv=.26,
                       t_override=-.30)
        # Force descending coved ST.
        st0 = 0.075
        st1 = 0.220
        w = _smooth_window(t, st0, st1, .012)
        y += (-.20*np.clip((t-st0)/(st1-st0),0,1))*w
        return y
    return _lead_base(t, lead)

def _af_beat(t, lead):
    # No organized P; QRS/T otherwise near normal.
    return _lead_base(t, lead, p_override=0.0)

def _flutter_beat(t, lead):
    # Base ventricular complex; flutter waves are generated separately in rhythm strip.
    return _lead_base(t, lead, p_override=0.0)

ATLAS = {
    "NORMAL_SINUS": {
        "name":"Ritmo sinusal / morfología de referencia",
        "family":"rhythm",
        "generator":"normal",
        "key_leads":["I","II","V1","V5"],
        "v24_rules":["RHYTHM_SINUS"],
        "measurement_constraints":{"heart_rate_bpm":[60,100],"pr_ms":[120,200],"qrs_ms":[0,119]},
        "visual_mode":"beat",
        "criteria_summary":["P organizada antes de QRS","PR 120-200 ms","QRS estrecho en adulto"],
        "sources":["AHA_III_2009"],
    },
    "ATRIAL_FIBRILLATION": {
        "name":"Fibrilación auricular · patrón visual",
        "family":"rhythm",
        "generator":"af",
        "key_leads":["II","V1"],
        "v24_rules":["RHYTHM_AF"],
        "measurement_constraints":{},
        "visual_mode":"rhythm_feature",
        "criteria_summary":["RR irregular","sin P reproducible","actividad auricular desorganizada"],
        "sources":["AHA_III_2009"],
    },
    "ATRIAL_FLUTTER": {
        "name":"Flutter auricular · patrón visual",
        "family":"rhythm",
        "generator":"flutter",
        "key_leads":["II","III","aVF","V1"],
        "v24_rules":["RHYTHM_FLUTTER"],
        "measurement_constraints":{},
        "visual_mode":"rhythm_feature",
        "criteria_summary":["actividad auricular regular rápida","ondas F/sawtooth, típicamente inferiores"],
        "sources":["AHA_III_2009"],
    },
    "AV_BLOCK_I": {
        "name":"Bloqueo AV de primer grado",
        "family":"conduction",
        "generator":"bav1",
        "key_leads":["II","V1"],
        "v24_rules":["AVB1"],
        "measurement_constraints":{"pr_ms":[201,500]},
        "visual_mode":"beat",
        "criteria_summary":["PR >200 ms","conducción AV 1:1"],
        "sources":["AHA_III_2009"],
    },
    "RBBB": {
        "name":"Bloqueo completo de rama derecha",
        "family":"conduction",
        "generator":"rbbb",
        "key_leads":["V1","V2","I","V6"],
        "v24_rules":["RBBB"],
        "measurement_constraints":{"qrs_ms":[120,250]},
        "visual_mode":"beat",
        "criteria_summary":["QRS ≥120 ms","rsR'/qR/R terminal en V1-V2","S terminal ancha en I/V6"],
        "sources":["AHA_III_2009"],
    },
    "LBBB": {
        "name":"Bloqueo completo de rama izquierda",
        "family":"conduction",
        "generator":"lbbb",
        "key_leads":["V1","V2","I","aVL","V5","V6"],
        "v24_rules":["LBBB"],
        "measurement_constraints":{"qrs_ms":[120,260]},
        "visual_mode":"beat",
        "criteria_summary":["QRS ancho","V1-V2 predominantemente negativos","R lateral ancha/entallada","repolarización discordante"],
        "sources":["AHA_III_2009"],
    },
    "LAFB": {
        "name":"Hemibloqueo anterior izquierdo",
        "family":"conduction",
        "generator":"lafb",
        "key_leads":["I","aVL","II","III","aVF"],
        "v24_rules":["LAFB"],
        "measurement_constraints":{"axis_deg":[-180,-45]},
        "visual_mode":"beat",
        "criteria_summary":["eje izquierdo","qR I/aVL","rS inferior"],
        "sources":["AHA_III_2009"],
    },
    "PREEXCITATION": {
        "name":"Preexcitación ventricular / WPW-like",
        "family":"conduction",
        "generator":"wpw",
        "key_leads":["I","II","V1","V4","V5"],
        "v24_rules":["PREEXCITATION"],
        "measurement_constraints":{"pr_ms":[0,119],"qrs_ms":[110,260]},
        "visual_mode":"beat",
        "criteria_summary":["PR corto","onda delta","QRS ensanchado"],
        "sources":["AHA_III_2009"],
    },
    "LVH": {
        "name":"Hipertrofia ventricular izquierda por voltaje/strain",
        "family":"chambers",
        "generator":"lvh",
        "key_leads":["aVL","V1","V3","V5","V6"],
        "v24_rules":["LVH"],
        "measurement_constraints":{},
        "visual_mode":"beat_amplitude",
        "criteria_summary":["Sokolow-Lyon/Cornell según datos disponibles","alto voltaje","strain lateral puede acompañar"],
        "sources":["AHA_V_2009"],
    },
    "LOW_VOLTAGE": {
        "name":"Bajo voltaje QRS",
        "family":"voltage",
        "generator":"low_voltage",
        "key_leads":LEADS,
        "v24_rules":["LOW_VOLTAGE"],
        "measurement_constraints":{},
        "visual_mode":"beat_amplitude",
        "criteria_summary":["QRS <0.5 mV en miembros y/o <1.0 mV en precordiales"],
        "sources":["AHA_V_2009"],
    },
    "STEMI_INFERIOR_PATTERN": {
        "name":"Patrón de elevación ST inferior",
        "family":"ischemia",
        "generator":"stemi_inferior",
        "key_leads":["II","III","aVF","I","aVL"],
        "v24_rules":["ST_ELEVATION_ACUTE"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["ST elevado en ≥2 inferiores contiguas","cambios recíprocos laterales pueden acompañar"],
        "sources":["UDMI_2018","AHA_VI_2009"],
    },
    "STEMI_ANTERIOR_PATTERN": {
        "name":"Patrón de elevación ST anterior",
        "family":"ischemia",
        "generator":"stemi_anterior",
        "key_leads":["V2","V3","V4","V5"],
        "v24_rules":["ST_ELEVATION_ACUTE"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["ST elevado en derivaciones anteriores contiguas con umbrales V2-V3 dependientes de edad/sexo"],
        "sources":["UDMI_2018","AHA_VI_2009"],
    },
    "ST_DEPRESSION_PATTERN": {
        "name":"Patrón de depresión ST",
        "family":"ischemia",
        "generator":"std_ischemia",
        "key_leads":["I","aVL","V4","V5","V6"],
        "v24_rules":["ST_DEPRESSION"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["depresión ST territorial reproducible"],
        "sources":["UDMI_2018","AHA_VI_2009"],
    },
    "WELLENS_LIKE": {
        "name":"Patrón tipo Wellens",
        "family":"ischemia",
        "generator":"wellens",
        "key_leads":["V2","V3","V4"],
        "v24_rules":["WELLENS_LIKE"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["T bifásica o profundamente negativa en V2-V3","ST mínimo","sin Q precordial significativa"],
        "sources":["WELLENS_REVIEW"],
    },
    "DE_WINTER": {
        "name":"Patrón tipo de Winter",
        "family":"ischemia",
        "generator":"dewinter",
        "key_leads":["V2","V3","V4","V5","aVR"],
        "v24_rules":["DE_WINTER"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["ST deprimido ascendente en precordiales","T altas/simétricas","aVR puede elevarse"],
        "sources":["DEWINTER_REVIEW_2020"],
    },
    "PERICARDITIS_LIKE": {
        "name":"Patrón tipo pericarditis",
        "family":"repolarization",
        "generator":"pericarditis",
        "key_leads":["I","II","aVL","aVF","V4","V5","V6","aVR"],
        "v24_rules":["PERICARDITIS_LIKE"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["ST difusamente elevado","PR deprimido puede acompañar","aVR suele ser recíproco"],
        "sources":["AHA_IV_2009"],
    },
    "BRUGADA_TYPE1_LIKE": {
        "name":"Patrón tipo 1 de Brugada-like",
        "family":"channelopathy",
        "generator":"brugada1",
        "key_leads":["V1","V2"],
        "v24_rules":["BRUGADA_TYPE1_LIKE"],
        "measurement_constraints":{},
        "visual_mode":"beat",
        "criteria_summary":["elevación coved del ST ≥2 mm V1-V2","descenso ST","T negativa"],
        "sources":["BRUGADA_CONSENSUS_2016"],
    },
}

_GENERATORS = {
    "normal":_normal, "af":_af_beat, "flutter":_flutter_beat, "bav1":_bav1,
    "rbbb":_rbbb, "lbbb":_lbbb, "lafb":_lafb, "wpw":_wpw, "lvh":_lvh,
    "low_voltage":_low_voltage, "stemi_inferior":_stemi_inferior,
    "stemi_anterior":_stemi_anterior, "std_ischemia":_std_ischemia,
    "wellens":_wellens, "dewinter":_dewinter, "pericarditis":_pericarditis,
    "brugada1":_brugada1,
}

def beat_time(fs=FS):
    return np.arange(int(round((BEAT_END-BEAT_START)*fs)), dtype=float)/fs + BEAT_START

def generate_beat(pattern_id: str, lead: str, fs=FS):
    if pattern_id not in ATLAS:
        raise KeyError(pattern_id)
    if lead not in LEADS:
        raise KeyError(lead)
    t = beat_time(fs)
    gen = _GENERATORS[ATLAS[pattern_id]["generator"]]
    y = gen(t, lead)
    return t, y.astype(np.float32)

def _repeat_beats(pattern_id: str, lead: str, duration=10.0, fs=FS, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration*fs), dtype=float)/fs
    y = np.zeros_like(t)

    if pattern_id == "ATRIAL_FIBRILLATION":
        rr = []
        pos = 0.55
        while pos < duration-0.6:
            interval = float(np.clip(rng.normal(.78,.18), .42, 1.35))
            rr.append(interval); pos += interval
        centers = np.cumsum([.55]+rr[:-1])
    elif pattern_id == "ATRIAL_FLUTTER":
        centers = np.arange(.65, duration-.4, .88)
    else:
        centers = np.arange(.60, duration-.4, 1.00)

    bt, by = generate_beat(pattern_id, lead, fs)
    for c in centers:
        idx = np.round((c+bt)*fs).astype(int)
        ok = (idx>=0)&(idx<len(y))
        y[idx[ok]] += by[ok]

    if pattern_id == "ATRIAL_FIBRILLATION":
        # Fine fibrillatory baseline, deterministic seed.
        f = 6.5
        y += .025*np.sin(2*np.pi*f*t + .4) + .012*np.sin(2*np.pi*8.2*t)
    elif pattern_id == "ATRIAL_FLUTTER":
        # Sawtooth-like atrial waves, 4.5 Hz (~270/min), most visible inferiorly.
        phase = (t*4.5) % 1.0
        saw = 2.0*phase - 1.0
        amp = .10 if lead in {"II","III","aVF"} else .045
        if lead=="aVR": amp *= -1
        y += amp*saw
    return t.astype(np.float32), y.astype(np.float32)

def generate_rhythm_strip(pattern_id: str, lead="II", duration=10.0, fs=FS):
    return _repeat_beats(pattern_id, lead, duration=duration, fs=fs)

def atlas_registry_jsonable():
    # Copy to plain JSON structures.
    return json.loads(json.dumps(ATLAS))

def save_registry(path):
    Path(path).write_text(json.dumps(atlas_registry_jsonable(), indent=2, ensure_ascii=False), encoding="utf-8")

def render_pattern_png(pattern_id: str, out_path: str | Path):
    """
    Single-axes stacked 12-lead synthetic reference. No textbook image is copied.
    """
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    t = beat_time()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111)
    offsets = np.arange(len(LEADS))[::-1] * 2.3
    for i, lead in enumerate(LEADS):
        _, y = generate_beat(pattern_id, lead)
        ax.plot(t, y + offsets[i])
        ax.text(BEAT_START-0.035, offsets[i], lead, ha="right", va="center", fontsize=8)
    ax.set_xlim(BEAT_START-0.08, BEAT_END)
    ax.set_ylim(-1.4, offsets[0]+1.4)
    ax.set_xlabel("Tiempo respecto del QRS (s)")
    ax.set_ylabel("Derivaciones apiladas (mV + offset)")
    ax.set_title(f"MEDCALC V25 · {ATLAS[pattern_id]['name']} · referencia sintética")
    ax.grid(True, linewidth=.4, alpha=.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def render_rhythm_png(pattern_id: str, out_path: str | Path, lead="II"):
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    t,y = generate_rhythm_strip(pattern_id, lead=lead)
    fig = plt.figure(figsize=(12,3))
    ax = fig.add_subplot(111)
    ax.plot(t,y)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("mV")
    ax.set_title(f"MEDCALC V25 · {ATLAS[pattern_id]['name']} · {lead}")
    ax.grid(True, linewidth=.4, alpha=.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def render_all(out_dir: str | Path):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = []
    for pid in ATLAS:
        paths.append(render_pattern_png(pid, out/f"{pid}.png"))
        if ATLAS[pid]["visual_mode"] == "rhythm_feature":
            paths.append(render_rhythm_png(pid, out/f"{pid}__RHYTHM_II.png", lead="II"))
    return paths

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    save_registry(here/"ecg_v25_atlas_registry.json")
    print(f"Atlas: {len(ATLAS)} patterns")
    for p in render_all(here/"atlas_assets"):
        print(p)
