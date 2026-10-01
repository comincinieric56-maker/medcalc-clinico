from __future__ import annotations

from typing import Any
import numpy as np

RELATIVE_R_AMPLITUDE_MIN = 0.15

def candidate_r_amplitude_mv(signal_mv: np.ndarray, sample: int, fs: int) -> float | None:
    """Return local-baseline amplitude for an R candidate."""
    x=np.asarray(signal_mv,dtype=float).reshape(-1); sample=int(sample)
    if fs<=0 or sample<0 or sample>=x.size or not np.isfinite(x[sample]): return None
    a=max(0,sample-int(round(.200*fs))); b=max(a,sample-int(round(.120*fs)))
    if b-a < max(5,int(round(.020*fs))):
        a=max(0,sample-int(round(.250*fs))); b=max(a,sample-int(round(.060*fs)))
    w=x[a:b]; w=w[np.isfinite(w)]
    if w.size<5: return None
    return abs(float(x[sample])-float(np.median(w)))

def upper_half_r_reference(values: list[float]) -> float | None:
    z=sorted(float(v) for v in values if np.isfinite(float(v)) and float(v)>0)
    if len(z)<3: return None
    upper=z[len(z)//2:]
    if not upper: return None
    ref=float(np.median(np.asarray(upper,dtype=float)))
    return ref if np.isfinite(ref) and ref>0 else None

def filter_relative_r_amplitude(raw_r_samples: list[int], signal_mv: np.ndarray, fs: int, threshold: float=RELATIVE_R_AMPLITUDE_MIN) -> tuple[bool,list[int]]:
    """Behavior-preserving relative-amplitude filter; unevaluable candidates are retained."""
    raw=sorted(set(int(v) for v in raw_r_samples))
    amplitudes=[]
    for sample in raw:
        amp=candidate_r_amplitude_mv(signal_mv,sample,fs)
        if amp is not None: amplitudes.append((sample,float(amp)))
    reference=upper_half_r_reference([amp for _,amp in amplitudes])
    if reference is None: return False,raw
    by_sample=dict(amplitudes)
    kept=[]
    for sample in raw:
        amp=by_sample.get(sample)
        if amp is None or float(amp/reference)>=float(threshold): kept.append(sample)
    return True,kept
