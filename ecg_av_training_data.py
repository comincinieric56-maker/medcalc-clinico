"""New AV training corpus; independent of frozen Synthetic-1000 fixtures."""
from __future__ import annotations

import hashlib
import numpy as np

DATA_VERSION = "R28_AV_SYNTHETIC_V2_T_AUX"
WAVEFORM_SEED_VERSION = "R28_AV_SYNTHETIC_V1"
CLASSES = ("SINUS", "FIRST_DEGREE", "MOBITZ_I", "MOBITZ_II", "TWO_TO_ONE",
           "HIGH_GRADE", "AV_DISSOCIATION", "OTHER")
FS = 250
DURATION_S = 10.0


def synthetic_case(index: int, namespace: str = "train") -> dict:
    # Keep the V1 waveform seed and RNG call order byte-stable so the T-auxiliary
    # experiment changes supervision, not the underlying synthetic ECGs.
    seed = int.from_bytes(
        hashlib.sha256(f"{WAVEFORM_SEED_VERSION}:{namespace}:{index}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    label = index % len(CLASSES)
    pp = rng.uniform(.55, 1.15)
    p_amp, p_width = rng.uniform(.035, .22), rng.uniform(.018, .038)
    qrs_width = rng.uniform(.008, .024)
    # Define onset-to-onset PR using the declared 2-sigma onset convention.
    pr_onset_s = rng.uniform(.23, .34) if label == 1 else rng.uniform(.13, .19)
    delay = pr_onset_s - 2 * (p_width - qrs_width)
    p = np.arange(rng.uniform(.15, .55), DURATION_S - .1, pp)
    p = p + rng.normal(0, .004, len(p))
    if label in (0, 1):
        r = p + delay
    elif label == 2:
        cycle = int(rng.integers(4, 6))
        step = rng.uniform(.025, .04)
        r = np.asarray([t + delay + (i % cycle) * step
                        for i, t in enumerate(p) if i % cycle < cycle - 1])
    elif label == 3:
        cycle = int(rng.integers(3, 6))
        r = np.asarray([t + delay for i, t in enumerate(p) if i % cycle != cycle - 1])
    elif label == 4:
        parity = int(rng.integers(0, 2))
        r = p[parity::2] + delay
    elif label == 5:
        ratio = int(rng.integers(3, 5))
        r = p[::ratio] + delay
    elif label == 6:
        # Two autonomous clocks; occasional coincidences retained.
        rr = rng.uniform(1.15, 1.85)
        r = np.arange(rng.uniform(.2, .9), DURATION_S, rr)
    else:
        subtype = index // len(CLASSES) % 3
        if subtype == 0:  # AF-like irregular ventricular events, no organized P.
            p = np.asarray([])
            r = np.cumsum(rng.uniform(.35, 1.1, 35))
        elif subtype == 1:  # Premature atrial event with conduction.
            p[2::4] -= .18 * pp
            r = p + delay
        else:  # Junctional rhythm, no visible P.
            p = np.asarray([])
            r = np.arange(.4, DURATION_S, rng.uniform(.7, 1.1))
    p, r = p[(p >= 0) & (p < DURATION_S)], r[(r >= 0) & (r < DURATION_S)]

    t = np.arange(int(FS * DURATION_S)) / FS
    x = np.zeros((2, len(t)), dtype=np.float32)
    t_s_by_lead = []
    for lead in range(2):
        polarity = 1 if lead == 0 else rng.choice([-1, 1])
        for event in p:
            x[lead] += polarity * p_amp * rng.uniform(.7, 1.3) * np.exp(-.5 * ((t - event) / p_width) ** 2)
        lead_t = []
        for event in r:
            scale = rng.uniform(.55, 1.35) * (1 if lead == 0 else rng.choice([-1, 1]))
            x[lead] += scale * np.exp(-.5 * ((t - event) / qrs_width) ** 2)
            x[lead] -= .25 * scale * np.exp(-.5 * ((t - event - .035) / .013) ** 2)
            # Preserve the exact V1 RNG call order and arithmetic expression.
            t_amp = rng.uniform(.08, .38)
            t_offset = rng.uniform(.22, .33)
            t_width = rng.uniform(.04, .075)
            x[lead] += t_amp * np.exp(-.5 * ((t - event - t_offset) / t_width) ** 2)
            t_center = event + t_offset
            if 0 <= t_center < DURATION_S:
                lead_t.append(t_center)
        t_s_by_lead.append(np.asarray(lead_t, dtype=float))
        x[lead] += rng.uniform(.01, .09) * np.sin(2 * np.pi * rng.uniform(.15, .5) * t + rng.uniform(0, 6.28))
        x[lead] += rng.normal(0, rng.uniform(.002, .015), len(t))

    # Simulated quantization/resampling of a digitalized trace, not image validation.
    quantum = rng.uniform(.001, .008)
    x = (np.round(x / quantum) * quantum).astype(np.float32)

    # P/QRS targets are shared; T targets are lead-specific because V1 generated
    # lead-specific T timing. This preserves the waveform distribution exactly.
    y = np.zeros((2, 3, len(t)), dtype=np.float32)
    for lead in range(2):
        for channel, events, half_width in (
            (0, p, .024), (1, r, .016), (2, t_s_by_lead[lead], .040)
        ):
            for event in events:
                y[lead, channel, abs(t - event) <= half_width] = 1
    return {"case_id": f"{namespace}-{index}", "patient_id": f"synthetic-{namespace}-{index}",
            "label": label, "signal": x, "targets": y, "p_s": p, "r_s": r,
            "t_s": t_s_by_lead[0], "t_s_by_lead": t_s_by_lead,
            "fs": FS, "duration_s": DURATION_S,
            "onset_definition": "GAUSSIAN_CENTER_MINUS_2_SIGMA",
            "base_pr_onset_s": pr_onset_s,
            "source": DATA_VERSION, "waveform_seed_source": WAVEFORM_SEED_VERSION,
            "role": "SYNTHETIC_DEVELOPMENT_ONLY"}
