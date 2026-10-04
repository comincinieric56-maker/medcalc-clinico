"""Independent research audit of axis-aligned red ECG grids.

Abstains unless minor lines and five-line major modulation are both visible.
Never imported by the clinical digitizer; does not change its output scale.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import find_peaks


def inspect_profile(profile):
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 1 or not np.isfinite(profile).all() or profile.size < 40:
        return {"accepted": False, "reason": "INVALID_PROFILE"}
    peaks, _ = find_peaks(profile, prominence=max(float(np.ptp(profile)) * .1, 1.))
    if len(peaks) < 25:
        return {"accepted": False, "reason": "INSUFFICIENT_LINES"}
    gaps = np.diff(peaks)
    spacing = float(np.median(gaps))
    regular = float(np.mean(np.abs(gaps / spacing - 1) <= .15))
    if regular < .9:
        return {"accepted": False, "reason": "IRREGULAR_LINES", "regular_fraction": regular}
    # Spatial phase, not peak ordinal: missing lines cannot shift the phase.
    slots = np.rint((peaks - peaks[0]) / spacing).astype(int)
    # Integrate each line: printed major lines can be wider, with the same
    # peak color as minor lines. Keep windows narrower than half a cell.
    radius = max(1, int(np.floor(spacing / 3)))
    strengths = np.array([np.sum(profile[max(0, p-radius):p+radius+1]) for p in peaks])
    phases = [float(np.median(strengths[slots % 5 == phase]))
              for phase in range(5)]
    strongest = int(np.argmax(phases))
    others = np.delete(phases, strongest)
    contrast = phases[strongest] / max(float(np.max(others)), 1e-9)
    if contrast < 1.2:
        return {"accepted": False, "reason": "MAJOR_MINOR_AMBIGUOUS", "major_contrast": contrast}
    return {"accepted": True, "minor_spacing_px": spacing,
            "major_spacing_px": 5 * spacing, "mm_per_pixel": 1 / spacing,
            "regular_fraction": regular, "major_contrast": contrast,
            "line_count": int(len(peaks))}


def inspect_image(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=float)
    redness = np.maximum(0, rgb[:, :, 0] - (rgb[:, :, 1] + rgb[:, :, 2]) / 2)
    axes = {name: inspect_profile(np.median(redness, axis=axis))
            for name, axis in (("x", 0), ("y", 1))}
    return {"image": Path(path).name, "axes": axes,
            "accepted": all(a["accepted"] for a in axes.values()),
            "clinical_scale_override_allowed": False, "training_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({"status": "RESEARCH_ONLY_RAW_GRID_AUDIT",
        "records": [inspect_image(p) for p in args.images]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
