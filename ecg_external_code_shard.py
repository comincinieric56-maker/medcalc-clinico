from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_external_code_test import _canonical_from_code, _predict
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_validation_guard import assert_external_dataset, load_registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--input-scale-mv", type=float, default=1.0)
    args = ap.parse_args()

    assert_external_dataset("code_test", load_registry())

    import h5py

    rows = []
    with h5py.File(args.h5, "r") as fh:
        traces = fh["tracings"]
        total = int(traces.shape[0])
        start = max(0, int(args.start))
        end = min(total, int(args.end))
        if start >= end:
            raise ValueError(f"Invalid shard [{start}, {end}) for total={total}")

        for idx in range(start, end):
            canonical = _canonical_from_code(
                np.asarray(traces[idx], dtype=float),
                input_scale_mv=float(args.input_scale_mv),
            )
            measurements = analyze_canonical_ecg(canonical)
            pred = _predict(measurements)
            flat = {k: v for k, v in pred.items() if k != "predictions"}
            for label, value in (pred.get("predictions") or {}).items():
                flat[f"pred_{label}"] = value
            rows.append({"record_index": idx, **flat})
            if (idx - start + 1) % 25 == 0 or idx + 1 == end:
                print(f"CODE_TEST_SHARD {start}:{end} {idx + 1 - start}/{end - start}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"CODE_TEST_SHARD_COMPLETE start={start} end={end} rows={len(rows)}")


if __name__ == "__main__":
    main()
