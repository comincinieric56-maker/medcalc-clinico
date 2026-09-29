from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

CLINICAL_ENGINE_FILES = (
    "ecg_signal_measurements.py",
    "ecg_measurement_consensus.py",
    "ecg_atrial_rhythm.py",
    "ecg_av_conduction.py",
    "ecg_crosslead_conduction.py",
    "ecg_domain_gating.py",
    "ecg_candidate_detectors.py",
    "ecg_evidence_fusion.py",
    "ecg_preexcitation.py",
    "ecg_reasoner.py",
)


def clinical_engine_fingerprint(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in CLINICAL_ENGINE_FILES:
        path = root / name
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:24]


def _safe_key(value: str | int) -> str:
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not key:
        raise ValueError("analysis cache record_key is empty after normalization")
    return key


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def analyze_with_cache(
    canonical_ecg: dict[str, Any],
    *,
    cache_dir: str | Path,
    record_key: str | int,
    analyzer: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    engine_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return a cached full ECG analysis when the clinical engine is unchanged.

    The cache namespace includes a fingerprint of the clinical engine source
    files. Audit-only changes therefore reuse analyses, while any clinical
    engine edit automatically uses a new namespace.
    """
    if analyzer is None:
        from ecg_signal_measurements import analyze_canonical_ecg
        analyzer = analyze_canonical_ecg

    fingerprint = engine_fingerprint or clinical_engine_fingerprint()
    root = Path(cache_dir) / fingerprint
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{_safe_key(record_key)}.json"

    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                return cached
        except Exception:
            # Corrupt/incomplete cache entries are recomputed, never trusted.
            pass

    analysis = analyzer(canonical_ecg)
    payload = json.dumps(
        analysis,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    normalized = json.loads(payload)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(root),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    return normalized


def _selftest() -> None:
    calls = {"n": 0}

    def fake_analyzer(ecg: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        return {
            "value": np.float64(12.5),
            "array": np.asarray([1, 2, 3], dtype=np.int64),
            "source": ecg.get("source"),
        }

    with tempfile.TemporaryDirectory() as tmp:
        first = analyze_with_cache(
            {"source": "A"},
            cache_dir=tmp,
            record_key=123,
            analyzer=fake_analyzer,
            engine_fingerprint="engine-a",
        )
        second = analyze_with_cache(
            {"source": "B"},
            cache_dir=tmp,
            record_key=123,
            analyzer=fake_analyzer,
            engine_fingerprint="engine-a",
        )
        assert calls["n"] == 1, calls
        assert first == second, (first, second)

        third = analyze_with_cache(
            {"source": "C"},
            cache_dir=tmp,
            record_key=123,
            analyzer=fake_analyzer,
            engine_fingerprint="engine-b",
        )
        assert calls["n"] == 2, calls
        assert third["source"] == "C", third

    print("ECG_ANALYSIS_CACHE_SELFTEST_PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
