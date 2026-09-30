from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import wfdb

from ecg_adult_diagnostic_dev_benchmark import (
    BASE,
    _canonical,
    _candidate_codes,
    _download,
    _ensure_record,
    _fast_gate_holdout_ids,
    _fusion_codes,
    _published_codes,
    select_records,
)
from ecg_analysis_cache import _json_default
from ecg_signal_measurements import analyze_canonical_ecg

VERSION = "MEDCALC_ECG_NUMERIC_DETERMINISM_AUDIT_V1"
SUBSET_N = 24


def _analysis_hash(analysis: dict[str, Any]) -> str:
    payload = json.dumps(
        analysis,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summary(analysis: dict[str, Any]) -> dict[str, Any]:
    global_metrics = analysis.get("global") or {}
    consensus = analysis.get("measurement_consensus") or {}
    metrics = consensus.get("metrics") or {}
    axis = analysis.get("multilead_frontal_qrs_axis") or {}
    atrial = analysis.get("atrial_rhythm") or {}
    return {
        "candidate_codes": sorted(_candidate_codes(analysis)),
        "fusion_codes": sorted(_fusion_codes(analysis)),
        "published_codes": sorted(_published_codes(analysis)),
        "heart_rate_bpm": global_metrics.get("heart_rate_bpm"),
        "global_qrs_ms": (global_metrics.get("qrs_ms") or {}).get("value"),
        "global_pr_ms": (global_metrics.get("pr_ms") or {}).get("value"),
        "consensus_qrs_ms": (metrics.get("qrs_ms") or {}).get("value"),
        "consensus_pr_ms": (metrics.get("pr_ms") or {}).get("value"),
        "multilead_axis_deg": axis.get("degrees"),
        "atrial_primary": atrial.get("primary_rhythm"),
        "af_score": atrial.get("af_score"),
        "flutter_score": atrial.get("flutter_score"),
    }


def audit(workdir: Path) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path = workdir / "ptbxl_database.csv"
    statements_path = workdir / "scp_statements.csv"
    _download(f"{BASE}/ptbxl_database.csv", metadata_path)
    _download(f"{BASE}/scp_statements.csv", statements_path)
    meta = pd.read_csv(metadata_path)
    selected, selection = select_records(
        meta,
        folds=[9],
        exclude_ecg_ids=_fast_gate_holdout_ids(),
    )
    negative_ids = list(selection["negative_control_ecg_ids"])[:SUBSET_N]
    rows = []
    records_root = workdir / "records"
    selected = selected.copy()
    selected["_ecg_id_int"] = selected["ecg_id"].astype(int)
    by_id = selected.set_index("_ecg_id_int", drop=False)

    for pos, ecg_id in enumerate(negative_ids, 1):
        row = by_id.loc[int(ecg_id)]
        local_base = _ensure_record(records_root, str(row["filename_hr"]))
        rec = wfdb.rdrecord(str(local_base))
        canonical = _canonical(
            rec.p_signal,
            int(round(float(rec.fs))),
            list(rec.sig_name),
            int(ecg_id),
        )
        first = analyze_canonical_ecg(canonical)
        second = analyze_canonical_ecg(canonical)
        h1 = _analysis_hash(first)
        h2 = _analysis_hash(second)
        s1 = _summary(first)
        s2 = _summary(second)
        rows.append({
            "ecg_id": int(ecg_id),
            "analysis_sha256_first": h1,
            "analysis_sha256_second": h2,
            "same_within_process": h1 == h2,
            "summary_first": s1,
            "summary_second": s2,
            "published_same_within_process": s1["published_codes"] == s2["published_codes"],
        })
        print(f"DETERMINISM_AUDIT {pos}/{len(negative_ids)} ecg_id={ecg_id}", flush=True)

    return {
        "version": VERSION,
        "fold": 9,
        "role": "AUDIT_ONLY_NO_CLINICAL_LOGIC_CHANGE",
        "subset_n": len(rows),
        "ecg_ids": [r["ecg_id"] for r in rows],
        "within_process_exact_match_n": sum(r["same_within_process"] for r in rows),
        "within_process_published_match_n": sum(r["published_same_within_process"] for r in rows),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.workdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "version": result["version"],
        "subset_n": result["subset_n"],
        "within_process_exact_match_n": result["within_process_exact_match_n"],
        "within_process_published_match_n": result["within_process_published_match_n"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
