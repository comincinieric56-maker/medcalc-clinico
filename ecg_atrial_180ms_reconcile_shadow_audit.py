from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_atrial_clean_r_mask_shadow_audit import _clean_selected_rhythm, _shadow_mask_inputs
from ecg_independent_atrial_evidence import recover_crosslead_atrial_candidates
from ecg_independent_atrial_evidence_audit import _shadow_harmonic_relation, _unseeded_pr_relation
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import CONTROL_N, DIAGNOSTIC_CASES_EACH, DIAGNOSTIC_GROUPS, ROLE, TOTAL_CASES, all_specs, canonical, make_signal

VERSION = "MEDCALC_ATRIAL_180MS_RECONCILE_SHADOW_AUDIT_V1"
REFRACTORY_MS = 180.0


def _events(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source, rows in (
        ("observed", (evidence.get("observed_consensus") or {}).get("events") or []),
        ("unseeded", evidence.get("unseeded_events") or []),
    ):
        for row in rows:
            if row.get("time_ms") is None:
                continue
            out.append({
                "time_ms": float(row["time_ms"]),
                "support_lead_n": int(row.get("support_lead_n") or row.get("support_n") or 0),
                "source": source,
            })
    return sorted(out, key=lambda x: x["time_ms"])


def _reconcile(evidence: dict[str, Any]) -> dict[str, Any]:
    """Audit-only refractory reconciliation; no timing synthesis and no clinical mutation."""
    rows = _events(evidence)
    clusters: list[list[dict[str, Any]]] = []
    for row in rows:
        if not clusters or row["time_ms"] - clusters[-1][-1]["time_ms"] >= REFRACTORY_MS:
            clusters.append([row])
        else:
            clusters[-1].append(row)

    chosen: list[dict[str, Any]] = []
    for cluster in clusters:
        # Prefer stronger cross-lead support. On ties prefer observed evidence.
        best = max(
            cluster,
            key=lambda r: (
                int(r["support_lead_n"]),
                1 if r["source"] == "observed" else 0,
            ),
        )
        chosen.append(best)

    times = np.asarray([float(r["time_ms"]) for r in chosen], dtype=float)
    pp = np.diff(times) if times.size >= 2 else np.asarray([], dtype=float)
    pp_median = float(np.median(pp)) if pp.size else None
    pp_cv = float(np.std(pp, ddof=1) / np.mean(pp)) if pp.size >= 2 and float(np.mean(pp)) > 0 else None
    organized = bool(
        len(chosen) >= 4
        and pp_median is not None
        and 300.0 <= pp_median <= 1500.0
        and pp_cv is not None
        and pp_cv <= 0.12
    )
    return {
        "unseeded_events": [{"time_ms": round(float(r["time_ms"]), 6)} for r in chosen],
        "unseeded_organized": organized,
        "event_n": len(chosen),
        "pp_median_ms": round(pp_median, 6) if pp_median is not None else None,
        "pp_cv": round(pp_cv, 6) if pp_cv is not None else None,
    }


def _blank() -> dict[str, int]:
    return {
        "n": 0,
        "organized_n": 0,
        "stable_pr_like_n": 0,
        "phase_dissociation_n": 0,
        "complete_block_mechanism_n": 0,
        "organized_complete_block_mechanism_n": 0,
        "event_total": 0,
    }


def _apply(dst: dict[str, int], rec: dict[str, Any], pr: dict[str, Any], h: dict[str, Any]) -> None:
    dst["n"] += 1
    dst["organized_n"] += int(bool(rec.get("unseeded_organized")))
    dst["stable_pr_like_n"] += int(bool(pr.get("stable_pr_like")))
    dst["phase_dissociation_n"] += int(bool(h.get("phase_dissociation")))
    dst["complete_block_mechanism_n"] += int(bool(h.get("complete_block_mechanism")))
    dst["organized_complete_block_mechanism_n"] += int(bool(rec.get("unseeded_organized")) and bool(h.get("complete_block_mechanism")))
    dst["event_total"] += int(rec.get("event_n") or 0)


def run_shard(index: int, count: int) -> dict[str, Any]:
    targets = {k: _blank() for k in DIAGNOSTIC_GROUPS}
    controls = _blank()
    errors = Counter()
    selected = [s for i, s in enumerate(all_specs()) if i % count == index]
    for spec in selected:
        try:
            sig = make_signal(spec)
            can = canonical(spec, sig)
            analysis = analyze_canonical_ecg(can)
            shadow_leads, _ = _shadow_mask_inputs(can, analysis)
            evidence = recover_crosslead_atrial_candidates(can, shadow_leads)
            rec = _reconcile(evidence)
            rhythm = _clean_selected_rhythm(can, analysis)
            fs = int(analysis.get("fs") or can.get("fs") or 0)
            pr = _unseeded_pr_relation(rec, rhythm, fs)
            h = _shadow_harmonic_relation(rec, rhythm, fs)
            if spec.get("kind") == "TARGET":
                _apply(targets[str(spec["target"])], rec, pr, h)
            else:
                _apply(controls, rec, pr, h)
        except Exception as exc:
            errors[type(exc).__name__] += 1
    return {
        "version": VERSION,
        "role": ROLE,
        "purpose": "DEVELOPMENT_ONLY_NONPUBLISHING_180MS_ATRIAL_RECONCILIATION_AUDIT",
        "clinical_output_changed": False,
        "case_count": len(selected),
        "shard_index": index,
        "shard_count": count,
        "metrics": {"per_target": targets, "controls": controls, "analysis_error_n": sum(errors.values()), "analysis_error_types": dict(errors)},
    }


def _merge(a: dict[str, int], b: dict[str, Any]) -> None:
    for k in a:
        a[k] += int(b.get(k) or 0)


def aggregate(path: Path) -> dict[str, Any]:
    rows = []
    for f in path.rglob("*.json"):
        try:
            x = json.loads(f.read_text())
        except Exception:
            continue
        if x.get("version") == VERSION and "shard_index" in x:
            rows.append(x)
    if not rows:
        raise SystemExit("No reconciliation shards found")
    expected = max(int(x["shard_count"]) for x in rows)
    if sorted(int(x["shard_index"]) for x in rows) != list(range(expected)):
        raise SystemExit("Missing shards")
    targets = {k: _blank() for k in DIAGNOSTIC_GROUPS}
    controls = _blank()
    total = errors = 0
    for x in rows:
        total += int(x["case_count"])
        m = x["metrics"]
        errors += int(m.get("analysis_error_n") or 0)
        for k, v in m["per_target"].items():
            _merge(targets[k], v)
        _merge(controls, m["controls"])
    if total != TOTAL_CASES or controls["n"] != CONTROL_N:
        raise SystemExit("cohort size mismatch")
    for k, v in targets.items():
        if v["n"] != DIAGNOSTIC_CASES_EACH:
            raise SystemExit(f"{k} count mismatch")
    def enrich(v: dict[str, int]) -> dict[str, Any]:
        n=max(v["n"],1)
        return {**v, "organized_fraction":v["organized_n"]/n, "stable_pr_like_fraction":v["stable_pr_like_n"]/n, "phase_dissociation_fraction":v["phase_dissociation_n"]/n, "complete_block_mechanism_fraction":v["complete_block_mechanism_n"]/n, "organized_complete_block_mechanism_fraction":v["organized_complete_block_mechanism_n"]/n, "event_mean":v["event_total"]/n}
    return {"version":VERSION,"role":ROLE,"purpose":"DEVELOPMENT_ONLY_NONPUBLISHING_180MS_ATRIAL_RECONCILIATION_AUDIT","clinical_output_changed":False,"case_count":total,"analysis_error_n":errors,"metrics":{"per_target":{k:enrich(v) for k,v in targets.items()},"controls":enrich(controls)}}


def selftest() -> None:
    e={"observed_consensus":{"events":[{"time_ms":1000,"support_lead_n":3},{"time_ms":2000,"support_lead_n":3}]},"unseeded_events":[{"time_ms":1080,"support_lead_n":5},{"time_ms":1500,"support_lead_n":4},{"time_ms":2500,"support_lead_n":4}]}
    r=_reconcile(e)
    assert r["event_n"] == 4, r
    assert r["unseeded_organized"], r
    print("MEDCALC_ATRIAL_180MS_RECONCILE_SHADOW_AUDIT_SELFTEST_PASS")


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--selftest",action="store_true")
    ap.add_argument("--shard-index",type=int)
    ap.add_argument("--shard-count",type=int,default=10)
    ap.add_argument("--aggregate-dir",type=Path)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    if args.selftest:
        selftest(); return
    out=aggregate(args.aggregate_dir) if args.aggregate_dir else run_shard(args.shard_index,args.shard_count)
    txt=json.dumps(out,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(txt)
    print(txt)

if __name__=="__main__":
    main()
