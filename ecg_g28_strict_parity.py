"""Exact-field ECG G26 differential-contract audit; research QA only.

This checker does not validate clinical diagnoses or ECG model performance.
It compares frozen base/main and additive feature outputs. It never reads
annotations, never changes measurements, and cannot deploy models.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from ecg_g29_seven_audit_ui import safe_seven_audit_rows

G26_MODULES = frozenset((
    "AVB2", "AVB3", "SVT", "TRIGEMINY", "ST_ELEVATION",
    "ISCHEMIA_GENERIC", "MI_HISTORY_Q_SCREEN",
))
TRIGEMINY_FIELDS = frozenset((
    "trigeminy_evaluable", "trigeminy_pattern", "trigeminy_repeated_cycles",
    "trigeminy_pattern_kind", "trigeminy_first_index", "review_only",
))
ECTOPY_PARENTS = frozenset((
    ("ectopy",),
    ("feature_graph", "specialist_evidence", "ectopy"),
    ("report", "ectopy"),
    ("report", "feature_graph", "specialist_evidence", "ectopy"),
    ("report", "digital_measurements_v2", "ectopy"),
    ("report", "digital_measurements_v2", "feature_graph", "specialist_evidence", "ectopy"),
))
ADDED_FIELDS = frozenset({("report", "unsupported_seven_evidence")}) | frozenset(p + (field,) for p in ECTOPY_PARENTS for field in TRIGEMINY_FIELDS)


def _walk(a, b, path=()):
    """Return added/removed/modified paths; compare all existing values exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            p = path + (str(key),)
            if key not in a:
                yield ("added", p)
            elif key not in b:
                yield ("removed", p)
            else:
                yield from _walk(a[key], b[key], p)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from _walk(x, y, path + (f"[{i}]",))
    elif type(a) != type(b) or a != b:
        yield ("modified", path)


def _manifest_ok(value):
    """Apply the same fail-closed G26 manifest contract as the display gate.

    This independent regression check never renders the UI or runs inference.
    Unknown categories/statuses, mismatched module IDs and altered policy
    declarations are rejected before accepting additive parity.
    """
    if not isinstance(value, dict):
        return False
    return safe_seven_audit_rows(
        {"unsupported_seven_evidence": value}
    ).get("status") == "ready"


def compare_pair(base, feature):
    if not base.get("success") or not feature.get("success"):
        return {"pass": False, "reason": "INFERENCE_FAILED", "unexpected": []}
    if base.get("subject") != feature.get("subject"):
        return {"pass": False, "reason": "SUBJECT_MISMATCH", "unexpected": []}
    a, b = base.get("fields"), feature.get("fields")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {"pass": False, "reason": "FIELDS_MISSING", "unexpected": []}
    seen = list(_walk(a, b))
    unexpected = [(mode, ".".join(path)) for mode, path in seen
                  if mode != "added" or path not in ADDED_FIELDS]
    actual_added = {path for mode, path in seen if mode == "added"}
    if actual_added != ADDED_FIELDS:
        unexpected.append(("added_field_contract_mismatch", f"actual:{len(actual_added)} expected:{len(ADDED_FIELDS)}"))
    seven = (b.get("report") or {}).get("unsupported_seven_evidence")
    if not _manifest_ok(seven):
        unexpected.append(("unsafe_seven_manifest", "report.unsupported_seven_evidence"))
    return {"pass": not unexpected, "reason": "PASS" if not unexpected else "DRIFT_OR_UNSAFE_SCHEMA",
            "subject":base.get("subject"), "added_n":len(actual_added),
            "modified_or_removed_n":sum(mode != "added" for mode,_ in seen),
            "unexpected":unexpected}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    baseline = json.loads(Path(args.base).read_text())
    feature = json.loads(Path(args.feature).read_text())
    if not isinstance(baseline, list) or not isinstance(feature, list) or len(baseline) != len(feature):
        raise SystemExit("INVALID_PAIR_LENGTH")
    cases = [compare_pair(x,y) for x,y in zip(baseline,feature)]
    report = {"contract":"MEDCALC_G28_STRICT_ADDITIVE_G26_AB", "n":len(cases),
              "passed":sum(x["pass"] for x in cases),
              "failed":sum(not x["pass"] for x in cases),
              "strict_contract_n_added_fields":len(ADDED_FIELDS),
              "source_sha256":{"base":sha(args.base),"feature":sha(args.feature)},
              "cases":cases,"clinically_validated":False,"diagnostic_deployment_allowed":False}
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))
    print(json.dumps({k:report[k] for k in ("n","passed","failed","strict_contract_n_added_fields")}))
    if report["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()