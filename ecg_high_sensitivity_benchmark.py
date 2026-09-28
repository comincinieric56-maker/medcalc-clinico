from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

from ecg_fn_waterfall import CODE_DOMAIN, classify_false_negative
from ecg_measurement_failure_audit import aggregate_measurement_audits


BENCHMARK_VERSION = "MEDCALC_ECG_HIGH_SENSITIVITY_DEVELOPMENT_BENCHMARK_V2"
CANDIDATE_SENSITIVITY_TARGET = 0.97
FINAL_SENSITIVITY_TARGET = 0.90


def _published_codes(analysis: Dict[str, Any]) -> set[str]:
    reasoning = analysis.get("specialist_reasoning") or {}
    summary = reasoning.get("diagnostic_summary") or {}
    return {
        str(row.get("code") or "")
        for row in (summary.get("findings") or [])
        if bool(row.get("publishable"))
    }


def score_labeled_development_cases(
    cases: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Score stage-by-stage sensitivity on development/regression labels only."""
    by_code: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in cases:
        code = str(row.get("expected_code") or "")
        analysis = row.get("analysis") or {}
        if not code:
            continue
        by_code[code].append(analysis)

    output: Dict[str, Any] = {}
    for code, analyses in sorted(by_code.items()):
        n = len(analyses)
        candidate_n = 0
        domain_eligible_n = 0
        fusion_n = 0
        final_n = 0
        waterfall = Counter()
        measurement_states = Counter()
        remeasure_case_n = 0
        unmeasurable_case_n = 0
        uncertain_case_n = 0
        measurement_audits = []

        for analysis in analyses:
            candidate_layer = analysis.get("high_recall_candidates") or {}
            candidate = (candidate_layer.get("by_code") or {}).get(code)
            if candidate is not None:
                candidate_n += 1

            domain = CODE_DOMAIN.get(code, "UNKNOWN")
            gate = (
                ((analysis.get("domain_gates") or {}).get("domains") or {}).get(domain)
                or {}
            )
            if bool(gate.get("eligible")):
                domain_eligible_n += 1

            fused = (
                ((analysis.get("evidence_fusion") or {}).get("by_code") or {}).get(code)
                or {}
            )
            if bool(fused.get("publishable")):
                fusion_n += 1

            if code in _published_codes(analysis):
                final_n += 1

            consensus = analysis.get("measurement_consensus") or {}
            for state in (consensus.get("measurement_states") or {}).values():
                measurement_states[str(state or "UNKNOWN")] += 1
            if consensus.get("remeasure_targets"):
                remeasure_case_n += 1
            if consensus.get("unmeasurable_targets"):
                unmeasurable_case_n += 1
            if consensus.get("uncertain_targets"):
                uncertain_case_n += 1

            audit = analysis.get("measurement_failure_audit") or {}
            if audit:
                measurement_audits.append(audit)

            wf = classify_false_negative(code, analysis)
            waterfall[str(wf.get("stage") or "UNKNOWN")] += 1

        def frac(x: int) -> float | None:
            return round(x / n, 6) if n else None

        candidate_sensitivity = frac(candidate_n)
        final_sensitivity = frac(final_n)
        output[code] = {
            "positive_n": n,
            "candidate_detected_n": candidate_n,
            "candidate_sensitivity": candidate_sensitivity,
            "domain_eligible_n": domain_eligible_n,
            "domain_eligible_rate": frac(domain_eligible_n),
            "fusion_publishable_n": fusion_n,
            "fusion_sensitivity": frac(fusion_n),
            "final_published_n": final_n,
            "final_sensitivity": final_sensitivity,
            "candidate_target": CANDIDATE_SENSITIVITY_TARGET,
            "final_target": FINAL_SENSITIVITY_TARGET,
            "candidate_target_met": bool(
                candidate_sensitivity is not None
                and candidate_sensitivity >= CANDIDATE_SENSITIVITY_TARGET
            ),
            "final_target_met": bool(
                final_sensitivity is not None
                and final_sensitivity >= FINAL_SENSITIVITY_TARGET
            ),
            "false_negative_waterfall": dict(sorted(waterfall.items())),
            "measurement_state_counts": dict(sorted(measurement_states.items())),
            "remeasure_case_rate": frac(remeasure_case_n),
            "unmeasurable_case_rate": frac(unmeasurable_case_n),
            "uncertain_measurement_case_rate": frac(uncertain_case_n),
            "measurement_failure_breakdown": aggregate_measurement_audits(
                measurement_audits
            ) if measurement_audits else None,
        }

    return {
        "version": BENCHMARK_VERSION,
        "scope": "DEVELOPMENT_OR_REGRESSION_DATA_ONLY",
        "frozen_external_record_debugging_allowed": False,
        "targets": {
            "candidate_sensitivity": CANDIDATE_SENSITIVITY_TARGET,
            "final_sensitivity": FINAL_SENSITIVITY_TARGET,
        },
        "diagnoses": output,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    cases = []
    with args.jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    result = score_labeled_development_cases(cases)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
