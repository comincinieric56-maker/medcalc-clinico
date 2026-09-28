from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_validation_harness import LEADS, generate_ground_truth_ecg
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_signal_report_adapter import build_signal_primary_structured_report
from ecg_machine_header import compose_final_report
from ecg_r27_consensus import TEMPORAL_MODULES, compare_r27_with_medcalc
from r27_local_runtime import ALL35


BENCHMARK_VERSION = "MEDCALC_ECG_DIAGNOSTIC_PIPELINE_DEV_V1"


def _canonical(signals: dict[str, np.ndarray], fs: int) -> dict[str, Any]:
    leads: dict[str, Any] = {}
    for lead in LEADS:
        x = np.asarray(signals[lead], dtype=float)
        leads[lead] = {
            "lead": lead,
            "signal_mv": [float(v) for v in x],
            "quality_mask": [2] * int(x.size),
            "fs": int(fs),
            "duration_s": float(x.size / fs),
            "source": "DEVELOPMENT_NATIVE_SYNTHETIC",
            "confidence": 1.0,
            "status": "MEASURABLE",
        }
    matrix = np.column_stack(
        [np.asarray(signals[lead], dtype=float) for lead in LEADS]
    )
    return {
        "version": "DEVELOPMENT_NATIVE_CANONICAL_DIAGNOSTIC_V1",
        "source": "DEVELOPMENT_NATIVE_SYNTHETIC",
        "fs": int(fs),
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "confidence": 1.0,
        },
        "uncertainty": {
            "timing_uncertainty_ms": 0.0,
            "amplitude_uncertainty_mv": 0.0,
            "source": "NATIVE_DIGITAL_DEVELOPMENT_SIGNAL",
        },
        "leads": leads,
        "lead_order": list(LEADS),
        "legacy_matrix_mv": matrix,
    }


def _payload(
    *,
    focus_module: str | None = None,
    focus_probability: float = 0.50,
    tiled: bool = False,
) -> dict[str, Any]:
    modules = {
        str(name): {
            "probability": 0.50,
            "interpretability": "PROBABILITY_ONLY",
        }
        for name in ALL35
    }
    if focus_module in modules:
        modules[str(focus_module)]["probability"] = float(focus_probability)
    if tiled:
        for name in TEMPORAL_MODULES:
            if name in modules:
                modules[name]["interpretability"] = "NOT_INTERPRETABLE_R27_TILED"
    return {
        "modules": modules,
        "input_adapter": {
            "mode": "DEVELOPMENT_NATIVE" if not tiled else "R27_TILED_DEVELOPMENT",
            "r27_tiled": bool(tiled),
            "research_only": bool(tiled),
        },
    }


def _line(text: str, prefix: str) -> str | None:
    for row in str(text or "").splitlines():
        if row.startswith(prefix):
            return row
    return None


def _snapshot(structured: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy({
        "measurement_summary": structured.get("measurement_summary"),
        "rhythm_screen": structured.get("rhythm_screen"),
        "diagnostic_summary": structured.get("diagnostic_summary"),
        "specialist_reasoning": structured.get("specialist_reasoning"),
        "domain_gates": structured.get("domain_gates"),
        "evidence_fusion": structured.get("evidence_fusion"),
    })


def benchmark() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for hr in (50.0, 75.0, 120.0):
        signals, truth = generate_ground_truth_ecg(heart_rate_bpm=hr)
        canonical = _canonical(signals, int(truth["fs"]))
        measured = analyze_canonical_ecg(canonical)
        structured = build_signal_primary_structured_report(canonical, measured)

        for required in (
            "high_recall_candidates",
            "domain_gates",
            "evidence_fusion",
            "specialist_reasoning",
        ):
            if not isinstance(measured.get(required), dict):
                raise AssertionError(f"{required} missing for HR={hr}")

        baseline_snapshot = _snapshot(structured)
        baseline_final = compose_final_report({}, structured, None)
        baseline_idx = str(baseline_final.get("idx") or "")
        baseline_conclusion = _line(
            str(baseline_final.get("text") or ""),
            "CONCLUSIÓN:",
        )

        neutral = _payload()
        neutral_qa = compare_r27_with_medcalc(structured, neutral)
        represented = [
            str(x.get("r27_module"))
            for x in neutral_qa.get("comparisons") or []
            if x.get("r27_module")
        ]
        focus = represented[0] if represented else "SINUS"

        support_payload = _payload(
            focus_module=focus,
            focus_probability=0.95,
        )
        discord_payload = _payload(
            focus_module=focus,
            focus_probability=0.05,
        )
        represented_set = set(represented)
        r27_only_module = next(
            (
                x for x in ("LBBB", "RBBB_COMPLETE", "WPW", "ST_ELEVATION")
                if x in ALL35 and x not in represented_set
            ),
            next(x for x in ALL35 if x not in represented_set),
        )
        r27_only_payload = _payload(
            focus_module=r27_only_module,
            focus_probability=0.95,
        )
        tiled_payload = _payload(
            focus_module=focus,
            focus_probability=0.99,
            tiled=True,
        )

        variants = {
            "support": support_payload,
            "discordance": discord_payload,
            "r27_only": r27_only_payload,
            "tiled": tiled_payload,
        }
        variant_rows: dict[str, Any] = {}

        for name, payload in variants.items():
            before = _snapshot(structured)
            final = compose_final_report({}, structured, payload)
            after = _snapshot(structured)
            if before != after or before != baseline_snapshot:
                raise AssertionError(
                    f"R27 mutated MEDCALC structured report for HR={hr}, variant={name}"
                )

            idx = str(final.get("idx") or "")
            conclusion = _line(str(final.get("text") or ""), "CONCLUSIÓN:")
            if idx != baseline_idx:
                raise AssertionError(
                    f"R27 changed IDX for HR={hr}, variant={name}: "
                    f"{baseline_idx!r} -> {idx!r}"
                )
            if conclusion != baseline_conclusion:
                raise AssertionError(
                    f"R27 changed conclusion for HR={hr}, variant={name}"
                )

            qa = final.get("r27_consensus") or {}
            if qa.get("measurement_mutation_allowed") is not False:
                raise AssertionError("R27 measurement mutation must remain false")
            if qa.get("diagnostic_mutation_allowed") is not False:
                raise AssertionError("R27 diagnostic mutation must remain false")

            variant_rows[name] = {
                "idx": idx,
                "conclusion": conclusion,
                "cross_engine_support_n": int(
                    qa.get("cross_engine_support_n") or 0
                ),
                "discordance_review_n": int(
                    qa.get("discordance_review_n") or 0
                ),
                "r27_only_review_n": len(
                    qa.get("r27_only_review_signals") or []
                ),
                "comparison_statuses": [
                    str(x.get("status") or "")
                    for x in qa.get("comparisons") or []
                ],
                "measurement_mutation_allowed": qa.get(
                    "measurement_mutation_allowed"
                ),
                "diagnostic_mutation_allowed": qa.get(
                    "diagnostic_mutation_allowed"
                ),
            }

        if represented:
            if variant_rows["support"]["cross_engine_support_n"] < 1:
                raise AssertionError(
                    f"Expected support audit for mapped module {focus}, HR={hr}"
                )
            if variant_rows["discordance"]["discordance_review_n"] < 1:
                raise AssertionError(
                    f"Expected discordance audit for mapped module {focus}, HR={hr}"
                )
            if (
                focus in TEMPORAL_MODULES
                and "NOT_COMPARABLE"
                not in variant_rows["tiled"]["comparison_statuses"]
            ):
                raise AssertionError(
                    f"Tiled temporal R27 module {focus} was not suppressed"
                )

        if variant_rows["r27_only"]["r27_only_review_n"] < 1:
            raise AssertionError(
                f"Expected R27-only review signal for {r27_only_module}, HR={hr}"
            )

        candidates = measured.get("high_recall_candidates") or {}
        gates = measured.get("domain_gates") or {}
        fusion = measured.get("evidence_fusion") or {}
        reasoner = measured.get("specialist_reasoning") or {}

        rows.append({
            "heart_rate_bpm": hr,
            "candidate_n": len(candidates.get("candidates") or []),
            "domain_gate_n": len(gates.get("domains") or {}),
            "fusion_finding_n": len(fusion.get("findings") or []),
            "fusion_publishable_n": len(
                fusion.get("publishable_findings") or []
            ),
            "reasoner_publication_allowed": bool(
                reasoner.get("publication_allowed")
            ),
            "diagnostic_summary": structured.get("diagnostic_summary"),
            "baseline_idx": baseline_idx,
            "baseline_conclusion": baseline_conclusion,
            "mapped_r27_modules": represented,
            "qa_variants": variant_rows,
        })

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "scope": "DEVELOPMENT_DIAGNOSTIC_PIPELINE_AND_R27_INVARIANCE",
        "clinical_validation_claim_allowed": False,
        "tuning_role": "DEVELOPMENT_ONLY",
        "policy": (
            "R27 executes only after MEDCALC candidate/domain-gate/fusion/reasoner/"
            "report generation and cannot mutate measurements, findings, conclusion, "
            "or IDX."
        ),
        "case_n": len(rows),
        "cases": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
