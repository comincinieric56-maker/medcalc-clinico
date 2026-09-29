from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from ecg_av_conduction import analyze_av_conduction
from ecg_candidate_detectors import _av_sequence_candidates, build_high_recall_candidates
from ecg_consistency_engine import evaluate_ecg_consistency
from ecg_crosslead_conduction import analyze_crosslead_conduction
from ecg_domain_gating import build_domain_gates
from ecg_reasoner import reason_ecg
from ecg_signal_measurements import analyze_canonical_ecg


REGRESSION_VERSION = "MEDCALC_ECG_KNOWN_CASE_REGRESSION_V1"
ALLOWED_ROLE = "DEVELOPMENT_REGRESSION_ONLY"


def _metric(value: float | None, confidence: float = 0.9) -> Dict[str, Any]:
    return {
        "value": value,
        "confidence": confidence if value is not None else 0.0,
        "status": "MEASURED" if value is not None else "NOT_MEASURABLE",
    }


def _load_provenance(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _provenance_sets(registry: Dict[str, Any]) -> tuple[set[str], set[str]]:
    development = {
        str(row.get("id") or "")
        for row in registry.get("development_contaminated") or []
        if str(row.get("id") or "")
    }
    forbidden = {
        str(row.get("id") or "")
        for section in ("external_baseline_consumed", "provisional_external_locked")
        for row in registry.get(section) or []
        if str(row.get("id") or "")
    }
    return development, forbidden


def validate_case_provenance(
    case: Dict[str, Any],
    registry: Dict[str, Any],
) -> None:
    source = dict(case.get("source") or {})
    dataset_id = str(source.get("dataset_id") or "").strip()
    role = str(source.get("usage_role") or "").strip()
    case_type = str(case.get("case_type") or "").strip()

    if role != ALLOWED_ROLE:
        raise ValueError(
            f"{case.get('case_id')}: usage_role must be {ALLOWED_ROLE}"
        )

    if case_type == "SYNTHETIC_SCENARIO":
        if dataset_id not in {"synthetic", "generated"}:
            raise ValueError(
                f"{case.get('case_id')}: synthetic case must use synthetic/generated source"
            )
        return

    development, forbidden = _provenance_sets(registry)
    if dataset_id in forbidden:
        raise ValueError(
            f"{case.get('case_id')}: dataset {dataset_id!r} is frozen/external and "
            "cannot enter the regression corpus"
        )
    if dataset_id not in development:
        raise ValueError(
            f"{case.get('case_id')}: dataset {dataset_id!r} is not registered as "
            "development-contaminated; fail closed"
        )

    if dataset_id == "ptb_xl":
        fold = source.get("fold")
        try:
            fold = int(fold)
        except Exception as exc:
            raise ValueError(
                f"{case.get('case_id')}: PTB-XL replay requires explicit fold"
            ) from exc
        if fold not in range(1, 9):
            raise ValueError(
                f"{case.get('case_id')}: PTB-XL fold {fold} is forbidden for "
                "known-case replay; only folds 1-8 may be used"
            )

    if not str(source.get("record_ref") or "").strip():
        raise ValueError(
            f"{case.get('case_id')}: real development replay requires record_ref"
        )

    if case_type == "CANONICAL_ECG_JSON":
        if not str(source.get("fixture_path") or "").strip():
            raise ValueError(
                f"{case.get('case_id')}: CANONICAL_ECG_JSON requires fixture_path"
            )
    elif case_type == "PTBXL_REMOTE_RECORD":
        if dataset_id != "ptb_xl":
            raise ValueError(
                f"{case.get('case_id')}: PTBXL_REMOTE_RECORD requires dataset_id=ptb_xl"
            )
        try:
            int(source.get("record_ref"))
        except Exception as exc:
            raise ValueError(
                f"{case.get('case_id')}: PTBXL_REMOTE_RECORD record_ref must be ecg_id"
            ) from exc
    else:
        raise ValueError(
            f"{case.get('case_id')}: unsupported real case_type {case_type!r}"
        )


def _scenario_bbb_preexcitation_warning() -> Dict[str, Any]:
    graph = {
        "global": {},
        "specialist_evidence": {
            "atrial_activity": {},
            "atrial_mechanism": {},
            "wide_complex_tachycardia": {},
            "ectopy": {},
        },
        "rhythm": {},
        "relations": {},
    }
    rbbb = {
        "code": "RBBB_MORPHOLOGY_COMPATIBLE",
        "domain": "BUNDLE_BRANCH",
        "publishable": True,
        "score": 0.88,
        "evidence": [
            "QRS_GE_120MS",
            "V1_R_PRIME_OR_TERMINAL_POSITIVE",
            "LATERAL_TERMINAL_S",
        ],
        "fusion_state": "ESTABLISHED_COMPATIBLE",
    }
    pre = {
        "code": "VENTRICULAR_PREEXCITATION_COMPATIBLE",
        "domain": "PREEXCITATION",
        "publishable": True,
        "score": 0.82,
        "evidence": ["SHORT_PR", "MULTILEAD_DELTA_SLUR", "QRS_GE_110MS"],
        "fusion_state": "ESTABLISHED_COMPATIBLE",
    }
    fusion = {
        "findings": [rbbb, pre],
        "by_code": {
            rbbb["code"]: rbbb,
            pre["code"]: pre,
        },
        "publishable_findings": [rbbb, pre],
    }
    consistency = {
        "status": "PASS_WITH_WARNINGS",
        "blocking_conflict": False,
        "conflicts": [{
            "code": "PREEXCITATION_CONFOUNDS_BUNDLE_BRANCH_PATTERN",
            "severity": "WARNING",
            "action": "REPORT_COEXISTING_PATTERNS_WITH_CONFOUNDING_REVIEW",
        }],
    }
    reasoned = reason_ecg(
        graph,
        {"findings": []},
        consistency,
        domain_gates={
            "domains": {
                "RHYTHM": {"eligible": False},
                "BUNDLE_BRANCH": {"eligible": True},
                "PREEXCITATION": {"eligible": True},
                "ECTOPY": {"eligible": False},
            }
        },
        evidence_fusion=fusion,
    )
    return {
        "analysis": {
            "specialist_reasoning": reasoned,
            "consistency": consistency,
        }
    }


def _scenario_fast_two_to_one_mapping() -> Dict[str, Any]:
    per_lead = {
        "II": {
            "evaluable": True,
            "fs": 500,
            "confidence": 0.95,
            "raw_p_peaks_samples": [50, 200, 350, 500, 650, 800],
            "r_peaks_samples": [100, 400, 700],
            "atrial_activity": {
                "p_candidate_n": 6,
                "p_wave_reproducible": True,
                "p_qrs_coupling_fraction": 0.50,
            },
        },
    }
    av = analyze_av_conduction(
        per_lead,
        {
            "p_wave_reproducible": True,
            "rhythm_p_qrs_coupling_fraction": 0.50,
        },
        global_metrics={"pr_ms": _metric(None)},
    )
    candidates = _av_sequence_candidates(per_lead)
    return {
        "analysis": {
            "av_conduction": av,
            "high_recall_candidates": {
                "candidates": candidates,
                "by_code": {
                    str(row.get("code") or ""): row
                    for row in candidates
                },
            },
        }
    }


def _scenario_rbbb_three_wide_leads_morphology_rescue() -> Dict[str, Any]:
    graph = {
        "global": {
            "qrs_ms": {
                "value": 118.0,
                "confidence": 0.30,
                "status": "REMEASURE",
            }
        },
        "specialist_evidence": {
            "measurement_consensus": {
                "remeasure_targets": ["qrs_ms"],
                "unusable_targets": ["qrs_ms"],
                "unmeasurable_targets": [],
                "uncertain_targets": ["qrs_ms"],
            },
            "qrs_morphology": {
                "per_lead": {
                    "V1": {
                        "evaluable": True,
                        "duration_ms": 130.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.15,
                        "terminal_negative_mv": -0.02,
                    },
                    "V2": {
                        "evaluable": True,
                        "duration_ms": 128.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.12,
                        "terminal_negative_mv": -0.02,
                    },
                    "I": {
                        "evaluable": True,
                        "duration_ms": 126.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.12,
                        "terminal_s_duration_ms": 40.0,
                        "terminal_positive_mv": 0.04,
                    },
                    "V6": {
                        "evaluable": True,
                        "duration_ms": 116.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.11,
                        "terminal_s_duration_ms": 38.0,
                        "terminal_positive_mv": 0.04,
                    },
                }
            },
            "fascicular_conduction": {},
        },
        "rhythm": {},
        "relations": {},
    }
    cross = analyze_crosslead_conduction(graph)
    consistency = evaluate_ecg_consistency(graph, cross)
    gates = build_domain_gates(graph, cross, consistency)
    candidates = build_high_recall_candidates(graph, cross, {})
    return {
        "analysis": {
            "crosslead_conduction": cross,
            "consistency": consistency,
            "domain_gates": gates,
            "high_recall_candidates": candidates,
        }
    }


def _scenario_rbbb_ge4_wide_wrong_distribution_no_rescue() -> Dict[str, Any]:
    graph = {
        "global": {
            "qrs_ms": {
                "value": 118.0,
                "confidence": 0.30,
                "status": "REMEASURE",
            }
        },
        "specialist_evidence": {
            "measurement_consensus": {
                "remeasure_targets": ["qrs_ms"],
                "unusable_targets": ["qrs_ms"],
                "unmeasurable_targets": [],
                "uncertain_targets": ["qrs_ms"],
            },
            "qrs_morphology": {
                "per_lead": {
                    "I": {
                        "evaluable": True,
                        "duration_ms": 128.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.12,
                        "terminal_s_duration_ms": 40.0,
                        "terminal_positive_mv": 0.04,
                    },
                    "II": {"evaluable": True, "duration_ms": 126.0},
                    "III": {"evaluable": True, "duration_ms": 124.0},
                    "aVL": {"evaluable": True, "duration_ms": 122.0},
                    "V1": {
                        "evaluable": True,
                        "duration_ms": 116.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.15,
                        "terminal_negative_mv": -0.02,
                    },
                    "V2": {
                        "evaluable": True,
                        "duration_ms": 116.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.12,
                        "terminal_negative_mv": -0.02,
                    },
                    "V6": {
                        "evaluable": True,
                        "duration_ms": 116.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.11,
                        "terminal_s_duration_ms": 38.0,
                        "terminal_positive_mv": 0.04,
                    },
                }
            },
            "fascicular_conduction": {},
        },
        "rhythm": {},
        "relations": {},
    }
    cross = analyze_crosslead_conduction(graph)
    consistency = evaluate_ecg_consistency(graph, cross)
    gates = build_domain_gates(graph, cross, consistency)
    candidates = build_high_recall_candidates(graph, cross, {})
    return {
        "analysis": {
            "crosslead_conduction": cross,
            "consistency": consistency,
            "domain_gates": gates,
            "high_recall_candidates": candidates,
        }
    }


def _scenario_multilead_qrs_rescue() -> Dict[str, Any]:
    graph = {
        "global": {
            "qrs_ms": {
                "value": 116.0,
                "confidence": 0.30,
                "status": "REMEASURE",
            }
        },
        "specialist_evidence": {
            "measurement_consensus": {
                "remeasure_targets": ["qrs_ms"],
                "unusable_targets": ["qrs_ms"],
                "unmeasurable_targets": [],
                "uncertain_targets": ["qrs_ms"],
            },
            "qrs_morphology": {
                "per_lead": {
                    "V1": {
                        "evaluable": True,
                        "duration_ms": 130.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.15,
                        "terminal_negative_mv": -0.02,
                    },
                    "V2": {
                        "evaluable": True,
                        "duration_ms": 128.0,
                        "r_prime_present": True,
                        "qrs_polarity": "R_DOMINANT",
                        "terminal_positive_mv": 0.12,
                        "terminal_negative_mv": -0.02,
                    },
                    "I": {
                        "evaluable": True,
                        "duration_ms": 126.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.12,
                        "terminal_s_duration_ms": 40.0,
                        "terminal_positive_mv": 0.04,
                    },
                    "V6": {
                        "evaluable": True,
                        "duration_ms": 124.0,
                        "qrs_polarity": "BIPHASIC",
                        "terminal_negative_mv": -0.11,
                        "terminal_s_duration_ms": 38.0,
                        "terminal_positive_mv": 0.04,
                    },
                }
            },
            "fascicular_conduction": {},
        },
        "rhythm": {},
        "relations": {},
    }
    cross = analyze_crosslead_conduction(graph)
    consistency = evaluate_ecg_consistency(graph, cross)
    gates = build_domain_gates(graph, cross, consistency)
    candidates = build_high_recall_candidates(graph, cross, {})
    return {
        "analysis": {
            "crosslead_conduction": cross,
            "consistency": consistency,
            "domain_gates": gates,
            "high_recall_candidates": candidates,
        }
    }


SYNTHETIC_SCENARIOS = {
    "BBB_PREEXCITATION_WARNING": _scenario_bbb_preexcitation_warning,
    "FAST_TWO_TO_ONE_MAPPING": _scenario_fast_two_to_one_mapping,
    "MULTILEAD_QRS_RESCUE": _scenario_multilead_qrs_rescue,
    "RBBB_THREE_WIDE_LEADS_MORPHOLOGY_RESCUE": _scenario_rbbb_three_wide_leads_morphology_rescue,
    "RBBB_GE4_WIDE_WRONG_DISTRIBUTION_NO_RESCUE": _scenario_rbbb_ge4_wide_wrong_distribution_no_rescue,
}


def _load_real_canonical_case(case: Dict[str, Any], root: Path) -> Dict[str, Any]:
    source = dict(case.get("source") or {})
    relative = str(source.get("fixture_path") or "").strip()
    if not relative:
        raise ValueError(
            f"{case.get('case_id')}: CANONICAL_ECG_JSON requires fixture_path"
        )
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"{case.get('case_id')}: fixture_path escapes repository")
    canonical = json.loads(path.read_text(encoding="utf-8"))
    return {"analysis": analyze_canonical_ecg(canonical)}


def _load_ptbxl_remote_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Download one frozen PTB-XL development record by ecg_id and replay it."""
    import pandas as pd
    import wfdb

    from ecg_adult_diagnostic_dev_benchmark import (
        BASE,
        _canonical,
        _download,
        _ensure_record,
    )

    source = dict(case.get("source") or {})
    ecg_id = int(source.get("record_ref"))
    expected_fold = int(source.get("fold"))

    with tempfile.TemporaryDirectory(prefix="medcalc-known-ptbxl-") as td:
        root = Path(td)
        metadata_path = root / "ptbxl_database.csv"
        _download(f"{BASE}/ptbxl_database.csv", metadata_path)
        meta = pd.read_csv(metadata_path)
        rows = meta.loc[meta["ecg_id"].astype(int) == ecg_id]
        if len(rows) != 1:
            raise ValueError(
                f"{case.get('case_id')}: PTB-XL ecg_id {ecg_id} not uniquely found"
            )
        row = rows.iloc[0]
        actual_fold = int(row["strat_fold"])
        if actual_fold != expected_fold:
            raise ValueError(
                f"{case.get('case_id')}: expected fold {expected_fold}, metadata says {actual_fold}"
            )
        if actual_fold not in range(1, 9):
            raise ValueError(
                f"{case.get('case_id')}: PTB-XL fold {actual_fold} is forbidden for replay"
            )

        filename_hr = str(row["filename_hr"])
        local_base = _ensure_record(root / "records", filename_hr)
        rec = wfdb.rdrecord(str(local_base))
        canonical = _canonical(
            rec.p_signal,
            int(round(float(rec.fs))),
            list(rec.sig_name),
            ecg_id,
        )
        analysis = analyze_canonical_ecg(canonical)
        return {
            "analysis": analysis,
            "source_audit": {
                "dataset_id": "ptb_xl",
                "ecg_id": ecg_id,
                "fold": actual_fold,
                "filename_hr": filename_hr,
                "usage_role": ALLOWED_ROLE,
                "validation_claim_allowed": False,
            },
        }


def _codes_from_analysis(analysis: Dict[str, Any]) -> Dict[str, set[str]]:
    reasoned = analysis.get("specialist_reasoning") or {}
    findings = ((reasoned.get("diagnostic_summary") or {}).get("findings") or [])
    final_codes = {
        str(row.get("code") or "")
        for row in findings
        if str(row.get("code") or "")
    }

    candidates = analysis.get("high_recall_candidates") or {}
    candidate_codes = {
        str(row.get("code") or "")
        for row in (candidates.get("candidates") or [])
        if str(row.get("code") or "")
    }

    conflicts = analysis.get("consistency") or {}
    warning_codes = {
        str(row.get("code") or "")
        for row in (conflicts.get("conflicts") or [])
        if str(row.get("severity") or "") == "WARNING"
    }
    return {
        "final_codes": final_codes,
        "candidate_codes": candidate_codes,
        "warning_codes": warning_codes,
    }


def _get_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _assert_expected(
    case: Dict[str, Any],
    payload: Dict[str, Any],
) -> list[str]:
    analysis = dict(payload.get("analysis") or {})
    codes = _codes_from_analysis(analysis)
    expected = dict(case.get("expected") or {})
    failures: list[str] = []

    for field, actual_key in (
        ("final_codes_all", "final_codes"),
        ("candidate_codes_all", "candidate_codes"),
        ("warning_codes_all", "warning_codes"),
    ):
        wanted = {str(x) for x in expected.get(field) or []}
        missing = wanted - codes[actual_key]
        if missing:
            failures.append(f"{field} missing {sorted(missing)}")

    forbidden = {str(x) for x in expected.get("final_codes_none") or []}
    present_forbidden = forbidden & codes["final_codes"]
    if present_forbidden:
        failures.append(
            f"final_codes_none unexpectedly present {sorted(present_forbidden)}"
        )

    findings = (
        ((analysis.get("specialist_reasoning") or {})
         .get("diagnostic_summary") or {})
        .get("findings") or []
    )
    by_final_code = {
        str(row.get("code") or ""): dict(row)
        for row in findings
        if str(row.get("code") or "")
    }
    for check in expected.get("final_finding_checks") or []:
        check = dict(check or {})
        code = str(check.get("code") or "")
        row = by_final_code.get(code)
        if row is None:
            failures.append(f"final_finding_checks missing code {code!r}")
            continue
        for key, wanted in dict(check.get("equals") or {}).items():
            actual = row.get(key)
            if actual != wanted:
                failures.append(
                    f"final finding {code}.{key}: expected {wanted!r}, got {actual!r}"
                )
        for key, wanted_items in dict(check.get("contains") or {}).items():
            actual_items = row.get(key) or []
            if not isinstance(actual_items, (list, tuple, set)):
                failures.append(
                    f"final finding {code}.{key}: expected list-like value"
                )
                continue
            missing_items = set(wanted_items or []) - set(actual_items)
            if missing_items:
                failures.append(
                    f"final finding {code}.{key}: missing {sorted(missing_items)}"
                )

    for path, wanted in dict(expected.get("attributes") or {}).items():
        actual = _get_path(analysis, str(path))
        if actual != wanted:
            failures.append(
                f"attribute {path}: expected {wanted!r}, got {actual!r}"
            )
    return failures


def run_manifest(
    manifest_path: Path,
    provenance_path: Path,
    root: Path,
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry = _load_provenance(provenance_path)

    if str(manifest.get("role") or "") != ALLOWED_ROLE:
        raise ValueError(f"manifest role must be {ALLOWED_ROLE}")
    if bool(manifest.get("report_as_validation")):
        raise ValueError("regression corpus can never be reported as validation")

    results: list[Dict[str, Any]] = []
    for case in manifest.get("cases") or []:
        validate_case_provenance(case, registry)
        case_type = str(case.get("case_type") or "")
        if case_type == "SYNTHETIC_SCENARIO":
            scenario = str(case.get("scenario") or "")
            fn = SYNTHETIC_SCENARIOS.get(scenario)
            if fn is None:
                raise ValueError(
                    f"{case.get('case_id')}: unknown synthetic scenario {scenario!r}"
                )
            payload = fn()
        elif case_type == "CANONICAL_ECG_JSON":
            payload = _load_real_canonical_case(case, root)
        elif case_type == "PTBXL_REMOTE_RECORD":
            payload = _load_ptbxl_remote_case(case)
        else:
            raise ValueError(
                f"{case.get('case_id')}: unsupported case_type {case_type!r}"
            )

        failures = _assert_expected(case, payload)
        results.append({
            "case_id": case.get("case_id"),
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
            "source": case.get("source"),
            "case_type": case_type,
        })

    failed = [row for row in results if row["status"] != "PASS"]
    return {
        "version": REGRESSION_VERSION,
        "role": ALLOWED_ROLE,
        "report_as_validation": False,
        "case_n": len(results),
        "pass_n": len(results) - len(failed),
        "fail_n": len(failed),
        "status": "PASS" if not failed else "FAIL",
        "results": results,
        "invariant": (
            "KNOWN_CASE_REPLAY_IS_ENGINEERING_REGRESSION_ONLY_AND_MUST_NEVER_"
            "BE_REPORTED_AS_INTERNAL_OR_EXTERNAL_VALIDATION"
        ),
    }


def _guard_selftest(provenance_path: Path) -> None:
    registry = _load_provenance(provenance_path)
    invalid = [
        {
            "case_id": "FORBID_FOLD9",
            "case_type": "PTBXL_REMOTE_RECORD",
            "source": {
                "dataset_id": "ptb_xl",
                "fold": 9,
                "record_ref": "100",
                "usage_role": ALLOWED_ROLE,
            },
        },
        {
            "case_id": "FORBID_FOLD10",
            "case_type": "PTBXL_REMOTE_RECORD",
            "source": {
                "dataset_id": "ptb_xl",
                "fold": 10,
                "record_ref": "101",
                "usage_role": ALLOWED_ROLE,
            },
        },
        {
            "case_id": "FORBID_EXTERNAL",
            "case_type": "CANONICAL_ECG_JSON",
            "source": {
                "dataset_id": "sph",
                "record_ref": "x",
                "fixture_path": "x.json",
                "usage_role": ALLOWED_ROLE,
            },
        },
        {
            "case_id": "FORBID_LOCKED_EXTERNAL",
            "case_type": "CANONICAL_ECG_JSON",
            "source": {
                "dataset_id": "heedb",
                "record_ref": "x",
                "fixture_path": "x.json",
                "usage_role": ALLOWED_ROLE,
            },
        },
    ]
    for case in invalid:
        try:
            validate_case_provenance(case, registry)
        except ValueError:
            continue
        raise AssertionError(f"provenance guard accepted forbidden case: {case}")

    allowed = {
        "case_id": "ALLOW_PTBXL_FOLD1",
        "case_type": "PTBXL_REMOTE_RECORD",
        "source": {
            "dataset_id": "ptb_xl",
            "fold": 1,
            "record_ref": "102",
            "usage_role": ALLOWED_ROLE,
        },
    }
    validate_case_provenance(allowed, registry)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("ecg_regression_corpus.json"),
    )
    ap.add_argument(
        "--provenance",
        type=Path,
        default=Path("ecg_dataset_provenance.json"),
    )
    ap.add_argument("--output", type=Path)
    ap.add_argument("--guard-selftest", action="store_true")
    args = ap.parse_args()

    if args.guard_selftest:
        _guard_selftest(args.provenance)
        print("MEDCALC_ECG_KNOWN_CASE_PROVENANCE_GUARD_PASS")
        return

    result = run_manifest(
        args.manifest,
        args.provenance,
        Path("."),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
