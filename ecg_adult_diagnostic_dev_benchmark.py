from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_signal_measurements import analyze_canonical_ecg

PTBXL_VERSION = "1.0.3"
BASE = f"https://physionet.org/files/ptb-xl/{PTBXL_VERSION}"
BENCHMARK_VERSION = "MEDCALC_ADULT_DIAGNOSTIC_DEVELOPMENT_PTBXL_V1"
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

TARGETS: dict[str, dict[str, Any]] = {
    "AF": {
        "scp": {"AFIB", "AF"},
        "medcalc": {"AF_COMPATIBLE"},
    },
    "FLUTTER": {
        "scp": {"AFLT", "AFLUT", "AFL"},
        "medcalc": {"FLUTTER_OR_AT_COMPATIBLE"},
    },
    "SINUS_BRADY": {
        "scp": {"SBRAD", "SB"},
        "medcalc": {"SINUS_BRADYCARDIA_COMPATIBLE"},
    },
    "SINUS_TACHY": {
        "scp": {"STACH", "ST"},
        "medcalc": {"SINUS_TACHYCARDIA_COMPATIBLE"},
    },
    "RBBB_COMPLETE": {
        "scp": {"RBBB", "CRBBB"},
        "medcalc": {"RBBB_MORPHOLOGY_COMPATIBLE"},
    },
    "LBBB": {
        "scp": {"LBBB", "CLBBB"},
        "medcalc": {"LBBB_MORPHOLOGY_COMPATIBLE"},
    },
    "LAFB": {
        "scp": {"LAFB", "LAnFB"},
        "medcalc": {"LAFB_COMPATIBLE"},
    },
    "LPFB": {
        "scp": {"LPFB"},
        "medcalc": {"LPFB_COMPATIBLE"},
    },
    "AVB1": {
        "scp": {"1AVB", "IAVB"},
        "medcalc": {"FIRST_DEGREE_AV_DELAY_COMPATIBLE"},
    },
    "AVB2": {
        "scp": {"2AVB", "IIAVB"},
        "medcalc": {
            "MOBITZ_I_WENCKEBACH_COMPATIBLE",
            "MOBITZ_II_COMPATIBLE",
            "TWO_TO_ONE_AV_BLOCK_COMPATIBLE",
            "HIGH_GRADE_AV_BLOCK_COMPATIBLE",
        },
    },
    "AVB3": {
        "scp": {"3AVB", "IIIAVB", "CAVB"},
        "medcalc": {"COMPLETE_AV_BLOCK_COMPATIBLE"},
    },
    "WPW": {
        "scp": {"WPW", "PREX"},
        "medcalc": {"VENTRICULAR_PREEXCITATION_COMPATIBLE"},
    },
}

MIN_LABEL_LIKELIHOOD = 80.0
MAX_POS_PER_TARGET = 80
NEGATIVE_CONTROL_N = 400
INTERNAL_VALIDATION_FOLD = 9
TARGET_CANDIDATE_SENSITIVITY = 0.97
TARGET_FINAL_SENSITIVITY = 0.90
SPECIFICITY_GUARDRAIL = 0.90
MIN_POSITIVE_N_FOR_GATE = 20


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "MEDCALC-ECG-development/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)


def _hash(value: str) -> str:
    return hashlib.sha256(f"MEDCALC_ADULT_PTBXL_V1|{value}".encode()).hexdigest()


def _parse_codes(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except Exception:
            return {}
    out: dict[str, float] = {}
    if not isinstance(raw, dict):
        return out
    for key, val in raw.items():
        try:
            score = float(val)
        except Exception:
            score = 0.0
        out[str(key).strip()] = score
    return out


def _target_positive(codes: dict[str, float], aliases: set[str]) -> bool:
    upper = {str(k).upper(): float(v) for k, v in codes.items()}
    return any(upper.get(a.upper(), 0.0) >= MIN_LABEL_LIKELIHOOD for a in aliases)


def _any_target_positive(codes: dict[str, float]) -> bool:
    return any(_target_positive(codes, spec["scp"]) for spec in TARGETS.values())


def _adult_rows(df: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["_age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["_fold"] = pd.to_numeric(out.get("strat_fold"), errors="coerce")
    fold_set = {int(x) for x in folds}
    out = out[(out["_age"] >= 18.0) & (out["_fold"].isin(fold_set))].copy()
    out["_codes"] = out["scp_codes"].map(_parse_codes)
    out["_hash"] = out["ecg_id"].astype(str).map(_hash)
    return out


def select_records(
    df: pd.DataFrame,
    folds: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = [INTERNAL_VALIDATION_FOLD] if folds is None else [int(x) for x in folds]
    adult = _adult_rows(df, folds)
    selected_ids: set[int] = set()
    target_ids: dict[str, list[int]] = {}
    availability: dict[str, int] = {}

    for target, spec in TARGETS.items():
        pos = adult[adult["_codes"].map(lambda x, a=spec["scp"]: _target_positive(x, a))].copy()
        pos = pos.sort_values(["_hash", "ecg_id"])
        availability[target] = int(len(pos))
        ids = [int(x) for x in pos["ecg_id"].head(MAX_POS_PER_TARGET).tolist()]
        target_ids[target] = ids
        selected_ids.update(ids)

    neg = adult[~adult["_codes"].map(_any_target_positive)].copy()
    neg = neg.sort_values(["_hash", "ecg_id"]).head(NEGATIVE_CONTROL_N)
    negative_ids = [int(x) for x in neg["ecg_id"].tolist()]
    selected_ids.update(negative_ids)

    selected = adult[adult["ecg_id"].astype(int).isin(selected_ids)].copy()
    selected = selected.sort_values(["_hash", "ecg_id"]).reset_index(drop=True)

    summary = {
        "fold": folds[0] if len(folds) == 1 else None,
        "folds": folds,
        "adult_records_in_fold": int(len(adult)),
        "selected_unique_records": int(len(selected)),
        "negative_control_n": int(len(negative_ids)),
        "positive_available_by_target": availability,
        "positive_selected_by_target": {k: len(v) for k, v in target_ids.items()},
        "target_positive_ecg_ids": target_ids,
        "negative_control_ecg_ids": negative_ids,
        "selection_namespace": "MEDCALC_ADULT_PTBXL_V1",
        "label_likelihood_min": MIN_LABEL_LIKELIHOOD,
        "external_validation_claim_allowed": False,
    }
    return selected, summary


def _lead_name(name: str) -> str:
    raw = str(name or "").strip().upper()
    return {"AVR":"aVR","AVL":"aVL","AVF":"aVF"}.get(raw, raw)


def _canonical(signal: np.ndarray, fs: int, sig_names: list[str], ecg_id: int) -> dict[str, Any]:
    names = [_lead_name(x) for x in sig_names]
    if set(names) != set(LEADS):
        raise ValueError(f"Expected standard 12 leads, got {names}")
    order = [names.index(x) for x in LEADS]
    x = np.asarray(signal, dtype=float)[:, order]
    leads: dict[str, Any] = {}
    for j, lead in enumerate(LEADS):
        y = np.asarray(x[:, j], dtype=float)
        leads[lead] = {
            "lead": lead,
            "signal_mv": [float(v) if math.isfinite(float(v)) else None for v in y],
            "quality_mask": np.where(np.isfinite(y), 2, 0).astype(np.uint8).tolist(),
            "fs": int(fs),
            "duration_s": float(len(y) / fs),
            "source": "PTBXL_DEVELOPMENT_DIGITAL_SIGNAL",
            "confidence": 1.0,
            "status": "MEASURABLE",
        }
    return {
        "version": "MEDCALC_CANONICAL_ECG_SIGNAL_V1",
        "source": "PTBXL_DEVELOPMENT_DIGITAL_SIGNAL",
        "fs": int(fs),
        "lead_order": list(LEADS),
        "leads": leads,
        "calibration": {
            "speed_mm_per_s": 25.0,
            "gain_mm_per_mv": 10.0,
            "confidence": 1.0,
            "source": "NATIVE_DIGITAL_DEVELOPMENT",
        },
        "validation_provenance": {
            "dataset_id": "ptb_xl",
            "ecg_id": int(ecg_id),
            "development_contaminated": True,
            "external_validation_claim_allowed": False,
        },
    }


def _published_codes(analysis: dict[str, Any]) -> set[str]:
    summary = ((analysis.get("specialist_reasoning") or {}).get("diagnostic_summary") or {})
    return {
        str(row.get("code") or "")
        for row in summary.get("findings") or []
        if bool(row.get("publishable"))
    }


def _candidate_codes(analysis: dict[str, Any]) -> set[str]:
    return set(((analysis.get("high_recall_candidates") or {}).get("by_code") or {}).keys())


def _candidate_audit(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_code = ((analysis.get("high_recall_candidates") or {}).get("by_code") or {})
    out: dict[str, dict[str, Any]] = {}
    for code, row_raw in by_code.items():
        row = dict(row_raw or {})
        out[str(code)] = {
            "score": float(row.get("score") or 0.0),
            "evidence": sorted(str(x) for x in (row.get("evidence") or [])),
            "source_groups": sorted(str(x) for x in (row.get("source_groups") or [])),
            "independent_evidence_n": int(row.get("independent_evidence_n") or 0),
            "specialist_confirmed": bool(row.get("specialist_confirmed")),
            "required_measurements": sorted(
                str(x) for x in (row.get("required_measurements") or [])
            ),
        }
    return out


def _fusion_codes(analysis: dict[str, Any]) -> set[str]:
    by_code = ((analysis.get("evidence_fusion") or {}).get("by_code") or {})
    return {str(k) for k, v in by_code.items() if bool((v or {}).get("publishable"))}


def _fusion_audit(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_code = ((analysis.get("evidence_fusion") or {}).get("by_code") or {})
    out: dict[str, dict[str, Any]] = {}
    for code, row_raw in by_code.items():
        row = dict(row_raw or {})
        gate = dict(row.get("domain_gate") or {})
        out[str(code)] = {
            "publishable": bool(row.get("publishable")),
            "fusion_state": str(row.get("fusion_state") or ""),
            "fusion_reason": str(row.get("fusion_reason") or ""),
            "score": float(row.get("score") or 0.0),
            "prospective_score_threshold": float(row.get("prospective_score_threshold") or 0.0),
            "independent_evidence_n": int(row.get("independent_evidence_n") or 0),
            "prospective_min_independent_sources": int(row.get("prospective_min_independent_sources") or 0),
            "unresolved_required_measurements": sorted(
                str(x) for x in (row.get("unresolved_required_measurements") or [])
            ),
            "boundary_failure_metrics": sorted(
                str(x.get("metric") or "")
                for x in (row.get("boundary_failures") or [])
                if str(x.get("metric") or "")
            ),
            "blocked_by_conflicts": sorted(
                str(x) for x in (gate.get("blocked_by_conflicts") or [])
            ),
            "gate_unusable_measurements": sorted(
                str(x) for x in (gate.get("unusable_measurements") or [])
            ),
        }
    return out


def _reasoner_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    reasoning = dict(analysis.get("specialist_reasoning") or {})
    summary = dict(reasoning.get("diagnostic_summary") or {})
    consistency = dict(analysis.get("consistency") or {})
    pre = dict(reasoning.get("preexcitation_finding") or {})
    conduction = [
        str(row.get("code") or "")
        for row in (reasoning.get("conduction_findings") or [])
        if str(row.get("code") or "")
    ]
    abstentions = [
        {
            "domain": str(row.get("domain") or ""),
            "reason": str(row.get("reason") or ""),
            "conflicts": sorted(str(x) for x in (row.get("conflicts") or [])),
            "remeasure_targets": sorted(str(x) for x in (row.get("remeasure_targets") or [])),
        }
        for row in (summary.get("abstentions") or [])
    ]
    blocking_conflicts = sorted(
        str(row.get("code") or "")
        for row in (consistency.get("conflicts") or [])
        if str(row.get("severity") or "") == "BLOCKING" and str(row.get("code") or "")
    )
    warning_conflicts = sorted(
        str(row.get("code") or "")
        for row in (consistency.get("conflicts") or [])
        if str(row.get("severity") or "") == "WARNING" and str(row.get("code") or "")
    )
    return {
        "preexcitation_published": bool(pre),
        "preexcitation_code": str(pre.get("code") or ""),
        "conduction_findings": sorted(conduction),
        "blocking_conflicts": blocking_conflicts,
        "warning_conflicts": warning_conflicts,
        "abstentions": abstentions,
    }


def _record_local_path(root: Path, filename_hr: str) -> Path:
    rel = Path(str(filename_hr))
    return root / rel


def _ensure_record(root: Path, filename_hr: str) -> Path:
    base = _record_local_path(root, filename_hr)
    for suffix in (".hea", ".dat"):
        rel = f"{filename_hr}{suffix}"
        _download(f"{BASE}/{rel}", root / rel)
    return base


def _score_target(
    target: str,
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    negative_ids: set[int],
) -> dict[str, Any]:
    positives = [
        r for r in rows
        if _target_positive(r["codes"], spec["scp"])
    ]
    negatives = [r for r in rows if int(r["ecg_id"]) in negative_ids]

    def hit(r: dict[str, Any], layer: str) -> bool:
        codes = set(r[layer])
        return bool(codes & set(spec["medcalc"]))

    n = len(positives)
    candidate_n = sum(hit(r, "candidate_codes") for r in positives)
    fusion_n = sum(hit(r, "fusion_codes") for r in positives)
    final_n = sum(hit(r, "published_codes") for r in positives)
    fp = sum(hit(r, "published_codes") for r in negatives)

    candidate_sens = candidate_n / n if n else None
    fusion_sens = fusion_n / n if n else None
    final_sens = final_n / n if n else None
    specificity = (len(negatives) - fp) / len(negatives) if negatives else None
    gate_eligible = n >= MIN_POSITIVE_N_FOR_GATE

    candidate_miss_n = n - candidate_n
    candidate_to_fusion_loss_n = max(candidate_n - fusion_n, 0)
    fusion_to_final_loss_n = max(fusion_n - final_n, 0)

    suppression_reasons: dict[str, int] = {}
    suppression_states: dict[str, int] = {}
    boundary_failure_metrics: dict[str, int] = {}
    unresolved_measurements: dict[str, int] = {}
    blocked_conflicts: dict[str, int] = {}
    insufficient_score_n = 0
    insufficient_sources_n = 0
    candidate_evidence_counts: dict[str, int] = {}
    candidate_source_group_counts: dict[str, int] = {}
    candidate_evidence_signatures: dict[str, int] = {}
    candidate_source_group_signatures: dict[str, int] = {}
    candidate_specialist_confirmed_n = 0

    reasoner_preexcitation_suppression_n = 0
    reasoner_preexcitation_suppression_reference_wpw_n = 0
    reasoner_preexcitation_suppression_without_reference_wpw_n = 0
    reasoner_blocking_conflicts: dict[str, int] = {}
    reasoner_warning_conflicts: dict[str, int] = {}
    reasoner_abstention_domains: dict[str, int] = {}
    reasoner_abstention_reasons: dict[str, int] = {}

    expected_codes = set(spec["medcalc"])
    for r in positives:
        candidate_hits = expected_codes & set(r["candidate_codes"])
        if not candidate_hits:
            continue
        audit_map = r.get("fusion_audit") or {}
        audited = [dict(audit_map.get(code) or {}) for code in candidate_hits]
        if any(bool(a.get("publishable")) for a in audited):
            continue

        candidate_map = r.get("candidate_audit") or {}
        for code in sorted(candidate_hits):
            c = dict(candidate_map.get(code) or {})
            evidence = sorted(str(x) for x in (c.get("evidence") or []))
            source_groups = sorted(str(x) for x in (c.get("source_groups") or []))
            for item in evidence:
                candidate_evidence_counts[item] = candidate_evidence_counts.get(item, 0) + 1
            for item in source_groups:
                candidate_source_group_counts[item] = candidate_source_group_counts.get(item, 0) + 1
            evidence_sig = "|".join(evidence) if evidence else "<NONE>"
            source_sig = "|".join(source_groups) if source_groups else "<NONE>"
            candidate_evidence_signatures[evidence_sig] = candidate_evidence_signatures.get(evidence_sig, 0) + 1
            candidate_source_group_signatures[source_sig] = candidate_source_group_signatures.get(source_sig, 0) + 1
            if bool(c.get("specialist_confirmed")):
                candidate_specialist_confirmed_n += 1

        for a in audited:
            reason = str(a.get("fusion_reason") or "UNKNOWN")
            state = str(a.get("fusion_state") or "UNKNOWN")
            suppression_reasons[reason] = suppression_reasons.get(reason, 0) + 1
            suppression_states[state] = suppression_states.get(state, 0) + 1

            score = float(a.get("score") or 0.0)
            score_threshold = float(a.get("prospective_score_threshold") or 0.0)
            source_n = int(a.get("independent_evidence_n") or 0)
            min_sources = int(a.get("prospective_min_independent_sources") or 0)
            if score < score_threshold:
                insufficient_score_n += 1
            if source_n < min_sources:
                insufficient_sources_n += 1

            for metric in a.get("boundary_failure_metrics") or []:
                boundary_failure_metrics[str(metric)] = boundary_failure_metrics.get(str(metric), 0) + 1
            for metric in a.get("unresolved_required_measurements") or []:
                unresolved_measurements[str(metric)] = unresolved_measurements.get(str(metric), 0) + 1
            for conflict in a.get("blocked_by_conflicts") or []:
                blocked_conflicts[str(conflict)] = blocked_conflicts.get(str(conflict), 0) + 1

    for r in positives:
        fused_hits = expected_codes & set(r["fusion_codes"])
        final_hits = expected_codes & set(r["published_codes"])
        if not fused_hits or final_hits:
            continue
        audit = dict(r.get("reasoner_audit") or {})
        if bool(audit.get("preexcitation_published")) and any(
            str(code).startswith(("RBBB_", "LBBB_")) for code in fused_hits
        ):
            reasoner_preexcitation_suppression_n += 1
            if _target_positive(r["codes"], TARGETS["WPW"]["scp"]):
                reasoner_preexcitation_suppression_reference_wpw_n += 1
            else:
                reasoner_preexcitation_suppression_without_reference_wpw_n += 1
        for code in audit.get("blocking_conflicts") or []:
            reasoner_blocking_conflicts[str(code)] = reasoner_blocking_conflicts.get(str(code), 0) + 1
        for code in audit.get("warning_conflicts") or []:
            reasoner_warning_conflicts[str(code)] = reasoner_warning_conflicts.get(str(code), 0) + 1
        for abst in audit.get("abstentions") or []:
            domain = str(abst.get("domain") or "UNKNOWN")
            reason = str(abst.get("reason") or "UNKNOWN")
            reasoner_abstention_domains[domain] = reasoner_abstention_domains.get(domain, 0) + 1
            reasoner_abstention_reasons[reason] = reasoner_abstention_reasons.get(reason, 0) + 1

    return {
        "positive_n": n,
        "negative_control_n": len(negatives),
        "candidate_detected_n": candidate_n,
        "fusion_publishable_n": fusion_n,
        "final_published_n": final_n,
        "false_positive_n_on_clean_controls": fp,
        "candidate_evidence_audit": {
            "suppressed_candidate_n": candidate_to_fusion_loss_n,
            "evidence_counts": candidate_evidence_counts,
            "source_group_counts": candidate_source_group_counts,
            "evidence_signatures": candidate_evidence_signatures,
            "source_group_signatures": candidate_source_group_signatures,
            "specialist_confirmed_n": candidate_specialist_confirmed_n,
        },
        "fusion_suppression_audit": {
            "fusion_reasons": suppression_reasons,
            "fusion_states": suppression_states,
            "insufficient_score_n": insufficient_score_n,
            "insufficient_sources_n": insufficient_sources_n,
            "boundary_failure_metrics": boundary_failure_metrics,
            "unresolved_measurements": unresolved_measurements,
            "blocked_conflicts": blocked_conflicts,
        },
        "reasoner_suppression_audit": {
            "preexcitation_suppression_n": reasoner_preexcitation_suppression_n,
            "preexcitation_suppression_reference_wpw_n": reasoner_preexcitation_suppression_reference_wpw_n,
            "preexcitation_suppression_without_reference_wpw_n": reasoner_preexcitation_suppression_without_reference_wpw_n,
            "blocking_conflicts": reasoner_blocking_conflicts,
            "warning_conflicts": reasoner_warning_conflicts,
            "abstention_domains": reasoner_abstention_domains,
            "abstention_reasons": reasoner_abstention_reasons,
        },
        "candidate_miss_n": candidate_miss_n,
        "candidate_to_fusion_loss_n": candidate_to_fusion_loss_n,
        "fusion_to_final_loss_n": fusion_to_final_loss_n,
        "candidate_miss_fraction": (candidate_miss_n / n) if n else None,
        "candidate_to_fusion_loss_fraction": (candidate_to_fusion_loss_n / n) if n else None,
        "fusion_to_final_loss_fraction": (fusion_to_final_loss_n / n) if n else None,
        "candidate_sensitivity": candidate_sens,
        "fusion_sensitivity": fusion_sens,
        "final_sensitivity": final_sens,
        "specificity_clean_controls": specificity,
        "engineering_gate_eligible": gate_eligible,
        "candidate_target": TARGET_CANDIDATE_SENSITIVITY,
        "final_target": TARGET_FINAL_SENSITIVITY,
        "specificity_guardrail": SPECIFICITY_GUARDRAIL,
        "candidate_target_met": bool(
            gate_eligible and candidate_sens is not None
            and candidate_sens >= TARGET_CANDIDATE_SENSITIVITY
        ),
        "final_target_met": bool(
            gate_eligible and final_sens is not None
            and final_sens >= TARGET_FINAL_SENSITIVITY
        ),
        "specificity_guardrail_met": bool(
            specificity is not None and specificity >= SPECIFICITY_GUARDRAIL
        ),
    }


def benchmark(workdir: Path, output: Path, folds: list[int] | None = None) -> dict[str, Any]:
    folds = [INTERNAL_VALIDATION_FOLD] if folds is None else [int(x) for x in folds]
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path = workdir / "ptbxl_database.csv"
    statements_path = workdir / "scp_statements.csv"
    _download(f"{BASE}/ptbxl_database.csv", metadata_path)
    _download(f"{BASE}/scp_statements.csv", statements_path)

    meta = pd.read_csv(metadata_path)
    selected, selection = select_records(meta, folds=folds)
    negative_ids = set(selection["negative_control_ecg_ids"])

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    records_root = workdir / "records"

    for idx, row in selected.iterrows():
        ecg_id = int(row["ecg_id"])
        filename_hr = str(row["filename_hr"])
        try:
            local_base = _ensure_record(records_root, filename_hr)
            rec = wfdb.rdrecord(str(local_base))
            analysis = analyze_canonical_ecg(
                _canonical(rec.p_signal, int(round(float(rec.fs))), list(rec.sig_name), ecg_id)
            )
            rows.append({
                "ecg_id": ecg_id,
                "codes": dict(row["_codes"]),
                "candidate_codes": sorted(_candidate_codes(analysis)),
                "fusion_codes": sorted(_fusion_codes(analysis)),
                "candidate_audit": _candidate_audit(analysis),
                "fusion_audit": _fusion_audit(analysis),
                "reasoner_audit": _reasoner_audit(analysis),
                "published_codes": sorted(_published_codes(analysis)),
                "remeasure_required": bool(
                    (analysis.get("measurement_consensus") or {}).get("remeasure_required")
                ),
            })
        except Exception as exc:
            errors.append({
                "ecg_id": ecg_id,
                "error": f"{type(exc).__name__}:{exc}",
            })
        if (idx + 1) % 25 == 0:
            print(f"MEDCALC_ADULT_PTBXL {idx+1}/{len(selected)}", flush=True)

    metrics = {
        target: _score_target(target, spec, rows, negative_ids)
        for target, spec in TARGETS.items()
    }

    result = {
        "benchmark_version": BENCHMARK_VERSION,
        "dataset": "PTB-XL",
        "dataset_version": PTBXL_VERSION,
        "population": "ADULT_AGE_GE_18",
        "role": (
            "DEVELOPMENT_INTERNAL_VALIDATION_ONLY"
            if folds == [INTERNAL_VALIDATION_FOLD]
            else "DEVELOPMENT_TUNING_ONLY"
        ),
        "external_validation_claim_allowed": False,
        "fold_policy": {
            "tuning_folds": [1,2,3,4,5,6,7,8],
            "internal_validation_fold": 9,
            "internal_confirmation_fold": 10,
            "executed_folds": folds,
        },
        "selection": selection,
        "records_analyzed": len(rows),
        "analysis_error_n": len(errors),
        "analysis_failure_rate": len(errors) / max(len(selected), 1),
        "remeasure_required_rate": (
            float(np.mean([bool(r["remeasure_required"]) for r in rows])) if rows else None
        ),
        "metrics": metrics,
        "diagnostic_waterfall": {
            target: {
                "positive_n": m["positive_n"],
                "candidate_miss_n": m["candidate_miss_n"],
                "candidate_to_fusion_loss_n": m["candidate_to_fusion_loss_n"],
                "fusion_to_final_loss_n": m["fusion_to_final_loss_n"],
                "candidate_sensitivity": m["candidate_sensitivity"],
                "fusion_sensitivity": m["fusion_sensitivity"],
                "final_sensitivity": m["final_sensitivity"],
                "specificity_clean_controls": m["specificity_clean_controls"],
                "candidate_evidence_audit": m["candidate_evidence_audit"],
                "fusion_suppression_audit": m["fusion_suppression_audit"],
                "reasoner_suppression_audit": m["reasoner_suppression_audit"],
            }
            for target, m in metrics.items()
        },
        "overall_engineering_gate_pass": bool(
            all(
                (not m["engineering_gate_eligible"])
                or (
                    m["candidate_target_met"]
                    and m["final_target_met"]
                    and m["specificity_guardrail_met"]
                )
                for m in metrics.values()
            )
        ),
        "constraints": {
            "consumed_external_datasets_used_for_tuning": False,
            "sph_used_for_tuning": False,
            "mimic_used_for_tuning": False,
            "zzu_used_for_tuning": False,
            "heedb_accessed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def selftest() -> None:
    df = pd.DataFrame([
        {
            "ecg_id": 1, "age": 55, "strat_fold": 9,
            "scp_codes": "{'AFIB': 100}", "filename_hr": "records500/00000/00001_hr",
        },
        {
            "ecg_id": 2, "age": 17, "strat_fold": 9,
            "scp_codes": "{'AFIB': 100}", "filename_hr": "records500/00000/00002_hr",
        },
        {
            "ecg_id": 3, "age": 70, "strat_fold": 9,
            "scp_codes": "{'RBBB': 100}", "filename_hr": "records500/00000/00003_hr",
        },
        {
            "ecg_id": 4, "age": 44, "strat_fold": 9,
            "scp_codes": "{'NORM': 100}", "filename_hr": "records500/00000/00004_hr",
        },
        {
            "ecg_id": 5, "age": 44, "strat_fold": 8,
            "scp_codes": "{'AFIB': 100}", "filename_hr": "records500/00000/00005_hr",
        },
    ])
    selected, summary = select_records(df, folds=[9])
    ids = set(selected["ecg_id"].astype(int))
    assert 1 in ids and 3 in ids and 4 in ids, (ids, summary)
    assert 2 not in ids and 5 not in ids, (ids, summary)
    assert summary["positive_available_by_target"]["AF"] == 1, summary
    assert summary["positive_available_by_target"]["RBBB_COMPLETE"] == 1, summary
    assert summary["negative_control_n"] == 1, summary
    audit_rows = [{
        "ecg_id": 10,
        "codes": {"LAFB": 100.0},
        "candidate_codes": ["LAFB_COMPATIBLE"],
        "fusion_codes": [],
        "published_codes": [],
        "candidate_audit": {
            "LAFB_COMPATIBLE": {
                "evidence": ["LEFT_AXIS", "POSITIVE_I_AVL"],
                "source_groups": ["AXIS", "SUPERIOR_LIMB_MORPHOLOGY"],
                "specialist_confirmed": False,
            }
        },
        "fusion_audit": {
            "LAFB_COMPATIBLE": {
                "publishable": False,
                "fusion_reason": "INSUFFICIENT_FUSED_EVIDENCE",
                "fusion_state": "CANDIDATE_REVIEW",
                "score": 0.60,
                "prospective_score_threshold": 0.65,
                "independent_evidence_n": 2,
                "prospective_min_independent_sources": 2,
            }
        },
        "reasoner_audit": {},
    }]
    audit_metric = _score_target("LAFB", TARGETS["LAFB"], audit_rows, set())
    ca = audit_metric["candidate_evidence_audit"]
    assert ca["evidence_counts"]["LEFT_AXIS"] == 1, ca
    assert ca["source_group_counts"]["AXIS"] == 1, ca
    assert ca["evidence_signatures"]["LEFT_AXIS|POSITIVE_I_AVL"] == 1, ca

    pre_rows = [{
        "ecg_id": 11,
        "codes": {"RBBB": 100.0, "WPW": 100.0},
        "candidate_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
        "fusion_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
        "published_codes": [],
        "candidate_audit": {},
        "fusion_audit": {},
        "reasoner_audit": {
            "preexcitation_published": True,
            "blocking_conflicts": [],
            "warning_conflicts": ["PREEXCITATION_CONFOUNDS_BUNDLE_BRANCH_PATTERN"],
            "abstentions": [],
        },
    }, {
        "ecg_id": 12,
        "codes": {"RBBB": 100.0},
        "candidate_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
        "fusion_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
        "published_codes": [],
        "candidate_audit": {},
        "fusion_audit": {},
        "reasoner_audit": {
            "preexcitation_published": True,
            "blocking_conflicts": [],
            "warning_conflicts": ["PREEXCITATION_CONFOUNDS_BUNDLE_BRANCH_PATTERN"],
            "abstentions": [],
        },
    }]
    pre_metric = _score_target(
        "RBBB_COMPLETE", TARGETS["RBBB_COMPLETE"], pre_rows, set()
    )
    ra = pre_metric["reasoner_suppression_audit"]
    assert ra["preexcitation_suppression_n"] == 2, ra
    assert ra["preexcitation_suppression_reference_wpw_n"] == 1, ra
    assert ra["preexcitation_suppression_without_reference_wpw_n"] == 1, ra

    selected_tuning, tuning_summary = select_records(df, folds=[8])
    tuning_ids = set(selected_tuning["ecg_id"].astype(int))
    assert 5 in tuning_ids and 1 not in tuning_ids, (tuning_ids, tuning_summary)
    assert tuning_summary["folds"] == [8], tuning_summary
    print("MEDCALC_ADULT_DIAGNOSTIC_DEV_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/medcalc-ptbxl-dev"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/MEDCALC_ADULT_PTBXL_DEV.json"))
    ap.add_argument(
        "--folds",
        type=str,
        default=str(INTERNAL_VALIDATION_FOLD),
        help="Comma-separated PTB-XL folds. Use 1-8 for tuning, 9 for internal validation, 10 for confirmation.",
    )
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        if not folds:
            raise ValueError("At least one fold is required")
        benchmark(args.workdir, args.output, folds=folds)


if __name__ == "__main__":
    main()
