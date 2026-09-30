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
from ecg_analysis_cache import analyze_with_cache
from ecg_measurement_consensus import threshold_relation

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
FAST_GATE_HOLDOUT_MANIFEST = Path(__file__).with_name("ecg_fast_gate_100_manifest.json")
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


def _fast_gate_holdout_ids() -> set[int]:
    if not FAST_GATE_HOLDOUT_MANIFEST.exists():
        return set()
    data = json.loads(FAST_GATE_HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    if str(data.get("version") or "") != "MEDCALC_ECG_FAST_GATE_100_V1":
        raise ValueError("Unexpected FAST-GATE-100 manifest version")
    if str((data.get("usage_policy") or {}).get("purpose") or "") != "FIXED_POST_CHANGE_EVALUATION_ONLY":
        raise ValueError("FAST-GATE-100 manifest is not evaluation-only")
    return {int(row["ecg_id"]) for row in (data.get("cases") or [])}


def select_records(
    df: pd.DataFrame,
    folds: list[int] | None = None,
    exclude_ecg_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    folds = [INTERNAL_VALIDATION_FOLD] if folds is None else [int(x) for x in folds]
    adult = _adult_rows(df, folds)
    excluded = {int(x) for x in (exclude_ecg_ids or set())}
    excluded_present = sorted(
        int(x) for x in adult.loc[
            adult["ecg_id"].astype(int).isin(excluded), "ecg_id"
        ].tolist()
    )
    if excluded:
        adult = adult.loc[
            ~adult["ecg_id"].astype(int).isin(excluded)
        ].copy()
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
        "excluded_holdout_n": len(excluded_present),
        "excluded_holdout_ecg_ids": excluded_present,
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


def _av_candidate_miss_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    av = dict(analysis.get("av_conduction") or {})
    consensus = dict(analysis.get("measurement_consensus") or {})
    pr_consensus = dict(((consensus.get("metrics") or {}).get("pr_ms") or {}))
    global_pr = dict(((analysis.get("global") or {}).get("pr_ms") or {}))
    global_atrial = dict(analysis.get("atrial_activity") or {})
    pr_value = av.get("pr_median_ms")
    try:
        pr_value = float(pr_value) if pr_value is not None else None
    except Exception:
        pr_value = None
    global_pr_value = global_pr.get("value")
    try:
        global_pr_value = (
            float(global_pr_value) if global_pr_value is not None else None
        )
    except Exception:
        global_pr_value = None
    global_pr_conf = float(global_pr.get("confidence") or 0.0)
    global_coupling = float(
        global_atrial.get("rhythm_p_qrs_coupling_fraction") or 0.0
    )
    global_p_repro = bool(global_atrial.get("p_wave_reproducible"))
    global_pr_relation = threshold_relation(consensus, "pr_ms", 200.0)
    avb1_candidate_gate_components_met = bool(
        global_pr_value is not None
        and global_pr_relation == "ABOVE"
        and global_pr_conf >= 0.40
        and global_p_repro
        and global_coupling >= 0.55
    )
    return {
        "evaluable": bool(av.get("evaluable")),
        "classification": str(av.get("classification") or ""),
        "reason": str(av.get("reason") or ""),
        "lead": str(av.get("lead") or ""),
        "confidence": float(av.get("confidence") or 0.0),
        "one_to_one": bool(av.get("one_to_one")),
        "stable_pr": bool(av.get("stable_pr")),
        "atrial_sequence_regular": bool(av.get("atrial_sequence_regular")),
        "av_dissociation_phase": bool(av.get("av_dissociation_phase")),
        "nonconducted_p_n": int(av.get("nonconducted_p_n") or 0),
        "conducted_p_n": int(av.get("conducted_p_n") or 0),
        "p_qrs_coupling_fraction": float(av.get("p_qrs_coupling_fraction") or 0.0),
        "pr_median_ms": pr_value,
        "pr_relation_200": (
            "GT_200" if pr_value is not None and pr_value > 200.0
            else "LE_200" if pr_value is not None
            else "MISSING"
        ),
        "pr_measurement_state": str(pr_consensus.get("measurement_state") or ""),
        "pr_unusable": bool(pr_consensus.get("unusable")),
        "global_pr_ms": global_pr_value,
        "global_pr_confidence": global_pr_conf,
        "global_pr_relation_200": global_pr_relation,
        "global_p_wave_reproducible": global_p_repro,
        "global_p_qrs_coupling_fraction": global_coupling,
        "avb1_candidate_gate_components_met": avb1_candidate_gate_components_met,
        "basis": sorted(str(x) for x in (av.get("basis") or [])),
    }


def _bbb_qrs_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    cross = dict(analysis.get("crosslead_conduction") or {})
    criteria = dict(cross.get("criteria") or {})
    consensus = dict(analysis.get("measurement_consensus") or {})
    qrs_consensus = dict(((consensus.get("metrics") or {}).get("qrs_ms") or {}))
    global_qrs = dict(((analysis.get("global") or {}).get("qrs_ms") or {}))

    value = global_qrs.get("value")
    try:
        value = float(value) if value is not None else None
    except Exception:
        value = None
    confidence = float(global_qrs.get("confidence") or 0.0)

    wide_n = int(criteria.get("wide_qrs_lead_n") or 0)
    limb_n = int(criteria.get("wide_qrs_limb_lead_n") or 0)
    precordial_n = int(criteria.get("wide_qrs_precordial_lead_n") or 0)
    shape_ok = bool(limb_n >= 1 and precordial_n >= 2)

    qrs_rows = (
        (((analysis.get("feature_graph") or {}).get("specialist_evidence") or {})
         .get("qrs_morphology") or {})
        .get("per_lead") or {}
    )
    durations: list[float] = []
    for row in qrs_rows.values():
        row = dict(row or {})
        if not bool(row.get("evaluable")):
            continue
        try:
            duration = float(row.get("duration_ms"))
        except Exception:
            continue
        if math.isfinite(duration):
            durations.append(duration)
    ge115_n = sum(x >= 115.0 for x in durations)
    ge118_n = sum(x >= 118.0 for x in durations)

    return {
        "global_qrs_ms": value,
        "global_qrs_confidence": confidence,
        "global_qrs_status": str(global_qrs.get("status") or ""),
        "global_qrs_relation_120": threshold_relation(consensus, "qrs_ms", 120.0),
        "qrs_measurement_state": str(qrs_consensus.get("measurement_state") or ""),
        "qrs_remeasure": bool(qrs_consensus.get("remeasure")),
        "qrs_unusable": bool(qrs_consensus.get("unusable")),
        "wide_qrs_lead_n": wide_n,
        "wide_qrs_limb_lead_n": limb_n,
        "wide_qrs_precordial_lead_n": precordial_n,
        "qrs_evaluable_lead_n": len(durations),
        "qrs_ge_115_lead_n": ge115_n,
        "qrs_ge_118_lead_n": ge118_n,
        "qrs_perlead_median_ms": (
            round(float(np.median(durations)), 3) if durations else None
        ),
        "qrs_perlead_max_ms": (
            round(float(max(durations)), 3) if durations else None
        ),
        "multilead_qrs_ge_120_rescue": bool(
            criteria.get("multilead_qrs_ge_120_rescue")
        ),
        "rescue_shape_support": shape_ok,
        "one_wide_lead_short_of_rescue": bool(wide_n == 3 and shape_ok),
        "ge4_wide_but_shape_insufficient": bool(wide_n >= 4 and not shape_ok),
        "rbbb_morphology": bool(criteria.get("rbbb_morphology")),
        "lbbb_morphology": bool(criteria.get("lbbb_morphology")),
    }


def _pr_multilead_audit(analysis: dict[str, Any]) -> dict[str, Any]:
    consensus = dict(analysis.get("measurement_consensus") or {})
    pr = dict(((consensus.get("metrics") or {}).get("pr_ms") or {}))
    values = dict(pr.get("candidate_values") or {})
    confidences = dict(pr.get("candidate_confidences") or {})

    usable: list[tuple[str, float, float]] = []
    for lead, raw_value in values.items():
        try:
            value = float(raw_value)
            confidence = float(confidences.get(lead) or 0.0)
        except Exception:
            continue
        if math.isfinite(value) and math.isfinite(confidence) and confidence >= 0.50:
            usable.append((str(lead), value, confidence))

    long_leads = sorted(lead for lead, value, _ in usable if value > 200.0)
    short_leads = sorted(
        lead for lead, value, _ in usable
        if 70.0 <= value < 120.0
    )
    usable_values = np.asarray([value for _, value, _ in usable], dtype=float)
    median = float(np.median(usable_values)) if usable_values.size else None
    mad = (
        float(np.median(np.abs(usable_values - median)))
        if usable_values.size >= 2 and median is not None
        else 0.0
    )

    return {
        "measurement_state": str(pr.get("measurement_state") or ""),
        "global_unusable": bool(pr.get("unusable")),
        "usable_lead_n_conf_ge_0_50": len(usable),
        "usable_lead_median_ms": round(median, 6) if median is not None else None,
        "usable_lead_mad_ms": round(mad, 6),
        "pr_gt_200_leads_n": len(long_leads),
        "pr_70_to_lt120_leads_n": len(short_leads),
        "ge2_pr_gt_200_leads": len(long_leads) >= 2,
        "ge2_pr_70_to_lt120_leads": len(short_leads) >= 2,
    }


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


def _rbbb_qrs_counterfactual_audit(
    rows: list[dict[str, Any]],
    negative_ids: set[int],
) -> dict[str, Any]:
    """Aggregate-only prospective rescue audit; never changes diagnosis."""
    spec = TARGETS["RBBB_COMPLETE"]
    expected = set(spec["medcalc"])

    positives = [
        r for r in rows
        if _target_positive(r.get("codes") or {}, spec["scp"])
    ]
    baseline_final_positive_n = sum(
        bool(expected & set(r.get("published_codes") or []))
        for r in positives
    )
    baseline_negative_fp_n = sum(
        bool(expected & set(r.get("published_codes") or []))
        for r in rows
        if int(r.get("ecg_id")) in negative_ids
    )

    policies = {
        "THREE_WIDE_DISTRIBUTED_FULL_MORPH": lambda b: (
            int(b.get("wide_qrs_lead_n") or 0) >= 3
            and int(b.get("wide_qrs_limb_lead_n") or 0) >= 1
            and int(b.get("wide_qrs_precordial_lead_n") or 0) >= 2
            and bool(b.get("rbbb_morphology"))
        ),
        "THREE_WIDE_DISTRIBUTED_PLUS_GE4_118_FULL_MORPH": lambda b: (
            int(b.get("wide_qrs_lead_n") or 0) >= 3
            and int(b.get("wide_qrs_limb_lead_n") or 0) >= 1
            and int(b.get("wide_qrs_precordial_lead_n") or 0) >= 2
            and int(b.get("qrs_ge_118_lead_n") or 0) >= 4
            and bool(b.get("rbbb_morphology"))
        ),
        "THREE_WIDE_DISTRIBUTED_PLUS_GE4_115_FULL_MORPH": lambda b: (
            int(b.get("wide_qrs_lead_n") or 0) >= 3
            and int(b.get("wide_qrs_limb_lead_n") or 0) >= 1
            and int(b.get("wide_qrs_precordial_lead_n") or 0) >= 2
            and int(b.get("qrs_ge_115_lead_n") or 0) >= 4
            and bool(b.get("rbbb_morphology"))
        ),
    }

    out: dict[str, Any] = {}
    for policy_name, policy_fn in policies.items():
        lost_positive_n = 0
        recoverable_positive_n = 0
        unresolved_negative_candidate_n = 0
        incremental_negative_trigger_n = 0

        for r in rows:
            candidate_present = bool(
                expected & set(r.get("candidate_codes") or [])
            )
            if not candidate_present:
                continue
            b = dict(r.get("bbb_qrs_audit") or {})
            relation = str(b.get("global_qrs_relation_120") or "")
            unresolved_qrs = bool(
                b.get("qrs_remeasure")
                or b.get("qrs_unusable")
                or relation in {"OVERLAPS", "UNUSABLE", "UNCERTAIN"}
            )
            if not unresolved_qrs:
                continue

            policy_hit = bool(policy_fn(b))
            is_positive = _target_positive(
                r.get("codes") or {},
                spec["scp"],
            )
            final_hit = bool(
                expected & set(r.get("published_codes") or [])
            )

            if is_positive and not final_hit:
                lost_positive_n += 1
                recoverable_positive_n += int(policy_hit)
            elif int(r.get("ecg_id")) in negative_ids and not final_hit:
                unresolved_negative_candidate_n += 1
                incremental_negative_trigger_n += int(policy_hit)

        projected_final_tp_n = (
            baseline_final_positive_n + recoverable_positive_n
        )
        projected_final_sensitivity = (
            projected_final_tp_n / len(positives)
            if positives else None
        )
        projected_fp_upper_n = (
            baseline_negative_fp_n + incremental_negative_trigger_n
        )
        projected_specificity_lower = (
            (len(negative_ids) - projected_fp_upper_n) / len(negative_ids)
            if negative_ids else None
        )

        out[policy_name] = {
            "positive_n": len(positives),
            "baseline_final_positive_n": baseline_final_positive_n,
            "baseline_final_sensitivity": (
                baseline_final_positive_n / len(positives)
                if positives else None
            ),
            "lost_positive_candidate_n": lost_positive_n,
            "recoverable_positive_n": recoverable_positive_n,
            "recoverable_positive_fraction_of_lost": (
                recoverable_positive_n / lost_positive_n
                if lost_positive_n else None
            ),
            "projected_final_positive_n_if_all_triggers_publish": projected_final_tp_n,
            "projected_final_sensitivity_if_all_triggers_publish": projected_final_sensitivity,
            "negative_control_n": len(negative_ids),
            "baseline_negative_fp_n": baseline_negative_fp_n,
            "unresolved_negative_candidate_n": unresolved_negative_candidate_n,
            "incremental_negative_trigger_n": incremental_negative_trigger_n,
            "projected_negative_fp_upper_n_if_all_triggers_publish": projected_fp_upper_n,
            "projected_specificity_lower_bound_if_all_triggers_publish": projected_specificity_lower,
            "interpretation": (
                "AGGREGATE_COUNTERFACTUAL_UPPER_BOUND_ONLY; "
                "DOES_NOT_CHANGE_GATE_OR_FUSION"
            ),
        }
    return out


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

    fp_evidence_counts: dict[str, int] = {}
    fp_source_group_counts: dict[str, int] = {}
    fp_evidence_signatures: dict[str, int] = {}
    fp_source_group_signatures: dict[str, int] = {}
    fp_specialist_confirmed_n = 0
    fp_candidate_present_n = 0
    fp_fusion_score_bands: dict[str, int] = {}

    av_candidate_miss_classifications: dict[str, int] = {}
    av_candidate_miss_reasons: dict[str, int] = {}
    av_candidate_miss_pr_relations: dict[str, int] = {}
    av_candidate_miss_pr_states: dict[str, int] = {}
    av_candidate_miss_one_to_one_n = 0
    av_candidate_miss_stable_pr_n = 0
    av_candidate_miss_evaluable_n = 0
    av_candidate_miss_atrial_regular_n = 0
    av_candidate_miss_av_dissociation_n = 0
    av_candidate_miss_nonconducted_p_present_n = 0
    av_candidate_miss_ge2_nonconducted_p_n = 0
    av_candidate_miss_conducted_p_ge2_n = 0
    av_candidate_miss_coupling_bands: dict[str, int] = {}
    av_candidate_miss_global_pr_relations: dict[str, int] = {}
    av_candidate_miss_global_p_repro_n = 0
    av_candidate_miss_global_coupling_ge_0_55_n = 0
    av_candidate_miss_global_pr_conf_ge_0_40_n = 0
    av_candidate_miss_avb1_gate_components_met_n = 0

    pr_ge2_long_positive_n = 0
    pr_ge2_short_positive_n = 0
    pr_ge2_long_candidate_miss_n = 0
    pr_ge2_short_candidate_miss_n = 0
    pr_ge2_long_fusion_loss_n = 0
    pr_ge2_short_fusion_loss_n = 0
    pr_global_unusable_ge2_long_n = 0
    pr_global_unusable_ge2_short_n = 0
    pr_negative_ge2_long_n = 0
    pr_negative_ge2_short_n = 0
    pr_negative_ge2_long_not_final_n = 0
    pr_negative_ge2_short_not_final_n = 0

    reasoner_preexcitation_suppression_n = 0
    reasoner_preexcitation_suppression_reference_wpw_n = 0
    reasoner_preexcitation_suppression_without_reference_wpw_n = 0
    reasoner_blocking_conflicts: dict[str, int] = {}
    reasoner_warning_conflicts: dict[str, int] = {}
    reasoner_abstention_domains: dict[str, int] = {}
    reasoner_abstention_reasons: dict[str, int] = {}

    bbb_qrs_global_relation_120: dict[str, int] = {}
    bbb_qrs_measurement_states: dict[str, int] = {}
    bbb_qrs_global_statuses: dict[str, int] = {}
    bbb_qrs_global_value_bands: dict[str, int] = {}
    bbb_qrs_wide_lead_n: dict[str, int] = {}
    bbb_qrs_wide_limb_n: dict[str, int] = {}
    bbb_qrs_wide_precordial_n: dict[str, int] = {}
    bbb_qrs_ge115_lead_n: dict[str, int] = {}
    bbb_qrs_ge118_lead_n: dict[str, int] = {}
    bbb_qrs_perlead_median_bands: dict[str, int] = {}
    bbb_qrs_perlead_max_bands: dict[str, int] = {}
    bbb_qrs_rescue_active_n = 0
    bbb_qrs_rescue_shape_support_n = 0
    bbb_qrs_one_wide_lead_short_n = 0
    bbb_qrs_ge4_shape_insufficient_n = 0
    bbb_qrs_target_morphology_n = 0

    expected_codes = set(spec["medcalc"])

    for r in positives:
        pr_audit = dict(r.get("pr_multilead_audit") or {})
        has_candidate = bool(expected_codes & set(r.get("candidate_codes") or []))
        has_fusion = bool(expected_codes & set(r.get("fusion_codes") or []))
        ge2_long = bool(pr_audit.get("ge2_pr_gt_200_leads"))
        ge2_short = bool(pr_audit.get("ge2_pr_70_to_lt120_leads"))
        global_unusable = bool(pr_audit.get("global_unusable"))

        if ge2_long:
            pr_ge2_long_positive_n += 1
            pr_ge2_long_candidate_miss_n += int(not has_candidate)
            pr_ge2_long_fusion_loss_n += int(has_candidate and not has_fusion)
            pr_global_unusable_ge2_long_n += int(global_unusable)
        if ge2_short:
            pr_ge2_short_positive_n += 1
            pr_ge2_short_candidate_miss_n += int(not has_candidate)
            pr_ge2_short_fusion_loss_n += int(has_candidate and not has_fusion)
            pr_global_unusable_ge2_short_n += int(global_unusable)

    for r in negatives:
        pr_audit = dict(r.get("pr_multilead_audit") or {})
        ge2_long = bool(pr_audit.get("ge2_pr_gt_200_leads"))
        ge2_short = bool(pr_audit.get("ge2_pr_70_to_lt120_leads"))
        final_hit = hit(r, "published_codes")
        pr_negative_ge2_long_n += int(ge2_long)
        pr_negative_ge2_short_n += int(ge2_short)
        pr_negative_ge2_long_not_final_n += int(ge2_long and not final_hit)
        pr_negative_ge2_short_not_final_n += int(ge2_short and not final_hit)

        if not final_hit:
            continue
        candidate_hits = expected_codes & set(r.get("candidate_codes") or [])
        candidate_map = r.get("candidate_audit") or {}
        fusion_map = r.get("fusion_audit") or {}
        if candidate_hits:
            fp_candidate_present_n += 1
        for code in sorted(candidate_hits):
            c = dict(candidate_map.get(code) or {})
            evidence = sorted(str(x) for x in (c.get("evidence") or []))
            source_groups = sorted(str(x) for x in (c.get("source_groups") or []))
            for item in evidence:
                fp_evidence_counts[item] = fp_evidence_counts.get(item, 0) + 1
            for item in source_groups:
                fp_source_group_counts[item] = fp_source_group_counts.get(item, 0) + 1
            evidence_sig = "|".join(evidence) if evidence else "<NONE>"
            source_sig = "|".join(source_groups) if source_groups else "<NONE>"
            fp_evidence_signatures[evidence_sig] = fp_evidence_signatures.get(evidence_sig, 0) + 1
            fp_source_group_signatures[source_sig] = fp_source_group_signatures.get(source_sig, 0) + 1
            if bool(c.get("specialist_confirmed")):
                fp_specialist_confirmed_n += 1

            fusion_row = dict(fusion_map.get(code) or {})
            score = float(fusion_row.get("score") or c.get("score") or 0.0)
            band = (
                "GE_0_90" if score >= 0.90
                else "0_80_TO_0_899" if score >= 0.80
                else "0_70_TO_0_799" if score >= 0.70
                else "0_60_TO_0_699" if score >= 0.60
                else "LT_0_60"
            )
            fp_fusion_score_bands[band] = fp_fusion_score_bands.get(band, 0) + 1

    if target in {"AVB1", "AVB2", "AVB3"}:
        for r in positives:
            candidate_hits = expected_codes & set(r["candidate_codes"])
            if candidate_hits:
                continue
            audit = dict(r.get("av_candidate_miss_audit") or {})
            classification = str(audit.get("classification") or "UNKNOWN")
            reason = str(audit.get("reason") or "NONE")
            pr_relation = str(audit.get("pr_relation_200") or "UNKNOWN")
            pr_state = str(audit.get("pr_measurement_state") or "UNKNOWN")
            av_candidate_miss_classifications[classification] = (
                av_candidate_miss_classifications.get(classification, 0) + 1
            )
            av_candidate_miss_reasons[reason] = (
                av_candidate_miss_reasons.get(reason, 0) + 1
            )
            av_candidate_miss_pr_relations[pr_relation] = (
                av_candidate_miss_pr_relations.get(pr_relation, 0) + 1
            )
            av_candidate_miss_pr_states[pr_state] = (
                av_candidate_miss_pr_states.get(pr_state, 0) + 1
            )
            av_candidate_miss_evaluable_n += int(bool(audit.get("evaluable")))
            av_candidate_miss_one_to_one_n += int(bool(audit.get("one_to_one")))
            av_candidate_miss_stable_pr_n += int(bool(audit.get("stable_pr")))
            av_candidate_miss_atrial_regular_n += int(
                bool(audit.get("atrial_sequence_regular"))
            )
            av_candidate_miss_av_dissociation_n += int(
                bool(audit.get("av_dissociation_phase"))
            )
            nonconducted_p_n = int(audit.get("nonconducted_p_n") or 0)
            conducted_p_n = int(audit.get("conducted_p_n") or 0)
            av_candidate_miss_nonconducted_p_present_n += int(nonconducted_p_n >= 1)
            av_candidate_miss_ge2_nonconducted_p_n += int(nonconducted_p_n >= 2)
            av_candidate_miss_conducted_p_ge2_n += int(conducted_p_n >= 2)
            coupling = float(audit.get("p_qrs_coupling_fraction") or 0.0)
            coupling_band = (
                "GE_0_90" if coupling >= 0.90
                else "0_70_TO_0_899" if coupling >= 0.70
                else "0_50_TO_0_699" if coupling >= 0.50
                else "LT_0_50"
            )
            av_candidate_miss_coupling_bands[coupling_band] = (
                av_candidate_miss_coupling_bands.get(coupling_band, 0) + 1
            )
            global_pr_relation = str(
                audit.get("global_pr_relation_200") or "UNKNOWN"
            )
            av_candidate_miss_global_pr_relations[global_pr_relation] = (
                av_candidate_miss_global_pr_relations.get(global_pr_relation, 0) + 1
            )
            av_candidate_miss_global_p_repro_n += int(
                bool(audit.get("global_p_wave_reproducible"))
            )
            av_candidate_miss_global_coupling_ge_0_55_n += int(
                float(audit.get("global_p_qrs_coupling_fraction") or 0.0) >= 0.55
            )
            av_candidate_miss_global_pr_conf_ge_0_40_n += int(
                float(audit.get("global_pr_confidence") or 0.0) >= 0.40
            )
            av_candidate_miss_avb1_gate_components_met_n += int(
                bool(audit.get("avb1_candidate_gate_components_met"))
            )

    for r in positives:
        candidate_hits = expected_codes & set(r["candidate_codes"])
        if not candidate_hits:
            continue
        audit_map = r.get("fusion_audit") or {}
        audited = [dict(audit_map.get(code) or {}) for code in candidate_hits]
        if any(bool(a.get("publishable")) for a in audited):
            continue

        if target in {"RBBB_COMPLETE", "LBBB"}:
            bbb = dict(r.get("bbb_qrs_audit") or {})
            relation = str(bbb.get("global_qrs_relation_120") or "UNKNOWN")
            state = str(bbb.get("qrs_measurement_state") or "UNKNOWN")
            status = str(bbb.get("global_qrs_status") or "UNKNOWN")
            bbb_qrs_global_relation_120[relation] = (
                bbb_qrs_global_relation_120.get(relation, 0) + 1
            )
            bbb_qrs_measurement_states[state] = (
                bbb_qrs_measurement_states.get(state, 0) + 1
            )
            bbb_qrs_global_statuses[status] = (
                bbb_qrs_global_statuses.get(status, 0) + 1
            )

            qrs_value = bbb.get("global_qrs_ms")
            try:
                qrs_value = float(qrs_value) if qrs_value is not None else None
            except Exception:
                qrs_value = None
            value_band = (
                "MISSING" if qrs_value is None
                else "GE_120" if qrs_value >= 120.0
                else "115_TO_119_9" if qrs_value >= 115.0
                else "110_TO_114_9" if qrs_value >= 110.0
                else "LT_110"
            )
            bbb_qrs_global_value_bands[value_band] = (
                bbb_qrs_global_value_bands.get(value_band, 0) + 1
            )

            wide_n = int(bbb.get("wide_qrs_lead_n") or 0)
            limb_n = int(bbb.get("wide_qrs_limb_lead_n") or 0)
            precordial_n = int(bbb.get("wide_qrs_precordial_lead_n") or 0)
            for bucket, value in (
                (bbb_qrs_wide_lead_n, wide_n),
                (bbb_qrs_wide_limb_n, limb_n),
                (bbb_qrs_wide_precordial_n, precordial_n),
                (bbb_qrs_ge115_lead_n, int(bbb.get("qrs_ge_115_lead_n") or 0)),
                (bbb_qrs_ge118_lead_n, int(bbb.get("qrs_ge_118_lead_n") or 0)),
            ):
                key = str(value) if value < 7 else "GE_7"
                bucket[key] = bucket.get(key, 0) + 1

            def _duration_band(value: Any) -> str:
                try:
                    x = float(value)
                except Exception:
                    return "MISSING"
                return (
                    "GE_120" if x >= 120.0
                    else "118_TO_119_9" if x >= 118.0
                    else "115_TO_117_9" if x >= 115.0
                    else "110_TO_114_9" if x >= 110.0
                    else "LT_110"
                )

            median_band = _duration_band(bbb.get("qrs_perlead_median_ms"))
            max_band = _duration_band(bbb.get("qrs_perlead_max_ms"))
            bbb_qrs_perlead_median_bands[median_band] = (
                bbb_qrs_perlead_median_bands.get(median_band, 0) + 1
            )
            bbb_qrs_perlead_max_bands[max_band] = (
                bbb_qrs_perlead_max_bands.get(max_band, 0) + 1
            )

            bbb_qrs_rescue_active_n += int(
                bool(bbb.get("multilead_qrs_ge_120_rescue"))
            )
            bbb_qrs_rescue_shape_support_n += int(
                bool(bbb.get("rescue_shape_support"))
            )
            bbb_qrs_one_wide_lead_short_n += int(
                bool(bbb.get("one_wide_lead_short_of_rescue"))
            )
            bbb_qrs_ge4_shape_insufficient_n += int(
                bool(bbb.get("ge4_wide_but_shape_insufficient"))
            )
            morphology_key = (
                "rbbb_morphology" if target == "RBBB_COMPLETE"
                else "lbbb_morphology"
            )
            bbb_qrs_target_morphology_n += int(bool(bbb.get(morphology_key)))

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
        "pr_multilead_threshold_audit": {
            "positive_n_with_ge2_pr_gt_200_leads": pr_ge2_long_positive_n,
            "positive_n_with_ge2_pr_70_to_lt120_leads": pr_ge2_short_positive_n,
            "candidate_miss_with_ge2_pr_gt_200_leads": pr_ge2_long_candidate_miss_n,
            "candidate_miss_with_ge2_pr_70_to_lt120_leads": pr_ge2_short_candidate_miss_n,
            "fusion_loss_with_ge2_pr_gt_200_leads": pr_ge2_long_fusion_loss_n,
            "fusion_loss_with_ge2_pr_70_to_lt120_leads": pr_ge2_short_fusion_loss_n,
            "global_unusable_with_ge2_pr_gt_200_leads": pr_global_unusable_ge2_long_n,
            "global_unusable_with_ge2_pr_70_to_lt120_leads": pr_global_unusable_ge2_short_n,
            "negative_control_n_with_ge2_pr_gt_200_leads": pr_negative_ge2_long_n,
            "negative_control_n_with_ge2_pr_70_to_lt120_leads": pr_negative_ge2_short_n,
            "negative_control_not_final_with_ge2_pr_gt_200_leads": pr_negative_ge2_long_not_final_n,
            "negative_control_not_final_with_ge2_pr_70_to_lt120_leads": pr_negative_ge2_short_not_final_n,
            "baseline_false_positive_n": fp,
            "projected_false_positive_upper_n_if_all_ge2_long_publish": (
                fp + pr_negative_ge2_long_not_final_n
            ),
            "projected_false_positive_upper_n_if_all_ge2_short_publish": (
                fp + pr_negative_ge2_short_not_final_n
            ),
            "projected_specificity_lower_bound_if_all_ge2_long_publish": (
                (
                    len(negatives)
                    - fp
                    - pr_negative_ge2_long_not_final_n
                ) / len(negatives)
                if negatives else None
            ),
            "projected_specificity_lower_bound_if_all_ge2_short_publish": (
                (
                    len(negatives)
                    - fp
                    - pr_negative_ge2_short_not_final_n
                ) / len(negatives)
                if negatives else None
            ),
            "lead_confidence_floor": 0.50,
            "long_pr_threshold_ms": 200.0,
            "short_pr_range_ms": [70.0, 120.0],
        },
        "false_positive_evidence_audit": {
            "final_false_positive_n": fp,
            "candidate_present_n": fp_candidate_present_n,
            "evidence_counts": fp_evidence_counts,
            "source_group_counts": fp_source_group_counts,
            "evidence_signatures": fp_evidence_signatures,
            "source_group_signatures": fp_source_group_signatures,
            "specialist_confirmed_n": fp_specialist_confirmed_n,
            "fusion_score_bands": fp_fusion_score_bands,
        },
        "candidate_miss_audit": {
            "av_classifications": av_candidate_miss_classifications,
            "av_reasons": av_candidate_miss_reasons,
            "pr_relation_200": av_candidate_miss_pr_relations,
            "pr_measurement_states": av_candidate_miss_pr_states,
            "av_evaluable_n": av_candidate_miss_evaluable_n,
            "one_to_one_n": av_candidate_miss_one_to_one_n,
            "stable_pr_n": av_candidate_miss_stable_pr_n,
            "atrial_sequence_regular_n": av_candidate_miss_atrial_regular_n,
            "av_dissociation_phase_n": av_candidate_miss_av_dissociation_n,
            "nonconducted_p_present_n": av_candidate_miss_nonconducted_p_present_n,
            "ge2_nonconducted_p_n": av_candidate_miss_ge2_nonconducted_p_n,
            "conducted_p_ge2_n": av_candidate_miss_conducted_p_ge2_n,
            "p_qrs_coupling_bands": av_candidate_miss_coupling_bands,
            "global_pr_relation_200": av_candidate_miss_global_pr_relations,
            "global_p_wave_reproducible_n": av_candidate_miss_global_p_repro_n,
            "global_coupling_ge_0_55_n": av_candidate_miss_global_coupling_ge_0_55_n,
            "global_pr_confidence_ge_0_40_n": av_candidate_miss_global_pr_conf_ge_0_40_n,
            "avb1_candidate_gate_components_met_n": av_candidate_miss_avb1_gate_components_met_n,
        },
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
        "bbb_qrs_fusion_loss_audit": {
            "suppressed_candidate_n": (
                candidate_to_fusion_loss_n
                if target in {"RBBB_COMPLETE", "LBBB"} else 0
            ),
            "global_qrs_relation_120": bbb_qrs_global_relation_120,
            "qrs_measurement_states": bbb_qrs_measurement_states,
            "global_qrs_statuses": bbb_qrs_global_statuses,
            "global_qrs_value_bands": bbb_qrs_global_value_bands,
            "wide_qrs_lead_n": bbb_qrs_wide_lead_n,
            "wide_qrs_limb_lead_n": bbb_qrs_wide_limb_n,
            "wide_qrs_precordial_lead_n": bbb_qrs_wide_precordial_n,
            "qrs_ge_115_lead_n": bbb_qrs_ge115_lead_n,
            "qrs_ge_118_lead_n": bbb_qrs_ge118_lead_n,
            "qrs_perlead_median_bands": bbb_qrs_perlead_median_bands,
            "qrs_perlead_max_bands": bbb_qrs_perlead_max_bands,
            "multilead_rescue_active_n": bbb_qrs_rescue_active_n,
            "rescue_shape_support_n": bbb_qrs_rescue_shape_support_n,
            "one_wide_lead_short_of_rescue_n": bbb_qrs_one_wide_lead_short_n,
            "ge4_wide_but_shape_insufficient_n": bbb_qrs_ge4_shape_insufficient_n,
            "target_morphology_present_n": bbb_qrs_target_morphology_n,
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


def benchmark(\n    workdir: Path,\n    output: Path,\n    folds: list[int] | None = None,\n    analysis_cache_dir: Path | None = None,\n) -> dict[str, Any]:
    folds = [INTERNAL_VALIDATION_FOLD] if folds is None else [int(x) for x in folds]
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path = workdir / "ptbxl_database.csv"
    statements_path = workdir / "scp_statements.csv"
    _download(f"{BASE}/ptbxl_database.csv", metadata_path)
    _download(f"{BASE}/scp_statements.csv", statements_path)

    meta = pd.read_csv(metadata_path)
    holdout_ids = _fast_gate_holdout_ids()
    selected, selection = select_records(
        meta,
        folds=folds,
        exclude_ecg_ids=holdout_ids,
    )
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
            canonical = _canonical(
                rec.p_signal,
                int(round(float(rec.fs))),
                list(rec.sig_name),
                ecg_id,
            )
            if analysis_cache_dir is not None:
                analysis = analyze_with_cache(
                    canonical,
                    cache_dir=analysis_cache_dir,
                    record_key=f"ptbxl-{PTBXL_VERSION}-{ecg_id}",
                )
            else:
                analysis = analyze_canonical_ecg(canonical)
            rows.append({
                "ecg_id": ecg_id,
                "codes": dict(row["_codes"]),
                "candidate_codes": sorted(_candidate_codes(analysis)),
                "fusion_codes": sorted(_fusion_codes(analysis)),
                "candidate_audit": _candidate_audit(analysis),
                "av_candidate_miss_audit": _av_candidate_miss_audit(analysis),
                "pr_multilead_audit": _pr_multilead_audit(analysis),
                "bbb_qrs_audit": _bbb_qrs_audit(analysis),
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
    rbbb_qrs_counterfactual = _rbbb_qrs_counterfactual_audit(
        rows,
        negative_ids,
    )

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
        "analysis_cache_enabled": bool(analysis_cache_dir is not None),
        "analysis_failure_rate": len(errors) / max(len(selected), 1),
        "remeasure_required_rate": (
            float(np.mean([bool(r["remeasure_required"]) for r in rows])) if rows else None
        ),
        "metrics": metrics,
        "rbbb_qrs_counterfactual_audit": rbbb_qrs_counterfactual,
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
                "false_positive_evidence_audit": m["false_positive_evidence_audit"],
                "pr_multilead_threshold_audit": m["pr_multilead_threshold_audit"],
                "candidate_miss_audit": m["candidate_miss_audit"],
                "candidate_evidence_audit": m["candidate_evidence_audit"],
                "fusion_suppression_audit": m["fusion_suppression_audit"],
                "bbb_qrs_fusion_loss_audit": m["bbb_qrs_fusion_loss_audit"],
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

    selected_holdout, summary_holdout = select_records(
        df,
        folds=[9],
        exclude_ecg_ids={1, 4},
    )
    holdout_ids = set(selected_holdout["ecg_id"].astype(int))
    assert 1 not in holdout_ids and 4 not in holdout_ids, (
        holdout_ids,
        summary_holdout,
    )
    assert 3 in holdout_ids, (holdout_ids, summary_holdout)
    assert summary_holdout["excluded_holdout_n"] == 2, summary_holdout
    assert summary_holdout["excluded_holdout_ecg_ids"] == [1, 4], summary_holdout
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

    pr_projection_rows = [
        {
            "ecg_id": 51,
            "codes": {"1AVB": 100.0},
            "candidate_codes": [],
            "fusion_codes": [],
            "published_codes": [],
            "candidate_audit": {},
            "av_candidate_miss_audit": {},
            "pr_multilead_audit": {
                "measurement_state": "REMEASURE_REQUIRED",
                "global_unusable": True,
                "ge2_pr_gt_200_leads": True,
                "ge2_pr_70_to_lt120_leads": False,
            },
            "fusion_audit": {},
            "reasoner_audit": {},
        },
        {
            "ecg_id": 52,
            "codes": {"NORM": 100.0},
            "candidate_codes": ["FIRST_DEGREE_AV_DELAY_COMPATIBLE"],
            "fusion_codes": ["FIRST_DEGREE_AV_DELAY_COMPATIBLE"],
            "published_codes": ["FIRST_DEGREE_AV_DELAY_COMPATIBLE"],
            "candidate_audit": {},
            "av_candidate_miss_audit": {},
            "pr_multilead_audit": {
                "measurement_state": "MEASURED_HIGH_CONFIDENCE",
                "global_unusable": False,
                "ge2_pr_gt_200_leads": False,
                "ge2_pr_70_to_lt120_leads": False,
            },
            "fusion_audit": {},
            "reasoner_audit": {},
        },
        {
            "ecg_id": 53,
            "codes": {"NORM": 100.0},
            "candidate_codes": [],
            "fusion_codes": [],
            "published_codes": [],
            "candidate_audit": {},
            "av_candidate_miss_audit": {},
            "pr_multilead_audit": {
                "measurement_state": "REMEASURE_REQUIRED",
                "global_unusable": True,
                "ge2_pr_gt_200_leads": True,
                "ge2_pr_70_to_lt120_leads": False,
            },
            "fusion_audit": {},
            "reasoner_audit": {},
        },
    ]
    pr_projection_metric = _score_target(
        "AVB1", TARGETS["AVB1"], pr_projection_rows, {52, 53}
    )
    pma = pr_projection_metric["pr_multilead_threshold_audit"]
    assert pma["baseline_false_positive_n"] == 1, pma
    assert pma["negative_control_not_final_with_ge2_pr_gt_200_leads"] == 1, pma
    assert pma["projected_false_positive_upper_n_if_all_ge2_long_publish"] == 2, pma
    assert pma["projected_specificity_lower_bound_if_all_ge2_long_publish"] == 0.0, pma

    bbb_rows = [{
        "ecg_id": 31,
        "codes": {"RBBB": 100.0},
        "candidate_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
        "fusion_codes": [],
        "published_codes": [],
        "candidate_audit": {
            "RBBB_MORPHOLOGY_COMPATIBLE": {
                "evidence": ["RIGHT_TERMINAL_R", "LATERAL_TERMINAL_S"],
                "source_groups": [
                    "RIGHT_PRECORDIAL_MORPHOLOGY",
                    "LATERAL_MORPHOLOGY",
                ],
                "specialist_confirmed": True,
            }
        },
        "bbb_qrs_audit": {
            "global_qrs_ms": 118.0,
            "global_qrs_relation_120": "UNCERTAIN",
            "qrs_measurement_state": "MEASURED_WITH_UNCERTAINTY",
            "global_qrs_status": "REMEASURE",
            "wide_qrs_lead_n": 3,
            "wide_qrs_limb_lead_n": 1,
            "wide_qrs_precordial_lead_n": 2,
            "qrs_ge_115_lead_n": 4,
            "qrs_ge_118_lead_n": 3,
            "qrs_perlead_median_ms": 119.0,
            "qrs_perlead_max_ms": 130.0,
            "multilead_qrs_ge_120_rescue": False,
            "rescue_shape_support": True,
            "one_wide_lead_short_of_rescue": True,
            "ge4_wide_but_shape_insufficient": False,
            "rbbb_morphology": True,
            "lbbb_morphology": False,
        },
        "fusion_audit": {
            "RBBB_MORPHOLOGY_COMPATIBLE": {
                "publishable": False,
                "fusion_reason": "REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED",
                "fusion_state": "MEASUREMENT_BOUNDARY_UNCERTAIN",
                "score": 0.70,
                "prospective_score_threshold": 0.65,
                "independent_evidence_n": 2,
                "prospective_min_independent_sources": 2,
                "boundary_failure_metrics": ["qrs_ms"],
            }
        },
        "reasoner_audit": {},
    }]
    bbb_metric = _score_target(
        "RBBB_COMPLETE", TARGETS["RBBB_COMPLETE"], bbb_rows, set()
    )
    bqa = bbb_metric["bbb_qrs_fusion_loss_audit"]
    assert bqa["suppressed_candidate_n"] == 1, bqa
    assert bqa["wide_qrs_lead_n"]["3"] == 1, bqa
    assert bqa["one_wide_lead_short_of_rescue_n"] == 1, bqa
    assert bqa["qrs_ge_115_lead_n"]["4"] == 1, bqa
    assert bqa["qrs_ge_118_lead_n"]["3"] == 1, bqa
    assert bqa["qrs_perlead_median_bands"]["118_TO_119_9"] == 1, bqa
    assert bqa["qrs_perlead_max_bands"]["GE_120"] == 1, bqa
    assert bqa["rescue_shape_support_n"] == 1, bqa
    assert bqa["target_morphology_present_n"] == 1, bqa
    assert bqa["global_qrs_value_bands"]["115_TO_119_9"] == 1, bqa

    rbbb_cf_rows = [
        {
            "ecg_id": 41,
            "codes": {"RBBB": 100.0},
            "candidate_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
            "fusion_codes": [],
            "published_codes": [],
            "bbb_qrs_audit": {
                "global_qrs_relation_120": "OVERLAPS",
                "qrs_remeasure": True,
                "qrs_unusable": False,
                "wide_qrs_lead_n": 3,
                "wide_qrs_limb_lead_n": 1,
                "wide_qrs_precordial_lead_n": 2,
                "qrs_ge_118_lead_n": 4,
                "qrs_ge_115_lead_n": 5,
                "rbbb_morphology": True,
            },
        },
        {
            "ecg_id": 42,
            "codes": {"NORM": 100.0},
            "candidate_codes": ["RBBB_MORPHOLOGY_COMPATIBLE"],
            "fusion_codes": [],
            "published_codes": [],
            "bbb_qrs_audit": {
                "global_qrs_relation_120": "OVERLAPS",
                "qrs_remeasure": True,
                "qrs_unusable": False,
                "wide_qrs_lead_n": 3,
                "wide_qrs_limb_lead_n": 1,
                "wide_qrs_precordial_lead_n": 2,
                "qrs_ge_118_lead_n": 3,
                "qrs_ge_115_lead_n": 4,
                "rbbb_morphology": False,
            },
        },
    ]
    cf = _rbbb_qrs_counterfactual_audit(rbbb_cf_rows, {42})
    cfa = cf["THREE_WIDE_DISTRIBUTED_PLUS_GE4_118_FULL_MORPH"]
    assert cfa["recoverable_positive_n"] == 1, cfa
    assert cfa["incremental_negative_trigger_n"] == 0, cfa
    assert cfa["projected_final_sensitivity_if_all_triggers_publish"] == 1.0, cfa
    assert cfa["projected_specificity_lower_bound_if_all_triggers_publish"] == 1.0, cfa

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

    av_miss_rows = [{
        "ecg_id": 20,
        "codes": {"1AVB": 100.0},
        "candidate_codes": [],
        "fusion_codes": [],
        "published_codes": [],
        "candidate_audit": {},
        "pr_multilead_audit": {
            "measurement_state": "REMEASURE_REQUIRED",
            "global_unusable": True,
            "ge2_pr_gt_200_leads": True,
            "ge2_pr_70_to_lt120_leads": False,
        },
        "av_candidate_miss_audit": {
            "evaluable": True,
            "classification": "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED",
            "reason": "",
            "one_to_one": True,
            "stable_pr": True,
            "pr_relation_200": "GT_200",
            "pr_measurement_state": "MEASURED_WITH_UNCERTAINTY",
            "atrial_sequence_regular": True,
            "av_dissociation_phase": False,
            "nonconducted_p_n": 1,
            "conducted_p_n": 4,
            "p_qrs_coupling_fraction": 0.80,
            "global_pr_relation_200": "ABOVE",
            "global_p_wave_reproducible": True,
            "global_p_qrs_coupling_fraction": 0.80,
            "global_pr_confidence": 0.75,
            "avb1_candidate_gate_components_met": True,
        },
        "fusion_audit": {},
        "reasoner_audit": {},
    }]
    av_miss_metric = _score_target(
        "AVB1", TARGETS["AVB1"], av_miss_rows, set()
    )
    ama = av_miss_metric["candidate_miss_audit"]
    assert ama["av_classifications"]["NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED"] == 1, ama
    assert ama["pr_relation_200"]["GT_200"] == 1, ama
    assert ama["pr_measurement_states"]["MEASURED_WITH_UNCERTAINTY"] == 1, ama
    assert ama["one_to_one_n"] == 1, ama
    assert ama["atrial_sequence_regular_n"] == 1, ama
    assert ama["nonconducted_p_present_n"] == 1, ama
    assert ama["ge2_nonconducted_p_n"] == 0, ama
    assert ama["conducted_p_ge2_n"] == 1, ama
    assert ama["p_qrs_coupling_bands"]["0_70_TO_0_899"] == 1, ama
    assert ama["global_pr_relation_200"]["ABOVE"] == 1, ama
    assert ama["global_p_wave_reproducible_n"] == 1, ama
    assert ama["global_coupling_ge_0_55_n"] == 1, ama
    assert ama["global_pr_confidence_ge_0_40_n"] == 1, ama
    assert ama["avb1_candidate_gate_components_met_n"] == 1, ama
    pra = av_miss_metric["pr_multilead_threshold_audit"]
    assert pra["positive_n_with_ge2_pr_gt_200_leads"] == 1, pra
    assert pra["candidate_miss_with_ge2_pr_gt_200_leads"] == 1, pra
    assert pra["global_unusable_with_ge2_pr_gt_200_leads"] == 1, pra

    fp_rows = [{
        "ecg_id": 30,
        "codes": {"NORM": 100.0},
        "candidate_codes": ["AF_COMPATIBLE"],
        "fusion_codes": ["AF_COMPATIBLE"],
        "published_codes": ["AF_COMPATIBLE"],
        "pr_multilead_audit": {
            "measurement_state": "MEASURED_WITH_UNCERTAINTY",
            "global_unusable": False,
            "ge2_pr_gt_200_leads": False,
            "ge2_pr_70_to_lt120_leads": False,
        },
        "candidate_audit": {
            "AF_COMPATIBLE": {
                "score": 0.78,
                "evidence": ["NO_REPRODUCIBLE_P", "RR_IRREGULAR"],
                "source_groups": ["P_WAVE", "RR"],
                "specialist_confirmed": False,
            }
        },
        "fusion_audit": {
            "AF_COMPATIBLE": {
                "publishable": True,
                "score": 0.78,
            }
        },
        "av_candidate_miss_audit": {},
        "reasoner_audit": {},
    }]
    fp_metric = _score_target(
        "AF", TARGETS["AF"], fp_rows, {30}
    )
    fpa = fp_metric["false_positive_evidence_audit"]
    assert fpa["final_false_positive_n"] == 1, fpa
    assert fpa["candidate_present_n"] == 1, fpa
    assert fpa["evidence_counts"]["NO_REPRODUCIBLE_P"] == 1, fpa
    assert fpa["source_group_counts"]["RR"] == 1, fpa
    assert fpa["fusion_score_bands"]["0_70_TO_0_799"] == 1, fpa
    assert fpa["specialist_confirmed_n"] == 0, fpa

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
    ap.add_argument("--analysis-cache-dir", type=Path, default=None)
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
        benchmark(\n            args.workdir,\n            args.output,\n            folds=folds,\n            analysis_cache_dir=args.analysis_cache_dir,\n        )


if __name__ == "__main__":
    main()
