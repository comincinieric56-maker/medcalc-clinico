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


def _adult_rows(df: pd.DataFrame, fold: int) -> pd.DataFrame:
    out = df.copy()
    out["_age"] = pd.to_numeric(out.get("age"), errors="coerce")
    out["_fold"] = pd.to_numeric(out.get("strat_fold"), errors="coerce")
    out = out[(out["_age"] >= 18.0) & (out["_fold"] == int(fold))].copy()
    out["_codes"] = out["scp_codes"].map(_parse_codes)
    out["_hash"] = out["ecg_id"].astype(str).map(_hash)
    return out


def select_records(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    adult = _adult_rows(df, INTERNAL_VALIDATION_FOLD)
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
        "fold": INTERNAL_VALIDATION_FOLD,
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


def _fusion_codes(analysis: dict[str, Any]) -> set[str]:
    by_code = ((analysis.get("evidence_fusion") or {}).get("by_code") or {})
    return {str(k) for k, v in by_code.items() if bool((v or {}).get("publishable"))}


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

    return {
        "positive_n": n,
        "negative_control_n": len(negatives),
        "candidate_detected_n": candidate_n,
        "fusion_publishable_n": fusion_n,
        "final_published_n": final_n,
        "false_positive_n_on_clean_controls": fp,
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


def benchmark(workdir: Path, output: Path) -> dict[str, Any]:
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path = workdir / "ptbxl_database.csv"
    statements_path = workdir / "scp_statements.csv"
    _download(f"{BASE}/ptbxl_database.csv", metadata_path)
    _download(f"{BASE}/scp_statements.csv", statements_path)

    meta = pd.read_csv(metadata_path)
    selected, selection = select_records(meta)
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
        "role": "DEVELOPMENT_INTERNAL_VALIDATION_ONLY",
        "external_validation_claim_allowed": False,
        "fold_policy": {
            "tuning_folds": [1,2,3,4,5,6,7,8],
            "internal_validation_fold": 9,
            "internal_confirmation_fold": 10,
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
    selected, summary = select_records(df)
    ids = set(selected["ecg_id"].astype(int))
    assert 1 in ids and 3 in ids and 4 in ids, (ids, summary)
    assert 2 not in ids and 5 not in ids, (ids, summary)
    assert summary["positive_available_by_target"]["AF"] == 1, summary
    assert summary["positive_available_by_target"]["RBBB_COMPLETE"] == 1, summary
    assert summary["negative_control_n"] == 1, summary
    print("MEDCALC_ADULT_DIAGNOSTIC_DEV_SELFTEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/medcalc-ptbxl-dev"))
    ap.add_argument("--output", type=Path, default=Path("/tmp/MEDCALC_ADULT_PTBXL_DEV.json"))
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        benchmark(args.workdir, args.output)


if __name__ == "__main__":
    main()
