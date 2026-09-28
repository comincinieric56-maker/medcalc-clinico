from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


AUDIT_VERSION = "MEDCALC_MEASUREMENT_FAILURE_AUDIT_V1"


def _metric_reason(metric: Dict[str, Any]) -> str:
    state = str(metric.get("measurement_state") or "UNKNOWN")
    if state == "REMEASURE_REQUIRED":
        diff = metric.get("canonical_vs_median_abs_diff")
        tol = metric.get("tolerance_ms")
        mad = metric.get("candidate_mad")
        try:
            if diff is not None and tol is not None and float(diff) > 2.0 * float(tol):
                return "CANONICAL_VS_CROSSLEAD_STRONG_DISAGREEMENT"
        except Exception:
            pass
        try:
            if mad is not None and tol is not None and float(mad) > float(tol):
                return "CROSSLEAD_MAD_EXCEEDS_TOLERANCE"
        except Exception:
            pass
        return "STRONG_DISCORDANCE_UNSPECIFIED"

    if state == "UNMEASURABLE":
        if metric.get("canonical_value") is None:
            return "NO_CANONICAL_VALUE"
        if int(metric.get("candidate_n") or 0) < 2 and float(metric.get("canonical_confidence") or 0.0) < 0.20:
            return "LOW_CANONICAL_CONFIDENCE_WITHOUT_CROSSLEAD_SUPPORT"
        return "UNMEASURABLE_UNSPECIFIED"

    if state == "MEASURED_WITH_UNCERTAINTY":
        if int(metric.get("candidate_n") or 0) == 0:
            return "CANONICAL_ONLY"
        if int(metric.get("candidate_n") or 0) == 1:
            return "SINGLE_CROSSLEAD_SOURCE"
        return "MODERATE_CROSSLEAD_DISPERSION"

    if state == "MEASURED_HIGH_CONFIDENCE":
        return "HIGH_CONFIDENCE"

    return "UNKNOWN"


def audit_measurement_consensus(consensus: Dict[str, Any]) -> Dict[str, Any]:
    metrics = consensus.get("metrics") or {}
    rows: Dict[str, Dict[str, Any]] = {}
    reasons = Counter()

    for metric_name, item in metrics.items():
        reason = _metric_reason(item or {})
        state = str((item or {}).get("measurement_state") or "UNKNOWN")
        rows[str(metric_name)] = {
            "measurement_state": state,
            "reason": reason,
            "canonical_value": (item or {}).get("canonical_value"),
            "canonical_confidence": (item or {}).get("canonical_confidence"),
            "candidate_n": (item or {}).get("candidate_n"),
            "candidate_mad": (item or {}).get("candidate_mad"),
            "canonical_vs_median_abs_diff": (item or {}).get("canonical_vs_median_abs_diff"),
            "tolerance_ms": (item or {}).get("tolerance_ms"),
            "uncertainty_ms": (item or {}).get("uncertainty_ms"),
        }
        reasons[f"{metric_name}:{reason}"] += 1

    r = consensus.get("r_peak_verification") or {}
    r_reason = None
    if r.get("evaluable"):
        try:
            agreement = float(r.get("aggregate_agreement"))
        except Exception:
            agreement = None
        lead_n = int(r.get("evaluable_lead_n") or 0)
        if agreement is not None and lead_n >= 2 and agreement < 0.50:
            r_reason = "R_PEAK_XQRS_STRONG_DISAGREEMENT"
        elif agreement is not None and lead_n >= 2 and agreement < 0.65:
            r_reason = "R_PEAK_XQRS_PARTIAL_AGREEMENT"
        else:
            r_reason = "R_PEAK_XQRS_ACCEPTABLE"
    else:
        r_reason = str(r.get("reason") or "R_PEAK_XQRS_NOT_EVALUABLE")

    if "r_peaks" in set(consensus.get("remeasure_targets") or []):
        reasons[f"r_peaks:{r_reason}"] += 1

    return {
        "version": AUDIT_VERSION,
        "remeasure_required": bool(consensus.get("remeasure_required")),
        "remeasure_targets": list(consensus.get("remeasure_targets") or []),
        "unmeasurable_targets": list(consensus.get("unmeasurable_targets") or []),
        "uncertain_targets": list(consensus.get("uncertain_targets") or []),
        "metric_audit": rows,
        "r_peak_reason": r_reason,
        "reason_counts": dict(sorted(reasons.items())),
        "policy": "DIAGNOSTIC_ONLY_DO_NOT_CHANGE_MEASUREMENTS_OR_THRESHOLDS",
    }


def aggregate_measurement_audits(audits: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    remeasure_n = 0
    target_counts = Counter()
    reason_counts = Counter()
    state_counts = Counter()

    for audit in audits:
        total += 1
        if bool(audit.get("remeasure_required")):
            remeasure_n += 1
        for target in audit.get("remeasure_targets") or []:
            target_counts[str(target)] += 1
        for key, value in (audit.get("reason_counts") or {}).items():
            reason_counts[str(key)] += int(value)
        for row in (audit.get("metric_audit") or {}).values():
            state_counts[str((row or {}).get("measurement_state") or "UNKNOWN")] += 1

    def frac(n: int) -> float | None:
        return round(n / total, 6) if total else None

    return {
        "version": AUDIT_VERSION,
        "record_n": total,
        "remeasure_n": remeasure_n,
        "remeasure_rate": frac(remeasure_n),
        "remeasure_target_counts": dict(sorted(target_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))),
        "measurement_state_counts": dict(sorted(state_counts.items())),
        "policy": "AGGREGATE_DEVELOPMENT_AUDIT_ONLY",
    }
