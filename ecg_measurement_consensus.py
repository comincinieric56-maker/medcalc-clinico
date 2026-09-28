from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CONSENSUS_VERSION = "MEDCALC_MEASUREMENT_CONSENSUS_V2"

MEASURED_HIGH_CONFIDENCE = "MEASURED_HIGH_CONFIDENCE"
MEASURED_WITH_UNCERTAINTY = "MEASURED_WITH_UNCERTAINTY"
REMEASURE_REQUIRED = "REMEASURE_REQUIRED"
UNMEASURABLE = "UNMEASURABLE"


def _finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _longest_finite_run(x: np.ndarray) -> tuple[int, int] | None:
    finite = np.isfinite(x)
    if not finite.any():
        return None
    d = np.diff(np.r_[False, finite, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    if not len(starts):
        return None
    return max(((int(a), int(b)) for a, b in zip(starts, ends)), key=lambda z: z[1] - z[0])


def _quality_fraction(item: Dict[str, Any], a: int, b: int) -> float:
    q = np.asarray(item.get("quality_mask", []), dtype=np.uint8)
    if q.size < b or b <= a:
        return 0.0
    seg = q[a:b]
    return float(np.mean(np.where(seg == 2, 1.0, np.where(seg == 1, 0.55, 0.0))))


def _match_peaks(
    reference: np.ndarray,
    candidate: np.ndarray,
    tolerance_samples: int,
) -> tuple[int, list[int]]:
    if reference.size == 0 or candidate.size == 0:
        return 0, []
    used: set[int] = set()
    errors: list[int] = []
    for r in reference:
        distances = np.abs(candidate - int(r))
        order = np.argsort(distances)
        for idx in order:
            j = int(idx)
            if j in used:
                continue
            if int(distances[j]) <= int(tolerance_samples):
                used.add(j)
                errors.append(int(candidate[j]) - int(r))
            break
    return len(errors), errors


def _verify_r_peaks_xqrs(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    fs = int(canonical_ecg.get("fs") or 500)
    leads = canonical_ecg.get("leads") or {}
    rows: Dict[str, Any] = {}

    try:
        from wfdb import processing as wfdb_processing
    except Exception as exc:
        return {
            "evaluable": False,
            "reason": f"WFDB_XQRS_UNAVAILABLE:{type(exc).__name__}",
            "per_lead": {},
            "aggregate_agreement": None,
        }

    for lead in LEADS:
        item = dict(leads.get(lead) or {})
        signal = np.asarray(
            [np.nan if v is None else float(v) for v in item.get("signal_mv", [])],
            dtype=float,
        )
        span = _longest_finite_run(signal)
        native = np.asarray((per_lead.get(lead) or {}).get("r_peaks_samples") or [], dtype=int)
        if span is None:
            rows[lead] = {"evaluable": False, "reason": "NO_FINITE_SIGNAL"}
            continue
        a, b = span
        duration_s = (b - a) / float(fs)
        if duration_s < 1.8:
            rows[lead] = {
                "evaluable": False,
                "reason": "CONTIGUOUS_SIGNAL_LT_1_8S",
                "duration_s": round(duration_s, 3),
            }
            continue
        quality = _quality_fraction(item, a, b)
        if quality < 0.55:
            rows[lead] = {
                "evaluable": False,
                "reason": "LOW_SIGNAL_QUALITY",
                "quality": round(quality, 6),
            }
            continue

        segment = np.asarray(signal[a:b], dtype=float)
        try:
            xqrs = np.asarray(
                wfdb_processing.xqrs_detect(
                    sig=segment,
                    fs=fs,
                    learn=True,
                    verbose=False,
                ),
                dtype=int,
            )
        except Exception as exc:
            rows[lead] = {
                "evaluable": False,
                "reason": f"XQRS_FAILED:{type(exc).__name__}",
            }
            continue

        native_local = native[(native >= a) & (native < b)] - int(a)
        if native_local.size < 3 or xqrs.size < 3:
            rows[lead] = {
                "evaluable": False,
                "reason": "LT_3_PEAKS_FOR_COMPARISON",
                "native_n": int(native_local.size),
                "xqrs_n": int(xqrs.size),
            }
            continue

        tolerance = max(1, int(round(0.050 * fs)))
        matched, errors = _match_peaks(native_local, xqrs, tolerance)
        precision = matched / float(max(int(xqrs.size), 1))
        recall = matched / float(max(int(native_local.size), 1))
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        abs_err_ms = (
            np.abs(np.asarray(errors, dtype=float)) * 1000.0 / float(fs)
            if errors
            else np.asarray([], dtype=float)
        )
        median_error_ms = float(np.median(abs_err_ms)) if abs_err_ms.size else None
        p95_error_ms = float(np.percentile(abs_err_ms, 95)) if abs_err_ms.size else None
        timing_score = (
            float(np.clip(1.0 - float(median_error_ms or 50.0) / 50.0, 0.0, 1.0))
            if matched
            else 0.0
        )
        agreement = float(np.clip(0.75 * f1 + 0.25 * timing_score, 0.0, 1.0))
        rows[lead] = {
            "evaluable": True,
            "source": "WFDB_XQRS_INDEPENDENT_VERIFIER",
            "duration_s": round(duration_s, 3),
            "quality": round(quality, 6),
            "native_n": int(native_local.size),
            "xqrs_n": int(xqrs.size),
            "matched_n": int(matched),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "median_abs_timing_error_ms": (
                round(median_error_ms, 3) if median_error_ms is not None else None
            ),
            "p95_abs_timing_error_ms": (
                round(p95_error_ms, 3) if p95_error_ms is not None else None
            ),
            "agreement": round(agreement, 6),
        }

    scores = [
        float(v["agreement"])
        for v in rows.values()
        if v.get("evaluable") and v.get("agreement") is not None
    ]
    aggregate = float(np.median(scores)) if scores else None
    return {
        "evaluable": bool(scores),
        "source": "WFDB_XQRS_INDEPENDENT_VERIFIER",
        "per_lead": rows,
        "evaluable_lead_n": len(scores),
        "aggregate_agreement": round(aggregate, 6) if aggregate is not None else None,
        "status": (
            "AGREED"
            if aggregate is not None and aggregate >= 0.80
            else "PARTIAL_AGREEMENT"
            if aggregate is not None and aggregate >= 0.65
            else "DISCORDANT"
            if aggregate is not None
            else "INSUFFICIENT"
        ),
    }


def _metric_candidates(
    per_lead: Dict[str, Dict[str, Any]],
    metric_name: str,
) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for lead in LEADS:
        metric = ((per_lead.get(lead) or {}).get("metrics") or {}).get(metric_name) or {}
        value = _finite_float(metric.get("value"))
        confidence = _finite_float(metric.get("confidence")) or 0.0
        if value is None or confidence < 0.20:
            continue
        rows.append((lead, value, confidence))
    return rows


def _explicit_time_uncertainty_ms(canonical_ecg: Dict[str, Any]) -> tuple[float | None, str | None]:
    """Read digitizer/grid timing uncertainty when an upstream stage provides it."""
    containers = [
        canonical_ecg,
        canonical_ecg.get("calibration") or {},
        canonical_ecg.get("uncertainty") or {},
        canonical_ecg.get("metadata") or {},
    ]
    keys = (
        "timing_uncertainty_ms",
        "time_uncertainty_ms",
        "grid_time_uncertainty_ms",
        "horizontal_uncertainty_ms",
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = _finite_float(container.get(key))
            if value is not None and value >= 0:
                return float(value), key
    return None, None


def _estimate_uncertainty_ms(
    *,
    fs: int,
    mad: float,
    disagreement: float | None,
    canonical_confidence: float,
    explicit_digitization_uncertainty_ms: float | None,
) -> tuple[float, list[str]]:
    components: list[float] = []
    sources: list[str] = []

    # Two samples is a conservative timing floor for a reconstructed signal.
    if fs > 0:
        components.append(2.0 * 1000.0 / float(fs))
        sources.append("RECONSTRUCTED_SIGNAL_SAMPLING_FLOOR")

    if explicit_digitization_uncertainty_ms is not None:
        components.append(float(explicit_digitization_uncertainty_ms))
        sources.append("UPSTREAM_DIGITIZATION_OR_GRID_UNCERTAINTY")

    if mad > 0:
        components.append(1.4826 * float(mad))
        sources.append("CROSS_LEAD_MAD")

    if disagreement is not None and disagreement > 0:
        components.append(0.50 * float(disagreement))
        sources.append("CANONICAL_VS_CROSSLEAD_DISAGREEMENT")

    # Low confidence should widen, not invalidate, an otherwise finite value.
    if canonical_confidence < 0.65:
        confidence_penalty = (0.65 - max(canonical_confidence, 0.0)) * 20.0
        if confidence_penalty > 0:
            components.append(confidence_penalty)
            sources.append("CANONICAL_CONFIDENCE_PENALTY")

    uncertainty = max(components) if components else 10.0
    return float(uncertainty), sources


def _audit_metric(
    per_lead: Dict[str, Dict[str, Any]],
    global_metrics: Dict[str, Dict[str, Any]],
    metric_name: str,
    *,
    tolerance_ms: float,
    fs: int,
    explicit_digitization_uncertainty_ms: float | None,
) -> Dict[str, Any]:
    canonical = dict(global_metrics.get(metric_name) or {})
    canonical_value = _finite_float(canonical.get("value"))
    canonical_confidence = _finite_float(canonical.get("confidence")) or 0.0
    rows = _metric_candidates(per_lead, metric_name)
    values = np.asarray([v for _, v, _ in rows], dtype=float)

    if values.size:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) if values.size >= 2 else 0.0
        iqr = float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size >= 4 else None
        disagreement = abs(canonical_value - median) if canonical_value is not None else None
    else:
        median = None
        mad = 0.0
        iqr = None
        disagreement = None

    if canonical_value is None:
        agreement_status = "NO_CANONICAL_VALUE" if rows else "INSUFFICIENT"
        state = UNMEASURABLE
        uncertainty_ms = None
        uncertainty_sources: list[str] = []
    else:
        if len(rows) >= 2 and disagreement is not None and disagreement <= tolerance_ms and mad <= tolerance_ms / 2.0:
            agreement_status = "AGREED"
        elif len(rows) == 1:
            agreement_status = "SINGLE_SOURCE"
        elif not rows:
            agreement_status = "CANONICAL_ONLY"
        else:
            agreement_status = "DISCORDANT"

        uncertainty_ms, uncertainty_sources = _estimate_uncertainty_ms(
            fs=fs,
            mad=mad,
            disagreement=disagreement,
            canonical_confidence=canonical_confidence,
            explicit_digitization_uncertainty_ms=explicit_digitization_uncertainty_ms,
        )

        # Remeasure is reserved for strong contradictory evidence. Moderate
        # disagreement becomes an uncertainty statement rather than a veto.
        strong_discordance = bool(
            len(rows) >= 2
            and (
                (disagreement is not None and disagreement > 2.0 * tolerance_ms)
                or mad > tolerance_ms
            )
        )
        if strong_discordance:
            state = REMEASURE_REQUIRED
        elif canonical_confidence < 0.20 and len(rows) < 2:
            state = UNMEASURABLE
        elif (
            canonical_confidence >= 0.65
            and (
                (len(rows) >= 2 and disagreement is not None and disagreement <= tolerance_ms / 2.0 and mad <= tolerance_ms / 4.0)
                or (len(rows) == 1 and disagreement is not None and disagreement <= tolerance_ms / 2.0)
            )
        ):
            state = MEASURED_HIGH_CONFIDENCE
        else:
            state = MEASURED_WITH_UNCERTAINTY

    ci_low = (
        float(canonical_value - uncertainty_ms)
        if canonical_value is not None and uncertainty_ms is not None
        else None
    )
    ci_high = (
        float(canonical_value + uncertainty_ms)
        if canonical_value is not None and uncertainty_ms is not None
        else None
    )

    return {
        "metric": metric_name,
        "status": agreement_status,
        "measurement_state": state,
        "canonical_value": canonical_value,
        "canonical_confidence": canonical_confidence,
        "candidate_median": round(median, 6) if median is not None else None,
        "candidate_mad": round(mad, 6),
        "candidate_iqr": round(iqr, 6) if iqr is not None else None,
        "candidate_n": len(rows),
        "source_leads": [lead for lead, _, _ in rows],
        "candidate_values": {lead: round(value, 6) for lead, value, _ in rows},
        "candidate_confidences": {lead: round(conf, 6) for lead, _, conf in rows},
        "canonical_vs_median_abs_diff": (
            round(float(disagreement), 6) if disagreement is not None else None
        ),
        "tolerance_ms": tolerance_ms,
        "uncertainty_ms": round(float(uncertainty_ms), 6) if uncertainty_ms is not None else None,
        "uncertainty_interval": (
            [round(ci_low, 6), round(ci_high, 6)]
            if ci_low is not None and ci_high is not None
            else None
        ),
        "uncertainty_sources": uncertainty_sources,
        "remeasure": state == REMEASURE_REQUIRED,
        "unusable": state in {REMEASURE_REQUIRED, UNMEASURABLE},
        "usable_with_uncertainty": state == MEASURED_WITH_UNCERTAINTY,
    }


def threshold_relation(
    consensus: Dict[str, Any],
    metric_name: str,
    threshold: float,
) -> str:
    """Classify a numeric threshold against the measurement uncertainty interval."""
    item = ((consensus.get("metrics") or {}).get(metric_name) or {})
    state = str(item.get("measurement_state") or "")
    if state in {REMEASURE_REQUIRED, UNMEASURABLE, ""}:
        return "UNUSABLE"

    value = _finite_float(item.get("canonical_value"))
    interval = item.get("uncertainty_interval")
    if value is None:
        return "UNUSABLE"
    if isinstance(interval, (list, tuple)) and len(interval) == 2:
        low = _finite_float(interval[0])
        high = _finite_float(interval[1])
    else:
        low = high = value
    if low is None or high is None:
        low = high = value

    t = float(threshold)
    if low > t:
        return "ABOVE"
    if high < t:
        return "BELOW"
    if low <= t <= high:
        return "OVERLAPS"
    return "UNUSABLE"


def build_measurement_consensus(
    canonical_ecg: Dict[str, Any],
    per_lead: Dict[str, Dict[str, Any]],
    global_metrics: Dict[str, Dict[str, Any]],
    rhythm: Dict[str, Any],
    axis: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit canonical digital measurements without silently replacing them.

    V2 separates uncertainty from true contradictory evidence:
      - MEASURED_HIGH_CONFIDENCE
      - MEASURED_WITH_UNCERTAINTY
      - REMEASURE_REQUIRED
      - UNMEASURABLE

    A moderate cross-lead spread no longer makes the whole ECG "remeasure".
    Threshold-sensitive diagnostic rules can use the returned uncertainty
    interval and abstain only when the relevant boundary is actually crossed.
    """
    fs = int(canonical_ecg.get("fs") or 500)
    explicit_time_uncertainty_ms, explicit_time_uncertainty_source = (
        _explicit_time_uncertainty_ms(canonical_ecg)
    )

    metric_audit = {
        "pr_ms": _audit_metric(
            per_lead, global_metrics, "pr_ms",
            tolerance_ms=20.0, fs=fs,
            explicit_digitization_uncertainty_ms=explicit_time_uncertainty_ms,
        ),
        "qrs_ms": _audit_metric(
            per_lead, global_metrics, "qrs_ms",
            tolerance_ms=20.0, fs=fs,
            explicit_digitization_uncertainty_ms=explicit_time_uncertainty_ms,
        ),
        "qt_ms": _audit_metric(
            per_lead, global_metrics, "qt_ms",
            tolerance_ms=30.0, fs=fs,
            explicit_digitization_uncertainty_ms=explicit_time_uncertainty_ms,
        ),
        "p_duration_ms": _audit_metric(
            per_lead, global_metrics, "p_duration_ms",
            tolerance_ms=20.0, fs=fs,
            explicit_digitization_uncertainty_ms=explicit_time_uncertainty_ms,
        ),
    }
    r_verification = _verify_r_peaks_xqrs(canonical_ecg, per_lead)

    remeasure_targets = [
        name for name, item in metric_audit.items()
        if str(item.get("measurement_state") or "") == REMEASURE_REQUIRED
    ]
    unmeasurable_targets = [
        name for name, item in metric_audit.items()
        if str(item.get("measurement_state") or "") == UNMEASURABLE
    ]
    uncertain_targets = [
        name for name, item in metric_audit.items()
        if str(item.get("measurement_state") or "") == MEASURED_WITH_UNCERTAINTY
    ]

    r_score = _finite_float(r_verification.get("aggregate_agreement"))
    if (
        r_score is not None
        and r_score < 0.50
        and int(r_verification.get("evaluable_lead_n") or 0) >= 2
    ):
        remeasure_targets.append("r_peaks")
    elif (
        r_score is not None
        and r_score < 0.65
        and int(r_verification.get("evaluable_lead_n") or 0) >= 2
    ):
        uncertain_targets.append("r_peaks")

    confidences = [
        _finite_float((global_metrics.get(name) or {}).get("confidence"))
        for name in ("pr_ms", "qrs_ms", "qt_ms")
    ]
    confidences = [c for c in confidences if c is not None]
    base_quality = float(np.median(confidences)) if confidences else 0.0
    if r_score is not None:
        base_quality = 0.70 * base_quality + 0.30 * r_score

    unusable_targets = sorted(set(remeasure_targets) | set(unmeasurable_targets))
    state_counts: Dict[str, int] = {}
    for item in metric_audit.values():
        state = str(item.get("measurement_state") or "UNKNOWN")
        state_counts[state] = state_counts.get(state, 0) + 1

    return {
        "version": CONSENSUS_VERSION,
        "policy": (
            "VERIFY_DO_NOT_OVERRIDE; MODERATE_DISAGREEMENT_PROPAGATES_UNCERTAINTY; "
            "REMEASURE_ONLY_FOR_STRONG_CONTRADICTION"
        ),
        "canonical_source": "MEDCALC_DIGITAL_MEASUREMENT_ENGINE",
        "metrics": metric_audit,
        "measurement_states": {
            name: item.get("measurement_state")
            for name, item in metric_audit.items()
        },
        "state_counts": state_counts,
        "r_peak_verification": r_verification,
        "rhythm_reference": {
            "lead": rhythm.get("lead"),
            "heart_rate_bpm": rhythm.get("heart_rate_bpm"),
            "rr_cv": rhythm.get("rr_cv"),
            "confidence": rhythm.get("confidence"),
        },
        "axis_reference": {
            "degrees": axis.get("degrees"),
            "confidence": axis.get("confidence"),
            "source": axis.get("source"),
        },
        "explicit_digitization_time_uncertainty_ms": explicit_time_uncertainty_ms,
        "explicit_digitization_time_uncertainty_source": explicit_time_uncertainty_source,
        "remeasure_required": bool(remeasure_targets),
        "remeasure_targets": sorted(set(remeasure_targets)),
        "unmeasurable_targets": sorted(set(unmeasurable_targets)),
        "uncertain_targets": sorted(set(uncertain_targets)),
        "unusable_targets": unusable_targets,
        "overall_measurement_quality": round(float(np.clip(base_quality, 0.0, 1.0)), 6),
    }
