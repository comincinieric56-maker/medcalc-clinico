from __future__ import annotations

import math
from typing import Any, Dict


MEASUREMENT_SERVICE_VERSION = "MEDCALC_ECG_MEASUREMENT_SERVICE_V1"


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def build_measurement_service(
    global_metrics: Dict[str, Dict[str, Any]],
    per_lead: Dict[str, Dict[str, Any]],
    measurement_consensus: Dict[str, Any],
) -> Dict[str, Any]:
    """Expose MEDCALC measurements through a stable provenance-rich contract.

    This layer is descriptive only. It must never recalculate, replace, or
    mutate the canonical measurements selected by the digital measurement
    engine or their consensus states.
    """
    consensus_metrics = dict(measurement_consensus.get("metrics") or {})
    envelopes: Dict[str, Dict[str, Any]] = {}

    for metric_name, global_raw in global_metrics.items():
        global_metric = dict(global_raw or {})
        value = _finite(global_metric.get("value"))
        confidence = _finite(global_metric.get("confidence"))
        consensus = dict(consensus_metrics.get(metric_name) or {})

        source_leads = [
            str(x)
            for x in (consensus.get("source_leads") or [])
            if str(x)
        ]
        lead_values = dict(consensus.get("candidate_values") or {})
        lead_confidences = dict(consensus.get("candidate_confidences") or {})

        lead_beat_n: Dict[str, int] = {}
        for lead in source_leads:
            lead_metric = (
                ((per_lead.get(lead) or {}).get("metrics") or {})
                .get(metric_name) or {}
            )
            try:
                beat_n = int(lead_metric.get("beat_n") or 0)
            except Exception:
                beat_n = 0
            if beat_n > 0:
                lead_beat_n[lead] = beat_n

        interval = consensus.get("uncertainty_interval")
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            uncertainty_interval = [
                _finite(interval[0]),
                _finite(interval[1]),
            ]
        else:
            uncertainty_interval = None

        envelopes[str(metric_name)] = {
            "value": value,
            "unit": global_metric.get("unit"),
            "confidence": confidence,
            "status": global_metric.get("status"),
            "reason": global_metric.get("reason"),
            "measurement_state": consensus.get("measurement_state"),
            "usable": not bool(consensus.get("unusable")),
            "usable_with_uncertainty": bool(
                consensus.get("usable_with_uncertainty")
            ),
            "remeasure_required": bool(consensus.get("remeasure")),
            "uncertainty": {
                "absolute": _finite(consensus.get("uncertainty_ms")),
                "unit": "ms" if consensus.get("uncertainty_ms") is not None else None,
                "interval": uncertainty_interval,
                "sources": list(consensus.get("uncertainty_sources") or []),
            },
            "crosslead_dispersion": {
                "median": _finite(consensus.get("candidate_median")),
                "mad": _finite(consensus.get("candidate_mad")),
                "iqr": _finite(consensus.get("candidate_iqr")),
                "canonical_vs_median_abs_diff": _finite(
                    consensus.get("canonical_vs_median_abs_diff")
                ),
            },
            "provenance": {
                "canonical_source": "MEDCALC_DIGITAL_MEASUREMENT_ENGINE",
                "source_leads": source_leads,
                "source_lead_n": len(source_leads),
                "lead_values": lead_values,
                "lead_confidences": lead_confidences,
                "lead_beat_n": lead_beat_n,
                "beat_n_total": sum(lead_beat_n.values()),
            },
            "invariant": "DESCRIPTIVE_ENVELOPE_DOES_NOT_OVERRIDE_CANONICAL_VALUE",
        }

    return {
        "version": MEASUREMENT_SERVICE_VERSION,
        "source": "MEDCALC_DIGITAL_MEASUREMENT_ENGINE_PLUS_CONSENSUS_AUDIT",
        "metrics": envelopes,
        "overall_measurement_quality": _finite(
            measurement_consensus.get("overall_measurement_quality")
        ),
        "remeasure_required": bool(
            measurement_consensus.get("remeasure_required")
        ),
        "remeasure_targets": list(
            measurement_consensus.get("remeasure_targets") or []
        ),
        "unmeasurable_targets": list(
            measurement_consensus.get("unmeasurable_targets") or []
        ),
        "uncertain_targets": list(
            measurement_consensus.get("uncertain_targets") or []
        ),
        "policy": (
            "READ_ONLY_MEASUREMENT_ENVELOPE; CANONICAL_VALUES_AND_CONSENSUS_"
            "STATES_REMAIN_AUTHORITATIVE"
        ),
    }
