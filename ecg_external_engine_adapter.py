from __future__ import annotations

import math
from typing import Any, Dict, Mapping


EXTERNAL_ENGINE_ADAPTER_VERSION = "MEDCALC_ECG_EXTERNAL_ENGINE_ADAPTER_V1"

SUPPORTED_INPUT_KINDS = {
    "CANONICAL_12_LEAD_DIGITAL",
    "ECG_IMAGE",
    "ECG_PDF",
    "VENDOR_XML",
    "HL7_AECG",
}


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    x = _finite(value)
    if x is None or not (0.0 <= x <= 1.0):
        return None
    return x


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except Exception:
            values = [value]
    return sorted({
        str(item).strip()
        for item in values
        if str(item).strip()
    })


def normalize_external_engine_result(
    raw: Dict[str, Any],
    *,
    engine_id: str,
    engine_version: str,
    input_kind: str,
    code_map: Mapping[str, str] | None = None,
    source_digest: str | None = None,
) -> Dict[str, Any]:
    """Normalize a third-party ECG engine result into an evidence-only contract.

    External engines never receive direct publication, fusion, measurement-
    override, or reasoner authority through this adapter. Vendor findings are
    advisory evidence only until a separately validated integration explicitly
    enables their use in MEDCALC evidence fusion.
    """
    engine_id = str(engine_id or "").strip()
    engine_version = str(engine_version or "").strip()
    input_kind = str(input_kind or "").strip().upper()

    if not engine_id:
        raise ValueError("engine_id is required")
    if not engine_version:
        raise ValueError("engine_version is required")
    if input_kind not in SUPPORTED_INPUT_KINDS:
        raise ValueError(f"Unsupported external ECG input kind: {input_kind!r}")
    if not isinstance(raw, dict):
        raise TypeError("raw external engine result must be a dictionary")

    normalized_map = {
        str(k).strip(): str(v).strip()
        for k, v in dict(code_map or {}).items()
        if str(k).strip() and str(v).strip()
    }

    rejected: list[Dict[str, Any]] = []
    findings: list[Dict[str, Any]] = []

    raw_findings = raw.get("findings") or []
    if isinstance(raw_findings, dict):
        raw_findings = [raw_findings]

    for idx, finding_raw in enumerate(raw_findings):
        if not isinstance(finding_raw, dict):
            rejected.append({
                "kind": "finding",
                "index": idx,
                "reason": "FINDING_NOT_OBJECT",
            })
            continue

        finding = dict(finding_raw)
        vendor_code = str(
            finding.get("vendor_code")
            or finding.get("code")
            or ""
        ).strip()
        label = str(finding.get("label") or finding.get("name") or "").strip()

        if not vendor_code:
            rejected.append({
                "kind": "finding",
                "index": idx,
                "reason": "MISSING_VENDOR_CODE",
            })
            continue

        raw_conf = finding.get("confidence")
        conf = _confidence(raw_conf)
        if raw_conf is not None and conf is None:
            rejected.append({
                "kind": "finding",
                "index": idx,
                "vendor_code": vendor_code,
                "reason": "INVALID_CONFIDENCE_SCALE_EXPECTED_0_TO_1",
            })
            continue

        findings.append({
            "vendor_code": vendor_code,
            "canonical_code": normalized_map.get(vendor_code),
            "label": label or vendor_code,
            "confidence": conf,
            "evidence": _strings(finding.get("evidence")),
            "lead_support": _strings(
                finding.get("lead_support") or finding.get("leads")
            ),
            "source_group": f"EXTERNAL_ENGINE:{engine_id}",
            "independent_external_source": True,
            "advisory_only": True,
            "fusion_eligible": False,
            "publishable": False,
            "requires_validation_before_fusion": True,
        })

    measurements: Dict[str, Dict[str, Any]] = {}
    raw_measurements = raw.get("measurements") or {}
    measurement_items = (
        raw_measurements.items()
        if isinstance(raw_measurements, dict)
        else []
    )

    for name_raw, measurement_raw in measurement_items:
        name = str(name_raw or "").strip()
        if not name:
            continue

        if isinstance(measurement_raw, dict):
            measurement = dict(measurement_raw)
            value_raw = measurement.get("value")
            unit = str(measurement.get("unit") or "").strip() or None
            conf_raw = measurement.get("confidence")
        else:
            value_raw = measurement_raw
            unit = None
            conf_raw = None

        value = _finite(value_raw)
        if value is None:
            rejected.append({
                "kind": "measurement",
                "name": name,
                "reason": "NONFINITE_OR_NONNUMERIC_VALUE",
            })
            continue

        conf = _confidence(conf_raw)
        if conf_raw is not None and conf is None:
            rejected.append({
                "kind": "measurement",
                "name": name,
                "reason": "INVALID_CONFIDENCE_SCALE_EXPECTED_0_TO_1",
            })
            continue

        measurements[name] = {
            "value": value,
            "unit": unit,
            "confidence": conf,
            "status": "EXTERNAL_REPORTED",
            "source": f"EXTERNAL_ENGINE:{engine_id}",
            "usable_for_medcalc_measurement_consensus": False,
            "measurement_override_allowed": False,
            "requires_validation_before_consensus": True,
        }

    return {
        "version": EXTERNAL_ENGINE_ADAPTER_VERSION,
        "engine": {
            "id": engine_id,
            "version": engine_version,
        },
        "input_kind": input_kind,
        "source_digest": str(source_digest or "").strip() or None,
        "validation_status": "UNVALIDATED_EXTERNAL_ENGINE",
        "findings": findings,
        "measurements": measurements,
        "rejected_items": rejected,
        "policy": {
            "direct_publication_allowed": False,
            "fusion_allowed": False,
            "measurement_override_allowed": False,
            "reasoner_override_allowed": False,
            "raw_payload_retained": False,
        },
        "provenance": {
            "external_engine": True,
            "normalized_by": EXTERNAL_ENGINE_ADAPTER_VERSION,
            "mapped_vendor_codes": sorted(normalized_map),
            "finding_n": len(findings),
            "measurement_n": len(measurements),
            "rejected_item_n": len(rejected),
        },
        "invariant": (
            "EXTERNAL_ENGINE_OUTPUT_IS_ADVISORY_EVIDENCE_ONLY_UNTIL_"
            "EXPLICITLY_VALIDATED_FOR_FUSION"
        ),
    }
