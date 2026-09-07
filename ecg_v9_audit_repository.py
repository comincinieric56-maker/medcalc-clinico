from __future__ import annotations
import hashlib
from typing import Any, Dict, Optional, List


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _execute_data(resp):
    return getattr(resp, "data", None) or []


def save_training_annotation(
    client,
    *,
    raw_source_bytes: bytes,
    source_filename: str,
    page_number: int | None,
    annotation: Dict[str, Any],
    annotator_label: str | None = None,
) -> Optional[str]:
    """Persist human ground truth for future ECG model training.

    Must be called with a server-side Supabase service-role client.
    """
    payload = {
        "source_sha256": sha256_bytes(raw_source_bytes),
        "source_filename": source_filename or None,
        "page_number": page_number,
        "annotation_version": str(annotation.get("schema") or "MEDCALC_ECG_ANNOTATION_V9_4"),
        "orientation_deg": annotation.get("orientation_deg"),
        "layout_code": annotation.get("layout"),
        "lead_order": annotation.get("lead_order") or {},
        "lead_boxes": annotation.get("lead_boxes") or {},
        "calibration": annotation.get("calibration") or {},
        "quality": annotation.get("quality") or {},
        "clinical_truth": annotation.get("clinical_truth") or {},
        "diagnoses": annotation.get("diagnoses") or [],
        "exclude_from_training": bool(annotation.get("exclude_from_training", False)),
        "exclusion_reason": annotation.get("exclusion_reason") or None,
        "annotator_label": annotator_label or None,
        "notes": annotation.get("notes") or None,
    }
    try:
        resp = (
            client.table("ecg_training_annotations")
            .upsert(payload, on_conflict="source_sha256,page_number")
            .execute()
        )
        rows = _execute_data(resp)
        return rows[0].get("id") if rows else None
    except Exception as exc:
        raise RuntimeError(f"No se pudo guardar la anotación ECG en Supabase: {exc}") from exc


def list_training_annotations_for_source(client, *, raw_source_bytes: bytes) -> List[Dict[str, Any]]:
    source_sha = sha256_bytes(raw_source_bytes)
    try:
        resp = (
            client.table("ecg_training_annotations")
            .select("id,source_sha256,source_filename,page_number,annotation_version,orientation_deg,layout_code,lead_order,lead_boxes,calibration,quality,clinical_truth,diagnoses,exclude_from_training,exclusion_reason,annotator_label,notes,created_at,updated_at")
            .eq("source_sha256", source_sha)
            .order("page_number")
            .execute()
        )
        return _execute_data(resp)
    except Exception as exc:
        raise RuntimeError(f"No se pudo consultar el progreso de auditoría ECG: {exc}") from exc


def get_training_annotation(client, *, raw_source_bytes: bytes, page_number: int) -> Optional[Dict[str, Any]]:
    source_sha = sha256_bytes(raw_source_bytes)
    try:
        resp = (
            client.table("ecg_training_annotations")
            .select("*")
            .eq("source_sha256", source_sha)
            .eq("page_number", int(page_number))
            .limit(1)
            .execute()
        )
        rows = _execute_data(resp)
        return rows[0] if rows else None
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer la anotación ECG existente: {exc}") from exc


def annotation_row_to_record(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        "schema": row.get("annotation_version") or "MEDCALC_ECG_ANNOTATION_V9_4",
        "source_sha256": row.get("source_sha256"),
        "source_filename": row.get("source_filename"),
        "page_number": row.get("page_number"),
        "orientation_deg": row.get("orientation_deg"),
        "layout": row.get("layout_code"),
        "lead_order": row.get("lead_order") or {},
        "lead_boxes": row.get("lead_boxes") or {},
        "calibration": row.get("calibration") or {},
        "quality": row.get("quality") or {},
        "clinical_truth": row.get("clinical_truth") or {},
        "diagnoses": row.get("diagnoses") or [],
        "exclude_from_training": bool(row.get("exclude_from_training", False)),
        "exclusion_reason": row.get("exclusion_reason") or "",
        "notes": row.get("notes") or "",
        "saved_id": row.get("id"),
        "updated_at": row.get("updated_at"),
    }


def save_prediction_audit(
    client,
    *,
    case_id: str,
    verdict: str,
    annotation: Dict[str, Any],
    note: str = "",
) -> Optional[str]:
    if verdict not in {"correct", "corrected", "exclude"}:
        raise ValueError("verdict inválido")
    truth = annotation.get("clinical_truth") or {}
    payload = {
        "case_id": case_id,
        "verdict": verdict,
        "corrected_orientation_deg": annotation.get("orientation_deg"),
        "corrected_layout_code": annotation.get("layout"),
        "corrected_measurements": truth,
        "corrected_diagnoses": annotation.get("diagnoses") or [],
        "auditor_note": note or annotation.get("notes") or None,
    }
    try:
        resp = client.table("ecg_audits").insert(payload).execute()
        rows = _execute_data(resp)
        return rows[0].get("id") if rows else None
    except Exception as exc:
        raise RuntimeError(f"No se pudo guardar la auditoría ECG: {exc}") from exc
