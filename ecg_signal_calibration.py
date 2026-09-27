from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def collect_calibration_evidence(
    source: Path,
    *,
    pdf_page_index: int,
    prepared_image: Path,
) -> Dict[str, Any]:
    """Collect only non-clinical scale evidence before signal measurement.

    Printed settings and the calibration pulse may be read from the raster.
    These values establish physical scale; they are never used as the clinical
    measurement result for PR/QRS/QT/ST/rhythm.
    """
    machine: Dict[str, Any] = {}
    pulse: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    try:
        from ecg_machine_header import extract_machine_measurements

        raw = extract_machine_measurements(
            source.name,
            source.read_bytes(),
            int(pdf_page_index),
        )
        machine = {
            "speed_mm_per_s": raw.get("speed_mm_per_s"),
            "gain_mm_per_mV": raw.get("gain_mm_per_mV"),
            "source": raw.get("source"),
            "parsed_field_count": raw.get("parsed_field_count"),
        }
    except Exception as exc:
        errors["printed_settings"] = str(exc)

    try:
        from ecg_photo_engine import detect_calibration_pulse

        raw = detect_calibration_pulse(prepared_image.read_bytes())
        pulse = {
            "detected": bool(raw.get("detected")),
            "confidence": raw.get("confidence"),
            "speed_mm_s": raw.get("speed_mm_s"),
            "gain_mm_mV": raw.get("gain_mm_mV"),
            "height_small_boxes": raw.get("height_small_boxes"),
            "plateau_small_boxes": raw.get("plateau_small_boxes"),
            "reason": raw.get("reason"),
        }
    except Exception as exc:
        errors["calibration_pulse"] = str(exc)

    return {
        "schema": "MEDCALC_CALIBRATION_EVIDENCE_V1",
        "printed_settings": machine,
        "calibration_pulse": pulse,
        "errors": errors,
    }
