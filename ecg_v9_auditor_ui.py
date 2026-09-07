from __future__ import annotations
import io, json, hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

RHYTHMS = [
    "Ritmo sinusal",
    "Arritmia sinusal",
    "Taquicardia sinusal",
    "Bradicardia sinusal",
    "Fibrilación auricular",
    "Fibrilación auricular con respuesta ventricular rápida",
    "Fibrilación auricular con respuesta ventricular lenta",
    "Flutter auricular",
    "Taquicardia auricular",
    "Taquicardia auricular multifocal",
    "Taquicardia supraventricular regular",
    "Ritmo de la unión",
    "Ritmo idioventricular",
    "Taquicardia ventricular",
    "Marcapasos",
    "Otro",
    "No valorable",
]

CONDUCTION = [
    "Sin trastorno de conducción",
    "BAV de primer grado",
    "BAV Mobitz I",
    "BAV Mobitz II",
    "BAV completo",
    "BCRD completo",
    "BCRD incompleto",
    "BCRI completo",
    "BCRI incompleto",
    "Bloqueo fascicular anterior izquierdo",
    "Bloqueo fascicular posterior izquierdo",
    "Bloqueo bifascicular",
    "Trastorno inespecífico de conducción intraventricular",
    "Preexcitación / WPW",
    "Otro",
    "No valorable",
]

ST_OPTIONS = [
    "Isoeléctrico / sin alteración significativa",
    "Elevación del ST",
    "Depresión del ST",
    "Alteración secundaria de repolarización",
    "No valorable",
]

T_OPTIONS = [
    "Morfología/polaridad sin alteración significativa",
    "Invertida",
    "Alta/picuda",
    "Aplanada",
    "Bifásica",
    "Alteración secundaria de repolarización",
    "No valorable",
]

P_OPTIONS = [
    "Presente, organizada",
    "Ausente",
    "Actividad auricular no organizada / ondas f",
    "Ondas F de flutter",
    "Morfología variable",
    "No valorable",
]

LAYOUTS = ["3x4", "6x2", "otro", "no_valorable"]
ORIENTATIONS = [0, 90, 180, 270]

STANDARD_3X4 = {
    "row1": ["I", "aVR", "V1", "V4"],
    "row2": ["II", "aVL", "V2", "V5"],
    "row3": ["III", "aVF", "V3", "V6"],
}
STANDARD_6X2 = {
    "row1": ["I", "V1"],
    "row2": ["II", "V2"],
    "row3": ["III", "V3"],
    "row4": ["aVR", "V4"],
    "row5": ["aVL", "V5"],
    "row6": ["aVF", "V6"],
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def page_count(raw: bytes, filename: str) -> int:
    if filename.lower().endswith(".pdf"):
        if fitz is None:
            raise RuntimeError("PyMuPDF no está instalado")
        doc = fitz.open(stream=raw, filetype="pdf")
        n = len(doc)
        doc.close()
        return n
    return 1


def render_source_page(raw: bytes, filename: str, page_index: int = 0, dpi: int = 170) -> Image.Image:
    if filename.lower().endswith(".pdf"):
        if fitz is None:
            raise RuntimeError("PyMuPDF no está instalado")
        doc = fitz.open(stream=raw, filetype="pdf")
        page = doc[int(page_index)]
        scale = float(dpi) / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")


def rotate_clockwise(img: Image.Image, deg: int) -> Image.Image:
    deg = int(deg) % 360
    if deg == 90:
        return img.transpose(Image.Transpose.ROTATE_270)
    if deg == 180:
        return img.transpose(Image.Transpose.ROTATE_180)
    if deg == 270:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img.copy()


def _option_index(options, value, default=0):
    try:
        return options.index(value)
    except Exception:
        return default


def _nullable_number(st, label: str, key: str, max_value: float, step: float = 1.0, help_text: str | None = None, default=None):
    c1, c2 = st.columns([1, 3])
    measured_default = default is not None
    measured = c1.checkbox("Medido", key=f"{key}_measured", value=measured_default)
    start = float(default) if default is not None else 0.0
    value = c2.number_input(label, min_value=0.0, max_value=float(max_value), value=start, step=float(step), key=key, disabled=not measured, help=help_text)
    return float(value) if measured else None


def _signed_nullable_number(st, label: str, key: str, min_value: float, max_value: float, step: float = 1.0, default=None):
    c1, c2 = st.columns([1, 3])
    measured_default = default is not None
    measured = c1.checkbox("Medido", key=f"{key}_measured", value=measured_default)
    start = float(default) if default is not None else 0.0
    value = c2.number_input(label, min_value=float(min_value), max_value=float(max_value), value=start, step=float(step), key=key, disabled=not measured)
    return float(value) if measured else None


def _lead_order_for_layout(layout: str) -> Dict[str, Any]:
    if layout == "3x4":
        return STANDARD_3X4
    if layout == "6x2":
        return STANDARD_6X2
    return {}


def build_annotation(
    *,
    raw_source: bytes,
    filename: str,
    page_number: int | None,
    orientation_deg: int,
    layout: str,
    rhythm_strip_lead: str | None,
    quality: Dict[str, Any],
    calibration: Dict[str, Any],
    clinical_truth: Dict[str, Any],
    diagnoses: list[str],
    exclude_from_training: bool,
    exclusion_reason: str,
    notes: str,
) -> Dict[str, Any]:
    return {
        "schema": "MEDCALC_ECG_ANNOTATION_V9_4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_bytes(raw_source),
        "source_filename": filename,
        "page_number": page_number,
        "orientation_deg": int(orientation_deg),
        "layout": layout,
        "lead_order": _lead_order_for_layout(layout),
        "rhythm_strip_lead": rhythm_strip_lead,
        "lead_boxes": {},
        "quality": quality,
        "calibration": calibration,
        "clinical_truth": clinical_truth,
        "diagnoses": diagnoses,
        "exclude_from_training": bool(exclude_from_training),
        "exclusion_reason": exclusion_reason or "",
        "notes": notes or "",
    }


def render_auditor(st, *, raw_source: bytes, filename: str, page_index: int = 0, prediction: Optional[Dict[str, Any]] = None, existing_annotation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Render a full human-in-the-loop annotation panel and return the current record."""
    base = render_source_page(raw_source, filename, page_index=page_index, dpi=170)
    existing = existing_annotation or {}
    existing_quality = existing.get("quality") or {}
    existing_cal = existing.get("calibration") or {}
    existing_truth = existing.get("clinical_truth") or {}

    st.markdown("## 1. Verdad visual del ECG")
    st.caption("Primero corrige la hoja. La interpretación clínica se registra después de fijar orientación y layout.")

    c1, c2, c3 = st.columns([1, 1, 1])
    orientation = c1.selectbox("Rotación necesaria para dejarlo al derecho", ORIENTATIONS, index=_option_index(ORIENTATIONS, existing.get("orientation_deg"), 0), key=f"audit_orientation_{page_index}")
    layout = c2.selectbox("Layout real", LAYOUTS, index=_option_index(LAYOUTS, existing.get("layout"), 0), key=f"audit_layout_{page_index}")
    strip_options = ["II", "V1", "V5", "V6", "otra", "ninguna/no identificable"]
    existing_strip = existing.get("rhythm_strip_lead") or "ninguna/no identificable"
    strip = c3.selectbox("Tira larga", strip_options, index=_option_index(strip_options, existing_strip, 0), key=f"audit_strip_{page_index}")

    corrected = rotate_clockwise(base, orientation)
    st.image(corrected, caption=f"Vista auditada · rotación {orientation}° · layout {layout}", use_container_width=True)

    if prediction:
        with st.expander("Comparar con predicción del motor"):
            st.json(prediction)

    st.markdown("## 2. Calidad y calibración")
    q1, q2, q3, q4 = st.columns(4)
    readable = q1.checkbox("Trazado legible", value=bool(existing_quality.get("readable", True)), key=f"q_read_{page_index}")
    folded = q2.checkbox("Papel doblado/curvado", value=bool(existing_quality.get("folded_or_curved", False)), key=f"q_fold_{page_index}")
    shadow = q3.checkbox("Sombras/reflejos", value=bool(existing_quality.get("shadow_or_glare", False)), key=f"q_shadow_{page_index}")
    cropped = q4.checkbox("ECG recortado/incompleto", value=bool(existing_quality.get("cropped_or_incomplete", False)), key=f"q_crop_{page_index}")

    cal1, cal2, cal3 = st.columns(3)
    speed_options = [25, 50, "no demostrada"]
    gain_options = [10, 5, 20, "no demostrada"]
    existing_speed = existing_cal.get("speed_mm_s") if existing_cal.get("speed_mm_s") is not None else "no demostrada"
    existing_gain = existing_cal.get("gain_mm_mv") if existing_cal.get("gain_mm_mv") is not None else "no demostrada"
    speed = cal1.selectbox("Velocidad", speed_options, index=_option_index(speed_options, existing_speed, 0), key=f"speed_{page_index}")
    gain = cal2.selectbox("Ganancia", gain_options, index=_option_index(gain_options, existing_gain, 0), key=f"gain_{page_index}")
    pulse = cal3.checkbox("Pulso de calibración visible", value=bool(existing_cal.get("pulse_visible", True)), key=f"pulse_{page_index}")

    st.markdown("## 3. Verdad clínica")
    st.caption("No es obligatorio completar una medición si no puede sostenerse visualmente. Déjala como no medida.")

    rhythm = st.selectbox("Ritmo correcto", RHYTHMS, index=_option_index(RHYTHMS, existing_truth.get("rhythm"), 0), key=f"rhythm_{page_index}")
    rr_options = ["regular", "irregular", "irregularmente irregular", "no valorable"]
    rr_pattern = st.selectbox("Regularidad RR", rr_options, index=_option_index(rr_options, existing_truth.get("rr_pattern"), 0), key=f"rr_{page_index}")
    p_status = st.selectbox("Actividad auricular / onda P", P_OPTIONS, index=_option_index(P_OPTIONS, existing_truth.get("p_status"), 0), key=f"p_{page_index}")

    m1, m2 = st.columns(2)
    with m1:
        hr = _nullable_number(st, "Frecuencia ventricular (lpm)", f"hr_{page_index}", 400, 1, default=existing_truth.get("heart_rate_bpm"))
        p_ms = _nullable_number(st, "Duración P (ms)", f"pms_{page_index}", 300, 1, default=existing_truth.get("p_duration_ms"))
        pr_ms = _nullable_number(st, "PR (ms)", f"pr_{page_index}", 600, 1, default=existing_truth.get("pr_ms"))
        qrs_ms = _nullable_number(st, "QRS (ms)", f"qrs_{page_index}", 600, 1, default=existing_truth.get("qrs_ms"))
    with m2:
        qt_ms = _nullable_number(st, "QT (ms)", f"qt_{page_index}", 1000, 1, default=existing_truth.get("qt_ms"))
        qtc_f = _nullable_number(st, "QTc Fridericia (ms)", f"qtcf_{page_index}", 1000, 1, default=existing_truth.get("qtc_fridericia_ms"))
        qtc_b = _nullable_number(st, "QTc Bazett (ms)", f"qtcb_{page_index}", 1000, 1, default=existing_truth.get("qtc_bazett_ms"))
        axis = _signed_nullable_number(st, "Eje QRS (°)", f"axis_{page_index}", -180, 180, 1, default=existing_truth.get("axis_qrs_deg"))

    st_status = st.selectbox("Segmento ST", ST_OPTIONS, index=_option_index(ST_OPTIONS, existing_truth.get("st_status"), 0), key=f"st_{page_index}")
    t_status = st.selectbox("Onda T", T_OPTIONS, index=_option_index(T_OPTIONS, existing_truth.get("t_status"), 0), key=f"t_{page_index}")
    conduction = st.selectbox("Conducción", CONDUCTION, index=_option_index(CONDUCTION, existing_truth.get("conduction"), 0), key=f"cond_{page_index}")
    ectopy_options = ["Sin extrasístoles", "Extrasístoles supraventriculares", "Extrasístoles ventriculares", "Ectopia mixta", "No valorable"]
    ectopy = st.selectbox("Ectopia", ectopy_options, index=_option_index(ectopy_options, existing_truth.get("ectopy"), 0), key=f"ect_{page_index}")

    existing_dx = "\n".join(existing.get("diagnoses") or [])
    diagnoses_text = st.text_area("Diagnóstico(s) final(es) — uno por línea", value=existing_dx, key=f"dx_{page_index}", height=100)
    diagnoses = [x.strip() for x in diagnoses_text.splitlines() if x.strip()]

    st.markdown("## 4. Inclusión en entrenamiento")
    exclude = st.checkbox("Excluir este caso del entrenamiento", value=bool(existing.get("exclude_from_training", False)), key=f"exclude_{page_index}")
    exclusion_reason = st.text_input("Motivo de exclusión", value=existing.get("exclusion_reason") or "", key=f"exreason_{page_index}", disabled=not exclude)
    notes = st.text_area("Notas del auditor", value=existing.get("notes") or "", key=f"notes_{page_index}", height=80)

    quality = {
        "readable": readable,
        "folded_or_curved": folded,
        "shadow_or_glare": shadow,
        "cropped_or_incomplete": cropped,
    }
    calibration = {
        "speed_mm_s": speed if isinstance(speed, int) else None,
        "gain_mm_mv": gain if isinstance(gain, int) else None,
        "pulse_visible": pulse,
    }
    clinical_truth = {
        "rhythm": rhythm,
        "rr_pattern": rr_pattern,
        "p_status": p_status,
        "heart_rate_bpm": hr,
        "p_duration_ms": p_ms,
        "pr_ms": pr_ms,
        "qrs_ms": qrs_ms,
        "qt_ms": qt_ms,
        "qtc_fridericia_ms": qtc_f,
        "qtc_bazett_ms": qtc_b,
        "axis_qrs_deg": axis,
        "st_status": st_status,
        "t_status": t_status,
        "conduction": conduction,
        "ectopy": ectopy,
    }
    return build_annotation(
        raw_source=raw_source,
        filename=filename,
        page_number=page_index + 1,
        orientation_deg=orientation,
        layout=layout,
        rhythm_strip_lead=None if strip == "ninguna/no identificable" else strip,
        quality=quality,
        calibration=calibration,
        clinical_truth=clinical_truth,
        diagnoses=diagnoses,
        exclude_from_training=exclude,
        exclusion_reason=exclusion_reason,
        notes=notes,
    )
