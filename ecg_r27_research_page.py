from __future__ import annotations

import io
from typing import Any

import streamlit as st


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _load_photo_engine():
    # Lazy import: OpenCV/Pillow/PyMuPDF are loaded only when the ECG page is used.
    from ecg_photo_engine import (
        assess_ecg_photo,
        detect_calibration_pulse,
        digitize_standard_12lead_preview,
        enhanced_preview,
        pdf_page_count,
        prepare_ecg_image,
        rectify_ecg_photo,
        render_ecg_pdf_page,
    )
    return {
        "assess": assess_ecg_photo,
        "calibration": detect_calibration_pulse,
        "digitize_preview": digitize_standard_12lead_preview,
        "enhanced": enhanced_preview,
        "pdf_page_count": pdf_page_count,
        "prepare": prepare_ecg_image,
        "rectify": rectify_ecg_photo,
        "render_pdf": render_ecg_pdf_page,
    }


def _source_to_image_bytes(s, uploaded, engine):
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()

    if name.endswith(".pdf"):
        pages = int(engine["pdf_page_count"](raw))
        page_index = 0
        if pages > 1:
            page_number = s.number_input(
                "Página del PDF que contiene el ECG",
                min_value=1,
                max_value=pages,
                value=1,
                step=1,
                key="r27_photo_pdf_page",
            )
            page_index = int(page_number) - 1

        image_bytes, meta = engine["render_pdf"](
            raw,
            page_index=page_index,
            dpi=300,
            max_dimension=4200,
        )
        return image_bytes, {
            "source_type": "pdf",
            "filename": uploaded.name,
            **meta,
        }

    image_bytes, _, meta = engine["prepare"](
        raw,
        crop_header=False,
        max_dimension=4200,
    )
    return image_bytes, {
        "source_type": "image",
        "filename": uploaded.name,
        **meta,
    }


def _quality_summary(s, quality: dict, calibration: dict, rect_meta: dict) -> None:
    c1, c2, c3, c4 = s.columns(4)
    c1.metric("Calidad", str(quality.get("quality_label") or "—"))
    c2.metric("Puntaje", str(quality.get("quality_score") or "—"))
    c3.metric(
        "Cuadrícula",
        f"{float(quality.get('grid_confidence') or 0.0) * 100:.0f}%",
    )
    c4.metric(
        "Rectificación",
        "SÍ" if rect_meta.get("rectified") else "NO",
    )

    if calibration.get("speed_mm_s"):
        s.success(
            f"Calibración detectada: {calibration.get('speed_mm_s')} mm/s · "
            f"{calibration.get('gain_mm_mV')} mm/mV."
        )
    else:
        s.warning(
            "La velocidad/ganancia no quedó demostrada automáticamente. "
            "No se convertirán píxeles a tiempo/voltaje por suposición."
        )

    issues = quality.get("issues") or []
    if issues:
        with s.expander("Observaciones de calidad", expanded=False):
            for issue in issues:
                s.write("• " + str(issue))


def _render_photo_pdf_pipeline(s) -> None:
    s.markdown("### Cargar ECG")
    s.caption(
        "Entrada de usuario final: fotografía o PDF del electrocardiograma. "
        "No se solicita ECG digital/WFDB."
    )

    uploaded = s.file_uploader(
        "Foto o PDF del ECG",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        key="r27_photo_pdf_upload",
        help=(
            "Preferir imagen nítida, perpendicular al papel y con cuadrícula/calibración visibles. "
            "En PDF, seleccione la página que contiene el trazado."
        ),
    )

    if uploaded is None:
        s.info("Suba una fotografía o PDF para iniciar el procesamiento.")
        return

    try:
        engine = _load_photo_engine()
        source_bytes, source_meta = _source_to_image_bytes(s, uploaded, engine)
    except Exception as exc:
        s.error(f"No fue posible abrir el archivo: {exc}")
        return

    s.image(
        source_bytes,
        caption=f"Entrada procesada · {source_meta.get('filename')}",
        use_container_width=True,
    )

    if not s.button(
        "Procesar ECG",
        type="primary",
        use_container_width=True,
        key="r27_photo_process",
    ):
        return

    with s.spinner("Rectificando papel, evaluando cuadrícula y siguiendo el trazado…"):
        try:
            rectified_bytes, rect_meta = engine["rectify"](source_bytes)
            quality = engine["assess"](rectified_bytes)
            calibration = engine["calibration"](rectified_bytes, quality)
            preview = engine["digitize_preview"](rectified_bytes, quality)
            enhanced = engine["enhanced"](rectified_bytes)
        except Exception as exc:
            s.error(f"Falló el procesamiento del ECG: {exc}")
            return

    s.markdown("### Control de calidad de la digitalización")
    _quality_summary(s, quality, calibration, rect_meta)

    p1, p2 = s.columns(2)
    with p1:
        s.image(
            enhanced,
            caption="ECG rectificado / realzado",
            use_container_width=True,
        )
    with p2:
        s.image(
            preview["overlay_bytes"],
            caption="Seguimiento candidato del formato 12 derivaciones",
            use_container_width=True,
        )

    s.image(
        preview["reconstruction_bytes"],
        caption="Reconstrucción preliminar del trazado · control visual",
        use_container_width=True,
    )

    layout_conf = float(preview.get("layout_confidence") or 0.0)
    if quality.get("digitization_allowed") and layout_conf >= 0.55:
        s.success(
            f"Preprocesamiento aceptable para continuar al digitalizador neuronal. "
            f"Seguimiento de tinta: {layout_conf * 100:.0f}%."
        )
    else:
        s.error(
            "La imagen no alcanza todavía los criterios mínimos para una reconstrucción "
            "segura de señal. No se enviará a R27."
        )
        return

    s.divider()
    s.markdown("### Paso siguiente · U-Net → señal → R27")

    s.warning(
        "La interfaz foto/PDF ya está montada, pero el puente neuronal definitivo todavía "
        "no está habilitado. El código actual sólo hace rectificación, control de calidad y "
        "seguimiento preliminar de tinta. No se debe convertir esta reconstrucción clásica "
        "directamente en la entrada de R27."
    )

    s.code(
        "FOTO/PDF\n"
        "  ↓\n"
        "Rectificación + QC\n"
        "  ↓\n"
        "U-Net: segmentación del trazado\n"
        "  ↓\n"
        "calibración mm→mV y mm→s\n"
        "  ↓\n"
        "reconstrucción 12 derivaciones\n"
        "  ↓\n"
        "señal 500 Hz + señal 100 Hz\n"
        "  ↓\n"
        "controles de integridad\n"
        "  ↓\n"
        "R27\n"
        "  ↓\n"
        "35 probabilidades",
        language="text",
    )

    s.caption(
        "El análisis R27 desde foto/PDF se habilitará sólo cuando el adaptador U-Net "
        "demuestre paridad contra ECG digitales originales y falle de forma cerrada "
        "cuando una derivación, escala o calibración no sea recuperable."
    )


def page_ecg_r27_research(st_module=None):
    s = st_module or st

    s.markdown(
        """
        <div style="
            border:1px solid #d6e3e8;
            border-radius:16px;
            padding:1rem 1.1rem;
            background:#ffffff;
            margin-bottom:1rem;">
          <div style="font-size:.78rem;font-weight:700;letter-spacing:.08em;color:#667788">
            MEDCALC ECG · FOTO/PDF → R27
          </div>
          <div style="font-size:1.65rem;font-weight:750;color:#12202f;margin-top:.15rem">
            ❤️ Electrocardiograma
          </div>
          <div style="color:#667788;margin-top:.25rem">
            Entrada clínica prevista: fotografía o PDF del ECG.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s.error(
        "**MODO INVESTIGACIÓN. NO USAR COMO DIAGNÓSTICO CLÍNICO.** "
        "R27 conserva salida probability-only; el digitalizador foto/PDF todavía debe "
        "validarse antes de conectarlo al motor."
    )

    _render_photo_pdf_pipeline(s)
