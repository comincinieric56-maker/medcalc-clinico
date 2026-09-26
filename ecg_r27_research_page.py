from __future__ import annotations

import io
import json
from typing import Any, Dict

import streamlit as st
from PIL import Image

from ecg_unet_r27_bridge import (
    DIGITISER_COMMIT,
    MODEL_SHA256,
    ECGDigitiserError,
    digitize_photo_pdf_and_run_r27,
)
from r27_local_runtime import (
    ALL35,
    EXPECTED_CLINICAL_STATUS,
    EXPECTED_RELEASE_TYPE,
)


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _render_pdf_preview(pdf_bytes: bytes, page_index: int) -> tuple[bytes, int]:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if getattr(doc, "needs_pass", False):
            raise RuntimeError("El PDF está protegido con contraseña.")
        if doc.page_count < 1:
            raise RuntimeError("El PDF no contiene páginas.")
        if not 0 <= int(page_index) < int(doc.page_count):
            raise RuntimeError("Página PDF fuera de rango.")
        page = doc.load_page(int(page_index))
        pix = page.get_pixmap(matrix=fitz.Matrix(180 / 72.0, 180 / 72.0), alpha=False)
        return pix.tobytes("png"), int(doc.page_count)
    finally:
        doc.close()


def _validate_probability_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ECGDigitiserError("Respuesta R27 inválida.")
    if payload.get("release_type") != EXPECTED_RELEASE_TYPE:
        raise ECGDigitiserError("release_type R27 inesperado.")
    if payload.get("clinical_deployment_status") != EXPECTED_CLINICAL_STATUS:
        raise ECGDigitiserError("clinical_deployment_status R27 inesperado.")

    modules = payload.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(ALL35):
        raise ECGDigitiserError("La salida R27 no contiene exactamente los 35 módulos.")

    for module in ALL35:
        item = modules[module]
        p = float(item["probability"])
        if not 0.0 <= p <= 1.0:
            raise ECGDigitiserError(f"{module}: probability fuera de [0,1].")
        for forbidden in ("threshold", "binary_classification", "diagnostic_label"):
            if item.get(forbidden) is not None:
                raise ECGDigitiserError(f"{module}: {forbidden} no autorizado.")
        if item.get("clinical_diagnostic_claim_allowed") is not False:
            raise ECGDigitiserError(f"{module}: claim clínico no autorizado.")

    return payload


def _render_digitizer_metadata(s, meta: dict) -> None:
    signal = meta.get("signal") or {}
    image = meta.get("image") or {}

    c1, c2, c3, c4 = s.columns(4)
    c1.metric("Digitalización", str(meta.get("status") or "—"))
    c2.metric("Señal 500 Hz", "5000×12" if signal.get("shape_500") == [5000, 12] else "—")
    c3.metric("Señal 100 Hz", "1000×12" if signal.get("shape_100") == [1000, 12] else "—")
    c4.metric(
        "Cobertura mínima",
        f"{float(signal.get('min_observed_fraction') or 0.0) * 100:.1f}%",
    )

    with s.expander("Control de integridad de la digitalización", expanded=False):
        s.write(f"**Fuente:** {image.get('source_type') or '—'}")
        s.write(f"**Derivaciones:** {', '.join(signal.get('sig_names') or []) or '—'}")
        s.write(f"**Fracción finita:** {signal.get('finite_fraction', '—')}")
        s.write(
            "**Compatible con el contrato temporal R27:** "
            + ("SÍ" if signal.get("r27_input_compatible") else "NO")
        )

        coverage = signal.get("observed_fraction_by_lead") or {}
        if coverage:
            rows = [
                {"Derivación": lead, "Cobertura observada": float(value)}
                for lead, value in coverage.items()
            ]
            s.dataframe(
                rows,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Cobertura observada": s.column_config.ProgressColumn(
                        "Cobertura observada",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.3f",
                    )
                },
            )
        s.caption(signal.get("resampling_note") or "")


def _render_probabilities(s, payload: Dict[str, Any]) -> None:
    payload = _validate_probability_payload(payload)
    rows = []
    for module in ALL35:
        item = payload["modules"][module]
        rows.append(
            {
                "Módulo": module,
                "Probabilidad": float(item["probability"]),
                "Threshold": "NO DISPONIBLE",
                "Clasificación binaria": "NO AUTORIZADA",
            }
        )

    rows.sort(key=lambda x: x["Probabilidad"], reverse=True)

    s.success("R27 completado. Se muestran únicamente probabilidades.")
    s.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Probabilidad": s.column_config.ProgressColumn(
                "Probabilidad",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
            )
        },
    )
    s.warning(
        "Una probabilidad alta no equivale a un diagnóstico positivo. "
        "R27 permanece en release de investigación probability-only."
    )
    s.download_button(
        "Descargar resultado R27 (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name="medcalc_ecg_r27_probability_only.json",
        mime="application/json",
        use_container_width=True,
        key="r27_photo_result_download",
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
            MEDCALC ECG · FOTO/PDF → U-NET → R27
          </div>
          <div style="font-size:1.65rem;font-weight:750;color:#12202f;margin-top:.15rem">
            ❤️ Electrocardiograma
          </div>
          <div style="color:#667788;margin-top:.25rem">
            Digitalización neuronal del trazado y paso condicionado al motor R27.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s.error(
        "**MODO INVESTIGACIÓN. NO USAR COMO DIAGNÓSTICO CLÍNICO.** "
        "El digitalizador de imagen y R27 requieren validación externa conjunta. "
        "R27 conserva salida exclusivamente probability-only."
    )

    github_token = _get_secret("R27_GITHUB_TOKEN")
    if not github_token:
        s.error(
            "Falta R27_GITHUB_TOKEN en Streamlit Secrets. "
            "Debe tener Contents: Read-only sobre medcalc-r27-backend."
        )
        return

    uploaded = s.file_uploader(
        "Foto o PDF del ECG",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        key="r27_photo_pdf_upload",
    )

    if uploaded is None:
        s.info(
            "Suba una fotografía o PDF del electrocardiograma. "
            "No se solicitan archivos ECG digitales al usuario."
        )
        with s.expander("Motor de digitalización", expanded=False):
            s.caption(
                f"PhysioNet Challenge 2024 winner · nnU-Net M3 · commit {DIGITISER_COMMIT}"
            )
            s.caption(f"Checkpoint SHA-256: {MODEL_SHA256}")
        return

    raw = uploaded.getvalue()
    ext = uploaded.name.lower().rsplit(".", 1)[-1] if "." in uploaded.name else ""
    pdf_page_index = 0

    try:
        if ext == "pdf":
            preview0, page_count = _render_pdf_preview(raw, 0)
            if page_count > 1:
                page_number = s.number_input(
                    "Página del PDF que contiene el ECG",
                    min_value=1,
                    max_value=page_count,
                    value=1,
                    step=1,
                    key="r27_pdf_page",
                )
                pdf_page_index = int(page_number) - 1
                preview, _ = _render_pdf_preview(raw, pdf_page_index)
            else:
                preview = preview0
            s.image(
                preview,
                caption=f"PDF · página {pdf_page_index + 1}",
                use_container_width=True,
            )
        else:
            img = Image.open(io.BytesIO(raw))
            s.image(img, caption=uploaded.name, use_container_width=True)
    except Exception as exc:
        s.error(f"No fue posible previsualizar el archivo: {exc}")
        return

    c_age, c_sex = s.columns(2)
    with c_age:
        age = s.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=1.0,
            key="r27_photo_age",
        )
    with c_sex:
        sex = s.selectbox(
            "Sexo codificado para runtime",
            options=["0", "1"],
            key="r27_photo_sex",
            help="Se mantiene la codificación exacta requerida por el runtime R27 congelado.",
        )

    with s.expander("Detalles técnicos del digitalizador", expanded=False):
        s.write("**Modelo:** ECG-Digitiser M3 · nnU-Net 2D")
        s.write(f"**Commit fijado:** {DIGITISER_COMMIT}")
        s.write(f"**Checkpoint SHA-256:** {MODEL_SHA256}")
        s.caption(
            "La primera ejecución de una instancia nueva descarga y verifica un checkpoint "
            "de aproximadamente 475 MB. El proceso U-Net se ejecuta en un subprocess y se "
            "destruye antes de cargar R27 para reducir el pico de memoria."
        )

    if not s.button(
        "Digitalizar y analizar",
        type="primary",
        use_container_width=True,
        key="r27_photo_run",
    ):
        return

    with s.spinner(
        "Digitalizando ECG con U-Net. En el primer uso puede tardar por la descarga "
        "del checkpoint y la inferencia CPU…"
    ):
        try:
            result = digitize_photo_pdf_and_run_r27(
                str(github_token),
                source_name=uploaded.name,
                source_bytes=raw,
                age=float(age),
                sex=str(sex),
                pdf_page_index=int(pdf_page_index),
                timeout_seconds=1800,
            )
        except Exception as exc:
            s.error(str(exc))
            return

    meta = result.get("digitizer") or {}
    _render_digitizer_metadata(s, meta)

    payload = result.get("payload")
    if payload is None:
        s.warning(
            "El ECG fue digitalizado, pero R27 no se ejecutó. "
            + str(meta.get("reason") or "")
        )
        s.info(
            "En el formato impreso 3×4 habitual la página muestra aproximadamente "
            "2,5 s de la mayoría de las derivaciones, mientras R27 fue congelado para "
            "10 s completos de las 12 derivaciones. MEDCALC no rellena, repite ni "
            "inventa los segmentos que no existen en el papel."
        )
    else:
        _render_probabilities(s, payload)

    s.download_button(
        "Descargar auditoría de digitalización (JSON)",
        data=json.dumps(meta, indent=2, ensure_ascii=False),
        file_name="medcalc_ecg_digitization_audit.json",
        mime="application/json",
        use_container_width=True,
        key="r27_digitizer_audit_download",
    )
