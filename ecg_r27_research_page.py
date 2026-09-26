from __future__ import annotations

import io
import json
from typing import Any, Dict

import streamlit as st

from ecg_unet_r27_bridge import (
    ECGDigitiserError,
    digitiser_status,
    digitize_photo_pdf_and_run_r27,
)
from r27_local_runtime import ALL35


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _render_preview(s, uploaded, page_index: int) -> None:
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()

    try:
        if name.endswith(".pdf"):
            import fitz
            from PIL import Image

            doc = fitz.open(stream=raw, filetype="pdf")
            try:
                if doc.page_count < 1:
                    return
                idx = min(max(int(page_index), 0), int(doc.page_count) - 1)
                page = doc.load_page(idx)
                pix = page.get_pixmap(matrix=fitz.Matrix(160 / 72, 160 / 72), alpha=False)
                image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            finally:
                doc.close()
            s.image(image, caption=f"Vista previa · página {idx + 1}", use_container_width=True)
        else:
            s.image(raw, caption="Vista previa", use_container_width=True)
    except Exception:
        pass


def _render_digitizer_meta(s, meta: Dict[str, Any]) -> None:
    signal = meta.get("signal") or {}
    observed = signal.get("observed_fraction_by_lead") or {}

    s.markdown("### Digitalización")
    c1, c2, c3 = s.columns(3)
    c1.metric("Estado", str(meta.get("status") or "—"))
    c2.metric("Layout", str(signal.get("layout_name") or "—"))
    c3.metric(
        "Cobertura mínima",
        f"{float(signal.get('min_observed_fraction') or 0.0) * 100:.1f}%",
    )

    if observed:
        rows = [
            {
                "Derivación": lead,
                "Cobertura observada": float(observed.get(lead) or 0.0),
            }
            for lead in [
                "I","II","III","aVR","aVL","aVF",
                "V1","V2","V3","V4","V5","V6",
            ]
        ]
        s.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cobertura observada": s.column_config.ProgressColumn(
                    "Cobertura observada",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                )
            },
        )

    with s.expander("Trazabilidad del digitalizador", expanded=False):
        s.json(
            {
                "digitizer": meta.get("digitizer"),
                "digitizer_commit": meta.get("digitizer_commit"),
                "license": meta.get("license"),
                "segmentation_model_sha256": meta.get("segmentation_model_sha256"),
                "lead_model_sha256": meta.get("lead_model_sha256"),
                "reason": meta.get("reason"),
                "signal": signal,
            }
        )


def _render_probability_table(s, payload: Dict[str, Any]) -> None:
    modules = payload.get("modules") or {}
    if set(modules) != set(ALL35):
        s.error("La salida R27 no contiene exactamente los 35 módulos esperados.")
        return

    rows = []
    for module in ALL35:
        item = modules[module]
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
        use_container_width=True,
        hide_index=True,
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
        "Una probabilidad alta no equivale a diagnóstico positivo. "
        "R27 permanece probability-only y sin thresholds desplegables."
    )
    s.download_button(
        "Descargar resultado R27 (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name="medcalc_r27_probability_only_result.json",
        mime="application/json",
        use_container_width=True,
        key="r27_download",
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
            Entrada del usuario: fotografía o PDF del ECG.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s.error(
        "**MODO INVESTIGACIÓN. NO USAR COMO DIAGNÓSTICO CLÍNICO.** "
        "El digitalizador U-Net y el adaptador foto/PDF→R27 deben validarse "
        "antes de cualquier uso clínico."
    )

    github_token = _get_secret("R27_GITHUB_TOKEN")

    if not github_token:
        s.warning(
            "Falta R27_GITHUB_TOKEN en Streamlit Secrets. "
            "La digitalización U-Net puede verificarse localmente, pero R27 no podrá "
            "materializar su runtime privado hasta configurar ese secreto."
        )

    uploaded = s.file_uploader(
        "Foto o PDF del ECG",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        key="r27_photo_pdf_upload",
        help=(
            "Preferir toma perpendicular, nítida, con cuadrícula y calibración visibles. "
            "El sistema no solicita archivos WFDB al usuario."
        ),
    )
    if uploaded is None:
        s.info("Suba una fotografía o PDF para comenzar.")
        return

    page_index = 0
    if (uploaded.name or "").lower().endswith(".pdf"):
        try:
            import fitz

            doc = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
            try:
                page_count = int(doc.page_count)
            finally:
                doc.close()
        except Exception as exc:
            s.error(f"No fue posible abrir el PDF: {exc}")
            return

        if page_count < 1:
            s.error("El PDF no contiene páginas.")
            return

        page_number = s.number_input(
            "Página del PDF que contiene el ECG",
            min_value=1,
            max_value=page_count,
            value=1,
            step=1,
            key="r27_pdf_page_number",
        )
        page_index = int(page_number) - 1

    _render_preview(s, uploaded, page_index)

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
            help="Se conserva la codificación requerida por el runtime R27 congelado.",
        )

    with s.expander("Estado técnico del digitalizador", expanded=False):
        try:
            status = digitiser_status()
        except Exception as exc:
            s.error(f"Digitalizador no disponible: {exc}")
        else:
            s.success(
                "U-Net cargado en el repositorio · "
                f"{status['segmentation_model_size'] / 1024 / 1024:.1f} MB + "
                f"{status['lead_model_size'] / 1024 / 1024:.1f} MB"
            )
            s.caption(
                f"Fuente: {status['source_repository']} · commit {status['source_commit'][:12]} · "
                f"licencia {status['license']}"
            )

    if not s.button(
        "Digitalizar y analizar",
        type="primary",
        use_container_width=True,
        key="r27_photo_run",
    ):
        return

    if not github_token:
        s.error(
            "Configure primero R27_GITHUB_TOKEN en Streamlit Secrets. "
            "No se ejecutará R27 sin su runtime congelado."
        )
        return

    with s.spinner(
        "Ejecutando U-Net sobre la foto/PDF. El proceso neuronal termina antes de "
        "iniciar R27 para no mantener ambos modelos en memoria al mismo tiempo…"
    ):
        try:
            result = digitize_photo_pdf_and_run_r27(
                str(github_token),
                source_name=uploaded.name,
                source_bytes=uploaded.getvalue(),
                age=float(age),
                sex=str(sex),
                pdf_page_index=int(page_index),
                timeout_seconds=1800,
            )
        except ECGDigitiserError as exc:
            s.error(str(exc))
            return
        except Exception as exc:
            s.error(f"Fallo no esperado en foto/PDF→U-Net→R27: {exc}")
            return

    meta = result.get("digitizer") or {}
    _render_digitizer_meta(s, meta)

    payload = result.get("payload")

    if payload is None:
        s.warning(
            "El ECG fue digitalizado, pero R27 no se ejecutó. "
            "MEDCALC sólo entrega a R27 señales con 10 s completos y finitos en las 12 derivaciones."
        )
        s.info(
            str(meta.get("reason") or "")
            or (
                "En impresos 3×4 convencionales suelen existir aproximadamente 2.5 s "
                "observados por derivación; MEDCALC no repite ni inventa los segmentos faltantes."
            )
        )
        return

    _render_probability_table(s, payload)
