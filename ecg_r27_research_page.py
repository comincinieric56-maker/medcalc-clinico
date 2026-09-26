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
from ecg_machine_header import (
    compose_final_report,
    extract_machine_measurements,
)
from ecg_layout_detector import detect_ecg_layout_source


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_machine_measurements(source_name: str, source_bytes: bytes, page_index: int):
    return extract_machine_measurements(source_name, source_bytes, page_index)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_layout_detection(source_name: str, source_bytes: bytes, page_index: int):
    return detect_ecg_layout_source(source_name, source_bytes, page_index)


def _render_layout_preflight(s, result: Dict[str, Any]) -> None:
    s.markdown("### Reconocimiento de formato ECG")

    layout = result.get("layout") or "UNKNOWN"
    confidence = float(result.get("confidence") or 0.0)
    route = str(result.get("route") or "UNKNOWN")

    c1, c2, c3, c4 = s.columns(4)
    c1.metric("Formato", str(layout))
    c2.metric(
        "Geometría",
        (
            f"{result.get('rows')} × {result.get('columns')}"
            if result.get("rows") and result.get("columns")
            else "—"
        ),
    )
    c3.metric("Confianza", f"{confidence * 100:.1f}%")
    c4.metric(
        "Regiones geométricas",
        f"{int(result.get('leads_detected') or 0)}/12",
    )

    if route == "6X2_ACTIVE_SPAN_CANONICALIZER":
        s.success(
            "Ruta seleccionada: 6×2 → 6X2_ACTIVE_SPAN_CANONICALIZER → "
            "U-Net de señal → 12 derivaciones canónicas."
        )
    elif route == "STANDARD_3X4_DIGITIZER":
        s.success(
            "Ruta seleccionada: 3×4 → Open ECG Digitizer → "
            "12 derivaciones canónicas."
        )
    elif route == "AMBIGUOUS_LAYOUT_RESOLVER":
        s.warning(
            "Formato parcialmente ambiguo. MEDCALC solicitará evidencia adicional "
            "del digitalizador antes de aceptar el mapeo."
        )
    else:
        s.warning(
            "Formato no resuelto con confianza suficiente. MEDCALC no forzará "
            "una geometría."
        )

    s.caption(
        "El detector principal usa geometría de las regiones y no OCR de I/II/III/V1…V6. "
        f"Rotación estimada de la cuadrícula: {float(result.get('rotation_deg') or 0.0):.1f}°."
    )

    with s.expander("Trazabilidad del detector de layout", expanded=False):
        s.json(result)


def _render_machine_measurements(s, machine: Dict[str, Any]) -> None:
    s.markdown("### Mediciones impresas por el equipo")

    if not machine.get("detected"):
        s.info(
            "No se detectó con suficiente confianza un bloque de mediciones impresas. "
            "MEDCALC continuará con la digitalización del trazado."
        )
        return

    c1, c2, c3, c4 = s.columns(4)
    c1.metric(
        "FC",
        f"{machine['heart_rate_bpm']} LPM"
        if machine.get("heart_rate_bpm") is not None else "—",
    )

    pr_printed = machine.get("pr_printed_ms")
    if pr_printed == 0:
        pr_value = "NO CALCULABLE"
    elif machine.get("pr_ms") is not None:
        pr_value = f"{machine['pr_ms']} ms"
    else:
        pr_value = "—"
    c2.metric("PR", pr_value)

    c3.metric(
        "QRS",
        f"{machine['qrs_ms']} ms"
        if machine.get("qrs_ms") is not None else "—",
    )

    if machine.get("qt_ms") is not None and machine.get("qtc_ms") is not None:
        qt_text = f"{machine['qt_ms']}/{machine['qtc_ms']} ms"
    else:
        qt_text = "—"
    c4.metric("QT/QTc", qt_text)

    a1, a2, a3 = s.columns(3)
    qrs_axis = machine.get("qrs_axis_deg")
    a1.metric(
        "Eje QRS",
        f"{qrs_axis}°" if qrs_axis is not None else "—",
    )
    a2.metric(
        "Eje T",
        f"{machine['t_axis_deg']}°"
        if machine.get("t_axis_deg") is not None else "—",
    )

    calibration = []
    if machine.get("speed_mm_per_s") is not None:
        calibration.append(f"{machine['speed_mm_per_s']:g} mm/s")
    if machine.get("gain_mm_per_mV") is not None:
        calibration.append(f"{machine['gain_mm_per_mV']:g} mm/mV")
    a3.metric("Calibración", " · ".join(calibration) if calibration else "—")

    if pr_printed == 0:
        s.caption(
            "PR impreso = 0 ms: MEDCALC lo interpreta como intervalo PR no calculado "
            "por el equipo, no como un PR fisiológico de 0 ms."
        )

    s.caption(
        "Estas cifras provienen del encabezado impreso del electrocardiógrafo y se "
        "mantienen separadas de las mediciones derivadas del trazado digitalizado."
    )

    with s.expander("OCR del encabezado", expanded=False):
        s.code(
            (machine.get("ocr_header_text") or "") + "\n" + (machine.get("ocr_footer_text") or ""),
            language=None,
        )


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
                "layout_detector": meta.get("layout_detector"),
                "signal": signal,
            }
        )


def _render_structured_report(
    s,
    meta: Dict[str, Any],
    machine: Dict[str, Any],
) -> None:
    structured = meta.get("structured_report") or {}
    final_report = compose_final_report(machine, structured)
    text = str(final_report.get("text") or "").strip()

    s.markdown("### Informe electrocardiográfico automatizado")

    if not text:
        s.warning("No fue posible generar el informe estructurado.")
        return

    s.code(text, language=None)

    s.download_button(
        "Descargar informe ECG (.txt)",
        data=text,
        file_name="medcalc_informe_ecg.txt",
        mime="text/plain",
        use_container_width=True,
        key="ecg_report_txt",
    )

    if machine.get("detected"):
        s.success(
            "Las mediciones de FC, PR, QRS, QT/QTc y ejes impresos por el "
            "electrocardiógrafo se incorporaron con trazabilidad explícita."
        )

    with s.expander("Mediciones que sustentan el informe", expanded=False):
        s.json(
            {
                "machine_printed_measurements": {
                    k: v
                    for k, v in machine.items()
                    if k not in {"ocr_header_text", "ocr_footer_text"}
                },
                "digitized_signal_report": {
                    "rhythm": structured.get("rhythm"),
                    "axis": structured.get("axis"),
                    "repolarization": structured.get("repolarization"),
                    "limitations": structured.get("limitations"),
                    "error": structured.get("error"),
                    "input_quality_gate": structured.get("input_quality_gate"),
                },
                "final_report": {
                    "machine_measurements_used": final_report.get("machine_measurements_used"),
                    "trusted_signal_report": final_report.get("trusted_signal_report"),
                },
            }
        )

    s.caption(
        "Los valores impresos por el equipo tienen prioridad como mediciones documentales. "
        "La morfología del trazado sólo se incorpora cuando la asignación de derivaciones "
        "supera el control de calidad del digitalizador. R27 sigue siendo probability-only."
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

    try:
        with s.spinner("Reconociendo geometría del ECG…"):
            layout_preflight = _cached_layout_detection(
                uploaded.name,
                uploaded.getvalue(),
                int(page_index),
            )
    except Exception as exc:
        layout_preflight = {
            "layout": None,
            "confidence": 0.0,
            "route": "UNKNOWN",
            "error": str(exc),
        }

    _render_layout_preflight(s, layout_preflight)

    try:
        with s.spinner("Leyendo mediciones impresas del electrocardiógrafo…"):
            machine_measurements = _cached_machine_measurements(
                uploaded.name,
                uploaded.getvalue(),
                int(page_index),
            )
    except Exception as exc:
        machine_measurements = {
            "detected": False,
            "source": "machine_printed_header_ocr",
            "error": str(exc),
        }

    _render_machine_measurements(s, machine_measurements)

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
    _render_structured_report(s, meta, machine_measurements)

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
