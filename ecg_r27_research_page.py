from __future__ import annotations

import base64
import io
import json
from typing import Any, Dict

import streamlit as st

from ecg_unet_r27_bridge import (
    ECGDigitiserError,
    digitize_photo_pdf_github_actions,
    remote_digitizer_status,
)
from r27_local_runtime import ALL35
from ecg_machine_header import (
    compose_final_report,
    extract_machine_measurements,
)
from ecg_layout_detector import detect_ecg_layout_source
from ecg_report_pdf import build_ecg_report_pdf


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
    s.markdown("### Preanálisis geométrico del ECG")

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
        s.info(
            "El preanálisis geométrico no resolvió el formato. Esto no bloquea "
            "el análisis: el layout definitivo se decide después de la "
            "segmentación U-Net mediante hipótesis 3×4/6×2 y control de calidad."
        )

    s.caption(
        "Este preanálisis es orientativo y no decide el layout final. "
        "La decisión definitiva se toma en el runner después de U-Net, usando "
        "la señal segmentada y scoring de hipótesis. "
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

    calibration = signal.get("calibration") or {}
    s.markdown("#### Señal digital canónica")
    q1, q2, q3, q4 = s.columns(4)
    q1.metric(
        "Fuente clínica",
        str(signal.get("digital_signal_schema") or "NO DISPONIBLE"),
    )
    q2.metric(
        "Velocidad",
        (
            f"{float(calibration['speed_mm_s']):g} mm/s"
            if calibration.get("speed_mm_s") is not None else "—"
        ),
    )
    q3.metric(
        "Ganancia",
        (
            f"{float(calibration['gain_mm_mV']):g} mm/mV"
            if calibration.get("gain_mm_mV") is not None else "—"
        ),
    )
    q4.metric(
        "Confianza escala",
        f"{float(calibration.get('confidence') or 0.0):.2f}",
    )
    if calibration.get("quantitative_scale_verified"):
        s.success(
            "La escala física fue validada. El analizador clínico consume "
            "arrays digitales en mV/ms a frecuencia de muestreo conocida."
        )
    else:
        s.warning(
            "La escala física no alcanzó confianza suficiente. Los valores "
            "cuantitativos deben fallar cerrados antes de publicarse."
        )

    with s.expander("Trazabilidad del digitalizador", expanded=False):
        signal_audit = {
            k: v for k, v in signal.items()
            if k not in {"digital_ecg", "audit_assets"}
        }
        digital_ecg = signal.get("digital_ecg") or {}
        s.json(
            {
                "digitizer": meta.get("digitizer"),
                "digitizer_commit": meta.get("digitizer_commit"),
                "license": meta.get("license"),
                "segmentation_model_sha256": meta.get("segmentation_model_sha256"),
                "lead_model_sha256": meta.get("lead_model_sha256"),
                "reason": meta.get("reason"),
                "layout_detector": meta.get("layout_detector"),
                "signal": signal_audit,
                "digital_ecg_summary": {
                    "schema": digital_ecg.get("schema"),
                    "source": digital_ecg.get("source"),
                    "fs": digital_ecg.get("fs"),
                    "units": digital_ecg.get("units"),
                    "source_layout": digital_ecg.get("source_layout"),
                    "layout_used_only_for_reconstruction": digital_ecg.get(
                        "layout_used_only_for_reconstruction"
                    ),
                    "recovered_lead_count": digital_ecg.get(
                        "recovered_lead_count"
                    ),
                    "global_confidence": digital_ecg.get("global_confidence"),
                },
            }
        )


def _render_digital_audit(s, meta: Dict[str, Any]) -> None:
    signal = meta.get("signal") or {}
    audit = signal.get("audit_assets") or {}
    reconstruction = str(audit.get("reconstruction_png_data_uri") or "")
    overlay = str(audit.get("segmentation_overlay_png_data_uri") or "")
    if not reconstruction and not overlay:
        return

    def decode(uri: str):
        marker = "data:image/png;base64,"
        if not uri.startswith(marker):
            return None
        try:
            return base64.b64decode(uri[len(marker):])
        except Exception:
            return None

    s.markdown("### Auditoría visual de la reconstrucción")
    s.caption(
        "Estas imágenes son para comprobar el trabajo del digitizer. "
        "MEDCALC no vuelve a medir sobre ellas."
    )
    a, b = s.columns(2)
    with a:
        data = decode(reconstruction)
        if data:
            s.image(data, caption="ECG reconstruido desde arrays digitales")
    with b:
        data = decode(overlay)
        if data:
            s.image(
                data,
                caption=(
                    "Centerline sobre espacio rectificado de segmentación U-Net"
                ),
            )
    s.caption(
        "El overlay actual está en coordenadas rectificadas del U-Net; permite "
        "auditar seguimiento de línea, ROI, fila y pérdida de señal. No se "
        "presenta como superposición geométrica exacta sobre el raster original."
    )


def _render_motor_measurements(
    s,
    meta: Dict[str, Any],
    machine: Dict[str, Any],
) -> None:
    structured = meta.get("structured_report") or {}
    motor = structured.get("measurement_summary") or {}
    rhythm = structured.get("rhythm") or {}
    screen = structured.get("rhythm_screen") or {}

    s.markdown("### Lectura del motor MEDCALC")
    s.caption(
        "Arquitectura V2: FC, RR, intervalos, ST y morfología se calculan "
        "principalmente sobre la señal ECG digital canónica reconstruida. "
        "El layout y la imagen original ya no son la interfaz del analizador clínico."
    )

    if not motor:
        s.info("El motor de medición no produjo valores utilizables.")
        return

    def _fmt(value, suffix="", digits=0, unavailable="NO EVALUABLE"):
        try:
            v = float(value)
            if digits == 0:
                return f"{v:.0f}{suffix}"
            return f"{v:.{digits}f}{suffix}"
        except Exception:
            return unavailable

    # Section 1 — clinically readable measurement cards.
    s.markdown("#### Mediciones principales")
    confidence = motor.get("confidence") or {}
    c1, c2, c3 = s.columns(3)
    c1.metric(
        "Frecuencia cardiaca",
        _fmt(motor.get("heart_rate_bpm"), " LPM"),
        help=f"Confianza: {_fmt(confidence.get('heart_rate'), '', 2)}",
    )
    c2.metric(
        "QRS",
        _fmt(motor.get("qrs_ms"), " ms"),
        help=f"Confianza: {_fmt(confidence.get('qrs'), '', 2)}",
    )
    qtm = motor.get("qt_ms")
    qtcm = motor.get("qtc_bazett_ms")
    qtt = (
        f"{float(qtm):.0f}/{float(qtcm):.0f} ms"
        if qtm is not None and qtcm is not None
        else "NO EVALUABLE"
    )
    c3.metric("QT / QTc", qtt)

    d1, d2, d3 = s.columns(3)
    d1.metric(
        "PR",
        _fmt(motor.get("pr_ms"), " ms"),
        help=f"Confianza: {_fmt(confidence.get('pr'), '', 2)}",
    )
    d2.metric(
        "Eje QRS",
        _fmt(motor.get("axis_deg"), "°"),
        help=f"Confianza: {_fmt(confidence.get('axis'), '', 2)}",
    )
    d3.metric(
        "Cobertura mínima",
        f"{float((meta.get('signal') or {}).get('min_observed_fraction') or 0.0) * 100:.1f}%",
    )

    # Section 2 — rhythm as a separate QC block.
    s.markdown("#### Ritmo y calidad temporal")
    rhythm_label = str(screen.get("label") or "RITMO NO EVALUABLE")
    if screen.get("evaluable"):
        s.success(f"**{rhythm_label}**")
        r1, r2, r3 = s.columns(3)
        r1.metric("RR CV", _fmt(motor.get("rr_cv"), "", 3))
        r2.metric("QRS detectados", _fmt(motor.get("beat_n")))
        r3.metric("P/QRS", _fmt(motor.get("p_before_qrs_ratio"), "", 2))
    else:
        reason_code = str(rhythm.get("reason") or "UNSPECIFIED")
        reason_text = {
            "INDEPENDENT_TEMPORAL_REFERENCE_INSUFFICIENT": (
                "La tira temporal independiente no alcanzó la calidad mínima para "
                "clasificar regularidad. MEDCALC conserva las mediciones "
                "morfológicas, pero no fuerza una conclusión de ritmo."
            ),
            "INDEPENDENT_TEMPORAL_REFERENCE_REQUIRED": (
                "No se dispone de una tira temporal independiente suficientemente "
                "confiable para clasificar el ritmo."
            ),
        }.get(
            reason_code,
            "La señal temporal no superó el control de calidad para clasificar el ritmo.",
        )
        s.warning(f"**Ritmo no evaluable.** {reason_text}")
        r1, r2, r3 = s.columns(3)
        r1.metric("RR CV", "NO EVALUABLE")
        r2.metric("QRS para ritmo", "NO EVALUABLE")
        r3.metric("P/QRS temporal", "NO EVALUABLE")
        with s.expander("Detalle técnico del control de calidad", expanded=False):
            s.code(reason_code, language=None)
            failures = rhythm.get("candidate_failures") or []
            if failures:
                s.json({"candidate_failures": failures})

    # Section 3 — comparison is useful audit information, but not the visual
    # centerpiece. Keep it collapsed instead of leaving a large raw table in the page.
    comparisons = [
        ("FC", machine.get("heart_rate_bpm"), motor.get("heart_rate_bpm"), "LPM", 10.0),
        (
            "PR",
            None if machine.get("pr_printed_ms") == 0 else machine.get("pr_ms"),
            motor.get("pr_ms"),
            "ms",
            30.0,
        ),
        ("QRS", machine.get("qrs_ms"), motor.get("qrs_ms"), "ms", 20.0),
        ("Eje QRS", machine.get("qrs_axis_deg"), motor.get("axis_deg"), "°", 20.0),
        ("QT", machine.get("qt_ms"), motor.get("qt_ms"), "ms", 40.0),
        ("QTc", machine.get("qtc_ms"), motor.get("qtc_bazett_ms"), "ms", 40.0),
    ]

    rows = []
    for name, printed, measured, unit, tolerance in comparisons:
        try:
            p = float(printed) if printed is not None else None
        except Exception:
            p = None
        try:
            m = float(measured) if measured is not None else None
        except Exception:
            m = None

        delta = abs(p - m) if p is not None and m is not None else None
        status = (
            "CONCORDANTE"
            if delta is not None and delta <= tolerance
            else "DISCORDANTE"
            if delta is not None
            else "NO COMPARABLE"
        )
        rows.append(
            {
                "Medición": name,
                "Equipo": f"{p:.0f} {unit}" if p is not None else "—",
                "MEDCALC": f"{m:.0f} {unit}" if m is not None else "—",
                "Diferencia": f"{delta:.0f} {unit}" if delta is not None else "—",
                "Estado": status,
            }
        )

    with s.expander("Comparar con mediciones impresas del electrocardiógrafo", expanded=False):
        s.dataframe(rows, width="stretch", hide_index=True)
        s.caption(
            "Esta comparación es una auditoría. El encabezado impreso no sustituye "
            "las mediciones calculadas sobre la señal digitalizada."
        )



def _render_structured_report(
    s,
    meta: Dict[str, Any],
    machine: Dict[str, Any],
    payload: Dict[str, Any] | None = None,
    *,
    r27_error: str | None = None,
    source_name: str | None = None,
    source_bytes: bytes | None = None,
    pdf_page_index: int = 0,
    age: float | None = None,
    sex_code: str | None = None,
) -> None:
    structured = meta.get("structured_report") or {}
    final_report = compose_final_report(machine, structured, payload)
    text = str(final_report.get("text") or "").strip()

    s.markdown("### Informe electrocardiográfico automatizado")

    if not text:
        s.warning("No fue posible generar el informe estructurado.")
        return

    s.code(text, language=None)

    try:
        pdf_bytes = build_ecg_report_pdf(
            final_report,
            machine=machine,
            structured_report=structured,
            digitizer=meta,
            r27_payload=payload,
            r27_error=r27_error,
            source_name=source_name,
            source_bytes=source_bytes,
            pdf_page_index=int(pdf_page_index),
            age=age,
            sex_code=sex_code,
        )
    except Exception as exc:
        pdf_bytes = None
        s.warning(f"No fue posible construir el PDF del informe: {exc}")

    d1, d2 = s.columns(2)
    with d1:
        s.download_button(
            "Descargar informe ECG (.txt)",
            data=text,
            file_name="medcalc_informe_ecg.txt",
            mime="text/plain",
            use_container_width=True,
            key="ecg_report_txt",
        )
    with d2:
        if pdf_bytes:
            s.download_button(
                "Descargar informe ECG (.pdf)",
                data=pdf_bytes,
                file_name="medcalc_informe_ecg.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="ecg_report_pdf",
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
                    "rhythm_screen": structured.get("rhythm_screen"),
                    "measurement_summary": structured.get("measurement_summary"),
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
        "MEDCALC mantiene separadas las mediciones impresas y las calculadas por el motor; "
        "ninguna fuente sustituye automáticamente a la otra. Las discordancias quedan "
        "explícitas. La morfología sólo se incorpora cuando el mapeo de derivaciones supera "
        "el control de calidad. R27 sigue siendo probability-only."
    )


def _render_r27_input_adapter(s, payload: Dict[str, Any]) -> None:
    adapter = payload.get("input_adapter") or {}
    if not adapter:
        return

    tiled = bool(adapter.get("r27_tiled"))
    mode = str(adapter.get("mode") or "—")

    if not tiled:
        s.success("Entrada R27: 10 s reales en las 12 derivaciones.")
        return

    s.error(
        "**R27-TILED · MODO EXPERIMENTAL.** "
        "R27 se ejecutó sobre una señal de compatibilidad de 10 s. "
        "Las porciones no observadas de las derivaciones incompletas se obtuvieron "
        "repitiendo exactamente el segmento real observado. Este resultado NO ha "
        "demostrado equivalencia con un ECG real de 10 s × 12 derivaciones."
    )

    provenance = adapter.get("provenance") or {}
    leads = provenance.get("lead_provenance") or {}
    rows = []
    for lead in [
        "I","II","III","aVR","aVL","aVF",
        "V1","V2","V3","V4","V5","V6",
    ]:
        info = leads.get(lead) or {}
        rows.append(
            {
                "Derivación": lead,
                "Entrada": (
                    "10 s reales"
                    if info.get("mode") == "REAL_10S"
                    else "segmento observado repetido"
                ),
                "Segundos reales usados": float(info.get("source_seconds") or 0.0),
                "Cobertura original": float(info.get("observed_fraction") or 0.0),
                "Repeticiones": int(info.get("repeat_count_ceiling") or 0),
                "Fracción repetida": float(info.get("repeated_output_fraction") or 0.0),
            }
        )

    s.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cobertura original": s.column_config.ProgressColumn(
                "Cobertura original",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "Fracción repetida": s.column_config.ProgressColumn(
                "Fracción repetida",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
        },
    )

    s.caption(
        f"Modo: {mode}. La señal repetida conserva exactamente la morfología observada; "
        "no se interpola ni se crean latidos nuevos. La periodicidad introducida puede "
        "alterar módulos sensibles a dinámica temporal, por lo que las salidas siguen "
        "siendo probabilidades de investigación."
    )


def _render_probability_table(
    s,
    payload: Dict[str, Any],
    structured_report: Dict[str, Any] | None = None,
) -> None:
    _render_r27_input_adapter(s, payload)

    structured_report = structured_report or {}
    repol = structured_report.get("repolarization") or {}
    modules = payload.get("modules") or {}
    if set(modules) != set(ALL35):
        s.error("La salida R27 no contiene exactamente los 35 módulos esperados.")
        return

    adapter = payload.get("input_adapter") or {}
    tiled = bool(adapter.get("r27_tiled"))

    rows = []
    for module in ALL35:
        item = modules[module]
        interpretability = str(item.get("interpretability") or "")
        rows.append(
            {
                "Módulo": module,
                "Probabilidad": float(item["probability"]),
                "Threshold": "NO DISPONIBLE",
                "Interpretabilidad": (
                    "NO INTERPRETABLE · R27-TILED"
                    if interpretability == "NOT_INTERPRETABLE_R27_TILED"
                    else "PROBABILITY-ONLY"
                ),
                "Clasificación binaria": "NO AUTORIZADA",
            }
        )

    rows.sort(key=lambda x: x["Probabilidad"], reverse=True)

    rhythm_keys = ["AF", "FLUTTER", "SVT", "SINUS", "SINUS_TACHY", "SINUS_ARRHYTHMIA"]
    rhythm_rows = [
        {
            "Módulo de ritmo": key,
            "Probabilidad": float(modules[key]["probability"]),
        }
        for key in rhythm_keys
        if key in modules
    ]
    rhythm_rows.sort(key=lambda x: x["Probabilidad"], reverse=True)

    if tiled:
        s.markdown("### Perfil R27 de ritmo")
        s.warning(
            "No se interpreta el perfil temporal de R27 porque la entrada contiene "
            "segmentos repetidos. La repetición exacta puede crear periodicidad "
            "artificial y sesgar AF, flutter, SVT, ectopia y bloqueos dependientes "
            "de secuencia. Para ritmo se prioriza la señal nativa observada y el "
            "strip largo real cuando esté disponible."
        )
    else:
        s.markdown("### Perfil R27 de ritmo")
        s.dataframe(
            rhythm_rows,
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
        s.caption(
            "AF, flutter, SVT, sinus y sinus tachy se muestran como probabilidades "
            "del R27 congelado. No existe threshold desplegable para convertirlas "
            "automáticamente en diagnósticos binarios."
        )

    st_model = modules.get("ST_ELEVATION") or {}
    try:
        st_model_score = float(st_model.get("probability"))
    except Exception:
        st_model_score = None
    st_depression_leads = list(repol.get("st_depression_leads") or [])
    st_elevation_leads = list(repol.get("st_elevation_leads") or [])
    if (
        st_model_score is not None
        and st_model_score >= 0.70
        and len(st_depression_leads) >= 2
        and len(st_depression_leads) > len(st_elevation_leads)
    ):
        measured = "depresión ST en " + ", ".join(st_depression_leads)
        if st_elevation_leads:
            measured += "; elevación ST en " + ", ".join(st_elevation_leads)
        s.warning(
            f"**Discordancia R27 vs medición directa:** ST_ELEVATION tiene score "
            f"{st_model_score:.2f}, pero el motor morfológico midió {measured}. "
            "Este score R27 no se interpreta como elevación del ST."
        )

    display_cutoff = 0.70
    highlighted = [
        row for row in rows
        if row["Probabilidad"] >= display_cutoff
        and row["Interpretabilidad"] != "NO INTERPRETABLE · R27-TILED"
    ]

    s.markdown("### Señales R27 destacadas")
    if highlighted:
        s.dataframe(
            highlighted,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probabilidad": s.column_config.ProgressColumn(
                    "Score R27",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.2f",
                )
            },
        )
        s.caption(
            "Se muestran únicamente módulos con score R27 ≥ 0.70 y con salida "
            "interpretable para el tipo de entrada actual."
        )
    else:
        s.info(
            "Ningún módulo interpretable alcanzó score R27 ≥ 0.70 en este ECG."
        )

    s.warning(
        "El corte de 0.70 es un filtro de visualización, no un umbral diagnóstico "
        "validado. Los scores R27 siguen siendo probability-only."
    )

    with s.expander("Auditoría técnica · ver los 35 módulos", expanded=False):
        s.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probabilidad": s.column_config.ProgressColumn(
                    "Score R27",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f",
                )
            },
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
    ecg_api_url = _get_secret(
        "ECG_R27_API_URL",
        "https://medcalc-ecg-r27.onrender.com",
    )
    ecg_api_token = _get_secret("ECG_R27_API_TOKEN")

    if not github_token:
        s.warning(
            "Falta R27_GITHUB_TOKEN en Streamlit Secrets. "
            "Se necesita para iniciar el runner privado de GitHub Actions."
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
        if not ecg_api_url:
            s.error("Backend ECG remoto no configurado.")
        else:
            try:
                status = remote_digitizer_status(
                    str(ecg_api_url),
                    str(ecg_api_token) if ecg_api_token else None,
                )
            except Exception as exc:
                s.error(f"Backend ECG no disponible: {exc}")
            else:
                remote_info = status.get("remote_digitizer") or {}
                if remote_info.get("ready"):
                    s.success(
                        "U-Net aislado del proceso Streamlit · "
                        f"backend remoto · {remote_info.get('high_fidelity_resample', 2000)} px"
                    )
                else:
                    s.error("Backend respondió, pero el digitalizador remoto no está listo.")
                s.caption(
                    "El backend sólo hace staging. El cálculo pesado se ejecuta en "
                    "GitHub Actions; Streamlit no carga PyTorch/U-Net."
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
            "Configure R27_GITHUB_TOKEN en Streamlit Secrets con acceso al "
            "repositorio privado medcalc-r27-backend y permiso Actions: write. "
            "El U-Net local permanece deshabilitado para proteger la RAM."
        )
        return

    with s.spinner(
        "Enviando el ECG a un runner privado de GitHub Actions. "
        "U-Net 2000 px + mediciones + R27 se ejecutan fuera de Streamlit…"
    ):
        try:
            result = digitize_photo_pdf_github_actions(
                str(ecg_api_url),
                str(ecg_api_token) if ecg_api_token else None,
                str(github_token),
                source_name=uploaded.name,
                source_bytes=uploaded.getvalue(),
                age=float(age),
                sex=str(sex),
                pdf_page_index=int(page_index),
                timeout_seconds=2400,
            )
        except ECGDigitiserError as exc:
            s.error(str(exc))
            return
        except Exception as exc:
            s.error(f"Fallo no esperado en runner privado foto/PDF→U-Net→R27: {exc}")
            return

    meta = result.get("digitizer") or {}
    payload = result.get("payload")
    r27_error = str(result.get("r27_error") or "").strip() or None

    _render_digitizer_meta(s, meta)
    _render_digital_audit(s, meta)
    _render_motor_measurements(s, meta, machine_measurements)
    _render_structured_report(
        s,
        meta,
        machine_measurements,
        payload,
        r27_error=r27_error,
        source_name=uploaded.name,
        source_bytes=uploaded.getvalue(),
        pdf_page_index=int(page_index),
        age=float(age),
        sex_code=str(sex),
    )

    if payload is None:
        if r27_error:
            s.warning(
                "La digitalización y el reporte del ECG se completaron, pero el runtime "
                "R27 falló después. El informe y su PDF permanecen disponibles."
            )
            short_error = r27_error
            marker = "REAL_BUILD_BLOCKER:"
            if marker in r27_error:
                short_error = marker + r27_error.split(marker, 1)[1].splitlines()[0]
            s.error(short_error[:1200])
        else:
            s.warning(
                "El ECG fue digitalizado, pero la entrada no cumplió los requisitos "
                "para ejecutar R27."
            )
            s.info(
                str(meta.get("reason") or "")
                or (
                    "La señal reconstruida no aportó cobertura suficiente en todas "
                    "las derivaciones para la ruta experimental R27."
                )
            )
        return

    _render_probability_table(
        s,
        payload,
        structured_report=structured_report,
    )
