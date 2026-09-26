from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict
from xml.sax.saxutils import escape

from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_VERSION = "MEDCALC_ECG_PDF_V2"
LEAD_ORDER = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
RHYTHM_MODULES = ["AF","FLUTTER","SVT","SINUS","SINUS_TACHY","SINUS_ARRHYTHMIA"]


def _finite(value: Any) -> float | None:
    try:
        z = float(value)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _ascii(text: Any) -> str:
    value = str(text or "")
    replacements = {
        "\u2192": "->",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00b7": "-",
        "\u2265": ">=",
        "\u2264": "<=",
        "\u00d7": "x",
        "\u00b0": " deg",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _p(value: Any, style) -> Paragraph:
    return Paragraph(escape(_ascii(value)), style)


def _metric(value: Any, suffix: str = "", digits: int = 0) -> str:
    z = _finite(value)
    if z is None:
        return "-"
    return f"{z:.{digits}f}{suffix}"


def _short_runtime_error(value: Any) -> str | None:
    text = _ascii(value).strip()
    if not text:
        return None
    match = re.search(r"(REAL_BUILD_BLOCKER:[^\n\r]+)", text)
    if match:
        return match.group(1)[:900]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else text)[:900]


def _source_preview_png(
    source_name: str | None,
    source_bytes: bytes | None,
    page_index: int,
) -> bytes | None:
    if not source_bytes:
        return None

    name = str(source_name or "").lower()
    try:
        if name.endswith(".pdf"):
            import pymupdf

            doc = pymupdf.open(stream=source_bytes, filetype="pdf")
            try:
                if doc.page_count < 1:
                    return None
                idx = min(max(int(page_index), 0), int(doc.page_count) - 1)
                page = doc.load_page(idx)
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(120 / 72, 120 / 72),
                    alpha=False,
                )
                return pix.tobytes("png")
            finally:
                doc.close()

        from PIL import Image, ImageOps

        image = ImageOps.exif_transpose(Image.open(io.BytesIO(source_bytes))).convert("RGB")
        image.thumbnail((1800, 1300))
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception:
        return None


def _scaled_image(png_bytes: bytes, max_w: float, max_h: float) -> RLImage:
    from PIL import Image

    pil = Image.open(io.BytesIO(png_bytes))
    w, h = pil.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    return RLImage(
        io.BytesIO(png_bytes),
        width=max(1.0, w * scale),
        height=max(1.0, h * scale),
    )


def _qr_drawing(payload: str, size: float = 31 * mm) -> Drawing:
    widget = QrCodeWidget(payload)
    x0, y0, x1, y1 = widget.getBounds()
    w = max(1.0, x1 - x0)
    h = max(1.0, y1 - y0)
    drawing = Drawing(
        size,
        size,
        transform=[size / w, 0, 0, size / h, 0, 0],
    )
    drawing.add(widget)
    return drawing


def _ecg_panel(lead: str, evidence: Dict[str, Any] | None) -> Drawing:
    """Vector image of the representative complex used as evidence."""
    evidence = evidence or {}
    width = 79 * mm
    height = 37 * mm
    drawing = Drawing(width, height)

    # ECG-paper-like neutral grid.
    for i in range(1, 20):
        x = width * i / 20.0
        drawing.add(
            Line(
                x,
                4,
                x,
                height - 4,
                strokeColor=colors.HexColor("#e6eaed"),
                strokeWidth=0.25 if i % 5 else 0.45,
            )
        )
    for i in range(1, 10):
        y = 4 + (height - 8) * i / 10.0
        drawing.add(
            Line(
                4,
                y,
                width - 4,
                y,
                strokeColor=colors.HexColor("#e6eaed"),
                strokeWidth=0.25 if i % 5 else 0.45,
            )
        )

    drawing.add(String(6, height - 11, lead, fontName="Helvetica-Bold", fontSize=8.5))

    duration = _finite(evidence.get("duration_s"))
    method = str(evidence.get("representative_complex_method") or "")
    if duration is not None:
        drawing.add(
            String(
                width - 6,
                height - 10,
                f"observado {duration:.1f} s",
                fontName="Helvetica",
                fontSize=5.5,
                textAnchor="end",
            )
        )

    vals = evidence.get("representative_complex_mv")
    times = evidence.get("representative_complex_time_s")
    using_complex = isinstance(vals, list) and len(vals) >= 4

    if not using_complex:
        vals = evidence.get("trace_mv")
        times = evidence.get("trace_time_s")

    pairs = []
    if isinstance(vals, list):
        for i, raw in enumerate(vals):
            v = _finite(raw)
            if v is None:
                pairs.append(None)
                continue
            t = None
            if isinstance(times, list) and i < len(times):
                t = _finite(times[i])
            pairs.append((float(i if t is None else t), v))

    good = [x for x in pairs if x is not None]
    if len(good) < 4:
        drawing.add(
            String(
                width / 2,
                height * 0.46,
                "NO EVALUABLE",
                fontName="Helvetica-Bold",
                fontSize=7,
                textAnchor="middle",
            )
        )
        return drawing

    t0 = min(x[0] for x in good)
    t1 = max(x[0] for x in good)
    if t1 <= t0:
        t0, t1 = 0.0, float(len(good) - 1)

    amp = max(max(abs(x[1]) for x in good), 0.05)
    x_left, x_right = 7.0, width - 7.0
    y_mid = height * 0.47
    y_scale = (height * 0.50) / (2.2 * amp)

    segments = []
    current = []
    for item in pairs:
        if item is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        t, v = item
        px = x_left + (x_right - x_left) * (t - t0) / max(t1 - t0, 1e-9)
        py = y_mid + v * y_scale
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)

    for points in segments:
        drawing.add(
            PolyLine(
                points,
                strokeColor=colors.HexColor("#182330"),
                strokeWidth=0.8,
            )
        )

    label = "complejo representativo" if using_complex else "segmento observado"
    if method:
        label += " - " + method.replace("_", " ").lower()
    drawing.add(
        String(
            6,
            5,
            label[:58],
            fontName="Helvetica",
            fontSize=5.2,
        )
    )
    return drawing


def _table(data, widths, *, header=True, fontsize=7.4) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), fontsize),
        ("LEADING", (0, 0), (-1, -1), fontsize + 2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.30, colors.HexColor("#cbd5de")),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header and data:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf2")),
        ]
    t.setStyle(TableStyle(style))
    return t


def build_ecg_report_pdf(
    final_report: Dict[str, Any] | None,
    *,
    machine: Dict[str, Any] | None = None,
    structured_report: Dict[str, Any] | None = None,
    digitizer: Dict[str, Any] | None = None,
    r27_payload: Dict[str, Any] | None = None,
    r27_error: str | None = None,
    source_name: str | None = None,
    source_bytes: bytes | None = None,
    pdf_page_index: int = 0,
    age: float | None = None,
    sex_code: str | None = None,
) -> bytes:
    """Create a traceable multi-page ECG PDF report.

    The report separates:
    - values printed by the electrocardiograph,
    - measurements calculated from the digitized signal,
    - descriptive signal interpretation,
    - R27 probabilities,
    - per-lead waveform evidence.

    It never turns probability-only R27 output into a thresholded diagnosis.
    """
    final_report = final_report or {}
    machine = machine or {}
    structured_report = structured_report or {}
    digitizer = digitizer or {}
    signal = digitizer.get("signal") or {}
    layout_detector = digitizer.get("layout_detector") or signal.get("preflight_layout") or {}
    motor = structured_report.get("measurement_summary") or {}
    rhythm = structured_report.get("rhythm") or {}
    rhythm_screen = structured_report.get("rhythm_screen") or {}
    repol = structured_report.get("repolarization") or {}
    assets = digitizer.get("assets") or {}

    source_sha = hashlib.sha256(source_bytes or b"").hexdigest() if source_bytes else None
    study_seed = (
        (source_sha or "NO_SOURCE")
        + "|"
        + str(pdf_page_index)
        + "|"
        + str(age)
        + "|"
        + str(sex_code)
    )
    study_id = "ECG-" + hashlib.sha256(study_seed.encode("utf-8")).hexdigest()[:16].upper()
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    adapter = (r27_payload or {}).get("input_adapter") or {}
    r27_mode = str(adapter.get("mode") or ("REAL_10S_12_LEAD" if r27_payload else "NOT_EXECUTED"))
    tiled = bool(adapter.get("r27_tiled", False))

    qr_payload = json.dumps(
        {
            "v": REPORT_VERSION,
            "study_id": study_id,
            "source_sha256": source_sha,
            "page": int(pdf_page_index) + 1,
            "layout": signal.get("layout_name") or layout_detector.get("layout"),
            "r27_mode": r27_mode,
            "r27_tiled": tiled,
            "research_only": True,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=15 * mm,
        bottomMargin=16 * mm,
        title="MEDCALC - Informe electrocardiografico",
        author="MEDCALC",
        subject=study_id,
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_LEFT,
        spaceAfter=3,
        textColor=colors.HexColor("#12202f"),
    )
    subtitle = ParagraphStyle(
        "ST",
        parent=styles["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#607184"),
        spaceAfter=5,
    )
    heading = ParagraphStyle(
        "H",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=7,
        spaceAfter=5,
        textColor=colors.HexColor("#12202f"),
    )
    body = ParagraphStyle(
        "B",
        parent=styles["BodyText"],
        fontSize=8.6,
        leading=11.5,
        spaceAfter=3,
    )
    small = ParagraphStyle(
        "S",
        parent=body,
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#5f6f7e"),
    )
    warning = ParagraphStyle(
        "W",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#8a3b00"),
        backColor=colors.HexColor("#fff2d8"),
        borderPadding=5,
        spaceBefore=4,
        spaceAfter=5,
    )
    report_line = ParagraphStyle(
        "RL",
        parent=body,
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        leftIndent=2 * mm,
        rightIndent=2 * mm,
        spaceAfter=2,
    )

    preview = _source_preview_png(source_name, source_bytes, pdf_page_index)
    qr = _qr_drawing(qr_payload)

    identity_rows = [
        [_p("ID estudio", small), _p(study_id, body)],
        [_p("Generado UTC", small), _p(generated, body)],
        [_p("Archivo", small), _p(source_name or "-", body)],
        [_p("Pagina analizada", small), _p(int(pdf_page_index) + 1, body)],
        [_p("Edad", small), _p("-" if age is None else f"{float(age):.0f} anos", body)],
        [_p("Sexo runtime", small), _p(sex_code if sex_code is not None else "-", body)],
    ]
    identity = _table(identity_rows, [31 * mm, 76 * mm], header=False, fontsize=7.3)

    top_left = [
        Paragraph("MEDCALC CLINICO", title),
        Paragraph("Informe electrocardiografico automatizado - foto/PDF", subtitle),
        identity,
        Spacer(1, 2 * mm),
        Paragraph(
            "QR de trazabilidad: contiene ID del estudio, SHA-256 del archivo fuente, "
            "layout y modo de entrada R27. No contiene nombre del paciente.",
            small,
        ),
    ]
    top = Table([[top_left, qr]], colWidths=[128 * mm, 36 * mm], hAlign="LEFT")
    top.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("BOX", (0,0), (-1,-1), 0, colors.white)]))

    story = [top]

    if preview:
        story += [
            Spacer(1, 3 * mm),
            Paragraph("Trazado fuente", heading),
            KeepTogether([
                _scaled_image(preview, 178 * mm, 83 * mm),
                Paragraph(
                    "Vista del documento original utilizado como fuente. Las imagenes de "
                    "complejos mostradas mas adelante proceden de la senal digitalizada, "
                    "no son recortes fotografico-forenses del papel.",
                    small,
                ),
            ]),
        ]

    layout = signal.get("layout_name") or layout_detector.get("layout") or "-"
    confidence = _finite(layout_detector.get("confidence"))
    observed = signal.get("observed_fraction_by_lead") or {}

    trace_rows = [
        ["Parametro", "Valor"],
        ["Layout aceptado", _ascii(layout)],
        ["Confianza detector frontal", f"{100*confidence:.1f}%" if confidence is not None else "-"],
        ["Estado digitalizador", digitizer.get("status") or "-"],
        ["Cobertura minima", f"{100*float(signal.get('min_observed_fraction') or 0):.1f}%"],
        ["Modo R27", r27_mode],
        ["R27-TILED", "SI - EXPERIMENTAL" if tiled else "NO"],
        ["Digitizer", assets.get("source_repository") or "Ahus-AIM/Open-ECG-Digitizer"],
        ["Commit digitizer", assets.get("source_commit") or "-"],
        ["SHA modelo segmentacion", assets.get("segmentation_model_sha256") or "-"],
        ["SHA modelo derivaciones", assets.get("lead_model_sha256") or "-"],
    ]
    story += [Paragraph("Trazabilidad tecnica", heading), _table(trace_rows, [54*mm, 112*mm])]

    # Measurement comparison.
    pr_machine = None if machine.get("pr_printed_ms") == 0 else machine.get("pr_ms")
    overlap = [
        ("FC", machine.get("heart_rate_bpm"), motor.get("heart_rate_bpm"), "LPM", 10),
        ("PR", pr_machine, motor.get("pr_ms"), "ms", 30),
        ("QRS", machine.get("qrs_ms"), motor.get("qrs_ms"), "ms", 20),
        ("QT", machine.get("qt_ms"), motor.get("qt_ms"), "ms", 40),
        ("QTc", machine.get("qtc_ms"), motor.get("qtc_bazett_ms"), "ms", 40),
        ("Eje QRS", machine.get("qrs_axis_deg"), motor.get("axis_deg"), "deg", 20),
    ]
    rows = [["Medicion", "Equipo impreso", "Motor MEDCALC", "Delta", "Estado"]]
    for name, printed, measured, unit, tol in overlap:
        pval, mval = _finite(printed), _finite(measured)
        delta = abs(pval-mval) if pval is not None and mval is not None else None
        state = "CONCORDANTE" if delta is not None and delta <= tol else "DISCORDANTE" if delta is not None else "NO COMPARABLE"
        if name == "PR" and machine.get("pr_printed_ms") == 0:
            ptxt = "NO CALCULABLE (0 ms)"
        else:
            ptxt = _metric(pval, " "+unit)
        rows.append([
            name,
            ptxt,
            _metric(mval, " "+unit),
            _metric(delta, " "+unit),
            state,
        ])

    story += [
        Paragraph("Mediciones: equipo vs motor", heading),
        _table(rows, [26*mm, 39*mm, 39*mm, 27*mm, 35*mm], fontsize=6.8),
        Paragraph(
            "Los datos impresos no se usan como verdad unica. El motor vuelve a medir "
            "sobre la senal digitalizada y las discordancias permanecen visibles.",
            small,
        ),
    ]

    machine_extra = [
        ["Medicion impresa adicional", "Valor"],
        ["Eje P", _metric(machine.get("p_axis_deg"), " deg")],
        ["Eje T", _metric(machine.get("t_axis_deg"), " deg")],
        ["Velocidad", _metric(machine.get("speed_mm_per_s"), " mm/s")],
        ["Ganancia", _metric(machine.get("gain_mm_per_mV"), " mm/mV")],
    ]
    motor_extra = [
        ["Medicion del motor", "Valor"],
        ["Derivacion de ritmo", rhythm.get("lead") or "-"],
        ["Duracion evaluada", _metric(rhythm.get("duration_s"), " s", 2)],
        ["Latidos/QRS detectados", _metric(motor.get("beat_n"))],
        ["RR CV", _metric(motor.get("rr_cv"), "", 3)],
        ["P antes de QRS", _metric(motor.get("p_before_qrs_ratio"), "", 2)],
        ["Extrasistolia - eventos", _metric(motor.get("premature_pattern_count"))],
        ["Screening de ritmo", rhythm_screen.get("label") or "NO EVALUABLE"],
    ]
    extras = Table(
        [[_table(machine_extra, [43*mm, 38*mm], fontsize=6.6), _table(motor_extra, [45*mm, 40*mm], fontsize=6.6)]],
        colWidths=[84*mm, 88*mm],
    )
    extras.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),2)]))
    story += [Spacer(1, 2*mm), extras]

    # Coverage.
    coverage_rows = [["Derivacion", "Cobertura observada"]]
    for lead in LEAD_ORDER:
        coverage_rows.append([lead, f"{100*float(observed.get(lead) or 0):.1f}%"])
    story += [
        Paragraph("Cobertura por derivacion", heading),
        _table(coverage_rows, [45*mm, 45*mm], fontsize=6.8),
    ]

    # Final interpretation.
    report_text = _ascii(final_report.get("text") or "").strip()
    story += [PageBreak(), Paragraph("Informe electrocardiografico automatizado", heading)]
    if report_text:
        for line in report_text.splitlines():
            if line.strip():
                story.append(_p(line.strip(), report_line))
    else:
        story.append(_p("No fue posible generar el informe estructurado.", body))

    if rhythm_screen:
        basis = rhythm_screen.get("basis") or []
        story += [
            Paragraph("Fundamento del screening de ritmo", heading),
            _p(rhythm_screen.get("label") or "NO EVALUABLE", body),
        ]
        if basis:
            story.append(_p(" | ".join(map(str, basis)), small))

    # Per-lead ST/T.
    per_lead = repol.get("per_lead") or {}
    st_rows = [["Derivacion", "ST motor (mV)", "T motor (mV)", "Evaluable"]]
    for lead in LEAD_ORDER:
        item = per_lead.get(lead) or {}
        st_rows.append([
            lead,
            _metric(item.get("st_mv"), "", 3),
            _metric(item.get("t_mv"), "", 3),
            "SI" if item.get("evaluable") else "NO",
        ])
    story += [
        Paragraph("Repolarizacion por derivacion", heading),
        _table(st_rows, [33*mm, 42*mm, 42*mm, 30*mm], fontsize=6.7),
    ]

    # R27 sections.
    if isinstance(r27_payload, dict):
        modules = r27_payload.get("modules") or {}
        if tiled:
            story += [
                Paragraph("R27-TILED - advertencia", heading),
                Paragraph(
                    "MODO EXPERIMENTAL: una o mas derivaciones fueron extendidas hasta "
                    "10 s mediante repeticion exacta del segmento observado. No se ha "
                    "demostrado equivalencia con un ECG real de 10 s x 12 derivaciones. "
                    "Las salidas permanecen probability-only.",
                    warning,
                ),
            ]
            lead_prov = (adapter.get("provenance") or {}).get("lead_provenance") or {}
            prov_rows = [["Lead","Entrada","Segundos reales","Fraccion repetida"]]
            for lead in LEAD_ORDER:
                info = lead_prov.get(lead) or {}
                prov_rows.append([
                    lead,
                    "REAL 10 s" if info.get("mode") == "REAL_10S" else "REPETIDO",
                    _metric(info.get("source_seconds"), " s", 2),
                    _metric(100*_finite(info.get("repeated_output_fraction")) if _finite(info.get("repeated_output_fraction")) is not None else None, "%", 1),
                ])
            story.append(_table(prov_rows, [25*mm, 42*mm, 48*mm, 48*mm], fontsize=6.5))

        rhythm_rows = [["Modulo R27 de ritmo", "Probabilidad"]]
        for key in RHYTHM_MODULES:
            item = modules.get(key)
            if isinstance(item, dict):
                pval = _finite(item.get("probability"))
                rhythm_rows.append([key, "-" if pval is None else f"{pval:.4f}"])
        story += [
            Paragraph("R27 - perfil de ritmo", heading),
            _table(rhythm_rows, [75*mm, 48*mm], fontsize=7),
            Paragraph(
                "Probabilidades sin umbral desplegable: no equivalen por si solas a "
                "un diagnostico binario.",
                small,
            ),
        ]

        all_rows = [["Modulo R27", "Probabilidad", "Clasificacion"]]
        for key in sorted(modules):
            item = modules.get(key) or {}
            pval = _finite(item.get("probability"))
            all_rows.append([
                key,
                "-" if pval is None else f"{pval:.4f}",
                "NO AUTORIZADA",
            ])
        story += [
            Paragraph("R27 - 35 modulos", heading),
            _table(all_rows, [77*mm, 43*mm, 48*mm], fontsize=6.2),
        ]

    runtime_error = _short_runtime_error(r27_error)
    if runtime_error:
        story += [
            Paragraph("Incidencia de runtime R27", heading),
            _p(runtime_error, body),
            Paragraph(
                "La digitalizacion y las mediciones del motor permanecen disponibles "
                "aunque el runtime R27 falle posteriormente.",
                small,
            ),
        ]

    # Evidence panels.
    evidence_by_lead = structured_report.get("evidence_by_lead") or {}
    story += [PageBreak(), Paragraph("Complejos representativos por derivacion", heading)]
    story.append(
        Paragraph(
            "Cada panel muestra la senal digitalizada que sustento el analisis automatizado "
            "de esa derivacion. Cuando fue posible, se centra un complejo representativo "
            "alrededor de un QRS detectado; si no, se muestra el segmento observado. "
            "No se inventa una forma de onda ausente.",
            small,
        )
    )
    panels = [_ecg_panel(lead, evidence_by_lead.get(lead)) for lead in LEAD_ORDER]
    grid_rows = [[panels[i], panels[i+1]] for i in range(0, 12, 2)]
    ev_table = Table(grid_rows, colWidths=[84*mm,84*mm], rowHeights=[39*mm]*6)
    ev_table.setStyle(
        TableStyle([
            ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#cbd5de")),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("LEFTPADDING",(0,0),(-1,-1),1),
            ("RIGHTPADDING",(0,0),(-1,-1),1),
            ("TOPPADDING",(0,0),(-1,-1),1),
            ("BOTTOMPADDING",(0,0),(-1,-1),1),
        ])
    )
    story.append(ev_table)

    # Final limitations / traceability.
    story += [PageBreak(), Paragraph("Limitaciones, trazabilidad y verificacion", heading)]
    limitations = structured_report.get("limitations") or []
    for item in limitations:
        story.append(_p("- " + str(item), small))
    story.append(
        Paragraph(
            "Este documento es una salida de investigacion. Debe contrastarse con el "
            "ECG fuente y el contexto clinico. R27 permanece probability-only, sin "
            "thresholds desplegables y sin autorizacion de clasificacion binaria.",
            warning,
        )
    )
    hash_rows = [
        ["Elemento", "Identificador"],
        ["Version PDF", REPORT_VERSION],
        ["ID estudio", study_id],
        ["SHA-256 fuente", source_sha or "-"],
        ["Layout", layout],
        ["Modo R27", r27_mode],
        ["R27-TILED", "SI" if tiled else "NO"],
        ["Digitizer commit", assets.get("source_commit") or "-"],
        ["Segmentation SHA-256", assets.get("segmentation_model_sha256") or "-"],
        ["Lead model SHA-256", assets.get("lead_model_sha256") or "-"],
    ]
    story.append(_table(hash_rows, [47*mm, 119*mm], fontsize=6.5))
    story += [Spacer(1,4*mm), Paragraph("QR de trazabilidad", heading), qr]
    story.append(_p("Contenido QR: " + qr_payload, small))

    def _footer(canvas, pdf_doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d6dde3"))
        canvas.line(14*mm, 12.5*mm, A4[0]-14*mm, 12.5*mm)
        canvas.setFont("Helvetica", 6.6)
        canvas.setFillColor(colors.HexColor("#687887"))
        canvas.drawString(14*mm, 8.5*mm, f"MEDCALC - Modo investigacion - {study_id}")
        canvas.drawRightString(A4[0]-14*mm, 8.5*mm, f"Pagina {pdf_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
