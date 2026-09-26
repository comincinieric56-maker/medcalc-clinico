from __future__ import annotations

import io
import math
import re
from typing import Any, Dict
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.graphics.shapes import Drawing, Line, PolyLine, String
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _finite(value: Any) -> float | None:
    try:
        z = float(value)
        return z if math.isfinite(z) else None
    except Exception:
        return None


def _ascii_punctuation(text: Any) -> str:
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
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def _short_runtime_error(value: Any) -> str | None:
    text = _ascii_punctuation(value).strip()
    if not text:
        return None

    # Preserve the clinically useful runtime failure identifier without placing
    # several thousand characters of stdout/stderr into the clinical PDF.
    match = re.search(r"(REAL_BUILD_BLOCKER:[^\n\r]+)", text)
    if match:
        return match.group(1)[:500]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1] if lines else text)[:500]


def _metric(value: Any, suffix: str = "") -> str:
    z = _finite(value)
    if z is None:
        return "-"
    return f"{z:.0f}{suffix}"


def _lead_preview(lead: str, evidence: Dict[str, Any] | None) -> Drawing:
    evidence = evidence or {}
    width = 49 * mm
    height = 25 * mm
    drawing = Drawing(width, height)

    drawing.add(String(3, height - 10, str(lead), fontName="Helvetica-Bold", fontSize=7.5))
    duration = _finite(evidence.get("duration_s"))
    if duration is not None:
        drawing.add(
            String(
                width - 3,
                height - 10,
                f"{duration:.1f} s",
                fontName="Helvetica",
                fontSize=5.5,
                textAnchor="end",
            )
        )

    drawing.add(Line(3, height * 0.46, width - 3, height * 0.46, strokeWidth=0.25))

    values = evidence.get("representative_complex_mv")
    if not isinstance(values, list) or len(values) < 4:
        values = evidence.get("trace_mv")

    numeric = []
    for value in values or []:
        z = _finite(value)
        numeric.append(z)

    finite_values = [z for z in numeric if z is not None]
    if len(finite_values) < 4:
        drawing.add(
            String(
                width / 2,
                height * 0.40,
                "NO EVALUABLE",
                fontName="Helvetica",
                fontSize=6,
                textAnchor="middle",
            )
        )
        return drawing

    lo = min(finite_values)
    hi = max(finite_values)
    amp = max(abs(lo), abs(hi), 0.05)
    x0, x1 = 3.0, width - 3.0
    y0 = height * 0.12
    y1 = height * 0.80
    mid = (y0 + y1) / 2.0
    scale = (y1 - y0) / (2.2 * amp)

    segments = []
    current = []
    n = max(1, len(numeric) - 1)
    for i, value in enumerate(numeric):
        if value is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue
        px = x0 + (x1 - x0) * i / n
        py = mid + value * scale
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)

    for points in segments:
        drawing.add(PolyLine(points, strokeWidth=0.65))

    return drawing


def build_ecg_report_pdf(
    final_report: Dict[str, Any] | None,
    *,
    machine: Dict[str, Any] | None = None,
    structured_report: Dict[str, Any] | None = None,
    digitizer: Dict[str, Any] | None = None,
    r27_payload: Dict[str, Any] | None = None,
    r27_error: str | None = None,
) -> bytes:
    """Create the downloadable MEDCALC ECG report as a compact A4 PDF.

    The PDF is documentary: it preserves printed machine measurements, the
    descriptive report derived from the digitized signal, and the state of R27.
    It does not convert R27 probabilities into thresholded diagnoses.
    """

    final_report = final_report or {}
    machine = machine or {}
    structured_report = structured_report or {}
    digitizer = digitizer or {}
    signal = digitizer.get("signal") or {}
    layout_detector = digitizer.get("layout_detector") or {}
    motor = structured_report.get("measurement_summary") or {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="MEDCALC - Informe electrocardiografico automatizado",
        author="MEDCALC",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "MedcalcTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "MedcalcSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#526273"),
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "MedcalcHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=13,
        spaceBefore=8,
        spaceAfter=5,
        textColor=colors.HexColor("#12202f"),
    )
    body = ParagraphStyle(
        "MedcalcBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )
    small = ParagraphStyle(
        "MedcalcSmall",
        parent=body,
        fontSize=7.6,
        leading=10,
        textColor=colors.HexColor("#596979"),
    )
    report_line = ParagraphStyle(
        "MedcalcReportLine",
        parent=body,
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.5,
        leftIndent=2 * mm,
        rightIndent=2 * mm,
        spaceAfter=3,
    )

    story = [
        Paragraph("MEDCALC", title),
        Paragraph(
            "Informe electrocardiografico automatizado - entrada foto/PDF",
            subtitle,
        ),
    ]

    layout = (
        layout_detector.get("layout")
        or signal.get("layout_name")
        or "-"
    )
    confidence = _finite(layout_detector.get("confidence"))
    confidence_text = f"{100.0 * confidence:.1f}%" if confidence is not None else "-"
    r27_state = "EJECUTADO" if isinstance(r27_payload, dict) else "NO DISPONIBLE"
    if r27_error:
        r27_state = "FALLO DE RUNTIME"

    qc_rows = [
        ["Fuente", "Foto/PDF"],
        ["Formato detectado", _ascii_punctuation(layout)],
        ["Confianza de formato", confidence_text],
        ["Ruta de digitalizacion", _ascii_punctuation(signal.get("canonicalizer") or signal.get("layout_source") or "-")],
        ["Estado R27", r27_state],
    ]
    qc_table = Table(qc_rows, colWidths=[50 * mm, 110 * mm], hAlign="LEFT")
    qc_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f7")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cad4de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [Paragraph("Trazabilidad tecnica", heading), qc_table, Spacer(1, 4 * mm)]

    pr_printed = machine.get("pr_printed_ms")
    if pr_printed == 0:
        pr_machine = "NO CALCULABLE (0 ms impreso)"
    else:
        pr_machine = _metric(machine.get("pr_ms"), " ms")

    machine_rows = [
        ["Medicion", "Equipo impreso", "Motor MEDCALC"],
        ["FC", _metric(machine.get("heart_rate_bpm"), " LPM"), _metric(motor.get("heart_rate_bpm"), " LPM")],
        ["PR", pr_machine, _metric(motor.get("pr_ms"), " ms")],
        ["QRS", _metric(machine.get("qrs_ms"), " ms"), _metric(motor.get("qrs_ms"), " ms")],
        ["Eje QRS", _metric(machine.get("qrs_axis_deg"), " deg"), _metric(motor.get("axis_deg"), " deg")],
        [
            "QT/QTc",
            (
                f"{_metric(machine.get('qt_ms'))}/{_metric(machine.get('qtc_ms'))} ms"
                if _finite(machine.get("qt_ms")) is not None and _finite(machine.get("qtc_ms")) is not None
                else "-"
            ),
            (
                f"{_metric(motor.get('qt_ms'))}/{_metric(motor.get('qtc_bazett_ms'))} ms"
                if _finite(motor.get("qt_ms")) is not None and _finite(motor.get("qtc_bazett_ms")) is not None
                else "-"
            ),
        ],
    ]
    measurements = Table(machine_rows, colWidths=[36 * mm, 62 * mm, 62 * mm], hAlign="LEFT")
    measurements.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf2")),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.7),
                ("LEADING", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cad4de")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story += [Paragraph("Mediciones", heading), measurements, Spacer(1, 4 * mm)]

    report_text = _ascii_punctuation(final_report.get("text") or "").strip()
    story.append(Paragraph("Reporte automatizado", heading))
    if report_text:
        for line in report_text.splitlines():
            if line.strip():
                story.append(Paragraph(escape(line.strip()), report_line))
    else:
        story.append(Paragraph("No fue posible generar el reporte estructurado.", body))

    if isinstance(r27_payload, dict):
        modules = r27_payload.get("modules") or {}
        rhythm_probs = []
        for key in ("AF", "FLUTTER", "SVT", "SINUS", "SINUS_TACHY"):
            item = modules.get(key)
            if isinstance(item, dict) and _finite(item.get("probability")) is not None:
                rhythm_probs.append(
                    f"{key}: {float(item['probability']):.3f}"
                )
        if rhythm_probs:
            story += [
                Paragraph("R27 - probabilidades de ritmo", heading),
                Paragraph(
                    escape(" | ".join(rhythm_probs)),
                    body,
                ),
                Paragraph(
                    "Estas probabilidades se muestran sin umbral diagnostico y no equivalen por si solas a un diagnostico positivo.",
                    small,
                ),
            ]

    evidence_by_lead = structured_report.get("evidence_by_lead") or {}
    if evidence_by_lead:
        story.append(Paragraph("Evidencia digitalizada por derivacion", heading))
        story.append(
            Paragraph(
                "Vista vectorial compacta de la senal reconstruida. Se usa el complejo representativo cuando esta disponible; de lo contrario, una vista resumida del segmento observado.",
                small,
            )
        )
        lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        panels = [_lead_preview(lead, evidence_by_lead.get(lead)) for lead in lead_order]
        rows = [panels[i:i + 3] for i in range(0, len(panels), 3)]
        trace_table = Table(rows, colWidths=[53 * mm] * 3, rowHeights=[28 * mm] * 4)
        trace_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cfd8df")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        story += [trace_table, Spacer(1, 3 * mm)]

    runtime_error = _short_runtime_error(r27_error)
    if runtime_error:
        story += [
            Paragraph("Incidencia de runtime R27", heading),
            Paragraph(escape(runtime_error), body),
            Paragraph(
                "El fallo de R27 no invalida automaticamente las mediciones documentales impresas ni el reporte descriptivo que haya superado el control de calidad de la senal digitalizada.",
                small,
            ),
        ]

    limitations = structured_report.get("limitations") or []
    story.append(Paragraph("Limitaciones y uso", heading))
    story.append(
        Paragraph(
            "La senal fue reconstruida desde una fotografia o PDF. Debe correlacionarse con el trazado original y con el contexto clinico. MEDCALC mantiene separadas las mediciones impresas, las mediciones derivadas de la senal y las probabilidades R27.",
            small,
        )
    )
    for item in limitations:
        story.append(Paragraph("- " + escape(_ascii_punctuation(item)), small))

    def _footer(canvas, pdf_doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#6d7b88"))
        canvas.drawString(17 * mm, 9 * mm, "MEDCALC - Modo investigacion")
        canvas.drawRightString(
            A4[0] - 17 * mm,
            9 * mm,
            f"Pagina {pdf_doc.page}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
