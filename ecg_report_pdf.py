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


REPORT_VERSION = "MEDCALC_ECG_PDF_V4"
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



def _observed_trace_pairs(evidence: Dict[str, Any] | None) -> list:
    """Return time/mV pairs from observed native samples prepared for PDF."""
    evidence = evidence or {}
    vals = evidence.get("pdf_trace_mv")
    times = evidence.get("pdf_trace_time_s")
    if not isinstance(vals, list) or len(vals) < 2:
        vals = evidence.get("trace_mv")
        times = evidence.get("trace_time_s")

    pairs = []
    if not isinstance(vals, list):
        return pairs
    for i, raw in enumerate(vals):
        v = _finite(raw)
        if v is None:
            pairs.append(None)
            continue
        t = None
        if isinstance(times, list) and i < len(times):
            t = _finite(times[i])
        pairs.append((float(i if t is None else t), float(v)))
    return pairs


def _ecg_trace_panel(
    lead: str,
    evidence: Dict[str, Any] | None,
    *,
    width: float = 82 * mm,
    height: float = 28 * mm,
) -> Drawing:
    """Render the full observed digitalized segment for one lead."""
    evidence = evidence or {}
    drawing = Drawing(width, height)

    # Neutral ECG-like grid for visual audit. Vertical scale is auto-fit per lead
    # because printed-photo calibration may be absent; actual values remain mV.
    for i in range(1, 26):
        x = 4 + (width - 8) * i / 26.0
        drawing.add(
            Line(
                x, 4, x, height - 4,
                strokeColor=colors.HexColor("#edf0f2"),
                strokeWidth=0.20 if i % 5 else 0.38,
            )
        )
    for i in range(1, 8):
        y = 4 + (height - 8) * i / 8.0
        drawing.add(
            Line(
                4, y, width - 4, y,
                strokeColor=colors.HexColor("#edf0f2"),
                strokeWidth=0.20 if i % 4 else 0.38,
            )
        )

    duration = _finite(evidence.get("duration_s"))
    drawing.add(String(5, height - 9, lead, fontName="Helvetica-Bold", fontSize=7.2))
    if duration is not None:
        drawing.add(
            String(
                width - 5,
                height - 9,
                f"{duration:.2f} s observados",
                fontName="Helvetica",
                fontSize=5.0,
                textAnchor="end",
            )
        )

    pairs = _observed_trace_pairs(evidence)
    good = [p for p in pairs if p is not None]
    if len(good) < 4:
        drawing.add(
            String(
                width / 2, height * 0.46, "NO EVALUABLE",
                fontName="Helvetica-Bold", fontSize=6.5, textAnchor="middle",
            )
        )
        return drawing

    t0 = min(p[0] for p in good)
    t1 = max(p[0] for p in good)
    if t1 <= t0:
        t0, t1 = 0.0, float(len(good) - 1)

    abs_vals = sorted(abs(p[1]) for p in good)
    q_index = min(len(abs_vals) - 1, max(0, int(round(0.98 * (len(abs_vals) - 1)))))
    amp = max(abs_vals[q_index], 0.05)
    x_left, x_right = 5.0, width - 5.0
    y_mid = height * 0.48
    y_scale = (height * 0.55) / (2.2 * amp)

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
        py = y_mid + max(-1.15 * amp, min(1.15 * amp, v)) * y_scale
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)

    for points in segments:
        drawing.add(
            PolyLine(
                points,
                strokeColor=colors.HexColor("#17222d"),
                strokeWidth=0.65,
            )
        )

    drawing.add(
        String(
            5, 3.5,
            f"senal digitalizada observada | autoescala +/-{amp:.2f} mV",
            fontName="Helvetica",
            fontSize=4.6,
        )
    )
    return drawing


def _rhythm_strip_drawing(
    evidence: Dict[str, Any] | None,
    rhythm: Dict[str, Any] | None,
    *,
    lead: str,
    fs: int,
    width: float = 168 * mm,
    height: float = 45 * mm,
) -> Drawing:
    """Render the native observed strip used by the rhythm engine."""
    evidence = evidence or {}
    rhythm = rhythm or {}
    drawing = Drawing(width, height)

    for i in range(1, 41):
        x = 5 + (width - 10) * i / 42.0
        drawing.add(
            Line(
                x, 5, x, height - 5,
                strokeColor=colors.HexColor("#e7ebee"),
                strokeWidth=0.20 if i % 5 else 0.45,
            )
        )
    for i in range(1, 10):
        y = 5 + (height - 10) * i / 10.0
        drawing.add(
            Line(
                5, y, width - 5, y,
                strokeColor=colors.HexColor("#e7ebee"),
                strokeWidth=0.20 if i % 5 else 0.45,
            )
        )

    pairs = _observed_trace_pairs(evidence)
    good = [p for p in pairs if p is not None]
    drawing.add(
        String(
            6, height - 10,
            f"Derivacion {lead} - strip nativo observado",
            fontName="Helvetica-Bold", fontSize=8.2,
        )
    )
    if len(good) < 4:
        drawing.add(
            String(
                width / 2, height * 0.45, "STRIP NO EVALUABLE",
                fontName="Helvetica-Bold", fontSize=8, textAnchor="middle",
            )
        )
        return drawing

    t0 = min(p[0] for p in good)
    t1 = max(p[0] for p in good)
    abs_vals = sorted(abs(p[1]) for p in good)
    q_index = min(len(abs_vals) - 1, max(0, int(round(0.98 * (len(abs_vals) - 1)))))
    amp = max(abs_vals[q_index], 0.05)

    x_left, x_right = 7.0, width - 7.0
    y_mid = height * 0.48
    y_scale = (height * 0.54) / (2.2 * amp)

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
        py = y_mid + max(-1.15 * amp, min(1.15 * amp, v)) * y_scale
        current.append((px, py))
    if len(current) >= 2:
        segments.append(current)
    for points in segments:
        drawing.add(
            PolyLine(
                points,
                strokeColor=colors.HexColor("#101a24"),
                strokeWidth=0.75,
            )
        )

    # QRS markers are derived from the selected native strip, not R27-TILED.
    r_peaks = rhythm.get("r_peaks_local") or []
    valid_r = []
    for raw in r_peaks:
        try:
            rp = int(raw)
        except Exception:
            continue
        rt = rp / float(fs)
        if t0 <= rt <= t1:
            valid_r.append(rt)
            px = x_left + (x_right - x_left) * (rt - t0) / max(t1 - t0, 1e-9)
            drawing.add(
                Line(
                    px, height - 17, px, height - 12,
                    strokeColor=colors.HexColor("#315c79"),
                    strokeWidth=0.8,
                )
            )

    drawing.add(
        String(
            6, 4,
            (
                f"{t1 - t0:.2f} s | {len(valid_r)} QRS marcados | "
                f"{fs} Hz | centerline observada, sin repeticion temporal"
            ),
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



PDF_COLORS = {
    "ink": colors.HexColor("#13212B"),
    "muted": colors.HexColor("#647481"),
    "line": colors.HexColor("#D7E0E6"),
    "paper": colors.HexColor("#F7F9FB"),
    "navy": colors.HexColor("#14384A"),
    "teal": colors.HexColor("#1C7B86"),
    "blue": colors.HexColor("#2E6F95"),
    "green": colors.HexColor("#2C7A5A"),
    "amber": colors.HexColor("#B7791F"),
    "red": colors.HexColor("#A94B4B"),
    "soft_teal": colors.HexColor("#EAF5F5"),
    "soft_blue": colors.HexColor("#EBF3F8"),
    "soft_green": colors.HexColor("#EAF5EF"),
    "soft_amber": colors.HexColor("#FFF4DE"),
    "soft_red": colors.HexColor("#FBECEC"),
    "white": colors.white,
}


def _tone_pair(tone: str) -> tuple:
    tone = str(tone or "blue").lower()
    mapping = {
        "teal": (PDF_COLORS["teal"], PDF_COLORS["soft_teal"]),
        "green": (PDF_COLORS["green"], PDF_COLORS["soft_green"]),
        "amber": (PDF_COLORS["amber"], PDF_COLORS["soft_amber"]),
        "red": (PDF_COLORS["red"], PDF_COLORS["soft_red"]),
        "navy": (PDF_COLORS["navy"], PDF_COLORS["soft_blue"]),
        "blue": (PDF_COLORS["blue"], PDF_COLORS["soft_blue"]),
    }
    return mapping.get(tone, mapping["blue"])


def _clean_report_fields(report_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in str(report_text or "").splitlines():
        line = _ascii(raw).strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().upper()
        value = value.strip()
        if key and value:
            fields[key] = value
    return fields


def _status_tone(state: Any) -> str:
    value = str(state or "").upper()
    if "CONCORDANTE" in value and "DISCORDANTE" not in value:
        return "green"
    if "DISCORDANTE" in value:
        return "red"
    if "NO EVALUABLE" in value or "NOT_EXECUTED" in value or "NO COMPARABLE" in value:
        return "amber"
    return "blue"


def _mini_badge(text: Any, *, tone: str = "blue") -> Table:
    accent, soft = _tone_pair(tone)
    style = ParagraphStyle(
        "MiniBadge",
        fontName="Helvetica-Bold",
        fontSize=6.1,
        leading=7.2,
        textColor=accent,
        alignment=TA_CENTER,
    )
    t = Table([[_p(str(text or "-").upper(), style)]], colWidths=[34 * mm])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), soft),
            ("BOX", (0, 0), (-1, -1), 0.55, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    return t


def _info_card(
    label: Any,
    value: Any,
    *,
    subtitle: Any = "",
    width: float = 52 * mm,
    tone: str = "blue",
    value_size: float = 12.5,
) -> Table:
    accent, soft = _tone_pair(tone)
    label_style = ParagraphStyle(
        "CardLabel",
        fontName="Helvetica-Bold",
        fontSize=6.2,
        leading=7.4,
        textColor=PDF_COLORS["muted"],
        spaceAfter=2,
    )
    value_style = ParagraphStyle(
        "CardValue",
        fontName="Helvetica-Bold",
        fontSize=value_size,
        leading=value_size + 2.0,
        textColor=PDF_COLORS["ink"],
        spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "CardSub",
        fontName="Helvetica",
        fontSize=6.3,
        leading=7.6,
        textColor=PDF_COLORS["muted"],
    )
    flow = [
        _p(str(label or "").upper(), label_style),
        _p(value if value not in (None, "") else "-", value_style),
    ]
    if str(subtitle or "").strip():
        flow.append(_p(subtitle, sub_style))
    t = Table([[flow]], colWidths=[width])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), soft),
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    return t


def _text_panel(
    title_text: Any,
    body_text: Any,
    *,
    width: float = 166 * mm,
    tone: str = "blue",
    compact: bool = False,
) -> Table:
    accent, soft = _tone_pair(tone)
    label_style = ParagraphStyle(
        "PanelTitle",
        fontName="Helvetica-Bold",
        fontSize=7.1 if compact else 8.2,
        leading=9.0 if compact else 10.2,
        textColor=accent,
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "PanelBody",
        fontName="Helvetica",
        fontSize=7.4 if compact else 8.4,
        leading=9.4 if compact else 11.2,
        textColor=PDF_COLORS["ink"],
    )
    flow = [
        _p(str(title_text or "").upper(), label_style),
        _p(body_text if body_text not in (None, "") else "-", body_style),
    ]
    t = Table([[flow]], colWidths=[width])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), soft),
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return t


def _section_label(
    title_text: Any,
    *,
    eyebrow: Any = None,
    subtitle: Any = None,
    width: float = 166 * mm,
) -> Table:
    eyebrow_style = ParagraphStyle(
        "SectionEyebrow",
        fontName="Helvetica-Bold",
        fontSize=6.2,
        leading=7.3,
        textColor=PDF_COLORS["teal"],
        spaceAfter=1,
    )
    title_style = ParagraphStyle(
        "SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=12.2,
        leading=14.5,
        textColor=PDF_COLORS["navy"],
        spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "SectionSub",
        fontName="Helvetica",
        fontSize=7.0,
        leading=8.6,
        textColor=PDF_COLORS["muted"],
    )
    flow = []
    if eyebrow:
        flow.append(_p(str(eyebrow).upper(), eyebrow_style))
    flow.append(_p(title_text, title_style))
    if subtitle:
        flow.append(_p(subtitle, sub_style))
    t = Table([[flow]], colWidths=[width])
    t.setStyle(
        TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.8, PDF_COLORS["line"]),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    return t


def _comparison_card(
    label: str,
    printed: Any,
    measured: Any,
    delta: Any,
    state: str,
    *,
    width: float = 80 * mm,
) -> Table:
    tone = _status_tone(state)
    accent, soft = _tone_pair(tone)
    label_style = ParagraphStyle(
        "CmpLabel",
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.6,
        textColor=PDF_COLORS["navy"],
    )
    cap_style = ParagraphStyle(
        "CmpCap",
        fontName="Helvetica-Bold",
        fontSize=5.7,
        leading=6.7,
        textColor=PDF_COLORS["muted"],
    )
    val_style = ParagraphStyle(
        "CmpVal",
        fontName="Helvetica-Bold",
        fontSize=9.0,
        leading=10.8,
        textColor=PDF_COLORS["ink"],
    )
    delta_style = ParagraphStyle(
        "CmpDelta",
        fontName="Helvetica",
        fontSize=6.1,
        leading=7.3,
        textColor=accent,
    )
    top = Table(
        [[_p(label, label_style), _mini_badge(state, tone=tone)]],
        colWidths=[width - 37 * mm, 34 * mm],
    )
    top.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ])
    )
    vals = Table(
        [
            [_p("EQUIPO", cap_style), _p("MEDCALC", cap_style)],
            [_p(printed, val_style), _p(measured, val_style)],
        ],
        colWidths=[(width - 14) / 2.0, (width - 14) / 2.0],
    )
    vals.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    flow = [top, vals, Spacer(1, 1.2 * mm), _p(f"Delta: {delta}", delta_style)]
    t = Table([[flow]], colWidths=[width])
    t.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.55, PDF_COLORS["line"]),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, accent),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    return t


def _coverage_bar(lead: str, fraction: float, *, width: float = 78 * mm) -> Drawing:
    frac = min(1.0, max(0.0, float(fraction or 0.0)))
    pct = 100.0 * frac
    tone = "green" if frac >= 0.70 else "amber" if frac >= 0.40 else "red"
    accent, soft = _tone_pair(tone)
    h = 10.5 * mm
    d = Drawing(width, h)
    d.add(String(0, h - 9, lead, fontName="Helvetica-Bold", fontSize=7.2, fillColor=PDF_COLORS["ink"]))
    d.add(String(width, h - 9, f"{pct:.1f}%", fontName="Helvetica-Bold", fontSize=6.7, fillColor=accent, textAnchor="end"))
    bar_y = 3.0
    bar_h = 4.0
    d.add(Rect(0, bar_y, width, bar_h, fillColor=soft, strokeColor=PDF_COLORS["line"], strokeWidth=0.35))
    if frac > 0:
        d.add(Rect(0, bar_y, width * frac, bar_h, fillColor=accent, strokeColor=None))
    return d


def _repol_card(
    lead: str,
    item: Dict[str, Any],
    *,
    width: float = 38 * mm,
) -> Table:
    evaluable = bool(item.get("evaluable"))
    tone = "teal" if evaluable else "amber"
    accent, soft = _tone_pair(tone)
    lead_style = ParagraphStyle(
        "RepolLead",
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=9.5,
        textColor=PDF_COLORS["navy"],
    )
    cap = ParagraphStyle(
        "RepolCap",
        fontName="Helvetica-Bold",
        fontSize=5.5,
        leading=6.5,
        textColor=PDF_COLORS["muted"],
    )
    val = ParagraphStyle(
        "RepolVal",
        fontName="Helvetica-Bold",
        fontSize=7.3,
        leading=8.6,
        textColor=PDF_COLORS["ink"],
    )
    st_value = _finite(item.get("st_mv"))
    st = _metric(st_value, " mV", 3)
    tv = _metric(item.get("t_mv"), " mV", 3)
    if not evaluable or st_value is None:
        st_badge = "NO EVALUABLE"
        st_badge_tone = "amber"
    elif st_value > 0.10:
        st_badge = "ELEVACION ST"
        st_badge_tone = "amber"
    elif st_value < -0.10:
        st_badge = "DEPRESION ST"
        st_badge_tone = "amber"
    else:
        st_badge = "ST SIN DESVIACION >0.10 mV"
        st_badge_tone = "teal"
    vals = Table(
        [
            [_p("ST", cap), _p("T", cap)],
            [_p(st, val), _p(tv, val)],
        ],
        colWidths=[(width - 12) / 2.0, (width - 12) / 2.0],
    )
    vals.setStyle(TableStyle([
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    flow = [
        _p(lead, lead_style),
        vals,
        _mini_badge(st_badge, tone=st_badge_tone),
    ]
    t = Table([[flow]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
        ("LINEBEFORE", (0, 0), (0, -1), 2.0, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _r27_probability_card(
    module: str,
    probability: float,
    *,
    width: float = 51 * mm,
) -> Table:
    p = min(1.0, max(0.0, float(probability)))
    tone = "teal" if p >= 0.70 else "blue"
    accent, soft = _tone_pair(tone)
    title_style = ParagraphStyle(
        "R27CardTitle",
        fontName="Helvetica-Bold",
        fontSize=6.4,
        leading=7.6,
        textColor=PDF_COLORS["navy"],
        spaceAfter=2,
    )
    value_style = ParagraphStyle(
        "R27CardValue",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=17,
        textColor=accent,
    )
    sub_style = ParagraphStyle(
        "R27CardSub",
        fontName="Helvetica",
        fontSize=5.7,
        leading=6.8,
        textColor=PDF_COLORS["muted"],
    )
    flow = [
        _p(module, title_style),
        _p(f"{p:.2f}", value_style),
        _p("SCORE DE MODELO - NO HALLAZGO MEDIDO", sub_style),
    ]
    t = Table([[flow]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), soft),
        ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
        ("LINEBEFORE", (0, 0), (0, -1), 2.4, accent),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
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
    - the reconstructed digitalized ECG and native rhythm strip,
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
    evidence_by_lead = structured_report.get("evidence_by_lead") or {}
    rhythm_evidence_override = structured_report.get("rhythm_evidence") or {}
    rhythm_signal_source = structured_report.get("rhythm_signal_source")
    digital_calibration = (
        structured_report.get("calibration")
        or signal.get("calibration")
        or {}
    )
    sampling_rate_hz = int(structured_report.get("sampling_rate_hz") or 500)
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

    report_text = _ascii(final_report.get("text") or "").strip()
    report_fields = _clean_report_fields(report_text)
    layout = signal.get("layout_name") or layout_detector.get("layout") or "-"
    layout_router = (
        signal.get("layout_hypothesis_router")
        or digitizer.get("layout_router")
        or {}
    )
    confidence = _finite(layout_router.get("selected_score"))
    if confidence is None:
        confidence = _finite(layout_detector.get("confidence"))
    observed = signal.get("observed_fraction_by_lead") or {}
    min_coverage = float(signal.get("min_observed_fraction") or 0.0)

    rhythm_label = (
        rhythm_screen.get("label")
        or report_fields.get("RITMO")
        or "NO EVALUABLE"
    )
    rhythm_tone = (
        "amber"
        if "NO EVALUABLE" in str(rhythm_label).upper()
        else "teal"
    )
    r27_status = (
        "R27-TILED"
        if tiled
        else "R27 ACTIVO"
        if isinstance(r27_payload, dict)
        else "NO EJECUTADO"
    )
    r27_tone = "teal" if isinstance(r27_payload, dict) else "amber"

    brand_style = ParagraphStyle(
        "BrandV4",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=20,
        textColor=PDF_COLORS["white"],
        spaceAfter=1,
    )
    brand_sub = ParagraphStyle(
        "BrandSubV4",
        fontName="Helvetica",
        fontSize=7.4,
        leading=9,
        textColor=colors.HexColor("#D8E8EE"),
    )
    meta_label = ParagraphStyle(
        "MetaLabelV4",
        fontName="Helvetica-Bold",
        fontSize=5.8,
        leading=6.8,
        textColor=PDF_COLORS["muted"],
    )
    meta_value = ParagraphStyle(
        "MetaValueV4",
        fontName="Helvetica-Bold",
        fontSize=7.1,
        leading=8.5,
        textColor=PDF_COLORS["ink"],
    )
    note_style = ParagraphStyle(
        "NoteV4",
        fontName="Helvetica",
        fontSize=6.7,
        leading=8.4,
        textColor=PDF_COLORS["muted"],
    )

    header_left = [
        _p("MEDCALC CLINICO", brand_style),
        _p("Informe electrocardiografico automatizado - foto/PDF", brand_sub),
        Spacer(1, 1.5 * mm),
        _mini_badge("MODO INVESTIGACION", tone="teal"),
    ]
    header = Table(
        [[header_left, qr]],
        colWidths=[128 * mm, 34 * mm],
        rowHeights=[37 * mm],
        hAlign="LEFT",
    )
    header.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["navy"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (0, 0), 9),
            ("RIGHTPADDING", (0, 0), (0, 0), 8),
            ("TOPPADDING", (0, 0), (0, 0), 7),
            ("BOTTOMPADDING", (0, 0), (0, 0), 7),
            ("LEFTPADDING", (1, 0), (1, 0), 2),
            ("RIGHTPADDING", (1, 0), (1, 0), 4),
            ("TOPPADDING", (1, 0), (1, 0), 3),
            ("BOTTOMPADDING", (1, 0), (1, 0), 3),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ])
    )

    study_meta = Table(
        [[
            [
                _p("ID ESTUDIO", meta_label),
                _p(study_id, meta_value),
            ],
            [
                _p("ARCHIVO", meta_label),
                _p(source_name or "-", meta_value),
            ],
            [
                _p("PACIENTE RUNTIME", meta_label),
                _p(
                    (
                        ("-" if age is None else f"{float(age):.0f} anos")
                        + " | sexo "
                        + str(sex_code if sex_code is not None else "-")
                    ),
                    meta_value,
                ),
            ],
            [
                _p("GENERADO UTC", meta_label),
                _p(generated, meta_value),
            ],
        ]],
        colWidths=[42 * mm, 48 * mm, 38 * mm, 38 * mm],
    )
    study_meta.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["paper"]),
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, PDF_COLORS["line"]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ])
    )

    story = [
        header,
        Spacer(1, 3 * mm),
        study_meta,
        Spacer(1, 4 * mm),
    ]

    if preview:
        preview_box = Table(
            [[_scaled_image(preview, 162 * mm, 72 * mm)]],
            colWidths=[166 * mm],
        )
        preview_box.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.55, PDF_COLORS["line"]),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story += [
            _section_label(
                "Trazado fuente",
                eyebrow="Documento original",
                subtitle=(
                    "La imagen fuente se conserva como referencia visual; las mediciones "
                    "se recalculan sobre la senal digitalizada."
                ),
            ),
            Spacer(1, 2 * mm),
            preview_box,
            Spacer(1, 2 * mm),
        ]

    story += [
        _section_label(
            "Resumen de lectura",
            eyebrow="Primera vista",
            subtitle=(
                "Sintesis visual de los hallazgos automatizados, su calidad tecnica "
                "y el estado de los motores."
            ),
        ),
        Spacer(1, 2 * mm),
        _text_panel(
            "Ritmo automatizado",
            rhythm_label,
            tone=rhythm_tone,
        ),
        Spacer(1, 2.5 * mm),
    ]

    summary_cards = Table(
        [[
            _info_card(
                "Frecuencia",
                _metric(motor.get("heart_rate_bpm"), " LPM"),
                subtitle=(
                    "Equipo: " + _metric(machine.get("heart_rate_bpm"), " LPM")
                ),
                width=39 * mm,
                tone="teal",
            ),
            _info_card(
                "QRS",
                _metric(motor.get("qrs_ms"), " ms"),
                subtitle=(
                    "Equipo: " + _metric(machine.get("qrs_ms"), " ms")
                ),
                width=39 * mm,
                tone=(
                    "red"
                    if _finite(machine.get("qrs_ms")) is not None
                    and _finite(motor.get("qrs_ms")) is not None
                    and abs(float(machine.get("qrs_ms")) - float(motor.get("qrs_ms"))) > 20
                    else "blue"
                ),
            ),
            _info_card(
                "Eje QRS",
                _metric(motor.get("axis_deg"), " deg"),
                subtitle=(
                    "Equipo: " + _metric(machine.get("qrs_axis_deg"), " deg")
                ),
                width=39 * mm,
                tone="blue",
            ),
            _info_card(
                "Estado R27",
                r27_status,
                subtitle=(
                    "Probability-only"
                    if isinstance(r27_payload, dict)
                    else "Sin probabilidades para este registro"
                ),
                width=39 * mm,
                tone=r27_tone,
                value_size=10.5,
            ),
        ]],
        colWidths=[41.5 * mm] * 4,
    )
    summary_cards.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(summary_cards)

    conclusion_text = (
        report_fields.get("CONCLUSION")
        or report_fields.get("CONCLUSIÓN")
        or report_fields.get("IDX")
        or "No fue posible generar una conclusion estructurada."
    )
    story += [
        Spacer(1, 3 * mm),
        _text_panel(
            "Conclusion automatizada",
            conclusion_text,
            tone="navy",
        ),
        Spacer(1, 3 * mm),
    ]

    quality_cards = Table(
        [[
            _info_card(
                "Layout",
                layout,
                subtitle=(
                    f"Confianza {100*confidence:.1f}%"
                    if confidence is not None
                    else "Confianza no disponible"
                ),
                width=52 * mm,
                tone="blue",
            ),
            _info_card(
                "Cobertura minima",
                f"{100*min_coverage:.1f}%",
                subtitle="Menor cobertura observada entre las 12 derivaciones",
                width=52 * mm,
                tone=(
                    "green" if min_coverage >= 0.70
                    else "amber" if min_coverage >= 0.40
                    else "red"
                ),
            ),
            _info_card(
                "Fuente temporal",
                (
                    rhythm_signal_source
                    or rhythm.get("signal_source")
                    or "NO EVALUABLE"
                ),
                subtitle=(
                    "Strip observado: "
                    + (
                        str(signal.get("rhythm_strip_lead"))
                        if signal.get("rhythm_strip_observed")
                        else "NO"
                    )
                ),
                width=52 * mm,
                tone=(
                    "teal"
                    if signal.get("rhythm_strip_observed")
                    else "amber"
                ),
                value_size=8.6,
            ),
        ]],
        colWidths=[55.3 * mm] * 3,
    )
    quality_cards.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(quality_cards)

    if digital_calibration:
        speed = _finite(digital_calibration.get("speed_mm_per_s"))
        gain = _finite(digital_calibration.get("gain_mm_per_mv"))
        cal_conf = _finite(digital_calibration.get("confidence"))
        cal_parts = []
        if speed is not None:
            cal_parts.append(
                f"Velocidad {speed:g} mm/s"
                + (
                    " (asumida)"
                    if digital_calibration.get("speed_assumed")
                    else " (detectada)"
                )
            )
        if gain is not None:
            cal_parts.append(
                f"Ganancia {gain:g} mm/mV"
                + (
                    " (asumida)"
                    if digital_calibration.get("gain_assumed")
                    else " (detectada)"
                )
            )
        gx = _finite(digital_calibration.get("mm_per_pixel_x"))
        gy = _finite(digital_calibration.get("mm_per_pixel_y"))
        if gx is not None and gy is not None:
            cal_parts.append(
                f"Grid {gx:.4f} mm/px horizontal | {gy:.4f} mm/px vertical"
            )
        if cal_conf is not None:
            cal_parts.append(f"Confianza de calibración {cal_conf:.2f}")
        if cal_parts:
            story += [
                Spacer(1, 2 * mm),
                _text_panel(
                    "Calibración física de la señal digital",
                    " | ".join(cal_parts),
                    tone=(
                        "teal"
                        if cal_conf is not None and cal_conf >= 0.80
                        else "amber"
                    ),
                    compact=True,
                ),
            ]

    # ------------------------------------------------------------------
    # Page 2 - structured interpretation and measurement concordance.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "Interpretacion estructurada",
            eyebrow="Lectura didactica",
            subtitle=(
                "Cada bloque separa lo que informa el motor, lo que imprimio el equipo "
                "y las discordancias que requieren revision."
            ),
        ),
        Spacer(1, 3 * mm),
    ]

    interpretation_items = [
        ("Ritmo", report_fields.get("RITMO") or rhythm_label, rhythm_tone),
        ("Frecuencia cardiaca", report_fields.get("FC") or _metric(motor.get("heart_rate_bpm"), " LPM"), "teal"),
        ("Eje", report_fields.get("EJE") or _metric(motor.get("axis_deg"), " deg"), "blue"),
        ("Segmento PR", report_fields.get("SEGMENTO PR") or "NO EVALUABLE", "blue"),
        ("Complejo QRS", report_fields.get("COMPLEJO QRS") or _metric(motor.get("qrs_ms"), " ms"), "blue"),
        ("QT / QTc", report_fields.get("QT/QTC") or report_fields.get("QT/QTc") or "NO EVALUABLE", "blue"),
        ("Segmento ST", report_fields.get("SEGMENTO ST") or "NO EVALUABLE", "blue"),
        ("Onda T", report_fields.get("ONDA T") or "NO EVALUABLE", "blue"),
    ]
    interpretation_cards = []
    for i in range(0, len(interpretation_items), 2):
        row = []
        for label_text, value_text, tone in interpretation_items[i:i+2]:
            row.append(
                _text_panel(
                    label_text,
                    value_text,
                    width=80 * mm,
                    tone=tone,
                    compact=True,
                )
            )
        if len(row) == 1:
            row.append("")
        interpretation_cards.append(row)
    interp_table = Table(
        interpretation_cards,
        colWidths=[83 * mm, 83 * mm],
    )
    interp_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(interp_table)

    story += [
        Spacer(1, 2 * mm),
        _section_label(
            "Equipo vs motor MEDCALC",
            eyebrow="Concordancia tecnica",
            subtitle=(
                "Las discordancias no se ocultan. La impresion del equipo se usa como "
                "referencia comparativa, no como verdad unica."
            ),
        ),
        Spacer(1, 2.5 * mm),
    ]

    pr_machine = None if machine.get("pr_printed_ms") == 0 else machine.get("pr_ms")
    overlap = [
        ("FC", machine.get("heart_rate_bpm"), motor.get("heart_rate_bpm"), "LPM", 10),
        ("PR", pr_machine, motor.get("pr_ms"), "ms", 30),
        ("QRS", machine.get("qrs_ms"), motor.get("qrs_ms"), "ms", 20),
        ("QT", machine.get("qt_ms"), motor.get("qt_ms"), "ms", 40),
        ("QTc", machine.get("qtc_ms"), motor.get("qtc_bazett_ms"), "ms", 40),
        ("Eje QRS", machine.get("qrs_axis_deg"), motor.get("axis_deg"), "deg", 20),
    ]
    comparison_cards = []
    for name, printed, measured, unit, tol in overlap:
        pval = _finite(printed)
        mval = _finite(measured)
        delta = abs(pval - mval) if pval is not None and mval is not None else None
        state = (
            "CONCORDANTE"
            if delta is not None and delta <= tol
            else "DISCORDANTE"
            if delta is not None
            else "NO COMPARABLE"
        )
        ptxt = (
            "NO CALCULABLE (0 ms)"
            if name == "PR" and machine.get("pr_printed_ms") == 0
            else _metric(pval, " " + unit)
        )
        mtxt = _metric(mval, " " + unit)
        dtxt = _metric(delta, " " + unit)
        comparison_cards.append(
            _comparison_card(
                name,
                ptxt,
                mtxt,
                dtxt,
                state,
                width=80 * mm,
            )
        )
    comp_rows = [
        [comparison_cards[i], comparison_cards[i + 1]]
        for i in range(0, len(comparison_cards), 2)
    ]
    comp_table = Table(comp_rows, colWidths=[83 * mm, 83 * mm])
    comp_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(comp_table)

    additional_cards = Table(
        [[
            _info_card("Eje P impreso", _metric(machine.get("p_axis_deg"), " deg"), width=39 * mm, tone="blue"),
            _info_card("Eje T impreso", _metric(machine.get("t_axis_deg"), " deg"), width=39 * mm, tone="blue"),
            _info_card("Velocidad", _metric(machine.get("speed_mm_per_s"), " mm/s"), width=39 * mm, tone="blue"),
            _info_card("Ganancia", _metric(machine.get("gain_mm_per_mV"), " mm/mV"), width=39 * mm, tone="blue"),
        ]],
        colWidths=[41.5 * mm] * 4,
    )
    additional_cards.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story += [
        Spacer(1, 1 * mm),
        additional_cards,
        Spacer(1, 3 * mm),
        _text_panel(
            "Como leer la concordancia",
            (
                "CONCORDANTE significa que la diferencia cae dentro de la tolerancia "
                "tecnica definida para ese parametro. DISCORDANTE significa que la "
                "diferencia supera esa tolerancia y debe revisarse junto con la senal "
                "digitalizada. NO COMPARABLE indica que una de las dos fuentes no pudo "
                "proporcionar una medicion util."
            ),
            tone="blue",
        ),
    ]

    # ------------------------------------------------------------------
    # Page 3 - native rhythm evidence and R27.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "Ritmo y evidencia temporal",
            eyebrow="Strip nativo",
            subtitle=(
                "El analisis de regularidad debe provenir de senal temporal observada, "
                "no de segmentos repetidos para compatibilidad R27."
            ),
        ),
        Spacer(1, 3 * mm),
    ]

    rhythm_lead = str(
        rhythm.get("lead")
        or signal.get("rhythm_strip_lead")
        or "II"
    )
    rhythm_evidence = (
        rhythm_evidence_override
        or evidence_by_lead.get(rhythm_lead)
        or {}
    )
    story.append(
        _rhythm_strip_drawing(
            rhythm_evidence,
            rhythm,
            lead=rhythm_lead,
            fs=sampling_rate_hz,
        )
    )

    rhythm_metrics = Table(
        [[
            _info_card("Duracion", _metric(rhythm.get("duration_s"), " s", 2), width=30 * mm, tone="blue"),
            _info_card("QRS", _metric(rhythm.get("r_count")), width=30 * mm, tone="blue"),
            _info_card("FC motor", _metric(rhythm.get("heart_rate_bpm"), " LPM"), width=30 * mm, tone="teal"),
            _info_card("RR CV crudo", _metric(rhythm.get("rr_cv"), "", 3), width=30 * mm, tone="blue"),
            _info_card("RR CV robusto", _metric(rhythm.get("rr_cv_robust"), "", 3), width=30 * mm, tone="blue"),
        ]],
        colWidths=[33 * mm] * 5,
    )
    rhythm_metrics.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story += [Spacer(1, 2 * mm), rhythm_metrics]

    basis = rhythm_screen.get("basis") or []
    story += [
        Spacer(1, 2.5 * mm),
        _text_panel(
            "Screening de ritmo",
            rhythm_label,
            tone=rhythm_tone,
        ),
        Spacer(1, 1.5 * mm),
        _text_panel(
            "Fundamento",
            (
                " | ".join(map(str, basis))
                if basis
                else "No hay fundamento temporal suficiente para clasificacion automatica."
            ),
            tone="blue",
            compact=True,
        ),
        Spacer(1, 1.5 * mm),
        _text_panel(
            "Fuente temporal usada",
            (
                str(rhythm_signal_source or rhythm.get("signal_source") or "NO EVALUABLE")
                + " | "
                + str(rhythm_evidence.get("pdf_trace_source") or "senal observada")
            ),
            tone="blue",
            compact=True,
        ),
    ]

    story += [
        Spacer(1, 4 * mm),
        _section_label(
            "Motores R27",
            eyebrow="Scores de modelos - no equivalen a mediciones directas",
            subtitle=(
                "R27 permanece probability-only. Un score alto puede ser discordante "
                "con la morfologia medida sobre la senal digitalizada y nunca sustituye "
                "la medicion directa de ST, QRS, QT o ritmo."
            ),
        ),
        Spacer(1, 2.5 * mm),
    ]

    if isinstance(r27_payload, dict):
        modules = r27_payload.get("modules") or {}
        if tiled:
            story += [
                _text_panel(
                    "Advertencia R27-TILED",
                    (
                        "Una o mas derivaciones fueron extendidas hasta 10 s mediante "
                        "repeticion exacta del segmento observado. No se ha demostrado "
                        "equivalencia con un ECG real de 10 s x 12 derivaciones. Para "
                        "ritmo se prioriza la senal nativa observada."
                    ),
                    tone="amber",
                ),
                Spacer(1, 2 * mm),
            ]

            lead_prov = (adapter.get("provenance") or {}).get("lead_provenance") or {}
            prov_parts = []
            for lead in LEAD_ORDER:
                info = lead_prov.get(lead) or {}
                if info.get("mode") == "REAL_10S":
                    continue
                sec = _finite(info.get("source_seconds"))
                if sec is not None:
                    prov_parts.append(f"{lead}: {sec:.2f} s reales")
            if prov_parts:
                story += [
                    _text_panel(
                        "Proveniencia R27-TILED",
                        " | ".join(prov_parts),
                        tone="blue",
                        compact=True,
                    ),
                    Spacer(1, 2 * mm),
                ]

        st_model = modules.get("ST_ELEVATION") or {}
        st_model_score = _finite(st_model.get("probability"))
        st_depression_leads = list(repol.get("st_depression_leads") or [])
        st_elevation_leads = list(repol.get("st_elevation_leads") or [])
        if (
            st_model_score is not None
            and st_model_score >= 0.70
            and len(st_depression_leads) >= 2
            and len(st_depression_leads) > len(st_elevation_leads)
        ):
            measured_summary = (
                "depresion ST medida en "
                + ", ".join(st_depression_leads)
            )
            if st_elevation_leads:
                measured_summary += (
                    "; elevacion ST medida en "
                    + ", ".join(st_elevation_leads)
                )
            story += [
                _text_panel(
                    "Discordancia R27 vs medicion directa",
                    (
                        f"R27 ST_ELEVATION = {st_model_score:.2f}, pero la medicion "
                        f"directa sobre la senal muestra {measured_summary}. "
                        "El score R27 NO se interpreta como elevacion del ST."
                    ),
                    tone="amber",
                ),
                Spacer(1, 2 * mm),
            ]

        highlighted = []
        for key in sorted(modules):
            item = modules.get(key) or {}
            pval = _finite(item.get("probability"))
            not_interpretable = (
                str(item.get("interpretability") or "")
                == "NOT_INTERPRETABLE_R27_TILED"
            )
            if pval is None or pval < 0.70 or not_interpretable:
                continue
            highlighted.append((key, pval))

        if highlighted:
            cards = [
                _r27_probability_card(key, pval, width=51 * mm)
                for key, pval in highlighted[:9]
            ]
            rows = []
            for i in range(0, len(cards), 3):
                row = cards[i:i+3]
                while len(row) < 3:
                    row.append("")
                rows.append(row)
            r27_grid = Table(rows, colWidths=[55 * mm, 55 * mm, 55 * mm])
            r27_grid.setStyle(
                TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ])
            )
            story.append(r27_grid)
        else:
            story.append(
                _text_panel(
                    "Senales destacadas",
                    "Ningun modulo interpretable alcanzo score R27 >= 0.70.",
                    tone="blue",
                    compact=True,
                )
            )

        if not tiled:
            rhythm_probs = []
            for key in RHYTHM_MODULES:
                item = modules.get(key)
                if not isinstance(item, dict):
                    continue
                pval = _finite(item.get("probability"))
                if pval is not None:
                    rhythm_probs.append((key, pval))
            if rhythm_probs:
                story += [
                    Spacer(1, 2 * mm),
                    _text_panel(
                        "Perfil R27 de ritmo",
                        " | ".join(f"{k}: {p:.4f}" for k, p in rhythm_probs),
                        tone="blue",
                        compact=True,
                    ),
                ]

        story += [
            Spacer(1, 2 * mm),
            _text_panel(
                "Interpretacion de los scores",
                (
                    "El corte 0.70 es solo un filtro visual del informe. No constituye "
                    "un umbral diagnostico validado. Los scores R27 no reemplazan las "
                    "mediciones morfologicas directas; cualquier discordancia se marca "
                    "explicitamente. Los 35 scores crudos permanecen en el JSON de auditoria."
                ),
                tone="blue",
                compact=True,
            ),
        ]
    else:
        runtime_error = _short_runtime_error(r27_error)
        unavailable_reason = (
            runtime_error
            or signal.get("r27_tiled_rejection_reason")
            or "El registro no produjo una entrada compatible para R27."
        )
        story += [
            _text_panel(
                "R27 no ejecutado",
                unavailable_reason,
                tone="amber",
            ),
            Spacer(1, 2 * mm),
            _text_panel(
                "Que significa",
                (
                    "La ausencia de R27 no invalida la digitalizacion ni las mediciones "
                    "descriptivas del motor. Indica que el registro no cumplio el contrato "
                    "de entrada requerido por los motores probabilisticos."
                ),
                tone="blue",
                compact=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Page 4 - reconstructed 12-lead ECG.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "ECG digitalizado - senal reconstruida",
            eyebrow="Auditoria visual",
            subtitle=(
                "Cada panel usa solamente tramos observados de la centerline. Los huecos "
                "se conservan como huecos y nunca se rellenan con rectas."
            ),
        ),
        Spacer(1, 2 * mm),
        _text_panel(
            "Como leer esta pagina",
            (
                "La escala vertical se autoajusta por derivacion para facilitar la "
                "inspeccion de forma. La duracion indicada en cada panel corresponde a "
                "senal realmente observada, no a tiempo sintetico."
            ),
            tone="blue",
            compact=True,
        ),
        Spacer(1, 2 * mm),
    ]

    trace_panels = [
        _ecg_trace_panel(lead, evidence_by_lead.get(lead))
        for lead in LEAD_ORDER
    ]
    trace_rows = [
        [trace_panels[i], trace_panels[i + 1]]
        for i in range(0, 12, 2)
    ]
    trace_table = Table(
        trace_rows,
        colWidths=[84 * mm, 84 * mm],
        rowHeights=[29 * mm] * 6,
    )
    trace_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E9EEF1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(trace_table)

    # ------------------------------------------------------------------
    # Page 5 - quality by lead and repolarization.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "Calidad por derivacion",
            eyebrow="Cobertura observada",
            subtitle=(
                "Las barras muestran cuanto del intervalo canonico de cada derivacion "
                "esta sustentado por senal observada."
            ),
        ),
        Spacer(1, 3 * mm),
    ]

    coverage_drawings = [
        _coverage_bar(lead, float(observed.get(lead) or 0.0))
        for lead in LEAD_ORDER
    ]
    coverage_grid = Table(
        [
            [coverage_drawings[i], coverage_drawings[i + 1]]
            for i in range(0, 12, 2)
        ],
        colWidths=[82 * mm, 82 * mm],
    )
    coverage_grid.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(coverage_grid)

    per_lead = repol.get("per_lead") or {}
    story += [
        Spacer(1, 4 * mm),
        _section_label(
            "Repolarizacion por derivacion",
            eyebrow="ST y onda T",
            subtitle=(
                "Cada tarjeta muestra las mediciones disponibles. NO EVALUABLE significa "
                "que la senal observada no fue suficiente para publicar esa medicion."
            ),
        ),
        Spacer(1, 3 * mm),
    ]
    repol_cards = [
        _repol_card(lead, per_lead.get(lead) or {}, width=38 * mm)
        for lead in LEAD_ORDER
    ]
    repol_rows = [
        repol_cards[i:i+4]
        for i in range(0, 12, 4)
    ]
    repol_grid = Table(
        repol_rows,
        colWidths=[41.5 * mm] * 4,
    )
    repol_grid.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(repol_grid)

    # ------------------------------------------------------------------
    # Page 6 - representative complexes.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "Complejos representativos por derivacion",
            eyebrow="Evidencia morfologica",
            subtitle=(
                "Cada panel muestra la senal digitalizada que sustento el analisis. "
                "Cuando fue posible se centra un complejo alrededor de un QRS detectado; "
                "si no, se conserva el segmento observado."
            ),
        ),
        Spacer(1, 3 * mm),
    ]
    panels = [
        _ecg_panel(lead, evidence_by_lead.get(lead))
        for lead in LEAD_ORDER
    ]
    grid_rows = [
        [panels[i], panels[i + 1]]
        for i in range(0, 12, 2)
    ]
    ev_table = Table(
        grid_rows,
        colWidths=[84 * mm, 84 * mm],
        rowHeights=[39 * mm] * 6,
    )
    ev_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E9EEF1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ("LEFTPADDING", (0, 0), (-1, -1), 1),
            ("RIGHTPADDING", (0, 0), (-1, -1), 1),
            ("TOPPADDING", (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ])
    )
    story.append(ev_table)

    # ------------------------------------------------------------------
    # Page 7 - limitations and traceability appendix.
    # ------------------------------------------------------------------
    story += [
        PageBreak(),
        _section_label(
            "Limitaciones y trazabilidad",
            eyebrow="Auditoria final",
            subtitle=(
                "Esta pagina concentra las limitaciones metodologicas y los identificadores "
                "necesarios para reproducir o auditar el analisis."
            ),
        ),
        Spacer(1, 3 * mm),
    ]

    limitations = structured_report.get("limitations") or []
    if limitations:
        for i, item in enumerate(limitations, start=1):
            story += [
                _text_panel(
                    f"Limitacion {i}",
                    str(item),
                    tone="amber",
                    compact=True,
                ),
                Spacer(1, 1.5 * mm),
            ]
    else:
        story += [
            _text_panel(
                "Limitaciones",
                (
                    "Reporte descriptivo automatizado derivado de la senal reconstruida "
                    "desde foto/PDF. Los campos no demostrables se informan como NO EVALUABLE."
                ),
                tone="amber",
            ),
            Spacer(1, 2 * mm),
        ]

    story += [
        _text_panel(
            "Uso del documento",
            (
                "Este documento es una salida de investigacion y debe contrastarse con "
                "el ECG fuente y el contexto clinico. R27 permanece probability-only, "
                "sin thresholds desplegables y sin autorizacion de clasificacion binaria."
            ),
            tone="amber",
        ),
        Spacer(1, 3 * mm),
    ]

    audit_rows = [
        ["Version PDF", REPORT_VERSION],
        ["ID estudio", study_id],
        ["SHA-256 fuente", source_sha or "-"],
        ["Layout", layout],
        ["Modo R27", r27_mode],
        ["R27-TILED", "SI" if tiled else "NO"],
        ["Contrato senal", signal.get("native_signal_contract") or "-"],
        ["Digitizer commit", assets.get("source_commit") or "-"],
        ["Segmentation SHA-256", assets.get("segmentation_model_sha256") or "-"],
        ["Lead model SHA-256", assets.get("lead_model_sha256") or "-"],
    ]
    audit = Table(
        [
            [
                _p(k, meta_label),
                _p(v, note_style),
            ]
            for k, v in audit_rows
        ],
        colWidths=[43 * mm, 121 * mm],
    )
    audit.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["paper"]),
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#E6ECEF")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )

    qr_block = Table(
        [[
            [
                _p("QR DE TRAZABILIDAD", meta_label),
                _p(
                    (
                        "Contiene ID del estudio, SHA-256 de la fuente, pagina, "
                        "layout y modo R27. No contiene nombre del paciente."
                    ),
                    note_style,
                ),
                Spacer(1, 2 * mm),
                _p("Contenido QR:", meta_label),
                _p(qr_payload, note_style),
            ],
            qr,
        ]],
        colWidths=[126 * mm, 38 * mm],
    )
    qr_block.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PDF_COLORS["soft_blue"]),
            ("BOX", (0, 0), (-1, -1), 0.45, PDF_COLORS["line"]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])
    )

    story += [
        audit,
        Spacer(1, 4 * mm),
        qr_block,
    ]

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
