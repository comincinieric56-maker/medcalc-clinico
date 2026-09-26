from __future__ import annotations

import io
import math
import re
from functools import lru_cache
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageOps


def _source_image(source_name: str, source_bytes: bytes, pdf_page_index: int) -> Image.Image:
    name = (source_name or "").lower()
    if name.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=source_bytes, filetype="pdf")
        try:
            if doc.page_count < 1:
                raise ValueError("PDF sin páginas.")
            idx = min(max(int(pdf_page_index), 0), int(doc.page_count) - 1)
            page = doc.load_page(idx)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(220 / 72, 220 / 72), alpha=False)
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        finally:
            doc.close()

    return ImageOps.exif_transpose(Image.open(io.BytesIO(source_bytes))).convert("RGB")


def _bounded(img: Image.Image, max_width: int = 2200) -> Image.Image:
    if img.width <= max_width:
        return img
    ratio = max_width / float(img.width)
    return img.resize((max_width, max(1, int(round(img.height * ratio)))))


def _ocr_variants(img: Image.Image) -> list[Image.Image]:
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    # Variant A: simple dark-pixel threshold. On ECG paper this preserves the
    # black printer text even when some red grid remains; Tesseract handles the
    # residual grid better than aggressive morphology.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, th110 = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    _, th135 = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)

    # Variant B: suppress pixels that are strongly chromatic red and keep dark
    # near-neutral ink. Useful for some scanners where the grid is saturated.
    maxc = np.max(rgb, axis=2)
    minc = np.min(rgb, axis=2)
    neutral_dark = ((maxc < 180) & ((maxc - minc) < 55)).astype(np.uint8) * 255
    neutral = 255 - neutral_dark

    return [
        Image.fromarray(th110, mode="L"),
        Image.fromarray(th135, mode="L"),
        Image.fromarray(neutral, mode="L"),
    ]


def _ocr(img: Image.Image) -> str:
    import pytesseract

    texts = []
    for processed in _ocr_variants(img):
        for psm in (6, 11):
            try:
                txt = pytesseract.image_to_string(
                    processed,
                    config=f"--oem 3 --psm {psm}",
                    lang="eng",
                )
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
    return "\n".join(texts)


def _ocr_int(token: str) -> int:
    cleaned = (
        str(token)
        .replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("L", "1")
    )
    cleaned = re.sub(r"[^0-9+-]", "", cleaned)
    return int(cleaned)


def _first_int(patterns, text: str) -> int | None:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I | re.S)
        if m:
            try:
                return _ocr_int(m.group(1))
            except Exception:
                pass
    return None


def _first_pair(patterns, text: str) -> tuple[int | None, int | None]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I | re.S)
        if m:
            try:
                return _ocr_int(m.group(1)), _ocr_int(m.group(2))
            except Exception:
                pass
    return None, None


def _parse_axes(text: str) -> Dict[str, int | None]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        compact = re.sub(r"\s+", "", line).lower()
        if "prtaxis" in compact or ("prt" in compact and "axis" in compact):
            block = " ".join(lines[i:i+3])
            tail = re.split(r"axis\s*:?", block, maxsplit=1, flags=re.I)
            tail = tail[-1] if tail else block
            nums = [int(v) for v in re.findall(r"(?<!\d)[+-]?\d{1,3}(?!\d)", tail)]
            star = "*" in tail
            # P/R/T axes. When the machine cannot calculate P it commonly prints
            # an asterisk followed by QRS and T axes.
            if star and len(nums) >= 2:
                return {"p_axis_deg": None, "qrs_axis_deg": nums[0], "t_axis_deg": nums[1]}
            if len(nums) >= 3:
                return {"p_axis_deg": nums[0], "qrs_axis_deg": nums[1], "t_axis_deg": nums[2]}
            if len(nums) == 2:
                return {"p_axis_deg": None, "qrs_axis_deg": nums[0], "t_axis_deg": nums[1]}
    return {"p_axis_deg": None, "qrs_axis_deg": None, "t_axis_deg": None}


def _axis_category(deg: int | float | None) -> str | None:
    if deg is None:
        return None
    d = float(deg)
    if -30 <= d <= 90:
        return "EJE NO DESVIADO"
    if -90 <= d < -30:
        return "DESVIACIÓN IZQUIERDA"
    if 90 < d <= 180:
        return "DESVIACIÓN DERECHA"
    return "EJE EXTREMO"


@lru_cache(maxsize=8)
def extract_machine_measurements(
    source_name: str,
    source_bytes: bytes,
    pdf_page_index: int = 0,
) -> Dict[str, Any]:
    img = _bounded(_source_image(source_name, source_bytes, pdf_page_index))

    top = img.crop((0, 0, img.width, max(1, int(img.height * 0.18))))
    bottom = img.crop((0, max(0, int(img.height * 0.82)), img.width, img.height))

    header_text = _ocr(top)
    footer_text = _ocr(bottom)
    text = header_text + "\n" + footer_text

    num = r"([0-9OIlL]{1,4})"

    hr = _first_int(
        [
            rf"Heart\s*Rate.{{0,18}}?{num}\s*(?:b?pm|pm)",
            rf"Heart\s*Rate.{{0,18}}?{num}",
            rf"\bHR\b.{{0,12}}?{num}",
        ],
        text,
    )
    pr_printed = _first_int(
        [
            rf"\bPR\b.{{0,10}}?Int.{{0,18}}?{num}\s*(?:m?s|s)",
            rf"\bPR\b.{{0,10}}?(?:Interval|Int).{{0,18}}?{num}",
            rf"TPR\s*I\s*nt.{{0,10}}?{num}\s*(?:m?s|s)",
        ],
        text,
    )
    qrs = _first_int(
        [
            rf"QRS.{{0,12}}?Dur.{{0,20}}?{num}\s*(?:m?s|s)",
            rf"QRS.{{0,12}}?(?:Duration|Dur).{{0,20}}?{num}",
        ],
        text,
    )
    qt, qtc = _first_pair(
        [
            r"QT\s*/\s*QT[cC].{0,18}?([0-9OIlL]{2,4})\s*/\s*([0-9OIlL]{2,4})\s*(?:m?s|s)",
            r"QT\s*/\s*QT[cC].{0,18}?([0-9OIlL]{2,4})\s*/\s*([0-9OIlL]{2,4})",
        ],
        text,
    )

    axes = _parse_axes(text)

    gain_match = re.search(r"(\d+(?:[\.,]\d+)?)\s*mm\s*/\s*mV", text, flags=re.I)
    speed_match = re.search(
        r"(\d+(?:[\.,]\d+)?)\s*mm\s*/\s*(?:s|sec|seg)",
        text,
        flags=re.I,
    )

    gain = float(gain_match.group(1).replace(",", ".")) if gain_match else None
    speed = float(speed_match.group(1).replace(",", ".")) if speed_match else None

    parsed_count = sum(
        v is not None
        for v in [hr, pr_printed, qrs, qt, qtc, axes["qrs_axis_deg"], axes["t_axis_deg"]]
    )

    return {
        "detected": bool(parsed_count >= 2),
        "source": "machine_printed_header_ocr",
        "heart_rate_bpm": hr,
        # A printed PR=0 is not a physiological 0 ms interval. It means the
        # device did not calculate PR; preserve the raw printed value separately.
        "pr_printed_ms": pr_printed,
        "pr_ms": None if pr_printed in (None, 0) else pr_printed,
        "pr_status": (
            "NOT_CALCULATED_BY_MACHINE"
            if pr_printed == 0
            else "PRINTED_VALUE" if pr_printed is not None else "NOT_FOUND"
        ),
        "qrs_ms": qrs,
        "qt_ms": qt,
        "qtc_ms": qtc,
        **axes,
        "qrs_axis_category": _axis_category(axes["qrs_axis_deg"]),
        "gain_mm_per_mV": gain,
        "speed_mm_per_s": speed,
        "parsed_field_count": int(parsed_count),
        "ocr_header_text": header_text,
        "ocr_footer_text": footer_text,
    }


def compose_final_report(
    machine: Dict[str, Any] | None,
    structured_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    machine = machine or {}
    structured_report = structured_report or {}
    formatted = structured_report.get("formatted") or {}
    report_error = structured_report.get("error")

    trusted_signal_report = not report_error and bool(formatted.get("text"))

    rhythm = (
        str(formatted.get("rhythm_text") or "NO EVALUABLE")
        if trusted_signal_report
        else "NO EVALUABLE"
    )
    st = (
        str(formatted.get("st_text") or "NO EVALUABLE")
        if trusted_signal_report
        else "NO EVALUABLE"
    )
    twave = (
        str(formatted.get("t_text") or "NO EVALUABLE")
        if trusted_signal_report
        else "NO EVALUABLE"
    )
    ectopy = (
        str(formatted.get("ectopy_text") or "EXTRASISTOLIA NO EVALUABLE")
        if trusted_signal_report
        else "EXTRASISTOLIA NO EVALUABLE"
    )

    hr = machine.get("heart_rate_bpm")
    hr_text = f"{int(hr)} LPM (IMPRESO POR EL EQUIPO)" if hr is not None else (
        str(formatted.get("heart_rate_text") or "NO EVALUABLE")
        if trusted_signal_report else "NO EVALUABLE"
    )

    qrs_axis = machine.get("qrs_axis_deg")
    if qrs_axis is not None:
        cat = machine.get("qrs_axis_category")
        axis_text = f"{int(qrs_axis)}° ({cat})" if cat else f"{int(qrs_axis)}°"
    else:
        axis_text = str(formatted.get("axis_text") or "NO EVALUABLE") if trusted_signal_report else "NO EVALUABLE"

    pr_printed = machine.get("pr_printed_ms")
    if pr_printed == 0:
        pr_text = "NO CALCULABLE POR EL EQUIPO (VALOR IMPRESO: 0 MS)"
    elif machine.get("pr_ms") is not None:
        p = int(machine["pr_ms"])
        qual = "NORMAL" if 120 <= p <= 200 else "PROLONGADO" if p > 200 else "CORTO"
        pr_text = f"{p} MS ({qual}; IMPRESO POR EL EQUIPO)"
    else:
        pr_text = str(formatted.get("pr_text") or "NO EVALUABLE") if trusted_signal_report else "NO EVALUABLE"

    qrs = machine.get("qrs_ms")
    if qrs is not None:
        q = int(qrs)
        qrs_text = f"{q} MS ({'NO PROLONGADO' if q < 120 else 'PROLONGADO'}; IMPRESO POR EL EQUIPO)"
    else:
        qrs_text = str(formatted.get("qrs_text") or "NO EVALUABLE") if trusted_signal_report else "NO EVALUABLE"

    qt = machine.get("qt_ms")
    qtc = machine.get("qtc_ms")
    qt_text = (
        f"{int(qt)}/{int(qtc)} MS (QT/QTc IMPRESO POR EL EQUIPO)"
        if qt is not None and qtc is not None
        else "NO EVALUABLE"
    )

    conclusion_bits = []
    if hr is not None:
        conclusion_bits.append(f"FC IMPRESA {int(hr)} LPM")
    if qrs is not None:
        conclusion_bits.append(f"QRS {int(qrs)} MS {'NO PROLONGADO' if int(qrs) < 120 else 'PROLONGADO'}")
    if qrs_axis is not None:
        conclusion_bits.append(f"EJE QRS {int(qrs_axis)}°")
    if qt is not None and qtc is not None:
        conclusion_bits.append(f"QT/QTc {int(qt)}/{int(qtc)} MS")
    if pr_printed == 0:
        conclusion_bits.append("PR NO CALCULABLE POR EL EQUIPO")

    if trusted_signal_report:
        if st != "NO EVALUABLE":
            conclusion_bits.append(st)
        if twave != "NO EVALUABLE":
            conclusion_bits.append(twave)

    conclusion = (
        "ELECTROCARDIOGRAMA CON " + ", ".join(conclusion_bits) + "."
        if conclusion_bits
        else "MEDICIONES AUTOMATIZADAS NO DISPONIBLES."
    )

    idx = "REVISIÓN MANUAL DEL RITMO"
    if hr is not None and qrs is not None:
        if int(hr) >= 100 and int(qrs) < 120:
            idx = "TAQUICARDIA DE COMPLEJO QRS ESTRECHO; DEFINIR MECANISMO DEL RITMO EN EL TRAZADO"
        elif int(hr) >= 100:
            idx = "TAQUICARDIA; DEFINIR MECANISMO DEL RITMO EN EL TRAZADO"

    lines = [
        f"RITMO: {rhythm}.",
        f"FC: {hr_text}.",
        f"EJE: {axis_text}.",
        f"SEGMENTO PR: {pr_text}.",
        f"COMPLEJO QRS: {qrs_text}.",
        f"QT/QTc: {qt_text}.",
        f"SEGMENTO ST: {st}.",
        f"ONDA T: {twave}.",
        f"{ectopy}.",
        f"CONCLUSIÓN: {conclusion}",
        f"IDX: {idx}.",
    ]

    return {
        "text": "\n".join(lines),
        "trusted_signal_report": bool(trusted_signal_report),
        "machine_measurements_used": bool(machine.get("detected")),
    }
