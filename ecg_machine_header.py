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



def _group_contiguous(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        cur = int(raw)
        if cur > prev + max_gap:
            groups.append((start, prev))
            start = cur
        prev = cur
    groups.append((start, prev))
    return groups


def _ocr_row_texts(img: Image.Image) -> list[list[str]]:
    """OCR the fixed measurement rows used by Bionet/EKG2000-style headers.

    Detection is based on horizontal dark-ink projections, not hard-coded pixel
    coordinates, so it survives PDF rasterization/resizing.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]

    x0, x1 = int(w * 0.18), int(w * 0.48)
    y0, y1 = 0, int(h * 0.145)
    roi = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    dark = gray < 95
    proj = dark.sum(axis=1)
    threshold = max(8, int(round(roi.shape[1] * 0.018)))
    candidate_rows = np.flatnonzero(proj > threshold)
    groups = _group_contiguous(candidate_rows, max_gap=2)

    # Keep plausible text bands and the first six: HR, PR, QRS, QT/QTc,
    # PRTaxis label, axis values.
    cleaned: list[tuple[int, int]] = []
    for a, b in groups:
        height = b - a + 1
        if 2 <= height <= max(60, int(h * 0.06)):
            cleaned.append((a, b))
    cleaned = cleaned[:6]

    import pytesseract

    output: list[list[str]] = []
    pad = max(4, int(round(h * 0.006)))
    for a, b in cleaned:
        aa = max(0, a - pad)
        bb = min(roi.shape[0], b + pad + 1)
        band = roi[aa:bb, :]
        band_gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
        texts: list[str] = []
        for thr in (80, 90, 100, 110):
            _, binary = cv2.threshold(band_gray, thr, 255, cv2.THRESH_BINARY)
            # Upscale a small line instead of OCRing the entire ECG page.
            binary = cv2.resize(binary, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
            try:
                txt = pytesseract.image_to_string(
                    binary,
                    config="--oem 3 --psm 7",
                    lang="eng",
                ).strip()
            except Exception:
                txt = ""
            if txt:
                texts.append(txt)
        output.append(texts)
    return output


def _numeric_candidates(texts: list[str], *, signed: bool = False) -> list[int]:
    out: list[int] = []
    for txt in texts:
        t = (
            txt.replace("O", "0").replace("o", "0")
            .replace("I", "1").replace("l", "1").replace("L", "1")
        )
        # Join OCR-spaced digits such as "9 2" -> "92".
        t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
        pattern = r"(?<!\d)[+-]?\d{1,4}(?!\d)" if signed else r"(?<!\d)\d{1,4}(?!\d)"
        for token in re.findall(pattern, t):
            try:
                out.append(int(token))
            except Exception:
                pass
    return out


def _mode_plausible(values: list[int], low: int, high: int) -> int | None:
    vals = [int(v) for v in values if low <= int(v) <= high]
    if not vals:
        return None
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], abs(kv[0])))[0][0]


def _parse_rowwise_machine_panel(img: Image.Image) -> dict[str, Any]:
    rows = _ocr_row_texts(img)
    if len(rows) < 4:
        return {}

    hr = _mode_plausible(_numeric_candidates(rows[0]), 20, 300)
    pr = _mode_plausible(_numeric_candidates(rows[1]), 0, 400)
    qrs = _mode_plausible(_numeric_candidates(rows[2]), 20, 250)

    qt = qtc = None
    qt_pairs: list[tuple[int, int]] = []
    for txt in rows[3]:
        t = (
            txt.replace("O", "0").replace("o", "0")
            .replace("I", "1").replace("l", "1").replace("L", "1")
        )
        t = re.sub(r"(?<=\d)\s+(?=\d)", "", t)
        for a, b in re.findall(r"(\d{2,4})\s*/\s*(\d{2,4})", t):
            aa, bb = int(a), int(b)
            if 100 <= aa <= 700 and 100 <= bb <= 800:
                qt_pairs.append((aa, bb))
    if qt_pairs:
        # Most repeated pair across threshold variants.
        counts: dict[tuple[int, int], int] = {}
        for pair in qt_pairs:
            counts[pair] = counts.get(pair, 0) + 1
        qt, qtc = sorted(counts.items(), key=lambda kv: -kv[1])[0][0]

    p_axis = qrs_axis = t_axis = None
    if len(rows) >= 6:
        pairs: list[tuple[int, int]] = []
        for txt in rows[5]:
            vals = [v for v in _numeric_candidates([txt], signed=True) if -180 <= v <= 180]
            if len(vals) >= 2:
                pairs.append((vals[-2], vals[-1]))
        if pairs:
            abs_first = _mode_plausible([abs(a) for a, _ in pairs], 0, 180)
            abs_second = _mode_plausible([abs(b) for _, b in pairs], 0, 180)
            if abs_first is not None:
                near = [a for a, _ in pairs if abs(abs(a) - abs_first) <= 3]
                qrs_axis = -abs_first if any(a < 0 for a in near) else abs_first
            if abs_second is not None:
                near = [b for _, b in pairs if abs(abs(b) - abs_second) <= 3]
                t_axis = -abs_second if any(b < 0 for b in near) else abs_second

    return {
        "heart_rate_bpm": hr,
        "pr_printed_ms": pr,
        "qrs_ms": qrs,
        "qt_ms": qt,
        "qtc_ms": qtc,
        "p_axis_deg": p_axis,
        "qrs_axis_deg": qrs_axis,
        "t_axis_deg": t_axis,
        "row_ocr": rows,
    }


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

    rowwise = _parse_rowwise_machine_panel(img)

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
            r"QRS.{0,12}?Dur.{0,20}?([0-9OIlL]{2,3})\s*(?:m?s|s)",
            r"QRS.{0,12}?(?:Duration|Dur).{0,20}?([0-9OIlL]{2,3})",
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

    # The row-wise panel parser is more reliable on red-grid scans than generic
    # whole-header OCR. Use each row-wise value only when it passed a physiological
    # range check; otherwise retain the generic result.
    hr = rowwise.get("heart_rate_bpm") if rowwise.get("heart_rate_bpm") is not None else hr
    pr_printed = rowwise.get("pr_printed_ms") if rowwise.get("pr_printed_ms") is not None else pr_printed
    qrs = rowwise.get("qrs_ms") if rowwise.get("qrs_ms") is not None else qrs
    qt = rowwise.get("qt_ms") if rowwise.get("qt_ms") is not None else qt
    qtc = rowwise.get("qtc_ms") if rowwise.get("qtc_ms") is not None else qtc
    if rowwise.get("qrs_axis_deg") is not None:
        axes["qrs_axis_deg"] = rowwise["qrs_axis_deg"]
    if rowwise.get("t_axis_deg") is not None:
        axes["t_axis_deg"] = rowwise["t_axis_deg"]

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
        "rowwise_panel_ocr": rowwise.get("row_ocr"),
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
