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



def _measurement_panel_image(
    source_name: str,
    source_bytes: bytes,
    pdf_page_index: int,
) -> Image.Image:
    """Render/crop only the machine measurement panel at high resolution.

    This preserves small minus signs and 2-digit values without keeping the
    entire ECG page at high DPI in RAM.
    """
    name = (source_name or "").lower()

    if name.endswith(".pdf"):
        import pymupdf

        doc = pymupdf.open(stream=source_bytes, filetype="pdf")
        try:
            if doc.page_count < 1:
                raise ValueError("PDF sin páginas.")
            idx = min(max(int(pdf_page_index), 0), int(doc.page_count) - 1)
            page = doc.load_page(idx)
            rect = page.rect
            clip = pymupdf.Rect(
                rect.x0 + rect.width * 0.18,
                rect.y0,
                rect.x0 + rect.width * 0.48,
                rect.y0 + rect.height * 0.145,
            )
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(300 / 72, 300 / 72),
                clip=clip,
                alpha=False,
            )
            return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        finally:
            doc.close()

    full = ImageOps.exif_transpose(Image.open(io.BytesIO(source_bytes))).convert("RGB")
    w, h = full.size
    panel = full.crop((int(w * 0.18), 0, int(w * 0.48), int(h * 0.145)))
    if panel.width > 3600:
        ratio = 3600 / float(panel.width)
        panel = panel.resize((3600, max(1, int(round(panel.height * ratio)))))
    return panel


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


def _ocr_row_texts(panel: Image.Image) -> list[list[str]]:
    """OCR the fixed measurement rows from a pre-cropped high-res panel."""
    rgb = np.asarray(panel.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    dark = gray < 95
    proj = dark.sum(axis=1)
    threshold = max(12, int(round(w * 0.018)))
    candidate_rows = np.flatnonzero(proj > threshold)
    groups = _group_contiguous(candidate_rows, max_gap=3)

    cleaned: list[tuple[int, int]] = []
    for a, b in groups:
        height = b - a + 1
        if 3 <= height <= max(120, int(h * 0.14)):
            cleaned.append((a, b))

    # First six text bands are HR, PR, QRS, QT/QTc, PRTaxis label, axis values.
    cleaned = cleaned[:6]

    import pytesseract

    output: list[list[str]] = []
    pad = max(6, int(round(h * 0.025)))
    for a, b in cleaned:
        aa = max(0, a - pad)
        bb = min(h, b + pad + 1)
        band_gray = gray[aa:bb, :]
        texts: list[str] = []
        for thr in (80, 90, 100, 110):
            _, binary = cv2.threshold(band_gray, thr, 255, cv2.THRESH_BINARY)
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



def _ocr_axis_value(
    panel: Image.Image,
    x0f: float,
    x1f: float,
) -> int | None:
    """Read one P/QRS/T axis cell from the printed PRTaxis row."""
    rgb = np.asarray(panel.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    roi = rgb[
        int(h * 0.68) : int(h * 0.87),
        int(w * x0f) : int(w * x1f),
    ]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    import pytesseract

    values: list[int] = []
    saw_minus = False
    for thr in (80, 90, 100, 110, 120, 130, 140, 150):
        _, binary = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        try:
            txt = pytesseract.image_to_string(
                binary,
                config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789-*+",
                lang="eng",
            ).strip()
        except Exception:
            txt = ""
        if not txt:
            continue
        saw_minus = saw_minus or ("-" in txt)
        for token in re.findall(r"[+-]?\d{1,3}", txt):
            try:
                value = int(token)
            except Exception:
                continue
            if -180 <= value <= 180:
                values.append(value)

    if not values:
        return None

    # Prefer repeated 2-3 digit magnitudes; isolated one-character OCR noise
    # is common on the red grid.
    abs_values = [abs(v) for v in values]
    multi = [v for v in abs_values if v >= 10]
    base_pool = multi if multi else abs_values
    counts: dict[int, int] = {}
    for value in base_pool:
        counts[value] = counts.get(value, 0) + 1
    magnitude = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    near = [v for v in values if abs(abs(v) - magnitude) <= 1]
    negative = any(v < 0 for v in near) or saw_minus
    return -magnitude if negative else magnitude


def _parse_rowwise_machine_panel(panel: Image.Image) -> dict[str, Any]:
    rows = _ocr_row_texts(panel)
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

    # Bionet/EKG2000 prints P, QRS and T axes in fixed adjacent cells.
    # Read those cells directly from the high-resolution panel rather than
    # inferring them from noisy whole-line OCR.
    p_axis = _ocr_axis_value(panel, 0.31, 0.40)
    qrs_axis = _ocr_axis_value(panel, 0.41, 0.54)
    t_axis = _ocr_axis_value(panel, 0.55, 0.66)

    # The P-axis cell contains an asterisk when P could not be calculated.
    # Treat spurious isolated OCR digits in that cell as missing if the PR
    # printed by the machine is 0 ms.
    if pr == 0:
        p_axis = None

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
    panel = _measurement_panel_image(source_name, source_bytes, pdf_page_index)

    top = img.crop((0, 0, img.width, max(1, int(img.height * 0.18))))
    bottom = img.crop((0, max(0, int(img.height * 0.82)), img.width, img.height))

    header_text = _ocr(top)
    footer_text = _ocr(bottom)
    text = header_text + "\n" + footer_text

    rowwise = _parse_rowwise_machine_panel(panel)

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
    r27_payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    machine = machine or {}
    structured_report = structured_report or {}
    formatted = structured_report.get("formatted") or {}
    rhythm = structured_report.get("rhythm") or {}
    rhythm_screen = structured_report.get("rhythm_screen") or {}
    axis = structured_report.get("axis") or {}
    repol = structured_report.get("repolarization") or {}
    motor = structured_report.get("measurement_summary") or {}
    report_error = structured_report.get("error")

    trusted_signal_report = not report_error and bool(formatted.get("text"))

    def _num(v):
        try:
            z = float(v)
            return z if math.isfinite(z) else None
        except Exception:
            return None

    def _pair_text(
        label: str,
        machine_value,
        motor_value,
        *,
        unit: str,
        tolerance: float,
        decimals: int = 0,
    ) -> tuple[str, dict]:
        mv = _num(machine_value)
        rv = _num(motor_value)
        fmt = f"{{:.{decimals}f}}"

        if mv is not None and rv is not None:
            delta = abs(mv - rv)
            concordant = bool(delta <= tolerance)
            text = (
                f"{fmt.format(mv)} {unit} (EQUIPO) / "
                f"{fmt.format(rv)} {unit} (MOTOR)"
            )
            if not concordant:
                text += f" · DISCORDANCIA {fmt.format(delta)} {unit}"
            return text, {
                "label": label,
                "machine": mv,
                "motor": rv,
                "delta": delta,
                "tolerance": tolerance,
                "concordant": concordant,
            }

        if mv is not None:
            return f"{fmt.format(mv)} {unit} (EQUIPO) / NO EVALUABLE (MOTOR)", {
                "label": label,
                "machine": mv,
                "motor": None,
                "delta": None,
                "tolerance": tolerance,
                "concordant": None,
            }

        if rv is not None:
            return f"NO DISPONIBLE (EQUIPO) / {fmt.format(rv)} {unit} (MOTOR)", {
                "label": label,
                "machine": None,
                "motor": rv,
                "delta": None,
                "tolerance": tolerance,
                "concordant": None,
            }

        return "NO EVALUABLE", {
            "label": label,
            "machine": None,
            "motor": None,
            "delta": None,
            "tolerance": tolerance,
            "concordant": None,
        }

    hr_text, hr_cmp = _pair_text(
        "FC",
        machine.get("heart_rate_bpm"),
        motor.get("heart_rate_bpm"),
        unit="LPM",
        tolerance=10.0,
    )

    qrs_text, qrs_cmp = _pair_text(
        "QRS",
        machine.get("qrs_ms"),
        motor.get("qrs_ms"),
        unit="MS",
        tolerance=20.0,
    )

    axis_text, axis_cmp = _pair_text(
        "EJE QRS",
        machine.get("qrs_axis_deg"),
        motor.get("axis_deg"),
        unit="°",
        tolerance=20.0,
    )

    pr_printed = machine.get("pr_printed_ms")
    motor_pr = _num(motor.get("pr_ms"))
    if pr_printed == 0:
        if motor_pr is not None:
            pr_text = (
                f"NO CALCULABLE POR EL EQUIPO (0 MS IMPRESO) / "
                f"{motor_pr:.0f} MS (MOTOR)"
            )
        else:
            pr_text = "NO CALCULABLE POR EL EQUIPO (0 MS IMPRESO) / NO EVALUABLE (MOTOR)"
        pr_cmp = {
            "label": "PR",
            "machine": None,
            "machine_printed_raw": 0,
            "motor": motor_pr,
            "delta": None,
            "tolerance": 30.0,
            "concordant": None,
        }
    else:
        pr_text, pr_cmp = _pair_text(
            "PR",
            machine.get("pr_ms"),
            motor_pr,
            unit="MS",
            tolerance=30.0,
        )

    qt_machine = _num(machine.get("qt_ms"))
    qtc_machine = _num(machine.get("qtc_ms"))
    qt_motor = _num(motor.get("qt_ms"))
    qtc_motor = _num(motor.get("qtc_bazett_ms"))

    qt_parts = []
    if qt_machine is not None or qtc_machine is not None:
        qt_parts.append(
            f"{qt_machine:.0f}/{qtc_machine:.0f} MS (QT/QTc EQUIPO)"
            if qt_machine is not None and qtc_machine is not None
            else "QT/QTc EQUIPO INCOMPLETO"
        )
    if qt_motor is not None or qtc_motor is not None:
        qt_parts.append(
            f"{qt_motor:.0f}/{qtc_motor:.0f} MS (QT/QTc MOTOR)"
            if qt_motor is not None and qtc_motor is not None
            else "QT/QTc MOTOR INCOMPLETO"
        )
    qt_text = " / ".join(qt_parts) if qt_parts else "NO EVALUABLE"

    comparisons = [hr_cmp, pr_cmp, qrs_cmp, axis_cmp]
    discrepancies = [
        c for c in comparisons
        if c.get("concordant") is False
    ]

    if trusted_signal_report:
        st = str(formatted.get("st_text") or "NO EVALUABLE")
        twave = str(formatted.get("t_text") or "NO EVALUABLE")
        ectopy = str(formatted.get("ectopy_text") or "EXTRASISTOLIA NO EVALUABLE")
    else:
        st = "NO EVALUABLE"
        twave = "NO EVALUABLE"
        ectopy = "EXTRASISTOLIA NO EVALUABLE"

    rhythm_label = str(rhythm_screen.get("label") or "").strip()
    if not trusted_signal_report or not rhythm_label:
        rhythm_label = "RITMO NO EVALUABLE"

    # R27 is probability-only. When it genuinely ran, expose the rhythm-related
    # probabilities without turning them into a thresholded diagnosis.
    r27_rhythm = {}
    if isinstance(r27_payload, dict):
        modules = r27_payload.get("modules") or {}
        for key in ("AF", "FLUTTER", "SVT", "SINUS", "SINUS_TACHY"):
            item = modules.get(key)
            if isinstance(item, dict) and item.get("probability") is not None:
                try:
                    r27_rhythm[key] = float(item["probability"])
                except Exception:
                    pass

    r27_line = None
    if r27_rhythm:
        ordered = sorted(r27_rhythm.items(), key=lambda kv: kv[1], reverse=True)
        r27_line = " | ".join(f"{k} {v:.3f}" for k, v in ordered)

    conclusion_bits = []
    if rhythm_label != "RITMO NO EVALUABLE":
        conclusion_bits.append(rhythm_label)
    if _num(machine.get("heart_rate_bpm")) is not None or _num(motor.get("heart_rate_bpm")) is not None:
        conclusion_bits.append(f"FC {hr_text}")
    if _num(machine.get("qrs_ms")) is not None or _num(motor.get("qrs_ms")) is not None:
        conclusion_bits.append(f"QRS {qrs_text}")
    if _num(machine.get("qrs_axis_deg")) is not None or _num(motor.get("axis_deg")) is not None:
        conclusion_bits.append(f"EJE {axis_text}")
    if qt_text != "NO EVALUABLE":
        conclusion_bits.append(qt_text)
    if pr_text != "NO EVALUABLE":
        conclusion_bits.append(f"PR {pr_text}")
    if trusted_signal_report and st != "NO EVALUABLE":
        conclusion_bits.append(st)
    if trusted_signal_report and twave != "NO EVALUABLE":
        conclusion_bits.append(twave)
    if trusted_signal_report and "SIN EXTRASÍSTOLES" in ectopy:
        conclusion_bits.append("SIN EXTRASÍSTOLES EVIDENTES")

    if discrepancies:
        conclusion_bits.append(
            "DISCORDANCIA ENTRE MEDICIONES IMPRESAS Y MOTOR EN "
            + ", ".join(str(c["label"]) for c in discrepancies)
        )

    conclusion = (
        "ELECTROCARDIOGRAMA CON " + ", ".join(conclusion_bits) + "."
        if conclusion_bits
        else "MEDICIONES AUTOMATIZADAS NO DISPONIBLES."
    )

    screen_code = str(rhythm_screen.get("code") or "")
    if screen_code == "AF_COMPATIBLE":
        idx = "PATRÓN COMPATIBLE CON FIBRILACIÓN AURICULAR CON RESPUESTA VENTRICULAR RÁPIDA"
    elif screen_code == "SVT_COMPATIBLE":
        idx = "PATRÓN COMPATIBLE CON TAQUICARDIA SUPRAVENTRICULAR REGULAR DE QRS ESTRECHO"
    elif screen_code == "SINUS_TACHY_COMPATIBLE":
        idx = "PATRÓN COMPATIBLE CON TAQUICARDIA SINUSAL"
    elif screen_code == "NARROW_TACHY_UNCLASSIFIED":
        idx = "TAQUICARDIA DE QRS ESTRECHO NO CLASIFICADA POR EL SCREENING DE RITMO"
    elif screen_code == "SINUS_COMPATIBLE":
        idx = "PATRÓN COMPATIBLE CON RITMO SINUSAL REGULAR"
    else:
        idx = "RITMO NO CLASIFICADO; REVISIÓN MANUAL"

    lines = [
        f"RITMO: {rhythm_label}.",
        f"FC: {hr_text}.",
        f"EJE: {axis_text}.",
        f"SEGMENTO PR: {pr_text}.",
        f"COMPLEJO QRS: {qrs_text}.",
        f"QT/QTc: {qt_text}.",
        f"SEGMENTO ST: {st}.",
        f"ONDA T: {twave}.",
        f"{ectopy}.",
    ]
    if r27_line:
        lines.append(
            "R27 RITMO (PROBABILIDADES; SIN UMBRAL DIAGNÓSTICO): " + r27_line + "."
        )
    lines.extend([
        f"CONCLUSIÓN: {conclusion}",
        f"IDX: {idx}.",
    ])

    return {
        "text": "\n".join(lines),
        "trusted_signal_report": bool(trusted_signal_report),
        "machine_measurements_used": bool(machine.get("detected")),
        "motor_measurements_used": bool(motor),
        "comparisons": comparisons,
        "discrepancies": discrepancies,
        "rhythm_screen": rhythm_screen,
        "r27_rhythm_probabilities": r27_rhythm,
        "idx": idx,
    }

