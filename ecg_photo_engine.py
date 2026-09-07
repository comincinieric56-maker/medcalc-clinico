"""MedCalc Clínico · ECG desde fotografía/PDF · V8.3.2 determinista.

Sin IA generativa, sin modelos de visión y sin APIs externas.
Esta capa realiza únicamente procesamiento clásico de imagen/señal:
- normalización de imagen;
- rasterización local de páginas PDF a alta resolución;
- rectificación geométrica conservadora;
- evaluación de calidad y cuadrícula;
- búsqueda experimental del pulso de calibración;
- segmentación candidata del formato estándar 3x4 + tira larga;
- reconstrucción preliminar de trazas para control visual;
- estimación de FC solo cuando la escala temporal queda suficientemente sustentada.

La V8.3.1 NO emite diagnósticos electrocardiográficos. Primero valida que la
foto puede digitalizarse de forma reproducible. Si no puede, devuelve no medible.
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ECG_SCHEMA_VERSION = "ECG_PHOTO_DETERMINISTIC_V1"
STANDARD_LEADS = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]




def pdf_page_count(pdf_bytes: bytes) -> int:
    """Devuelve el número de páginas de un PDF sin enviarlo a servicios externos."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "Falta PyMuPDF. Añada 'PyMuPDF>=1.24,<2' al requirements.txt para admitir PDF."
        ) from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el PDF: {exc}") from exc
    try:
        if getattr(doc, "needs_pass", False):
            raise ValueError("El PDF está protegido con contraseña y no puede procesarse.")
        count = int(doc.page_count)
        if count < 1:
            raise ValueError("El PDF no contiene páginas.")
        return count
    finally:
        doc.close()


def render_ecg_pdf_page(
    pdf_bytes: bytes,
    *,
    page_index: int = 0,
    dpi: int = 300,
    max_dimension: int = 4200,
) -> Tuple[bytes, Dict[str, Any]]:
    """Rasteriza una página PDF localmente y devuelve JPEG apto para el pipeline ECG.

    No usa OCR, IA ni servicios externos. Se renderiza a 300 dpi por defecto para
    preservar cuadrícula y trazado; después se limita la dimensión máxima para
    controlar memoria.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "Falta PyMuPDF. Añada 'PyMuPDF>=1.24,<2' al requirements.txt para admitir PDF."
        ) from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el PDF: {exc}") from exc

    try:
        if getattr(doc, "needs_pass", False):
            raise ValueError("El PDF está protegido con contraseña y no puede procesarse.")
        page_count = int(doc.page_count)
        if page_count < 1:
            raise ValueError("El PDF no contiene páginas.")
        if not 0 <= int(page_index) < page_count:
            raise ValueError(f"Página fuera de rango: {page_index + 1} de {page_count}.")

        page = doc.load_page(int(page_index))
        render_dpi = max(144, min(int(dpi), 400))
        zoom = render_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    rendered_size = img.size
    scale = min(1.0, float(max_dimension) / max(img.size))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    return _pil_to_jpeg_bytes(img, quality=95), {
        "source_type": "pdf",
        "page_index": int(page_index),
        "page_number": int(page_index) + 1,
        "page_count": page_count,
        "render_dpi": render_dpi,
        "rendered_width": rendered_size[0],
        "rendered_height": rendered_size[1],
        "processed_width": img.width,
        "processed_height": img.height,
    }

def _as_rgb(image_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    return ImageOps.exif_transpose(img).convert("RGB")


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 94) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _cv_to_jpeg_bytes(bgr: np.ndarray, quality: int = 94) -> bytes:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("No se pudo codificar la imagen procesada.")
    return enc.tobytes()


def prepare_ecg_image(
    image_bytes: bytes,
    *,
    crop_header: bool = False,
    header_fraction: float = 0.0,
    max_dimension: int = 3600,
) -> Tuple[bytes, str, Dict[str, Any]]:
    """Normaliza EXIF, color y tamaño. Todo ocurre localmente en Streamlit."""
    img = _as_rgb(image_bytes)
    original_size = img.size
    cropped = False
    if crop_header and 0 < header_fraction < 0.30:
        y0 = int(round(img.height * header_fraction))
        if y0 < img.height - 400:
            img = img.crop((0, y0, img.width, img.height))
            cropped = True

    scale = min(1.0, float(max_dimension) / max(img.size))
    if scale < 1.0:
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )

    return _pil_to_jpeg_bytes(img), "image/jpeg", {
        "original_width": original_size[0],
        "original_height": original_size[1],
        "processed_width": img.width,
        "processed_height": img.height,
        "header_cropped": cropped,
        "header_fraction": header_fraction if cropped else 0.0,
    }


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def _warp_quad(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = _order_quad(quad)
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    width = max(width, 300)
    height = max(height, 220)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
    return cv2.warpPerspective(bgr, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify_ecg_photo(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Busca el contorno del papel y corrige perspectiva solo si la evidencia es fuerte.

    Si no encuentra un cuadrilátero grande y plausible, conserva la imagen original.
    """
    pil = _as_rgb(image_bytes)
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else bgr.copy()
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    image_area = float(sh * sw)
    best = None
    best_score = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        area_frac = area / max(image_area, 1.0)
        if area_frac < 0.35:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        if min(rw, rh) < 100:
            continue
        rectangularity = min(1.0, area / max(rw * rh, 1.0))
        score = 0.72 * min(1.0, area_frac / 0.82) + 0.28 * rectangularity
        if score > best_score:
            best_score = score
            best = pts

    if best is None or best_score < 0.55:
        return image_bytes, {
            "rectified": False,
            "confidence": round(float(best_score), 3),
            "reason": "No se identificó un borde de papel suficientemente robusto; se conserva la geometría original.",
        }

    best_full = best / scale if scale < 1 else best
    warped = _warp_quad(bgr, best_full)
    wh, ww = warped.shape[:2]
    if ww < wh:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        wh, ww = warped.shape[:2]

    return _cv_to_jpeg_bytes(warped), {
        "rectified": True,
        "confidence": round(float(best_score), 3),
        "output_width": int(ww),
        "output_height": int(wh),
        "reason": "Borde de papel detectado y transformado mediante homografía.",
    }


def _laplacian_variance(gray: np.ndarray) -> float:
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    return float(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F).var())


def _smooth_1d(x: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    if len(x) < window:
        return np.full_like(x, float(np.mean(x)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def _periodicity_score(signal: np.ndarray, min_lag: int = 4, max_lag: int = 160) -> Tuple[Optional[float], float]:
    x = np.asarray(signal, dtype=float)
    if x.size < 80:
        return None, 0.0
    x = x - _smooth_1d(x, max(15, min(101, (x.size // 12) * 2 + 1)))
    x = x - np.mean(x)
    sd = float(np.std(x))
    if sd < 1e-6:
        return None, 0.0
    x /= sd
    max_lag = min(max_lag, x.size // 4)
    if max_lag <= min_lag:
        return None, 0.0
    vals = []
    for lag in range(min_lag, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        vals.append(float(np.mean(a * b)) if a.size >= 20 else 0.0)
    vals = np.asarray(vals)
    idx = int(np.argmax(vals))
    best = float(vals[idx])
    background = float(np.percentile(vals, 75)) if vals.size else 0.0
    confidence = max(0.0, min(1.0, (best - background) / 0.45))
    return float(min_lag + idx), confidence


def _grid_signal(rgb: np.ndarray) -> Tuple[np.ndarray, str, float]:
    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)
    red_excess = r - 0.5 * (g + b)
    red_mask = (red_excess > 18) & (r > 115)
    red_fraction = float(np.mean(red_mask))
    if red_fraction >= 0.004:
        return np.clip(red_excess, 0, None), "red_grid", red_fraction
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(245.0 - gray, 0, None), "monochrome_or_unknown", red_fraction


def _regional_periodicity(score_img: np.ndarray, axis: int) -> Tuple[Optional[float], float, float]:
    if axis == 0:
        proj = np.mean(score_img, axis=0)
        pieces = np.array_split(score_img, 3, axis=0)
        local = [_periodicity_score(np.mean(p, axis=0)) for p in pieces]
    else:
        proj = np.mean(score_img, axis=1)
        pieces = np.array_split(score_img, 3, axis=1)
        local = [_periodicity_score(np.mean(p, axis=1)) for p in pieces]
    lag, conf = _periodicity_score(proj)
    lags = [x[0] for x in local if x[0] is not None and x[1] >= 0.15]
    variation = 0.0
    if len(lags) >= 2 and np.mean(lags) > 0:
        variation = float((max(lags) - min(lags)) / np.mean(lags))
    return lag, conf, variation


def assess_ecg_photo(image_bytes: bytes) -> Dict[str, Any]:
    img = _as_rgb(image_bytes)
    scale = min(1.0, 1800.0 / max(img.size))
    if scale < 1:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(float)

    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    sharpness = _laplacian_variance(gray)
    dark_fraction = float(np.mean(gray < 90))
    white_fraction = float(np.mean(gray > 245))

    score_img, grid_kind, red_fraction = _grid_signal(rgb)
    gx, cx, vx = _regional_periodicity(score_img, axis=0)
    gy, cy, vy = _regional_periodicity(score_img, axis=1)
    grid_conf = float((cx + cy) / 2.0)
    perspective_variation = float(max(vx, vy))

    small_candidates = []
    for lag in (gx, gy):
        if lag is None:
            continue
        lag = float(lag)
        small_candidates.append(lag / 5.0 if lag > 35.0 else lag)
    small_grid_px = float(np.median(small_candidates)) if small_candidates else None

    resolution_score = min(1.0, min(img.width, img.height) / 1000.0)
    contrast_score = min(1.0, max(0.0, (contrast - 35.0) / 100.0))
    sharp_score = min(1.0, max(0.0, math.log1p(max(sharpness, 0.0)) / math.log1p(900.0)))
    exposure_score = 1.0 - min(1.0, max(0.0, (white_fraction - 0.92) / 0.08)) * 0.5
    perspective_score = max(0.0, 1.0 - perspective_variation / 0.35)
    q = 100.0 * (
        0.24 * resolution_score + 0.20 * contrast_score + 0.22 * sharp_score
        + 0.18 * grid_conf + 0.08 * exposure_score + 0.08 * perspective_score
    )
    quality_score = int(round(max(0.0, min(100.0, q))))
    label = "ALTA" if quality_score >= 80 else "ADECUADA" if quality_score >= 65 else "LIMITADA" if quality_score >= 45 else "INSUFICIENTE"

    issues: List[str] = []
    if min(img.width, img.height) < 700:
        issues.append("Resolución baja para una digitalización fina.")
    if contrast < 50:
        issues.append("Contraste bajo entre trazado, papel y cuadrícula.")
    if sharpness < 35:
        issues.append("Posible desenfoque o movimiento de cámara.")
    if grid_conf < 0.20:
        issues.append("Cuadrícula no detectada de forma robusta; no se puede convertir píxeles a milímetros.")
    if perspective_variation > 0.20:
        issues.append("Persisten diferencias de escala compatibles con perspectiva oblicua.")
    if dark_fraction < 0.002:
        issues.append("Trazado oscuro escaso o demasiado tenue.")

    return {
        "schema": ECG_SCHEMA_VERSION,
        "width": int(img.width),
        "height": int(img.height),
        "aspect_ratio": round(img.width / max(1, img.height), 3),
        "quality_score": quality_score,
        "quality_label": label,
        "contrast_range": round(contrast, 1),
        "sharpness_index": round(sharpness, 1),
        "grid_kind": grid_kind,
        "red_grid_fraction": round(red_fraction, 4),
        "grid_spacing_x_px_candidate": round(gx, 1) if gx else None,
        "grid_spacing_y_px_candidate": round(gy, 1) if gy else None,
        "small_grid_square_px_candidate": round(small_grid_px, 2) if small_grid_px else None,
        "grid_confidence": round(grid_conf, 3),
        "perspective_variation": round(perspective_variation, 3),
        "digitization_allowed": bool(quality_score >= 55 and grid_conf >= 0.20 and min(img.width, img.height) >= 650),
        "precision_measurements_allowed": bool(quality_score >= 70 and grid_conf >= 0.32 and perspective_variation <= 0.16),
        "issues": issues,
    }


def _black_trace_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    gray = (0.299 * r + 0.587 * g + 0.114 * b)
    red_excess = r - ((g + b) / 2.0)
    # Favorece tinta negra/gris y suprime cuadrícula roja.
    mask = (gray < 145) & (red_excess < 32)
    mask = mask.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask > 0


def detect_calibration_pulse(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Busca un pulso rectangular de calibración en el margen izquierdo.

    La inferencia de 25/50 mm/s se acepta solo con una forma compatible con un
    pulso de 200 ms y una altura cercana a 10 mm. Si no, devuelve no medible.
    """
    q = quality or assess_ecg_photo(image_bytes)
    gpx = q.get("small_grid_square_px_candidate")
    if not gpx or float(q.get("grid_confidence") or 0) < 0.28:
        return {"detected": False, "confidence": 0.0, "speed_mm_s": None, "gain_mm_mV": None, "reason": "Cuadrícula insuficiente."}
    gpx = float(gpx)
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    h, w = rgb.shape[:2]
    roi = rgb[:, : max(int(w * 0.22), int(22 * gpx))]
    mask = _black_trace_mask(roi).astype(np.uint8) * 255
    k = max(2, int(round(gpx * 0.30)))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Dict[str, Any]] = None
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        hg = bh / gpx
        wg = bw / gpx
        if not (7.0 <= hg <= 13.5 and 2.2 <= wg <= 12.5):
            continue
        if bw < 8 or bh < 20:
            continue
        height_score = math.exp(-0.5 * ((hg - 10.0) / 1.8) ** 2)
        # El bounding box suele incluir los segmentos basales antes/después del pulso.
        # Para inferir velocidad se mide la meseta horizontal superior, no el ancho total.
        crop = mask[y:y+bh, x:x+bw]
        top_band = crop[:max(3, int(round(bh * 0.35))), :]
        plateau_px = 0
        for rr in range(top_band.shape[0]):
            row = top_band[rr] > 0
            if not np.any(row):
                continue
            padded = np.r_[False, row, False].astype(np.int8)
            d = np.diff(padded)
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            if len(starts) and len(ends):
                plateau_px = max(plateau_px, int(np.max(ends - starts)))
        plateau_g = plateau_px / gpx if plateau_px else 0.0
        width25 = math.exp(-0.5 * ((plateau_g - 5.0) / 0.85) ** 2)
        width50 = math.exp(-0.5 * ((plateau_g - 10.0) / 1.15) ** 2)
        width_score = max(width25, width50)
        left_score = max(0.0, 1.0 - x / max(1.0, roi.shape[1] * 0.9))
        # Una calibración es predominantemente línea, no un bloque relleno.
        fill = float(np.mean(crop > 0)) if crop.size else 1.0
        fill_score = 1.0 if 0.03 <= fill <= 0.42 else max(0.0, 1.0 - abs(fill - 0.20) / 0.50)
        score = 0.37 * height_score + 0.39 * width_score + 0.12 * left_score + 0.12 * fill_score
        cand = {"score": score, "x": x, "y": y, "w": bw, "h": bh, "height_grid": hg, "width_grid": wg, "plateau_grid": plateau_g, "width25": width25, "width50": width50}
        if best is None or score > best["score"]:
            best = cand

    if best is None or best["score"] < 0.70:
        return {"detected": False, "confidence": round(best["score"], 3) if best else 0.0, "speed_mm_s": None, "gain_mm_mV": None, "reason": "No se identificó un pulso de calibración con geometría suficiente."}

    speed = 25.0 if best["width25"] >= best["width50"] else 50.0
    plateau_g = float(best.get("plateau_grid") or 0.0)
    width_close = abs(plateau_g - (5.0 if speed == 25.0 else 10.0)) <= (1.0 if speed == 25.0 else 1.4)
    # La altura ~10 cuadros pequeños se interpreta como pulso estándar 1 mV -> 10 mm/mV.
    confidence = float(best["score"])
    calibrated = bool(confidence >= 0.78 and width_close)
    return {
        "detected": True,
        "confidence": round(confidence, 3),
        "speed_mm_s": speed if calibrated else None,
        "gain_mm_mV": 10.0 if calibrated else None,
        "candidate_speed_mm_s": speed,
        "bbox": [int(best["x"]), int(best["y"]), int(best["w"]), int(best["h"])],
        "height_small_squares": round(float(best["height_grid"]), 2),
        "width_small_squares": round(float(best["width_grid"]), 2),
        "plateau_small_squares": round(plateau_g, 2),
        "reason": "Pulso geométricamente compatible, pero la meseta temporal no permite validar 25/50 mm/s." if not calibrated else "Pulso compatible con calibración estándar detectado automáticamente.",
    }


def _layout_regions(width: int, height: int) -> List[Dict[str, Any]]:
    # Márgenes conservadores para reducir textos periféricos.
    x0, x1 = int(width * 0.025), int(width * 0.985)
    y0, y1 = int(height * 0.05), int(height * 0.96)
    usable_w, usable_h = x1 - x0, y1 - y0
    row_h = usable_h / 4.0
    col_w = usable_w / 4.0
    regions: List[Dict[str, Any]] = []
    for r in range(3):
        for c in range(4):
            xa = int(round(x0 + c * col_w)); xb = int(round(x0 + (c + 1) * col_w))
            ya = int(round(y0 + r * row_h)); yb = int(round(y0 + (r + 1) * row_h))
            regions.append({"lead": STANDARD_LEADS[r][c], "rect": [xa, ya, xb, yb], "row": r, "col": c})
    regions.append({"lead": "RHYTHM", "rect": [x0, int(round(y0 + 3 * row_h)), x1, y1], "row": 3, "col": 0})
    return regions


def _trace_track(region_rgb: np.ndarray, grid_px: Optional[float] = None) -> Tuple[np.ndarray, float, Dict[str, float]]:
    h, w = region_rgb.shape[:2]
    # Evita rótulo de derivación al inicio y bordes de panel.
    sx0, sx1 = int(w * 0.10), int(w * 0.98)
    sy0, sy1 = int(h * 0.08), int(h * 0.92)
    crop = region_rgb[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return np.array([]), 0.0, {"coverage": 0.0, "amplitude_px": 0.0}
    mask = _black_trace_mask(crop)
    ch, cw = mask.shape
    row_counts = mask.sum(axis=1)
    if row_counts.max(initial=0) <= 1:
        return np.array([]), 0.0, {"coverage": 0.0, "amplitude_px": 0.0}

    # Inicio cerca de la zona con mayor continuidad horizontal.
    baseline = float(np.argmax(_smooth_1d(row_counts.astype(float), max(3, int(ch * 0.03) | 1))))
    radius = max(8, int(round((grid_px or max(4.0, ch / 40.0)) * 2.6)))
    ytrack = np.full(cw, np.nan, dtype=float)
    prev = baseline
    found = 0
    for x in range(cw):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            near = ys[np.abs(ys - prev) <= radius]
            if near.size:
                y = float(near[np.argmin(np.abs(near - prev))])
                found += 1
            else:
                y = prev
        else:
            y = prev
        ytrack[x] = y
        prev = y

    coverage = found / max(cw, 1)
    sig = baseline - ytrack
    # Remueve deriva lenta, no altera la forma rápida para el preview.
    detrend_window = max(15, int((grid_px or 6.0) * 15) | 1)
    sig = sig - _smooth_1d(sig, min(detrend_window, max(15, (len(sig)//3)*2+1)))
    amplitude = float(np.percentile(sig, 95) - np.percentile(sig, 5)) if len(sig) else 0.0
    continuity_score = min(1.0, coverage / 0.70)
    amplitude_score = min(1.0, amplitude / max(4.0, (grid_px or 6.0) * 2.0))
    confidence = 0.72 * continuity_score + 0.28 * amplitude_score
    return sig, float(confidence), {"coverage": float(coverage), "amplitude_px": amplitude, "offset_x": float(sx0), "offset_y": float(sy0), "baseline_y": float(baseline)}


def digitize_standard_12lead_preview(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Segmenta 3x4+tira larga y reconstruye trazas solo para CONTROL VISUAL.

    No etiqueta esto como medición clínica. El usuario debe comprobar que las
    líneas reconstruidas siguen la tinta original antes de habilitar la V0.2.
    """
    q = quality or assess_ecg_photo(image_bytes)
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    h, w = rgb.shape[:2]
    regions = _layout_regions(w, h)
    grid_px = q.get("small_grid_square_px_candidate")

    overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    recon = np.full((max(700, h), max(1200, w), 3), 255, dtype=np.uint8)
    panel_h = recon.shape[0] // 4
    panel_w = recon.shape[1] // 4

    per_lead = []
    confs = []
    for reg in regions:
        xa, ya, xb, yb = reg["rect"]
        cv2.rectangle(overlay, (xa, ya), (xb, yb), (60, 90, 60), 2)
        cv2.putText(overlay, reg["lead"], (xa + 8, ya + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 60, 30), 2, cv2.LINE_AA)
        region_rgb = rgb[ya:yb, xa:xb]
        sig, conf, stats = _trace_track(region_rgb, float(grid_px) if grid_px else None)
        confs.append(conf)
        per_lead.append({
            "lead": reg["lead"],
            "confidence": round(conf, 3),
            "coverage": round(stats.get("coverage", 0.0), 3),
            "amplitude_px": round(stats.get("amplitude_px", 0.0), 1),
            "samples": int(len(sig)),
        })
        if len(sig) < 10:
            continue
        if reg["lead"] == "RHYTHM":
            rx0, ry0, rw, rh = 0, 3 * panel_h, recon.shape[1], panel_h
        else:
            rx0, ry0, rw, rh = reg["col"] * panel_w, reg["row"] * panel_h, panel_w, panel_h
        cv2.rectangle(recon, (rx0, ry0), (min(recon.shape[1]-1, rx0+rw-1), min(recon.shape[0]-1, ry0+rh-1)), (220,220,220), 1)
        cv2.putText(recon, reg["lead"], (rx0 + 8, ry0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80,80,80), 1, cv2.LINE_AA)
        xs = np.linspace(rx0 + 10, rx0 + rw - 10, len(sig)).astype(np.int32)
        center = ry0 + rh // 2
        # Escala visual únicamente; conserva proporciones dentro de cada panel.
        s95 = max(1.0, float(np.percentile(np.abs(sig), 95)))
        ys = (center - np.clip(sig / s95, -1.0, 1.0) * (rh * 0.34)).astype(np.int32)
        pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
        cv2.polylines(recon, [pts], False, (0,0,0), 1, cv2.LINE_AA)

    layout_conf = float(np.mean(confs)) if confs else 0.0
    # La mera segmentación geométrica nunca obtiene confianza clínica alta en V0.1.
    status = "ADECUADA PARA CONTROL VISUAL" if layout_conf >= 0.55 and q.get("digitization_allowed") else "REVISAR SEGMENTACIÓN"
    return {
        "schema": ECG_SCHEMA_VERSION,
        "layout": "3x4_plus_rhythm_candidate",
        "layout_status": status,
        "layout_confidence": round(layout_conf, 3),
        "overlay_bytes": _cv_to_jpeg_bytes(overlay, 92),
        "reconstruction_bytes": _cv_to_jpeg_bytes(recon, 92),
        "per_lead": per_lead,
        "warning": "Reconstrucción preliminar para validar seguimiento de tinta; no usar todavía para PR/QRS/QT/ST ni diagnóstico.",
    }


def estimate_rhythm_strip_hr(
    image_bytes: bytes,
    quality: Optional[Dict[str, Any]] = None,
    *,
    speed_mm_s: Optional[float] = None,
) -> Dict[str, Any]:
    q = quality or assess_ecg_photo(image_bytes)
    if speed_mm_s is None or float(speed_mm_s) <= 0:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Velocidad del papel no validada automáticamente."}
    speed_mm_s = float(speed_mm_s)
    grid_px = q.get("small_grid_square_px_candidate")
    if not grid_px or float(grid_px) < 3 or float(q.get("grid_confidence") or 0) < 0.28:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Cuadrícula insuficiente para convertir píxeles a tiempo."}
    if float(q.get("perspective_variation") or 0) > 0.22:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Perspectiva excesiva para medición temporal."}

    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    h, w = rgb.shape[:2]
    regions = _layout_regions(w, h)
    reg = regions[-1]
    xa, ya, xb, yb = reg["rect"]
    strip = rgb[ya:yb, xa:xb]
    signal, track_conf, _ = _trace_track(strip, float(grid_px))
    if len(signal) < 300 or track_conf < 0.35:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Tira de ritmo no recuperada con continuidad suficiente."}

    long_window = max(31, int(round(float(grid_px) * 12)) | 1)
    centered = signal - _smooth_1d(signal, min(long_window, max(31, (len(signal)//3)*2+1)))
    deriv = np.diff(centered, prepend=centered[0])
    energy = _smooth_1d(deriv * deriv, max(3, int(round(float(grid_px) * 0.35)) | 1))
    energy = energy - np.mean(energy)
    sd = float(np.std(energy))
    if sd < 1e-6:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Periodicidad QRS no recuperable."}
    energy /= sd

    sec_per_px = (1.0 / speed_mm_s) / float(grid_px)
    lag_min = max(5, int(round((60.0 / 220.0) / sec_per_px)))
    lag_max = min(len(energy) // 2, int(round((60.0 / 30.0) / sec_per_px)))
    if lag_max <= lag_min + 5:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Ventana temporal insuficiente."}

    vals = np.asarray([float(np.mean(energy[:-lag] * energy[lag:])) for lag in range(lag_min, lag_max + 1)])
    idx_max = int(np.argmax(vals)); best = float(vals[idx_max])
    strong = max(0.12, best * 0.62)
    candidates = [i for i in range(1, len(vals)-1) if vals[i] >= strong and vals[i] >= vals[i-1] and vals[i] >= vals[i+1]]
    idx = candidates[0] if candidates else idx_max
    chosen = float(vals[idx]); lag = lag_min + idx
    hr = 60.0 / (lag * sec_per_px)
    background = float(np.percentile(vals, 75))
    specificity = max(0.0, chosen - background)
    conf_score = max(0.0, min(1.0, 0.45 * track_conf + 0.35 * max(0.0, chosen) + 1.2 * specificity))
    confidence = "alta" if conf_score >= 0.72 else "media" if conf_score >= 0.50 else "baja"
    if not (30 <= hr <= 220) or chosen < 0.12 or conf_score < 0.45:
        return {"value": None, "unit": "lpm", "confidence": "no_medido", "reason": "Periodicidad fisiológicamente plausible no suficientemente robusta."}
    return {
        "value": round(float(hr), 1), "unit": "lpm", "confidence": confidence,
        "confidence_score": round(conf_score, 3),
        "reason": "FC por periodicidad de la tira inferior digitalizada y escala temporal validada automáticamente.",
    }


def enhanced_preview(image_bytes: bytes) -> bytes:
    img = _as_rgb(image_bytes)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return _pil_to_jpeg_bytes(img, 92)


# =============================================================================
# V8.3.3 · ANALISIS CLINICO DE RITMO DESDE FOTO/PDF · DETERMINISTA
# =============================================================================
# Esta capa NO usa IA ni modelos externos. Corrige tres limitaciones de V8.3.2:
# 1) estima mejor la reticula fina a partir de la periodicidad de las lineas mayores;
# 2) busca calibracion tambien en el margen derecho/tira larga;
# 3) para ritmos irregulares detecta QRS individuales y analiza los RR, en vez de
#    exigir periodicidad global.


def _autocorr_grid_geometry(rgb: np.ndarray) -> Dict[str, Any]:
    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)
    red_score = np.clip(r - 0.5 * (g + b), 0, None)
    combined: Dict[int, List[float]] = {}
    for axis in (0, 1):
        sig = np.mean(red_score, axis=axis).astype(float)
        sig -= float(np.mean(sig))
        sd = float(np.std(sig))
        if sd < 1e-6:
            continue
        sig /= sd
        max_lag = min(140, max(12, len(sig) // 3))
        for lag in range(4, max_lag + 1):
            a, bb = sig[:-lag], sig[lag:]
            if a.size < 40:
                continue
            combined.setdefault(lag, []).append(float(np.mean(a * bb)))
    scores = {k: float(np.mean(v)) for k, v in combined.items() if len(v) >= 2 and 6 <= k <= 120}
    if not scores:
        return {"small_grid_px": None, "major_grid_px": None, "confidence": 0.0}
    best_score = max(scores.values())
    strong = sorted(k for k, v in scores.items() if v >= max(0.18, 0.60 * best_score))
    fundamental = strong[0] if strong else max(scores, key=scores.get)
    # En ECG fotografiados suelen verse mejor las lineas de 5 mm que las de 1 mm.
    # Si la periodicidad elegida es >= ~8 px, se trata como linea mayor y se divide /5.
    major = float(fundamental)
    small = major / 5.0
    # Si la subperiodicidad 1/5 es visible, aumenta confianza.
    sub = max(1, int(round(major / 5.0)))
    sub_score = float(scores.get(sub, 0.0))
    harmonic_support = min(1.0, max(0.0, sub_score / max(abs(scores.get(fundamental, 1e-6)), 1e-6)))
    conf = max(0.0, min(1.0, 0.65 * max(0.0, best_score) + 0.35 * harmonic_support))
    if not (1.2 <= small <= 30.0):
        return {"small_grid_px": None, "major_grid_px": major, "confidence": 0.0}
    return {
        "small_grid_px": float(small),
        "major_grid_px": float(major),
        "confidence": float(conf),
        "fundamental_lag_px": int(fundamental),
        "best_periodicity": float(best_score),
        "subperiodicity_support": float(sub_score),
    }


def assess_ecg_photo(image_bytes: bytes) -> Dict[str, Any]:
    img = _as_rgb(image_bytes)
    scale = min(1.0, 2200.0 / max(img.size))
    if scale < 1:
        img = img.resize((max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale)))), Image.Resampling.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(float)
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float(p95 - p5)
    sharpness = _laplacian_variance(gray)
    dark_fraction = float(np.mean(gray < 120))
    geom = _autocorr_grid_geometry(rgb)
    grid_conf = float(geom.get("confidence") or 0.0)
    small_grid = geom.get("small_grid_px")
    resolution_score = min(1.0, max(0.0, min(img.width, img.height) / 750.0))
    contrast_score = min(1.0, max(0.0, (contrast - 25.0) / 90.0))
    sharp_score = min(1.0, max(0.0, math.log1p(max(sharpness, 0.0)) / math.log1p(1200.0)))
    q = 100.0 * (0.34 * resolution_score + 0.24 * contrast_score + 0.24 * sharp_score + 0.18 * grid_conf)
    quality_score = int(round(max(0.0, min(100.0, q))))
    label = "ALTA" if quality_score >= 80 else "ADECUADA" if quality_score >= 60 else "LIMITADA" if quality_score >= 42 else "INSUFICIENTE"
    issues: List[str] = []
    if min(img.width, img.height) < 550:
        issues.append("Resolucion limitada para intervalos finos; el analisis de ritmo puede seguir siendo util si los QRS son visibles.")
    if contrast < 45:
        issues.append("Contraste reducido entre trazado y papel.")
    if sharpness < 30:
        issues.append("Posible desenfoque o movimiento.")
    if grid_conf < 0.20:
        issues.append("Reticula no demostrada con suficiente confianza para convertir pixeles a milimetros.")
    if dark_fraction < 0.001:
        issues.append("Trazado oscuro insuficiente.")
    return {
        "schema": "ECG_PHOTO_DETERMINISTIC_V2",
        "width": int(img.width), "height": int(img.height),
        "aspect_ratio": round(img.width / max(1, img.height), 3),
        "quality_score": quality_score, "quality_label": label,
        "contrast_range": round(contrast, 1), "sharpness_index": round(sharpness, 1),
        "grid_kind": "red_grid" if grid_conf >= 0.18 else "grid_not_confirmed",
        "small_grid_square_px_candidate": round(float(small_grid), 3) if small_grid else None,
        "major_grid_square_px_candidate": round(float(geom.get("major_grid_px")), 3) if geom.get("major_grid_px") else None,
        "grid_confidence": round(grid_conf, 3),
        "perspective_variation": 0.0,  # V8.3.3 no usa esta metrica para bloquear ritmo.
        "digitization_allowed": bool(quality_score >= 42 and min(img.width, img.height) >= 350),
        "precision_measurements_allowed": bool(quality_score >= 68 and grid_conf >= 0.35 and small_grid is not None),
        "rhythm_analysis_allowed": bool(quality_score >= 40 and img.width >= 500 and img.height >= 300),
        "issues": issues,
    }


def _trace_mask_adaptive(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    red_excess = r - ((g + b) / 2.0)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    p95_red = float(np.percentile(red_excess, 95))
    if p95_red > 25:
        mask = (val < 225) & (red_excess < 18)
    else:
        mask = (val < 220) & (sat < 65)
    mask |= (val < 90)
    m = mask.astype(np.uint8) * 255
    h, w = m.shape
    # Retira lineas largas del marco/panel sin borrar la morfologia corta del ECG.
    hlen = max(25, w // 12)
    vlen = max(18, h // 2)
    horizontal = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1)))
    vertical = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    return cv2.subtract(m, cv2.bitwise_or(horizontal, vertical))


def _component_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        out.append((x, y, w, h, area))
    return out


def _calibration_candidate_in_roi(mask: np.ndarray, grid_px: float, xoff: int, yoff: int) -> Optional[Dict[str, Any]]:
    g = float(grid_px)
    if mask.size == 0 or g <= 0:
        return None
    hk = max(3, int(round(3.0 * g)))
    vk = max(3, int(round(6.0 * g)))
    hor = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)))
    ver = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)))
    hs = [c for c in _component_boxes(hor) if 3.0 * g <= c[2] <= 11.5 * g and c[3] <= 4.0 * g]
    vs = [c for c in _component_boxes(ver) if 7.0 * g <= c[3] <= 13.5 * g and c[2] <= 4.0 * g]
    best = None
    for hx, hy, hw, hh, _ in hs:
        left = [v for v in vs if abs((v[0] + v[2] / 2) - hx) <= 2.8 * g and abs(v[1] - hy) <= 3.5 * g]
        right = [v for v in vs if abs((v[0] + v[2] / 2) - (hx + hw)) <= 2.8 * g and abs(v[1] - hy) <= 3.5 * g]
        if not left or not right:
            continue
        lv = min(left, key=lambda v: abs(v[3] / g - 10.0))
        rv = min(right, key=lambda v: abs(v[3] / g - 10.0))
        height_boxes = 0.5 * (lv[3] + rv[3]) / g
        plateau_boxes = hw / g
        height_score = math.exp(-0.5 * ((height_boxes - 10.0) / 2.1) ** 2)
        s25 = math.exp(-0.5 * ((plateau_boxes - 5.0) / 1.7) ** 2)
        s50 = math.exp(-0.5 * ((plateau_boxes - 10.0) / 2.0) ** 2)
        speed = 25.0 if s25 >= s50 else 50.0
        width_score = max(s25, s50)
        score = 0.58 * height_score + 0.42 * width_score
        cand = {
            "confidence": float(score), "speed_mm_s": speed,
            "gain_mm_mV": 10.0 if height_score >= 0.45 else None,
            "height_small_boxes": float(height_boxes), "plateau_small_boxes": float(plateau_boxes),
            "bbox": [int(xoff + hx), int(yoff + hy), int(hw), int(max(lv[3], rv[3]))],
        }
        if best is None or cand["confidence"] > best["confidence"]:
            best = cand
    return best


def detect_calibration_pulse(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = quality or assess_ecg_photo(image_bytes)
    g = q.get("small_grid_square_px_candidate")
    if not g or float(q.get("grid_confidence") or 0) < 0.18:
        return {"detected": False, "confidence": 0.0, "speed_mm_s": None, "gain_mm_mV": None, "reason": "Reticula insuficiente para evaluar calibracion."}
    g = float(g)
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    full_mask = _trace_mask_adaptive(rgb)
    h, w = full_mask.shape
    rois = [
        (int(w * 0.74), int(h * 0.70), w, h),      # esquina inferior derecha: frecuente en tira larga
        (0, int(h * 0.70), int(w * 0.28), h),      # esquina inferior izquierda
        (0, 0, int(w * 0.20), h),                  # margen izquierdo completo
        (int(w * 0.80), 0, w, h),                  # margen derecho completo
    ]
    candidates = []
    for x0, y0, x1, y1 in rois:
        cand = _calibration_candidate_in_roi(full_mask[y0:y1, x0:x1], g, x0, y0)
        if cand:
            candidates.append(cand)
    if not candidates:
        return {"detected": False, "confidence": 0.0, "speed_mm_s": None, "gain_mm_mV": None, "reason": "No se identifico un pulso rectangular de 1 mV con geometria suficiente."}
    best = max(candidates, key=lambda c: c["confidence"])
    conf = float(best["confidence"])
    accepted = conf >= 0.58 and best.get("gain_mm_mV") is not None
    return {
        "detected": bool(accepted), "confidence": round(conf, 3),
        "speed_mm_s": best.get("speed_mm_s") if accepted else None,
        "gain_mm_mV": best.get("gain_mm_mV") if accepted else None,
        "height_small_boxes": round(float(best.get("height_small_boxes") or 0), 2),
        "plateau_small_boxes": round(float(best.get("plateau_small_boxes") or 0), 2),
        "bbox": best.get("bbox"),
        "reason": (
            f"Pulso candidato: altura {best.get('height_small_boxes'):.1f} cuadros pequenos y meseta {best.get('plateau_small_boxes'):.1f}; "
            f"compatible con {int(best.get('speed_mm_s'))} mm/s y 10 mm/mV."
            if accepted else "Se encontro una forma candidata, pero no supera el umbral conservador de calibracion."
        ),
    }


def _rhythm_strip_region(rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = rgb.shape[:2]
    x0, x1 = int(round(w * 0.03)), int(round(w * 0.97))
    y0, y1 = int(round(h * 0.755)), int(round(h * 0.985))
    return rgb[y0:y1, x0:x1], (x0, y0, x1, y1)


def _smooth_vector(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(1, -1)
    return cv2.GaussianBlur(arr, (0, 0), max(0.35, float(sigma))).ravel()


def _detect_qrs_columns(clean: np.ndarray, small_grid_px: Optional[float]) -> Dict[str, Any]:
    h, w = clean.shape
    density = (clean > 0).sum(axis=0).astype(float)
    span = np.zeros(w, dtype=float)
    for x in range(w):
        ys = np.flatnonzero(clean[:, x] > 0)
        if ys.size:
            span[x] = float(ys[-1] - ys[0] + 1)
    compactness = density / np.maximum(span, 1.0)
    raw = density * np.sqrt(np.clip(compactness, 0.0, 1.0))
    raw[compactness < 0.38] *= 0.20
    score = _smooth_vector(raw, 0.8)
    lo, hi = int(round(w * 0.08)), int(round(w * 0.95))
    core = score[lo:hi]
    if core.size < 80:
        return {"positions": [], "strengths": [], "score": score, "density": density, "span": span, "confidence": 0.0}
    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med))) + 1e-6
    threshold = max(float(np.percentile(core, 72)), med + 1.2 * mad, 2.5)
    candidates = [i for i in range(lo + 1, hi - 1) if score[i] >= threshold and score[i] >= score[i - 1] and score[i] >= score[i + 1]]
    g = float(small_grid_px or max(1.5, w / 280.0))
    min_distance = max(8, int(round(4.2 * g)))
    selected: List[int] = []
    for idx in sorted(candidates, key=lambda j: float(score[j]), reverse=True):
        if all(abs(idx - prev) >= min_distance for prev in selected):
            selected.append(int(idx))
    selected.sort()
    selected = [idx for idx in selected if compactness[idx] >= 0.42 and span[idx] >= max(4.0, 1.8 * g)]
    # Si aparecen dos candidatos absurdamente próximos respecto al RR dominante,
    # conserva el más fuerte. Esto elimina T/P o restos de borde sin penalizar una
    # taquicardia verdadera, porque en ésta el RR corto sería el RR mediano.
    changed = True
    while changed and len(selected) >= 6:
        changed = False
        rr = np.diff(np.asarray(selected, dtype=float))
        med_rr = float(np.median(rr)) if rr.size else 0.0
        if med_rr <= 0:
            break
        for j, d in enumerate(rr):
            if d < 0.45 * med_rr:
                a, b = selected[j], selected[j + 1]
                rem = a if score[a] < score[b] else b
                selected.remove(rem)
                changed = True
                break
    # Limpia un candidato espurio en el borde de la tira (rotulo/calibracion) cuando
    # solo el primer o ultimo RR rompe una secuencia por lo demas estable.
    if len(selected) >= 7:
        rr_edge = np.diff(np.asarray(selected, dtype=float))
        med_edge = float(np.median(rr_edge)) if rr_edge.size else 0.0
        if med_edge > 0 and rr_edge[0] < 0.65 * med_edge and selected[0] < 0.13 * w:
            selected = selected[1:]
        if len(selected) >= 7:
            rr_edge = np.diff(np.asarray(selected, dtype=float))
            med_edge = float(np.median(rr_edge)) if rr_edge.size else 0.0
            if med_edge > 0 and rr_edge[-1] < 0.65 * med_edge and selected[-1] > 0.90 * w:
                selected = selected[:-1]
    strengths = [float(score[i]) for i in selected]
    conf = 0.0
    if len(selected) >= 6:
        prominence = float(np.median(strengths) / max(np.median(core) + mad, 1e-6)) if strengths else 0.0
        conf = max(0.0, min(1.0, 0.50 + 0.045 * min(len(selected), 12) + 0.10 * min(prominence, 2.0)))
    return {"positions": selected, "strengths": strengths, "score": score, "density": density, "span": span, "confidence": conf}


def _p_wave_reproducibility(clean: np.ndarray, qrs: List[int]) -> Optional[float]:
    if len(qrs) < 6:
        return None
    rr = np.diff(np.asarray(qrs, dtype=float))
    med_rr = float(np.median(rr)) if rr.size else 0.0
    if med_rr < 8:
        return None
    patches = []
    for q in qrs[1:-1]:
        x0 = max(0, int(round(q - 0.45 * med_rr)))
        x1 = min(clean.shape[1], int(round(q - 0.08 * med_rr)))
        if x1 - x0 < 6:
            continue
        patch = clean[:, x0:x1]
        patch = cv2.resize(patch, (40, 64), interpolation=cv2.INTER_AREA).astype(float) / 255.0
        if float(np.std(patch)) > 1e-5:
            patches.append(patch)
    if len(patches) < 4:
        return None
    corrs = []
    for i in range(len(patches)):
        a = patches[i].ravel()
        for j in range(i + 1, len(patches)):
            b = patches[j].ravel()
            if np.std(a) > 1e-6 and np.std(b) > 1e-6:
                c = float(np.corrcoef(a, b)[0, 1])
                if math.isfinite(c):
                    corrs.append(c)
    return float(np.median(corrs)) if corrs else None


def _rhythm_overlay_bytes(rgb: np.ndarray, strip_rect: Tuple[int, int, int, int], qrs: List[int], calibration: Optional[Dict[str, Any]] = None) -> bytes:
    bgr = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = strip_rect
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (30, 130, 210), 2)
    for pos in qrs:
        x = x0 + int(pos)
        cv2.line(bgr, (x, y0 + 4), (x, y1 - 4), (20, 40, 220), 1, cv2.LINE_AA)
    bbox = (calibration or {}).get("bbox")
    if bbox:
        bx, by, bw, bh = [int(v) for v in bbox]
        cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (30, 170, 60), 2)
    return _cv_to_jpeg_bytes(bgr, 94)


def analyze_rhythm_clinical(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None, calibration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q = quality or assess_ecg_photo(image_bytes)
    cal = calibration or detect_calibration_pulse(image_bytes, q)
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    strip, rect = _rhythm_strip_region(rgb)
    clean = _trace_mask_adaptive(strip)
    g = q.get("small_grid_square_px_candidate")
    det = _detect_qrs_columns(clean, float(g) if g else None)
    qrs = list(det.get("positions") or [])
    out: Dict[str, Any] = {
        "schema": "ECG_RHYTHM_CLINICAL_V1", "qrs_count": len(qrs),
        "qrs_confidence": round(float(det.get("confidence") or 0.0), 3),
        "rhythm": "NO_CLASIFICABLE", "rhythm_label": "Ritmo no clasificable",
        "rhythm_confidence": "insuficiente", "heart_rate_bpm": None,
        "heart_rate_confidence": "no_medido", "rr_pattern": "NO_MEDIBLE",
        "p_organization": "NO_MEDIBLE", "p_reproducibility": None,
        "interpretation": "No hay QRS suficientes para una clasificacion automatica de ritmo.",
        "urgent_flag": False,
    }
    if len(qrs) < 6:
        out["overlay_bytes"] = _rhythm_overlay_bytes(rgb, rect, qrs, cal)
        out["reason"] = "Se requieren al menos 6 complejos QRS detectables en la tira larga."
        return out
    rr = np.diff(np.asarray(qrs, dtype=float))
    med_rr = float(np.median(rr))
    mean_rr = float(np.mean(rr))
    rr_cv = float(np.std(rr) / max(mean_rr, 1e-6))
    rr_nmad = float(np.median(np.abs(rr - med_rr)) / max(med_rr, 1e-6))
    rr_sdiff = float(np.median(np.abs(np.diff(rr))) / max(med_rr, 1e-6)) if rr.size >= 3 else 0.0
    regular_fraction = float(np.mean(np.abs(rr - med_rr) / max(med_rr, 1e-6) <= 0.08))
    p_rep = _p_wave_reproducibility(clean, qrs)
    if p_rep is None:
        p_org = "INDETERMINADA"
    elif p_rep >= 0.62:
        p_org = "ORGANIZADA_REPRODUCIBLE"
    elif p_rep <= 0.42:
        p_org = "NO_REPRODUCIBLE"
    else:
        p_org = "INDETERMINADA"

    regular = rr_cv <= 0.085 and rr_nmad <= 0.07 and rr_sdiff <= 0.10
    irregularly_irregular = rr_cv >= 0.13 and rr_nmad >= 0.09 and rr_sdiff >= 0.12 and regular_fraction <= 0.55
    if regular:
        rr_label = "REGULAR"
    elif irregularly_irregular:
        rr_label = "IRREGULARMENTE_IRREGULAR"
    else:
        rr_label = "IRREGULAR"

    speed = cal.get("speed_mm_s")
    if speed and g and float(g) > 0:
        sec_per_px = 1.0 / (float(speed) * float(g))
        hr = 60.0 / max(mean_rr * sec_per_px, 1e-6)
        if 20 <= hr <= 300:
            out["heart_rate_bpm"] = round(float(hr), 1)
            out["heart_rate_confidence"] = "alta" if float(cal.get("confidence") or 0) >= 0.72 and float(det.get("confidence") or 0) >= 0.72 else "media"
    # Solo como informacion tecnica, nunca como FC clinica si la velocidad no esta validada.
    if g and float(g) > 0:
        rr_mm = mean_rr / float(g)
        out["hr_if_25_mm_s"] = round(60.0 * 25.0 / max(rr_mm, 1e-6), 1)
        out["hr_if_50_mm_s"] = round(60.0 * 50.0 / max(rr_mm, 1e-6), 1)

    if irregularly_irregular and p_org == "NO_REPRODUCIBLE":
        out.update({
            "rhythm": "FIBRILACION_AURICULAR_PROBABLE",
            "rhythm_label": "Patrón compatible con fibrilación auricular",
            "rhythm_confidence": "alta" if float(det.get("confidence") or 0) >= 0.75 and p_rep is not None and p_rep <= 0.30 else "media",
            "interpretation": "Respuesta ventricular irregularmente irregular y ausencia de un patrón auricular/P reproducible en la tira de ritmo; hallazgos compatibles con fibrilación auricular. Requiere confirmación visual del ECG original por un profesional.",
        })
    elif regular and p_org == "ORGANIZADA_REPRODUCIBLE":
        out.update({
            "rhythm": "RITMO_SINUSAL_PROBABLE",
            "rhythm_label": "Patrón compatible con ritmo sinusal regular",
            "rhythm_confidence": "alta" if float(det.get("confidence") or 0) >= 0.75 and p_rep is not None and p_rep >= 0.75 else "media",
            "interpretation": "RR regulares con actividad auricular pre-QRS reproducible; patrón compatible con ritmo sinusal. La morfología P y el eje de P aún no se validan en esta versión.",
        })
    elif irregularly_irregular:
        out.update({
            "rhythm": "RITMO_IRREGULARMENTE_IRREGULAR",
            "rhythm_label": "Ritmo irregularmente irregular",
            "rhythm_confidence": "media",
            "interpretation": "Se demuestra irregularidad RR marcada, pero la organización auricular no puede clasificarse con suficiente seguridad. Considerar fibrilación auricular entre los diferenciales y confirmar visualmente ondas P/actividad fibrilatoria.",
        })
    elif not regular:
        out.update({
            "rhythm": "RITMO_IRREGULAR",
            "rhythm_label": "Ritmo irregular",
            "rhythm_confidence": "media",
            "interpretation": "Se detecta variabilidad RR, pero no cumple el patrón determinista de irregularidad absoluta usado para sugerir fibrilación auricular.",
        })
    else:
        out.update({
            "rhythm": "RITMO_REGULAR_NO_CLASIFICADO",
            "rhythm_label": "Ritmo regular no clasificado",
            "rhythm_confidence": "media",
            "interpretation": "Los RR son regulares, pero la actividad auricular no es suficientemente reproducible para clasificar el mecanismo con seguridad.",
        })

    out.update({
        "rr_pattern": rr_label,
        "rr_cv": round(rr_cv, 3), "rr_nmad": round(rr_nmad, 3), "rr_successive_variation": round(rr_sdiff, 3),
        "rr_regular_fraction": round(regular_fraction, 3),
        "p_organization": p_org,
        "p_reproducibility": round(float(p_rep), 3) if p_rep is not None else None,
        "qrs_positions_px": [int(v) for v in qrs],
        "overlay_bytes": _rhythm_overlay_bytes(rgb, rect, qrs, cal),
        "calibration": cal,
        "grid_small_px": g,
    })
    return out
