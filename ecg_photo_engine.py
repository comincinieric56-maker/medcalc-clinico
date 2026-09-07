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

# =============================================================================
# V8.3.4 · INTERPRETACION CLINICA DETERMINISTA MULTIPARAMETRO
# =============================================================================

def _trace_component(clean: np.ndarray) -> Tuple[np.ndarray, float]:
    """Aísla la componente horizontal continua más compatible con el trazado ECG."""
    if clean.size == 0:
        return clean, 0.0
    h, w = clean.shape
    dil = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((dil > 0).astype(np.uint8), 8)
    best_i = None
    best_score = -1e9
    for i in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[i]]
        coverage = ww / max(w, 1)
        if coverage < 0.42 or hh < 4 or hh > 0.72 * h:
            continue
        touch_penalty = 0.12 if y <= 1 else 0.0
        score = 1.15 * coverage + 0.20 * min(1.0, area / max(ww * 5.0, 1.0)) - 0.30 * (hh / max(h, 1)) - touch_penalty
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        return clean, 0.25
    comp = (labels == best_i).astype(np.uint8)
    # Incluye la tinta original cubierta por la componente dilatada.
    out = cv2.bitwise_and(clean, clean, mask=comp)
    xs = np.flatnonzero(np.any(out > 0, axis=0))
    coverage = (xs[-1] - xs[0] + 1) / max(w, 1) if xs.size else 0.0
    return out, float(max(0.0, min(1.0, coverage)))


def _signal_from_component(clean: np.ndarray) -> Dict[str, Any]:
    """Obtiene una señal 1-D aproximada preservando la geometría original."""
    comp, coverage = _trace_component(clean)
    h, w = comp.shape
    y = np.full(w, np.nan, dtype=float)
    span = np.zeros(w, dtype=float)
    density = np.zeros(w, dtype=float)
    for x in range(w):
        ys = np.flatnonzero(comp[:, x] > 0)
        if ys.size:
            y[x] = float(np.median(ys))
            span[x] = float(ys[-1] - ys[0] + 1)
            density[x] = float(ys.size)
    idx = np.flatnonzero(np.isfinite(y))
    if idx.size < max(20, int(0.25 * w)):
        return {"ok": False, "coverage": coverage, "mask": comp}
    y = np.interp(np.arange(w), idx, y[idx])
    y = _smooth_vector(y, 1.0)
    # La línea de base se estima excluyendo columnas verticalmente extensas (QRS/texto).
    valid = span <= max(4.0, float(np.percentile(span[span > 0], 70)) if np.any(span > 0) else 4.0)
    baseline = float(np.median(y[valid])) if np.any(valid) else float(np.median(y))
    signal = baseline - y
    return {
        "ok": True, "coverage": coverage, "mask": comp, "y": y,
        "signal_px": signal, "baseline_y": baseline, "span": span, "density": density,
    }


def _measure_qrs_bounds(span: np.ndarray, qrs: List[int], grid_px: float, speed_mm_s: float) -> Dict[str, Any]:
    widths_ms: List[float] = []
    bounds: List[Tuple[int, int]] = []
    g = max(float(grid_px), 1.2)
    for q in qrs:
        if q <= 1 or q >= len(span) - 2:
            continue
        peak = float(span[q])
        if peak <= 0:
            continue
        thr = max(3.0, 0.15 * peak)
        L = max(0, int(round(q - 1.9 * g)))
        R = min(len(span) - 1, int(round(q + 1.9 * g)))
        active = span[L:R + 1] >= thr
        # Puentea huecos de hasta 2 columnas por antialiasing/intersección con cuadrícula.
        for k in range(1, len(active) - 1):
            if (not active[k]) and active[k - 1] and active[k + 1]:
                active[k] = True
        for k in range(1, len(active) - 2):
            if (not active[k]) and (not active[k + 1]) and active[k - 1] and active[k + 2]:
                active[k:k + 2] = True
        inds = np.flatnonzero(active)
        if inds.size == 0:
            continue
        seed = int(inds[np.argmin(np.abs(inds - (q - L)))])
        lo = seed
        hi = seed
        while lo > 0 and active[lo - 1]:
            lo -= 1
        while hi < len(active) - 1 and active[hi + 1]:
            hi += 1
        onset = L + lo
        offset = L + hi
        width_px = offset - onset + 1
        width_ms = 1000.0 * width_px / max(float(speed_mm_s) * g, 1e-6)
        if 35.0 <= width_ms <= 220.0:
            widths_ms.append(float(width_ms))
            bounds.append((int(onset), int(offset)))
    if not widths_ms:
        return {"value_ms": None, "confidence": 0.0, "bounds": [], "n": 0}
    med = float(np.median(widths_ms))
    mad = float(np.median(np.abs(np.asarray(widths_ms) - med)))
    dispersion = mad / max(med, 1e-6)
    conf = min(1.0, 0.45 + 0.045 * min(len(widths_ms), 10) + 0.28 * max(0.0, 1.0 - dispersion / 0.20))
    return {
        "value_ms": round(med, 1), "confidence": round(conf, 3), "bounds": bounds,
        "n": len(widths_ms), "mad_ms": round(mad, 1), "values_ms": [round(v, 1) for v in widths_ms],
    }


def _p_pr_measurements(signal_px: np.ndarray, qrs_bounds: List[Tuple[int, int]], grid_px: float, speed_mm_s: float, rhythm: str, p_org: str) -> Dict[str, Any]:
    if rhythm == "FIBRILACION_AURICULAR_PROBABLE" or p_org == "NO_REPRODUCIBLE":
        return {
            "p_present": False, "p_duration_ms": None, "p_amplitude_mv": None, "pr_ms": None,
            "confidence": 0.95, "reason": "No se identifican ondas P sinusales reproducibles; el PR no es medible en este ritmo."
        }
    if p_org != "ORGANIZADA_REPRODUCIBLE" or len(qrs_bounds) < 4:
        return {"p_present": None, "p_duration_ms": None, "p_amplitude_mv": None, "pr_ms": None, "confidence": 0.0, "reason": "Actividad auricular insuficientemente organizada para medir P/PR."}
    px_per_ms = float(speed_mm_s) * float(grid_px) / 1000.0
    p_durs, p_amps, prs = [], [], []
    # Detrend lento para reducir deriva de línea de base.
    sig = np.asarray(signal_px, dtype=float)
    sigma = max(2.0, 120.0 * px_per_ms)
    slow = cv2.GaussianBlur(sig.reshape(1, -1).astype(np.float32), (0, 0), sigma).ravel()
    hp = sig - slow
    for onset, _ in qrs_bounds[1:]:
        a = max(0, int(round(onset - 320 * px_per_ms)))
        b = max(a + 1, int(round(onset - 70 * px_per_ms)))
        if b - a < 5:
            continue
        seg = hp[a:b]
        pk = a + int(np.argmax(np.abs(seg)))
        amp_px = float(abs(hp[pk]))
        if amp_px < max(0.55, 0.14 * float(grid_px)):
            continue
        th = max(0.25, 0.22 * amp_px)
        lo, hi = pk, pk
        while lo > a and abs(hp[lo - 1]) >= th:
            lo -= 1
        while hi < b - 1 and abs(hp[hi + 1]) >= th:
            hi += 1
        pdur = (hi - lo + 1) / max(px_per_ms, 1e-6)
        pr = (onset - lo) / max(px_per_ms, 1e-6)
        pamp = amp_px / max(float(grid_px) * 10.0, 1e-6)  # mV a 10 mm/mV
        if 45 <= pdur <= 160 and 80 <= pr <= 320:
            p_durs.append(float(pdur)); prs.append(float(pr)); p_amps.append(float(pamp))
    if len(prs) < 3:
        return {"p_present": True, "p_duration_ms": None, "p_amplitude_mv": None, "pr_ms": None, "confidence": 0.35, "reason": "P reproducible, pero no hubo suficientes complejos con bordes P/PR estables."}
    med_pr = float(np.median(prs)); med_pd = float(np.median(p_durs)); med_pa = float(np.median(p_amps))
    disp = float(np.median(np.abs(np.asarray(prs) - med_pr)) / max(med_pr, 1e-6))
    conf = min(0.96, 0.62 + 0.07 * min(len(prs), 5) + 0.18 * max(0.0, 1.0 - disp / 0.18))
    return {
        "p_present": True, "p_duration_ms": round(med_pd, 1), "p_amplitude_mv": round(med_pa, 3),
        "pr_ms": round(med_pr, 1), "confidence": round(conf, 3), "n": len(prs), "reason": "Mediana de ondas P pre-QRS reproducibles en la tira larga."
    }


def _qt_measurements(signal_px: np.ndarray, qrs_bounds: List[Tuple[int, int]], qrs_positions: List[int], grid_px: float, speed_mm_s: float) -> Dict[str, Any]:
    if len(qrs_bounds) < 4 or len(qrs_positions) < 5:
        return {"qt_ms": None, "qtc_f_ms": None, "qtc_b_ms": None, "confidence": 0.0, "n": 0}
    px_per_ms = float(speed_mm_s) * float(grid_px) / 1000.0
    sig = np.asarray(signal_px, dtype=float)
    slow = cv2.GaussianBlur(sig.reshape(1, -1).astype(np.float32), (0, 0), max(2.0, 180.0 * px_per_ms)).ravel()
    hp = sig - slow
    smooth = _smooth_vector(hp, 1.2)
    der = np.gradient(smooth)
    qt_vals, qtf_vals, qtb_vals = [], [], []
    # Empareja bounds con posiciones por proximidad.
    bound_by_q = []
    for q in qrs_positions:
        if not qrs_bounds:
            continue
        b = min(qrs_bounds, key=lambda z: abs(((z[0] + z[1]) / 2.0) - q))
        if abs(((b[0] + b[1]) / 2.0) - q) <= 2.5 * grid_px:
            bound_by_q.append((q, b))
    for j in range(1, len(bound_by_q) - 1):
        q, (onset, offset) = bound_by_q[j]
        prev_q = bound_by_q[j - 1][0]
        next_q = bound_by_q[j + 1][0]
        rr_ms = (q - prev_q) / max(px_per_ms, 1e-6)
        if rr_ms < 300 or rr_ms > 2000:
            continue
        a = int(round(offset + 90 * px_per_ms))
        b = min(int(round(onset + 520 * px_per_ms)), int(round(next_q - 90 * px_per_ms)))
        if b - a < max(6, int(120 * px_per_ms)):
            continue
        seg = smooth[a:b]
        pk = a + int(np.argmax(np.abs(seg)))
        amp = float(smooth[pk])
        if abs(amp) < max(0.5, 0.12 * grid_px):
            continue
        # Método de tangente sobre la pendiente terminal de T.
        ds1 = min(b, int(round(pk + 180 * px_per_ms)))
        if ds1 <= pk + 2:
            continue
        if amp >= 0:
            s = pk + int(np.argmin(der[pk:ds1]))
        else:
            s = pk + int(np.argmax(der[pk:ds1]))
        slope = float(der[s])
        if abs(slope) < 0.04:
            continue
        tend = float(s - smooth[s] / slope)
        if tend <= pk or tend >= next_q - 50 * px_per_ms:
            continue
        qt = (tend - onset) / max(px_per_ms, 1e-6)
        if not (250 <= qt <= 550):
            continue
        rr_s = rr_ms / 1000.0
        qt_s = qt / 1000.0
        qtf = 1000.0 * qt_s / (rr_s ** (1.0 / 3.0))
        qtb = 1000.0 * qt_s / math.sqrt(rr_s)
        if 250 <= qtf <= 650:
            qt_vals.append(qt); qtf_vals.append(qtf); qtb_vals.append(qtb)
    if len(qt_vals) < 2:
        return {"qt_ms": None, "qtc_f_ms": None, "qtc_b_ms": None, "confidence": 0.25 if qt_vals else 0.0, "n": len(qt_vals)}
    med = float(np.median(qt_vals)); mad = float(np.median(np.abs(np.asarray(qt_vals) - med)))
    conf = min(0.92, 0.50 + 0.08 * min(len(qt_vals), 4) + 0.25 * max(0.0, 1.0 - mad / 55.0))
    return {
        "qt_ms": round(med, 1), "qtc_f_ms": round(float(np.median(qtf_vals)), 1),
        "qtc_b_ms": round(float(np.median(qtb_vals)), 1), "confidence": round(conf, 3),
        "n": len(qt_vals), "mad_ms": round(mad, 1),
    }


def _lead_regions(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """Formato 3x4 estándar. Devuelve regiones internas evitando separadores y rótulos."""
    h, w = rgb.shape[:2]
    x0, x1 = int(round(0.028 * w)), int(round(0.972 * w))
    y0, y1 = int(round(0.045 * h)), int(round(0.752 * h))
    cw = (x1 - x0) / 4.0
    rh = (y1 - y0) / 3.0
    out: Dict[str, np.ndarray] = {}
    for r, row in enumerate(STANDARD_LEADS):
        for c, lead in enumerate(row):
            xa = int(round(x0 + c * cw + 0.025 * cw))
            xb = int(round(x0 + (c + 1) * cw - 0.025 * cw))
            ya = int(round(y0 + r * rh + 0.05 * rh))
            yb = int(round(y0 + (r + 1) * rh - 0.05 * rh))
            out[lead] = rgb[max(0, ya):min(h, yb), max(0, xa):min(w, xb)]
    return out


def _lead_qrs_features(panel: np.ndarray, grid_px: float, speed_mm_s: float, gain_mm_mv: float) -> Dict[str, Any]:
    clean = _trace_mask_adaptive(panel)
    # Rótulos suelen estar en el extremo izquierdo/superior.
    hh, ww = clean.shape
    clean[:max(1, int(0.12 * hh)), :max(1, int(0.22 * ww))] = 0
    s = _signal_from_component(clean)
    if not s.get("ok") or float(s.get("coverage") or 0) < 0.45:
        return {"ok": False, "confidence": float(s.get("coverage") or 0)}
    det = _detect_qrs_columns(s["mask"], grid_px)
    qrs = list(det.get("positions") or [])
    if len(qrs) < 1:
        return {"ok": False, "confidence": 0.25 * float(s.get("coverage") or 0)}
    qb = _measure_qrs_bounds(s["span"], qrs, grid_px, speed_mm_s)
    sig = np.asarray(s["signal_px"], dtype=float)
    nets, sts, t_amps = [], [], []
    for q in qrs:
        if qb.get("bounds"):
            bnd = min(qb["bounds"], key=lambda z: abs(((z[0] + z[1]) / 2.0) - q))
            onset, offset = bnd
        else:
            onset, offset = max(0, q - int(1.2 * grid_px)), min(len(sig)-1, q + int(1.2 * grid_px))
        # Línea de base pre-QRS local (TP/PR aproximada).
        pre0 = max(0, int(round(onset - 5.0 * grid_px)))
        pre1 = max(pre0 + 1, int(round(onset - 1.6 * grid_px)))
        base = float(np.median(sig[pre0:pre1])) if pre1 > pre0 else 0.0
        qseg = sig[onset:offset + 1] - base
        if qseg.size:
            pos = float(np.max(qseg)); neg = float(np.min(qseg))
            nets.append(pos + neg)
        # ST: J + 60 ms.
        px60 = 0.060 * speed_mm_s * grid_px
        sx = int(round(offset + px60))
        if 0 <= sx < len(sig):
            sts.append(float((sig[sx] - base) / max(grid_px * gain_mm_mv, 1e-6)))
        # T amplitud/polaridad aproximada 100-420 ms pos-QRS.
        ta = int(round(offset + 0.10 * speed_mm_s * grid_px))
        tb = min(len(sig), int(round(offset + 0.42 * speed_mm_s * grid_px)))
        if tb - ta >= 3:
            seg = sig[ta:tb] - base
            pk = float(seg[np.argmax(np.abs(seg))])
            t_amps.append(pk / max(grid_px * gain_mm_mv, 1e-6))
    if not nets:
        return {"ok": False, "confidence": 0.3}
    conf = min(0.95, 0.45 + 0.25 * float(s.get("coverage") or 0) + 0.12 * min(len(nets), 3) + 0.12 * float(det.get("confidence") or 0))
    return {
        "ok": True, "confidence": round(conf, 3), "coverage": round(float(s.get("coverage") or 0), 3),
        "net_qrs_px": round(float(np.median(nets)), 3),
        "qrs_ms": qb.get("value_ms"), "qrs_confidence": qb.get("confidence"),
        "st_mv": round(float(np.median(sts)), 3) if sts else None,
        "t_amp_mv": round(float(np.median(t_amps)), 3) if t_amps else None,
        "qrs_count": len(qrs),
    }


def _axis_from_leads(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    vals = {}
    for lead in ("I", "II", "aVF"):
        f = features.get(lead) or {}
        if f.get("ok") and float(f.get("confidence") or 0) >= 0.55 and f.get("net_qrs_px") is not None:
            vals[lead] = float(f["net_qrs_px"])
    if "I" not in vals or "aVF" not in vals:
        return {"axis_deg": None, "axis_label": "NO VALORABLE", "confidence": 0.0}
    i, avf = vals["I"], vals["aVF"]
    # Exige magnitud mínima para no decidir con ruido cercano a cero.
    mag = math.hypot(i, avf)
    if mag < 0.8:
        return {"axis_deg": None, "axis_label": "NO VALORABLE", "confidence": 0.25}
    deg = math.degrees(math.atan2(avf, i))
    if deg > 180:
        deg -= 360
    if deg <= -180:
        deg += 360
    if i >= 0 and avf >= 0:
        label = "NORMAL"
    elif i >= 0 and avf < 0:
        if vals.get("II", 1.0) >= 0:
            label = "LIMÍTROFE IZQUIERDO"
        else:
            label = "DESVIADO A IZQUIERDA"
    elif i < 0 and avf >= 0:
        label = "DESVIADO A DERECHA"
    else:
        label = "EJE EXTREMO"
    conf = min(0.94, 0.58 + 0.16 * min(float(features["I"].get("confidence") or 0), float(features["aVF"].get("confidence") or 0)) + 0.12 * min(1.0, mag / 4.0))
    return {"axis_deg": round(float(deg), 1), "axis_label": label, "confidence": round(conf, 3)}


def _st_t_summary(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    st_values = {k: float(v["st_mv"]) for k, v in features.items() if v.get("ok") and v.get("st_mv") is not None and float(v.get("confidence") or 0) >= 0.55}
    t_values = {k: float(v["t_amp_mv"]) for k, v in features.items() if v.get("ok") and v.get("t_amp_mv") is not None and float(v.get("confidence") or 0) >= 0.55}
    if len(st_values) < 6:
        st_status = "NO VALORABLE"; st_conf = 0.0
    else:
        elev, depr = [], []
        for lead, val in st_values.items():
            # Umbral técnico conservador; V2-V3 se trata con mayor tolerancia porque los criterios diagnósticos dependen de edad/sexo.
            lim = 0.20 if lead in {"V2", "V3"} else 0.10
            if val >= lim: elev.append(lead)
            if val <= -0.10: depr.append(lead)
        if elev:
            st_status = "ELEVACIÓN SIGNIFICATIVA DETECTADA EN " + ", ".join(elev)
        elif depr:
            st_status = "DEPRESIÓN SIGNIFICATIVA DETECTADA EN " + ", ".join(depr)
        else:
            st_status = "ISOELÉCTRICO / SIN DESVIACIÓN SIGNIFICATIVA DETECTADA"
        st_conf = min(0.93, 0.50 + 0.045 * len(st_values))
    core_t = {k: v for k, v in t_values.items() if k in {"I", "II", "V3", "V4", "V5", "V6"}}
    inverted = [k for k, v in core_t.items() if v <= -0.10]
    peaked = []
    for k, v in t_values.items():
        lim = 1.0 if k.startswith("V") else 0.5
        if v >= lim:
            peaked.append(k)
    if len(t_values) < 5:
        t_status = "NO VALORABLE"; t_conf = 0.0
    elif inverted or peaked:
        parts = []
        if inverted: parts.append("T INVERTIDA EN " + ", ".join(inverted))
        if peaked: parts.append("T DE AMPLITUD MARCADAMENTE AUMENTADA EN " + ", ".join(peaked))
        t_status = " · ".join(parts); t_conf = min(0.90, 0.48 + 0.05 * len(t_values))
    else:
        t_status = "SIN INVERSIÓN PATOLÓGICA NI PICOSIDAD MARCADA DETECTADA"
        t_conf = min(0.90, 0.48 + 0.05 * len(t_values))
    return {
        "st_status": st_status, "st_confidence": round(st_conf, 3), "st_by_lead_mv": st_values,
        "t_status": t_status, "t_confidence": round(t_conf, 3), "t_by_lead_mv": t_values,
    }


def _ectopy_summary(qrs_ms_values: List[float], qrs_positions: List[int], grid_px: float, speed_mm_s: float, rhythm: str) -> Dict[str, Any]:
    if len(qrs_positions) < 6:
        return {"status": "NO VALORABLE", "count": None, "confidence": 0.0}
    rr = np.diff(np.asarray(qrs_positions, dtype=float))
    med_rr = float(np.median(rr)) if rr.size else 0.0
    widths = np.asarray(qrs_ms_values, dtype=float) if qrs_ms_values else np.asarray([], dtype=float)
    pvc = 0
    if widths.size >= 5:
        med_w = float(np.median(widths))
        # Solo marca PVC si existe QRS claramente ancho/discrepante; la irregularidad aislada en FA no cuenta como extrasístole.
        for i, w in enumerate(widths[:len(rr)]):
            if w >= 120 and w >= 1.35 * max(med_w, 1.0):
                if rhythm == "FIBRILACION_AURICULAR_PROBABLE" or (i < len(rr) and rr[i] < 0.85 * med_rr):
                    pvc += 1
    if pvc:
        return {"status": f"EXTRASÍSTOLES VENTRICULARES PROBABLES: {pvc}", "count": pvc, "confidence": 0.72}
    if widths.size >= 5:
        return {"status": "SIN EXTRASÍSTOLES VENTRICULARES ANCHAS EVIDENTES EN LA TIRA ANALIZADA", "count": 0, "confidence": 0.72}
    return {"status": "NO VALORABLE", "count": None, "confidence": 0.25}


def _conf_word(v: Optional[float]) -> str:
    x = float(v or 0.0)
    if x >= 0.82: return "ALTA"
    if x >= 0.62: return "MEDIA-ALTA"
    if x >= 0.45: return "MEDIA"
    if x > 0: return "BAJA"
    return "NO MEDIDO"


def analyze_ecg_full_clinical(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None, calibration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Integra ritmo + intervalos + eje + ST/T de forma determinista y trazable.

    V8.3.4 no declara normalidad cuando un parámetro no alcanza confianza mínima.
    """
    q = quality or assess_ecg_photo(image_bytes)
    cal = calibration or detect_calibration_pulse(image_bytes, q)
    rhythm = analyze_rhythm_clinical(image_bytes, q, cal)
    out: Dict[str, Any] = {"schema": "ECG_FULL_CLINICAL_V1", "rhythm": rhythm, "quality": q, "calibration": cal}
    grid = q.get("small_grid_square_px_candidate")
    speed = cal.get("speed_mm_s")
    gain = cal.get("gain_mm_mV")
    if not (grid and speed and gain):
        out.update({
            "measurements_available": False,
            "reason": "Calibración temporal/vertical no validada; solo se conserva la interpretación de ritmo disponible.",
            "report": None,
        })
        return out
    grid = float(grid); speed = float(speed); gain = float(gain)
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    strip, _ = _rhythm_strip_region(rgb)
    clean = _trace_mask_adaptive(strip)
    trace = _signal_from_component(clean)
    # Si la componente larga es mejor que la máscara completa, usa esa también para QRS.
    qdet = _detect_qrs_columns(trace.get("mask") if trace.get("ok") else clean, grid)
    qrs_pos = list(qdet.get("positions") or rhythm.get("qrs_positions_px") or [])
    qrs_m = _measure_qrs_bounds(trace.get("span") if trace.get("ok") else np.zeros(clean.shape[1]), qrs_pos, grid, speed) if trace.get("ok") else {"value_ms": None,"confidence":0,"bounds":[],"values_ms":[]}
    ppr = _p_pr_measurements(
        np.asarray(trace.get("signal_px") if trace.get("signal_px") is not None else [], dtype=float), qrs_m.get("bounds") or [], grid, speed,
        str(rhythm.get("rhythm") or ""), str(rhythm.get("p_organization") or "")
    ) if trace.get("ok") else {"p_present": None,"p_duration_ms":None,"p_amplitude_mv":None,"pr_ms":None,"confidence":0,"reason":"Tira no trazable."}
    qt = _qt_measurements(
        np.asarray(trace.get("signal_px") if trace.get("signal_px") is not None else [], dtype=float), qrs_m.get("bounds") or [], qrs_pos, grid, speed
    ) if trace.get("ok") else {"qt_ms":None,"qtc_f_ms":None,"qtc_b_ms":None,"confidence":0,"n":0}

    lead_features: Dict[str, Dict[str, Any]] = {}
    for lead, panel in _lead_regions(rgb).items():
        lead_features[lead] = _lead_qrs_features(panel, grid, speed, gain)
    axis = _axis_from_leads(lead_features)
    stt = _st_t_summary(lead_features)

    # Consenso QRS: tira larga + derivaciones con medición válida.
    qrs_candidates = []
    if qrs_m.get("value_ms") is not None and float(qrs_m.get("confidence") or 0) >= 0.50:
        qrs_candidates.append(float(qrs_m["value_ms"]))
    for f in lead_features.values():
        if f.get("qrs_ms") is not None and float(f.get("qrs_confidence") or 0) >= 0.50:
            qrs_candidates.append(float(f["qrs_ms"]))
    if qrs_candidates:
        qrs_global = float(np.median(qrs_candidates))
        qrs_mad = float(np.median(np.abs(np.asarray(qrs_candidates) - qrs_global)))
        qrs_conf = min(0.97, 0.58 + 0.035 * min(len(qrs_candidates), 8) + 0.22 * max(0.0, 1.0 - qrs_mad / 25.0))
    else:
        qrs_global = None; qrs_mad = None; qrs_conf = 0.0

    ectopy = _ectopy_summary(qrs_m.get("values_ms") or [], qrs_pos, grid, speed, str(rhythm.get("rhythm") or ""))
    conduction = "NO VALORABLE"
    if qrs_global is not None and qrs_conf >= 0.55:
        if qrs_global < 120:
            conduction = "QRS ESTRECHO; SIN CRITERIO DE BLOQUEO COMPLETO DE RAMA POR DURACIÓN"
        else:
            v1 = lead_features.get("V1") or {}; v6 = lead_features.get("V6") or {}; lead_i = lead_features.get("I") or {}
            if all(x.get("ok") for x in (v1, v6, lead_i)):
                if float(v1.get("net_qrs_px") or 0) < 0 and float(v6.get("net_qrs_px") or 0) > 0 and float(lead_i.get("net_qrs_px") or 0) > 0:
                    conduction = "PATRÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA IZQUIERDA"
                elif float(v1.get("net_qrs_px") or 0) > 0 and float(v6.get("net_qrs_px") or 0) < 0:
                    conduction = "PATRÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA DERECHA"
                else:
                    conduction = "QRS PROLONGADO / TRASTORNO DE CONDUCCIÓN INTRAVENTRICULAR NO CLASIFICADO"
            else:
                conduction = "QRS PROLONGADO; MORFOLOGÍA DE RAMA NO VALORABLE CON CONFIANZA"

    # Confianza clínica global: no depende de una única métrica técnica.
    conf_parts = [float(cal.get("confidence") or 0), float(rhythm.get("qrs_confidence") or 0)]
    for v in (qrs_conf, float(axis.get("confidence") or 0), float(stt.get("st_confidence") or 0), float(stt.get("t_confidence") or 0)):
        if v > 0: conf_parts.append(v)
    clinical_conf = float(np.median(conf_parts)) if conf_parts else 0.0

    out.update({
        "measurements_available": True,
        "heart_rate_bpm": rhythm.get("heart_rate_bpm"),
        "heart_rate_confidence": rhythm.get("heart_rate_confidence"),
        "qrs_ms": round(qrs_global,1) if qrs_global is not None else None,
        "qrs_confidence": round(qrs_conf,3), "qrs_consensus_n": len(qrs_candidates), "qrs_mad_ms": round(qrs_mad,1) if qrs_mad is not None else None,
        "p": ppr, "qt": qt, "axis": axis, "stt": stt, "ectopy": ectopy, "conduction": conduction,
        "lead_features": lead_features,
        "clinical_confidence": round(clinical_conf,3), "clinical_confidence_label": _conf_word(clinical_conf),
    })

    # Formato clínico solicitado: todo parámetro no confiable se declara NO VALORABLE.
    rhy = str(rhythm.get("rhythm") or "")
    if rhy == "FIBRILACION_AURICULAR_PROBABLE":
        rhythm_text = "FIBRILACIÓN AURICULAR; RITMO IRREGULARMENTE IRREGULAR"
        idx = "FIBRILACIÓN AURICULAR"
    elif rhy == "RITMO_SINUSAL_PROBABLE":
        rhythm_text = "SINUSAL Y REGULAR"
        idx = "RITMO SINUSAL"
    else:
        rhythm_text = str(rhythm.get("rhythm_label") or "NO CLASIFICABLE").upper()
        idx = str(rhythm.get("rhythm_label") or "RITMO NO CLASIFICABLE").upper()
    hr = rhythm.get("heart_rate_bpm")
    hr_text = f"{float(hr):.1f} LPM" if hr is not None else "NO MEDIBLE"
    axis_text = str(axis.get("axis_label") or "NO VALORABLE")
    if axis.get("axis_deg") is not None and float(axis.get("confidence") or 0) >= 0.55:
        axis_text += f" ({float(axis['axis_deg']):+.0f}°)"
    if ppr.get("p_present") is False:
        p_text = "NO SE IDENTIFICAN ONDAS P SINUSALES REPRODUCIBLES"
    elif ppr.get("p_duration_ms") is not None and float(ppr.get("confidence") or 0) >= 0.55:
        p_text = f"P REPRODUCIBLE · DURACIÓN {float(ppr['p_duration_ms']):.0f} MS · AMPLITUD ~{float(ppr.get('p_amplitude_mv') or 0):.2f} MV"
    elif ppr.get("p_present") is True:
        p_text = "P REPRODUCIBLE, PERO DURACIÓN/AMPLITUD NO VALORABLES CON CONFIANZA"
    else:
        p_text = "NO VALORABLE"
    pr_text = f"{float(ppr['pr_ms']):.0f} MS" if ppr.get("pr_ms") is not None and float(ppr.get("confidence") or 0) >= 0.55 else ("NO MEDIBLE POR AUSENCIA DE P SINUSAL" if ppr.get("p_present") is False else "NO VALORABLE")
    qrs_text = "NO VALORABLE"
    if qrs_global is not None and qrs_conf >= 0.55:
        qrs_text = f"{qrs_global:.0f} MS · " + ("NO PROLONGADO" if qrs_global < 120 else "PROLONGADO")
    qt_text = "NO VALORABLE"
    if qt.get("qt_ms") is not None and float(qt.get("confidence") or 0) >= 0.50:
        qt_text = f"QT {float(qt['qt_ms']):.0f} MS · QTC FRIDERICIA {float(qt['qtc_f_ms']):.0f} MS · QTC BAZETT {float(qt['qtc_b_ms']):.0f} MS"
    st_text = str(stt.get("st_status") or "NO VALORABLE") if float(stt.get("st_confidence") or 0) >= 0.45 else "NO VALORABLE"
    t_text = str(stt.get("t_status") or "NO VALORABLE") if float(stt.get("t_confidence") or 0) >= 0.45 else "NO VALORABLE"
    ect_text = str(ectopy.get("status") or "NO VALORABLE")

    # Conclusión cuidadosa: no afirma ausencia absoluta de isquemia; describe criterios automatizados.
    conclusions = []
    if rhy == "FIBRILACION_AURICULAR_PROBABLE":
        conclusions.append("RITMO COMPATIBLE CON FIBRILACIÓN AURICULAR")
        if hr is not None:
            conclusions.append(f"RESPUESTA VENTRICULAR PROMEDIO {float(hr):.0f} LPM")
    elif rhy == "RITMO_SINUSAL_PROBABLE":
        conclusions.append("RITMO SINUSAL")
    if qrs_global is not None and qrs_conf >= 0.55:
        conclusions.append("QRS NO PROLONGADO" if qrs_global < 120 else "QRS PROLONGADO")
    if axis.get("axis_label") != "NO VALORABLE" and float(axis.get("confidence") or 0) >= 0.55:
        conclusions.append(f"EJE {axis.get('axis_label')}")
    if "ISOELÉCTRICO" in st_text:
        conclusions.append("SIN DESVIACIÓN SIGNIFICATIVA DEL ST DETECTADA POR EL ALGORITMO")
    if "SIN INVERSIÓN" in t_text:
        conclusions.append("SIN ALTERACIÓN PATOLÓGICA DE T DETECTADA EN DERIVACIONES EVALUABLES")
    if ectopy.get("count") == 0:
        conclusions.append("SIN EXTRASÍSTOLES VENTRICULARES ANCHAS EVIDENTES EN LA TIRA ANALIZADA")
    conclusion_text = ", ".join(conclusions) + "." if conclusions else "INTERPRETACIÓN PARCIAL; EXISTEN PARÁMETROS NO VALORABLES."

    report = "\n".join([
        "INTERPRETACIÓN ELECTROCARDIOGRAMA",
        f"RITMO: {rhythm_text}.",
        f"FC: {hr_text}.",
        f"EJE: {axis_text}.",
        f"ONDA P: {p_text}.",
        f"INTERVALO PR: {pr_text}.",
        f"COMPLEJO QRS: {qrs_text}.",
        f"INTERVALO QT/QTC: {qt_text}.",
        f"SEGMENTO ST: {st_text}.",
        f"ONDA T: {t_text}.",
        f"EXTRASÍSTOLES: {ect_text}.",
        f"CONDUCCIÓN: {conduction}.",
        f"CONCLUSIÓN: {conclusion_text}",
        f"IDX: {idx}.",
    ])
    out["report"] = report
    return out

# --- V8.3.4 refinamientos de confianza y preservación de imagen ----------------

def _cv_to_jpeg_bytes(bgr: np.ndarray, quality: int = 94) -> bytes:
    """Compatibilidad interna: desde V8.3.4 codifica PNG lossless para no degradar retícula/trazo."""
    ok, enc = cv2.imencode('.png', bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not ok:
        raise ValueError('No se pudo codificar la imagen procesada.')
    return enc.tobytes()


def _pil_to_jpeg_bytes(img: Image.Image, quality: int = 94) -> bytes:
    """Compatibilidad interna: PNG lossless para mediciones finas."""
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False, compress_level=3)
    return buf.getvalue()


def prepare_ecg_image(
    image_bytes: bytes,
    *,
    crop_header: bool = False,
    header_fraction: float = 0.0,
    max_dimension: int = 4200,
) -> Tuple[bytes, str, Dict[str, Any]]:
    img = _as_rgb(image_bytes)
    original_size = img.size
    cropped = False
    if crop_header and 0 < header_fraction < 0.30:
        y0 = int(round(img.height * header_fraction))
        if y0 < img.height - 400:
            img = img.crop((0, y0, img.width, img.height)); cropped = True
    scale = min(1.0, float(max_dimension) / max(img.size))
    if scale < 1.0:
        img = img.resize((max(1, int(round(img.width*scale))), max(1, int(round(img.height*scale)))), Image.Resampling.LANCZOS)
    return _pil_to_jpeg_bytes(img), 'image/png', {
        'original_width': original_size[0], 'original_height': original_size[1],
        'processed_width': img.width, 'processed_height': img.height,
        'header_cropped': cropped, 'header_fraction': header_fraction if cropped else 0.0,
        'lossless_processing': True,
    }


def _rhythm_from_component(trace_mask: np.ndarray, qrs: List[int], grid_px: float, cal: Dict[str, Any], base_rhythm: Dict[str, Any]) -> Dict[str, Any]:
    """Rescate de ritmo cuando la máscara global falla pero la componente continua es sólida."""
    if len(qrs) < 6:
        return base_rhythm
    rr = np.diff(np.asarray(qrs, dtype=float))
    med_rr = float(np.median(rr)); mean_rr = float(np.mean(rr))
    rr_cv = float(np.std(rr) / max(mean_rr, 1e-6))
    rr_nmad = float(np.median(np.abs(rr-med_rr)) / max(med_rr,1e-6))
    rr_sdiff = float(np.median(np.abs(np.diff(rr))) / max(med_rr,1e-6)) if rr.size >= 3 else 0.0
    regular_fraction = float(np.mean(np.abs(rr-med_rr)/max(med_rr,1e-6) <= 0.08))
    p_rep = _p_wave_reproducibility(trace_mask, qrs)
    p_org = 'INDETERMINADA' if p_rep is None else ('ORGANIZADA_REPRODUCIBLE' if p_rep >= .62 else 'NO_REPRODUCIBLE' if p_rep <= .42 else 'INDETERMINADA')
    regular = rr_cv <= .085 and rr_nmad <= .07 and rr_sdiff <= .10
    irr_abs = rr_cv >= .13 and rr_nmad >= .09 and rr_sdiff >= .12 and regular_fraction <= .55
    rr_label = 'REGULAR' if regular else 'IRREGULARMENTE_IRREGULAR' if irr_abs else 'IRREGULAR'
    out = dict(base_rhythm)
    out.update({
        'qrs_count': len(qrs), 'qrs_positions_px': [int(x) for x in qrs],
        'qrs_confidence': max(float(out.get('qrs_confidence') or 0), 0.86),
        'rr_pattern': rr_label, 'rr_cv': round(rr_cv,3), 'rr_nmad': round(rr_nmad,3),
        'rr_successive_variation': round(rr_sdiff,3), 'rr_regular_fraction': round(regular_fraction,3),
        'p_organization': p_org, 'p_reproducibility': round(float(p_rep),3) if p_rep is not None else None,
    })
    speed = cal.get('speed_mm_s')
    if speed and grid_px:
        hr = 60.0 * float(speed) * float(grid_px) / max(mean_rr, 1e-6)
        if 20 <= hr <= 300:
            out['heart_rate_bpm'] = round(hr,1); out['heart_rate_confidence'] = 'alta' if float(cal.get('confidence') or 0)>=.75 else 'media'
    if irr_abs and p_org == 'NO_REPRODUCIBLE':
        out.update({
            'rhythm':'FIBRILACION_AURICULAR_PROBABLE','rhythm_label':'Patrón compatible con fibrilación auricular',
            'rhythm_confidence':'alta' if p_rep is not None and p_rep <= .30 else 'media',
            'interpretation':'Respuesta ventricular irregularmente irregular, sin ondas P sinusales reproducibles; patrón compatible con fibrilación auricular. Confirmar visualmente sobre el ECG original.'
        })
    elif regular and p_org == 'ORGANIZADA_REPRODUCIBLE':
        out.update({'rhythm':'RITMO_SINUSAL_PROBABLE','rhythm_label':'Patrón compatible con ritmo sinusal regular','rhythm_confidence':'alta' if p_rep is not None and p_rep>=.75 else 'media'})
    elif irr_abs:
        out.update({'rhythm':'RITMO_IRREGULARMENTE_IRREGULAR','rhythm_label':'Ritmo irregularmente irregular','rhythm_confidence':'media'})
    elif regular:
        out.update({'rhythm':'RITMO_REGULAR_NO_CLASIFICADO','rhythm_label':'Ritmo regular no clasificado','rhythm_confidence':'media'})
    else:
        out.update({'rhythm':'RITMO_IRREGULAR','rhythm_label':'Ritmo irregular','rhythm_confidence':'media'})
    return out


def _axis_from_leads(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    vals: Dict[str,float] = {}
    for lead in ('I','II','III','aVR','aVL','aVF'):
        f=features.get(lead) or {}
        if f.get('ok') and float(f.get('confidence') or 0)>=.55 and f.get('net_qrs_px') is not None:
            vals[lead]=float(f['net_qrs_px'])
    if 'I' not in vals or 'aVF' not in vals:
        return {'axis_deg':None,'axis_label':'NO VALORABLE','confidence':0.0,'consistency':None}
    # Control fisiológico de consistencia entre derivaciones frontales.
    scale=float(np.median([abs(v) for v in vals.values()])) if vals else 0.0
    residuals=[]
    if scale>0 and all(k in vals for k in ('I','II','III')):
        residuals.append(abs(vals['II']-(vals['I']+vals['III']))/scale)
    if scale>0 and all(k in vals for k in ('aVR','aVL','aVF')):
        residuals.append(abs(vals['aVR']+vals['aVL']+vals['aVF'])/scale)
    consistency=float(np.median(residuals)) if residuals else None
    if consistency is not None and consistency>0.55:
        return {'axis_deg':None,'axis_label':'NO VALORABLE','confidence':0.30,'consistency':round(consistency,3), 'reason':'Las amplitudes frontales no cumplen suficiente consistencia vectorial entre derivaciones.'}
    i,avf=vals['I'],vals['aVF']; mag=math.hypot(i,avf)
    if mag<0.8:
        return {'axis_deg':None,'axis_label':'NO VALORABLE','confidence':0.25,'consistency':round(consistency,3) if consistency is not None else None}
    deg=math.degrees(math.atan2(avf,i))
    if i>=0 and avf>=0: label='NORMAL'
    elif i>=0 and avf<0: label='LIMÍTROFE IZQUIERDO' if vals.get('II',1)>=0 else 'DESVIADO A IZQUIERDA'
    elif i<0 and avf>=0: label='DESVIADO A DERECHA'
    else: label='EJE EXTREMO'
    cpen=1.0 if consistency is None else max(0.0,1.0-consistency/0.55)
    conf=min(.94,.55+.18*cpen+.10*min(1.0,mag/4.0))
    return {'axis_deg':round(deg,1),'axis_label':label,'confidence':round(conf,3),'consistency':round(consistency,3) if consistency is not None else None}


def _contiguous_pair_present(leads: List[str], groups: List[set]) -> bool:
    s=set(leads)
    for g in groups:
        if len(s & g)>=2:
            return True
    return False


def _st_t_summary(features: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    st_values={k:float(v['st_mv']) for k,v in features.items() if v.get('ok') and v.get('st_mv') is not None and float(v.get('confidence') or 0)>=.58 and float(v.get('coverage') or 0)>=.60}
    t_values={k:float(v['t_amp_mv']) for k,v in features.items() if v.get('ok') and v.get('t_amp_mv') is not None and float(v.get('confidence') or 0)>=.58 and float(v.get('coverage') or 0)>=.60}
    groups=[{'II','III','aVF'},{'I','aVL','V5','V6'},{'V1','V2','V3','V4'},{'V3','V4','V5','V6'}]
    elev=[];depr=[]
    for lead,val in st_values.items():
        lim=.20 if lead in {'V2','V3'} else .10
        if val>=lim:elev.append(lead)
        if val<=-.10:depr.append(lead)
    if len(st_values)<6:
        st_status='NO VALORABLE';st_conf=0.0
    elif _contiguous_pair_present(elev,groups):
        st_status='ELEVACIÓN SIGNIFICATIVA CONCORDANTE EN '+', '.join(elev);st_conf=min(.94,.55+.04*len(st_values))
    elif _contiguous_pair_present(depr,groups):
        st_status='DEPRESIÓN SIGNIFICATIVA CONCORDANTE EN '+', '.join(depr);st_conf=min(.94,.55+.04*len(st_values))
    else:
        st_status='ISOELÉCTRICO / SIN DESVIACIÓN SIGNIFICATIVA CONCORDANTE EN DERIVACIONES CONTIGUAS';st_conf=min(.92,.52+.04*len(st_values))
    inv=[k for k,v in t_values.items() if k in {'I','II','V3','V4','V5','V6'} and v<=-.10]
    peak=[]
    for k,v in t_values.items():
        lim=1.0 if k.startswith('V') else .5
        if v>=lim:peak.append(k)
    if len(t_values)<5:
        t_status='NO VALORABLE';t_conf=0.0
    elif _contiguous_pair_present(inv,groups) or _contiguous_pair_present(peak,groups):
        parts=[]
        if _contiguous_pair_present(inv,groups):parts.append('T INVERTIDA CONCORDANTE EN '+', '.join(inv))
        if _contiguous_pair_present(peak,groups):parts.append('T DE AMPLITUD MARCADAMENTE AUMENTADA CONCORDANTE EN '+', '.join(peak))
        t_status=' · '.join(parts);t_conf=min(.91,.52+.04*len(t_values))
    else:
        t_status='SIN INVERSIÓN PATOLÓGICA CONCORDANTE NI PICOSIDAD MARCADA DETECTADA';t_conf=min(.90,.50+.04*len(t_values))
    return {'st_status':st_status,'st_confidence':round(st_conf,3),'st_by_lead_mv':st_values,'t_status':t_status,'t_confidence':round(t_conf,3),'t_by_lead_mv':t_values}

# Reemplazo final del integrador para usar rescate de componente continua.
_analyze_ecg_full_clinical_v834_base = analyze_ecg_full_clinical

def analyze_ecg_full_clinical(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None, calibration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    q=quality or assess_ecg_photo(image_bytes); cal=calibration or detect_calibration_pulse(image_bytes,q)
    rgb=np.asarray(_as_rgb(image_bytes),dtype=np.uint8); strip,_=_rhythm_strip_region(rgb); clean=_trace_mask_adaptive(strip); tr=_signal_from_component(clean)
    base_r=analyze_rhythm_clinical(image_bytes,q,cal)
    grid=q.get('small_grid_square_px_candidate')
    if tr.get('ok') and grid:
        d=_detect_qrs_columns(tr['mask'],float(grid)); base_r=_rhythm_from_component(tr['mask'],list(d.get('positions') or []),float(grid),cal,base_r)
    # Llama el integrador base y luego sustituye el ritmo por el más robusto; para que P/PR use el ritmo rescatado,
    # se replica temporalmente mediante una ruta compacta: si difieren, recalcula las secciones dependientes.
    out=_analyze_ecg_full_clinical_v834_base(image_bytes,q,cal)
    old=out.get('rhythm') or {}
    if base_r.get('qrs_count',0) >= old.get('qrs_count',0):
        out['rhythm']=base_r; out['heart_rate_bpm']=base_r.get('heart_rate_bpm'); out['heart_rate_confidence']=base_r.get('heart_rate_confidence')
        # Recalcula P/PR y ectopia con el ritmo definitivo si hay datos geométricos.
        if tr.get('ok') and grid and cal.get('speed_mm_s'):
            d=_detect_qrs_columns(tr['mask'],float(grid)); qp=list(d.get('positions') or [])
            qm=_measure_qrs_bounds(tr['span'],qp,float(grid),float(cal['speed_mm_s']))
            out['p']=_p_pr_measurements(np.asarray(tr['signal_px'],dtype=float),qm.get('bounds') or [],float(grid),float(cal['speed_mm_s']),str(base_r.get('rhythm') or ''),str(base_r.get('p_organization') or ''))
            out['ectopy']=_ectopy_summary(qm.get('values_ms') or [],qp,float(grid),float(cal['speed_mm_s']),str(base_r.get('rhythm') or ''))
    # Reconstruye reporte final con la lógica base pero usando ritmo/P definitivos. Se invoca una pequeña función local para evitar duplicar cálculo.
    rhy=str((out.get('rhythm') or {}).get('rhythm') or ''); rdat=out.get('rhythm') or {}; ppr=out.get('p') or {}; axis=out.get('axis') or {}; qt=out.get('qt') or {}; stt=out.get('stt') or {}; ect=out.get('ectopy') or {}
    qrs=out.get('qrs_ms'); qconf=float(out.get('qrs_confidence') or 0)
    if rhy=='FIBRILACION_AURICULAR_PROBABLE': rhythm_text='FIBRILACIÓN AURICULAR; RITMO IRREGULARMENTE IRREGULAR';idx='FIBRILACIÓN AURICULAR'
    elif rhy=='RITMO_SINUSAL_PROBABLE': rhythm_text='SINUSAL Y REGULAR';idx='RITMO SINUSAL'
    else: rhythm_text=str(rdat.get('rhythm_label') or 'RITMO NO CLASIFICABLE').upper();idx=str(rdat.get('rhythm_label') or 'RITMO NO CLASIFICABLE').upper()
    hr=rdat.get('heart_rate_bpm');hr_text=f'{float(hr):.1f} LPM' if hr is not None else 'NO MEDIBLE'
    axis_text=str(axis.get('axis_label') or 'NO VALORABLE')
    if axis.get('axis_deg') is not None and float(axis.get('confidence') or 0)>=.55:axis_text+=f" ({float(axis['axis_deg']):+.0f}°)"
    if ppr.get('p_present') is False:p_text='NO SE IDENTIFICAN ONDAS P SINUSALES REPRODUCIBLES'
    elif ppr.get('p_duration_ms') is not None and float(ppr.get('confidence') or 0)>=.55:p_text=f"P REPRODUCIBLE · DURACIÓN {float(ppr['p_duration_ms']):.0f} MS · AMPLITUD ~{float(ppr.get('p_amplitude_mv') or 0):.2f} MV"
    elif ppr.get('p_present') is True:p_text='P REPRODUCIBLE, PERO DURACIÓN/AMPLITUD NO VALORABLES CON CONFIANZA'
    else:p_text='NO VALORABLE'
    pr_text=f"{float(ppr['pr_ms']):.0f} MS" if ppr.get('pr_ms') is not None and float(ppr.get('confidence') or 0)>=.55 else ('NO MEDIBLE POR AUSENCIA DE P SINUSAL' if ppr.get('p_present') is False else 'NO VALORABLE')
    qrs_text='NO VALORABLE' if qrs is None or qconf<.55 else f"{float(qrs):.0f} MS · {'NO PROLONGADO' if float(qrs)<120 else 'PROLONGADO'}"
    qt_text='NO VALORABLE' if qt.get('qt_ms') is None or float(qt.get('confidence') or 0)<.50 else f"QT {float(qt['qt_ms']):.0f} MS · QTC FRIDERICIA {float(qt['qtc_f_ms']):.0f} MS · QTC BAZETT {float(qt['qtc_b_ms']):.0f} MS"
    st_text=str(stt.get('st_status') or 'NO VALORABLE') if float(stt.get('st_confidence') or 0)>=.45 else 'NO VALORABLE';t_text=str(stt.get('t_status') or 'NO VALORABLE') if float(stt.get('t_confidence') or 0)>=.45 else 'NO VALORABLE';ect_text=str(ect.get('status') or 'NO VALORABLE')
    conduction=str(out.get('conduction') or 'NO VALORABLE')
    concl=[]
    if rhy=='FIBRILACION_AURICULAR_PROBABLE':concl.append('RITMO COMPATIBLE CON FIBRILACIÓN AURICULAR');
    elif rhy=='RITMO_SINUSAL_PROBABLE':concl.append('RITMO SINUSAL')
    if hr is not None and rhy=='FIBRILACION_AURICULAR_PROBABLE':concl.append(f'RESPUESTA VENTRICULAR PROMEDIO {float(hr):.0f} LPM')
    if qrs is not None and qconf>=.55:concl.append('QRS NO PROLONGADO' if float(qrs)<120 else 'QRS PROLONGADO')
    if axis.get('axis_label')!='NO VALORABLE' and float(axis.get('confidence') or 0)>=.55:concl.append(f"EJE {axis.get('axis_label')}")
    if 'ISOELÉCTRICO' in st_text:concl.append('SIN DESVIACIÓN SIGNIFICATIVA CONCORDANTE DEL ST DETECTADA')
    if 'SIN INVERSIÓN' in t_text:concl.append('SIN ALTERACIÓN PATOLÓGICA CONCORDANTE DE T DETECTADA')
    if ect.get('count')==0:concl.append('SIN EXTRASÍSTOLES VENTRICULARES ANCHAS EVIDENTES')
    conclusion=', '.join(concl)+'.' if concl else 'INTERPRETACIÓN PARCIAL; EXISTEN PARÁMETROS NO VALORABLES.'
    out['report']='\n'.join(['INTERPRETACIÓN ELECTROCARDIOGRAMA',f'RITMO: {rhythm_text}.',f'FC: {hr_text}.',f'EJE: {axis_text}.',f'ONDA P: {p_text}.',f'INTERVALO PR: {pr_text}.',f'COMPLEJO QRS: {qrs_text}.',f'INTERVALO QT/QTC: {qt_text}.',f'SEGMENTO ST: {st_text}.',f'ONDA T: {t_text}.',f'EXTRASÍSTOLES: {ect_text}.',f'CONDUCCIÓN: {conduction}.',f'CONCLUSIÓN: {conclusion}',f'IDX: {idx}.'])
    # Confianza clínica global recalculada con ritmo definitivo.
    vals=[float(cal.get('confidence') or 0),float(base_r.get('qrs_confidence') or 0),float(out.get('qrs_confidence') or 0),float((out.get('axis') or {}).get('confidence') or 0),float((out.get('stt') or {}).get('st_confidence') or 0),float((out.get('stt') or {}).get('t_confidence') or 0)]
    vals=[v for v in vals if v>0];cc=float(np.median(vals)) if vals else 0
    out['clinical_confidence']=round(cc,3);out['clinical_confidence_label']=_conf_word(cc)
    return out

# =============================================================================
# V8.3.5 · GEOMETRÍA ADAPTATIVA · ORIENTACIÓN + 3x4 / 6x2 + BASE LOCAL
# =============================================================================

V835_LAYOUT_3X4 = "3X4_RHYTHM"
V835_LAYOUT_6X2 = "6X2_RHYTHM"


def apply_ecg_rotation(image_bytes: bytes, rotation_deg: int = 0) -> bytes:
    """Rota el ECG en múltiplos de 90° sin interpolación adicional."""
    img = _as_rgb(image_bytes)
    deg = int(rotation_deg) % 360
    if deg == 90:
        img = img.transpose(Image.Transpose.ROTATE_270)  # horario
    elif deg == 180:
        img = img.transpose(Image.Transpose.ROTATE_180)
    elif deg == 270:
        img = img.transpose(Image.Transpose.ROTATE_90)   # antihorario
    return _pil_to_jpeg_bytes(img)


def _v835_text_balance(rgb: np.ndarray) -> float:
    """Heurística determinista: cabeceras impresas suelen aportar más tinta arriba que abajo.

    Solo se usa para decidir entre dos rotaciones horizontales equivalentes. Si la
    diferencia es pequeña la orientación se marca como ambigua y la UI obliga a
    confirmación visual.
    """
    mask = _trace_mask_adaptive(rgb)
    h, w = mask.shape
    # Evita los laterales donde suelen quedar marcas de escáner.
    x0, x1 = int(.08*w), int(.92*w)
    top = float(np.mean(mask[:max(1,int(.16*h)), x0:x1] > 0))
    bot = float(np.mean(mask[int(.84*h):, x0:x1] > 0))
    return top - bot


def auto_orient_ecg(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Orienta el papel para que el tiempo discurra de izquierda a derecha.

    No usa OCR. Primero lleva a paisaje cuando la página está en retrato y luego
    decide entre las dos orientaciones horizontales por distribución de tinta.
    La decisión 0/180 queda explícitamente marcada como ambigua cuando la evidencia
    es baja; el usuario puede corregirla en la interfaz antes de interpretar.
    """
    img = _as_rgb(image_bytes)
    base = np.asarray(img, dtype=np.uint8)
    # Candidatos que dejan la imagen en formato paisaje.
    candidates = []
    for deg in (0, 90, 180, 270):
        b = apply_ecg_rotation(image_bytes, deg)
        arr = np.asarray(_as_rgb(b), dtype=np.uint8)
        h, w = arr.shape[:2]
        landscape = w >= h * 1.08
        # Si todos son casi cuadrados, no penaliza fuerte.
        aspect_bonus = 1.0 if landscape else (0.55 if max(w,h)/max(1,min(w,h)) < 1.12 else 0.0)
        balance = _v835_text_balance(arr)
        score = 1.6 * aspect_bonus + 10.0 * balance
        candidates.append((score, deg, b, arr, balance, landscape))
    candidates.sort(key=lambda x: x[0], reverse=True)
    best, second = candidates[0], candidates[1]
    margin = float(best[0] - second[0])
    # 0 vs 180 o 90 vs 270 pueden ser estructuralmente equivalentes; la confianza
    # se basa más en la diferencia de tinta superior/inferior que en el aspecto.
    conf = max(0.0, min(1.0, 0.45 + 0.35*min(1.0, abs(best[4])/0.02) + 0.20*min(1.0, margin/0.25)))
    ambiguous = bool(abs(best[4]) < 0.006 or margin < 0.08)
    return best[2], {
        "rotation_deg": int(best[1]),
        "confidence": round(conf,3),
        "ambiguous": ambiguous,
        "reason": "Orientación automática por geometría de página y distribución superior/inferior de tinta; sin OCR.",
        "candidate_scores": [{"rotation_deg":int(x[1]),"score":round(float(x[0]),3),"top_bottom_balance":round(float(x[4]),5)} for x in candidates],
    }


def _v835_vertical_dividers(rgb: np.ndarray) -> List[Dict[str, Any]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    edges = cv2.Canny(gray, 55, 145)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180.0,
                            threshold=max(55, int(h*.045)),
                            minLineLength=max(45, int(h*.18)), maxLineGap=max(8,int(h*.018)))
    raw = []
    if lines is not None:
        # OpenCV puede devolver HoughLinesP como (N,1,4), (N,4) o incluso
        # una variante contigua equivalente según versión/plataforma. Normalizar
        # siempre a filas [x1,y1,x2,y2] evita el error "cannot unpack non-iterable numpy.int32 object".
        line_rows = np.asarray(lines, dtype=np.int32).reshape(-1, 4)
        for x1, y1, x2, y2 in line_rows:
            dx, dy = abs(int(x2)-int(x1)), abs(int(y2)-int(y1))
            if dy >= .18*h and dx/max(dy,1) <= .055:
                raw.append((0.5*(int(x1)+int(x2)), min(int(y1),int(y2)), max(int(y1),int(y2)), dy))
    if not raw:
        return []
    raw.sort(key=lambda z:z[0])
    groups: List[List[Tuple[float,int,int,int]]] = []
    for item in raw:
        if not groups or item[0]-groups[-1][-1][0] > max(7,w*.006):
            groups.append([item])
        else:
            groups[-1].append(item)
    out=[]
    for g in groups:
        x=float(np.median([z[0] for z in g])); y0=int(np.min([z[1] for z in g])); y1=int(np.max([z[2] for z in g])); span=(y1-y0)/max(h,1)
        if .05*w < x < .95*w and span >= .18:
            out.append({"x":x,"x_frac":x/w,"y0":y0,"y1":y1,"span_frac":span})
    return out


def detect_ecg_layout(image_bytes: bytes) -> Dict[str, Any]:
    """Distingue formatos 3×4 y 6×2 sin leer etiquetas por OCR."""
    rgb = np.asarray(_as_rgb(image_bytes), dtype=np.uint8)
    h,w=rgb.shape[:2]
    divs=_v835_vertical_dividers(rgb)
    targets=(.25,.50,.75)
    matches=[]
    for t in targets:
        cand=[d for d in divs if abs(d['x_frac']-t)<=.075 and d['span_frac']>=.22]
        if cand:
            matches.append(min(cand,key=lambda d:abs(d['x_frac']-t)))
    if len(matches)>=3:
        layout=V835_LAYOUT_3X4
        # Panel superior: intersección robusta de los tres separadores.
        top=int(np.median([m['y0'] for m in matches])); bottom=int(np.median([m['y1'] for m in matches]))
        # Evita que una línea parcial de cabecera recorte el cuerpo.
        top=max(0,min(top,int(.35*h))); bottom=max(int(.55*h),min(bottom,int(.86*h)))
        confidence=min(.98,.70+.08*len(matches))
    else:
        layout=V835_LAYOUT_6X2
        top=int(.055*h); bottom=int(.86*h)
        confidence=.78 if len(matches)<=1 else .62
    return {
        "layout":layout,"confidence":round(confidence,3),"vertical_dividers":divs,
        "matched_dividers":matches,"standard_top":top,"standard_bottom":bottom,
        "has_long_rhythm_strip":True,
        "description":"3×4 + tira larga" if layout==V835_LAYOUT_3X4 else "6×2 + tira larga / seis filas",
    }


def _v835_local_grid_px(panel: np.ndarray, fallback: Optional[float]) -> Tuple[Optional[float], float]:
    """Escala local para tolerar papel curvado/doblado."""
    if panel.size == 0:
        return fallback, 0.0
    geom=_autocorr_grid_geometry(panel)
    g=geom.get('small_grid_px'); c=float(geom.get('confidence') or 0)
    # Fallback monocromo: periodicidad de proyecciones de intensidad.
    if not g or c<.12:
        gray=cv2.cvtColor(panel,cv2.COLOR_RGB2GRAY).astype(float)
        inv=255.0-gray
        vals=[]
        for axis in (0,1):
            sig=np.mean(inv,axis=axis); sig=sig-np.mean(sig)
            sd=np.std(sig)
            if sd<1e-6: continue
            sig=sig/sd
            n=len(sig); maxlag=min(80,max(10,n//5))
            scores=[]
            for lag in range(3,maxlag+1):
                a,b=sig[:-lag],sig[lag:]
                if len(a)<30: continue
                scores.append((lag,float(np.mean(a*b))))
            strong=[z for z in scores if z[1]>=.22]
            if strong:
                # Menor periodo fuerte compatible con cuadro pequeño o línea mayor.
                vals.append(min(strong,key=lambda z:z[0]))
        if vals:
            lag=float(np.median([v[0] for v in vals])); sc=float(np.median([v[1] for v in vals]))
            # Si la periodicidad visible es la línea de 5 mm, divide por 5 solo cuando
            # ello la aproxima al fallback global.
            candidates=[lag,lag/5.0]
            if fallback:
                cand=min(candidates,key=lambda x:abs(x-float(fallback)))
            else:
                cand=lag/5.0 if lag>12 else lag
            if 1.2<=cand<=35:
                g,c=cand,max(c,min(.65,max(0.0,sc)))
    if g and fallback:
        # Evita saltos locales absurdos por texto/separadores.
        ratio=float(g)/max(float(fallback),1e-6)
        if not .55<=ratio<=1.8:
            return float(fallback), max(.30,c*.4)
    return (float(g) if g else (float(fallback) if fallback else None)), float(c)


def _v835_lead_regions(rgb: np.ndarray, layout_meta: Dict[str,Any]) -> Dict[str,Dict[str,Any]]:
    h,w=rgb.shape[:2]; out: Dict[str,Dict[str,Any]]={}
    layout=layout_meta.get('layout')
    if layout==V835_LAYOUT_3X4:
        divs=layout_meta.get('matched_dividers') or []
        xs=[int(.02*w)] + [int(d['x']) for d in sorted(divs,key=lambda d:d['x'])] + [int(.985*w)]
        if len(xs)!=5:
            xs=[int(.02*w),int(.255*w),int(.50*w),int(.745*w),int(.985*w)]
        y0=int(layout_meta.get('standard_top',.06*h)); y3=int(layout_meta.get('standard_bottom',.76*h)); rh=(y3-y0)/3.0
        for r,row in enumerate(STANDARD_LEADS):
            for c,lead in enumerate(row):
                xa=xs[c]+max(2,int(.025*(xs[c+1]-xs[c]))); xb=xs[c+1]-max(2,int(.02*(xs[c+1]-xs[c])))
                ya=int(y0+r*rh+.04*rh); yb=int(y0+(r+1)*rh-.04*rh)
                out[lead]={"panel":rgb[max(0,ya):min(h,yb),max(0,xa):min(w,xb)],"rect":[xa,ya,xb,yb]}
        ry0=max(y3+2,int(.74*h)); ry1=int(.975*h)
        out['RHYTHM']={"panel":rgb[ry0:ry1,int(.02*w):int(.985*w)],"rect":[int(.02*w),ry0,int(.985*w),ry1]}
    else:
        # 6 filas: I/II/III/aVR/aVL/aVF a izquierda; V1..V6 a derecha.
        # El papel puede curvarse; se usan bandas relativamente amplias y luego una
        # línea de base local dentro de cada panel.
        centers=np.asarray([.115,.245,.375,.505,.635,.765])*h
        row_h=.118*h
        left_labels=['I','II','III','aVR','aVL','aVF']; right_labels=['V1','V2','V3','V4','V5','V6']
        split=int(.49*w)
        for i,c in enumerate(centers):
            ya=max(0,int(c-row_h*.47)); yb=min(h,int(c+row_h*.47))
            out[left_labels[i]]={"panel":rgb[ya:yb,int(.02*w):int(.495*w)],"rect":[int(.02*w),ya,int(.495*w),yb]}
            out[right_labels[i]]={"panel":rgb[ya:yb,int(.505*w):int(.985*w)],"rect":[int(.505*w),ya,int(.985*w),yb]}
        ry0=int(.835*h); ry1=int(.975*h)
        out['RHYTHM']={"panel":rgb[ry0:ry1,int(.02*w):int(.985*w)],"rect":[int(.02*w),ry0,int(.985*w),ry1]}
    return out


def _v835_rhythm_analysis(rgb: np.ndarray, regions: Dict[str,Dict[str,Any]], global_grid: Optional[float], cal: Dict[str,Any]) -> Dict[str,Any]:
    rr=regions.get('RHYTHM') or regions.get('II')
    panel=np.asarray(rr.get('panel'),dtype=np.uint8); rect=rr.get('rect')
    g,gc=_v835_local_grid_px(panel,global_grid)
    clean=_trace_mask_adaptive(panel); tr=_signal_from_component(clean)
    mask=tr.get('mask') if tr.get('ok') else clean
    det=_detect_qrs_columns(mask,g)
    qrs=list(det.get('positions') or [])
    # Reutiliza la lógica de irregularidad/P pero sin la antigua región fija.
    base={"qrs_count":len(qrs),"qrs_positions_px":qrs,"qrs_confidence":float(det.get('confidence') or 0),"grid_small_px":g}
    if len(qrs)>=4:
        d=np.diff(np.asarray(qrs,dtype=float)); med=float(np.median(d)); mean=float(np.mean(d));
        cv=float(np.std(d)/max(mean,1e-6)); nmad=float(np.median(np.abs(d-med))/max(med,1e-6)); sd=float(np.median(np.abs(np.diff(d)))/max(med,1e-6)) if len(d)>=3 else 0.0
        regfrac=float(np.mean(np.abs(d-med)/max(med,1e-6)<=.08))
        prep=_p_wave_reproducibility(mask,qrs)
        p_org='INDETERMINADA' if prep is None else ('ORGANIZADA_REPRODUCIBLE' if prep>=.62 else 'NO_REPRODUCIBLE' if prep<=.38 else 'INDETERMINADA')
        regular=cv<=.085 and nmad<=.07 and sd<=.10
        irr=cv>=.13 and nmad>=.09 and sd>=.12 and regfrac<=.60
        base.update({'rr_cv':round(cv,3),'rr_nmad':round(nmad,3),'rr_successive_variation':round(sd,3),'rr_regular_fraction':round(regfrac,3),'p_reproducibility':round(float(prep),3) if prep is not None else None,'p_organization':p_org,'rr_pattern':'REGULAR' if regular else 'IRREGULARMENTE_IRREGULAR' if irr else 'IRREGULAR'})
        speed=cal.get('speed_mm_s')
        if speed and g:
            hr=60*float(speed)*float(g)/max(float(np.mean(d)),1e-6)
            if 20<=hr<=320: base['heart_rate_bpm']=round(hr,1);base['heart_rate_confidence']='alta' if float(cal.get('confidence') or 0)>=.72 else 'media'
        if irr and p_org=='NO_REPRODUCIBLE':
            base.update({'rhythm':'FIBRILACION_AURICULAR_PROBABLE','rhythm_label':'Patrón compatible con fibrilación auricular','rhythm_confidence':'alta' if prep is not None and prep<=.28 else 'media'})
        elif regular and p_org=='ORGANIZADA_REPRODUCIBLE':
            base.update({'rhythm':'RITMO_SINUSAL_PROBABLE','rhythm_label':'Patrón compatible con ritmo sinusal regular','rhythm_confidence':'alta' if prep is not None and prep>=.72 else 'media'})
        elif regular:
            base.update({'rhythm':'RITMO_REGULAR_NO_CLASIFICADO','rhythm_label':'Ritmo regular no clasificado','rhythm_confidence':'media'})
        elif irr:
            base.update({'rhythm':'RITMO_IRREGULARMENTE_IRREGULAR','rhythm_label':'Ritmo irregularmente irregular','rhythm_confidence':'media'})
        else:
            base.update({'rhythm':'RITMO_IRREGULAR','rhythm_label':'Ritmo irregular','rhythm_confidence':'media'})
    else:
        base.update({'rhythm':'NO_CLASIFICABLE','rhythm_label':'Ritmo no clasificable','rhythm_confidence':'insuficiente'})
    # Marcadores puntuales; no líneas verticales atravesando el papel.
    bgr=cv2.cvtColor(rgb.copy(),cv2.COLOR_RGB2BGR)
    x0,y0,x1,y1=[int(v) for v in rect]
    for x in qrs:
        xx=x0+int(x)
        # busca el punto oscuro más cercano al centro local en ±25% de la banda
        col=panel[:,max(0,min(panel.shape[1]-1,int(x)))]
        gray=cv2.cvtColor(col.reshape(-1,1,3),cv2.COLOR_RGB2GRAY).ravel(); yy=int(np.argmin(gray)) if gray.size else panel.shape[0]//2
        cv2.circle(bgr,(xx,y0+yy),4,(30,40,220),-1,cv2.LINE_AA)
    bbox=cal.get('bbox')
    if bbox:
        bx,by,bw,bh=[int(v) for v in bbox];cv2.rectangle(bgr,(bx,by),(bx+bw,by+bh),(30,160,60),2)
    base['overlay_bytes']=_cv_to_jpeg_bytes(bgr);base['strip_rect']=rect;base['local_grid_confidence']=round(gc,3)
    base['_trace']=tr;base['_panel']=panel;base['_local_grid']=g
    return base


def _v835_conduction(qrs_ms: Optional[float], qconf: float, feats: Dict[str,Dict[str,Any]]) -> Dict[str,Any]:
    if qrs_ms is None or qconf<.55:
        return {'status':'NO VALORABLE','pattern':None,'confidence':0.0}
    if qrs_ms<120:
        return {'status':'QRS NO PROLONGADO; SIN CRITERIO DE BLOQUEO COMPLETO DE RAMA POR DURACIÓN','pattern':'QRS_ESTRECHO','confidence':qconf}
    v1,v6,li=(feats.get('V1') or {}),(feats.get('V6') or {}),(feats.get('I') or {})
    usable=all(f.get('ok') and float(f.get('confidence') or 0)>=.52 for f in (v1,v6,li))
    if usable:
        n1=float(v1.get('net_qrs_px') or 0);n6=float(v6.get('net_qrs_px') or 0);ni=float(li.get('net_qrs_px') or 0)
        if n1<0 and n6>0 and ni>0:
            return {'status':'PATRÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA IZQUIERDA','pattern':'BCRI','confidence':min(.94,.62+.30*qconf)}
        if n1>0 and (n6<0 or ni<0):
            return {'status':'PATRÓN COMPATIBLE CON BLOQUEO COMPLETO DE RAMA DERECHA','pattern':'BCRD','confidence':min(.90,.58+.28*qconf)}
    return {'status':'QRS PROLONGADO / TRASTORNO DE CONDUCCIÓN INTRAVENTRICULAR; MORFOLOGÍA DE RAMA NO CLASIFICADA','pattern':'IVCD','confidence':qconf*.78}


def analyze_ecg_full_clinical(image_bytes: bytes, quality: Optional[Dict[str, Any]] = None, calibration: Optional[Dict[str, Any]] = None, layout_override: Optional[str] = None) -> Dict[str,Any]:
    """Integrador V8.3.5: no asume 3×4, usa retícula y baseline local por región."""
    q=quality or assess_ecg_photo(image_bytes);cal=calibration or detect_calibration_pulse(image_bytes,q)
    rgb=np.asarray(_as_rgb(image_bytes),dtype=np.uint8)
    lm=detect_ecg_layout(image_bytes)
    if layout_override in {V835_LAYOUT_3X4,V835_LAYOUT_6X2}:
        lm=dict(lm);lm['layout']=layout_override;lm['description']='3×4 + tira larga' if layout_override==V835_LAYOUT_3X4 else '6×2 + tira larga / seis filas';lm['confidence']=1.0;lm['manual_override']=True
    regs=_v835_lead_regions(rgb,lm)
    global_grid=q.get('small_grid_square_px_candidate')
    rhythm=_v835_rhythm_analysis(rgb,regs,global_grid,cal)
    out={'schema':'ECG_FULL_CLINICAL_V835','layout':lm,'rhythm':{k:v for k,v in rhythm.items() if not k.startswith('_')},'quality':q,'calibration':cal}
    speed=cal.get('speed_mm_s');gain=cal.get('gain_mm_mV')
    # Si no se detectó pulso, mantiene ritmo pero no inventa ms/mV.
    if not (speed and gain and global_grid):
        out.update({'measurements_available':False,'reason':'Escala temporal/vertical no demostrada; no se emiten PR/QRS/QT/ST en unidades clínicas.','report':None,'lead_features':{}})
        return out
    speed=float(speed);gain=float(gain);global_grid=float(global_grid)
    lead_features={}; local_grids={}
    for lead,item in regs.items():
        if lead=='RHYTHM':continue
        panel=np.asarray(item['panel'],dtype=np.uint8);lg,lc=_v835_local_grid_px(panel,global_grid);local_grids[lead]={'px_per_mm':lg,'confidence':round(lc,3)}
        lead_features[lead]=_lead_qrs_features(panel,float(lg or global_grid),speed,gain)
    # Tira de ritmo: intervalos con escala local.
    tr=rhythm.get('_trace') or {}; rg=float(rhythm.get('_local_grid') or global_grid);qpos=list(rhythm.get('qrs_positions_px') or [])
    if tr.get('ok'):
        qm=_measure_qrs_bounds(tr.get('span'),qpos,rg,speed)
        ppr=_p_pr_measurements(np.asarray(tr.get('signal_px'),dtype=float),qm.get('bounds') or [],rg,speed,str(rhythm.get('rhythm') or ''),str(rhythm.get('p_organization') or ''))
        qt=_qt_measurements(np.asarray(tr.get('signal_px'),dtype=float),qm.get('bounds') or [],qpos,rg,speed)
    else:
        qm={'value_ms':None,'confidence':0,'values_ms':[],'bounds':[]};ppr={'p_present':None,'p_duration_ms':None,'p_amplitude_mv':None,'pr_ms':None,'confidence':0};qt={'qt_ms':None,'qtc_f_ms':None,'qtc_b_ms':None,'confidence':0}
    qvals=[]
    if qm.get('value_ms') is not None and float(qm.get('confidence') or 0)>=.5:qvals.append(float(qm['value_ms']))
    for f in lead_features.values():
        if f.get('qrs_ms') is not None and float(f.get('qrs_confidence') or 0)>=.5:qvals.append(float(f['qrs_ms']))
    if qvals:
        qrs=float(np.median(qvals));mad=float(np.median(np.abs(np.asarray(qvals)-qrs)));qconf=min(.97,.58+.035*min(len(qvals),8)+.22*max(0,1-mad/25))
    else:qrs=None;mad=None;qconf=0.0
    axis=_axis_from_leads(lead_features);stt=_st_t_summary(lead_features);ect=_ectopy_summary(qm.get('values_ms') or [],qpos,rg,speed,str(rhythm.get('rhythm') or ''))
    cond=_v835_conduction(qrs,qconf,lead_features)
    # Evita falsa FA si hay un latido ancho aislado sobre un ritmo por lo demás organizado.
    if str(rhythm.get('rhythm'))=='FIBRILACION_AURICULAR_PROBABLE' and ect.get('count') and float(rhythm.get('p_reproducibility') or 0)>=.30:
        rhythm['rhythm']='RITMO_IRREGULAR';rhythm['rhythm_label']='Ritmo irregular con ectopia posible; FA no confirmada';rhythm['rhythm_confidence']='media'
    confs=[float(cal.get('confidence') or 0),float(rhythm.get('qrs_confidence') or 0),qconf,float(axis.get('confidence') or 0),float(stt.get('st_confidence') or 0),float(stt.get('t_confidence') or 0),float(lm.get('confidence') or 0)]
    confs=[x for x in confs if x>0];cc=float(np.median(confs)) if confs else 0
    rhy=str(rhythm.get('rhythm') or '')
    if rhy=='FIBRILACION_AURICULAR_PROBABLE':rt='FIBRILACIÓN AURICULAR; RITMO IRREGULARMENTE IRREGULAR';idx='FIBRILACIÓN AURICULAR'
    elif rhy=='RITMO_SINUSAL_PROBABLE':rt='SINUSAL Y REGULAR';idx='RITMO SINUSAL'
    else:rt=str(rhythm.get('rhythm_label') or 'NO CLASIFICABLE').upper();idx=rt
    hr=rhythm.get('heart_rate_bpm');hrtxt=f'{float(hr):.1f} LPM' if hr is not None else 'NO MEDIBLE'
    axt=str(axis.get('axis_label') or 'NO VALORABLE');
    if axis.get('axis_deg') is not None and float(axis.get('confidence') or 0)>=.55:axt+=f" ({float(axis['axis_deg']):+.0f}°)"
    if ppr.get('p_present') is False:pt='NO SE IDENTIFICAN ONDAS P SINUSALES REPRODUCIBLES';pr='NO MEDIBLE POR AUSENCIA DE P SINUSAL'
    elif ppr.get('p_duration_ms') is not None and float(ppr.get('confidence') or 0)>=.55:pt=f"P REPRODUCIBLE · DURACIÓN {float(ppr['p_duration_ms']):.0f} MS · AMPLITUD ~{float(ppr.get('p_amplitude_mv') or 0):.2f} MV";pr=f"{float(ppr['pr_ms']):.0f} MS" if ppr.get('pr_ms') is not None else 'NO VALORABLE'
    elif ppr.get('p_present') is True:pt='P REPRODUCIBLE; TAMAÑO NO VALORABLE CON CONFIANZA';pr='NO VALORABLE'
    else:pt='NO VALORABLE';pr='NO VALORABLE'
    qtxt='NO VALORABLE' if qrs is None or qconf<.55 else f"{qrs:.0f} MS · {'NO PROLONGADO' if qrs<120 else 'PROLONGADO'}"
    qtt='NO VALORABLE' if qt.get('qt_ms') is None or float(qt.get('confidence') or 0)<.50 else f"QT {float(qt['qt_ms']):.0f} MS · QTC FRIDERICIA {float(qt['qtc_f_ms']):.0f} MS · QTC BAZETT {float(qt['qtc_b_ms']):.0f} MS"
    stxt=str(stt.get('st_status') or 'NO VALORABLE') if float(stt.get('st_confidence') or 0)>=.45 else 'NO VALORABLE';ttxt=str(stt.get('t_status') or 'NO VALORABLE') if float(stt.get('t_confidence') or 0)>=.45 else 'NO VALORABLE';etxt=str(ect.get('status') or 'NO VALORABLE')
    concl=[]
    if rhy=='FIBRILACION_AURICULAR_PROBABLE':concl.append('RITMO COMPATIBLE CON FIBRILACIÓN AURICULAR')
    elif rhy=='RITMO_SINUSAL_PROBABLE':concl.append('RITMO SINUSAL')
    if qrs is not None and qconf>=.55:concl.append('QRS NO PROLONGADO' if qrs<120 else 'QRS PROLONGADO')
    if cond.get('pattern') in {'BCRI','BCRD'}:concl.append(cond['status'])
    if axis.get('axis_label')!='NO VALORABLE' and float(axis.get('confidence') or 0)>=.55:concl.append('EJE '+str(axis.get('axis_label')))
    if ect.get('count'):concl.append(str(ect.get('status')))
    report='\n'.join(['INTERPRETACIÓN ELECTROCARDIOGRAMA',f'RITMO: {rt}.',f'FC: {hrtxt}.',f'EJE: {axt}.',f'ONDA P: {pt}.',f'INTERVALO PR: {pr}.',f'COMPLEJO QRS: {qtxt}.',f'INTERVALO QT/QTC: {qtt}.',f'SEGMENTO ST: {stxt}.',f'ONDA T: {ttxt}.',f'EXTRASÍSTOLES: {etxt}.',f'CONDUCCIÓN: {cond.get("status")}.',f'CONCLUSIÓN: {", ".join(concl)+"." if concl else "INTERPRETACIÓN PARCIAL; HAY PARÁMETROS NO VALORABLES."}',f'IDX: {idx}.'])
    out.update({'rhythm':{k:v for k,v in rhythm.items() if not k.startswith('_')},'measurements_available':True,'p':ppr,'qt':qt,'qrs_ms':round(qrs,1) if qrs is not None else None,'qrs_confidence':round(qconf,3),'qrs_consensus_n':len(qvals),'qrs_mad_ms':round(mad,1) if mad is not None else None,'axis':axis,'stt':stt,'ectopy':ect,'conduction':cond.get('status'),'conduction_detail':cond,'lead_features':lead_features,'local_grids':local_grids,'clinical_confidence':round(cc,3),'clinical_confidence_label':_conf_word(cc),'report':report})
    return out

# --- V8.3.5 final: retícula robusta y orientación conservadora -----------------

def _v835_grid_spacing(rgb: np.ndarray) -> Dict[str, Any]:
    """Estima px/mm desde periodicidad de líneas de retícula, roja o monocroma.

    A diferencia de la autocorrelación antigua, no interpreta el primer lag corto
    como una línea mayor. Busca picos reales de proyección y usa el modo robusto
    de sus separaciones en ambos ejes.
    """
    r=rgb[...,0].astype(float); g=rgb[...,1].astype(float); b=rgb[...,2].astype(float)
    red_mask=(r-g>12)&(r-b>12)&(r>100)
    red_frac=float(np.mean(red_mask))
    per_axis=[]
    for axis in (0,1):
        if red_frac>.004:
            score=np.clip(r-0.5*(g+b),0,None)
            sig=np.mean(score,axis=axis).astype(np.float32)
        else:
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
            sig=np.mean(255.0-gray,axis=axis).astype(np.float32)
        n=len(sig)
        if n<80: continue
        sigma=max(3.0,n/180.0)
        sm=cv2.GaussianBlur(sig.reshape(1,-1),(0,0),sigmaX=sigma).ravel()
        hp=sig-sm
        thr=float(np.mean(hp)+0.35*np.std(hp))
        cand=[i for i in range(2,n-2) if hp[i]>=thr and hp[i]>=hp[i-1] and hp[i]>=hp[i+1]]
        # Non-max suppression mínimo 3 px.
        sel=[]
        for i in sorted(cand,key=lambda j:float(hp[j]),reverse=True):
            if all(abs(i-k)>=3 for k in sel): sel.append(i)
        sel=sorted(sel)
        if len(sel)<8: continue
        d=np.diff(np.asarray(sel,dtype=float)); d=d[(d>=4)&(d<=40)]
        if len(d)<5: continue
        hist=np.zeros(41,dtype=float)
        for x in d:
            k=int(round(float(x)))
            if 0<=k<len(hist): hist[k]+=1
        smh=np.convolve(hist,np.asarray([1,2,3,2,1],dtype=float),mode='same')
        lo,hi=6,25
        best=int(lo+np.argmax(smh[lo:hi+1]))
        near=d[np.abs(d-best)<=2]
        if len(near)<3: continue
        spacing=float(np.median(near)); support=float(len(near)/max(len(d),1))
        per_axis.append((spacing,support))
    if not per_axis:
        return {'small_grid_px':None,'confidence':0.0,'red_fraction':round(red_frac,4)}
    spacing=float(np.median([x[0] for x in per_axis]))
    agreement=1.0 if len(per_axis)<2 else max(0.0,1.0-abs(per_axis[0][0]-per_axis[1][0])/max(spacing,1e-6))
    support=float(np.mean([x[1] for x in per_axis]))
    conf=max(0.0,min(1.0,.42+.20*len(per_axis)+.20*agreement+.18*min(1.0,support/.35)))
    return {'small_grid_px':round(spacing,3),'confidence':round(conf,3),'red_fraction':round(red_frac,4),'axis_estimates':[round(x[0],3) for x in per_axis],'axis_support':[round(x[1],3) for x in per_axis]}


def assess_ecg_photo(image_bytes: bytes) -> Dict[str, Any]:
    img=_as_rgb(image_bytes)
    scale=min(1.0,2400.0/max(img.size))
    if scale<1:
        img=img.resize((max(1,int(round(img.width*scale))),max(1,int(round(img.height*scale)))),Image.Resampling.BILINEAR)
    rgb=np.asarray(img,dtype=np.uint8)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
    p5,p95=np.percentile(gray,[5,95]);contrast=float(p95-p5);sharpness=_laplacian_variance(gray)
    geom=_v835_grid_spacing(rgb); grid_conf=float(geom.get('confidence') or 0); small=geom.get('small_grid_px')
    resolution_score=min(1.0,min(img.width,img.height)/800.0);contrast_score=min(1.0,max(0.0,(contrast-22)/90));sharp_score=min(1.0,max(0.0,math.log1p(max(sharpness,0))/math.log1p(1200)))
    score=int(round(100*max(0,min(1,.34*resolution_score+.22*contrast_score+.22*sharp_score+.22*grid_conf))))
    label='ALTA' if score>=80 else 'ADECUADA' if score>=60 else 'LIMITADA' if score>=42 else 'INSUFICIENTE'
    issues=[]
    if min(img.width,img.height)<550:issues.append('Resolución limitada para mediciones finas.')
    if contrast<45:issues.append('Contraste reducido entre trazado y papel.')
    if sharpness<30:issues.append('Posible desenfoque o movimiento.')
    if not small or grid_conf<.35:issues.append('Retícula no demostrada con suficiente confianza para convertir píxeles a milímetros.')
    return {'schema':'ECG_PHOTO_DETERMINISTIC_V835','width':img.width,'height':img.height,'aspect_ratio':round(img.width/max(img.height,1),3),'quality_score':score,'quality_label':label,'contrast_range':round(contrast,1),'sharpness_index':round(sharpness,1),'grid_kind':'red_grid' if geom.get('red_fraction',0)>.004 else 'monochrome_grid','small_grid_square_px_candidate':small,'major_grid_square_px_candidate':round(float(small)*5,3) if small else None,'grid_confidence':grid_conf,'grid_geometry':geom,'perspective_variation':0.0,'digitization_allowed':bool(score>=42 and small),'precision_measurements_allowed':bool(score>=60 and grid_conf>=.55 and small),'rhythm_analysis_allowed':bool(score>=40),'issues':issues}


def _v835_local_grid_px(panel: np.ndarray, fallback: Optional[float]) -> Tuple[Optional[float], float]:
    geom=_v835_grid_spacing(panel);g=geom.get('small_grid_px');c=float(geom.get('confidence') or 0)
    if g and fallback:
        ratio=float(g)/max(float(fallback),1e-6)
        if not .55<=ratio<=1.80:
            return float(fallback),max(.30,c*.4)
    return (float(g) if g else (float(fallback) if fallback else None)),c


def auto_orient_ecg(image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """Orientación automática conservadora.

    - Si ya viene apaisado, NO aplica 180° automáticamente: conserva el original.
    - Si viene en retrato, compara 90° horario vs antihorario con calibración y
      balance de tinta. La UI siempre permite corregir manualmente.
    """
    img=_as_rgb(image_bytes);w,h=img.size
    if w>=h*1.08:
        # Mantiene la orientación de origen; evita invertir ECG ya correctos.
        return _pil_to_jpeg_bytes(img),{'rotation_deg':0,'confidence':0.72,'ambiguous':True,'reason':'La página ya está apaisada; V8.3.5 conserva su orientación y solicita verificación visual antes de interpretar.','candidate_scores':[]}
    candidates=[]
    for deg in (90,270):
        b=apply_ecg_rotation(image_bytes,deg);arr=np.asarray(_as_rgb(b),dtype=np.uint8)
        q=assess_ecg_photo(b);cal=detect_calibration_pulse(b,q);bal=_v835_text_balance(arr)
        cal_score=(1.0 if cal.get('detected') else 0.0)+float(cal.get('confidence') or 0)
        # Si ambas opciones son plausibles, el balance superior/inferior decide.
        score=1.5*cal_score+6.0*bal+0.25*float(q.get('grid_confidence') or 0)
        candidates.append((score,deg,b,bal,cal,q))
    candidates.sort(key=lambda x:x[0],reverse=True);best,second=candidates[0],candidates[1]
    margin=float(best[0]-second[0]);amb=bool(margin<.18)
    conf=max(.45,min(.97,.62+.22*min(1,margin/.5)+.15*float(best[4].get('confidence') or 0)))
    return best[2],{'rotation_deg':int(best[1]),'confidence':round(conf,3),'ambiguous':amb,'reason':'Página en retrato: orientación elegida comparando calibración, retícula y distribución de tinta; sin OCR.','candidate_scores':[{'rotation_deg':int(x[1]),'score':round(float(x[0]),3),'calibration_detected':bool(x[4].get('detected')),'calibration_confidence':round(float(x[4].get('confidence') or 0),3),'top_bottom_balance':round(float(x[3]),5)} for x in candidates]}

# --- V8.3.5 final geometry: panel borders for 3x4 -----------------------------

def _v835_horizontal_borders(rgb: np.ndarray) -> List[Dict[str,Any]]:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); h,w=gray.shape
    binary=(gray<175).astype(np.uint8)*255
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(max(40,int(.15*w)),1))
    ho=cv2.morphologyEx(binary,cv2.MORPH_OPEN,k)
    dens=(ho>0).sum(axis=1)/max(w,1)
    idx=np.where(dens>.155)[0]
    groups=[]
    if len(idx):
        s=pr=int(idx[0])
        for y in idx[1:]:
            y=int(y)
            if y-pr>5:
                groups.append((s,pr,float(np.max(dens[s:pr+1]))));s=y
            pr=y
        groups.append((s,pr,float(np.max(dens[s:pr+1]))))
    merged=[]
    for a,b,d in groups:
        y=.5*(a+b)
        if merged and y-merged[-1]['y']<.018*h:
            if d>merged[-1]['density']:
                merged[-1]={'y':y,'density':d}
        else:
            merged.append({'y':y,'density':d})
    return merged


def _v835_choose_3x4_borders(rgb: np.ndarray) -> Optional[List[float]]:
    h=rgb.shape[0]; c=[x['y'] for x in _v835_horizontal_borders(rgb) if .04*h<x['y']<.97*h]
    if len(c)<4:return None
    # Busca secuencia de 5 bordes aproximadamente equiespaciados: top, 3 filas, rhythm bottom.
    best=None
    from itertools import combinations
    for seq in combinations(c,5):
        arr=np.asarray(seq,dtype=float); ds=np.diff(arr); med=float(np.median(ds))
        if med<.10*h or med>.28*h: continue
        cv=float(np.std(ds)/max(np.mean(ds),1e-6))
        span=(arr[-1]-arr[0])/h
        if not (.55<=span<=.92): continue
        score=1.0-cv-.15*abs(span-.72)
        if best is None or score>best[0]:best=(score,arr)
    if best is None:
        # 4 bordes: infer top or bottom from median spacing.
        for seq in combinations(c,4):
            arr=np.asarray(seq,dtype=float);ds=np.diff(arr);med=float(np.median(ds));cv=float(np.std(ds)/max(np.mean(ds),1e-6))
            if med<.10*h or med>.28*h or cv>.25:continue
            cand1=np.r_[arr[0]-med,arr];cand2=np.r_[arr,arr[-1]+med]
            for arr5 in (cand1,cand2):
                if arr5[0]>=0 and arr5[-1]<=h:
                    span=(arr5[-1]-arr5[0])/h;score=.8-cv-.15*abs(span-.72)
                    if best is None or score>best[0]:best=(score,arr5)
    return [float(x) for x in best[1]] if best else None


def detect_ecg_layout(image_bytes: bytes) -> Dict[str, Any]:
    rgb=np.asarray(_as_rgb(image_bytes),dtype=np.uint8);h,w=rgb.shape[:2]
    divs=_v835_vertical_dividers(rgb);targets=(.25,.50,.75);matches=[]
    for t in targets:
        cand=[d for d in divs if abs(d['x_frac']-t)<=.08 and d['span_frac']>=.20]
        if cand:matches.append(min(cand,key=lambda d:abs(d['x_frac']-t)))
    borders=_v835_choose_3x4_borders(rgb) if len(matches)>=2 else None
    is3=bool(len(matches)>=3 or (len(matches)>=2 and borders is not None))
    if is3:
        layout=V835_LAYOUT_3X4;conf=min(.98,.70+.06*len(matches)+(.08 if borders else 0))
        if borders:
            top,b1,b2,b3,b4=[int(round(x)) for x in borders]
        else:
            top=int(np.median([m['y0'] for m in matches]));b3=int(np.median([m['y1'] for m in matches]));rh=(b3-top)/3;b1=int(top+rh);b2=int(top+2*rh);b4=min(h-1,int(b3+rh))
    else:
        layout=V835_LAYOUT_6X2;conf=.78 if len(matches)<=1 else .62
        top,b1,b2,b3,b4=int(.04*h),None,None,int(.82*h),int(.98*h)
    return {'layout':layout,'confidence':round(conf,3),'vertical_dividers':divs,'matched_dividers':matches,'horizontal_borders':borders,'standard_top':top,'standard_row_borders':[top,b1,b2,b3] if is3 else None,'standard_bottom':b3,'rhythm_bottom':b4,'has_long_rhythm_strip':True,'description':'3×4 + tira larga' if is3 else '6×2 + tira larga / seis filas'}


def _v835_lead_regions(rgb: np.ndarray, layout_meta: Dict[str,Any]) -> Dict[str,Dict[str,Any]]:
    h,w=rgb.shape[:2];out={};layout=layout_meta.get('layout')
    if layout==V835_LAYOUT_3X4:
        divs=layout_meta.get('matched_dividers') or [];xs=[int(.015*w)]+[int(d['x']) for d in sorted(divs,key=lambda d:d['x'])]+[int(.99*w)]
        if len(xs)!=5:xs=[int(.015*w),int(.255*w),int(.50*w),int(.745*w),int(.99*w)]
        borders=layout_meta.get('horizontal_borders')
        if borders and len(borders)==5:
            ys=[int(round(x)) for x in borders]
        else:
            y0=int(layout_meta.get('standard_top',.08*h));y3=int(layout_meta.get('standard_bottom',.73*h));rh=(y3-y0)/3;ys=[y0,int(y0+rh),int(y0+2*rh),y3,int(layout_meta.get('rhythm_bottom',min(h-1,y3+rh)))]
        for r,row in enumerate(STANDARD_LEADS):
            for c,lead in enumerate(row):
                xa=xs[c]+max(3,int(.018*(xs[c+1]-xs[c])));xb=xs[c+1]-max(3,int(.018*(xs[c+1]-xs[c])))
                ya=ys[r]+max(3,int(.055*(ys[r+1]-ys[r])));yb=ys[r+1]-max(3,int(.055*(ys[r+1]-ys[r])))
                out[lead]={'panel':rgb[ya:yb,xa:xb],'rect':[xa,ya,xb,yb]}
        ry0=ys[3]+max(3,int(.06*(ys[4]-ys[3])));ry1=ys[4]-max(3,int(.06*(ys[4]-ys[3])))
        out['RHYTHM']={'panel':rgb[ry0:ry1,int(.015*w):int(.99*w)],'rect':[int(.015*w),ry0,int(.99*w),ry1]}
    else:
        # Six standard rows + optional rhythm row; template follows common Bionet/GE printouts.
        centers=np.asarray([.115,.245,.375,.505,.635,.765])*h;row_h=.115*h
        left=['I','II','III','aVR','aVL','aVF'];right=['V1','V2','V3','V4','V5','V6']
        for i,c in enumerate(centers):
            ya=max(0,int(c-row_h*.47));yb=min(h,int(c+row_h*.47))
            out[left[i]]={'panel':rgb[ya:yb,int(.015*w):int(.495*w)],'rect':[int(.015*w),ya,int(.495*w),yb]}
            out[right[i]]={'panel':rgb[ya:yb,int(.505*w):int(.99*w)],'rect':[int(.505*w),ya,int(.99*w),yb]}
        ry0=int(.835*h);ry1=int(.965*h)
        out['RHYTHM']={'panel':rgb[ry0:ry1,int(.015*w):int(.99*w)],'rect':[int(.015*w),ry0,int(.99*w),ry1]}
    return out
