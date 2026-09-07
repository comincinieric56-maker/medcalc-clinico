from __future__ import annotations
import json, math, hashlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageFilter

ORIENT_CLASSES = [0, 90, 180, 270]
LAYOUT_CLASSES = ['3x4', '6x2']
FEATURE_VERSION = 'V9_SHADOW_FEATURES_001'
MODEL_NAME = 'v9_shadow_orientation_layout'


def rotate_clockwise(img: Image.Image, deg: int) -> Image.Image:
    d = int(deg) % 360
    if d == 90:
        return img.transpose(Image.Transpose.ROTATE_270)
    if d == 180:
        return img.transpose(Image.Transpose.ROTATE_180)
    if d == 270:
        return img.transpose(Image.Transpose.ROTATE_90)
    return img.copy()


def _resize_gray(img: Image.Image, size=(24, 24)) -> np.ndarray:
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    a = np.asarray(g.resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    # ink = high values, paper = low values
    return 1.0 - a


def extract_features(img: Image.Image) -> np.ndarray:
    """Lightweight visual descriptor for online shadow learning.

    It intentionally contains no OCR and no clinical interpretation. It captures
    coarse ink geometry, edge distribution and horizontal/vertical projections.
    """
    x = _resize_gray(img, (24, 24))
    # simple gradients on the 24x24 ink map
    gx = np.zeros_like(x); gy = np.zeros_like(x)
    gx[:, 1:-1] = np.abs(x[:, 2:] - x[:, :-2]) * 0.5
    gy[1:-1, :] = np.abs(x[2:, :] - x[:-2, :]) * 0.5
    edge = np.sqrt(gx * gx + gy * gy)
    # projections
    row = x.mean(axis=1)
    col = x.mean(axis=0)
    erow = edge.mean(axis=1)
    ecol = edge.mean(axis=0)
    # block pooled structure 8x8
    pooled = np.asarray(Image.fromarray((x * 255).astype(np.uint8)).resize((8,8), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    feat = np.concatenate([x.ravel(), edge.ravel(), row, col, erow, ecol, pooled.ravel()]).astype(np.float32)
    # normalize vector energy while preserving zero-like paper pages
    mu = float(feat.mean()); sd = float(feat.std())
    if sd > 1e-6:
        feat = (feat - mu) / sd
    return feat


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40, 40))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _fit_softmax(X: np.ndarray, y: np.ndarray, n_classes: int, epochs: int = 260, lr: float = 0.08, l2: float = 1e-3) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float64)
    b = np.zeros((n_classes,), dtype=np.float64)
    Y = np.zeros((n, n_classes), dtype=np.float64)
    Y[np.arange(n), y] = 1.0
    # class balancing to avoid domination by one layout early on
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.ones(n, dtype=np.float64)
    for c in range(n_classes):
        if counts[c] > 0:
            weights[y == c] = n / (n_classes * counts[c])
    weights = weights / max(weights.mean(), 1e-9)
    for _ in range(int(epochs)):
        P = softmax(X @ W + b)
        E = (P - Y) * weights[:, None]
        gW = (X.T @ E) / n + l2 * W
        gb = E.mean(axis=0)
        W -= lr * gW
        b -= lr * gb
    P = softmax(X @ W + b)
    acc = float((P.argmax(axis=1) == y).mean())
    return {'W': W.astype(np.float32).tolist(), 'b': b.astype(np.float32).tolist(), 'train_accuracy': acc, 'n': int(n), 'd': int(d)}


def _predict(model: Dict[str, Any], feat: np.ndarray) -> Tuple[int, float, List[float]]:
    if not model or not model.get('W'):
        return 0, 0.0, []
    W = np.asarray(model['W'], dtype=np.float32)
    b = np.asarray(model['b'], dtype=np.float32)
    p = softmax((feat.astype(np.float32) @ W + b)[None, :])[0]
    idx = int(np.argmax(p))
    return idx, float(p[idx]), [float(v) for v in p]


def make_training_feature_record(img: Image.Image, *, source_sha256: str, page_number: int, orientation_deg: int, layout: str, exclude: bool = False) -> Dict[str, Any]:
    orientation_deg = int(orientation_deg) % 360
    views = {}
    labels = {}
    for r in ORIENT_CLASSES:
        view = rotate_clockwise(img, r)
        views[str(r)] = extract_features(view).tolist()
        labels[str(r)] = int((orientation_deg - r) % 360)
    upright = rotate_clockwise(img, orientation_deg)
    return {
        'source_sha256': source_sha256,
        'page_number': int(page_number),
        'feature_version': FEATURE_VERSION,
        'orientation_views': views,
        'orientation_labels': labels,
        'layout_features': extract_features(upright).tolist(),
        'orientation_deg': orientation_deg,
        'layout_code': layout if layout in LAYOUT_CLASSES else None,
        'exclude_from_training': bool(exclude),
    }


def rebuild_state(rows: List[Dict[str, Any]], previous_version: int = 0) -> Dict[str, Any]:
    Xo=[]; yo=[]; Xl=[]; yl=[]; n_cases=0
    for r in rows:
        if r.get('exclude_from_training'):
            continue
        if r.get('feature_version') != FEATURE_VERSION:
            continue
        views = r.get('orientation_views') or {}
        labels = r.get('orientation_labels') or {}
        for key, vec in views.items():
            lab = labels.get(key)
            if lab in ORIENT_CLASSES:
                Xo.append(vec); yo.append(ORIENT_CLASSES.index(int(lab)))
        layout = r.get('layout_code')
        lf = r.get('layout_features')
        if layout in LAYOUT_CLASSES and lf:
            Xl.append(lf); yl.append(LAYOUT_CLASSES.index(layout))
        n_cases += 1
    state = {
        'schema': 'MEDCALC_ECG_SHADOW_MODEL_V9_5',
        'model_name': MODEL_NAME,
        'model_version': int(previous_version) + 1,
        'feature_version': FEATURE_VERSION,
        'n_cases': int(n_cases),
        'orientation_model': None,
        'layout_model': None,
        'metrics': {},
    }
    if Xo:
        om = _fit_softmax(np.asarray(Xo), np.asarray(yo), 4)
        state['orientation_model'] = om
        state['metrics']['orientation_train_accuracy'] = om['train_accuracy']
        state['metrics']['orientation_training_views'] = om['n']
    if Xl:
        lm = _fit_softmax(np.asarray(Xl), np.asarray(yl), 2)
        state['layout_model'] = lm
        state['metrics']['layout_train_accuracy'] = lm['train_accuracy']
        state['metrics']['layout_training_cases'] = lm['n']
    return state


def predict_shadow(img: Image.Image, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not state or int(state.get('n_cases') or 0) < 1:
        return {
            'available': False,
            'reason': 'Aún no hay ECG auditados suficientes para el aprendiz sombra.',
            'model_version': int((state or {}).get('model_version') or 0),
            'n_cases': int((state or {}).get('n_cases') or 0),
        }
    om = state.get('orientation_model')
    if not om:
        return {'available': False, 'reason':'El modelo de orientación aún no está entrenado.', 'model_version':state.get('model_version',0), 'n_cases':state.get('n_cases',0)}
    oi, oc, op = _predict(om, extract_features(img))
    orientation = ORIENT_CLASSES[oi]
    upright_pred = rotate_clockwise(img, orientation)
    layout = None; lc = 0.0; lp=[]
    if state.get('layout_model'):
        li, lc, lp = _predict(state['layout_model'], extract_features(upright_pred))
        layout = LAYOUT_CLASSES[li]
    return {
        'available': True,
        'mode': 'SHADOW_ONLY_NOT_CLINICAL',
        'model_version': int(state.get('model_version') or 0),
        'n_cases': int(state.get('n_cases') or 0),
        'orientation_deg': orientation,
        'orientation_confidence': oc,
        'orientation_probabilities': {str(c): op[i] for i,c in enumerate(ORIENT_CLASSES)} if op else {},
        'layout': layout,
        'layout_confidence': lc,
        'layout_probabilities': {c: lp[i] for i,c in enumerate(LAYOUT_CLASSES)} if lp else {},
        'metrics': state.get('metrics') or {},
    }
