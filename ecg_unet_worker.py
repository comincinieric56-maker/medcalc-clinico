from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ecg_layout_detector import (
    build_rows_from_signal_probability,
    build_weighted_rows_counterfactual,
    canonicalize_extracted_rows,
    detect_ecg_layout,
    detect_rows_from_signal_probability,
    recover_rhythm_center_from_preflight,
    recover_signal_geometry_from_preflight,
    route_layout_hypotheses,
    route_temporal_rhythm_reference,
)

from ecg_signal_reconstruction import (
    canonical_to_worker_payload,
    reconstruct_canonical_ecg,
)
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_signal_report_adapter import build_signal_primary_structured_report


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

LOW_MEMORY_RESAMPLE_SIZE = int(os.environ.get("MEDCALC_ECG_LOW_MEMORY_RESAMPLE", "1200"))
HIGH_FIDELITY_RESAMPLE_SIZE = int(os.environ.get("MEDCALC_ECG_HIGH_FIDELITY_RESAMPLE", "2000"))
LOW_MEMORY_IMAGE_MAX_DIM = max(1400, LOW_MEMORY_RESAMPLE_SIZE + 100)
HIGH_FIDELITY_IMAGE_MAX_DIM = max(2300, HIGH_FIDELITY_RESAMPLE_SIZE + 300)


class ForcedLayoutCorroborationError(RuntimeError):
    """Raised when geometry is not independently corroborated by extracted rows."""


def _centerline_audit_overlay_png_base64(
    signal_prob: np.ndarray,
    physical_rows: np.ndarray,
) -> str | None:
    """Render selected centerlines over the aligned/dewarped U-Net probability map."""
    prob = np.asarray(signal_prob, dtype=np.float32)
    rows = np.asarray(physical_rows, dtype=np.float64)
    if prob.ndim != 2 or rows.ndim != 2 or prob.size == 0:
        return None

    lo = float(np.nanpercentile(prob, 2))
    hi = float(np.nanpercentile(prob, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, max(1.0, float(np.nanmax(prob)))
    gray = np.clip((prob - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (255.0 - 215.0 * gray).astype(np.uint8)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")

    max_width = 1200
    scale = min(1.0, max_width / max(1, image.width))
    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    draw = ImageDraw.Draw(image)
    source_width = rows.shape[1]
    sx = image.width / max(1, source_width)
    sy = image.height / max(1, prob.shape[0])
    palette = [
        (220, 35, 35),
        (20, 110, 210),
        (20, 155, 90),
        (185, 80, 190),
        (225, 130, 20),
        (20, 160, 170),
        (110, 70, 210),
        (190, 70, 90),
        (60, 120, 60),
        (50, 80, 180),
        (165, 105, 25),
        (20, 140, 130),
        (230, 30, 140),
    ]
    for r, line in enumerate(rows):
        finite = np.isfinite(line)
        transitions = np.diff(
            np.r_[False, finite, False].astype(np.int8)
        )
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        color = palette[r % len(palette)]
        for a, b in zip(starts, ends):
            if b - a < 2:
                continue
            step = max(1, int((b - a) / 1200))
            xs = np.arange(a, b, step, dtype=int)
            pts = [
                (float(x * sx), float(line[x] * sy))
                for x in xs
                if np.isfinite(line[x])
            ]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _prepare_source_image(
    source: Path,
    page_index: int,
    destination: Path,
    *,
    max_dimension: int = LOW_MEMORY_IMAGE_MAX_DIM,
    pdf_dpi: float = 150.0,
) -> dict:
    ext = source.suffix.lower()

    if ext == ".pdf":
        import fitz

        doc = fitz.open(stream=source.read_bytes(), filetype="pdf")
        try:
            if getattr(doc, "needs_pass", False):
                raise RuntimeError("PDF protegido con contraseña.")
            if doc.page_count < 1:
                raise RuntimeError("PDF sin páginas.")
            if not 0 <= page_index < doc.page_count:
                raise RuntimeError(
                    f"Página PDF fuera de rango: {page_index + 1}/{doc.page_count}."
                )
            page = doc.load_page(page_index)
            # Rasterize only to the resolution required by the selected
            # inference path. High-fidelity 6x2 uses a larger on-disk image,
            # while the fallback neural-layout path stays deliberately compact.
            pix = page.get_pixmap(
                matrix=fitz.Matrix(float(pdf_dpi) / 72.0, float(pdf_dpi) / 72.0),
                alpha=False,
            )
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            page_count = int(doc.page_count)
        finally:
            doc.close()
        source_type = "pdf"
    else:
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
        source_type = "image"
        page_count = 1

    original_size = image.size

    # Bound the decoded RGB tensor before Torch. The high-fidelity path is
    # created only after a high-confidence preflight and remains in the worker
    # subprocess, so Streamlit never retains the large raster itself.
    max_dimension = int(max_dimension)
    scale = min(1.0, float(max_dimension) / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.LANCZOS,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)

    return {
        "source_type": source_type,
        "pdf_page_index": int(page_index) if source_type == "pdf" else None,
        "pdf_page_count": page_count,
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "processed_width": int(image.width),
        "processed_height": int(image.height),
        "max_dimension_requested": int(max_dimension),
        "pdf_dpi": float(pdf_dpi) if source_type == "pdf" else None,
    }


def _normalize_high_fidelity_input(
    image_path: Path,
    layout_preflight: dict,
) -> dict:
    """Normalize only acquisition geometry before U-Net; never alter signal QC."""
    image = Image.open(image_path).convert("RGB")
    before = [int(image.width), int(image.height)]
    rotation = float(layout_preflight.get("rotation_deg") or 0.0)
    applied_rotation = 0.0

    # The preflight estimator uses long paper-grid lines, not waveform slope.
    # Deskew only when the estimate is materially non-zero and still inside the
    # detector's conservative skew domain. Expand preserves all source pixels.
    if 0.75 <= abs(rotation) <= 6.0:
        image = image.rotate(
            -rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )
        applied_rotation = -rotation

    # Low-resolution/JPEG ECGs can arrive smaller than the model's 2000 px
    # segmentation scale. Avoid a second smoothing resize: nearest-neighbour
    # enlargement preserves thin trace/grid edges; the digitizer performs its
    # own model-scale normalization afterwards.
    min_model_dim = int(HIGH_FIDELITY_RESAMPLE_SIZE)
    scale = max(1.0, float(min_model_dim) / max(image.size))
    upscale_method = "NONE"
    if scale > 1.0:
        image = image.resize(
            (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            ),
            Image.Resampling.NEAREST,
        )
        upscale_method = "NEAREST_EDGE_PRESERVING"

    image.save(image_path, format="PNG", optimize=True)
    return {
        "input_size": before,
        "output_size": [int(image.width), int(image.height)],
        "preflight_rotation_deg": rotation,
        "applied_rotation_deg": applied_rotation,
        "upscale_factor": round(float(scale), 6),
        "upscale_method": upscale_method,
        "signal_qc_thresholds_changed": False,
    }


def _load_digitizer(
    vendor_root: Path,
    segmentation_model: Path,
    lead_model: Path,
    *,
    resample_size: int = LOW_MEMORY_RESAMPLE_SIZE,
):
    import torch

    deterministic = str(
        os.environ.get("MEDCALC_ECG_DETERMINISTIC_INFERENCE", "1")
    ).strip().lower() not in {"0", "false", "no", "off"}
    deterministic_strict = str(
        os.environ.get("MEDCALC_ECG_DETERMINISTIC_STRICT", "0")
    ).strip().lower() in {"1", "true", "yes", "on"}
    deterministic_seed = int(os.environ.get("MEDCALC_ECG_INFERENCE_SEED", "1729"))

    if deterministic:
        # The delineation layer is sensitive to one-sample changes at waveform
        # tails, so the digitizer itself must be reproducible for the same ECG.
        # This is not a clinical threshold change; it constrains numerical
        # execution so repeated inference cannot change interval measurements.
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        np.random.seed(deterministic_seed)
        torch.manual_seed(deterministic_seed)
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=not deterministic_strict,
            )
        except TypeError:
            torch.use_deterministic_algorithms(True)
        try:
            import cv2
            cv2.setNumThreads(1)
            cv2.setRNGSeed(int(deterministic_seed))
        except Exception:
            pass

    sys.path.insert(0, str(vendor_root))

    from src.config.default import get_cfg
    from src.model.inference_wrapper import InferenceWrapper

    config_path = vendor_root / "src" / "config" / "inference_wrapper_george-moody-2024.yml"
    layout_path = vendor_root / "src" / "config" / "lead_layouts_all.yml"
    lead_unet_config = vendor_root / "src" / "config" / "lead_name_unet.yml"

    cfg = get_cfg(str(config_path))

    # CPU-only inference for Streamlit Community Cloud.
    cfg.MODEL.KWARGS.device = "cpu"
    # Resolution is route-specific. The upstream model documents 3000 px and
    # 2000 px as its reduced-memory setting. MEDCALC uses a higher-resolution
    # segmentation-only path only when geometry is strongly corroborated; the
    # neural-layout fallback remains low-memory.
    cfg.MODEL.KWARGS.resample_size = int(resample_size)
    # Grid-based dewarping is enabled on the remote high-fidelity route.
    # The 1200 px reference/fallback path stays compact and deterministic.
    cfg.MODEL.KWARGS.apply_dewarping = bool(
        int(os.environ.get("MEDCALC_ECG_ENABLE_DEWARP", "1"))
        and int(resample_size) >= HIGH_FIDELITY_RESAMPLE_SIZE
    )
    cfg.MODEL.KWARGS.enable_timing = False

    inner = cfg.MODEL.KWARGS.config
    inner.SEGMENTATION_MODEL.weight_path = str(segmentation_model)
    inner.LAYOUT_IDENTIFIER.config_path = str(layout_path)
    inner.LAYOUT_IDENTIFIER.unet_config_path = str(lead_unet_config)
    inner.LAYOUT_IDENTIFIER.unet_weight_path = str(lead_model)
    inner.LAYOUT_IDENTIFIER.KWARGS.device = "cpu"
    # Standard MEDCALC uploads are displayed in normal orientation after EXIF/PDF
    # normalization. Avoid the extra flipped-layout branch in low-memory mode.
    inner.LAYOUT_IDENTIFIER.KWARGS.possibly_flipped = False

    # R27's frozen signal contract is 10 s at 500 Hz.
    # This sets the output grid length only; unobserved printed portions remain
    # NaN in the canonical lead tensor and are checked before R27 can run.
    inner.LAYOUT_IDENTIFIER.KWARGS.target_num_samples = 5000
    inner.LAYOUT_IDENTIFIER.KWARGS.required_valid_samples = 2

    torch_threads = int(os.environ.get("MEDCALC_ECG_TORCH_THREADS", "4"))
    torch_threads = max(1, min(4, torch_threads))
    if deterministic:
        torch_threads = 1
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model = InferenceWrapper(**cfg.MODEL.KWARGS)
    model.eval()
    return model


def _digitize_image(
    image_path: Path,
    model,
    *,
    layout_hint: str | None = None,
) -> tuple[np.ndarray, dict]:
    import torch
    from torchvision.io import decode_image

    image = decode_image(str(image_path), mode="RGB")[:3].unsqueeze(0)

    with torch.no_grad():
        result = model(
            image,
            layout_should_include_substring=layout_hint,
            skip_identifier=False,
        )

    canonical = result.get("signal", {}).get("canonical_lines")
    if canonical is None:
        raise RuntimeError("El U-Net no produjo canonical_lines.")

    signal_info = result.get("signal", {}) or {}
    layout_name = str(result.get("layout_name") or "")
    layout_source = "lead_name_unet"
    original_layout_name = layout_name

    identifier_rows = signal_info.get("identifier_rows_in_layout")
    identifier_n_detected = signal_info.get("identifier_n_detected")
    identifier_defaulted = bool(signal_info.get("identifier_defaulted_layout", False))
    extractor_num_peaks = signal_info.get("signal_extractor_num_peaks")
    fallback_rows = None
    fallback_layout = None

    # The lead-name U-Net may fail at the low-memory 1200 px inference size
    # even though the signal extractor correctly recovered the page geometry.
    # Use the identifier's own row count first, then the extractor's peak count,
    # and only then recompute from the compact raw-line tensor.
    low_confidence_layout = bool(
        layout_name == "Unknown layout"
        or identifier_defaulted
        or identifier_n_detected is None
        or int(identifier_n_detected) <= 2
    )

    raw_lines = signal_info.get("raw_lines")
    avg_ppmm = (result.get("pixel_spacing_mm") or {}).get("average_pixel_per_mm")

    row_candidates = []
    for candidate in (identifier_rows, extractor_num_peaks):
        try:
            if candidate is not None:
                row_candidates.append(int(candidate))
        except Exception:
            pass

    merged = None
    if raw_lines is not None and model.identifier is not None:
        try:
            merged = model.identifier._merge_nonoverlapping_lines(raw_lines)
            row_candidates.append(int(merged.shape[0]))
        except Exception:
            merged = None

    if low_confidence_layout:
        # Prefer the richest plausible physical-row geometry when sources
        # disagree. In this Bionet/EKG2000 family, a seven-row signal cannot be
        # represented by 3x4+1R without discarding V1-V6. Earlier code selected
        # the first candidate and could therefore collapse [4, 7, 7] to 4.
        standard_counts = [c for c in row_candidates if c in (4, 6, 7)]
        if 7 in standard_counts:
            fallback_rows = 7
        elif 6 in standard_counts:
            fallback_rows = 6
        elif 4 in standard_counts:
            fallback_rows = 4

        if fallback_rows == 7:
            fallback_layout = "6x2+1R"
        elif fallback_rows == 6:
            fallback_layout = "6x2"
        elif fallback_rows == 4:
            fallback_layout = "3x4+1R"

        if (
            fallback_layout is not None
            and raw_lines is not None
            and avg_ppmm is not None
            and model.identifier is not None
        ):
            if merged is None:
                merged = model.identifier._merge_nonoverlapping_lines(raw_lines)

            # Use identifier-normalized rows only when their row count matches
            # the selected geometry and they contain finite signal. Otherwise
            # normalize the recomputed merged raw rows.
            normalized = None
            identifier_lines = signal_info.get("identifier_lines")
            if identifier_lines is not None:
                try:
                    same_rows = int(identifier_lines.shape[0]) == int(fallback_rows)
                    finite_n = int((~identifier_lines.isnan()).sum().item())
                    if same_rows and finite_n > 0:
                        normalized = identifier_lines
                except Exception:
                    normalized = None

            if normalized is None:
                if merged is None:
                    merged = model.identifier._merge_nonoverlapping_lines(raw_lines)
                if int(merged.shape[0]) != int(fallback_rows):
                    # Do not force a layout onto a line tensor with a different
                    # physical-row count; keep the neural result fail-closed.
                    fallback_layout = None
                else:
                    normalized = -model.identifier.normalize(
                        merged,
                        float(avg_ppmm),
                        0.1,
                    )

            if fallback_layout is not None and normalized is not None:
                canonical_candidate = model.identifier._canonicalize_lines(
                    normalized,
                    {"layout": fallback_layout, "flip": False},
                )
                finite_by_lead = (~canonical_candidate.isnan()).float().mean(dim=1)
                recovered_leads = int((finite_by_lead > 0.10).sum().item())

                # A standard 12-lead page must recover most leads. Reject a
                # fallback that merely has the right label but produces an
                # empty/degenerate canonical tensor.
                if recovered_leads >= 10:
                    canonical = canonical_candidate
                    layout_name = fallback_layout
                    layout_source = (
                        f"geometric_fallback_rows_{fallback_rows}"
                        f"_detected_{identifier_n_detected}"
                    )
                else:
                    fallback_layout = None

    signal_uv = canonical.detach().cpu().numpy().astype(np.float64)
    if signal_uv.shape != (12, 5000):
        raise RuntimeError(
            f"Forma canónica inesperada: {signal_uv.shape}; se esperaba (12, 5000)."
        )

    signal_uv = signal_uv.T  # samples x leads

    finite = np.isfinite(signal_uv)
    coverage = finite.mean(axis=0)

    layout_cost = result.get("signal", {}).get("layout_matching_cost")
    try:
        layout_cost = float(layout_cost)
    except Exception:
        layout_cost = None

    pixel = result.get("pixel_spacing_mm") or {}

    meta = {
        "shape_500_candidate": [int(v) for v in signal_uv.shape],
        "sig_names": LEADS,
        "observed_fraction_by_lead": {
            lead: round(float(coverage[i]), 6)
            for i, lead in enumerate(LEADS)
        },
        "observed_seconds_by_lead": {
            lead: round(float(coverage[i]) * 10.0, 6)
            for i, lead in enumerate(LEADS)
        },
        "native_signal_contract": "OBSERVED_ONLY_NAN_MASKED_500HZ_12LEAD",
        "observed_mask_preserved": True,
        "min_observed_fraction": round(float(np.min(coverage)), 6),
        "all_samples_observed": bool(np.all(finite)),
        "layout_name": layout_name,
        "layout_source": layout_source,
        "layout_name_original": original_layout_name,
        "geometric_fallback_rows": fallback_rows,
        "identifier_rows_in_layout": identifier_rows,
        "identifier_n_detected": identifier_n_detected,
        "identifier_defaulted_layout": identifier_defaulted,
        "signal_extractor_num_peaks": extractor_num_peaks,
        "row_candidates": row_candidates,
        "layout_matching_cost": layout_cost,
        "pixel_spacing_mm": {
            "x": float(pixel["x"]) if pixel.get("x") is not None else None,
            "y": float(pixel["y"]) if pixel.get("y") is not None else None,
        },
        "units_from_digitizer": "uV",
        "target_samples": 5000,
    }
    return signal_uv, meta


def _digitize_temporal_strip_only(
    image_path: Path,
    model,
    *,
    layout_hint: str,
) -> tuple[np.ndarray, dict]:
    """Extract only the observed long rhythm strip at 1200 px.

    Full 12-lead layout acceptance is intentionally not required here.  The
    primary 2000 px route already established the layout; this route answers a
    narrower question: is there a sufficiently observed native long strip for
    RR timing?  The returned tensor contains only lead II and cannot be used for
    morphology, axis or R27.
    """
    import torch
    from torchvision.io import decode_image

    image = decode_image(str(image_path), mode="RGB")[:3].unsqueeze(0)
    with torch.no_grad():
        result = model(
            image,
            layout_should_include_substring=None,
            skip_identifier=True,
        )

    signal_info = result.get("signal", {}) or {}
    raw_lines = signal_info.get("raw_lines")
    aligned_signal_prob = signal_info.get("aligned_signal_prob")
    pixel = result.get("pixel_spacing_mm") or {}
    avg_ppmm = pixel.get("average_pixel_per_mm")

    if raw_lines is None:
        raise RuntimeError("TEMPORAL_STRIP_ONLY: U-Net sin raw_lines.")
    if aligned_signal_prob is None:
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY: U-Net sin aligned_signal_prob."
        )
    if avg_ppmm is None:
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY: escala física no disponible."
        )

    if hasattr(aligned_signal_prob, "detach"):
        prob = aligned_signal_prob.detach().cpu().numpy().astype(np.float32)
    else:
        prob = np.asarray(aligned_signal_prob, dtype=np.float32)

    routed = route_temporal_rhythm_reference(
        prob,
        raw_lines,
        avg_pixel_per_mm=float(avg_ppmm),
        layout_hint=layout_hint,
        threshold=0.12,
        min_rhythm_coverage=0.55,
    )
    signal_leads_samples = np.asarray(
        routed.pop("signal_uv"),
        dtype=np.float64,
    )
    if signal_leads_samples.shape != (12, 5000):
        raise RuntimeError(
            "TEMPORAL_STRIP_ONLY: forma esperada (12, 5000), "
            f"recibida {signal_leads_samples.shape}."
        )

    signal_uv = signal_leads_samples.T
    observed = np.isfinite(signal_uv).mean(axis=0)
    meta = {
        "shape_500_candidate": [5000, 12],
        "sig_names": LEADS,
        "layout_name": f"TEMPORAL_STRIP_ONLY_{str(layout_hint).split('+',1)[0]}",
        "layout_source": "LOW_MEMORY_1200_TEMPORAL_STRIP_ONLY",
        "observed_fraction_by_lead": {
            lead: round(float(observed[i]), 6)
            for i, lead in enumerate(LEADS)
        },
        "observed_seconds_by_lead": {
            lead: round(float(observed[i]) * 10.0, 6)
            for i, lead in enumerate(LEADS)
        },
        "min_observed_fraction": 0.0,
        "all_samples_observed": False,
        "native_signal_contract": "OBSERVED_ONLY_NAN_MASKED_500HZ_RHYTHM_ONLY",
        "observed_mask_preserved": True,
        "rhythm_strip_detected": True,
        "rhythm_strip_observed": True,
        "rhythm_strip_lead": "II",
        "rhythm_strip_coverage": routed.get("rhythm_strip_coverage"),
        "rhythm_strip_longest_contiguous_fraction": routed.get(
            "rhythm_strip_longest_contiguous_fraction"
        ),
        "rhythm_strip_quality": "USABLE_LONG_STRIP",
        "rhythm_strip_center_source": "POST_UNET_TEMPORAL_STRIP_ONLY",
        "temporal_strip_router": routed,
        "units_from_digitizer": "uV",
        "target_samples": 5000,
    }
    return signal_uv, meta


class LayoutHypothesisRoutingError(RuntimeError):
    """Raised when post-U-Net evidence cannot select a standard layout."""


def _digitize_layout_hypotheses(
    image_path: Path,
    model,
    *,
    layout_preflight: dict | None = None,
    speed_mm_per_s: float | None = 25.0,
    gain_mm_per_mv: float | None = 10.0,
) -> tuple[np.ndarray, dict]:
    """Digitize from U-Net signal support with geometry-first layout guidance.

    A strong preflight 3x4/6x2 geometry supplies expected physical row
    positions, which are locally re-centered on the U-Net probability map.
    If that signal-corroborated geometry route is unavailable, the original
    competing post-U-Net hypotheses remain the fail-closed fallback.
    """
    import torch
    from torchvision.io import decode_image

    image = decode_image(str(image_path), mode="RGB")[:3].unsqueeze(0)
    with torch.no_grad():
        result = model(
            image,
            layout_should_include_substring=None,
            skip_identifier=True,
        )

    signal_info = result.get("signal", {}) or {}
    raw_lines = signal_info.get("raw_lines")
    aligned_signal_prob = signal_info.get("aligned_signal_prob")
    pixel = result.get("pixel_spacing_mm") or {}
    dewarping_info = result.get("dewarping") or {}
    aligned_active_x = signal_info.get("aligned_active_x")
    avg_ppmm = pixel.get("average_pixel_per_mm")

    if raw_lines is None:
        raise LayoutHypothesisRoutingError(
            "La segmentación no produjo raw_lines."
        )
    if aligned_signal_prob is None:
        raise LayoutHypothesisRoutingError(
            "La segmentación no produjo aligned_signal_prob."
        )
    if avg_ppmm is None:
        raise LayoutHypothesisRoutingError(
            "No se pudo estimar la escala física del ECG."
        )

    if hasattr(aligned_signal_prob, "detach"):
        signal_prob_np = (
            aligned_signal_prob.detach().cpu().numpy().astype(np.float32)
        )
    else:
        signal_prob_np = np.asarray(aligned_signal_prob, dtype=np.float32)

    selected = None
    public_router: dict = {}
    guided_error = None

    preflight = dict(layout_preflight or {})
    guided_layout = _trusted_preflight_layout(preflight)
    if guided_layout is not None:
        try:
            guided_geometry = recover_signal_geometry_from_preflight(
                signal_prob_np,
                preflight,
                threshold=0.08,
                aligned_active_x=aligned_active_x,
            )
            guided_rows, guided_sources, guided_debug = (
                build_rows_from_signal_probability(
                    signal_prob_np,
                    raw_lines,
                    guided_geometry,
                )
            )
            primary_quality = [
                row for row in (guided_debug.get("source_quality") or [])
                if not bool(row.get("is_rhythm_row"))
            ]
            coverages = [
                float(row.get("selected_active_coverage") or 0.0)
                for row in primary_quality
            ]
            median_coverage = (
                float(np.median(coverages)) if coverages else 0.0
            )
            min_support = min(
                [
                    float(v)
                    for v in (
                        guided_geometry.get("primary_row_support") or []
                    )
                ]
                or [0.0]
            )
            expected_rows = 6 if guided_layout == "6x2" else 3
            primary_row_n = len(
                guided_geometry.get("primary_centers_y") or []
            )
            guided_accept = bool(
                primary_row_n == expected_rows
                and len(primary_quality) >= expected_rows
                and median_coverage >= 0.25
                and min_support >= 0.10
            )
            guided_score = float(
                np.clip(
                    0.55 * float(preflight.get("confidence") or 0.0)
                    + 0.45 * median_coverage,
                    0.0,
                    1.0,
                )
            )
            public_router = {
                "router_version": "MEDCALC_GEOMETRY_FIRST_SIGNAL_GUIDED_V1",
                "decision": (
                    "SELECTED_GEOMETRY_GUIDED"
                    if guided_accept
                    else "GEOMETRY_GUIDED_SIGNAL_QC_FAILED"
                ),
                "selected_layout": guided_layout if guided_accept else None,
                "selected_score": guided_score if guided_accept else None,
                "preflight_confidence": float(
                    preflight.get("confidence") or 0.0
                ),
                "median_row_coverage": round(median_coverage, 6),
                "min_primary_row_unet_support": round(min_support, 6),
                "expected_primary_rows": expected_rows,
                "recovered_primary_rows": primary_row_n,
                "row_debug": guided_debug,
            }
            if guided_accept:
                selected = {
                    "layout": guided_layout,
                    "score": guided_score,
                    "geometry": guided_geometry,
                    "row_sources": guided_sources,
                    "row_debug": guided_debug,
                    "physical_rows_y_px": guided_rows,
                    "canonical_meta": {},
                }
        except Exception as exc:
            guided_error = f"{type(exc).__name__}:{exc}"
            public_router = {
                "router_version": "MEDCALC_GEOMETRY_FIRST_SIGNAL_GUIDED_V1",
                "decision": "GEOMETRY_GUIDED_ROUTE_FAILED",
                "selected_layout": None,
                "guided_error": guided_error,
            }

    if selected is None:
        routed = route_layout_hypotheses(
            signal_prob_np,
            raw_lines,
            avg_pixel_per_mm=float(avg_ppmm),
            threshold=0.12,
            min_winner_score=0.66,
            min_margin=0.07,
        )
        selected = routed.get("_selected_candidate")
        fallback_public = {
            k: v for k, v in routed.items()
            if k != "_selected_candidate"
        }
        public_router = {
            "geometry_guided": public_router,
            "fallback_hypothesis_router": fallback_public,
            "router_version": "MEDCALC_GEOMETRY_FIRST_WITH_HYPOTHESIS_FALLBACK_V1",
            "decision": (
                "SELECTED_POST_UNET_FALLBACK"
                if selected is not None
                else "NO_LAYOUT_ROUTE_ACCEPTED"
            ),
            "selected_layout": (
                selected.get("layout") if selected is not None else None
            ),
            "selected_score": (
                float(selected.get("score")) if selected is not None else None
            ),
        }

    if selected is None:
        raise LayoutHypothesisRoutingError(
            "LAYOUT_HYPOTHESES_UNRESOLVED: "
            + json.dumps(public_router, ensure_ascii=False, sort_keys=True)
        )

    layout = str(selected["layout"])
    geometry = selected.get("geometry") or {}
    if aligned_active_x is not None:
        if (len(aligned_active_x) != 2
                or not 0 <= aligned_active_x[0] < aligned_active_x[1] < signal_prob_np.shape[1]):
            raise LayoutHypothesisRoutingError("INVALID_ALIGNED_ACTIVE_X")
        geometry["active_x"] = list(aligned_active_x)
        geometry["active_x_debug"] = {"source": "EXTRACTOR_ALIGNED_CANVAS_PIXELS"}
    rhythm_detected = geometry.get("rhythm_center_y") is not None
    physical_rows = np.asarray(
        selected.get("physical_rows_y_px"),
        dtype=np.float64,
    )
    if physical_rows.ndim != 2:
        raise LayoutHypothesisRoutingError(
            "La ruta U-Net no expuso centerlines físicas 2-D."
        )

    audit_overlay_b64 = _centerline_audit_overlay_png_base64(
        signal_prob_np,
        physical_rows,
    )

    canonical_ecg = reconstruct_canonical_ecg(
        physical_rows,
        layout=layout,
        rhythm_strip=bool(rhythm_detected),
        active_x=geometry.get("active_x"),
        pixel_spacing_mm={
            "x": pixel.get("x"),
            "y": pixel.get("y"),
        },
        speed_mm_per_s=speed_mm_per_s,
        gain_mm_per_mv=gain_mm_per_mv,
        fs=500,
        layout_confidence=float(selected.get("score") or 0.0),
        row_sources=list(selected.get("row_sources") or []),
        speed_source="MEDCALC_FIXED_ACQUISITION_PROTOCOL_25_MM_S",
        gain_source="MEDCALC_FIXED_ACQUISITION_PROTOCOL_10_MM_MV",
    )
    signal_mv = np.asarray(
        canonical_ecg["legacy_matrix_mv"],
        dtype=np.float64,
    )
    if signal_mv.shape != (5000, 12):
        raise LayoutHypothesisRoutingError(
            f"Forma digital canónica inesperada: {signal_mv.shape}."
        )
    signal_uv = signal_mv * 1000.0
    quality_matrix = np.asarray(
        canonical_ecg["legacy_quality_mask"],
        dtype=np.uint8,
    )
    finite = np.isfinite(signal_uv)

    # Clinical coverage is relative to the duration expected from the selected
    # physical layout, not the 10 s legacy/R27 matrix.
    canonical_coverage = dict(canonical_ecg.get("coverage_by_lead") or {})
    coverage = np.asarray(
        [float(canonical_coverage.get(lead) or 0.0) for lead in LEADS],
        dtype=float,
    )
    lead_ii = (canonical_ecg.get("leads") or {}).get("II") or {}
    lead_ii_coverage = float(lead_ii.get("observed_fraction") or 0.0)
    rhythm_observed = bool(
        float(lead_ii.get("duration_s") or 0.0) >= 5.0
        and lead_ii_coverage >= 0.45
    )

    measurement_error = None
    signal_primary_report = None
    digital_measurements = None
    try:
        digital_measurements = analyze_canonical_ecg(canonical_ecg)
        signal_primary_report = build_signal_primary_structured_report(
            canonical_ecg,
            digital_measurements,
        )
    except Exception as exc:
        measurement_error = str(exc)

    dev_row_counterfactual = None
    if str(os.environ.get("MEDCALC_ECG_DEV_ROW_COUNTERFACTUAL", "0")).strip().lower() in {
        "1", "true", "yes", "on"
    }:
        try:
            cf_rows, cf_sources, cf_row_debug = build_weighted_rows_counterfactual(
                signal_prob_np,
                geometry,
            )
            cf_canonical = reconstruct_canonical_ecg(
                cf_rows,
                layout=layout,
                rhythm_strip=bool(rhythm_detected),
                active_x=geometry.get("active_x"),
                pixel_spacing_mm={
                    "x": pixel.get("x"),
                    "y": pixel.get("y"),
                },
                speed_mm_per_s=speed_mm_per_s,
                gain_mm_per_mv=gain_mm_per_mv,
                fs=500,
                layout_confidence=float(selected.get("score") or 0.0),
                row_sources=cf_sources,
                speed_source="MEDCALC_FIXED_ACQUISITION_PROTOCOL_25_MM_S",
                gain_source="MEDCALC_FIXED_ACQUISITION_PROTOCOL_10_MM_MV",
            )
            cf_analysis = analyze_canonical_ecg(cf_canonical)

            def _cf_metric(name: str) -> float | None:
                item = (cf_analysis.get("global") or {}).get(name) or {}
                value = item.get("value")
                return float(value) if value is not None else None

            def _primary_metric(name: str) -> float | None:
                item = ((digital_measurements or {}).get("global") or {}).get(name) or {}
                value = item.get("value")
                return float(value) if value is not None else None

            dev_row_counterfactual = {
                "status": "OK",
                "route": "ALL_WEIGHTED_BAND_FROM_ALIGNED_UNET_PROBABILITY",
                "primary_metrics": {
                    name: _primary_metric(name)
                    for name in ("qrs_ms", "pr_ms", "qt_ms", "heart_rate_bpm")
                },
                "counterfactual_metrics": {
                    name: _cf_metric(name)
                    for name in ("qrs_ms", "pr_ms", "qt_ms", "heart_rate_bpm")
                },
                "row_debug": cf_row_debug,
            }
        except Exception as exc:
            dev_row_counterfactual = {
                "status": "FAIL",
                "reason": str(exc),
            }

    canonical_meta = selected.get("canonical_meta") or {}
    meta = {
        "shape_500_candidate": [5000, 12],
        "sig_names": LEADS,
        "observed_fraction_by_lead": {
            lead: round(float(coverage[i]), 6)
            for i, lead in enumerate(LEADS)
        },
        "coverage_definition": "LONGEST_CONTIGUOUS_SUPPORTED_SECONDS_DIVIDED_BY_LAYOUT_EXPECTED_SECONDS",
        "expected_duration_by_lead_s": dict(
            canonical_ecg.get("expected_duration_by_lead_s") or {}
        ),
        "observed_seconds_by_lead": dict(
            canonical_ecg.get("observed_seconds_by_lead") or {}
        ),
        "total_supported_seconds_by_lead": dict(
            canonical_ecg.get("total_supported_seconds_by_lead") or {}
        ),
        "legacy_10s_coverage_by_lead": dict(
            canonical_ecg.get("legacy_10s_coverage_by_lead") or {}
        ),
        "native_signal_contract": "CALIBRATED_DIGITAL_SIGNAL_V2_500HZ_12LEAD_NAN_MASKED",
        "observed_mask_preserved": True,
        "min_observed_fraction": round(float(np.min(coverage)), 6),
        "all_samples_observed": bool(np.all(finite)),
        "layout_name": (
            f"{layout}+1R" if rhythm_detected else layout
        ),
        "layout_source": "POST_UNET_LAYOUT_HYPOTHESIS_ROUTER_V2",
        "layout_name_original": str(result.get("layout_name") or ""),
        "layout_matching_cost": None,
        "layout_hypothesis_router": public_router,
        "canonicalizer": canonical_meta.get("canonicalizer"),
        "canonicalizer_meta": canonical_meta,
        "signal_geometry": geometry,
        "rhythm_strip_detected": bool(rhythm_detected),
        "rhythm_strip_observed": bool(rhythm_observed),
        "rhythm_strip_lead": "II" if rhythm_observed else None,
        "rhythm_strip_coverage": round(lead_ii_coverage, 6),
        "rhythm_strip_quality": (
            "USABLE_LONG_STRIP"
            if rhythm_observed
            else "DETECTED_BUT_INSUFFICIENT_COVERAGE"
            if rhythm_detected
            else "NOT_DETECTED"
        ),
        "rhythm_strip_center_source": (
            "POST_UNET_SIGNAL_HYPOTHESIS"
            if rhythm_detected else "NOT_DETECTED"
        ),
        "row_sources": selected.get("row_sources") or [],
        "row_assignment_debug": selected.get("row_debug") or {},
        "signal_extractor_num_peaks": signal_info.get(
            "signal_extractor_num_peaks"
        ),
        "pixel_spacing_mm": {
            "x": float(pixel["x"]) if pixel.get("x") is not None else None,
            "y": float(pixel["y"]) if pixel.get("y") is not None else None,
        },
        "units_from_digitizer": "uV",
        "target_samples": 5000,
        "calibrated_digital_signal": canonical_to_worker_payload(canonical_ecg),
        "calibration": canonical_ecg.get("calibration"),
        "geometric_correction": {
            "perspective": "APPLIED_BY_OPEN_ECG_PIPELINE",
            "dewarping": dewarping_info,
        },
        "clinical_measurement_source": "CALIBRATED_DIGITAL_SIGNAL_V2",
        "signal_primary_structured_report": signal_primary_report,
        "digital_measurements_v2": digital_measurements,
        "signal_primary_measurement_error": measurement_error,
        "development_row_counterfactual": dev_row_counterfactual,
        "audit_centerline_overlay_png_base64": audit_overlay_b64,
        "audit_overlay_coordinate_system": (
            "POST_PERSPECTIVE_POST_DEWARP_U_NET_PROBABILITY_MAP"
        ),
    }
    return signal_uv, meta


def _validate_forced_6x2_corroboration(
    signal_geometry: dict,
    row_debug: dict,
) -> dict:
    """Require independent signal-extractor support before forcing 6x2.

    The cheap preflight geometry detector is never sufficient on its own. A
    forced 6x2 route is accepted only when the segmentation probability map has
    six coherent row centers and Open-ECG's own extracted centerlines can be
    assigned to nearly all of them. Otherwise the worker falls back to the
    neural layout identifier.
    """
    centers = np.asarray(
        signal_geometry.get("primary_centers_y") or [],
        dtype=float,
    )
    failures: list[str] = []

    if centers.size != 6:
        failures.append(f"primary_centers={int(centers.size)}")
        spacing_cv = None
    else:
        spacing = np.diff(np.sort(centers))
        spacing_mean = float(np.mean(spacing)) if spacing.size else 0.0
        spacing_cv = (
            float(np.std(spacing) / spacing_mean)
            if spacing_mean > 0
            else None
        )
        if spacing_cv is None or spacing_cv > 0.35:
            failures.append(
                "row_spacing_cv="
                + ("NA" if spacing_cv is None else f"{spacing_cv:.3f}")
            )

    assignment = row_debug.get("assignment") or {}
    try:
        assigned_count = int(assignment.get("assigned_count") or 0)
    except Exception:
        assigned_count = 0
    if assigned_count < 5:
        failures.append(f"open_ecg_assigned_rows={assigned_count}")

    qualities = [
        q for q in (row_debug.get("source_quality") or [])
        if not bool(q.get("is_rhythm_row"))
    ]
    selected_coverages = [
        float(q.get("selected_active_coverage") or 0.0)
        for q in qualities
    ]
    median_selected_coverage = (
        float(np.median(selected_coverages))
        if selected_coverages
        else 0.0
    )
    if median_selected_coverage < 0.35:
        failures.append(
            f"median_selected_coverage={median_selected_coverage:.3f}"
        )

    result = {
        "accepted": not failures,
        "expected_layout": "6x2",
        "primary_center_count": int(centers.size),
        "row_spacing_cv": (
            round(float(spacing_cv), 6)
            if spacing_cv is not None
            else None
        ),
        "open_ecg_assigned_rows": int(assigned_count),
        "median_selected_active_coverage": round(
            median_selected_coverage,
            6,
        ),
        "failures": failures,
    }
    if failures:
        raise ForcedLayoutCorroborationError(
            "6x2 preflight no corroborado por la señal: "
            + "; ".join(failures)
        )
    return result


def _digitize_forced_layout(
    image_path: Path,
    model,
    *,
    layout_preflight: dict,
    allow_unconfirmed_probe: bool = False,
) -> tuple[np.ndarray, dict]:
    """High-fidelity 6x2 route with independent post-U-Net corroboration.

    The preflight geometry detector proposes 6x2, but it is not trusted alone.
    The segmentation probability map and Open-ECG centerlines must independently
    support the six physical rows before MEDCALC skips the lead-name U-Net.
    Unprinted intervals remain NaN.
    """
    import torch
    from torchvision.io import decode_image

    layout = str(layout_preflight.get("layout") or "")
    confidence = float(layout_preflight.get("confidence") or 0.0)
    if layout != "6x2":
        raise RuntimeError("La ruta forzada sólo acepta hipótesis 6x2.")
    if confidence < 0.85 and not bool(allow_unconfirmed_probe):
        raise RuntimeError(
            "La ruta forzada exige preflight 6x2 >=0.85 salvo una sonda "
            "explícita que después debe ser corroborada por la señal U-Net."
        )

    image = decode_image(str(image_path), mode="RGB")[:3].unsqueeze(0)

    with torch.no_grad():
        result = model(
            image,
            layout_should_include_substring=None,
            skip_identifier=True,
        )

    signal_info = result.get("signal", {}) or {}
    raw_lines = signal_info.get("raw_lines")
    aligned_signal_prob = signal_info.get("aligned_signal_prob")
    pixel = result.get("pixel_spacing_mm") or {}
    avg_ppmm = pixel.get("average_pixel_per_mm")

    if raw_lines is None:
        raise RuntimeError("La U-Net no produjo raw_lines para la ruta 6x2.")
    if aligned_signal_prob is None:
        raise RuntimeError(
            "La U-Net no produjo aligned_signal_prob para la ruta 6x2."
        )
    if avg_ppmm is None:
        raise RuntimeError("No se pudo recuperar la escala física del ECG.")

    if hasattr(aligned_signal_prob, "detach"):
        signal_prob_np = (
            aligned_signal_prob.detach().cpu().numpy().astype(np.float32)
        )
    else:
        signal_prob_np = np.asarray(aligned_signal_prob, dtype=np.float32)

    signal_geometry = detect_rows_from_signal_probability(
        signal_prob_np,
        layout="6x2",
        rhythm_strip_hint=bool(layout_preflight.get("rhythm_strip")),
        threshold=0.12,
    )

    rhythm_recovery_source = (
        "SIGNAL_PROBABILITY"
        if signal_geometry.get("rhythm_center_y") is not None
        else "NOT_DETECTED"
    )
    if (
        bool(layout_preflight.get("rhythm_strip"))
        and signal_geometry.get("rhythm_center_y") is None
    ):
        recovered_center, rhythm_recovery_source = recover_rhythm_center_from_preflight(
            signal_prob_np,
            signal_geometry,
            layout_preflight,
            threshold=0.08,
        )
        if recovered_center is not None:
            signal_geometry["rhythm_center_y"] = float(recovered_center)
            signal_geometry["rhythm_center_recovery"] = rhythm_recovery_source

    row_lines, row_sources, row_debug = build_rows_from_signal_probability(
        signal_prob_np,
        raw_lines,
        signal_geometry,
    )

    forced_validation = _validate_forced_6x2_corroboration(
        signal_geometry,
        row_debug,
    )

    rhythm_detected = signal_geometry.get("rhythm_center_y") is not None

    canonical_uv, canonical_meta = canonicalize_extracted_rows(
        row_lines,
        avg_pixel_per_mm=float(avg_ppmm),
        layout="6x2",
        rhythm_strip=bool(rhythm_detected),
        target_num_samples=5000,
        required_valid_samples=2,
        active_x=signal_geometry.get("active_x"),
    )

    if canonical_uv.shape != (12, 5000):
        raise RuntimeError(
            f"Forma canónica 6x2 inesperada: {canonical_uv.shape}."
        )

    signal_uv = canonical_uv.T
    finite = np.isfinite(signal_uv)
    coverage = finite.mean(axis=0)

    # A detected row center is not enough to claim a usable long rhythm strip.
    # The old implementation labelled +1R as "observed" even when only ~19% of
    # lead II had actually been recovered. Require substantial native coverage.
    lead_ii_coverage = float(coverage[LEADS.index("II")])
    rhythm_observed = bool(rhythm_detected and lead_ii_coverage >= 0.70)
    layout_name = "6x2+1R" if rhythm_detected else "6x2"

    meta = {
        "shape_500_candidate": [5000, 12],
        "sig_names": LEADS,
        "observed_fraction_by_lead": {
            lead: round(float(coverage[i]), 6)
            for i, lead in enumerate(LEADS)
        },
        "observed_seconds_by_lead": {
            lead: round(float(coverage[i]) * 10.0, 6)
            for i, lead in enumerate(LEADS)
        },
        "native_signal_contract": "OBSERVED_ONLY_NAN_MASKED_500HZ_12LEAD",
        "observed_mask_preserved": True,
        "min_observed_fraction": round(float(np.min(coverage)), 6),
        "all_samples_observed": bool(np.all(finite)),
        "layout_name": layout_name,
        "layout_source": (
            "POST_UNET_SIGNAL_CORROBORATED_6X2_PROBE"
            if bool(allow_unconfirmed_probe)
            else "PRE_UNET_GEOMETRY_ROUTER"
        ),
        "layout_name_original": str(result.get("layout_name") or ""),
        "layout_matching_cost": None,
        "canonicalizer": canonical_meta.get("canonicalizer"),
        "canonicalizer_meta": canonical_meta,
        "signal_geometry": signal_geometry,
        "rhythm_strip_detected": bool(rhythm_detected),
        "rhythm_strip_observed": bool(rhythm_observed),
        "rhythm_strip_lead": "II" if rhythm_observed else None,
        "rhythm_strip_coverage": round(lead_ii_coverage, 6),
        "rhythm_strip_quality": (
            "USABLE_LONG_STRIP"
            if rhythm_observed
            else "DETECTED_BUT_INSUFFICIENT_COVERAGE"
            if rhythm_detected
            else "NOT_DETECTED"
        ),
        "rhythm_strip_center_source": rhythm_recovery_source,
        "row_sources": row_sources,
        "row_assignment_debug": row_debug,
        "forced_layout_validation": forced_validation,
        "preflight_layout": {
            "layout": layout_preflight.get("layout"),
            "confidence": float(layout_preflight.get("confidence") or 0.0),
            "route": layout_preflight.get("route"),
            "rows": layout_preflight.get("rows"),
            "columns": layout_preflight.get("columns"),
            "rhythm_strip": bool(layout_preflight.get("rhythm_strip")),
            "rotation_deg": layout_preflight.get("rotation_deg"),
            "unconfirmed_probe": bool(allow_unconfirmed_probe),
            "probe_reason": layout_preflight.get("probe_reason"),
        },
        "signal_extractor_num_peaks": signal_info.get(
            "signal_extractor_num_peaks"
        ),
        "pixel_spacing_mm": {
            "x": float(pixel["x"]) if pixel.get("x") is not None else None,
            "y": float(pixel["y"]) if pixel.get("y") is not None else None,
        },
        "units_from_digitizer": "uV",
        "target_samples": 5000,
    }
    return signal_uv, meta


def _finite_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if not mask.any():
        return []
    d = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b > a]


def _build_r27_tiled_signal(
    signal_uv: np.ndarray,
    *,
    fs: int = 500,
    target_samples: int = 5000,
    min_real_seconds: float = 1.5,
) -> tuple[np.ndarray, dict]:
    """Build an explicit research-only 10 s compatibility signal.

    Each lead is handled independently:
    - if all 10 s are genuinely observed, preserve them unchanged;
    - otherwise take the longest contiguous finite observed segment,
      repeat that exact segment end-to-end, and truncate to 10 s.

    No interpolation, smoothing, phase shifting, cross-fading or synthetic
    morphology is introduced. This adapter changes temporal repetition only.
    """
    x = np.asarray(signal_uv, dtype=np.float64)
    if x.shape != (target_samples, 12):
        raise RuntimeError(
            f"Forma inesperada para R27-TILED: {x.shape}; "
            f"se esperaba ({target_samples}, 12)."
        )

    min_real_samples = int(round(float(min_real_seconds) * float(fs)))
    out = np.empty_like(x)
    lead_meta: dict[str, dict] = {}

    for j, lead in enumerate(LEADS):
        col = x[:, j]
        finite = np.isfinite(col)
        observed_total = int(finite.sum())
        observed_fraction = float(observed_total / target_samples)

        if observed_total == target_samples:
            out[:, j] = col
            lead_meta[lead] = {
                "mode": "REAL_10S",
                "observed_total_samples": observed_total,
                "observed_fraction": round(observed_fraction, 6),
                "source_start_sample": 0,
                "source_end_sample": target_samples,
                "source_samples": target_samples,
                "source_seconds": round(target_samples / fs, 6),
                "repeat_count_ceiling": 1,
                "repeated_output_fraction": 0.0,
                "seam_count": 0,
            }
            continue

        runs = _finite_runs(finite)
        if not runs:
            raise RuntimeError(
                f"R27-TILED no puede ejecutarse: {lead} no tiene muestras observadas."
            )

        a, b = max(runs, key=lambda ab: ab[1] - ab[0])
        segment = np.asarray(col[a:b], dtype=np.float64)
        n = int(segment.size)

        if n < min_real_samples:
            raise RuntimeError(
                f"R27-TILED no puede ejecutarse: {lead} sólo tiene "
                f"{n / fs:.2f} s contiguos observados; mínimo requerido "
                f"{min_real_seconds:.2f} s."
            )
        if not np.isfinite(segment).all():
            raise RuntimeError(
                f"R27-TILED encontró valores no finitos dentro del segmento de {lead}."
            )

        reps = int(np.ceil(target_samples / n))
        tiled = np.tile(segment, reps)[:target_samples]
        if tiled.shape != (target_samples,) or not np.isfinite(tiled).all():
            raise RuntimeError(f"R27-TILED produjo una señal inválida para {lead}.")

        out[:, j] = tiled
        lead_meta[lead] = {
            "mode": "EXACT_REPEAT_OF_OBSERVED_SEGMENT",
            "observed_total_samples": observed_total,
            "observed_fraction": round(observed_fraction, 6),
            "source_start_sample": int(a),
            "source_end_sample": int(b),
            "source_samples": n,
            "source_seconds": round(n / fs, 6),
            "repeat_count_ceiling": reps,
            "repeated_output_fraction": round(
                float(max(0, target_samples - n) / target_samples),
                6,
            ),
            "seam_count": max(0, reps - 1),
        }

    if not np.isfinite(out).all():
        raise RuntimeError("R27-TILED produjo valores no finitos.")

    repeated_leads = [
        lead for lead, info in lead_meta.items()
        if info["mode"] != "REAL_10S"
    ]
    real_10s_leads = [
        lead for lead, info in lead_meta.items()
        if info["mode"] == "REAL_10S"
    ]

    return out, {
        "adapter": "R27_SYNTHETIC_10S_FROM_OBSERVED_SEGMENT_REPEAT",
        "adapter_version": "1.0",
        "research_only": True,
        "validated_equivalent_to_real_10s": False,
        "fs_hz": int(fs),
        "target_samples": int(target_samples),
        "target_seconds": round(target_samples / fs, 6),
        "min_real_seconds_per_lead": float(min_real_seconds),
        "repeated_leads": repeated_leads,
        "real_10s_leads": real_10s_leads,
        "lead_provenance": lead_meta,
        "transformation": (
            "Longest contiguous finite observed segment repeated exactly end-to-end "
            "and truncated to 10 s; no interpolation or new morphology."
        ),
    }


def _build_r27_tiled_with_reference(
    primary_signal_uv: np.ndarray,
    *,
    reference_signal_uv: np.ndarray | None = None,
    fs: int = 500,
    target_samples: int = 5000,
    min_real_seconds: float = 1.5,
) -> tuple[np.ndarray, dict, str, str | None]:
    """Prefer the high-fidelity route, then retry on an independent 1200px route.

    The fallback is whole-route, never a lead-by-lead splice. This preserves a
    coherent calibration/provenance contract for R27-TILED while allowing the
    older, validated temporal extraction route to rescue fragmentation introduced
    by the higher-resolution segmentation path.
    """
    try:
        tiled_uv, tiled_meta = _build_r27_tiled_signal(
            primary_signal_uv,
            fs=fs,
            target_samples=target_samples,
            min_real_seconds=min_real_seconds,
        )
        return (
            tiled_uv,
            tiled_meta,
            "PRIMARY_HIGH_FIDELITY_ROUTE",
            None,
        )
    except RuntimeError as primary_exc:
        primary_reason = str(primary_exc)
        if "R27-TILED no puede ejecutarse:" not in primary_reason:
            raise
        if reference_signal_uv is None:
            raise

    try:
        tiled_uv, tiled_meta = _build_r27_tiled_signal(
            reference_signal_uv,
            fs=fs,
            target_samples=target_samples,
            min_real_seconds=min_real_seconds,
        )
    except RuntimeError as reference_exc:
        reference_reason = str(reference_exc)
        if "R27-TILED no puede ejecutarse:" not in reference_reason:
            raise
        raise RuntimeError(
            primary_reason
            + " | Ruta temporal 1200 px también rechazada: "
            + reference_reason
        ) from reference_exc

    return (
        tiled_uv,
        tiled_meta,
        "LOW_MEMORY_1200_FORCED_6X2_REFERENCE_FALLBACK",
        primary_reason,
    )


def _mark_r27_tiled_unavailable(meta: dict, tiled_reason: str) -> None:
    signal = meta.setdefault("signal", {})
    signal["r27_input_compatible"] = False
    signal["r27_input_mode"] = (
        "UNAVAILABLE_INSUFFICIENT_CONTIGUOUS_LEAD_COVERAGE"
    )
    signal["r27_tiled"] = False
    signal["r27_tiled_rejection_reason"] = str(tiled_reason)
    signal["r27_compatibility_rule"] = (
        "R27-TILED requires at least 1.50 s of genuinely observed "
        "contiguous signal in every incomplete lead. The threshold "
        "was not lowered."
    )
    meta["status"] = "DIGITIZED_ONLY"
    meta["reason"] = (
        "Digitalización U-Net completada y reporte estructurado conservado. "
        "R27 se omitió porque al menos una derivación no alcanzó 1.50 s "
        "contiguos observados para R27-TILED. "
        + str(tiled_reason)
    )


def _write_wfdb_pair(
    signal_uv: np.ndarray,
    output500: Path,
    output100: Path,
    record_name: str,
) -> dict:
    import wfdb
    from scipy.signal import resample_poly

    if signal_uv.shape != (5000, 12):
        raise RuntimeError(f"Forma inesperada antes de WFDB: {signal_uv.shape}")
    if not np.isfinite(signal_uv).all():
        raise RuntimeError(
            "La señal contiene muestras no observadas. No se permite imputarlas antes de R27."
        )

    # Open ECG Digitizer reports microvolts. PTB-XL / R27 reads physical ECG
    # amplitudes in millivolts through WFDB.
    signal500_mv = signal_uv / 1000.0

    if not np.isfinite(signal500_mv).all():
        raise RuntimeError("Conversión uV→mV produjo valores no finitos.")

    signal100_mv = resample_poly(signal500_mv, up=1, down=5, axis=0)
    if signal100_mv.shape != (1000, 12):
        raise RuntimeError(f"Resample 100 Hz inesperado: {signal100_mv.shape}")
    if not np.isfinite(signal100_mv).all():
        raise RuntimeError("Resample 100 Hz produjo valores no finitos.")

    output500.mkdir(parents=True, exist_ok=True)
    output100.mkdir(parents=True, exist_ok=True)

    common = dict(
        units=["mV"] * 12,
        sig_name=LEADS,
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
    )

    wfdb.wrsamp(
        record_name,
        fs=500,
        p_signal=signal500_mv,
        write_dir=str(output500),
        **common,
    )
    wfdb.wrsamp(
        record_name,
        fs=100,
        p_signal=signal100_mv,
        write_dir=str(output100),
        **common,
    )

    return {
        "shape_500": [5000, 12],
        "shape_100": [1000, 12],
        "fs_500": 500,
        "fs_100": 100,
        "units_for_r27": "mV",
        "adapter_100hz": "scipy.signal.resample_poly(up=1, down=5)",
    }


def _primary_route_needs_temporal_reference(
    signal_uv: np.ndarray,
    signal_meta: dict,
    *,
    allow_r27_tiled: bool,
) -> tuple[bool, dict]:
    """Run the 1200 px reference route only when it can still change the result.

    The high-fidelity route remains the primary clinical source.  The compact
    reference is retained as a rescue path for ambiguous layout, fragmented
    morphology, insufficient native rhythm, or R27 tiling incompatibility.
    """
    reasons: list[str] = []
    router = dict(signal_meta.get("layout_hypothesis_router") or {})
    selected_score = float(router.get("selected_score") or 0.0)
    margin = float(router.get("margin") or 0.0)
    decision = str(router.get("decision") or "")
    selected_layout = str(router.get("selected_layout") or "")
    selected_candidate = None
    for candidate in router.get("candidates") or []:
        if str(candidate.get("layout") or "") == selected_layout:
            selected_candidate = candidate
            break
    metrics = dict((selected_candidate or {}).get("metrics") or {})

    if decision != "SELECTED":
        reasons.append("LAYOUT_NOT_SELECTED")
    if selected_score < 0.90:
        reasons.append(f"LAYOUT_SCORE_LT_0_90:{selected_score:.3f}")
    if margin < 0.10:
        reasons.append(f"LAYOUT_MARGIN_LT_0_10:{margin:.3f}")
    if int(metrics.get("recovered_leads") or 0) < 12:
        reasons.append("LT_12_RECOVERED_LEADS")
    if float(metrics.get("continuity_score") or 0.0) < 0.90:
        reasons.append("CONTINUITY_LT_0_90")

    rhythm_detected = bool(
        signal_meta.get("rhythm_strip_detected")
        or metrics.get("rhythm_detected")
    )
    if rhythm_detected:
        rhythm_observed = bool(signal_meta.get("rhythm_strip_observed"))
        rhythm_coverage = float(signal_meta.get("rhythm_strip_coverage") or 0.0)
        rhythm_quality = float(metrics.get("rhythm_quality") or 0.0)
        if not rhythm_observed:
            reasons.append("RHYTHM_STRIP_NOT_OBSERVED")
        if rhythm_coverage < 0.90:
            reasons.append(f"RHYTHM_COVERAGE_LT_0_90:{rhythm_coverage:.3f}")
        if rhythm_quality < 0.90:
            reasons.append(f"RHYTHM_QUALITY_LT_0_90:{rhythm_quality:.3f}")

    r27_primary_ready = True
    r27_primary_reason = None
    if bool(allow_r27_tiled) and not bool(signal_meta.get("all_samples_observed")):
        try:
            _build_r27_tiled_signal(
                signal_uv,
                fs=500,
                target_samples=5000,
                min_real_seconds=1.5,
            )
        except RuntimeError as exc:
            r27_primary_ready = False
            r27_primary_reason = str(exc)
            reasons.append("PRIMARY_R27_TILING_NOT_READY")

    return bool(reasons), {
        "policy": "ADAPTIVE_REFERENCE_V1",
        "reference_required": bool(reasons),
        "reasons": reasons,
        "selected_score": round(selected_score, 6),
        "margin": round(margin, 6),
        "recovered_leads": int(metrics.get("recovered_leads") or 0),
        "continuity_score": float(metrics.get("continuity_score") or 0.0),
        "rhythm_detected": rhythm_detected,
        "rhythm_strip_observed": bool(signal_meta.get("rhythm_strip_observed")),
        "rhythm_strip_coverage": signal_meta.get("rhythm_strip_coverage"),
        "rhythm_quality": metrics.get("rhythm_quality"),
        "r27_primary_ready": r27_primary_ready,
        "r27_primary_reason": r27_primary_reason,
    }


def _should_probe_unknown_dense_6x2(layout_preflight: dict) -> bool:
    """Select a guarded 6x2 signal probe for noisy scanned ECG geometry.

    Some portrait/phone-scanned 6x2+1R pages create extra projection-profile
    peaks around tall QRS complexes. The lightweight preflight can then return
    8-10 row candidates and no layout, even though the page is a standard 6x2
    family ECG. A probe is safe because _digitize_forced_layout still requires
    six coherent U-Net signal rows plus independent Open-ECG corroboration
    before the layout is accepted.
    """
    if layout_preflight.get("layout") is not None:
        return False
    centers = list(layout_preflight.get("row_centers_y") or [])
    if not (7 <= len(centers) <= 10):
        return False
    try:
        rotation = abs(float(layout_preflight.get("rotation_deg") or 0.0))
    except Exception:
        return False
    return bool(rotation <= 6.0)


def _layout_family(value: object) -> str | None:
    """Normalize layout labels emitted by geometry and neural routes."""
    s = str(value or "").strip().lower()
    if "6x2" in s:
        return "6x2"
    if "3x4" in s:
        return "3x4"
    if "12x1" in s or "cabrera" in s:
        return "12x1"
    return None


def _trusted_preflight_layout(layout_preflight: dict) -> str | None:
    """Return a standard geometry-first layout only when preflight is strong.

    The preflight detector uses page/grid geometry rather than lead-name
    semantics.  A strong 3x4/6x2 result constrains later neural identification,
    but does not bypass signal extraction or quality gates.
    """
    family = _layout_family(layout_preflight.get("layout"))
    if family not in {"3x4", "6x2"}:
        return None
    try:
        confidence = float(layout_preflight.get("confidence") or 0.0)
    except Exception:
        return None
    try:
        leads_detected = int(layout_preflight.get("leads_detected") or 0)
    except Exception:
        leads_detected = 0
    if confidence < 0.80 or leads_detected < 12:
        return None
    return family


def _assert_layout_constraint(
    signal_meta: dict,
    expected_layout: str | None,
) -> None:
    if expected_layout is None:
        return
    actual = _layout_family(signal_meta.get("layout_name"))
    if actual != expected_layout:
        raise RuntimeError(
            "TRUSTED_PREFLIGHT_LAYOUT_CONSTRAINT_NOT_HONORED: "
            f"expected={expected_layout}; actual={actual}; "
            f"raw={signal_meta.get('layout_name')}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor-root", required=True)
    ap.add_argument("--segmentation-model", required=True)
    ap.add_argument("--lead-model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--pdf-page-index", type=int, default=0)
    ap.add_argument("--speed-mm-per-s", type=float, default=25.0)
    ap.add_argument("--gain-mm-per-mv", type=float, default=10.0)
    ap.add_argument(
        "--allow-r27-tiled",
        action="store_true",
        help=(
            "Research-only: repeat each lead's longest observed segment to 10 s "
            "when a true 10 s x 12 lead record is unavailable."
        ),
    )
    ap.add_argument(
        "--force-low-memory",
        action="store_true",
        help=(
            "Disable the higher-resolution segmentation-only route. Used as an "
            "automatic retry after a worker OOM/resource failure."
        ),
    )
    args = ap.parse_args()
    worker_started = time.perf_counter()

    vendor_root = Path(args.vendor_root).resolve()
    segmentation_model = Path(args.segmentation_model).resolve()
    lead_model = Path(args.lead_model).resolve()
    source = Path(args.source).resolve()
    output_root = Path(args.output_root).resolve()
    meta_path = Path(args.meta).resolve()

    image_dir = output_root / "input"
    output500 = output_root / "wfdb500"
    output100 = output_root / "wfdb100"
    image_dir.mkdir(parents=True, exist_ok=True)

    record_name = "medcalc_photo_ecg"
    preflight_image_path = image_dir / f"{record_name}_preflight.png"
    high_fidelity_image_path = image_dir / f"{record_name}_high_fidelity.png"

    meta: dict = {
        "status": "STARTED",
        "digitizer": "Ahus-AIM/Open-ECG-Digitizer",
        "digitizer_commit": "97a15087d4abcda843da8c58ee74b1d8f47e6f9a",
        "segmentation_model_sha256": "17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b",
        "lead_model_sha256": "840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2",
        "license": "CC BY-SA 4.0",
        "performance": {},
    }

    try:
        print("[ECG-U-NET] PREPARE_IMAGE", flush=True)
        meta["image"] = _prepare_source_image(
            source,
            int(args.pdf_page_index),
            preflight_image_path,
            max_dimension=LOW_MEMORY_IMAGE_MAX_DIM,
            pdf_dpi=150.0,
        )
        meta["image"]["role"] = "PREFLIGHT_AND_LOW_MEMORY_FALLBACK"

        print("[ECG-LAYOUT] PREFLIGHT_START", flush=True)
        layout_preflight = detect_ecg_layout(preflight_image_path)
        meta["layout_detector"] = layout_preflight
        print(
            "[ECG-LAYOUT] "
            f"layout={layout_preflight.get('layout')} "
            f"confidence={float(layout_preflight.get('confidence') or 0.0):.3f} "
            f"route={layout_preflight.get('route')} "
            f"rows={layout_preflight.get('rows')} "
            f"columns={layout_preflight.get('columns')} "
            f"rhythm={layout_preflight.get('rhythm_strip')}",
            flush=True,
        )

        # Geometry-first authority: a strong physical 3x4/6x2 preflight
        # constrains later layout identification. U-Net remains authoritative
        # for trace segmentation and signal quality, but may not silently
        # relabel a page whose standard geometry is already well established.
        trusted_preflight_layout = _trusted_preflight_layout(layout_preflight)
        meta["trusted_preflight_layout"] = trusted_preflight_layout
        meta["trusted_preflight_layout_policy"] = (
            "CONSTRAIN_NEURAL_LAYOUT_DO_NOT_BYPASS_SIGNAL_QC"
            if trusted_preflight_layout
            else "NO_STRONG_STANDARD_GEOMETRY_CONSTRAINT"
        )
        inference_image_path = preflight_image_path
        inference_resample = LOW_MEMORY_RESAMPLE_SIZE
        fidelity_mode = "LOW_MEMORY_NEURAL_LAYOUT"

        if not args.force_low_memory:
            print("[ECG-U-NET] PREPARE_HIGH_FIDELITY_IMAGE", flush=True)
            meta["high_fidelity_image"] = _prepare_source_image(
                source,
                int(args.pdf_page_index),
                high_fidelity_image_path,
                max_dimension=HIGH_FIDELITY_IMAGE_MAX_DIM,
                pdf_dpi=240.0,
            )
            meta["high_fidelity_image"]["role"] = (
                "PRIMARY_SEGMENTATION_FOR_LAYOUT_HYPOTHESIS_ROUTER"
            )
            meta["high_fidelity_normalization"] = _normalize_high_fidelity_input(
                high_fidelity_image_path,
                layout_preflight,
            )
            inference_image_path = high_fidelity_image_path
            inference_resample = HIGH_FIDELITY_RESAMPLE_SIZE
            fidelity_mode = "HIGH_FIDELITY_LAYOUT_HYPOTHESIS_ROUTER_V2"

        primary_inference_started = time.perf_counter()
        print(
            "[ECG-U-NET] LOAD_MODELS "
            f"resample={int(inference_resample)} mode={fidelity_mode}",
            flush=True,
        )
        model = _load_digitizer(
            vendor_root,
            segmentation_model,
            lead_model,
            resample_size=int(inference_resample),
        )
        print("[ECG-U-NET] INFERENCE_START", flush=True)

        if not args.force_low_memory:
            try:
                signal_uv, signal_meta = _digitize_layout_hypotheses(
                    inference_image_path,
                    model,
                    layout_preflight=layout_preflight,
                    speed_mm_per_s=args.speed_mm_per_s,
                    gain_mm_per_mv=args.gain_mm_per_mv,
                )
                meta["layout_router"] = signal_meta.get(
                    "layout_hypothesis_router"
                )
                selected_family = _layout_family(signal_meta.get("layout_name"))
                if (
                    trusted_preflight_layout is not None
                    and selected_family != trusted_preflight_layout
                ):
                    meta["post_unet_layout_rejected"] = {
                        "selected_layout": selected_family,
                        "selected_raw": signal_meta.get("layout_name"),
                        "trusted_preflight_layout": trusted_preflight_layout,
                        "reason": "CONTRADICTS_STRONG_PHYSICAL_GEOMETRY",
                    }
                    raise LayoutHypothesisRoutingError(
                        "POST_UNET_LAYOUT_CONTRADICTS_TRUSTED_PREFLIGHT: "
                        f"preflight={trusted_preflight_layout}; "
                        f"post_unet={selected_family}"
                    )
                print(
                    "[ECG-LAYOUT] POST_UNET_SELECTED "
                    f"layout={signal_meta.get('layout_name')} "
                    f"score={((signal_meta.get('layout_hypothesis_router') or {}).get('selected_score'))}",
                    flush=True,
                )
            except LayoutHypothesisRoutingError as route_exc:
                # Ambiguous standard-layout evidence is not forced.  Run the
                # independent lead-name identifier at compact resolution.  This
                # fallback also preserves support for non-3x4/6x2 layouts such
                # as 12x1/Cabrera when Open-ECG can identify them.
                meta["layout_hypothesis_fallback_reason"] = str(route_exc)
                print(
                    "[ECG-LAYOUT] HYPOTHESES_UNRESOLVED -> "
                    "NEURAL_LAYOUT_FALLBACK: "
                    + str(route_exc),
                    flush=True,
                )
                del model
                gc.collect()
                model = _load_digitizer(
                    vendor_root,
                    segmentation_model,
                    lead_model,
                    resample_size=LOW_MEMORY_RESAMPLE_SIZE,
                )
                signal_uv, signal_meta = _digitize_image(
                    preflight_image_path,
                    model,
                    layout_hint=trusted_preflight_layout,
                )
                _assert_layout_constraint(
                    signal_meta,
                    trusted_preflight_layout,
                )
                signal_meta["preflight_layout_constraint"] = (
                    trusted_preflight_layout
                )
                fidelity_mode = (
                    "LOW_MEMORY_NEURAL_LAYOUT_CONSTRAINED_BY_GEOMETRY"
                    if trusted_preflight_layout
                    else "LOW_MEMORY_NEURAL_LAYOUT_AFTER_HYPOTHESIS_AMBIGUITY"
                )
                inference_resample = LOW_MEMORY_RESAMPLE_SIZE
        else:
            signal_uv, signal_meta = _digitize_image(
                preflight_image_path,
                model,
                layout_hint=trusted_preflight_layout,
            )
            _assert_layout_constraint(
                signal_meta,
                trusted_preflight_layout,
            )
            signal_meta["preflight_layout_constraint"] = (
                trusted_preflight_layout
            )

        meta["performance"]["primary_model_load_and_inference_seconds"] = round(
            time.perf_counter() - primary_inference_started,
            3,
        )
        print("[ECG-U-NET] INFERENCE_DONE", flush=True)

        # Release primary model before the independent temporal/reference pass.
        del model
        gc.collect()

        reference_signal_uv = None
        reference_signal_meta = None
        reference_route_label = None

        # The 1200 px route is a rescue/reference path, not a mandatory second
        # inference.  Skip it when the primary high-fidelity route is already
        # strong enough for layout, continuity, native rhythm and R27 input.
        reference_required = False
        reference_gate = {
            "policy": "NOT_APPLICABLE_NON_HIGH_FIDELITY",
            "reference_required": False,
        }
        if fidelity_mode == "HIGH_FIDELITY_LAYOUT_HYPOTHESIS_ROUTER_V2":
            reference_required, reference_gate = (
                _primary_route_needs_temporal_reference(
                    signal_uv,
                    signal_meta,
                    allow_r27_tiled=bool(args.allow_r27_tiled),
                )
            )
            meta["temporal_reference_gate"] = reference_gate

        if (
            fidelity_mode == "HIGH_FIDELITY_LAYOUT_HYPOTHESIS_ROUTER_V2"
            and reference_required
        ):
            reference_started = time.perf_counter()
            print(
                "[ECG-U-NET] TEMPORAL_REFERENCE_START "
                f"resample={LOW_MEMORY_RESAMPLE_SIZE}",
                flush=True,
            )
            reference_model = None
            primary_layout = str(
                signal_meta.get("layout_name") or ""
            ).split("+", 1)[0]
            try:
                reference_model = _load_digitizer(
                    vendor_root,
                    segmentation_model,
                    lead_model,
                    resample_size=LOW_MEMORY_RESAMPLE_SIZE,
                )
                try:
                    reference_signal_uv, reference_signal_meta = (
                        _digitize_layout_hypotheses(
                            preflight_image_path,
                            reference_model,
                            speed_mm_per_s=args.speed_mm_per_s,
                            gain_mm_per_mv=args.gain_mm_per_mv,
                        )
                    )

                    reference_layout = str(
                        reference_signal_meta.get("layout_name") or ""
                    ).split("+", 1)[0]
                    if reference_layout != primary_layout:
                        raise LayoutHypothesisRoutingError(
                            "La referencia temporal seleccionó un layout distinto "
                            f"({reference_layout}) al primario ({primary_layout})."
                        )

                    reference_route_label = (
                        "LOW_MEMORY_1200_LAYOUT_HYPOTHESIS_REFERENCE"
                    )
                    reference_reason = None
                except Exception as full_reference_exc:
                    # Open-ECG's inference wrapper is effectively single-use for
                    # repeated segmentation calls: after one forward pass some
                    # internal model attributes may be released.  Never reuse
                    # the failed full-layout wrapper for the strip-only pass.
                    # Reload a fresh 1200 px instance so rhythm recovery is an
                    # independent inference, not a second call on mutated state.
                    print(
                        "[ECG-U-NET] TEMPORAL_FULL_LAYOUT_REJECTED -> "
                        "STRIP_ONLY_REFERENCE_FRESH_MODEL: "
                        + str(full_reference_exc),
                        flush=True,
                    )
                    if reference_model is not None:
                        del reference_model
                        reference_model = None
                        gc.collect()
                    reference_model = _load_digitizer(
                        vendor_root,
                        segmentation_model,
                        lead_model,
                        resample_size=LOW_MEMORY_RESAMPLE_SIZE,
                    )
                    reference_signal_uv, reference_signal_meta = (
                        _digitize_temporal_strip_only(
                            preflight_image_path,
                            reference_model,
                            layout_hint=primary_layout,
                        )
                    )
                    reference_route_label = (
                        "LOW_MEMORY_1200_TEMPORAL_STRIP_ONLY_FRESH_MODEL"
                    )
                    reference_reason = str(full_reference_exc)

                meta["temporal_reference"] = {
                    "status": "PASS",
                    "route": reference_route_label,
                    "layout": reference_signal_meta.get("layout_name"),
                    "full_layout_rejection_reason": reference_reason,
                    "layout_router": reference_signal_meta.get(
                        "layout_hypothesis_router"
                    ),
                    "temporal_strip_router": reference_signal_meta.get(
                        "temporal_strip_router"
                    ),
                    "rhythm_strip_observed": bool(
                        reference_signal_meta.get("rhythm_strip_observed")
                    ),
                    "rhythm_strip_coverage": reference_signal_meta.get(
                        "rhythm_strip_coverage"
                    ),
                    "rhythm_strip_longest_contiguous_fraction": (
                        reference_signal_meta.get(
                            "rhythm_strip_longest_contiguous_fraction"
                        )
                    ),
                    "observed_fraction_by_lead": reference_signal_meta.get(
                        "observed_fraction_by_lead"
                    ),
                    "observed_seconds_by_lead": reference_signal_meta.get(
                        "observed_seconds_by_lead"
                    ),
                }
                print(
                    "[ECG-U-NET] TEMPORAL_REFERENCE_DONE "
                    f"route={reference_route_label} "
                    f"layout={reference_signal_meta.get('layout_name')} "
                    f"rhythm_coverage={float(reference_signal_meta.get('rhythm_strip_coverage') or 0.0):.3f}",
                    flush=True,
                )
            except Exception as reference_exc:
                reference_signal_uv = None
                reference_signal_meta = None
                reference_route_label = None
                meta["temporal_reference"] = {
                    "status": "FAIL",
                    "route": "LOW_MEMORY_1200_TEMPORAL_REFERENCE",
                    "reason": str(reference_exc),
                }
                print(
                    "[ECG-U-NET] TEMPORAL_REFERENCE_FAILED: "
                    + str(reference_exc),
                    flush=True,
                )
            finally:
                if reference_model is not None:
                    del reference_model
                gc.collect()
                meta["performance"]["temporal_reference_seconds"] = round(
                    time.perf_counter() - reference_started,
                    3,
                )
        elif fidelity_mode == "HIGH_FIDELITY_LAYOUT_HYPOTHESIS_ROUTER_V2":
            meta["temporal_reference"] = {
                "status": "SKIPPED_PRIMARY_QC_PASS",
                "route": None,
                "reason": (
                    "Primary high-fidelity route satisfied adaptive layout, "
                    "continuity, rhythm and R27-readiness gates."
                ),
                "gate": reference_gate,
            }
            meta["performance"]["temporal_reference_seconds"] = 0.0
            print(
                "[ECG-U-NET] TEMPORAL_REFERENCE_SKIPPED_PRIMARY_QC_PASS",
                flush=True,
            )

        meta["signal"] = signal_meta
        meta["signal"]["fidelity_mode"] = fidelity_mode
        meta["signal"]["inference_resample_max_dimension"] = int(
            inference_resample
        )
        meta["signal"]["force_low_memory"] = bool(args.force_low_memory)

        print("[ECG-U-NET] STRUCTURED_REPORT", flush=True)
        # Never turn an untrusted lead mapping into a clinical-looking report.
        # A standard 3x4 page is acceptable for descriptive measurements only
        # after either neural layout identification or the exact-four-row
        # deterministic fallback above.
        layout_trusted = signal_meta["layout_name"] != "Unknown layout"
        recovered_leads = sum(
            1 for v in signal_meta["observed_fraction_by_lead"].values()
            if float(v) >= 0.15
        )
        report_input_trusted = bool(layout_trusted and recovered_leads >= 10)

        if not report_input_trusted:
            meta["structured_report"] = {
                "version": "ECG_STRUCTURED_REPORT_V1",
                "error": "UNTRUSTED_DIGITIZED_LEAD_MAPPING",
                "formatted": {
                    "text": (
                        "RITMO: NO EVALUABLE.\n"
                        "FC: NO EVALUABLE.\n"
                        "EJE: NO EVALUABLE.\n"
                        "SEGMENTO PR: NO EVALUABLE.\n"
                        "COMPLEJO QRS: NO EVALUABLE.\n"
                        "SEGMENTO ST: NO EVALUABLE.\n"
                        "ONDA T: NO EVALUABLE.\n"
                        "EXTRASISTOLIA: NO EVALUABLE.\n"
                        "CONCLUSIÓN: DIGITALIZACIÓN INSUFICIENTE PARA INFORME ELECTROCARDIOGRÁFICO AUTOMATIZADO.\n"
                        "IDX: REVISIÓN MANUAL."
                    )
                },
            }
        else:
            signal_primary_report = signal_meta.get(
                "signal_primary_structured_report"
            )
            if isinstance(signal_primary_report, dict):
                # V2 architecture: the clinical analyzer consumes the calibrated
                # digital signal reconstructed from U-Net centerlines. The image
                # and preflight detector are no longer measurement sources.
                meta["structured_report"] = signal_primary_report
                meta["structured_report"]["input_quality_gate"] = {
                    "layout_trusted": True,
                    "recovered_leads_ge_15pct": int(recovered_leads),
                    "layout_source": signal_meta.get("layout_source"),
                    "clinical_measurement_source": "CALIBRATED_DIGITAL_SIGNAL_V2",
                    "calibration": signal_meta.get("calibration"),
                    "temporal_reference_role": "AUDIT_OR_FALLBACK_ONLY",
                    "temporal_reference_status": (
                        (meta.get("temporal_reference") or {}).get("status")
                    ),
                }
            else:
                # Compatibility path for legacy/neural-layout fallbacks that do
                # not yet expose the V2 calibrated per-lead contract. This still
                # measures a digitized signal, never the source raster.
                try:
                    from ecg_structured_report import build_structured_ecg_report
                    meta["structured_report"] = build_structured_ecg_report(
                        signal_uv,
                        fs=500,
                        lead_names=LEADS,
                    )
                    meta["structured_report"]["input_quality_gate"] = {
                        "layout_trusted": True,
                        "recovered_leads_ge_15pct": int(recovered_leads),
                        "layout_source": signal_meta.get("layout_source"),
                        "clinical_measurement_source": (
                            "LEGACY_DIGITIZED_SIGNAL_COMPATIBILITY"
                        ),
                        "v2_unavailable_reason": signal_meta.get(
                            "signal_primary_measurement_error"
                        ),
                    }
                except Exception as report_exc:
                    meta["structured_report"] = {
                        "version": "ECG_STRUCTURED_REPORT_V1",
                        "error": str(report_exc),
                        "formatted": {
                            "text": (
                                "RITMO: NO EVALUABLE.\n"
                                "FC: NO EVALUABLE.\n"
                                "EJE: NO EVALUABLE.\n"
                                "SEGMENTO PR: NO EVALUABLE.\n"
                                "COMPLEJO QRS: NO EVALUABLE.\n"
                                "SEGMENTO ST: NO EVALUABLE.\n"
                                "ONDA T: NO EVALUABLE.\n"
                                "EXTRASISTOLIA: NO EVALUABLE.\n"
                                "CONCLUSIÓN: REPORTE AUTOMATIZADO NO DISPONIBLE.\n"
                                "IDX: REVISIÓN MANUAL."
                            )
                        },
                    }

        # R27 input routing.
        #
        # Original R27 remains untouched. When all 12 leads contain true 10 s,
        # preserve the historical exact route. For printed ECGs, an explicitly
        # labeled research-only compatibility adapter may repeat the longest
        # observed contiguous segment of each incomplete lead to reach 10 s.
        if signal_meta["all_samples_observed"]:
            wfdb_meta = _write_wfdb_pair(
                signal_uv,
                output500,
                output100,
                record_name,
            )
            meta["signal"].update(wfdb_meta)
            meta["signal"]["r27_input_compatible"] = True
            meta["signal"]["r27_input_mode"] = "REAL_10S_12_LEAD"
            meta["signal"]["r27_tiled"] = False
            meta["signal"]["r27_signal_source"] = "PRIMARY_HIGH_FIDELITY_ROUTE"
            meta["signal"]["r27_compatibility_rule"] = (
                "12 standard leads; exactly 5000 genuinely observed finite "
                "samples/lead at 500 Hz"
            )
            meta["wfdb_500_base"] = str(output500 / record_name)
            meta["wfdb_100_base"] = str(output100 / record_name)
            meta["status"] = "PASS"
            meta["reason"] = (
                "Digitalización completa: 10 s observados y finitos en las 12 derivaciones."
            )

        elif bool(args.allow_r27_tiled):
            try:
                (
                    tiled_uv,
                    tiled_meta,
                    r27_signal_source,
                    primary_tiled_rejection,
                ) = _build_r27_tiled_with_reference(
                    signal_uv,
                    reference_signal_uv=reference_signal_uv,
                    fs=500,
                    target_samples=5000,
                    min_real_seconds=1.5,
                )
            except RuntimeError as tiled_exc:
                tiled_reason = str(tiled_exc)
                if "R27-TILED no puede ejecutarse:" not in tiled_reason:
                    raise

                # A short lead is an R27 compatibility failure, not a
                # digitization failure. Preserve the genuine U-Net output,
                # native rhythm evidence and structured report instead of
                # discarding the entire ECG because the research-only tiling
                # adapter cannot meet its minimum observed-duration gate.
                _mark_r27_tiled_unavailable(meta, tiled_reason)
            else:
                wfdb_meta = _write_wfdb_pair(
                    tiled_uv,
                    output500,
                    output100,
                    record_name,
                )
                meta["signal"].update(wfdb_meta)
                meta["signal"]["r27_input_compatible"] = True
                meta["signal"]["r27_input_mode"] = (
                    "R27_SYNTHETIC_10S_FROM_OBSERVED_SEGMENT_REPEAT"
                )
                meta["signal"]["r27_tiled"] = True
                meta["signal"]["r27_signal_source"] = r27_signal_source
                if primary_tiled_rejection:
                    meta["signal"]["r27_primary_route_rejection_reason"] = (
                        primary_tiled_rejection
                    )
                meta["signal"]["r27_tiled_provenance"] = tiled_meta
                meta["signal"]["r27_compatibility_rule"] = (
                    "Research-only compatibility route. Incomplete leads are expanded "
                    "to 10 s by exact repetition of the longest contiguous observed segment. "
                    "If the 2000 px morphology route fragments a lead below the 1.50 s "
                    "gate, MEDCALC may retry the complete 1200 px reference route; "
                    "signals are never spliced lead-by-lead across routes."
                )
                meta["signal"]["photo_domain_warning"] = (
                    "R27-TILED is not validated as equivalent to real 10 s x 12-lead input. "
                    "Probabilities must remain probability-only and be interpreted with the "
                    "lead-level tiling provenance."
                )
                meta["wfdb_500_base"] = str(output500 / record_name)
                meta["wfdb_100_base"] = str(output100 / record_name)
                meta["status"] = "PASS_TILED"
                meta["reason"] = (
                    "R27 activado en modo experimental R27-TILED usando "
                    + r27_signal_source
                    + ": las derivaciones incompletas fueron extendidas a 10 s "
                    "mediante repetición exacta del segmento observado. "
                    "No equivale a 10 s reales."
                )

        else:
            meta["status"] = "DIGITIZED_ONLY"
            meta["reason"] = (
                "El trazado fue digitalizado, pero no existen 10 s observados para "
                "las 12 derivaciones y R27-TILED está desactivado."
            )

    except Exception as exc:
        meta.setdefault("performance", {})["worker_total_seconds"] = round(
            time.perf_counter() - worker_started,
            3,
        )
        meta["status"] = "FAIL"
        meta["reason"] = str(exc)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        raise

    meta.setdefault("performance", {})["worker_total_seconds"] = round(
        time.perf_counter() - worker_started,
        3,
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
