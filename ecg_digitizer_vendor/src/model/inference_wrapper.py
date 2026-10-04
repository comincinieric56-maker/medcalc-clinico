import gc
import time
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.nn import Module
from yacs.config import CfgNode as CN

from src.utils import import_class_from_path


@contextmanager
def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
    """Context manager for timing code blocks.

    Args:
        name: Name of the section.
        times_dict: Dictionary to store timing.
    """
    start = time.time()
    yield
    times_dict[name] = time.time() - start


class InferenceWrapper(Module):
    def __init__(
        self,
        config: CN,
        device: str,
        resample_size: None | tuple[int, ...] = None,
        grid_class: int = 0,
        text_background_class: int = 1,
        signal_class: int = 2,
        background_class: int = 3,
        rotate_on_resample: bool = False,
        enable_timing: bool = False,
        minimum_image_size: int = 512,
        apply_dewarping: bool = True,
        segmentation_tile_size: int = 896,
        segmentation_tile_overlap: int = 192,
        segmentation_tiling_threshold: int = 1400,
    ) -> None:
        """Inference wrapper for ECG pipeline.

        Args:
            config: Configuration node.
            device: Torch device string.
            resample_size: Optional resample target size.
            grid_class: Grid class index.
            text_background_class: Text and background class index.
            signal_class: Signal class index.
            background_class: Background class index.
            rotate_on_resample: Whether to rotate on resample.
            enable_timing: Whether to print timings.
            minimum_image_size: Minimum allowed image size.
            apply_dewarping: Whether to apply dewarping (perspective correction is still performed regardless).
        """
        super().__init__()
        self.config = config
        self.device = device
        self.resample_size = resample_size
        self.grid_class = grid_class
        self.text_background_class = text_background_class
        self.signal_class = signal_class
        self.background_class = background_class
        self.rotate_on_resample = rotate_on_resample
        self._timing_enabled = enable_timing
        self.minimum_image_size = minimum_image_size
        self.apply_dewarping = apply_dewarping
        self.segmentation_tile_size = int(segmentation_tile_size)
        self.segmentation_tile_overlap = int(segmentation_tile_overlap)
        self.segmentation_tiling_threshold = int(segmentation_tiling_threshold)

        self.signal_extractor = self._load_signal_extractor()
        self.perspective_detector: Any = self._load_perspective_detector()
        self.segmentation_model: Any = self._load_segmentation_model().to(self.device)
        self.cropper: Any = self._load_cropper()
        self.pixel_size_finder: Any = self._load_pixel_size_finder()
        self.dewarper: Any = self._load_dewarper()

        # MEDCALC low-memory patch: do not keep the lead-name U-Net resident
        # while the segmentation U-Net is running. It is loaded only after the
        # segmentation model and its activations have been released.
        self.identifier = None
        self.times: dict[str, float] = {}

    @torch.no_grad()
    def forward(
        self,
        image: Tensor,
        layout_should_include_substring: None | str,
        skip_identifier: bool = False,
    ) -> dict[str, Tensor | str | float | None | dict[str, Any]]:
        """Performs full inference on an input image.

        Args:
            image: Input image tensor.
            layout_should_include_substring: Optional substring to filter layout names.

        Returns:
            Dictionary with processed outputs and intermediate results.
        """
        self._check_image_dimensions(image)
        image = self.min_max_normalize(image)
        image = image.to(self.device)

        self.times = {}
        image = self._resample_image(image)

        signal_prob, grid_prob, text_prob = self._get_feature_maps(
            image,
            need_text=not skip_identifier,
        )

        with timed_section("Perspective detection", self.times):
            alignment_params = self.perspective_detector(grid_prob)

        with timed_section("Cropping", self.times):
            source_points = self.cropper(signal_prob, alignment_params)

        if skip_identifier:
            # The high-fidelity route needs only signal + grid after perspective
            # estimation. Avoid materializing an aligned RGB image and a full
            # text map at 2000 px; those tensors add substantial RAM but do not
            # contribute to centerline extraction.
            del image
            aligned_signal_prob, aligned_grid_prob = self._align_signal_grid_only(
                signal_prob,
                grid_prob,
                source_points,
            )
            aligned_image = None
            aligned_text_prob = None
        else:
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._align_feature_maps(
                image, signal_prob, grid_prob, text_prob, source_points
            )

        with timed_section("Pixel size search", self.times):
            mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(aligned_grid_prob)
            avg_pixel_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2

        dewarping_status = (
            "NOT_REQUESTED" if not self.apply_dewarping else "REQUESTED"
        )
        dewarping_error = None
        with timed_section("Dewarping", self.times):
            if self.apply_dewarping:
                # Dewarping is useful for photographed/curved paper, but it is
                # a geometric refinement. A failure must not destroy an
                # otherwise valid perspective-corrected ECG. Run the internal
                # grid optimization with gradients enabled, then fail open to
                # the aligned maps while recording provenance.
                signal_before_dewarp = aligned_signal_prob
                grid_before_dewarp = aligned_grid_prob
                try:
                    with torch.enable_grad():
                        self.dewarper.fit(
                            aligned_grid_prob.squeeze().detach(),
                            avg_pixel_per_mm,
                        )
                    aligned_signal_prob = self.dewarper.transform(
                        aligned_signal_prob.squeeze()
                    )
                    aligned_grid_prob = self.dewarper.transform(
                        aligned_grid_prob.squeeze()
                    )
                    # Clinical pixel->mm conversion must use the same final
                    # coordinate system from which centerlines are extracted.
                    (
                        mm_per_pixel_x,
                        mm_per_pixel_y,
                    ) = self.pixel_size_finder(aligned_grid_prob)
                    avg_pixel_per_mm = (
                        1 / mm_per_pixel_x + 1 / mm_per_pixel_y
                    ) / 2
                    dewarping_status = "APPLIED"
                except Exception as exc:
                    aligned_signal_prob = signal_before_dewarp
                    aligned_grid_prob = grid_before_dewarp
                    dewarping_status = "FAILED_FALLBACK_PERSPECTIVE_ONLY"
                    dewarping_error = str(exc)

        with timed_section("Signal extraction", self.times):
            signals = self.signal_extractor(aligned_signal_prob.squeeze())

        self._print_profiling_results()

        extractor_num_peaks = getattr(self.signal_extractor, "num_peaks", None)

        if skip_identifier:
            # Keep only the compact aligned signal probability map required by
            # MEDCALC's row-aware canonicalizer. The second lead-name U-Net is
            # never loaded on this high-confidence geometry route.
            aligned_signal_prob_cpu = aligned_signal_prob.squeeze().cpu()
            from src.model.coordinate_contract import restore_aligned_lines
            signals, aligned_active_x = restore_aligned_lines(
                signals,
                int(aligned_signal_prob_cpu.shape[1]),
                getattr(self.signal_extractor, "last_crop_bounds", None),
            )

            del signal_prob
            del grid_prob
            del text_prob
            del aligned_signal_prob
            del aligned_grid_prob
            del source_points
            del alignment_params
            # aligned_image/text are intentionally absent on this route.
            aligned_image = None
            aligned_text_prob = None

            if hasattr(self, "segmentation_model"):
                del self.segmentation_model
            gc.collect()

            return {
                "layout_name": "PREFLIGHT_FORCED_LAYOUT",
                "signal": {
                    "canonical_lines": None,
                    "raw_lines": signals.cpu(),
                    "raw_lines_coordinate_system": "ALIGNED_CANVAS_PIXELS",
                    "aligned_active_x": aligned_active_x,
                    "aligned_signal_prob": aligned_signal_prob_cpu,
                    "identifier_lines": None,
                    "layout_matching_cost": None,
                    "layout_is_flipped": "False",
                    "identifier_rows_in_layout": None,
                    "identifier_n_detected": None,
                    "identifier_defaulted_layout": False,
                    "signal_extractor_num_peaks": (
                        int(extractor_num_peaks)
                        if extractor_num_peaks is not None
                        else None
                    ),
                },
                "pixel_spacing_mm": {
                    "x": mm_per_pixel_x,
                    "y": mm_per_pixel_y,
                    "average_pixel_per_mm": avg_pixel_per_mm,
                },
                "dewarping": {
                    "requested": bool(self.apply_dewarping),
                    "status": dewarping_status,
                    "error": dewarping_error,
                },
            }

        # MEDCALC low-memory patch: only aligned_text_prob is still required by
        # the lead identifier. Release the first U-Net and all no-longer-needed
        # tensors before loading the second U-Net.
        del image
        del signal_prob
        del grid_prob
        del text_prob
        del aligned_image
        del aligned_signal_prob
        del aligned_grid_prob
        del source_points
        del alignment_params

        if hasattr(self, "segmentation_model"):
            del self.segmentation_model
        gc.collect()

        if self.identifier is None:
            self.identifier = self._load_layout_identifier()

        layout = self.identifier(
            signals,
            aligned_text_prob,
            avg_pixel_per_mm,
            layout_should_include_substring=layout_should_include_substring,
        )
        layout_str = str(layout.get("layout") or "Unknown layout")
        layout_is_flipped = str(bool(layout.get("flip", False)))
        layout_cost = layout.get("cost", 1.0)
        identifier_rows = layout.get("rows_in_layout")
        identifier_n_detected = layout.get("n_detected")
        identifier_defaulted = bool(layout.get("defaulted_layout", False))

        # MEDCALC only consumes the canonical signal, layout metadata and
        # pixel spacing. Do not retain large intermediate image/feature tensors
        # in the returned object.
        return {
            "layout_name": layout_str,
            "signal": {
                "canonical_lines": layout.get("canonical_lines", None),
                # Small tensor (rows x width), retained only so MEDCALC can
                # apply a deterministic geometric fallback when the lead-name
                # U-Net cannot read the printed labels.
                "raw_lines": signals.cpu(),
                "identifier_lines": (
                    layout.get("lines").cpu()
                    if layout.get("lines") is not None
                    else None
                ),
                "layout_matching_cost": layout_cost,
                "layout_is_flipped": layout_is_flipped,
                "identifier_rows_in_layout": (
                    int(identifier_rows) if identifier_rows is not None else None
                ),
                "identifier_n_detected": (
                    int(identifier_n_detected)
                    if identifier_n_detected is not None
                    else None
                ),
                "identifier_defaulted_layout": identifier_defaulted,
                "signal_extractor_num_peaks": (
                    int(extractor_num_peaks)
                    if extractor_num_peaks is not None
                    else None
                ),
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": avg_pixel_per_mm,
            },
            "dewarping": {
                "requested": bool(self.apply_dewarping),
                "status": dewarping_status,
                "error": dewarping_error,
            },
        }

    def _align_signal_grid_only(
        self,
        signal_prob: Tensor,
        grid_prob: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Perspective-align only the maps needed for centerline extraction."""
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(
                signal_prob,
                source_points,
                fill_value=0,
            )
            aligned_grid_prob = self.cropper.apply_perspective(
                grid_prob,
                source_points,
                fill_value=0,
            )

            if self.rotate_on_resample and aligned_signal_prob.shape[2] > aligned_signal_prob.shape[3]:
                aligned_signal_prob = torch.rot90(
                    aligned_signal_prob,
                    k=3,
                    dims=(2, 3),
                )
                aligned_grid_prob = torch.rot90(
                    aligned_grid_prob,
                    k=3,
                    dims=(2, 3),
                )

            prob = torch.clamp(
                (aligned_signal_prob + aligned_grid_prob).squeeze().sum(dim=1)
                - (aligned_signal_prob + aligned_grid_prob).squeeze().sum(dim=1).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            if non_zero.numel() == 0:
                y1, y2 = 0, aligned_signal_prob.shape[2] - 1
            else:
                y1 = int(non_zero[0].item())
                y2 = int(non_zero[-1].item())

            slices = (
                slice(None),
                slice(None),
                slice(y1, y2 + 1),
                slice(None),
            )
            return aligned_signal_prob[slices], aligned_grid_prob[slices]


    def _align_feature_maps(
        self,
        image: Tensor,
        signal_prob: Tensor,
        grid_prob: Tensor,
        text_prob: Tensor,
        source_points: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Aligns image and feature maps using perspective cropping.

        Returns:
            Aligned image, signal, grid, and text tensors.
        """
        with timed_section("Feature map resampling", self.times):
            aligned_signal_prob = self.cropper.apply_perspective(signal_prob, source_points, fill_value=0)
            aligned_image = self.cropper.apply_perspective(image, source_points, fill_value=0)
            aligned_grid_prob = self.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
            aligned_text_prob = self.cropper.apply_perspective(text_prob, source_points, fill_value=0)
            if self.rotate_on_resample:
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._rotate_on_resample(
                    aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
                )
            aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob = self._crop_y(
                aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob
            )

            return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _crop_y(
        self, image: Tensor, signal_prob: Tensor, grid_prob: Tensor, text_prob: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Crops tensors in y and x using bounds from feature maps.

        Returns:
            Cropped image, signal, grid, and text tensors.
        """

        def get_bounds(tensor: Tensor) -> tuple[int, int]:
            prob = torch.clamp(
                tensor.squeeze().sum(dim=tensor.dim() - 3) - tensor.squeeze().sum(dim=tensor.dim() - 3).mean(),
                min=0,
            )
            non_zero = (prob > 0).nonzero(as_tuple=True)[0]
            if non_zero.numel() == 0:
                return 0, tensor.shape[2] - 1
            return int(non_zero[0].item()), int(non_zero[-1].item())

        y1, y2 = get_bounds(signal_prob + grid_prob)

        slices = (slice(None), slice(None), slice(y1, y2 + 1), slice(None))
        return image[slices], signal_prob[slices], grid_prob[slices], text_prob[slices]

    def _print_profiling_results(self) -> None:
        """Prints the timings for each timed section."""
        if not self._timing_enabled:
            return
        print(" Timing results:")
        max_length = max(len(section) for section in self.times.keys())
        for section, duration in self.times.items():
            print(f"    {section:<{max_length+2}}{duration:.2f} s")
        total_time = sum(self.times.values())
        print(f"Total time: {total_time:.2f} s")

    def _rotate_on_resample(
        self,
        aligned_image: Tensor,
        aligned_signal_prob: Tensor,
        aligned_grid_prob: Tensor,
        aligned_text_prob: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Rotates all tensors if height > width.

        Returns:
            Rotated tensors in same order.
        """
        if aligned_image.shape[2] > aligned_image.shape[3]:
            aligned_image = torch.rot90(aligned_image, k=3, dims=(2, 3))
            aligned_signal_prob = torch.rot90(aligned_signal_prob, k=3, dims=(2, 3))
            aligned_grid_prob = torch.rot90(aligned_grid_prob, k=3, dims=(2, 3))
            aligned_text_prob = torch.rot90(aligned_text_prob, k=3, dims=(2, 3))
        return aligned_image, aligned_signal_prob, aligned_grid_prob, aligned_text_prob

    def _resample_image(self, image: Tensor) -> Tensor:
        with timed_section("Initial resampling", self.times):
            if self.resample_size is None:
                return image

            height, width = image.shape[2], image.shape[3]
            min_dim = min(height, width)
            max_dim = max(height, width)

            if min_dim < self.minimum_image_size:
                scale: float = self.minimum_image_size / min_dim
                new_size: tuple[int, int] = (int(height * scale), int(width * scale))
                interpolated: Tensor = F.interpolate(image, size=new_size, mode="bilinear", align_corners=False)
                return interpolated

            if isinstance(self.resample_size, int):
                if max_dim > self.resample_size:
                    scale = self.resample_size / max_dim
                    new_size = (int(height * scale), int(width * scale))
                    return F.interpolate(image, size=new_size, mode="bilinear", align_corners=False, antialias=True)
                return image

            if isinstance(self.resample_size, tuple):
                interpolated = F.interpolate(
                    image, size=self.resample_size, mode="bilinear", align_corners=False, antialias=True
                )
                return interpolated

            raise ValueError(f"Invalid resample_size: {self.resample_size}. Expected int or tuple of (height, width).")

    def process_sparse_prob(self, signal_prob: Tensor) -> Tensor:
        """Normalize one probability map in place to avoid page-sized copies."""
        mean = signal_prob.mean()
        signal_prob.sub_(mean)
        signal_prob.clamp_(min=0)
        max_value = signal_prob.max()
        signal_prob.div_(max_value + 1e-9)
        return signal_prob

    def _get_feature_maps(
        self,
        image: Tensor,
        *,
        need_text: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with timed_section("Segmentation", self.times):
            h, w = int(image.shape[2]), int(image.shape[3])
            if (
                max(h, w) > self.segmentation_tiling_threshold
                and self.segmentation_tile_size > 0
            ):
                return self._get_feature_maps_tiled(
                    image,
                    need_text=need_text,
                )

            logits = self.segmentation_model(image)

            # Avoid materializing a second full 4-channel softmax tensor.
            # logsumexp gives the shared normalizer; only channels required
            # downstream are materialized.
            lse = torch.logsumexp(logits, dim=1, keepdim=True)
            signal_prob = torch.exp(logits[:, [self.signal_class], :, :] - lse)
            grid_prob = torch.exp(logits[:, [self.grid_class], :, :] - lse)
            if need_text:
                text_prob = torch.exp(
                    logits[:, [self.text_background_class], :, :] - lse
                )
            else:
                text_prob = torch.empty(
                    (0,),
                    dtype=signal_prob.dtype,
                    device=signal_prob.device,
                )
            del logits
            del lse

            signal_prob = self.process_sparse_prob(signal_prob)
            grid_prob = self.process_sparse_prob(grid_prob)
            if need_text:
                text_prob = self.process_sparse_prob(text_prob)

            return signal_prob, grid_prob, text_prob

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, tile_size - overlap)
        starts = list(range(0, max(1, length - tile_size + 1), stride))
        last = length - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _tile_blend_weight(
        self,
        height: int,
        width: int,
        *,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        full_h: int,
        full_w: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        weight = torch.ones((1, 1, height, width), dtype=dtype, device=device)
        fade = max(8, self.segmentation_tile_overlap // 2)

        fy = min(fade, max(1, height // 4))
        fx = min(fade, max(1, width // 4))

        if y0 > 0 and fy > 1:
            weight[:, :, :fy, :] *= torch.linspace(
                0.05, 1.0, fy, dtype=dtype, device=device
            ).view(1, 1, fy, 1)
        if y1 < full_h and fy > 1:
            weight[:, :, -fy:, :] *= torch.linspace(
                1.0, 0.05, fy, dtype=dtype, device=device
            ).view(1, 1, fy, 1)
        if x0 > 0 and fx > 1:
            weight[:, :, :, :fx] *= torch.linspace(
                0.05, 1.0, fx, dtype=dtype, device=device
            ).view(1, 1, 1, fx)
        if x1 < full_w and fx > 1:
            weight[:, :, :, -fx:] *= torch.linspace(
                1.0, 0.05, fx, dtype=dtype, device=device
            ).view(1, 1, 1, fx)

        return weight

    def _get_feature_maps_tiled(
        self,
        image: Tensor,
        *,
        need_text: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the segmentation U-Net in overlapping tiles at full resolution.

        This keeps the 2000 px centerline detail while bounding U-Net activation
        memory to roughly one 896x896 tile. Only compact probability maps are
        stitched across the full page; full-resolution network activations never
        coexist for the whole ECG.
        """
        h, w = int(image.shape[2]), int(image.shape[3])
        tile_size = max(512, int(self.segmentation_tile_size))
        overlap = min(
            max(64, int(self.segmentation_tile_overlap)),
            tile_size // 3,
        )
        ys = self._tile_starts(h, tile_size, overlap)
        xs = self._tile_starts(w, tile_size, overlap)

        signal_acc = torch.zeros(
            (1, 1, h, w),
            dtype=torch.float32,
            device=image.device,
        )
        grid_acc = torch.zeros_like(signal_acc)
        text_acc = torch.zeros_like(signal_acc) if need_text else None
        weight_acc = torch.zeros_like(signal_acc)

        for y0 in ys:
            y1 = min(h, y0 + tile_size)
            for x0 in xs:
                x1 = min(w, x0 + tile_size)
                tile = image[:, :, y0:y1, x0:x1]
                logits = self.segmentation_model(tile)
                lse = torch.logsumexp(logits, dim=1, keepdim=True)
                signal_tile = torch.exp(
                    logits[:, [self.signal_class], :, :] - lse
                )
                grid_tile = torch.exp(
                    logits[:, [self.grid_class], :, :] - lse
                )
                text_tile = (
                    torch.exp(
                        logits[:, [self.text_background_class], :, :] - lse
                    )
                    if need_text
                    else None
                )

                weight = self._tile_blend_weight(
                    y1 - y0,
                    x1 - x0,
                    y0=y0,
                    y1=y1,
                    x0=x0,
                    x1=x1,
                    full_h=h,
                    full_w=w,
                    dtype=signal_tile.dtype,
                    device=signal_tile.device,
                )

                signal_acc[:, :, y0:y1, x0:x1].add_(signal_tile * weight)
                grid_acc[:, :, y0:y1, x0:x1].add_(grid_tile * weight)
                if need_text and text_acc is not None and text_tile is not None:
                    text_acc[:, :, y0:y1, x0:x1].add_(text_tile * weight)
                weight_acc[:, :, y0:y1, x0:x1].add_(weight)

                del tile
                del logits
                del lse
                del signal_tile
                del grid_tile
                del text_tile
                del weight

        denom = weight_acc.clamp_min_(1e-6)
        signal_acc.div_(denom)
        grid_acc.div_(denom)
        signal_prob = signal_acc
        grid_prob = grid_acc
        if need_text and text_acc is not None:
            text_acc.div_(denom)
            text_prob = text_acc
        else:
            text_prob = torch.empty(
                (0,),
                dtype=signal_prob.dtype,
                device=signal_prob.device,
            )

        del signal_acc
        del grid_acc
        del text_acc
        del weight_acc
        del denom

        signal_prob = self.process_sparse_prob(signal_prob)
        grid_prob = self.process_sparse_prob(grid_prob)
        if need_text:
            text_prob = self.process_sparse_prob(text_prob)

        return signal_prob, grid_prob, text_prob

    def min_max_normalize(self, image: Tensor) -> Tensor:
        return (image - image.min()) / (image.max() - image.min())

    def _load_signal_extractor(self) -> Any:
        signal_extractor_class = import_class_from_path(self.config.SIGNAL_EXTRACTOR.class_path)
        extractor: Any = signal_extractor_class(**self.config.SIGNAL_EXTRACTOR.KWARGS)
        return extractor

    def _load_perspective_detector(self) -> Any:
        perspective_detector_class = import_class_from_path(self.config.PERSPECTIVE_DETECTOR.class_path)
        perspective_detector: Any = perspective_detector_class(**self.config.PERSPECTIVE_DETECTOR.KWARGS)
        return perspective_detector

    def _load_segmentation_model(self) -> Any:
        segmentation_model_class = import_class_from_path(self.config.SEGMENTATION_MODEL.class_path)
        segmentation_model: Any = segmentation_model_class(**self.config.SEGMENTATION_MODEL.KWARGS)
        self._load_segmentation_model_weights(segmentation_model)
        return segmentation_model.eval()

    def _load_cropper(self) -> Any:
        cropper_class = import_class_from_path(self.config.CROPPER.class_path)
        cropper: Any = cropper_class(**self.config.CROPPER.KWARGS)
        return cropper

    def _load_pixel_size_finder(self) -> Any:
        pixel_size_finder_class = import_class_from_path(self.config.PIXEL_SIZE_FINDER.class_path)
        pixel_size_finder: Any = pixel_size_finder_class(**self.config.PIXEL_SIZE_FINDER.KWARGS)
        return pixel_size_finder

    def _load_dewarper(self) -> Any:
        # MEDCALC inference keeps dewarping disabled.
        if not self.apply_dewarping:
            return None
        dewarper_class = import_class_from_path(self.config.DEWARPER.class_path)
        dewarper: Any = dewarper_class(**self.config.DEWARPER.KWARGS)
        return dewarper

    def _load_layout_identifier(self) -> Any:
        layouts = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.config_path, "r"))
        unet_cfg = yaml.safe_load(open(self.config.LAYOUT_IDENTIFIER.unet_config_path, "r"))
        unet_class = import_class_from_path(unet_cfg["MODEL"]["class_path"])
        unet: torch.nn.Module = unet_class(**unet_cfg["MODEL"]["KWARGS"])
        checkpoint = torch.load(self.config.LAYOUT_IDENTIFIER.unet_weight_path, map_location=self.device)
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        unet.load_state_dict(checkpoint)
        unet.eval()

        identifier_class = import_class_from_path(self.config.LAYOUT_IDENTIFIER.class_path)
        identifier: Any = identifier_class(
            layouts=layouts,
            unet=unet,
            **self.config.LAYOUT_IDENTIFIER.KWARGS,
        )
        return identifier

    def _load_segmentation_model_weights(self, segmentation_model: torch.nn.Module) -> None:
        """Loads weights for segmentation model.

        Args:
            segmentation_model: The model to load weights into.
        """
        checkpoint = torch.load(self.config.SEGMENTATION_MODEL.weight_path, weights_only=True, map_location=self.device)
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
        segmentation_model.load_state_dict(checkpoint)

    def _check_image_dimensions(self, image: Tensor) -> None:
        """Checks input image dimensions.

        Args:
            image: Image tensor.

        Raises:
            NotImplementedError: If batch or channel dims are incorrect.
        """
        if image.dim() != 4:
            raise NotImplementedError(f"Expected 4 dimensions, got tensor with {image.dim()} dimensions")
        if image.shape[0] != 1:
            raise NotImplementedError(f"Batch processing not supported, got tensor with shape {image.shape}")
        if image.shape[1] != 3:
            raise NotImplementedError(f"Expected 3 channels, got tensor with shape {image.shape}")
