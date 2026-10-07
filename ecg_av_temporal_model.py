"""Optional learned AV research branch. Importing this does not load PyTorch."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import find_peaks, resample_poly
from fractions import Fraction

from ecg_av_event_graph import TOKEN_FEATURES, build_av_event_graph
from ecg_av_training_data import CLASSES, FS

VERSION = "MEDCALC_R28_AV_TEMPORAL_V1"
MAX_EVENTS = 256
SUPPORTED_EVENT_CHANNELS = (("P", "QRS"), ("P", "QRS", "T"))


def create_networks(detector_channels: int = 2):
    import torch
    from torch import nn

    if detector_channels not in (2, 3):
        raise ValueError("AV detector supports only P/QRS or P/QRS/T channels")

    class EventDetector(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Conv1d(1, 24, 9, padding=4), nn.GELU()]
            for dilation in (1, 2, 4, 8, 16, 32):
                layers += [nn.Conv1d(24, 24, 7, padding=3 * dilation,
                                     dilation=dilation), nn.GELU()]
            self.layers = nn.Sequential(*layers, nn.Conv1d(24, detector_channels, 1))

        def forward(self, signal):
            return self.layers(signal)

    class TemporalClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(len(TOKEN_FEATURES), 32)
            layer = nn.TransformerEncoderLayer(32, 4, 64, dropout=.1,
                                               batch_first=True, norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, len(CLASSES)))

        def forward(self, tokens, padding_mask):
            x = self.encoder(self.project(tokens), src_key_padding_mask=padding_mask)
            usable = (~padding_mask).unsqueeze(-1)
            pooled = (x * usable).sum(1) / usable.sum(1).clamp(min=1)
            return self.head(pooled)

    return EventDetector(), TemporalClassifier()


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    x = x - np.median(x)
    return (x / max(float(np.quantile(abs(x), .995)), .1)).astype(np.float32)


def _peak_events(score: np.ndarray, fs: int, distance_s: float):
    indices, properties = find_peaks(score, height=.5, prominence=.05,
                                     distance=int(distance_s * fs))
    heights = properties.get("peak_heights", score[indices])
    return [{"time_s": float(i / fs), "confidence": float(h)}
            for i, h in zip(indices, heights)]


def events_from_probabilities(probabilities: np.ndarray, fs: int = FS, include_t: bool = False):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] not in (2, 3):
        raise ValueError("Expected P/QRS or P/QRS/T probability channels")
    if probabilities.shape[1] == 0 or not np.all(np.isfinite(probabilities)):
        raise ValueError("Event probabilities must be finite and non-empty")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Event probabilities must lie in [0,1]")

    # T is an auxiliary task only. It must never hard-veto a P candidate because
    # true atrial activity may overlap a T wave in AV block or tachycardia.
    p = _peak_events(probabilities[0], fs, .18)
    r = _peak_events(probabilities[1], fs, .22)
    t = _peak_events(probabilities[2], fs, .18) if probabilities.shape[0] == 3 else []
    return (p, r, t) if include_t else (p, r)


def token_batch(graphs: list[dict]):
    import torch
    size = max(1, max(len(g["tokens"]) for g in graphs))
    if size > MAX_EVENTS:
        raise ValueError("Event budget exceeded; do not truncate conduction sequences")
    array = np.zeros((len(graphs), size, len(TOKEN_FEATURES)), dtype=np.float32)
    mask = np.ones((len(graphs), size), dtype=bool)
    for i, graph in enumerate(graphs):
        count = len(graph["tokens"])
        if count:
            array[i, :count] = graph["tokens"]
            mask[i, :count] = False
        else:
            # Empty placeholder avoids all-masked attention; inference still abstains.
            mask[i, 0] = False
    return torch.from_numpy(array), torch.from_numpy(mask)


class AVResearchModel:
    def __init__(self, checkpoint: str | Path):
        import torch
        path = Path(checkpoint)
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        metadata = bundle["metadata"]
        event_channels = tuple(metadata.get("event_channels", ("P", "QRS")))
        if (metadata.get("version") != VERSION or metadata.get("classes") != list(CLASSES)
                or metadata.get("fs") != FS or metadata.get("token_features") != list(TOKEN_FEATURES)
                or metadata.get("diagnostic_claim_allowed") is not False
                or event_channels not in SUPPORTED_EVENT_CHANNELS):
            raise ValueError("Incompatible or unguarded research checkpoint")
        self.detector, self.classifier = create_networks(len(event_channels))
        self.detector.load_state_dict(bundle["detector"], strict=True)
        self.classifier.load_state_dict(bundle["classifier"], strict=True)
        self.detector.eval()
        self.classifier.eval()
        self.metadata = metadata
        self.event_channels = event_channels
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    def analyze_signal(self, signal: np.ndarray, fs: int) -> dict[str, Any]:
        import torch
        x = np.asarray(signal, dtype=np.float32)
        if x.ndim != 1 or not np.all(np.isfinite(x)) or fs < 100:
            raise ValueError("Expected one finite observed lead sampled at >=100 Hz")
        if len(x) / fs < 6 or len(x) / fs > 60:
            raise ValueError("Research window must contain 6-60 contiguous seconds")
        ratio = Fraction(FS, int(fs))
        x = resample_poly(x, ratio.numerator, ratio.denominator)
        with torch.inference_mode():
            logits = self.detector(torch.from_numpy(normalize_signal(x))[None, None])
            probabilities = logits.sigmoid()[0].numpy()
            p, r = events_from_probabilities(probabilities)
            graph = build_av_event_graph(p, r, len(x) / FS)
            if len(graph["tokens"]) > MAX_EVENTS:
                return self._result(graph, {}, "EXCESSIVE_EVENT_CANDIDATES")
            tokens, mask = token_batch([graph])
            scores = self.classifier(tokens, mask).softmax(-1)[0].numpy()
        reason = "UNVALIDATED_RESEARCH_MODEL"
        if len(p) < 4 or len(r) < 3:
            reason = "INSUFFICIENT_OBSERVED_EVENT_CANDIDATES"
        return self._result(graph, {k: float(v) for k, v in zip(CLASSES, scores)}, reason)

    def _result(self, graph, probabilities, reason):
        return {"version": VERSION, "status": "RESEARCH_ONLY", "abstain": True,
                "reason": reason, "graph": graph,
                "probabilities": probabilities,
                "probability_semantics": "UNCALIBRATED_RESEARCH_SOFTMAX",
                "model_sha256": self.sha256, "training_source": self.metadata["training_source"],
                "event_channels": list(self.event_channels),
                "diagnostic_claim_allowed": False, "clinical_fusion_allowed": False}


def analyze_av_research(canonical: dict, model: AVResearchModel) -> dict:
    """Use one continuous physical lead, without aligning sequential paper columns."""
    candidates = []
    for priority, name in enumerate(("II", "V1", "aVF", "I", "III", "aVL", "V5", "V6")):
        item = (canonical.get("leads") or {}).get(name) or {}
        try:
            x = np.asarray(item.get("signal_mv", []), dtype=float)
            quality = np.asarray(item.get("quality_mask", []))
            fs = int(item.get("fs") or canonical.get("fs") or 0)
        except (ValueError, TypeError):
            continue
        if x.ndim != 1 or quality.shape != x.shape or fs < 100:
            continue
        if (item.get("tiled") or item.get("repeated") or
                float(item.get("repeated_fraction") or 0) > 0):
            continue
        observed = np.isfinite(x) & (quality == 2)
        transitions = np.diff(np.r_[False, observed, False].astype(int))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            duration = (end - start) / fs
            if 6 <= duration <= 60:
                candidates.append((duration, -priority, name, int(start), int(end), fs))
    if not candidates:
        return {"version": VERSION, "status": "NOT_EVALUABLE", "abstain": True,
                "reason": "NO_CONTIGUOUS_OBSERVED_NATIVE_LEAD_GE_6S",
                "diagnostic_claim_allowed": False, "clinical_fusion_allowed": False}
    _, _, lead, start, end, fs = max(candidates)
    item = canonical["leads"][lead]
    result = model.analyze_signal(np.asarray(item["signal_mv"])[start:end], fs)
    result["observed_window"] = {"lead": lead, "start_sample": start, "end_sample": end,
                                 "fs": fs, "time_origin_ms": item.get("time_origin_ms", 0),
                                 "event_time_reference": "SECONDS_FROM_WINDOW_START",
                                 "crosslead_alignment_assumed": False}
    return result
