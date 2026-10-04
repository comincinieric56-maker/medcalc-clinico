"""Attach AV research evidence without changing the clinical engine entry point."""
from __future__ import annotations

from typing import Any


def analyze_ecg_with_av_research(canonical_ecg: dict, *, av_research_model: Any) -> dict:
    from ecg_signal_measurements import analyze_canonical_ecg
    result = analyze_canonical_ecg(canonical_ecg)
    try:
        from ecg_av_temporal_model import analyze_av_research
        result["av_research"] = analyze_av_research(canonical_ecg, av_research_model)
    except Exception as exc:
        result["av_research"] = {
            "status": "RESEARCH_ERROR", "abstain": True,
            "reason": type(exc).__name__, "diagnostic_claim_allowed": False,
            "clinical_fusion_allowed": False,
        }
    return result
