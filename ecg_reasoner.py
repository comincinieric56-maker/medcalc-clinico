from __future__ import annotations

from typing import Any, Dict


REASONER_VERSION = "MEDCALC_ECG_SPECIALIST_REASONER_V1"


def _candidate(code: str, confidence: float, basis: list[str], layer: str) -> Dict[str, Any]:
    return {
        "code": code,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
        "basis": basis,
        "layer": layer,
    }


def reason_ecg(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Evidence-constrained ECG reasoning.

    The reasoner may select among already-derived specialist hypotheses but
    cannot mutate numeric measurements or invent a diagnosis unsupported by
    the specialist layers.
    """
    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    wct = specialists.get("wide_complex_tachycardia") or {}
    rhythm = feature_graph.get("rhythm") or {}

    candidates: list[Dict[str, Any]] = []

    if bool(wct.get("wide_complex_tachycardia")):
        cls = str(wct.get("classification") or "")
        conf = float(wct.get("confidence") or 0.0)
        if cls == "VT_COMPATIBLE":
            candidates.append(_candidate(
                "VT_COMPATIBLE", conf,
                ["WCT_GATE_MET", "DIRECT_WCT_MORPHOLOGY"], "VENTRICULAR_ORIGIN"
            ))

    mechanism = str(atrial_mech.get("mechanism") or "")
    atrial_conf = float(atrial_mech.get("confidence") or 0.0)
    if mechanism in {
        "AF_COMPATIBLE",
        "FLUTTER_OR_AT_COMPATIBLE",
        "OTHER_SVT_COMPATIBLE",
    }:
        candidates.append(_candidate(
            mechanism,
            atrial_conf,
            ["NATIVE_ATRIAL_ANALYZER", "RR_AND_P_QRS_CONTEXT"],
            "ATRIAL_MECHANISM",
        ))
    elif bool(atrial.get("sinus_compatible")):
        candidates.append(_candidate(
            "SINUS_COMPATIBLE",
            min(
                float(rhythm.get("confidence") or 0.0),
                0.95,
            ),
            ["REPRODUCIBLE_P_QRS_COUPLING", "POSITIVE_P_IN_II"],
            "ATRIAL_MECHANISM",
        ))

    if candidates:
        vt = [c for c in candidates if c["code"] == "VT_COMPATIBLE"]
        primary = vt[0] if vt else max(candidates, key=lambda c: c["confidence"])
    else:
        primary = _candidate(
            "RHYTHM_MECHANISM_UNDETERMINED",
            0.0,
            ["INSUFFICIENT_SPECIALIST_SEPARATION"],
            "RHYTHM",
        )

    conduction_findings = list(crosslead_conduction.get("findings") or [])
    if "qrs_ms" in (consistency.get("remeasure_targets") or []):
        conduction_findings = []

    publication_allowed = bool(consistency.get("publication_allowed"))
    if not publication_allowed:
        primary = {
            **primary,
            "publish_as_established": False,
            "suppressed_by_consistency_engine": True,
        }
    else:
        primary = {
            **primary,
            "publish_as_established": bool(primary["confidence"] >= 0.70),
            "suppressed_by_consistency_engine": False,
        }

    return {
        "version": REASONER_VERSION,
        "primary_rhythm": primary,
        "rhythm_candidates": candidates,
        "conduction_findings": conduction_findings,
        "consistency_status": consistency.get("status"),
        "publication_allowed": publication_allowed,
        "measurement_mutation_allowed": False,
        "llm_role": "REPORT_WORDING_ONLY_NOT_CLINICAL_ARBITRATION",
        "source": "SPECIALIST_EVIDENCE_GRAPH",
    }
