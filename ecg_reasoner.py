from __future__ import annotations

from typing import Any, Dict


REASONER_VERSION = "MEDCALC_ECG_SPECIALIST_REASONER_V3_HIGH_SENSITIVITY"


def _candidate(code: str, confidence: float, basis: list[str], layer: str) -> Dict[str, Any]:
    return {
        "code": code,
        "confidence": round(float(max(0.0, min(1.0, confidence))), 6),
        "basis": basis,
        "layer": layer,
    }


def _global_value(feature_graph: Dict[str, Any], key: str) -> float | None:
    try:
        value = ((feature_graph.get("global") or {}).get(key) or {}).get("value")
        return float(value) if value is not None else None
    except Exception:
        return None


def _domain_ok(domain_gates: Dict[str, Any], domain: str) -> bool:
    if not domain_gates:
        return True
    return bool((((domain_gates.get("domains") or {}).get(domain) or {}).get("eligible")))


def _fused_map(evidence_fusion: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("code") or ""): dict(row)
        for row in (evidence_fusion.get("findings") or [])
        if str(row.get("code") or "")
    }


def _legacy_reason_ecg(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Backward-compatible path for direct callers that do not supply V3 layers."""
    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    wct = specialists.get("wide_complex_tachycardia") or {}
    ectopy = specialists.get("ectopy") or {}
    av = specialists.get("av_conduction") or {}
    preexcitation = specialists.get("preexcitation") or {}
    rhythm = feature_graph.get("rhythm") or {}

    candidates = []
    if bool(wct.get("wide_complex_tachycardia")) and str(wct.get("classification") or "") == "VT_COMPATIBLE":
        candidates.append(_candidate(
            "VT_COMPATIBLE",
            float(wct.get("confidence") or 0.0),
            ["WCT_GATE_MET", "DIRECT_WCT_MORPHOLOGY"],
            "VENTRICULAR_ORIGIN",
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
            ["NATIVE_ATRIAL_ANALYZER", "RR_P_QRS_AND_ECTOPY_CONTEXT"],
            "ATRIAL_MECHANISM",
        ))
    elif bool(atrial.get("sinus_compatible")):
        hr = _global_value(feature_graph, "heart_rate_bpm")
        base_conf = min(float(rhythm.get("confidence") or 0.0), 0.95)
        if hr is not None and hr < 60.0:
            code = "SINUS_BRADYCARDIA_COMPATIBLE"
            basis = ["SINUS_MECHANISM_ESTABLISHED", "VENTRICULAR_RATE_LT_60"]
        elif hr is not None and hr > 100.0:
            code = "SINUS_TACHYCARDIA_COMPATIBLE"
            basis = ["SINUS_MECHANISM_ESTABLISHED", "VENTRICULAR_RATE_GT_100"]
        else:
            code = "SINUS_COMPATIBLE"
            basis = ["REPRODUCIBLE_POSITIVE_P_IN_II", "P_QRS_COUPLING"]
        candidates.append(_candidate(code, base_conf, basis, "ATRIAL_MECHANISM"))

    if candidates:
        vt = [row for row in candidates if row["code"] == "VT_COMPATIBLE"]
        primary = vt[0] if vt else max(candidates, key=lambda row: row["confidence"])
    else:
        primary = _candidate(
            "RHYTHM_MECHANISM_UNDETERMINED",
            0.0,
            ["INSUFFICIENT_SPECIALIST_SEPARATION"],
            "RHYTHM",
        )

    blocking_codes = {
        str(row.get("code") or "")
        for row in (consistency.get("conflicts") or [])
        if str(row.get("severity") or "") == "BLOCKING"
    }

    conduction_findings = list(crosslead_conduction.get("findings") or [])
    if (
        "qrs_ms" in (consistency.get("remeasure_targets") or [])
        or blocking_codes & {
            "RBBB_LBBB_MUTUAL_CONFLICT",
            "COMPLETE_BBB_WITH_QRS_LT_120_CONFLICT",
            "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT",
        }
    ):
        conduction_findings = []

    av_finding = None
    av_cls = str(av.get("classification") or "")
    if (
        av.get("evaluable")
        and av_cls not in {"", "NO_HIGH_GRADE_AV_BLOCK_ESTABLISHED", "AV_CONDUCTION_NOT_EVALUABLE"}
        and not blocking_codes & {
            "FIRST_DEGREE_AV_DELAY_WITHOUT_PR_GT_200_OR_1_TO_1",
            "AV_BLOCK_WITHOUT_NONCONDUCTED_P_CONFLICT",
            "COMPLETE_AV_BLOCK_WITHOUT_AV_DISSOCIATION_SUPPORT",
        }
    ):
        av_finding = {
            "code": av_cls,
            "confidence": float(av.get("confidence") or 0.0),
            "basis": list(av.get("basis") or []),
        }

    preexcitation_finding = None
    if str(preexcitation.get("classification") or "") == "VENTRICULAR_PREEXCITATION_COMPATIBLE":
        preexcitation_finding = {
            "code": "VENTRICULAR_PREEXCITATION_COMPATIBLE",
            "confidence": float(preexcitation.get("confidence") or 0.0),
            "basis": ["SHORT_PR", "QRS_PROLONGATION", "MULTILEAD_DELTA_SLUR"],
        }
        for row in conduction_findings:
            if str(row.get("code") or "").startswith(("RBBB_", "LBBB_")):
                row["basis"] = sorted(set(
                    list(row.get("basis") or [])
                    + ["PREEXCITATION_CONFOUNDING_WARNING"]
                ))
                row["confounded_by_preexcitation"] = True

    ectopy_findings = []
    if int(ectopy.get("pvc_compatible_n") or 0) > 0:
        ectopy_findings.append({"code": "PVC_COMPATIBLE", "count": int(ectopy.get("pvc_compatible_n") or 0)})
    if int(ectopy.get("pac_or_narrow_premature_n") or 0) > 0:
        ectopy_findings.append({
            "code": "PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE",
            "count": int(ectopy.get("pac_or_narrow_premature_n") or 0),
        })

    publication_allowed = bool(consistency.get("publication_allowed"))
    primary = {
        **primary,
        "publish_as_established": bool(publication_allowed and primary["confidence"] >= 0.70),
        "suppressed_by_consistency_engine": not publication_allowed,
    }

    findings = []
    if primary.get("code") != "RHYTHM_MECHANISM_UNDETERMINED":
        findings.append({
            "domain": "RHYTHM",
            "code": primary.get("code"),
            "confidence": primary.get("confidence"),
            "publishable": bool(primary.get("publish_as_established")),
            "basis": list(primary.get("basis") or []),
        })
    for row in conduction_findings:
        findings.append({
            "domain": "CONDUCTION",
            "code": row.get("code"),
            "confidence": row.get("confidence"),
            "publishable": publication_allowed,
            "basis": list(row.get("basis") or []),
        })
    if av_finding:
        findings.append({"domain": "AV_CONDUCTION", **av_finding, "publishable": publication_allowed})
    if preexcitation_finding:
        findings.append({"domain": "PREEXCITATION", **preexcitation_finding, "publishable": publication_allowed})
    for row in ectopy_findings:
        findings.append({"domain": "ECTOPY", **row, "publishable": publication_allowed})

    abstentions = []
    if consistency.get("blocking_conflict"):
        abstentions.append({
            "domain": "GLOBAL",
            "reason": "BLOCKING_CONSISTENCY_CONFLICT",
            "conflicts": sorted(blocking_codes),
        })

    return {
        "version": REASONER_VERSION,
        "primary_rhythm": primary,
        "rhythm_candidates": candidates,
        "conduction_findings": conduction_findings,
        "av_conduction_finding": av_finding,
        "preexcitation_finding": preexcitation_finding,
        "ectopy_findings": ectopy_findings,
        "consistency_status": consistency.get("status"),
        "publication_allowed": publication_allowed,
        "publication_model": "LEGACY_GLOBAL_COMPATIBILITY",
        "domain_gates": {},
        "evidence_fusion": {},
        "measurement_mutation_allowed": False,
        "diagnostic_summary": {
            "authoritative": True,
            "findings": findings,
            "abstentions": abstentions,
            "publication_allowed": publication_allowed,
            "publication_model": "LEGACY_GLOBAL_COMPATIBILITY",
        },
        "report_authority": "SPECIALIST_REASONER_STRUCTURED_OUTPUT",
        "llm_role": "REPORT_WORDING_ONLY_NOT_CLINICAL_ARBITRATION",
        "source": "SPECIALIST_EVIDENCE_GRAPH_LEGACY_COMPATIBILITY",
    }


def reason_ecg(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    consistency: Dict[str, Any],
    *,
    domain_gates: Dict[str, Any] | None = None,
    evidence_fusion: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evidence-constrained V3 synthesis with domain-specific abstention.

    Numeric measurements remain immutable. A contradiction in one evidence
    domain can suppress only diagnoses that depend on that domain. High-recall
    candidates become publishable only after multisource evidence fusion.
    """
    if domain_gates is None and evidence_fusion is None:
        return _legacy_reason_ecg(feature_graph, crosslead_conduction, consistency)

    domain_gates = domain_gates or {}
    evidence_fusion = evidence_fusion or {}
    fused = _fused_map(evidence_fusion)

    specialists = feature_graph.get("specialist_evidence") or {}
    atrial = specialists.get("atrial_activity") or {}
    atrial_mech = specialists.get("atrial_mechanism") or {}
    wct = specialists.get("wide_complex_tachycardia") or {}
    ectopy = specialists.get("ectopy") or {}
    rhythm = feature_graph.get("rhythm") or {}

    rhythm_candidates: list[Dict[str, Any]] = []

    if bool(wct.get("wide_complex_tachycardia")) and _domain_ok(domain_gates, "RHYTHM"):
        cls = str(wct.get("classification") or "")
        conf = float(wct.get("confidence") or 0.0)
        if cls == "VT_COMPATIBLE":
            rhythm_candidates.append(_candidate(
                "VT_COMPATIBLE", conf,
                ["WCT_GATE_MET", "DIRECT_WCT_MORPHOLOGY"], "VENTRICULAR_ORIGIN"
            ))

    for code in (
        "AF_COMPATIBLE",
        "FLUTTER_OR_AT_COMPATIBLE",
        "SINUS_BRADYCARDIA_COMPATIBLE",
        "SINUS_TACHYCARDIA_COMPATIBLE",
    ):
        row = fused.get(code) or {}
        if bool(row.get("publishable")):
            rhythm_candidates.append(_candidate(
                code,
                float(row.get("score") or 0.0),
                list(row.get("evidence") or []),
                "EVIDENCE_FUSION",
            ))

    mechanism = str(atrial_mech.get("mechanism") or "")
    atrial_conf = float(atrial_mech.get("confidence") or 0.0)
    if (
        mechanism == "OTHER_SVT_COMPATIBLE"
        and atrial_conf >= 0.70
        and _domain_ok(domain_gates, "RHYTHM")
    ):
        rhythm_candidates.append(_candidate(
            mechanism,
            atrial_conf,
            ["NATIVE_ATRIAL_ANALYZER", "REGULAR_TACHYCARDIA_CONTEXT"],
            "ATRIAL_MECHANISM",
        ))

    # Normal sinus rhythm remains a specialist conclusion rather than a
    # high-recall candidate because it is a negative/normal diagnosis.
    if (
        not rhythm_candidates
        and bool(atrial.get("sinus_compatible"))
        and _domain_ok(domain_gates, "RHYTHM")
    ):
        hr = _global_value(feature_graph, "heart_rate_bpm")
        if hr is None or 60.0 <= hr <= 100.0:
            rhythm_candidates.append(_candidate(
                "SINUS_COMPATIBLE",
                min(float(rhythm.get("confidence") or 0.0), 0.95),
                ["REPRODUCIBLE_POSITIVE_P_IN_II", "P_QRS_COUPLING"],
                "ATRIAL_MECHANISM",
            ))

    if rhythm_candidates:
        vt = [c for c in rhythm_candidates if c["code"] == "VT_COMPATIBLE"]
        primary = vt[0] if vt else max(rhythm_candidates, key=lambda c: c["confidence"])
        primary = {
            **primary,
            "publish_as_established": True,
            "suppressed_by_consistency_engine": False,
        }
    else:
        primary = {
            **_candidate(
                "RHYTHM_MECHANISM_UNDETERMINED",
                0.0,
                ["NO_PUBLISHABLE_DOMAIN_SPECIFIC_RHYTHM_EVIDENCE"],
                "RHYTHM",
            ),
            "publish_as_established": False,
            "suppressed_by_consistency_engine": not _domain_ok(domain_gates, "RHYTHM"),
        }

    conduction_findings: list[Dict[str, Any]] = []
    for code in (
        "RBBB_MORPHOLOGY_COMPATIBLE",
        "LBBB_MORPHOLOGY_COMPATIBLE",
        "LAFB_COMPATIBLE",
        "LPFB_COMPATIBLE",
    ):
        row = fused.get(code) or {}
        if bool(row.get("publishable")):
            conduction_findings.append({
                "code": code,
                "confidence": float(row.get("score") or 0.0),
                "basis": list(row.get("evidence") or []),
                "fusion_state": row.get("fusion_state"),
            })

    # Preserve incomplete bundle findings from the specialist engine, but only
    # when the bundle-branch domain itself is eligible.
    if _domain_ok(domain_gates, "BUNDLE_BRANCH"):
        existing_codes = {str(x.get("code") or "") for x in conduction_findings}
        for row in crosslead_conduction.get("findings") or []:
            code = str(row.get("code") or "")
            if code in {
                "INCOMPLETE_RBBB_MORPHOLOGY_COMPATIBLE",
                "INCOMPLETE_LBBB_MORPHOLOGY_COMPATIBLE",
            } and code not in existing_codes:
                conduction_findings.append(dict(row))

    av_rows = [
        row for row in (evidence_fusion.get("publishable_findings") or [])
        if str(row.get("domain") or "") == "AV_CONDUCTION"
    ]
    av_finding = None
    if av_rows:
        av_row = max(av_rows, key=lambda x: float(x.get("score") or 0.0))
        av_finding = {
            "code": str(av_row.get("code") or ""),
            "confidence": float(av_row.get("score") or 0.0),
            "basis": list(av_row.get("evidence") or []),
            "fusion_state": av_row.get("fusion_state"),
        }

    pre_row = fused.get("VENTRICULAR_PREEXCITATION_COMPATIBLE") or {}
    preexcitation_finding = None
    if bool(pre_row.get("publishable")):
        preexcitation_finding = {
            "code": "VENTRICULAR_PREEXCITATION_COMPATIBLE",
            "confidence": float(pre_row.get("score") or 0.0),
            "basis": list(pre_row.get("evidence") or []),
            "fusion_state": pre_row.get("fusion_state"),
        }
        # Preexcitation may mimic a bundle-branch pattern, but this is a
        # consistency WARNING, not a blocking contradiction. If complete BBB
        # independently passed its own domain gate and evidence fusion, preserve
        # the morphology-compatible finding and mark it explicitly as confounded
        # rather than deleting it silently.
        for row in conduction_findings:
            if str(row.get("code") or "").startswith(("RBBB_", "LBBB_")):
                row["basis"] = sorted(set(
                    list(row.get("basis") or [])
                    + ["PREEXCITATION_CONFOUNDING_WARNING"]
                ))
                row["confounded_by_preexcitation"] = True

    ectopy_findings = []
    if _domain_ok(domain_gates, "ECTOPY"):
        if int(ectopy.get("pvc_compatible_n") or 0) > 0:
            ectopy_findings.append({
                "code": "PVC_COMPATIBLE",
                "count": int(ectopy.get("pvc_compatible_n") or 0),
            })
        if int(ectopy.get("pac_or_narrow_premature_n") or 0) > 0:
            ectopy_findings.append({
                "code": "PAC_OR_NARROW_PREMATURE_BEAT_COMPATIBLE",
                "count": int(ectopy.get("pac_or_narrow_premature_n") or 0),
            })

    final_findings = []
    if primary.get("code") != "RHYTHM_MECHANISM_UNDETERMINED":
        final_findings.append({
            "domain": "RHYTHM",
            "code": primary.get("code"),
            "confidence": primary.get("confidence"),
            "publishable": True,
            "basis": list(primary.get("basis") or []),
        })
    for row in conduction_findings:
        final_findings.append({
            "domain": (
                "FASCICULAR"
                if str(row.get("code") or "") in {"LAFB_COMPATIBLE", "LPFB_COMPATIBLE"}
                else "BUNDLE_BRANCH"
            ),
            "code": row.get("code"),
            "confidence": row.get("confidence"),
            "publishable": True,
            "basis": list(row.get("basis") or []),
            "confounded_by_preexcitation": bool(
                row.get("confounded_by_preexcitation")
            ),
        })
    if av_finding:
        final_findings.append({
            "domain": "AV_CONDUCTION",
            **av_finding,
            "publishable": True,
        })
    if preexcitation_finding:
        final_findings.append({
            "domain": "PREEXCITATION",
            **preexcitation_finding,
            "publishable": True,
        })
    for row in ectopy_findings:
        final_findings.append({
            "domain": "ECTOPY",
            **row,
            "publishable": True,
        })

    abstentions = []
    for domain, gate in (domain_gates.get("domains") or {}).items():
        if bool(gate.get("eligible")):
            continue
        abstentions.append({
            "domain": domain,
            "reason": "DOMAIN_EVIDENCE_NOT_PUBLISHABLE",
            "conflicts": list(gate.get("blocked_by_conflicts") or []),
            "remeasure_targets": list(gate.get("remeasure_targets") or []),
        })

    domains = domain_gates.get("domains") or {}
    any_domain_publishable = any(
        bool(row.get("eligible")) for row in domains.values()
    ) if domains else not bool(consistency.get("blocking_conflict"))

    return {
        "version": REASONER_VERSION,
        "primary_rhythm": primary,
        "rhythm_candidates": rhythm_candidates,
        "conduction_findings": conduction_findings,
        "av_conduction_finding": av_finding,
        "preexcitation_finding": preexcitation_finding,
        "ectopy_findings": ectopy_findings,
        "consistency_status": consistency.get("status"),
        "publication_allowed": any_domain_publishable,
        "publication_model": "DOMAIN_SPECIFIC",
        "domain_gates": domain_gates,
        "evidence_fusion": evidence_fusion,
        "measurement_mutation_allowed": False,
        "diagnostic_summary": {
            "authoritative": True,
            "findings": final_findings,
            "abstentions": abstentions,
            "publication_allowed": any_domain_publishable,
            "publication_model": "DOMAIN_SPECIFIC",
        },
        "report_authority": "SPECIALIST_REASONER_STRUCTURED_OUTPUT",
        "llm_role": "REPORT_WORDING_ONLY_NOT_CLINICAL_ARBITRATION",
        "source": "SPECIALIST_EVIDENCE_GRAPH_PLUS_HIGH_RECALL_FUSION",
    }
