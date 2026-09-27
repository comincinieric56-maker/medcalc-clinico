from __future__ import annotations

from typing import Any, Dict


DOMAIN_GATING_VERSION = "MEDCALC_ECG_DOMAIN_GATING_V1"

DOMAINS = (
    "RHYTHM",
    "RATE",
    "BUNDLE_BRANCH",
    "FASCICULAR",
    "AV_CONDUCTION",
    "PREEXCITATION",
    "ECTOPY",
    "REPOLARIZATION",
    "QT",
)

CONFLICT_DOMAINS = {
    "AF_VS_REPRODUCIBLE_P_QRS_CONFLICT": {"RHYTHM"},
    "SINUS_WITHOUT_REPRODUCIBLE_P_CONFLICT": {"RHYTHM"},
    "WCT_ACTIVATED_OUTSIDE_GATE": {"RHYTHM", "BUNDLE_BRANCH"},
    "RHYTHM_SOURCE_FAILS_SIGNAL_INTEGRITY_GATE": {"RHYTHM", "ECTOPY"},
    "RBBB_LBBB_MUTUAL_CONFLICT": {"BUNDLE_BRANCH"},
    "COMPLETE_BBB_WITH_QRS_LT_120_CONFLICT": {"BUNDLE_BRANCH"},
    "CONDUCTION_DEPENDS_ON_DISCORDANT_QRS_MEASUREMENT": {"BUNDLE_BRANCH"},
    "LAFB_WITHOUT_REQUIRED_AXIS_CONFLICT": {"FASCICULAR"},
    "FIRST_DEGREE_AV_DELAY_WITHOUT_PR_GT_200_OR_1_TO_1": {"AV_CONDUCTION"},
    "AV_BLOCK_WITHOUT_NONCONDUCTED_P_CONFLICT": {"AV_CONDUCTION"},
    "COMPLETE_AV_BLOCK_WITHOUT_AV_DISSOCIATION_SUPPORT": {"AV_CONDUCTION"},
}

REMEASURE_DOMAINS = {
    "r_peaks": {"RHYTHM", "RATE", "ECTOPY", "AV_CONDUCTION"},
    "qrs_ms": {"BUNDLE_BRANCH", "PREEXCITATION", "QT"},
    "pr_ms": {"AV_CONDUCTION", "PREEXCITATION"},
    "p_duration_ms": {"RHYTHM", "AV_CONDUCTION"},
    "qt_ms": {"QT"},
}


def _rate_consensus_usable(feature_graph: Dict[str, Any]) -> bool:
    rhythm = feature_graph.get("rhythm") or {}
    consensus = rhythm.get("rate_consensus") or {}
    try:
        confidence = float(consensus.get("confidence") or 0.0)
        source_n = int(consensus.get("source_n") or 0)
        hr = float(consensus.get("heart_rate_bpm"))
    except Exception:
        return False
    return bool(consensus.get("evaluable") and source_n >= 3 and confidence >= 0.50 and 25.0 <= hr <= 250.0)


def build_domain_gates(
    feature_graph: Dict[str, Any],
    crosslead_conduction: Dict[str, Any],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert global consistency output into diagnosis-domain eligibility.

    A conflict may suppress only diagnoses that depend on the affected evidence.
    This prevents, for example, a discordant QRS duration from suppressing an
    otherwise well-supported atrial rhythm diagnosis.
    """
    conflicts = list(consistency.get("conflicts") or [])
    remeasure = set(consistency.get("remeasure_targets") or [])
    gates: Dict[str, Dict[str, Any]] = {}

    for domain in DOMAINS:
        blocking = []
        for row in conflicts:
            if str(row.get("severity") or "") != "BLOCKING":
                continue
            code = str(row.get("code") or "")
            if domain in CONFLICT_DOMAINS.get(code, set()):
                blocking.append(code)

        relevant_remeasure = sorted(
            target for target in remeasure
            if domain in REMEASURE_DOMAINS.get(str(target), set())
        )

        # Robust multilead rate consensus can rescue absolute rate even when
        # one R detector disagrees. It does NOT rescue RR-sequence diagnoses.
        if domain == "RATE" and "r_peaks" in relevant_remeasure and _rate_consensus_usable(feature_graph):
            relevant_remeasure = [x for x in relevant_remeasure if x != "r_peaks"]

        eligible = not blocking and not relevant_remeasure
        gates[domain] = {
            "eligible": eligible,
            "blocked_by_conflicts": sorted(set(blocking)),
            "remeasure_targets": relevant_remeasure,
            "degraded": bool(blocking or relevant_remeasure),
        }

    return {
        "version": DOMAIN_GATING_VERSION,
        "policy": "ABSTENTION_IS_DOMAIN_SPECIFIC; UNRELATED_CONFLICTS_DO_NOT_SUPPRESS_OTHER_DOMAINS",
        "domains": gates,
        "global_blocking_conflict_present": bool(consistency.get("blocking_conflict")),
        "global_remeasure_present": bool(consistency.get("remeasure_required")),
        "rate_consensus_rescue_active": _rate_consensus_usable(feature_graph),
    }


def domain_eligible(gates: Dict[str, Any], domain: str) -> bool:
    row = ((gates.get("domains") or {}).get(domain) or {})
    return bool(row.get("eligible"))
