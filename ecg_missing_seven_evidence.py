"""MEDCALC ECG / G26. Read-only, evidence-only inventory for seven unsupported R27 categories.

Not a classifier. Does not supply probabilities, logits, labels, thresholds, or
clinical findings. Never feeds the reasoner or an existing module gate.
"""
from __future__ import annotations

from typing import Any, Mapping

VERSION = 'MEDCALC_ECG_G26_MISSING_SEVEN_EVIDENCE_V1'
MISSING_SEVEN = (
    'AVB2', 'AVB3', 'TRIGEMINY', 'SVT', 'ST_ELEVATION',
    'ISCHEMIA_GENERIC', 'MI_HISTORY_Q_SCREEN',
)
AVB2_CODES = frozenset({
    'MOBITZ_I_WENCKEBACH_COMPATIBLE', 'MOBITZ_II_COMPATIBLE',
    'TWO_TO_ONE_AV_BLOCK_COMPATIBLE', 'HIGH_GRADE_AV_BLOCK_COMPATIBLE',
})


def _mapping(x: Any) -> Mapping[str, Any]:
    return x if isinstance(x, Mapping) else {}


def _bounded_n(x: Any) -> int:
    if isinstance(x, bool):
        return 0
    try:
        n = int(x)
        return n if 0 <= n <= 100000 and float(x) == n else 0
    except (ValueError, TypeError, OverflowError):
        return 0


def _domain_available(gates: Mapping[str, Any], name: str) -> bool:
    row = _mapping(_mapping(gates.get('domains')).get(name))
    return row.get('eligible') is True


def _entry(code: str, status: str, reason: str, evidence: list[str]) -> dict[str, Any]:
    return {
        'module': code,
        'status': status,
        'reason': reason,
        'evidence': sorted(set(evidence)),
        'trained_model_available': False,
        'neural_output': None,
        'score': None,
        'binary_classification': None,
        'diagnostic_label': None,
        'clinical_diagnostic_claim_allowed': False,
        'clinical_deployment_allowed': False,
    }


def audit_missing_seven(
    *,
    av_conduction: Mapping[str, Any] | None = None,
    atrial_mechanism: Mapping[str, Any] | None = None,
    ectopy: Mapping[str, Any] | None = None,
    domain_gates: Mapping[str, Any] | None = None,
    repolarization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-actionable evidence manifest from existing measured features.

    All inputs must be upstream measurement/specialist evidence. No model
    probability/logit nor reference labels are accepted or interpreted.
    Missing domain-gate metadata fails closed rather than assuming eligibility.
    """
    av, atr, ec, gates, rep = map(_mapping, (
        av_conduction, atrial_mechanism, ectopy, domain_gates, repolarization,
    ))
    av_ok = _domain_available(gates, 'AV_CONDUCTION')
    rh_ok = _domain_available(gates, 'RHYTHM')
    ect_ok = _domain_available(gates, 'ECTOPY')
    rep_ok = _domain_available(gates, 'REPOLARIZATION')
    out: dict[str, dict[str, Any]] = {}

    av_class = str(av.get('classification') or '')
    av_counts = _bounded_n(av.get('nonconducted_p_n'))
    av_good = av.get('evaluable') is True and av_ok
    if not av_good:
        out['AVB2'] = _entry('AVB2', 'NOT_EVALUABLE', 'AV_EVIDENCE_OR_DOMAIN_GATE_NOT_USABLE', [])
        out['AVB3'] = _entry('AVB3', 'NOT_EVALUABLE', 'AV_EVIDENCE_OR_DOMAIN_GATE_NOT_USABLE', [])
    else:
        av2_pattern = av_class in AVB2_CODES
        av2_support = av.get('atrial_sequence_regular') is True and av_counts >= 1
        av2_status = 'RESEARCH_PATTERN_FOR_REVIEW' if av2_pattern and av2_support else ('INCONSISTENT_MECHANISTIC_EVIDENCE' if av2_pattern else 'NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE')
        out['AVB2'] = _entry('AVB2', av2_status, 'AV_CONDUCTION_MECHANISM_REQUIRES_MEDICAL_VERIFICATION', [av_class] if av2_pattern else [])
        av3_pattern = av_class == 'COMPLETE_AV_BLOCK_COMPATIBLE'
        av3_support = (
            av.get('av_dissociation_phase') is True
            and av.get('atrial_sequence_regular') is True
            and av.get('ventricular_sequence_regular') is True
        )
        av3_status = 'RESEARCH_PATTERN_FOR_REVIEW' if av3_pattern and av3_support else ('INCONSISTENT_MECHANISTIC_EVIDENCE' if av3_pattern else 'NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE')
        out['AVB3'] = _entry('AVB3', av3_status, 'AV_DISSOCIATION_REQUIRES_INDEPENDENT_CONFIRMATION', [av_class] if av3_pattern else [])

    if not ect_ok or ec.get('evaluable') is not True or ec.get('trigeminy_evaluable') is not True:
        out['TRIGEMINY'] = _entry('TRIGEMINY', 'NOT_EVALUABLE', 'BEAT_SEQUENCE_OR_ECTOPY_GATE_UNAVAILABLE', [])
    else:
        pattern = ec.get('trigeminy_pattern') is True and _bounded_n(ec.get('trigeminy_repeated_cycles')) >= 3
        out['TRIGEMINY'] = _entry('TRIGEMINY',
            'RESEARCH_PATTERN_FOR_REVIEW' if pattern else 'NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE',
            'REPETITIVE_ECTOPY_PATTERN_NOT_AUTONOMOUS_DIAGNOSIS',
            ['THREE_OR_MORE_REPEATED_NNV_CYCLES'] if pattern else [])

    atr_m = str(atr.get('mechanism') or '')
    if not rh_ok or atr.get('evaluable') is not True:
        out['SVT'] = _entry('SVT', 'NOT_EVALUABLE', 'RHYTHM_DOMAIN_OR_ATRIAL_MECHANISM_UNAVAILABLE', [])
    else:
        pattern = atr_m == 'OTHER_SVT_COMPATIBLE'
        out['SVT'] = _entry('SVT',
            'RESEARCH_PATTERN_FOR_REVIEW' if pattern else 'NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE',
            'ATRIAL_MECHANISM_ALONE_DOES_NOT_VERIFY_SVT_SUBTYPE',
            [atr_m] if pattern else [])

    leads = rep.get('st_elevation_leads')
    if not rep_ok or not isinstance(leads, (list, tuple)):
        out['ST_ELEVATION'] = _entry('ST_ELEVATION', 'NOT_EVALUABLE', 'CALIBRATED_LEADWISE_ST_MEASUREMENTS_UNAVAILABLE', [])
    else:
        named = [str(x) for x in leads if str(x) in {
            'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        }]
        out['ST_ELEVATION'] = _entry('ST_ELEVATION',
            'MEASURED_MORPHOLOGY_FOR_REVIEW' if named else 'NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE',
            'ST_PATTERN_NOT_STEMI_DIAGNOSIS; CONTIGUITY_AND_CONTEXT_REQUIRED',
            ['MEASURED_ST_ELEVATION_'+lead for lead in sorted(set(named))])

    out['ISCHEMIA_GENERIC'] = _entry('ISCHEMIA_GENERIC', 'CLINICAL_CONTEXT_REQUIRED',
        'ECG_MORPHOLOGY_ALONE_CANNOT_DIAGNOSE_MYOCARDIAL_ISCHEMIA', [])
    out['MI_HISTORY_Q_SCREEN'] = _entry('MI_HISTORY_Q_SCREEN', 'VALIDATED_Q_CRITERIA_REQUIRED',
        'PATHOLOGICAL_Q_SCREEN_NOT_EQUIVALENT_TO_PRIOR_INFARCTION', [])

    assert tuple(sorted(out)) == tuple(sorted(MISSING_SEVEN))
    return {
        'version': VERSION,
        'schema': 'NON_DIAGNOSTIC_SEVEN_MODULE_COVERAGE',
        'model_status': 'NO_NEW_TRAINED_MODEL',
        'clinical_deployment_status': 'BLOCKED',
        'diagnostic_claim_allowed': False,
        'measurement_mutation_allowed': False,
        'reasoner_mutation_allowed': False,
        'modules': out,
    }