"""Fail-closed, research-only UI for the seven unsupported ECG categories.

Intentionally read-only: no model scoring, no thresholding, no reasoner/measurement
mutation, and no insertion of these states into clinical PDF/TXT conclusions.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VERSION = "MEDCALC_ECG_G29_SEVEN_UI_AUDIT_V1"
CODES = (
    "AVB2", "AVB3", "SVT", "TRIGEMINY", "ST_ELEVATION",
    "ISCHEMIA_GENERIC", "MI_HISTORY_Q_SCREEN",
)
NAMES = {
    "AVB2": "Conducción AV: segundo grado",
    "AVB3": "Conducción AV: tercer grado",
    "SVT": "Taquicardia supraventricular",
    "TRIGEMINY": "Periodicidad ectópica: trigeminismo",
    "ST_ELEVATION": "Morfología del segmento ST",
    "ISCHEMIA_GENERIC": "Hallazgos compatibles con isquemia (requieren contexto)",
    "MI_HISTORY_Q_SCREEN": "Cribado morfológico de onda Q",
}
STATES = {
    "NOT_EVALUABLE": "No evaluable",
    "RESEARCH_PATTERN_FOR_REVIEW": "Patrón experimental; requiere revisión",
    "INCONSISTENT_MECHANISTIC_EVIDENCE": "Evidencia mecanística discordante",
    "NOT_ESTABLISHED_BY_AVAILABLE_EVIDENCE": "Sin patrón establecido con los datos disponibles",
    "MEASURED_MORPHOLOGY_FOR_REVIEW": "Morfología medida; revisar, no diagnóstico",
    "CLINICAL_CONTEXT_REQUIRED": "Requiere contexto clínico",
    "VALIDATED_Q_CRITERIA_REQUIRED": "Requiere criterios de onda Q validados",
}
NEVER_OUTPUT = ("score", "neural_output", "binary_classification", "diagnostic_label")


def safe_seven_audit_rows(structured_report: Any) -> dict[str, Any]:
    """Return a non-diagnostic display projection, never raw model inputs.

    'absent' is not an error for prior report versions. 'blocked' signals an
    inconsistent/malicious report where the UI must not render module states.
    """
    if not isinstance(structured_report, Mapping):
        return {"status": "blocked", "rows": [], "reason": "REPORT_INVALID"}
    if "unsupported_seven_evidence" not in structured_report:
        return {"status": "absent", "rows": [], "reason": "OLDER_REPORT_VERSION"}
    manifest = structured_report["unsupported_seven_evidence"]
    if not isinstance(manifest, Mapping):
        return {"status": "blocked", "rows": [], "reason": "MANIFEST_INVALID"}
    must_equal = {
        "schema": "NON_DIAGNOSTIC_SEVEN_MODULE_COVERAGE",
        "version": "MEDCALC_ECG_G26_MISSING_SEVEN_EVIDENCE_V1",
        "model_status": "NO_NEW_TRAINED_MODEL",
        "clinical_deployment_status": "BLOCKED",
        "diagnostic_claim_allowed": False,
        "measurement_mutation_allowed": False,
        "reasoner_mutation_allowed": False,
    }
    for key, expected in must_equal.items():
        # Require exact booleans rather than truthiness/int coercion.
        actual = manifest.get(key)
        if type(expected) is bool and type(actual) is not bool:
            return {"status": "blocked", "rows": [], "reason": "MANIFEST_POLICY_INVALID"}
        if actual != expected:
            return {"status": "blocked", "rows": [], "reason": "MANIFEST_POLICY_INVALID"}
    modules = manifest.get("modules")
    if not isinstance(modules, Mapping) or set(modules) != set(CODES):
        return {"status": "blocked", "rows": [], "reason": "MODULE_SET_INVALID"}
    rows = []
    for code in CODES:
        item = modules[code]
        if not isinstance(item, Mapping) or item.get("module") != code:
            return {"status": "blocked", "rows": [], "reason": "MODULE_INVALID"}
        for key in ("trained_model_available", "clinical_diagnostic_claim_allowed", "clinical_deployment_allowed"):
            if item.get(key) is not False:
                return {"status": "blocked", "rows": [], "reason": "CLINICAL_CLAIM_NOT_BLOCKED"}
        if any(item.get(k, object()) is not None for k in NEVER_OUTPUT):
            return {"status": "blocked", "rows": [], "reason": "DIAGNOSTIC_VALUE_LEAK"}
        state = item.get("status")
        if not isinstance(state, str) or state not in STATES:
            return {"status": "blocked", "rows": [], "reason": "UNRECOGNIZED_STATUS"}
        # Author-controlled display text; never display raw 'reason', free text,
        # probabilities, or proposed diagnoses from external model output.
        rows.append({"Categoría investigada": NAMES[code], "Evidencia": STATES[state]})
    return {"status": "ready", "rows": rows, "reason": "RESEARCH_ONLY"}


def render_seven_audit_ui(st: Any, structured_report: Any) -> str:
    """Render an isolated, collapsed Streamlit panel after the main report.

    Never calls the inference engine or changes the input dict.
    """
    projection = safe_seven_audit_rows(structured_report)
    status = projection["status"]
    if status == "absent":
        return status
    if status == "blocked":
        st.warning(
            "Auditoría G26 no disponible: contrato experimental inválido o incompleto. "
            "No se muestran puntuaciones ni conclusiones diagnósticas."
        )
        return status
    with st.expander("Auditoría de evidencia G26 · siete categorías (investigación)", expanded=False):
        st.warning(
            "NO USAR PARA DIAGNÓSTICO CLÍNICO. No son probabilidades, "
            "clasificaciones, diagnósticos confirmados ni recomendaciones de tratamiento."
        )
        st.dataframe(projection["rows"], use_container_width=True, hide_index=True)
        st.caption(
            "Sin modelos neuronales entrenados para estas siete salidas. "
            "Ninguna de estas filas modifica mediciones, IDX, conclusiones, "
            "informe TXT/PDF ni conducta clínica."
        )
    return status