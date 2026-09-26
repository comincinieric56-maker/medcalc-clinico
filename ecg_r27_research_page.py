from __future__ import annotations

import json
from typing import Any, Dict

import requests
import streamlit as st


ALL35 = [
    "SINUS","AF","FLUTTER","SINUS_BRADY","SINUS_TACHY","SINUS_ARRHYTHMIA",
    "PVC","PAC","BIGEMINY","TRIGEMINY","AVB1","AVB2","AVB3",
    "RBBB_COMPLETE","RBBB_INCOMPLETE","LBBB","LBBB_INCOMPLETE","IVCD",
    "LAFB","LPFB","WPW","LVH","RVH","LAE","RAE","LOW_VOLTAGE","Q_WAVE",
    "LONG_QT","ST_DEPRESSION","ST_ELEVATION","ISCHEMIA_GENERIC",
    "MI_HISTORY_Q_SCREEN","PACEMAKER","SVT","NORMAL_ECG",
]

EXPECTED_RELEASE_TYPE = "RESEARCH_PROBABILITY_ONLY_RELEASE"
EXPECTED_CLINICAL_STATUS = "CLINICAL_DEPLOYMENT_BLOCKED"


class R27ClientError(RuntimeError):
    pass


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _validate_probability_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise R27ClientError("Respuesta R27 inválida: se esperaba un objeto JSON.")

    if payload.get("release_type") != EXPECTED_RELEASE_TYPE:
        raise R27ClientError(
            "Respuesta rechazada: release_type no corresponde al release R27 autorizado."
        )

    if payload.get("clinical_deployment_status") != EXPECTED_CLINICAL_STATUS:
        raise R27ClientError(
            "Respuesta rechazada: clinical_deployment_status inesperado."
        )

    modules = payload.get("modules")
    if not isinstance(modules, dict):
        raise R27ClientError("Respuesta R27 sin diccionario 'modules'.")

    if set(modules) != set(ALL35):
        missing = sorted(set(ALL35) - set(modules))
        extra = sorted(set(modules) - set(ALL35))
        raise R27ClientError(
            f"Conjunto de módulos inválido. Faltan={missing}; extra={extra}"
        )

    for module in ALL35:
        item = modules[module]
        if not isinstance(item, dict):
            raise R27ClientError(f"{module}: salida inválida.")

        try:
            p = float(item["probability"])
        except Exception as exc:
            raise R27ClientError(f"{module}: probability ausente/no numérica.") from exc

        if not 0.0 <= p <= 1.0:
            raise R27ClientError(f"{module}: probability fuera de [0,1].")

        for forbidden in ("threshold", "binary_classification", "diagnostic_label"):
            if item.get(forbidden) is not None:
                raise R27ClientError(
                    f"{module}: el backend intentó exponer '{forbidden}', "
                    "lo cual no está autorizado por R27."
                )

        if item.get("clinical_diagnostic_claim_allowed") not in (False, None):
            raise R27ClientError(
                f"{module}: clinical_diagnostic_claim_allowed debe ser False."
            )

    return payload


def _post_r27(
    api_url: str,
    token: str | None,
    *,
    age: float,
    sex: str,
    hr_hea,
    hr_dat,
    lr_hea,
    lr_dat,
) -> Dict[str, Any]:
    endpoint = api_url.rstrip("/") + "/v1/ecg/probabilities"

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    files = {
        "hr_hea": (hr_hea.name, hr_hea.getvalue(), "text/plain"),
        "hr_dat": (hr_dat.name, hr_dat.getvalue(), "application/octet-stream"),
        "lr_hea": (lr_hea.name, lr_hea.getvalue(), "text/plain"),
        "lr_dat": (lr_dat.name, lr_dat.getvalue(), "application/octet-stream"),
    }

    data = {
        "age": str(float(age)),
        "sex": str(sex),
        "release": EXPECTED_RELEASE_TYPE,
    }

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            files=files,
            data=data,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise R27ClientError(f"No fue posible contactar el backend R27: {exc}") from exc

    if response.status_code != 200:
        detail = response.text[:2000]
        raise R27ClientError(
            f"Backend R27 respondió HTTP {response.status_code}: {detail}"
        )

    try:
        payload = response.json()
    except Exception as exc:
        raise R27ClientError("Backend R27 no devolvió JSON válido.") from exc

    return _validate_probability_payload(payload)


def page_ecg_r27_research(st_module=None):
    s = st_module or st

    s.markdown(
        """
        <div style="
            border:1px solid #d6e3e8;
            border-radius:16px;
            padding:1rem 1.1rem;
            background:#ffffff;
            margin-bottom:1rem;">
          <div style="font-size:.78rem;font-weight:700;letter-spacing:.08em;color:#667788">
            MEDCALC ECG · V41.0R27
          </div>
          <div style="font-size:1.65rem;font-weight:750;color:#12202f;margin-top:.15rem">
            ❤️ ECG · Señal cruda · Investigación
          </div>
          <div style="color:#667788;margin-top:.25rem">
            Release final probability-only. No emite diagnóstico binario ni aplica umbrales.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s.error(
        "**MODO INVESTIGACIÓN. NO USAR COMO DIAGNÓSTICO CLÍNICO.** "
        "R27 permite únicamente probabilidades. No existe un threshold desplegable "
        "vinculado al SHA actual para ninguno de los 35 módulos y la validación externa "
        "del modelo actual no está establecida bajo el gate R26."
    )

    api_url = _get_secret("ECG_R27_API_URL")
    api_token = _get_secret("ECG_R27_API_TOKEN")

    if not api_url:
        s.warning(
            "El frontend R27 está instalado, pero falta configurar "
            "`ECG_R27_API_URL` en Streamlit Secrets."
        )
        s.code(
            'ECG_R27_API_URL = "https://TU-BACKEND-R27"\n'
            'ECG_R27_API_TOKEN = "..."',
            language="toml",
        )
        return

    s.markdown("### Entrada exacta requerida")
    s.caption(
        "R27 fue cerrado sobre dos representaciones oficiales de la misma señal: "
        "500 Hz y 100 Hz. No se sintetiza ni remuestrea una frecuencia a partir de la otra."
    )

    c1, c2 = s.columns(2)

    with c1:
        s.markdown("**500 Hz**")
        hr_hea = s.file_uploader(
            "Archivo .hea (500 Hz)",
            type=["hea"],
            key="r27_hr_hea",
        )
        hr_dat = s.file_uploader(
            "Archivo .dat (500 Hz)",
            type=["dat"],
            key="r27_hr_dat",
        )

    with c2:
        s.markdown("**100 Hz**")
        lr_hea = s.file_uploader(
            "Archivo .hea (100 Hz)",
            type=["hea"],
            key="r27_lr_hea",
        )
        lr_dat = s.file_uploader(
            "Archivo .dat (100 Hz)",
            type=["dat"],
            key="r27_lr_dat",
        )

    c_age, c_sex = s.columns(2)

    with c_age:
        age = s.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=1.0,
            key="r27_age",
        )

    with c_sex:
        sex = s.selectbox(
            "Sexo codificado para runtime",
            options=["0", "1"],
            key="r27_sex",
            help=(
                "Se conserva la codificación requerida por el runtime congelado. "
                "No se infiere ni transforma automáticamente."
            ),
        )

    ready = all([hr_hea, hr_dat, lr_hea, lr_dat])

    if not ready:
        s.info("Cargue los cuatro archivos WFDB para habilitar el análisis.")
        return

    if not s.button(
        "Analizar con R27",
        type="primary",
        use_container_width=True,
        key="r27_run",
    ):
        return

    with s.spinner("Ejecutando runtime R27 congelado…"):
        try:
            payload = _post_r27(
                str(api_url),
                str(api_token) if api_token else None,
                age=age,
                sex=sex,
                hr_hea=hr_hea,
                hr_dat=hr_dat,
                lr_hea=lr_hea,
                lr_dat=lr_dat,
            )
        except R27ClientError as exc:
            s.error(str(exc))
            return

    rows = []

    for module in ALL35:
        item = payload["modules"][module]
        rows.append(
            {
                "Módulo": module,
                "Probabilidad": float(item["probability"]),
                "Threshold": "NO DISPONIBLE",
                "Clasificación binaria": "NO AUTORIZADA",
                "Validación externa SHA actual": "NO ESTABLECIDA",
            }
        )

    rows = sorted(
        rows,
        key=lambda x: x["Probabilidad"],
        reverse=True,
    )

    s.success("Runtime completado. Se muestran únicamente probabilidades.")

    s.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probabilidad": s.column_config.ProgressColumn(
                "Probabilidad",
                min_value=0.0,
                max_value=1.0,
                format="%.4f",
            ),
        },
    )

    s.warning(
        "Una probabilidad alta **no equivale a diagnóstico positivo**. "
        "R27 no dispone de thresholds clínicamente desplegables para estos modelos."
    )

    s.download_button(
        "Descargar resultado R27 (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name="medcalc_r27_probability_only_result.json",
        mime="application/json",
        use_container_width=True,
        key="r27_download",
    )
