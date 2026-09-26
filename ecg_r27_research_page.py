from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st

from r27_local_runtime import (
    ALL35,
    EXPECTED_CLINICAL_STATUS,
    EXPECTED_RELEASE_TYPE,
    R27_SOURCE_COMMIT,
    R27LocalError,
    run_r27_local,
    runtime_status,
)


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _validate_probability_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise R27LocalError("Respuesta R27 inválida: se esperaba un objeto JSON.")

    if payload.get("release_type") != EXPECTED_RELEASE_TYPE:
        raise R27LocalError("release_type no corresponde al release R27 autorizado.")

    if payload.get("clinical_deployment_status") != EXPECTED_CLINICAL_STATUS:
        raise R27LocalError("clinical_deployment_status R27 inesperado.")

    modules = payload.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(ALL35):
        raise R27LocalError("La salida no contiene exactamente los 35 módulos R27.")

    for module in ALL35:
        item = modules[module]
        p = float(item["probability"])
        if not 0.0 <= p <= 1.0:
            raise R27LocalError(f"{module}: probability fuera de [0,1].")
        for forbidden in ("threshold", "binary_classification", "diagnostic_label"):
            if item.get(forbidden) is not None:
                raise R27LocalError(f"{module}: '{forbidden}' no autorizado.")
        if item.get("clinical_diagnostic_claim_allowed") is not False:
            raise R27LocalError(f"{module}: claim clínico no autorizado.")

    return payload


def _render_probability_table(s, payload: Dict[str, Any]) -> None:
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

    rows.sort(key=lambda x: x["Probabilidad"], reverse=True)

    s.success("Runtime R27 completado localmente. Se muestran únicamente probabilidades.")
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
            )
        },
    )
    s.warning(
        "Una probabilidad alta no equivale a diagnóstico positivo. "
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


def _digital_ecg_tab(s, github_token: str | None) -> None:
    s.markdown("### ECG digital · WFDB 500 Hz + 100 Hz")
    s.caption(
        "El motor R27 se ejecuta dentro de la misma instancia de Streamlit. "
        "No usa Render ni un backend HTTP externo."
    )

    if not github_token:
        s.error(
            "Falta R27_GITHUB_TOKEN en Streamlit Secrets. "
            "Debe ser un token GitHub de solo lectura limitado al repositorio privado "
            "comincinieric56-maker/medcalc-r27-backend."
        )
        s.code('R27_GITHUB_TOKEN = "github_pat_..."', language="toml")
        return

    with s.expander("Estado del runtime local R27", expanded=False):
        s.caption(f"Fuente congelada: commit {R27_SOURCE_COMMIT}")
        if s.button("Verificar / materializar R27", key="r27_verify_runtime"):
            with s.spinner("Descargando y verificando el runtime R27 congelado…"):
                try:
                    status = runtime_status(str(github_token))
                except Exception as exc:
                    s.error(f"No fue posible preparar R27: {exc}")
                else:
                    s.success(
                        f"R27 listo · {status['modules']} módulos · "
                        f"{status['critical_sha_n']} SHA críticos verificados."
                    )

    c1, c2 = s.columns(2)
    with c1:
        s.markdown("**500 Hz**")
        hr_hea = s.file_uploader(
            "Archivo .hea (500 Hz)", type=["hea"], key="r27_hr_hea"
        )
        hr_dat = s.file_uploader(
            "Archivo .dat (500 Hz)", type=["dat"], key="r27_hr_dat"
        )

    with c2:
        s.markdown("**100 Hz**")
        lr_hea = s.file_uploader(
            "Archivo .hea (100 Hz)", type=["hea"], key="r27_lr_hea"
        )
        lr_dat = s.file_uploader(
            "Archivo .dat (100 Hz)", type=["dat"], key="r27_lr_dat"
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

    if not all([hr_hea, hr_dat, lr_hea, lr_dat]):
        s.info("Cargue los cuatro archivos WFDB para habilitar el análisis.")
        return

    if not s.button(
        "Analizar localmente con R27",
        type="primary",
        use_container_width=True,
        key="r27_run",
    ):
        return

    with s.spinner(
        "Ejecutando R27 dentro de Streamlit. La primera ejecución puede tardar "
        "mientras se materializa el runtime congelado…"
    ):
        try:
            payload = run_r27_local(
                str(github_token),
                age=float(age),
                sex=str(sex),
                hr_hea_name=hr_hea.name,
                hr_hea_bytes=hr_hea.getvalue(),
                hr_dat_name=hr_dat.name,
                hr_dat_bytes=hr_dat.getvalue(),
                lr_hea_name=lr_hea.name,
                lr_hea_bytes=lr_hea.getvalue(),
                lr_dat_name=lr_dat.name,
                lr_dat_bytes=lr_dat.getvalue(),
                timeout_seconds=900,
            )
            payload = _validate_probability_payload(payload)
        except R27LocalError as exc:
            s.error(str(exc))
            return
        except Exception as exc:
            s.error(f"Fallo no esperado al ejecutar R27: {exc}")
            return

    _render_probability_table(s, payload)


def _photo_pdf_tab(s) -> None:
    s.markdown("### Foto / PDF")
    s.info(
        "Esta entrada queda reservada para el digitalizador definitivo "
        "foto/PDF → U-Net → señal → control de calidad → R27."
    )
    s.warning(
        "Todavía no se envían fotografías ni PDF a R27. "
        "No se inventarán segmentos no impresos ni se completarán derivaciones "
        "por inferencia. La integración se habilitará sólo después de validar "
        "el adaptador foto→señal contra el ECG digital original."
    )
    s.caption(
        "El antiguo digitalizador y el ECG V9 Auditor quedan fuera de la navegación "
        "pública mientras se integra esta ruta definitiva."
    )


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
            MEDCALC ECG · V41.0R27 · LOCAL
          </div>
          <div style="font-size:1.65rem;font-weight:750;color:#12202f;margin-top:.15rem">
            ❤️ Electrocardiograma
          </div>
          <div style="color:#667788;margin-top:.25rem">
            Motor R27 dentro de Streamlit · release probability-only.
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

    github_token = _get_secret("R27_GITHUB_TOKEN")

    tab_photo, tab_digital = s.tabs(["📷 Foto / PDF", "📈 ECG digital"])

    with tab_photo:
        _photo_pdf_tab(s)

    with tab_digital:
        _digital_ecg_tab(s, str(github_token) if github_token else None)
