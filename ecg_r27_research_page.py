from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import streamlit as st

from r27_local_runtime import (
    ALL35,
    EXPECTED_CLINICAL_STATUS,
    EXPECTED_RELEASE_TYPE,
    R27LocalError,
    run_r27_local,
)


def _get_secret(name: str, default: Any = None) -> Any:
    try:
        return st.secrets[name]
    except Exception:
        return default


def _load_photo_engine():
    from ecg_photo_engine import (
        assess_ecg_photo,
        detect_calibration_pulse,
        pdf_page_count,
        prepare_ecg_image,
        rectify_ecg_photo,
        render_ecg_pdf_page,
    )
    return {
        "assess": assess_ecg_photo,
        "calibration": detect_calibration_pulse,
        "pdf_page_count": pdf_page_count,
        "prepare": prepare_ecg_image,
        "rectify": rectify_ecg_photo,
        "render_pdf": render_ecg_pdf_page,
    }


def _load_nnunet():
    from ecg_nnunet_digitizer import (
        ECGDigitizerError,
        digitize_with_pretrained_nnunet,
        make_r27_wfdb_payload,
    )
    return ECGDigitizerError, digitize_with_pretrained_nnunet, make_r27_wfdb_payload


def _source_to_image_bytes(s, uploaded, engine):
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()

    if name.endswith(".pdf"):
        pages = int(engine["pdf_page_count"](raw))
        page_index = 0
        if pages > 1:
            page_number = s.number_input(
                "Página del PDF que contiene el ECG",
                min_value=1,
                max_value=pages,
                value=1,
                step=1,
                key="r27_photo_pdf_page",
            )
            page_index = int(page_number) - 1

        image_bytes, meta = engine["render_pdf"](
            raw,
            page_index=page_index,
            dpi=300,
            max_dimension=4200,
        )
        return image_bytes, {"source_type": "pdf", "filename": uploaded.name, **meta}

    image_bytes, _, meta = engine["prepare"](
        raw,
        crop_header=False,
        max_dimension=4200,
    )
    return image_bytes, {"source_type": "image", "filename": uploaded.name, **meta}


def _validate_probability_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("release_type") != EXPECTED_RELEASE_TYPE:
        raise R27LocalError("release_type R27 inesperado.")
    if payload.get("clinical_deployment_status") != EXPECTED_CLINICAL_STATUS:
        raise R27LocalError("clinical_deployment_status R27 inesperado.")

    modules = payload.get("modules")
    if not isinstance(modules, dict) or set(modules) != set(ALL35):
        raise R27LocalError("La salida R27 no contiene exactamente los 35 módulos.")

    for module in ALL35:
        item = modules[module]
        p = float(item["probability"])
        if not 0.0 <= p <= 1.0:
            raise R27LocalError(f"{module}: probability fuera de [0,1].")
        if item.get("threshold") is not None:
            raise R27LocalError(f"{module}: threshold no autorizado.")
        if item.get("binary_classification") is not None:
            raise R27LocalError(f"{module}: clasificación binaria no autorizada.")
        if item.get("diagnostic_label") is not None:
            raise R27LocalError(f"{module}: etiqueta diagnóstica no autorizada.")
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
            }
        )
    rows.sort(key=lambda x: x["Probabilidad"], reverse=True)

    s.success("R27 completado. Se muestran únicamente probabilidades.")
    s.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Probabilidad": s.column_config.ProgressColumn(
                "Probabilidad", min_value=0.0, max_value=1.0, format="%.4f"
            )
        },
    )
    s.warning(
        "Una probabilidad alta no equivale a diagnóstico positivo. "
        "R27 no dispone de thresholds clínicamente desplegables."
    )
    s.download_button(
        "Descargar resultado R27 (JSON)",
        data=json.dumps(payload, indent=2, ensure_ascii=False),
        file_name="medcalc_r27_probability_only_result.json",
        mime="application/json",
        use_container_width=True,
        key="r27_download",
    )


def _render_coverage_table(s, result: Dict[str, Any]) -> None:
    rows = []
    for lead in result["lead_names"]:
        rows.append(
            {
                "Derivación": lead,
                "Duración observada (s)": float(result["duration_sec_by_lead"][lead]),
                "Cobertura finita": float(result["finite_coverage_by_lead"][lead]),
            }
        )
    s.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cobertura finita": s.column_config.ProgressColumn(
                "Cobertura finita", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
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
            MEDCALC ECG · FOTO/PDF → nnU-NET → R27
          </div>
          <div style="font-size:1.65rem;font-weight:750;color:#12202f;margin-top:.15rem">
            ❤️ Electrocardiograma
          </div>
          <div style="color:#667788;margin-top:.25rem">
            Entrada de usuario final: fotografía o PDF del trazado.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s.error(
        "**MODO INVESTIGACIÓN. NO USAR COMO DIAGNÓSTICO CLÍNICO.** "
        "El digitalizador neuronal y el adaptador foto→R27 deben validarse antes de uso clínico."
    )

    uploaded = s.file_uploader(
        "Foto o PDF del ECG",
        type=["jpg", "jpeg", "png", "webp", "pdf"],
        key="r27_photo_pdf_upload",
        help="Preferir toma perpendicular, nítida, con cuadrícula y calibración visibles.",
    )
    if uploaded is None:
        s.info("Suba una fotografía o PDF para comenzar.")
        return

    try:
        photo = _load_photo_engine()
        source_bytes, source_meta = _source_to_image_bytes(s, uploaded, photo)
        rectified_bytes, rect_meta = photo["rectify"](source_bytes)
        quality = photo["assess"](rectified_bytes)
        calibration = photo["calibration"](rectified_bytes, quality)
    except Exception as exc:
        s.error(f"No fue posible preparar la imagen: {exc}")
        return

    s.image(
        rectified_bytes,
        caption=f"Entrada rectificada · {source_meta.get('filename')}",
        use_container_width=True,
    )

    c1, c2, c3 = s.columns(3)
    c1.metric("Calidad", str(quality.get("quality_label") or "—"))
    c2.metric("Puntaje", str(quality.get("quality_score") or "—"))
    c3.metric("Cuadrícula", f"{float(quality.get('grid_confidence') or 0.0)*100:.0f}%")

    if calibration.get("speed_mm_s"):
        s.success(
            f"Calibración geométrica: {calibration.get('speed_mm_s')} mm/s · "
            f"{calibration.get('gain_mm_mV')} mm/mV."
        )
    else:
        s.warning(
            "La calibración automática no quedó demostrada por la capa geométrica. "
            "El nnU-Net puede segmentar, pero no se autorizará R27 si la reconstrucción "
            "no demuestra cobertura temporal completa."
        )

    if not quality.get("digitization_allowed"):
        s.error("Calidad insuficiente para ejecutar el digitalizador neuronal.")
        return

    s.caption(
        "La primera ejecución descarga el modelo M3 preentrenado del ganador del "
        "PhysioNet Challenge 2024 (~475 MB) y verifica su SHA-256."
    )

    if not s.button(
        "Digitalizar ECG con nnU-Net",
        type="primary",
        use_container_width=True,
        key="r27_run_nnunet",
    ):
        return

    try:
        ECGDigitizerError, digitize, make_wfdb = _load_nnunet()
    except Exception as exc:
        s.error(f"No fue posible cargar nnU-Net: {exc}")
        return

    with s.spinner("Segmentando 12 derivaciones y reconstruyendo la señal…"):
        try:
            result = digitize(rectified_bytes)
        except ECGDigitizerError as exc:
            s.error(str(exc))
            return
        except Exception as exc:
            s.error(f"Fallo no esperado en el digitalizador: {exc}")
            return

    s.image(
        result["mask_overlay_bytes"],
        caption="Máscara nnU-Net sobre el ECG",
        use_container_width=True,
    )

    s.markdown("### Cobertura recuperada")
    _render_coverage_table(s, result)

    s.caption(
        f"Modelo: {result['model_source']} · {result['model_name']} · "
        f"commit {result['model_commit'][:12]} · "
        f"rotación corregida {result['rotation_deg']:.2f}°"
    )

    if not result["r27_temporal_coverage_eligible"]:
        s.error(
            "ECG DIGITALIZADO, PERO R27 BLOQUEADO POR COBERTURA TEMPORAL. "
            "El motor R27 congelado requiere 10 s completos en las 12 derivaciones. "
            "Un impreso estándar 3×4 normalmente contiene ~2.5 s por derivación; "
            "los 7.5 s restantes no se inventarán, repetirán ni imputarán."
        )
        s.info(
            "Para habilitar R27 desde imagen, el documento debe contener 10 s completos "
            "de las 12 derivaciones, o habrá que desarrollar y validar un modelo nuevo "
            "específico para ECG impreso 3×4."
        )
        return

    github_token = _get_secret("R27_GITHUB_TOKEN")
    if not github_token:
        s.error(
            "La imagen sí cumple cobertura para R27, pero falta R27_GITHUB_TOKEN "
            "en Streamlit Secrets."
        )
        return

    s.success("Cobertura 10 s × 12 derivaciones demostrada. R27 puede ejecutarse.")

    c_age, c_sex = s.columns(2)
    with c_age:
        age = s.number_input(
            "Edad (años)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=1.0,
            key="r27_photo_age",
        )
    with c_sex:
        sex = s.selectbox(
            "Sexo codificado para runtime",
            options=["0", "1"],
            key="r27_photo_sex",
        )

    if not s.button(
        "Enviar señal reconstruida a R27",
        type="primary",
        use_container_width=True,
        key="r27_run_from_photo",
    ):
        return

    try:
        wf = make_wfdb(result["signals_500"], result["lead_names"])
    except Exception as exc:
        s.error(f"No fue posible construir el paquete WFDB foto→R27: {exc}")
        return

    with s.spinner("Ejecutando R27 sobre la señal reconstruida…"):
        try:
            payload = run_r27_local(
                str(github_token),
                age=float(age),
                sex=str(sex),
                hr_hea_name=wf["hr_hea_name"],
                hr_hea_bytes=wf["hr_hea_bytes"],
                hr_dat_name=wf["hr_dat_name"],
                hr_dat_bytes=wf["hr_dat_bytes"],
                lr_hea_name=wf["lr_hea_name"],
                lr_hea_bytes=wf["lr_hea_bytes"],
                lr_dat_name=wf["lr_dat_name"],
                lr_dat_bytes=wf["lr_dat_bytes"],
                timeout_seconds=900,
            )
            payload = _validate_probability_payload(payload)
        except Exception as exc:
            s.error(f"R27 no pudo completarse: {exc}")
            return

    _render_probability_table(s, payload)
