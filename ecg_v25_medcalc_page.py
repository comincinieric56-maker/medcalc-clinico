from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ecg_v23_medcalc_page import analyze_uploaded_ecg
from ecg_v24_criteria_engine import interpret_v23_zip, SOURCE_REGISTRY
from ecg_v25_visual_atlas import ATLAS, VERSION as ATLAS_VERSION
from ecg_v25_visual_matcher import (
    rank_visual_atlas,
    render_match_overlay,
    VERSION as V25_VERSION,
)


def _fmt_score(x):
    if x is None:
        return "NO VALORABLE"
    return f"{100.0*float(x):.0f}%"


def _render_measurements(st, measurements):
    st.markdown("### 1 · Mediciones V23.1")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("FC","NO VALORABLE" if measurements.get("heart_rate_bpm") is None else f"{measurements['heart_rate_bpm']:.1f} lpm")
    c2.metric("PR","NO VALORABLE" if measurements.get("pr_ms") is None else f"{measurements['pr_ms']:.0f} ms")
    c3.metric("QRS","NO VALORABLE" if measurements.get("qrs_ms") is None else f"{measurements['qrs_ms']:.0f} ms")
    c4.metric("QT","NO VALORABLE" if measurements.get("qt_ms") is None else f"{measurements['qt_ms']:.0f} ms")
    c5,c6,c7,c8=st.columns(4)
    c5.metric("QTcF","NO VALORABLE" if measurements.get("qtc_f_ms") is None else f"{measurements['qtc_f_ms']:.0f} ms")
    axis=measurements.get("axis_deg")
    c6.metric("Eje","NO VALORABLE" if axis is None else f"{float(axis):+.0f}°")
    rr=measurements.get("rr_median_sec")
    c7.metric("RR","NO VALORABLE" if rr is None else f"{float(rr):.3f} s")
    c8.metric("QC",str(measurements.get("measurement_qc") or "—"))


def _render_v24(st, v24):
    st.markdown("### 2 · Criterios electrocardiográficos V24")
    interp=(v24.get("interpretation") or {})
    urg=interp.get("urgent_flags") or []
    if urg:
        st.error("⚠️ Hay criterios automatizados de alto riesgo que requieren revisión inmediata del trazado y del contexto clínico.")
    for s in interp.get("final_statements") or []:
        st.markdown(f"- **{s}**")
    with st.expander("Ver auditoría V24 completa"):
        rows=[]
        for r in interp.get("rules") or []:
            rows.append({
                "Regla":r.get("rule_id"),
                "Hallazgo":r.get("label"),
                "Estado":r.get("status"),
                "Confianza":r.get("confidence"),
                "Criterios":" · ".join(r.get("criteria_met") or []) or "—",
                "Fuente":", ".join(r.get("sources") or []) or "—",
            })
        st.dataframe(rows,hide_index=True,use_container_width=True)


def _render_v25(st, v25):
    st.markdown("### 3 · Confirmación morfológica visual V25")
    st.caption(
        "La similitud visual se calcula sobre la señal digitalizada mV/seg, no sobre los píxeles de la foto. "
        "Esto reduce el efecto de cuadrícula, color, compresión, rotación y fondo."
    )
    matches=v25.get("matches") or []
    rows=[]
    for m in matches[:10]:
        rows.append({
            "Patrón":m.get("name"),
            "Familia":m.get("family"),
            "Visual":_fmt_score(m.get("visual_score")),
            "Criterios":_fmt_score(m.get("criteria_score")),
            "Mediciones":_fmt_score(m.get("measurement_score")),
            "Fusión":_fmt_score(m.get("fusion_score")),
            "Estado":m.get("fusion_status"),
        })
    st.dataframe(rows,hide_index=True,use_container_width=True)

    concordant=v25.get("concordant_matches") or []
    if concordant:
        st.success("Concordancia entre criterios y morfología visual:")
        for m in concordant[:5]:
            st.markdown(
                f"- **{m['name']}** · visual {_fmt_score(m.get('visual_score'))} · "
                f"fusión {_fmt_score(m.get('fusion_score'))} · `{m['fusion_status']}`"
            )
    else:
        st.info("No hubo patrón con concordancia visual + criterios suficiente para promoción automática.")

    discord=v25.get("discordant_matches") or []
    if discord:
        st.warning(
            "Existen coincidencias visuales discordantes. MEDCALC no las convierte en diagnóstico "
            "porque los criterios explícitos no las sostienen."
        )

    # Render top 3 visual references/overlays.
    shown=0
    for m in matches:
        if shown>=3: break
        if m.get("visual_score") is None: continue
        pid=m["pattern_id"]
        with st.expander(
            f"Comparación visual · {m['name']} · {_fmt_score(m.get('visual_score'))}",
            expanded=(shown==0),
        ):
            asset=Path(__file__).resolve().parent/"atlas_assets"/f"{pid}.png"
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Referencia sintética original V25**")
                if asset.exists():
                    st.image(str(asset),use_container_width=True)
                else:
                    st.caption("Referencia no renderizada en el paquete.")
            with c2:
                st.markdown("**Paciente vs referencia sintética**")
                try:
                    tmp=Path(tempfile.gettempdir())/f"medcalc_v25_overlay_{pid}.png"
                    render_match_overlay(v25,pid,tmp)
                    st.image(str(tmp),use_container_width=True)
                except Exception as exc:
                    st.caption(f"Overlay no disponible: {exc}")

            lead_rows=[]
            for lead,ls in (m.get("lead_scores") or {}).items():
                if not ls.get("reliable"): continue
                lead_rows.append({
                    "Derivación":lead,
                    "Visual":_fmt_score(ls.get("visual_score")),
                    "Forma":_fmt_score(ls.get("shape_score")),
                    "Amplitud":_fmt_score(ls.get("amplitude_score")),
                })
            if lead_rows:
                st.dataframe(lead_rows,hide_index=True,use_container_width=True)
            st.caption("Criterios prototipo: " + " · ".join(m.get("criteria_summary") or []))
            st.caption("Fuentes: " + ", ".join(m.get("sources") or []))
        shown+=1


def page_ecg_v25(st):
    st.markdown("## ❤️ MEDCALC ECG V25 · Visual Atlas Fusion")
    st.caption(
        "V23.1 mide · V24 aplica criterios de literatura · V25 compara la morfología "
        "digitalizada contra un atlas sintético original y fusiona la evidencia."
    )

    st.info(
        "Las imágenes del atlas NO son copias de libros. Son recreaciones digitales paramétricas "
        "basadas en los criterios morfológicos. El patrón visual nunca reemplaza un criterio ECG."
    )

    d1,d2=st.columns(2)
    with d1:
        age_on=st.checkbox("Informar edad",value=True,key="ecg_v25_age_on")
        age=st.number_input(
            "Edad (años)",min_value=0.0,max_value=120.0,value=40.0,step=1.0,key="ecg_v25_age"
        ) if age_on else None
    with d2:
        sx=st.selectbox(
            "Sexo para criterios sexoespecíficos",
            ["No informado","Masculino","Femenino"],key="ecg_v25_sex"
        )
        sex={"Masculino":"male","Femenino":"female"}.get(sx)

    uploaded=st.file_uploader(
        "ECG completo en PDF, JPG, JPEG o PNG",
        type=["pdf","jpg","jpeg","png"],
        key="ecg_v25_upload",
    )
    if uploaded is None:
        st.caption(
            "El motor necesita las 12 derivaciones y cuadrícula visible. "
            "La primera ejecución puede ser más lenta por carga de modelos."
        )
        return

    if not st.button(
        "Analizar · mediciones + criterios + atlas visual",
        type="primary",use_container_width=True,key="ecg_v25_run"
    ):
        return

    try:
        with st.spinner("1/3 · V23.1 digitalizando y midiendo…"):
            v23=analyze_uploaded_ecg(uploaded.getvalue(),uploaded.name)
        zip_path=v23.get("zip_path")
        if not zip_path or not Path(zip_path).exists():
            st.error("V23.1 no produjo un ZIP de auditoría.")
            return

        with st.spinner("2/3 · V24 aplicando criterios de literatura…"):
            v24=interpret_v23_zip(zip_path,age_years=age,sex=sex)
        if v24.get("status")!="PASS":
            st.error(f"V24 NO VALORABLE: {v24.get('reason')}")
            return

        with st.spinner("3/3 · V25 comparando morfología con atlas digital…"):
            v25=rank_visual_atlas(zip_path,age_years=age,sex=sex)
        if v25.get("status")!="PASS":
            st.error(f"V25 NO VALORABLE: {v25.get('reason')}")
            return
    except Exception as exc:
        st.exception(exc)
        return

    manifest=v23.get("manifest") or {}
    measurements=manifest.get("objective_measurements") or {}
    _render_measurements(st,measurements)
    _render_v24(st,v24)
    _render_v25(st,v25)

    st.markdown("### Conclusión de fusión")
    concord=v25.get("concordant_matches") or []
    if concord:
        for m in concord[:5]:
            st.markdown(f"- **{m['name']}** — soporte `{m['fusion_status']}`")
    else:
        st.markdown("- Sin concordancia morfológica + criterios suficiente para un patrón mayor.")

    st.warning(
        "La conclusión es una interpretación ECG asistida. No confirma por sí sola IAM, síndrome de Brugada, "
        "pericarditis, hipertrofia anatómica u otras enfermedades que requieren contexto clínico, biomarcadores, "
        "imagen, genética o ECG seriados."
    )

    # Avoid serializing patient beat arrays into the UI download.
    export=dict(v25)
    export.pop("patient_beats",None)
    payload=json.dumps(export,indent=2,ensure_ascii=False).encode("utf-8")
    st.download_button(
        "Descargar informe V25 (.json)",
        data=payload,
        file_name=f"{Path(uploaded.name).stem}__MEDCALC_ECG_V25.json",
        mime="application/json",
        use_container_width=True,
    )

    if zip_path and Path(zip_path).exists():
        st.download_button(
            "Descargar auditoría de señal V23.1 (.zip)",
            data=Path(zip_path).read_bytes(),
            file_name=Path(zip_path).name,
            mime="application/zip",
            use_container_width=True,
        )

    with st.expander("Bibliografía y trazabilidad"):
        rows=[]
        used=set()
        for m in v25.get("matches") or []:
            used.update(m.get("sources") or [])
        for sid in sorted(used):
            src=SOURCE_REGISTRY.get(sid) or {}
            rows.append({
                "ID":sid,
                "Fuente":src.get("title") or sid,
                "Año":src.get("year") or "—",
                "DOI":src.get("doi") or "—",
                "URL":src.get("url") or "—",
            })
        st.dataframe(rows,hide_index=True,use_container_width=True)
