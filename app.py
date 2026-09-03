from pathlib import Path
import math

import streamlit as st

from supabase_repository import SupabaseRepository, SCHEMA_VERSION, normalize_text
from medcalc_engine import (
    age_to_months,
    as_float,
    bedside_schwartz,
    bsa_mosteller,
    calculate_exposure_mgkg,
    calculate_pediatric_dose,
    ckdepi_2021,
    cockcroft_gault,
    normalize_crcl_to_173,
    quantity_to_ml,
    renal_biblio_band,
    rule_applies_demographics,
    select_renal_rule,
)

APP_VERSION = "V7.1 SUPABASE BETA"
REVIEW_DATE = "2026-09-03"
ROOT = Path(__file__).parent
FALLBACK_DB_PATH = ROOT / "medcalc.db"
CITUC_URL = "https://cituc.uc.cl/"
PAGES = ["Inicio", "Dosis pediátrica", "Ajuste renal", "Toxicología", "Base y fuentes"]

st.set_page_config(
    page_title="MedCalc Clínico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid #e6edf2;}
      .medcalc-kicker {font-size:.76rem; letter-spacing:.08em; text-transform:uppercase; color:#60717c; font-weight:700;}
      .medcalc-title {font-size:2rem; font-weight:800; line-height:1.15; margin:.2rem 0 .3rem; color:#17212b;}
      .medcalc-subtitle {color:#5f6f7a; margin-bottom:1rem;}
      .safe-card {border:1px solid #dbe5eb; border-radius:14px; padding:16px 18px; background:#fbfdfe; margin:.4rem 0 1rem;}
      .result-box {border-left:5px solid #176B87; background:#f5fafc; border-radius:10px; padding:14px 16px; margin:.6rem 0;}
      .status-ok {display:inline-block; padding:3px 9px; border-radius:999px; background:#eaf6ee; color:#216e39; font-size:.78rem; font-weight:700;}
      .status-off {display:inline-block; padding:3px 9px; border-radius:999px; background:#fff3e8; color:#8a4b08; font-size:.78rem; font-weight:700;}
      .status-ref {display:inline-block; padding:3px 9px; border-radius:999px; background:#edf4ff; color:#244f8f; font-size:.78rem; font-weight:700;}
      div[data-testid="stMetric"] {border:1px solid #e5edf1; padding:12px 14px; border-radius:12px; background:#fff;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "Faltan los Secrets de Streamlit: SUPABASE_URL y SUPABASE_PUBLISHABLE_KEY."
        ) from exc
    return SupabaseRepository(url, key, fallback_db_path=FALLBACK_DB_PATH)


try:
    db = get_db()
except Exception as exc:
    st.error(
        "**No se pudo conectar MedCalc con Supabase.**\n\n"
        f"{exc}\n\n"
        "Verifique Streamlit Secrets, que el proyecto Supabase esté activo y que las políticas RLS permitan SELECT de registros PUBLISHED."
    )
    st.stop()

COUNTS = db.counts()


def fmt_num(value, digits=1):
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_range(lo, hi, unit="mg"):
    if lo is None or hi is None:
        return "—"
    if abs(float(lo) - float(hi)) < 1e-9:
        return f"{fmt_num(lo, 2)} {unit}"
    return f"{fmt_num(lo, 2)}–{fmt_num(hi, 2)} {unit}"


def header(title, subtitle):
    st.markdown(f'<div class="medcalc-kicker">MEDCALC CLÍNICO · {APP_VERSION}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="medcalc-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="medcalc-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def source_block(source, url, revision=None):
    c1, c2 = st.columns([3, 1])
    c1.caption(f"Fuente: {source or 'No consignada'}" + (f" · revisión {revision}" if revision else ""))
    if url:
        c2.link_button("Abrir fuente", url, use_container_width=True)


def navigate(target, med_id=None):
    st.session_state["nav_page"] = target
    if med_id:
        st.session_state["selected_med_id"] = med_id


def medication_picker(prefix, title="Medicamento", help_text=None):
    """Explicit search + all 618 medication selector."""
    query = st.text_input(
        f"Buscar {title.lower()}",
        placeholder="Escriba parte del nombre, por ejemplo: amoxi, aciclovir, gabapentina…",
        key=f"{prefix}_med_query",
    )
    hits = db.search_medications(query, limit=618)
    if not hits:
        st.warning("No hay coincidencias en el catálogo maestro.")
        return None

    labels = [f"{r['principio_activo']} · {r['med_id']}" for r in hits]
    preferred = st.session_state.get("selected_med_id")
    index = 0
    if preferred:
        for i, r in enumerate(hits):
            if r["med_id"] == preferred:
                index = i
                break

    picked = st.selectbox(
        title,
        labels,
        index=index,
        key=f"{prefix}_med_select",
        help=help_text or "El selector proviene de la tabla maestra SQL de 618 medicamentos.",
    )
    row = hits[labels.index(picked)]
    st.session_state["selected_med_id"] = row["med_id"]
    return row


def status_badges(summary):
    c1, c2, c3 = st.columns(3)
    ped_n = int(summary.get("pediatric_rule_count") or 0)
    ren_n = int(summary.get("renal_rule_count") or 0)
    ref_n = int(summary.get("renal_biblio_count") or 0)
    tox = int(summary.get("toxicology_available") or 0)
    c1.markdown(
        f'<span class="status-{"ok" if ped_n else "off"}">PEDIATRÍA · {ped_n} reglas</span>',
        unsafe_allow_html=True,
    )
    renal_text = f"RENAL · {ren_n} auto" + (f" + {ref_n} ref." if ref_n else "")
    c2.markdown(
        f'<span class="status-{"ok" if (ren_n or ref_n) else "off"}">{renal_text}</span>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<span class="status-{"ok" if tox else "off"}">TOXICOLOGÍA · {"disponible" if tox else "sin ficha"}</span>',
        unsafe_allow_html=True,
    )


def resolve_renal_image(image_value=None, table_num=None):
    candidates = []
    if image_value:
        name = Path(str(image_value)).name
        candidates += [ROOT / str(image_value), ROOT / name, ROOT / "renal_fuente_2025" / name]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    if table_num is not None:
        matches = sorted((ROOT / "renal_fuente_2025").glob(f"tabla_{int(table_num):02d}_pag_*.png")) if (ROOT / "renal_fuente_2025").exists() else []
        if not matches:
            matches = sorted(ROOT.glob(f"tabla_{int(table_num):02d}_pag_*.png"))
        if matches:
            return matches[0]
    return None


def page_home():
    header("MedCalc Clínico", "Un único medicamento central enlazado a pediatría, ajuste renal y toxicología.")
    st.markdown(
        """<div class="safe-card"><strong>Arquitectura V7:</strong> la tabla SQL <code>medications</code> contiene los 618 MED-ID. Pediatría, renal y toxicología se relacionan con ese mismo identificador. Esto evita catálogos recortados por módulo.</div>""",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Catálogo maestro", f"{COUNTS['medications']} medicamentos")
    m2.metric("Pediatría", f"{COUNTS['pediatric_rules']} reglas · {COUNTS['pediatric_meds']} fármacos")
    m3.metric("Renal", f"{COUNTS['renal_rules']} auto · {COUNTS['renal_biblio']} ref.")
    m4.metric("Toxicología", f"{COUNTS['toxicology']} fichas")

    st.subheader("Buscador clínico global")
    med = medication_picker("home", "Medicamento")
    if not med:
        return
    summary = db.medication(med["med_id"])
    st.markdown(f"### {summary['principio_activo']} · {summary['med_id']}")
    status_badges(summary)

    ped_inds = db.pediatric_indications(summary["med_id"])
    renal_inds = db.renal_indications(summary["med_id"])
    renal_refs = db.renal_biblio(summary["med_id"])
    tox = db.toxicology(summary["med_id"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Indicaciones pediátricas cargadas")
        if ped_inds:
            for r in ped_inds:
                st.write(f"• **{r['indicacion']}** · vía(s): {r.get('vias') or '—'}")
        else:
            st.caption("Sin pauta pediátrica automatizada revisada.")
    with c2:
        st.markdown("#### Escenarios de ajuste renal")
        if renal_inds:
            for r in renal_inds:
                st.write(f"• **{r['indicacion']}** · vía(s): {r.get('vias') or '—'}")
        if renal_refs:
            st.caption(f"Además: {len(renal_refs)} referencia(s) de Nefrología al Día 2025 enlazadas al MED-ID.")
        if not renal_inds and not renal_refs:
            st.caption("Sin regla renal enlazada actualmente.")

    if tox:
        st.markdown("#### Toxicología")
        st.write(f"**Dosis registrada en la base bibliográfica:** {tox.get('dosis_toxica_base') or '—'}")
        st.caption(f"Estado de revisión: {tox.get('estado_revision') or '—'} · Clase: {tox.get('clase_toxicologica') or '—'}")

    st.markdown("#### Abrir módulo con este medicamento")
    b1, b2, b3 = st.columns(3)
    b1.button(
        "👶 Ir a Pediatría",
        use_container_width=True,
        disabled=not bool(ped_inds),
        on_click=navigate,
        args=("Dosis pediátrica", summary["med_id"]),
    )
    b2.button(
        "🧮 Ir a Ajuste renal",
        use_container_width=True,
        disabled=not bool(renal_inds or renal_refs),
        on_click=navigate,
        args=("Ajuste renal", summary["med_id"]),
    )
    b3.button(
        "☠️ Ir a Toxicología",
        use_container_width=True,
        disabled=not bool(tox),
        on_click=navigate,
        args=("Toxicología", summary["med_id"]),
    )


def page_pediatric():
    header("Dosis pediátrica", "Buscador explícito sobre los 618 medicamentos y cálculo por indicación/escenario.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Catálogo Supabase", f"{COUNTS['medications']} medicamentos")
    m2.metric("Con pauta cargada", f"{COUNTS['pediatric_meds']} medicamentos")
    m3.metric("Reglas pediátricas", f"{COUNTS['pediatric_rules']} reglas")

    med = medication_picker("ped", "Medicamento")
    if not med:
        return
    rules = db.pediatric_rules(med["med_id"])
    auto_rules = [r for r in rules if r.get("automatizable") == "SI"]
    if not auto_rules:
        st.warning(
            f"**{med['principio_activo']}: PENDIENTE DE REVISIÓN PEDIÁTRICA.** El fármaco permanece en el catálogo de 618, pero no se extrapola una pauta hasta disponer de indicación, edad, formulación y fuente revisadas."
        )
        return

    st.success(f"{med['principio_activo']}: {len(auto_rules)} regla(s) pediátrica(s) cargada(s).")
    indications = sorted({r["indicacion"] for r in auto_rules})
    indication = st.selectbox("Indicación / escenario", indications, key="ped_indication_sql")
    irules = [r for r in auto_rules if r["indicacion"] == indication]
    routes = sorted({r["via"] for r in irules})
    route = st.selectbox("Vía", routes, key="ped_route_sql")
    candidates = [r for r in irules if r["via"] == route]

    with st.form("ped_patient_sql", border=True):
        c1, c2, c3 = st.columns(3)
        age_value = c1.number_input("Edad", min_value=0.0, max_value=216.0, value=5.0, step=0.5)
        age_unit = c1.selectbox("Unidad", ["años", "meses", "días"])
        weight = c2.number_input("Peso (kg)", min_value=0.1, max_value=250.0, value=20.0, step=0.1)
        c3.write(f"**Escenario:** {indication}")
        c3.write(f"**Vía:** {route}")
        submitted = st.form_submit_button("Calcular dosis", type="primary", use_container_width=True)

    if submitted:
        age_mo = age_to_months(age_value, age_unit)
        applicable = [r for r in candidates if rule_applies_demographics(r, age_mo, weight)]
        if not applicable:
            st.error("No existe una regla validada compatible con esa edad/peso para el escenario seleccionado.")
            return
        st.session_state["ped_sql_result"] = {
            "med_id": med["med_id"], "rule_ids": [r["rule_id"] for r in applicable], "weight": weight,
            "age_value": age_value, "age_unit": age_unit, "indication": indication, "route": route,
        }
        st.session_state.pop("ped_sql_volume", None)

    snap = st.session_state.get("ped_sql_result")
    if not snap or snap.get("med_id") != med["med_id"] or snap.get("indication") != indication or snap.get("route") != route:
        st.caption("Complete los datos y pulse **Calcular dosis**.")
        return

    selected_rules = [r for r in rules if r["rule_id"] in snap["rule_ids"]]
    if len(selected_rules) > 1:
        labels = [f"{r['poblacion']} · {r['rule_id']}" for r in selected_rules]
        label = st.selectbox("Regla compatible", labels, key="ped_rule_sql")
        rule = selected_rules[labels.index(label)]
    else:
        rule = selected_rules[0]

    result = calculate_pediatric_dose(rule, snap["weight"])
    unit = result["unit"]
    st.markdown(f"### {rule['principio_activo']} · {rule['indicacion']}")
    st.markdown(f'<div class="result-box"><strong>{rule["rule_id"]}</strong> · {rule["poblacion"]} · {rule["via"]}</div>', unsafe_allow_html=True)
    if (rule.get("nivel_uso") or "GENERAL") != "GENERAL":
        st.warning(f"Nivel de uso: **{rule.get('nivel_uso')}**")

    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Dosis por administración", fmt_range(result["min_value"], result["max_value"], unit))
    x2.metric("Intervalo", f"cada {fmt_num(result['interval_h'],1)} h" if result.get("interval_h") else (rule.get("frecuencia_texto") or "Según regla"))
    x3.metric("Dosis diaria", fmt_range(result["daily_min_value"], result["daily_max_value"], f"{unit}/día") if result.get("daily_min_value") is not None else "Según frecuencia")
    x4.metric("Máximo por dosis", f"{fmt_num(result['max_single_value'],2)} {unit}" if result.get("max_single_value") is not None else "No cargado")
    st.caption(f"Trazabilidad: {result['formula']}")
    if rule.get("frecuencia_texto"): st.info(rule["frecuencia_texto"])
    if rule.get("duracion"): st.info(f"Duración: {rule['duracion']}")
    if rule.get("notas"): st.info(rule["notas"])
    if rule.get("nota_renal"): st.warning("Función renal: " + rule["nota_renal"])

    if str(rule.get("permite_conversion_volumen") or "SI").upper() == "SI":
        st.subheader("Conversión a volumen")
        with st.form("ped_volume_sql", border=True):
            q1, q2 = st.columns(2)
            default_amount = 100000.0 if unit.upper().startswith("U") else 100.0
            label_value = q1.number_input(f"Cantidad de fármaco ({unit})", min_value=0.0001, value=default_amount, step=1.0)
            label_ml = q2.number_input("Volumen correspondiente (mL)", min_value=0.01, value=5.0, step=0.5)
            cv = st.form_submit_button(f"Convertir {unit} → mL", use_container_width=True)
        if cv:
            st.session_state["ped_sql_volume"] = quantity_to_ml(result["min_value"], result["max_value"], label_value, label_ml)
        vol = st.session_state.get("ped_sql_volume")
        if vol:
            v1, v2 = st.columns(2)
            v1.metric("Concentración", f"{fmt_num(vol['unit_per_ml'],3)} {unit}/mL")
            v2.metric("Volumen por dosis", fmt_range(vol["min_ml"], vol["max_ml"], "mL"))

    source_block(rule.get("fuente"), rule.get("url_fuente"), rule.get("fecha_revision"))


def page_renal():
    header("Ajuste renal", "Los 618 medicamentos permanecen visibles; el SQL indica cuáles tienen ajuste automático y/o referencia renal.")
    med = medication_picker("renal", "Medicamento")
    if not med:
        return
    summary = db.medication(med["med_id"])
    auto_rules = [r for r in db.renal_rules(med["med_id"]) if r.get("automatizable") == "SI"]
    refs = db.renal_biblio(med["med_id"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Catálogo Supabase", f"{COUNTS['medications']} medicamentos")
    c2.metric("Reglas automáticas del fármaco", len(auto_rules))
    c3.metric("Referencias renales enlazadas", len(refs))
    if not auto_rules and not refs:
        st.warning("Este medicamento está en el catálogo maestro, pero todavía no tiene una regla renal enlazada. No se extrapola un ajuste desde otro fármaco.")

    with st.form("renal_patient_sql", border=True):
        a, b, c = st.columns(3)
        age = a.number_input("Edad (años)", 1, 120, 60, 1)
        sex = a.selectbox("Sexo para ecuación", ["Hombre", "Mujer"])
        creat = b.number_input("Creatinina sérica (mg/dL)", 0.1, 20.0, 1.0, 0.1)
        weight = b.number_input("Peso para Cockcroft–Gault (kg)", 1.0, 300.0, 70.0, 0.5)
        height = c.number_input("Talla (cm)", 30.0, 230.0, 170.0, 1.0)
        hd = c.checkbox("Hemodiálisis")
        calc = st.form_submit_button("Calcular función renal", type="primary", use_container_width=True)
    if calc:
        crcl = cockcroft_gault(age, sex, weight, creat)
        bsa = bsa_mosteller(height, weight)
        st.session_state["renal_sql_snapshot"] = {
            "age":age,"sex":sex,"creat":creat,"weight":weight,"height":height,"hd":hd,
            "crcl":crcl,"crcl_norm":normalize_crcl_to_173(crcl,bsa),"bsa":bsa,
            "egfr":ckdepi_2021(age,sex,creat) if age>=18 else None,
            "schwartz":bedside_schwartz(height,creat) if age<18 else None,
        }
    snap = st.session_state.get("renal_sql_snapshot")
    if snap:
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Cockcroft–Gault", f"{fmt_num(snap['crcl'],1)} mL/min")
        m2.metric("CrCl normalizado", f"{fmt_num(snap['crcl_norm'],1)} mL/min/1,73 m²")
        m3.metric("CKD-EPI 2021", f"{fmt_num(snap['egfr'],1)} mL/min/1,73 m²" if snap['egfr'] is not None else "Solo ≥18 años")
        m4.metric("Schwartz bedside", f"{fmt_num(snap['schwartz'],1)} mL/min/1,73 m²" if snap['schwartz'] is not None else "Solo <18 años")
        m5.metric("SC Mosteller", f"{fmt_num(snap['bsa'],2)} m²")

    tab1, tab2 = st.tabs(["Ajuste automatizado", "Bibliografía renal 2025"])
    with tab1:
        if not auto_rules:
            st.info("No hay ajuste renal automático validado para este MED-ID.")
        else:
            indications = sorted({r["indicacion"] for r in auto_rules})
            ind = st.selectbox("Indicación / régimen basal", indications, key="renal_ind_sql")
            irules = [r for r in auto_rules if r["indicacion"] == ind]
            if not snap:
                st.caption("Calcule primero la función renal para seleccionar la banda correspondiente.")
            elif snap["age"] < 18:
                st.error("Las reglas automáticas actuales son principalmente adultas. No se extrapolan a pediatría.")
            else:
                if st.button("Obtener ajuste", type="primary", use_container_width=True):
                    selected, value_used = select_renal_rule(
                        irules, snap["crcl"], snap["crcl_norm"], snap["egfr"], snap["hd"]
                    )
                    if selected:
                        st.success(selected.get("regimen_ajustado") or "—")
                        r1,r2,r3=st.columns(3)
                        r1.metric("Banda", selected.get("rango") or "—")
                        r2.metric("Métrica", selected.get("metrica_renal") or "—")
                        r3.metric("Valor usado", fmt_num(value_used,1) if value_used is not None else "Regla HD/no numérica")
                        if selected.get("notas"): st.info(selected["notas"])
                        source_block(selected.get("fuente"),selected.get("url_fuente"),selected.get("fecha_revision"))
                    else:
                        st.error("No existe una banda validada compatible con ese valor.")

    with tab2:
        if refs:
            labels=[f"Tabla {r['table']} · pág. {r['page']} · {r['principio_activo']}" for r in refs]
            pick=st.selectbox("Referencia enlazada al medicamento",labels,key="renal_ref_sql")
            ref=refs[labels.index(pick)]
            st.write(f"**Dosis con función renal normal:** {ref.get('dosis_fr_normal') or '—'}")
            st.write(f"**Método:** {ref.get('metodo') or '—'}")
            if snap and snap["age"]>=18:
                band=renal_biblio_band(snap["crcl"])
                st.success(ref.get(band) or "—")
                st.caption(f"Columna seleccionada por Cockcroft–Gault {fmt_num(snap['crcl'],1)} mL/min; se reproduce el contenido de la fuente, no se reinterpreta.")
            with st.expander("Ver todas las bandas"):
                st.write(f"100–50: {ref.get('crcl_100_50') or '—'}")
                st.write(f"50–10: {ref.get('crcl_50_10') or '—'}")
                st.write(f"<10: {ref.get('crcl_lt10') or '—'}")
                st.write(f"HD: {ref.get('suplemento_hd') or '—'}")
            img=resolve_renal_image(ref.get("imagen"),ref.get("table"))
            if img:
                with st.expander("Ver tabla original"):
                    st.image(str(img),use_container_width=True)
            source_block(ref.get("fuente"),ref.get("url_fuente"),ref.get("fecha_fuente"))
        else:
            st.info("No hay una referencia renal 2025 enlazada a este MED-ID.")
            with st.expander("Buscar manualmente en las referencias no enlazadas"):
                q=st.text_input("Buscar nombre en Nefrología al Día",value=med["principio_activo"],key="renal_manual_ref_q")
                mhits=db.search_renal_biblio(q)
                if mhits:
                    for r in mhits[:10]:
                        st.write(f"• {r['principio_activo']} · Tabla {r['table']} · pág. {r['page']} · {r.get('dosis_fr_normal') or '—'}")
                else:
                    st.caption("Sin coincidencias en las 127 filas verificadas.")


def page_toxicology():
    header("Toxicología", "Buscador Supabase del catálogo farmacológico completo y conservación de la dosis bibliográfica original.")
    tab1, tab2, tab3 = st.tabs(["Medicamentos", "Drogas/plaguicidas/metales", "Antídotos"])
    with tab1:
        med=medication_picker("tox","Medicamento")
        if not med: return
        tox=db.toxicology(med["med_id"])
        if not tox:
            st.warning("Sin ficha toxicológica enlazada.")
            return
        st.markdown(f"### {med['principio_activo']} · {med['med_id']}")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("#### 📚 Base bibliográfica original")
            st.write(f"**Dosis tóxica registrada:** {tox.get('dosis_toxica_base') or '—'}")
            st.write(f"**Síntomas:** {tox.get('sintomas_base') or '—'}")
            st.write(f"**Manejo/antídoto original:** {tox.get('antidoto_manejo_base') or '—'}")
        with c2:
            st.markdown("#### ✅ Capa revisada")
            st.write(f"**Criterio:** {tox.get('dosis_toxica_corregida') or '—'}")
            st.write(f"**Manifestaciones:** {tox.get('manifestaciones_clave') or '—'}")
            st.write(f"**Manejo:** {tox.get('manejo_corregido') or '—'}")
            st.write(f"**Terapia específica:** {tox.get('antidoto_especifico') or '—'}")
        st.caption(f"Estado: {tox.get('estado_revision') or '—'} · Nivel de evidencia: {tox.get('nivel_evidencia') or '—'}")
        threshold=as_float(tox.get("umbral_mgkg_automatizable"))
        if str(tox.get("permitir_comparacion_automatica") or "").upper()=="SI" and threshold is not None:
            st.subheader("Calculadora de exposición")
            with st.form("tox_exp_sql"):
                a,b=st.columns(2)
                total=a.number_input("Cantidad total ingerida (mg)",min_value=0.0,value=500.0,step=50.0)
                weight=b.number_input("Peso (kg)",min_value=0.1,value=20.0,step=0.1)
                submit=st.form_submit_button("Calcular mg/kg",use_container_width=True)
            if submit:
                exposure, ratio = calculate_exposure_mgkg(total,weight,threshold)
                st.metric("Exposición",f"{fmt_num(exposure,2)} mg/kg")
                if ratio is not None:
                    st.caption(f"Relación con la referencia cargada: {fmt_num(ratio,2)}×. Este cociente no sustituye la evaluación toxicológica.")
                st.info(tox.get("etiqueta_umbral") or f"Referencia cargada: {threshold:g} mg/kg")
        if tox.get("fuente_principal"):
            st.link_button("Abrir fuente principal",tox["fuente_principal"])

    with tab2:
        q=st.text_input("Buscar tóxico no farmacológico",key="other_tox_q")
        hits=db.search_other_tox(q)
        if hits:
            names=[r.get("toxico") or "—" for r in hits]
            pick=st.selectbox("Tóxico",names,key="other_tox_sel")
            r=hits[names.index(pick)]
            st.write(f"**Síntomas:** {r.get('sintomas_base') or '—'}")
            st.write(f"**Antídoto/tratamiento:** {r.get('antidoto_tratamiento_base') or '—'}")
            st.warning(f"Estado: {r.get('estado_validacion') or '—'}")
    with tab3:
        q=st.text_input("Buscar tóxico, síndrome o antídoto",key="antidote_q")
        hits=db.search_antidotes(q)
        if hits:
            labels=[f"{r.get('toxico_sindrome') or '—'} → {r.get('antidoto_base') or '—'}" for r in hits]
            pick=st.selectbox("Resultado",labels,key="antidote_sel")
            r=hits[labels.index(pick)]
            st.write(f"**Dosis registrada:** {r.get('dosis_base') or '—'}")
            st.write(f"**Observaciones:** {r.get('observaciones_base') or '—'}")
            st.warning(f"Estado: {r.get('estado_validacion') or '—'}")


def page_sources():
    header("Base clínica y fuentes", "Estructura SQL, cobertura y trazabilidad.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("MED-ID",COUNTS["medications"])
    c2.metric("Pediatría",COUNTS["pediatric_rules"])
    c3.metric("Renal automático",COUNTS["renal_rules"])
    c4.metric("Toxicología",COUNTS["toxicology"])
    st.markdown("#### Base Supabase")
    st.code("medications 1 ─── N pediatric_rules\nmedications 1 ─── N renal_rules\nmedications 1 ─── N renal_bibliography\nmedications 1 ─── 1 toxicology")
    st.caption(f"Schema Supabase: {SCHEMA_VERSION} · Datos: {db.metadata('data_version') or 'sin versión'}")
    st.markdown("#### Fuentes")
    for r in db.sources():
        with st.expander(r.get("fuente") or r.get("codigo") or "Fuente"):
            st.write(f"**Código:** {r.get('codigo') or '—'}")
            st.write(f"**Revisión:** {r.get('fecha_revision') or '—'}")
            if r.get("url"): st.link_button("Abrir fuente",r["url"])
    st.success("El catálogo clínico principal se consulta desde PostgreSQL/Supabase con RLS. `medcalc.db` permanece temporalmente solo como respaldo de los submódulos auxiliares de tóxicos no farmacológicos y antídotos, pendientes de migración completa.")


if "nav_page" not in st.session_state:
    st.session_state["nav_page"]="Inicio"

with st.sidebar:
    st.markdown("## 🩺 MedCalc Clínico")
    st.caption(APP_VERSION)
    page=st.radio("Navegación",PAGES,key="nav_page")
    st.divider()
    st.caption(f"Supabase schema {SCHEMA_VERSION} · {COUNTS['medications']} MED-ID")
    st.caption("Herramienta clínica en desarrollo. No usar como única fuente para prescripción o manejo toxicológico.")
    st.link_button("CITUC Chile",CITUC_URL,use_container_width=True)

if page=="Inicio": page_home()
elif page=="Dosis pediátrica": page_pediatric()
elif page=="Ajuste renal": page_renal()
elif page=="Toxicología": page_toxicology()
else: page_sources()

st.divider()
st.caption(f"MedCalc Clínico {APP_VERSION} · Base Supabase {SCHEMA_VERSION} · revisión {REVIEW_DATE} · apoyo clínico, no sustituto del juicio profesional.")
