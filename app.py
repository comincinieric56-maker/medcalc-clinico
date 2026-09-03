from pathlib import Path

import streamlit as st

from medcalc_engine import (
    age_to_months,
    as_float,
    bedside_schwartz,
    bsa_mosteller,
    calculate_exposure_mgkg,
    calculate_pediatric_dose,
    ckdepi_2021,
    cockcroft_gault,
    mg_to_ml,
    normalize_crcl_to_173,
    renal_biblio_band,
    rule_applies_demographics,
    select_renal_rule,
)
from repository import Repository

APP_VERSION = "V5.5.2 BETA · RENAL 2025"
REVIEW_DATE = "2026-09-03"
ROOT = Path(__file__).parent
BASE = ROOT / "data" if (ROOT / "data").exists() else ROOT
CITUC_URL = "https://cituc.uc.cl/"

st.set_page_config(
    page_title="MedCalc Clínico",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stSidebar"] {border-right: 1px solid #e6edf2;}
      .medcalc-kicker {font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; color:#5f6f7a; font-weight:700;}
      .medcalc-title {font-size:2rem; font-weight:800; line-height:1.15; margin:.2rem 0 .3rem 0; color:#17212b;}
      .medcalc-subtitle {color:#5f6f7a; margin-bottom:1rem;}
      .safe-card {border:1px solid #dbe5eb; border-radius:14px; padding:16px 18px; background:#fbfdfe; margin:.4rem 0 1rem 0;}
      .safe-card strong {color:#17212b;}
      .result-box {border-left:5px solid #176B87; background:#f5fafc; border-radius:10px; padding:14px 16px; margin:.6rem 0;}
      .status-ok {display:inline-block; padding:3px 9px; border-radius:999px; background:#eaf6ee; color:#216e39; font-size:.78rem; font-weight:700;}
      .status-off {display:inline-block; padding:3px 9px; border-radius:999px; background:#fff3e8; color:#8a4b08; font-size:.78rem; font-weight:700;}
      .tiny {font-size:.8rem; color:#667680;}
      div[data-testid="stMetric"] {border:1px solid #e5edf1; padding:12px 14px; border-radius:12px; background:#ffffff;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_repo():
    return Repository(BASE)


repo = get_repo()


def fmt_num(value, digits=1):
    if value is None:
        return "—"
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_dose_range(lo, hi, unit="mg"):
    if lo is None or hi is None:
        return "—"
    if abs(lo - hi) < 1e-9:
        return f"{fmt_num(lo, 2)} {unit}"
    return f"{fmt_num(lo, 2)}–{fmt_num(hi, 2)} {unit}"


def header(title, subtitle):
    st.markdown(f'<div class="medcalc-kicker">MEDCALC CLÍNICO · {APP_VERSION}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="medcalc-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="medcalc-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def source_block(source, url, revision=None):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"Fuente: {source or 'No consignada'}" + (f" · revisión {revision}" if revision else ""))
    with c2:
        if url:
            st.link_button("Abrir fuente", url, use_container_width=True)

def resolve_renal_source_image(image_value=None, table_num=None):
    """Locate a renal bibliography image whether GitHub stores it in root or renal_fuente_2025/."""
    candidates = []

    if image_value:
        raw = Path(str(image_value))
        name = raw.name
        candidates.extend([
            BASE / raw,
            ROOT / raw,
            BASE / name,
            ROOT / name,
            BASE / "renal_fuente_2025" / name,
            ROOT / "renal_fuente_2025" / name,
        ])

    seen = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file():
            return path

    if table_num is not None:
        pattern = f"tabla_{int(table_num):02d}_pag_*.png"
        folders = [
            BASE / "renal_fuente_2025",
            ROOT / "renal_fuente_2025",
            BASE,
            ROOT,
        ]
        checked = set()
        for folder in folders:
            folder_key = str(folder)
            if folder_key in checked:
                continue
            checked.add(folder_key)
            if folder.exists():
                matches = sorted(folder.glob(pattern))
                if matches:
                    return matches[0]

    return None


def tox_revision_badge(status):
    mapping = {
        "VALIDADO_ESPECIFICO": ("success", "✅ Validación específica"),
        "VALIDADO_POR_CLASE": ("info", "🔷 Validación por clase"),
        "CORREGIDO_ERROR_IMPORTANTE": ("error", "🛠️ Error importante corregido"),
        "NOMBRE_AMBIGUO_NO_AUTOMATIZAR": ("error", "⛔ Nombre ambiguo: no automatizar"),
        "NO_PRIORIZAR_COMO_TOXICO": ("warning", "ℹ️ No priorizar como tóxico primario"),
        "REVISADO_CONSERVADOR": ("warning", "🟡 Revisión conservadora / SDTE cuando corresponde"),
    }
    kind, text = mapping.get(status, ("warning", "🟡 Revisión conservadora"))
    getattr(st, kind)(text)




def is_sdte_text(value):
    text = (value or "").strip().upper()
    compact = text.replace(" ", "").replace("Ó", "O")
    return (
        not text
        or compact in {"SDTE", "STDE"}
        or text.startswith("SDTE —")
        or text.startswith("STDE —")
        or "SIN DOSIS TÓXICA ESPECÍFICA" in text
        or "SIN DOSIS TOXICA ESPECIFICA" in text
    )


def base_has_specific_toxic_dose(value):
    text = (value or "").strip().upper()
    if not text:
        return False
    # Registros del tipo SDTE/DOSIS MAX contienen una dosis terapéutica máxima,
    # no una dosis tóxica específica; se conservan como trazabilidad, pero no se
    # presentan como umbral toxicológico.
    if text.startswith("SDTE") or text.startswith("STDE"):
        return False
    return True


def page_home():
    header(
        "MedCalc Clínico",
        "Dosis pediátrica, ajuste renal y toxicología en una sola base clínica trazable.",
    )
    st.markdown(
        """
        <div class="safe-card"><strong>Uso previsto:</strong> herramienta de apoyo para personal sanitario. 
        Los resultados dependen de la indicación, edad, peso, vía, función renal y regla clínica cargada. 
        No sustituye protocolo institucional, ficha técnica local, toxicología clínica ni juicio médico.</div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Catálogo maestro", f"{len(repo.catalog)} medicamentos")
    m2.metric("Pediatría", f"{len(repo.ped_rules)} reglas")
    m3.metric("Renal", f"{len(repo.renal_rules)} auto · {len(repo.renal_biblio)} ref.")
    m4.metric("Toxicología", f"{len(repo.meds)} fichas")

    st.subheader("Buscador global")
    q = st.text_input("Medicamento", placeholder="Ej.: amoxicilina, aciclovir, paracetamol, gabapentina")
    hits = repo.search_catalog(q, limit=50)
    if q.strip() and not hits:
        st.warning("No hay coincidencias en el catálogo maestro.")
    elif hits:
        options = [f"{r['principio_activo']} · {r['med_id']}" for r in hits]
        picked = st.selectbox("Resultado", options)
        row = hits[options.index(picked)]
        tox = repo.tox_by_id.get(row["med_id"])

        st.markdown(f"### {row['principio_activo']}")
        c1, c2, c3 = st.columns(3)
        with c1:
            if int(row.get("reglas_pediatria") or 0) > 0:
                st.markdown(f'<span class="status-ok">PEDIATRÍA · {row["reglas_pediatria"]} reglas</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-off">PEDIATRÍA · no habilitado</span>', unsafe_allow_html=True)
        with c2:
            if int(row.get("reglas_renal") or 0) > 0:
                st.markdown(f'<span class="status-ok">RENAL · {row["reglas_renal"]} reglas</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-off">RENAL · no habilitado</span>', unsafe_allow_html=True)
        with c3:
            if tox:
                st.markdown('<span class="status-ok">TOXICOLOGÍA · disponible</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-off">TOXICOLOGÍA · sin ficha</span>', unsafe_allow_html=True)

        if tox:
            st.caption(f"Clase toxicológica: {tox.get('clase_toxicologica') or '—'} · Estado: {tox.get('estado_revision') or '—'}")
        st.info("Use el menú lateral para abrir el módulo correspondiente. La app no genera una pauta cuando el medicamento figura como NO HABILITADO.")

    st.divider()
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Principios de seguridad del motor")
        st.write(
            "• Una dosis pediátrica se selecciona por **medicamento + indicación + edad/peso + vía**.\n"
            "• El ajuste renal usa la **métrica exigida por cada regla**; no intercambia eGFR y Cockcroft–Gault de forma arbitraria.\n"
            "• En toxicología, **SDTE significa sin dosis tóxica específica** y no se convierte en un umbral inventado.\n"
            "• Toda regla automatizable conserva fuente y fecha de revisión."
        )
    with right:
        st.subheader("Toxicología Chile")
        st.write("Acceso rápido al Centro de Información Toxicológica de la Pontificia Universidad Católica de Chile.")
        st.link_button("Abrir CITUC", CITUC_URL, use_container_width=True)


def page_pediatric():
    header(
        "Dosis pediátrica",
        "Selecciona una pauta validada y calcula dosis por peso, máximos y conversión mg → mL.",
    )
    st.info(
        "La calculadora no usa una dosis genérica por medicamento. Primero exige una indicación/escenario y verifica que la regla sea compatible con edad y peso."
    )

    ped_drugs = sorted(repo.ped_by_drug)
    with st.form("ped_form", border=True):
        a, b, c = st.columns(3)
        with a:
            age_value = st.number_input("Edad", min_value=0.0, max_value=216.0, value=5.0, step=0.5)
            age_unit = st.selectbox("Unidad de edad", ["años", "meses", "días"])
        with b:
            weight = st.number_input("Peso (kg)", min_value=0.1, max_value=250.0, value=20.0, step=0.1)
            drug = st.selectbox("Medicamento", ped_drugs)
        with c:
            rules_for_drug = repo.ped_by_drug[drug]
            indications = sorted({r["indicacion"] for r in rules_for_drug})
            indication = st.selectbox("Indicación / escenario", indications)
            route_options = sorted({r["via"] for r in rules_for_drug if r["indicacion"] == indication})
            route = st.selectbox("Vía", route_options)
        submitted = st.form_submit_button("Calcular dosis", type="primary", use_container_width=True)

    # Streamlit vuelve a ejecutar todo el script al enviar el formulario de volumen.
    # Por eso persistimos los datos del último cálculo pediátrico válido.
    if submitted:
        age_mo = age_to_months(age_value, age_unit)
        candidates = [
            r for r in repo.ped_by_drug[drug]
            if r["indicacion"] == indication and r["via"] == route and r.get("automatizable") == "SI"
        ]
        applicable = [r for r in candidates if rule_applies_demographics(r, age_mo, weight)]

        # Un nuevo cálculo invalida cualquier conversión de volumen anterior.
        st.session_state.pop("ped_volume_snapshot", None)

        if not applicable:
            st.session_state.pop("ped_snapshot", None)
            st.error("No existe una regla automática validada para esta combinación de edad, peso, indicación y vía. No se extrapola una dosis.")
            with st.expander("Ver reglas disponibles para ese escenario"):
                for r in candidates:
                    st.write(f"• {r['rule_id']} · {r['poblacion']} · {r['via']} · {r.get('notas') or 'sin nota adicional'}")
            return

        st.session_state["ped_snapshot"] = {
            "age_value": age_value,
            "age_unit": age_unit,
            "age_mo": age_mo,
            "weight": weight,
            "drug": drug,
            "indication": indication,
            "route": route,
            "applicable_rule_ids": [r["rule_id"] for r in applicable],
        }
        # Restablecer selector si el nuevo escenario ya no contiene la regla previa.
        if st.session_state.get("ped_rule_choice") not in st.session_state["ped_snapshot"]["applicable_rule_ids"]:
            st.session_state["ped_rule_choice"] = st.session_state["ped_snapshot"]["applicable_rule_ids"][0]

    if "ped_snapshot" not in st.session_state:
        st.caption("Complete los datos y pulse **Calcular dosis**.")
        return

    snap = st.session_state["ped_snapshot"]
    applicable = []
    for rule_id in snap["applicable_rule_ids"]:
        for r in repo.ped_by_drug.get(snap["drug"], []):
            if r.get("rule_id") == rule_id:
                applicable.append(r)
                break

    if not applicable:
        st.session_state.pop("ped_snapshot", None)
        st.error("La regla previamente calculada ya no está disponible en la base cargada. Vuelva a calcular la dosis.")
        return

    if len(applicable) > 1:
        labels = {f"{r['poblacion']} · {r['rule_id']}": r["rule_id"] for r in applicable}
        current_id = st.session_state.get("ped_rule_choice", applicable[0]["rule_id"])
        current_label = next((label for label, rid in labels.items() if rid == current_id), list(labels)[0])
        chosen_label = st.selectbox(
            "Hay más de una regla compatible; seleccione",
            list(labels),
            index=list(labels).index(current_label),
            key="ped_rule_choice_label",
        )
        st.session_state["ped_rule_choice"] = labels[chosen_label]
        rule = next(r for r in applicable if r["rule_id"] == st.session_state["ped_rule_choice"])
    else:
        rule = applicable[0]
        st.session_state["ped_rule_choice"] = rule["rule_id"]

    try:
        result = calculate_pediatric_dose(rule, snap["weight"])
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown(f"### {rule['principio_activo']} · {rule['indicacion']}")
    st.markdown(
        f'<div class="result-box"><strong>Regla {rule["rule_id"]}</strong> · {rule["poblacion"]} · vía {rule["via"]}</div>',
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Dosis por administración", fmt_dose_range(result["min_mg"], result["max_mg"]))
    r2.metric("Intervalo", f"cada {fmt_num(result['interval_h'], 1)} h" if result["interval_h"] else "Dosis única / según regla")
    r3.metric("Dosis diaria", fmt_dose_range(result["daily_min_mg"], result["daily_max_mg"], "mg/día"))
    r4.metric("Máximo por dosis", f"{fmt_num(result['max_single_mg'], 0)} mg" if result["max_single_mg"] is not None else "No cargado")

    st.caption(
        f"Paciente calculado: {fmt_num(snap['weight'], 1)} kg · {fmt_num(snap['age_value'], 1)} {snap['age_unit']} · "
        f"Trazabilidad: {result['formula']}"
    )
    if result["daily_cap_labels"]:
        st.warning("Límites diarios de la regla: " + " · ".join(result["daily_cap_labels"]))
    if result["caps_applied"]:
        st.warning("Se aplicó automáticamente: " + " · ".join(result["caps_applied"]))
    if rule.get("duracion"):
        st.info(f"Duración consignada: {rule['duracion']}")
    if rule.get("notas"):
        st.info(rule["notas"])
    if rule.get("nota_renal"):
        st.warning("Función renal: " + rule["nota_renal"])

    st.subheader("Conversión a volumen")
    with st.form("ped_volume_form", border=True):
        x1, x2 = st.columns(2)
        label_mg = x1.number_input("Presentación: cantidad de fármaco (mg)", min_value=0.01, value=100.0, step=1.0)
        label_ml = x2.number_input("Presentación: volumen correspondiente (mL)", min_value=0.01, value=5.0, step=0.5)
        vol_submit = st.form_submit_button("Convertir mg → mL", use_container_width=True)

    if vol_submit:
        try:
            volume = mg_to_ml(result["min_mg"], result["max_mg"], label_mg, label_ml)
            st.session_state["ped_volume_snapshot"] = {
                "rule_id": rule["rule_id"],
                "min_mg": result["min_mg"],
                "max_mg": result["max_mg"],
                "label_mg": label_mg,
                "label_ml": label_ml,
                "volume": volume,
            }
        except ValueError as exc:
            st.session_state.pop("ped_volume_snapshot", None)
            st.error(str(exc))

    vol_snap = st.session_state.get("ped_volume_snapshot")
    if (
        vol_snap
        and vol_snap.get("rule_id") == rule["rule_id"]
        and abs(vol_snap.get("min_mg", -1) - result["min_mg"]) < 1e-9
        and abs(vol_snap.get("max_mg", -1) - result["max_mg"]) < 1e-9
    ):
        volume = vol_snap["volume"]
        v1, v2 = st.columns(2)
        v1.metric("Concentración ingresada", f"{fmt_num(volume['mg_per_ml'], 2)} mg/mL")
        v2.metric("Volumen por dosis", fmt_dose_range(volume["min_ml"], volume["max_ml"], "mL"))
        st.caption(
            f"Presentación utilizada: {fmt_num(vol_snap['label_mg'], 2)} mg en {fmt_num(vol_snap['label_ml'], 2)} mL. "
            "La app no asume automáticamente una concentración comercial."
        )

    source_block(rule.get("fuente"), rule.get("url_fuente"), rule.get("fecha_revision"))

def page_renal():
    header(
        "Ajuste renal",
        "Calcula función renal y consulta reglas automatizadas o la bibliografía renal Nefrología al Día 2025.",
    )
    st.warning(
        "Las reglas automáticas y la bibliografía estructurada son principalmente adultas. En lesión renal aguda, creatinina inestable, embarazo, extremos de masa muscular o diálisis no convencional, interprete las estimaciones con cautela."
    )

    with st.form("renal_patient_form", border=True):
        a, b, c = st.columns(3)
        with a:
            age = st.number_input("Edad (años)", min_value=1, max_value=120, value=60, step=1)
            sex = st.selectbox("Sexo para ecuación", ["Hombre", "Mujer"])
        with b:
            creat = st.number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
            weight = st.number_input("Peso usado para Cockcroft–Gault (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.5)
        with c:
            height = st.number_input("Talla (cm)", min_value=30.0, max_value=230.0, value=170.0, step=1.0)
            hemodialysis = st.checkbox("Paciente en hemodiálisis")
        patient_submit = st.form_submit_button("Calcular función renal", type="primary", use_container_width=True)

    if not patient_submit and "renal_snapshot" not in st.session_state:
        st.caption("Complete los datos y pulse **Calcular función renal**.")
        return

    if patient_submit:
        crcl = cockcroft_gault(age, sex, weight, creat)
        egfr = ckdepi_2021(age, sex, creat) if age >= 18 else None
        schwartz = bedside_schwartz(height, creat) if age < 18 else None
        bsa = bsa_mosteller(height, weight)
        crcl_norm = normalize_crcl_to_173(crcl, bsa)
        st.session_state["renal_snapshot"] = {
            "age": age,
            "sex": sex,
            "creat": creat,
            "weight": weight,
            "height": height,
            "hemodialysis": hemodialysis,
            "crcl": crcl,
            "egfr": egfr,
            "schwartz": schwartz,
            "bsa": bsa,
            "crcl_norm": crcl_norm,
        }

    snap = st.session_state["renal_snapshot"]
    crcl, egfr, schwartz, bsa, crcl_norm = snap["crcl"], snap["egfr"], snap["schwartz"], snap["bsa"], snap["crcl_norm"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cockcroft–Gault", f"{fmt_num(crcl, 1)} mL/min")
    m2.metric("CrCl normalizado", f"{fmt_num(crcl_norm, 1)} mL/min/1,73 m²")
    m3.metric("CKD-EPI 2021", f"{fmt_num(egfr, 1)} mL/min/1,73 m²" if egfr is not None else "Solo ≥18 años")
    m4.metric("Schwartz bedside", f"{fmt_num(schwartz, 1)} mL/min/1,73 m²" if schwartz is not None else "Solo <18 años")
    m5.metric("SC Mosteller", f"{fmt_num(bsa, 2)} m²")

    if snap["age"] < 18:
        st.error("La app muestra Schwartz, pero NO extrapola las bandas renales adultas a pediatría. Use una referencia renal pediátrica específica.")
        return

    tab_auto, tab_biblio = st.tabs(["Ajuste automatizado", "Bibliografía renal 2025"])

    with tab_auto:
        st.caption(f"{len(repo.renal_rules)} reglas estructuradas automatizables. Cada regla conserva su métrica renal y fuente propia.")
        renal_drugs = sorted(repo.renal_by_drug)
        with st.form("renal_drug_form", border=True):
            c1, c2 = st.columns(2)
            drug = c1.selectbox("Medicamento", renal_drugs, key="renal_auto_drug")
            rules = repo.renal_by_drug[drug]
            indication = c2.selectbox("Indicación / régimen basal", sorted({r["indicacion"] for r in rules}), key="renal_auto_ind")
            dose_submit = st.form_submit_button("Obtener ajuste", type="primary", use_container_width=True)

        if dose_submit:
            irules = [r for r in repo.renal_by_drug[drug] if r["indicacion"] == indication and r.get("automatizable") == "SI"]
            selected, value_used = select_renal_rule(
                irules,
                crcl=crcl,
                crcl_normalized=crcl_norm,
                egfr=egfr,
                hemodialysis=snap["hemodialysis"],
            )
            if not selected:
                st.error("No existe una banda renal validada para este valor/escenario dentro de la regla seleccionada. La app no extrapola.")
            else:
                st.markdown(f"### {selected['principio_activo']} · {selected['indicacion']}")
                if selected.get("tipo_regla") in {"CONTRAINDICACION", "PRECAUCION"}:
                    st.error(selected["regimen_ajustado"])
                else:
                    st.success(selected["regimen_ajustado"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Banda de la regla", selected.get("rango") or "—")
                c2.metric("Métrica usada", selected.get("metrica_renal") or "—")
                c3.metric("Valor usado", fmt_num(value_used, 1) if value_used is not None else "Hemodiálisis / regla no numérica")

                if selected.get("tipo_regla") in {"PORCENTAJE", "PROPORCIONAL", "FORMULACION"}:
                    st.warning("Esta regla requiere conocer el régimen basal o la formulación exacta; no debe interpretarse como una prescripción autónoma.")
                if selected.get("notas"):
                    st.info(selected["notas"])
                source_block(selected.get("fuente"), selected.get("url_fuente"), selected.get("fecha_revision"))

    with tab_biblio:
        st.info(
            "Fuente incorporada: **Nefrología al Día, FR-001, actualizado 24-05-2025**. "
            "Las celdas de las tablas se conservan como referencia bibliográfica. Esta capa NO reemplaza una ficha técnica ni convierte porcentajes/intervalos en una prescripción automática."
        )
        a, b, c = st.columns(3)
        a.metric("Filas verificadas", len(repo.renal_biblio))
        b.metric("Tablas fuente incluidas", 26)
        biblio_linked = sum(1 for r in repo.renal_biblio if r.get("med_id"))
        c.metric("Enlazadas a MED-ID", biblio_linked)

        q = st.text_input("Buscar en la bibliografía renal", placeholder="Ej.: aciclovir, ceftriaxona, fluconazol", key="renal_biblio_q")
        names = sorted(repo.renal_biblio_by_drug)
        if q.strip():
            nq = normalize_text(q)
            names = [n for n in names if nq in normalize_text(n)]
        if not names:
            st.warning("No hay coincidencias dentro de las filas ya transcritas y verificadas. Puede revisar las 26 tablas originales al final de esta pestaña.")
        else:
            selected_name = st.selectbox("Medicamento / combinación", names, key="renal_biblio_drug")
            refs = repo.renal_biblio_by_drug[selected_name]
            if len(refs) > 1:
                ref_labels = [f"Tabla {r['table']} · pág. {r['page']} · {r['categoria']}" for r in refs]
                pick = st.selectbox("Referencia", ref_labels, key="renal_biblio_ref")
                ref = refs[ref_labels.index(pick)]
            else:
                ref = refs[0]

            band = renal_biblio_band(crcl)
            band_labels = {
                "crcl_100_50": "100–50 mL/min",
                "crcl_50_10": "50–10 mL/min",
                "crcl_lt10": "<10 mL/min",
            }
            recommendation = ref.get(band) or "—"

            st.markdown(f"### {ref['principio_activo']}")
            if ref.get("med_id"):
                st.caption(f"Enlace con catálogo maestro: {ref['med_id']} · {ref.get('catalogo_nombre') or ref['principio_activo']}")
            else:
                st.caption("Referencia renal disponible, pero este nombre/combinación aún no está enlazado a un MED-ID de la base maestra.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Dosis F.R. normal (fuente)", ref.get("dosis_fr_normal") or "—")
            c2.metric("Método", ref.get("metodo") or "—")
            c3.metric("Banda según CrCl", band_labels.get(band, "—"))

            st.markdown("#### Ajuste de referencia para la banda calculada")
            st.success(recommendation)
            st.caption(f"Selección basada en Cockcroft–Gault: {fmt_num(crcl,1)} mL/min. La app selecciona la columna de la tabla; no reinterpreta su contenido.")

            if snap["hemodialysis"]:
                st.warning("Hemodiálisis — suplemento consignado en la fuente: " + (ref.get("suplemento_hd") or "No consignado"))
            if ref.get("dosis_hfvvc"):
                st.info("HFVVC consignada en la fuente: " + ref["dosis_hfvvc"])
            if ref.get("notas"):
                st.warning(ref["notas"])

            with st.expander("Ver las tres bandas y la dosis normal"):
                st.write(f"**Dosis F.R. normal:** {ref.get('dosis_fr_normal') or '—'}")
                st.write(f"**100–50 mL/min:** {ref.get('crcl_100_50') or '—'}")
                st.write(f"**50–10 mL/min:** {ref.get('crcl_50_10') or '—'}")
                st.write(f"**<10 mL/min:** {ref.get('crcl_lt10') or '—'}")
                st.write(f"**Suplemento HD:** {ref.get('suplemento_hd') or '—'}")
                st.write(f"**HFVVC:** {ref.get('dosis_hfvvc') or '—'}")

            source_block(ref.get("fuente"), ref.get("url_fuente"), ref.get("fecha_fuente"))
            image_path = resolve_renal_source_image(ref.get("imagen"), ref.get("table"))
            if image_path:
                with st.expander(f"Ver tabla original · Tabla {ref['table']} · página {ref['page']}"):
                    st.image(str(image_path), use_container_width=True)
            else:
                st.caption("Imagen original de la tabla no encontrada en el repositorio.")

        st.divider()
        with st.expander("Explorar las 26 tablas originales del PDF"):
            table_num = st.selectbox("Tabla", list(range(1, 27)), key="renal_source_table")
            table_rows = [r for r in repo.renal_ocr_index if str(r.get("table")) == str(table_num)]
            page_num = table_rows[0].get("page") if table_rows else "—"
            image_path = resolve_renal_source_image(table_num=table_num)
            st.caption(f"Tabla {table_num} · página {page_num}. La imagen es la fuente original; el índice OCR auxiliar NO se usa para calcular dosis.")
            if image_path:
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning("No se encontró la imagen de esta tabla en la raíz ni en renal_fuente_2025/.")


def page_toxicology():
    header(
        "Toxicología",
        "Fichas farmacológicas curadas, tóxicos no farmacológicos y base local de antídotos.",
    )
    st.link_button("CITUC Chile", CITUC_URL)

    n_specific = sum(r.get("estado_revision") == "VALIDADO_ESPECIFICO" for r in repo.meds)
    n_class = sum(r.get("estado_revision") == "VALIDADO_POR_CLASE" for r in repo.meds)
    n_conservative = sum(r.get("estado_revision") == "REVISADO_CONSERVADOR" for r in repo.meds)
    n_auto = sum(r.get("permitir_comparacion_automatica") == "SI" for r in repo.meds)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Medicamentos", len(repo.meds))
    s2.metric("Validación específica", n_specific)
    s3.metric("Validación por clase", n_class)
    s4.metric("Comparación mg/kg activa", n_auto)

    tab1, tab2, tab3, tab4 = st.tabs(["Medicamentos", "Otros tóxicos", "Antídotos", "Auditoría"])

    with tab1:
        q = st.text_input("Buscar medicamento", placeholder="Ej.: paracetamol, venlafaxina, verapamilo", key="tox_q")
        found = repo.search_meds(q)
        if not found:
            st.warning("Sin coincidencias.")
        else:
            labels = [f"{r['principio_activo']} · {r['id_revision']}" for r in found]
            chosen = st.selectbox(f"Coincidencias ({len(found)})", labels, key="tox_select")
            item = found[labels.index(chosen)]

            st.markdown(f"### {item['principio_activo']}")
            tox_revision_badge(item.get("estado_revision"))
            a, b, c, d = st.columns(4)
            a.metric("Clase", item.get("clase_toxicologica") or "—")
            b.metric("Tipo de umbral", item.get("tipo_umbral") or "—")
            c.metric("Revisión", item.get("fecha_revision") or "—")
            d.metric("Comparación automática", item.get("permitir_comparacion_automatica") or "NO")

            st.subheader("Dosis toxicológica")
            base_dose = (item.get("dosis_toxica_base") or "").strip()
            reviewed_dose = (item.get("dosis_toxica_corregida") or "").strip()

            if base_has_specific_toxic_dose(base_dose):
                st.markdown("**📚 Dosis registrada en la base bibliográfica original**")
                st.write(base_dose)
                if is_sdte_text(reviewed_dose):
                    st.info(
                        "Esta cifra se conserva porque estaba consignada en la base original. "
                        "La revisión externa de esta versión no la habilitó como umbral automático; "
                        "por eso se muestra como dato bibliográfico y no se usa por sí sola para decidir toxicidad."
                    )
                else:
                    st.markdown("**✅ Criterio toxicológico revisado para la app**")
                    st.write(reviewed_dose)
            else:
                st.markdown("**📚 Registro de la base bibliográfica original**")
                st.write(base_dose or "SDTE — sin dosis tóxica específica consignada")
                st.markdown("**Criterio toxicológico revisado para la app**")
                st.write(reviewed_dose or "—")

            if item.get("poblacion_umbral"):
                st.caption("Población/alcance del criterio automatizable: " + item["poblacion_umbral"])

            with st.expander("Cómo interpreta MedCalc estas dos capas"):
                st.write(
                    "• **Base bibliográfica original:** conserva la dosis que venía en el archivo fuente. "
                    "No se elimina aunque todavía no se haya documentado en la app la referencia bibliográfica exacta.\n"
                    "• **Criterio revisado/automatizable:** es la capa que puede activar comparaciones matemáticas. "
                    "Si figura SDTE aquí, la app no convierte automáticamente la cifra bibliográfica en una frontera diagnóstica.\n"
                    "• Un registro del tipo `SDTE / dosis máxima ...` se interpreta como SDTE; la dosis máxima terapéutica no se presenta como dosis tóxica."
                )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Manifestaciones clave")
                st.write(item.get("manifestaciones_clave") or "—")
            with c2:
                st.markdown("#### Manejo corregido")
                st.write(item.get("manejo_corregido") or "—")

            st.markdown("#### Antídoto / terapia específica")
            st.write(item.get("antidoto_especifico") or "—")
            if item.get("nota_revision"):
                st.info(item["nota_revision"])

            threshold = as_float(item.get("umbral_mgkg_automatizable"))
            with st.expander("Calculadora de exposición"):
                if item.get("permitir_comparacion_automatica") != "SI" or threshold is None:
                    st.warning("Esta ficha no permite comparación mg/kg automática. La dosis bibliográfica, cuando existe, se conserva visible pero no se transforma automáticamente en un umbral diagnóstico.")
                else:
                    st.caption(item.get("etiqueta_umbral") or "Referencia mg/kg")
                    with st.form(f"tox_exposure_{item['id_revision']}"):
                        x1, x2 = st.columns(2)
                        weight = x1.number_input("Peso (kg)", min_value=0.1, max_value=300.0, value=70.0, step=0.1)
                        total_mg = x2.number_input("Cantidad total ingerida (mg)", min_value=0.0, max_value=1000000.0, value=0.0, step=10.0)
                        tox_submit = st.form_submit_button("Calcular exposición", use_container_width=True)
                    if tox_submit:
                        exposure, ratio = calculate_exposure_mgkg(total_mg, weight, threshold)
                        x1, x2, x3 = st.columns(3)
                        x1.metric("Exposición", f"{fmt_num(exposure, 2)} mg/kg")
                        x2.metric("Referencia", f"{fmt_num(threshold, 2)} mg/kg")
                        x3.metric("Relación", f"{fmt_num(ratio, 2)}×" if ratio is not None else "—")
                        st.error("Esta relación es aritmética. No diagnostica intoxicación, no reemplaza clínica/tiempo/concentraciones y no autoriza alta.")

            st.markdown("#### Fuentes")
            fc1, fc2 = st.columns(2)
            if item.get("fuente_principal"):
                fc1.link_button("Fuente principal", item["fuente_principal"], use_container_width=True)
            if item.get("fuente_secundaria"):
                fc2.link_button("Fuente secundaria", item["fuente_secundaria"], use_container_width=True)

            with st.expander("Datos originales de la base (trazabilidad)"):
                st.write(f"**Presentación:** {item.get('concentracion') or '—'} {item.get('unidad_medida') or ''} · {item.get('unidad_referencia') or '—'}")
                st.write(f"**Dosis tóxica original:** {item.get('dosis_toxica_base') or '—'}")
                st.write("**Síntomas originales:**", item.get("sintomas_base") or "—")
                st.write("**Manejo/antídoto original:**", item.get("antidoto_manejo_base") or "—")

    with tab2:
        st.warning("Este bloque conserva la base local original y todavía no tiene auditoría registro por registro equivalente a la de medicamentos.")
        q = st.text_input("Buscar droga, plaguicida o metal", key="other_q")
        found = repo.search_other_tox(q)
        if found:
            labels = [r["toxico"] for r in found]
            selected = st.selectbox(f"Coincidencias ({len(found)})", labels, key="other_sel")
            item = found[labels.index(selected)]
            st.markdown(f"### {item['toxico']}")
            st.markdown("**Manifestaciones consignadas**")
            st.write(item.get("sintomas_base") or "—")
            st.markdown("**Antídoto / tratamiento consignado**")
            st.write(item.get("antidoto_tratamiento_base") or "—")
        else:
            st.warning("Sin coincidencias.")

    with tab3:
        st.warning("Las dosis de antídotos continúan identificadas como contenido de la base local hasta completar su auditoría específica.")
        q = st.text_input("Buscar antídoto, tóxico o síndrome", key="ant_q")
        found = repo.search_antidotes(q)
        if found:
            labels = [f"{r['toxico_sindrome']} — {r['antidoto_base']}" for r in found]
            selected = st.selectbox(f"Coincidencias ({len(found)})", labels, key="ant_sel")
            item = found[labels.index(selected)]
            st.markdown(f"### {item['toxico_sindrome']}")
            st.markdown("**Antídoto consignado**")
            st.write(item.get("antidoto_base") or "—")
            st.markdown("**Dosis consignada en base local**")
            st.write(item.get("dosis_base") or "—")
            st.markdown("**Observaciones**")
            st.write(item.get("observaciones_base") or "—")
        else:
            st.warning("Sin coincidencias.")

    with tab4:
        n_critical = sum(r.get("correccion_clinicamente_relevante") == "SI" for r in repo.meds)
        n_dupes = sum(r.get("duplicado_exacto_nombre") == "SI" for r in repo.meds)
        a, b, c, d = st.columns(4)
        a.metric("Total", len(repo.meds))
        b.metric("Revisión conservadora", n_conservative)
        c.metric("Correcciones relevantes", n_critical)
        d.metric("Duplicados marcados", n_dupes)
        st.write(
            "La política del módulo es conservadora: una dosis terapéutica máxima o una DL50 animal no se acepta como dosis tóxica humana. "
            "Cuando no existe un umbral defendible, se mantiene **SDTE — sin dosis tóxica específica**."
        )
        critical = [r["principio_activo"] for r in repo.meds if r.get("correccion_clinicamente_relevante") == "SI"]
        with st.expander(f"Ver {len(critical)} fármacos con corrección clínicamente relevante"):
            st.write(" · ".join(critical))


def page_sources():
    header(
        "Base clínica y fuentes",
        "Cobertura, versionado y trazabilidad de las reglas que utiliza la aplicación.",
    )
    st.subheader("Cobertura")
    c1, c2, c3 = st.columns(3)
    c1.metric("Medicamentos catálogo", len(repo.catalog))
    c2.metric("Medicamentos con pediatría", len(repo.ped_by_drug))
    c3.metric("Medicamentos con ajuste renal", len(repo.renal_by_drug))

    st.subheader("Fuentes de cálculo")
    for row in repo.sources:
        with st.expander(row.get("fuente") or row.get("codigo") or "Fuente"):
            st.write(f"**Código:** {row.get('codigo') or '—'}")
            st.write(f"**Fecha de revisión en MedCalc:** {row.get('fecha_revision') or '—'}")
            if row.get("url"):
                st.link_button("Abrir fuente", row["url"])

    st.subheader("Política de versionado")
    st.write(
        f"Versión de la interfaz: **{APP_VERSION}**. Fecha de revisión de las reglas incorporadas: **{REVIEW_DATE}**. "
        "Cada fila de pediatría y ajuste renal conserva su propia fuente y fecha de revisión. "
        "Los medicamentos sin regla validada permanecen en el catálogo, pero la app los muestra como **NO HABILITADOS** para cálculo."
    )


with st.sidebar:
    st.markdown("## 🩺 MedCalc Clínico")
    st.caption(APP_VERSION)
    page = st.radio(
        "Navegación",
        ["Inicio", "Dosis pediátrica", "Ajuste renal", "Toxicología", "Base y fuentes"],
        index=0,
    )
    st.divider()
    st.caption("Herramienta clínica en desarrollo. No usar como única fuente para prescripción o decisiones toxicológicas.")
    st.link_button("CITUC Chile", CITUC_URL, use_container_width=True)

if page == "Inicio":
    page_home()
elif page == "Dosis pediátrica":
    page_pediatric()
elif page == "Ajuste renal":
    page_renal()
elif page == "Toxicología":
    page_toxicology()
else:
    page_sources()

st.divider()
st.caption(f"MedCalc Clínico {APP_VERSION} · Base revisada {REVIEW_DATE} · Uso como apoyo clínico, no sustituto del juicio profesional.")
