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
    ckd_g_stage,
    dosing_band_from_egfr,
    stage_to_dosing_band,
    rule_applies_demographics,
    select_renal_rule,
)

APP_VERSION = "V7.4.4 · PEDIATRÍA BIBLIOGRÁFICA VISIBLE"
REVIEW_DATE = "2026-09-04"
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
      :root {
        --mc-bg:#f5f7fa;
        --mc-card:#ffffff;
        --mc-ink:#12202f;
        --mc-muted:#667788;
        --mc-line:#e3e9ef;
        --mc-primary:#0f6f78;
        --mc-primary-2:#16818b;
        --mc-blue:#2765c5;
        --mc-green:#237a57;
        --mc-amber:#a86414;
        --mc-red:#a93f45;
        --mc-soft:#eef6f7;
      }

      html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
      .stApp {background:linear-gradient(180deg,#f8fafc 0%,var(--mc-bg) 100%); color:var(--mc-ink);}
      .block-container {padding-top:1.15rem; padding-bottom:3.5rem; max-width:1420px;}

      /* Sidebar */
      [data-testid="stSidebar"] {
        background:linear-gradient(180deg,#ffffff 0%,#f6fafb 100%);
        border-right:1px solid var(--mc-line);
      }
      [data-testid="stSidebar"] > div:first-child {padding-top:.9rem;}
      [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius:12px;
        padding:.28rem .45rem;
        transition:background .15s ease;
      }
      [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#edf5f6;}
      .side-brand {
        border:1px solid #dce8eb;
        border-radius:16px;
        padding:14px 15px;
        background:white;
        box-shadow:0 8px 26px rgba(34,67,78,.06);
        margin-bottom:.7rem;
      }
      .side-brand-title {font-size:1.03rem;font-weight:800;color:#12313a;}
      .side-brand-sub {font-size:.75rem;color:#70808b;margin-top:2px;}
      .side-badge {display:inline-block;margin-top:8px;padding:3px 8px;border-radius:999px;background:#e7f3f4;color:#17636b;font-size:.68rem;font-weight:800;letter-spacing:.04em;}

      /* Headers */
      .page-head {margin:0 0 1.1rem;}
      .medcalc-kicker {font-size:.70rem;letter-spacing:.11em;text-transform:uppercase;color:#72818c;font-weight:800;}
      .medcalc-title {font-size:2.05rem;font-weight:850;line-height:1.12;margin:.18rem 0 .35rem;color:var(--mc-ink);letter-spacing:-.025em;}
      .medcalc-subtitle {color:var(--mc-muted);font-size:.98rem;max-width:900px;line-height:1.55;}

      /* Cards */
      .hero {
        position:relative;overflow:hidden;
        background:linear-gradient(135deg,#ffffff 0%,#edf7f8 100%);
        border:1px solid #d7e7e9;
        border-radius:22px;
        padding:24px 26px;
        margin:.4rem 0 1.1rem;
        box-shadow:0 12px 34px rgba(28,70,79,.07);
      }
      .hero:after {content:"";position:absolute;right:-55px;top:-70px;width:190px;height:190px;border-radius:50%;background:rgba(15,111,120,.07);}
      .hero-title {font-size:1.05rem;font-weight:850;color:#15353d;margin-bottom:.3rem;}
      .hero-copy {color:#657783;max-width:900px;line-height:1.55;}

      .module-card {
        border:1px solid var(--mc-line);
        border-radius:20px;
        padding:18px 18px 15px;
        background:white;
        min-height:172px;
        box-shadow:0 8px 28px rgba(31,56,72,.055);
      }
      .module-icon {font-size:1.45rem;line-height:1;margin-bottom:.7rem;}
      .module-title {font-size:1rem;font-weight:850;color:#172532;margin-bottom:.28rem;}
      .module-count {font-size:.79rem;color:#71808c;margin-bottom:.75rem;}

      .safe-card {border:1px solid var(--mc-line);border-radius:16px;padding:16px 18px;background:#fff;margin:.4rem 0 1rem;}
      .result-box {border:1px solid #d8e8ea;border-left:5px solid var(--mc-primary);background:#f4fafb;border-radius:14px;padding:14px 16px;margin:.65rem 0;line-height:1.55;}
      .renal-stage {border:1px solid #dbe6ec;border-radius:16px;padding:15px 17px;background:#fff;box-shadow:0 5px 18px rgba(31,56,72,.04);}

      /* Pills */
      .chip {display:inline-block;padding:5px 10px;border-radius:999px;background:#edf6f7;color:#1c626b;font-size:.73rem;font-weight:750;margin:2px 4px 3px 0;border:1px solid #dcebec;}
      .status-ok,.status-off,.status-ref {display:inline-block;padding:5px 10px;border-radius:999px;font-size:.72rem;font-weight:800;border:1px solid transparent;}
      .status-ok {background:#eaf7f0;color:#256b4d;border-color:#d6eee1;}
      .status-off {background:#fff5e9;color:#8a5614;border-color:#f3e2ca;}
      .status-ref {background:#edf4ff;color:#315d9a;border-color:#dce7f8;}

      /* Inputs */
      div[data-baseweb="input"] > div,
      div[data-baseweb="select"] > div {
        border-radius:12px !important;
        border-color:#dbe4ea !important;
        background:white !important;
      }
      div[data-baseweb="input"] > div:focus-within,
      div[data-baseweb="select"] > div:focus-within {border-color:#7bb2b8 !important;box-shadow:0 0 0 2px rgba(15,111,120,.08) !important;}
      [data-testid="stNumberInput"] input {border-radius:12px !important;}

      /* Metrics */
      div[data-testid="stMetric"] {
        border:1px solid var(--mc-line);
        padding:14px 16px;
        border-radius:17px;
        background:white;
        box-shadow:0 6px 22px rgba(31,56,72,.045);
        min-height:94px;
      }
      div[data-testid="stMetricLabel"] {color:#6a7b87;font-weight:700;}
      div[data-testid="stMetricValue"] {color:#172532;font-weight:800;letter-spacing:-.02em;}

      /* Buttons */
      div.stButton > button, div[data-testid="stLinkButton"] > a {
        border-radius:12px !important;
        font-weight:750 !important;
        min-height:2.65rem;
        transition:transform .08s ease,box-shadow .08s ease;
      }
      div.stButton > button:hover, div[data-testid="stLinkButton"] > a:hover {transform:translateY(-1px);}
      div.stButton > button[kind="primary"] {background:linear-gradient(135deg,var(--mc-primary),var(--mc-primary-2));border-color:transparent;box-shadow:0 7px 18px rgba(15,111,120,.18);}

      /* Tabs */
      button[data-baseweb="tab"] {font-weight:750 !important;color:#5f7180 !important;}
      button[data-baseweb="tab"][aria-selected="true"] {color:var(--mc-primary) !important;}

      /* Alerts */
      [data-testid="stAlert"] {border-radius:14px !important;border-width:1px !important;}

      /* Expanders */
      details {border-radius:14px !important;border-color:var(--mc-line) !important;background:#fff !important;}

      /* Footer */
      .mc-footer {font-size:.72rem;color:#7a8994;text-align:center;padding:.75rem 0 0;}

      /* Mobile */
      @media (max-width: 768px) {
        .block-container {padding-left:.9rem;padding-right:.9rem;padding-top:.7rem;}
        .medcalc-title {font-size:1.65rem;}
        .hero {padding:18px;border-radius:18px;}
        .module-card {min-height:unset;margin-bottom:.55rem;}
      }
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
        "Verifique Streamlit Secrets, que el proyecto Supabase esté activo y que las políticas RLS permitan SELECT de PUBLISHED y PENDING_REVIEW en pediatría."
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
    icons = {
        "MedCalc Clínico": "✦",
        "Dosis pediátrica": "👶",
        "Ajuste renal adulto": "🧮",
        "Toxicología": "☠️",
        "Base clínica y fuentes": "📚",
    }
    icon = icons.get(title, "🩺")
    st.markdown(
        f'<div class="page-head"><div class="medcalc-kicker">MEDCALC CLÍNICO · {APP_VERSION}</div>'
        f'<div class="medcalc-title">{icon}&nbsp;&nbsp;{title}</div>'
        f'<div class="medcalc-subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def source_block(source, url, revision=None, page=None):
    c1, c2 = st.columns([3, 1])
    text = f"Fuente: {source or 'No consignada'}"
    if page not in (None, ""):
        text += f" · pág. {page}"
    if revision:
        text += f" · revisión {revision}"
    c1.caption(text)
    if url:
        c2.link_button("Abrir fuente", url, use_container_width=True)


def navigate(target, med_id=None):
    # No modificar directamente la key del widget de navegación una vez creado.
    st.session_state["pending_nav_page"] = target
    if med_id:
        st.session_state["selected_med_id"] = med_id


def go_to_module(target, med_id=None):
    # El cambio se aplica al inicio del siguiente rerun, antes de crear el radio.
    st.session_state["pending_nav_page"] = target
    if med_id:
        st.session_state["selected_med_id"] = med_id
    st.rerun()

def medication_picker(prefix, title="Medicamento", help_text=None):
    """Buscador explícito sobre todo el catálogo maestro Supabase."""
    query = st.text_input(
        f"Buscar {title.lower()}",
        placeholder="Escriba parte del nombre, por ejemplo: amoxi, aciclovir, gabapentina…",
        key=f"{prefix}_med_query",
    )
    hits = db.search_medications(query, limit=max(COUNTS.get("medications", 1000), 1000))
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
        help=help_text or f"El selector proviene de la tabla maestra Supabase ({COUNTS.get('medications', '—')} MED-ID).",
    )
    row = hits[labels.index(picked)]
    st.session_state["selected_med_id"] = row["med_id"]
    return row


def status_badges(summary):
    c1, c2, c3 = st.columns(3)
    ped_n = int(summary.get("pediatric_rule_count") or 0)
    ped_pub = int(summary.get("pediatric_published_count") or 0)
    ped_pending = int(summary.get("pediatric_pending_count") or 0)
    ren_n = int(summary.get("renal_rule_count") or 0)
    ref_n = int(summary.get("renal_biblio_count") or 0)
    tox = int(summary.get("toxicology_available") or 0)
    ped_text = f"PEDIATRÍA · {ped_pub} publicadas"
    if ped_pending:
        ped_text += f" + {ped_pending} pendientes"
    if not ped_n:
        ped_text = "PEDIATRÍA · sin pauta"
    c1.markdown(
        f'<span class="status-{"ok" if ped_n else "off"}">{ped_text}</span>',
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
    header("MedCalc Clínico", "Buscador central con navegación directa a pediatría, función renal y toxicología.")
    st.markdown(
        f'<div class="hero"><div class="hero-title">Base clínica central · Supabase</div>'
        f'<div class="hero-copy">Consulta un medicamento una sola vez y navega entre sus pautas pediátricas, ajuste renal y toxicología. '
        f'Actualmente hay <strong>{COUNTS["medications"]} MED-ID</strong> activos en la base clínica.</div></div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Catálogo maestro", f"{COUNTS['medications']} medicamentos")
    m2.metric("Pediatría", f"{COUNTS['pediatric_rules']} reglas")
    m3.metric("Renal", f"{COUNTS['renal_rules']} auto · {COUNTS['renal_biblio']} ref.")
    m4.metric("Toxicología", f"{COUNTS['toxicology']} fichas")

    st.markdown("### 🔎 Buscar medicamento")
    st.caption("Escriba nombre genérico o parte del nombre. La selección se conserva al abrir otro módulo.")
    med = medication_picker("home", "Resultado")
    if not med:
        return
    summary = db.medication(med["med_id"])
    st.markdown(f"## {summary['principio_activo']} · {summary['med_id']}")
    status_badges(summary)

    ped_inds = db.pediatric_indications(summary["med_id"])
    renal_inds = db.renal_indications(summary["med_id"])
    renal_refs = db.renal_biblio(summary["med_id"])
    tox = db.toxicology(summary["med_id"])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="module-card"><div class="module-icon">👶</div><div class="module-title">Pediatría</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="module-count">{len(ped_inds)} indicación(es) con pauta cargada</div>', unsafe_allow_html=True)
        if ped_inds:
            for r in ped_inds[:6]:
                st.markdown(f'<span class="chip">{r["indicacion"]}</span>', unsafe_allow_html=True)
            if len(ped_inds) > 6:
                st.caption(f"+ {len(ped_inds)-6} escenarios adicionales")
        else:
            st.caption("Sin pauta pediátrica cargada.")
        if st.button("Abrir Pediatría", key="home_open_ped", use_container_width=True):
            go_to_module("Dosis pediátrica", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="module-card"><div class="module-icon">🧮</div><div class="module-title">Función renal</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="module-count">{len(renal_inds)} escenario(s) automáticos · {len(renal_refs)} referencia(s)</div>', unsafe_allow_html=True)
        if renal_inds:
            for r in renal_inds[:5]:
                st.markdown(f'<span class="chip">{r["indicacion"]}</span>', unsafe_allow_html=True)
        elif renal_refs:
            st.caption("Hay bibliografía renal enlazada aunque no exista regla automática.")
        else:
            st.caption("Pendiente de revisión renal.")
        if st.button("Abrir Ajuste renal", key="home_open_renal", use_container_width=True):
            go_to_module("Ajuste renal", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="module-card"><div class="module-icon">☠️</div><div class="module-title">Toxicología</div>', unsafe_allow_html=True)
        if tox:
            st.markdown('<div class="module-count">Ficha toxicológica disponible</div>', unsafe_allow_html=True)
            st.write(f"**Dosis bibliográfica:** {tox.get('dosis_toxica_base') or 'SDTE / no consignada'}")
            st.caption(f"Estado: {tox.get('estado_revision') or '—'}")
        else:
            st.caption("Pendiente de ficha toxicológica.")
        if st.button("Abrir Toxicología", key="home_open_tox", use_container_width=True):
            go_to_module("Toxicología", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)


def pediatric_rule_dose_text(rule):
    """Representación fiel de la dosis estructurada sin recalcularla."""
    dtype = str(rule.get("tipo_dosis") or "").upper()
    unit = str(rule.get("unidad_dosis") or "mg")
    lo = as_float(rule.get("dosis_valor"))
    hi = as_float(rule.get("dosis_valor_max"))
    fixed_lo = as_float(rule.get("dosis_fija_valor"))
    fixed_hi = as_float(rule.get("dosis_fija_valor_max"))

    def _rng(a, b, suffix=""):
        if a is None:
            return None
        b = a if b is None else b
        return fmt_range(a, b, unit) + suffix

    if fixed_lo is not None:
        text = _rng(fixed_lo, fixed_hi)
        if "DIA" in dtype:
            return text + "/día"
        return text

    if lo is not None:
        if "KG_DIA" in dtype:
            return _rng(lo, hi, "/kg/día")
        if "KG_DOSIS" in dtype:
            return _rng(lo, hi, "/kg/dosis")
        if "M2_DIA" in dtype or "M²_DIA" in dtype:
            return _rng(lo, hi, "/m²/día")
        if "M2_DOSIS" in dtype or "M²_DOSIS" in dtype:
            return _rng(lo, hi, "/m²/dosis")
        return _rng(lo, hi)

    # En varias reglas tópicas/inhaladas la pauta está codificada como texto
    # o como cantidad fija sin dosis_min/dose_max.
    return rule.get("frecuencia_texto") or "Dosis numérica no estructurada en esta fila"


def pediatric_rule_limits_text(rule):
    parts = []
    unit = str(rule.get("unidad_dosis") or "mg")
    mx = as_float(rule.get("max_dosis_valor"))
    md = as_float(rule.get("max_dia_valor"))
    mxkg = as_float(rule.get("max_dosis_valorkg"))
    mdkg = as_float(rule.get("max_dia_valorkg"))
    if mx is not None:
        parts.append(f"máx. por dosis {fmt_num(mx,2)} {unit}")
    if mxkg is not None:
        parts.append(f"máx. por dosis {fmt_num(mxkg,2)} {unit}/kg")
    if md is not None:
        parts.append(f"máx. diario {fmt_num(md,2)} {unit}")
    if mdkg is not None:
        parts.append(f"máx. diario {fmt_num(mdkg,2)} {unit}/kg")
    return " · ".join(parts) if parts else "No consignado"


def show_pending_pediatric_rules(pending_rules):
    if not pending_rules:
        return

    st.markdown("### 📚 Pautas bibliográficas pendientes de validación")
    st.warning(
        "Estas pautas **sí están cargadas en Supabase y se muestran como referencia bibliográfica**, aunque su estado sea "
        "`PENDING_REVIEW`. Conserve la indicación, población, vía, dosis, frecuencia, duración y fuente como referencia. "
        "**No se usan para el cálculo clínico validado hasta pasar a `PUBLISHED`.**"
    )

    for idx, rule in enumerate(pending_rules):
        title = (
            f"{rule.get('indicacion') or 'Sin indicación'} · "
            f"{rule.get('poblacion') or 'Pediatría'} · "
            f"{rule.get('via') or 'vía no consignada'}"
        )
        with st.expander(title, expanded=(idx == 0)):
            st.warning("🟡 DOSIS BIBLIOGRÁFICA · PENDING_REVIEW · todavía no validada como regla clínica publicada")
            c1, c2, c3 = st.columns(3)
            c1.metric("Dosis registrada", pediatric_rule_dose_text(rule))
            c2.metric(
                "Intervalo / frecuencia",
                f"cada {fmt_num(rule['intervalo_h'],1)} h"
                if as_float(rule.get("intervalo_h")) is not None
                else (rule.get("frecuencia_texto") or "No consignado"),
            )
            c3.metric("Máximos", pediatric_rule_limits_text(rule))

            st.write(f"**Población:** {rule.get('poblacion') or '—'}")
            st.write(f"**Vía:** {rule.get('via') or '—'}")
            if rule.get("duracion"):
                st.write(f"**Duración:** {rule['duracion']}")
            if rule.get("frecuencia_texto"):
                st.write(f"**Pauta textual de la fuente:** {rule['frecuencia_texto']}")
            if rule.get("notas"):
                st.info(rule["notas"])
            if rule.get("nota_renal"):
                st.warning("Función renal: " + str(rule["nota_renal"]))

            if str(rule.get("automatizable") or "").upper() == "SI":
                st.caption(
                    "La estructura de esta fila permite cálculo matemático, pero MedCalc lo mantiene "
                    "bloqueado mientras la revisión clínica siga en PENDING_REVIEW."
                )
            else:
                st.caption("Esta pauta está marcada como referencia no automatizable.")

            source_block(rule.get("fuente"), rule.get("url_fuente"), rule.get("fecha_revision"), rule.get("pagina_fuente"))


def page_pediatric():
    header(
        "Dosis pediátrica",
        "Las reglas PUBLISHED permiten cálculo cuando son automatizables. "
        "Las PENDING_REVIEW también se muestran como referencia bibliográfica, claramente diferenciadas.",
    )
    if st.button("← Volver al inicio", key="ped_back_home"):
        go_to_module("Inicio", st.session_state.get("selected_med_id"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Catálogo Supabase", f"{COUNTS['medications']} medicamentos")
    m2.metric("Con alguna pauta", f"{COUNTS.get('pediatric_meds', 0)} medicamentos")
    m3.metric("Reglas publicadas", f"{COUNTS.get('pediatric_rules_published', 0)}")
    m4.metric("Pendientes visibles", f"{COUNTS.get('pediatric_rules_pending', 0)}")

    med = medication_picker("ped", "Medicamento")
    if not med:
        return

    rules = db.pediatric_rules(med["med_id"])
    published_rules = [
        r for r in rules if str(r.get("estado") or "").upper() == "PUBLISHED"
    ]
    pending_rules = [
        r for r in rules if str(r.get("estado") or "").upper() == "PENDING_REVIEW"
    ]
    auto_rules = [
        r for r in published_rules if str(r.get("automatizable") or "").upper() == "SI"
    ]

    if not rules:
        st.warning(
            f"**{med['principio_activo']}: SIN PAUTA PEDIÁTRICA CARGADA.** "
            "El medicamento permanece en el catálogo, pero no hay una regla estructurada para mostrar."
        )
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Pautas cargadas", len(rules))
    c2.metric("Publicadas", len(published_rules))
    c3.metric("Pendientes de revisión", len(pending_rules))

    # Primero hacemos visibles las pautas PENDING_REVIEW, incluso si el fármaco
    # no tiene ninguna regla publicada/automatizable.
    show_pending_pediatric_rules(pending_rules)

    if not auto_rules:
        if published_rules:
            st.info(
                "Este medicamento tiene pauta(s) PUBLISHED, pero ninguna está habilitada para "
                "cálculo automático. La información permanece visible arriba o en sus fuentes."
            )
        elif pending_rules:
            st.warning(
                f"**{med['principio_activo']} tiene información pediátrica cargada, pero todavía no publicada.** "
                "Puede consultar la dosis y el escenario en el bloque bibliográfico anterior. "
                "El cálculo automático seguirá bloqueado hasta la validación."
            )
        return

    st.divider()
    st.markdown("### ✅ Calculadora con reglas publicadas")
    st.success(
        f"{med['principio_activo']}: {len(auto_rules)} regla(s) PUBLISHED y automatizable(s)."
    )

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
            st.error("No existe una regla PUBLISHED compatible con esa edad/peso para el escenario seleccionado.")
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
    st.markdown(
        f'<div class="result-box"><strong>{rule["rule_id"]}</strong> · '
        f'{rule["poblacion"]} · {rule["via"]} · PUBLISHED</div>',
        unsafe_allow_html=True,
    )
    if (rule.get("nivel_uso") or "GENERAL") != "GENERAL":
        st.warning(f"Nivel de uso: **{rule.get('nivel_uso')}**")

    x1, x2, x3, x4 = st.columns(4)
    x1.metric("Dosis por administración", fmt_range(result["min_value"], result["max_value"], unit))
    x2.metric(
        "Intervalo",
        f"cada {fmt_num(result['interval_h'],1)} h"
        if result.get("interval_h")
        else (rule.get("frecuencia_texto") or "Según regla"),
    )
    x3.metric(
        "Dosis diaria",
        fmt_range(result["daily_min_value"], result["daily_max_value"], f"{unit}/día")
        if result.get("daily_min_value") is not None
        else "Según frecuencia",
    )
    x4.metric(
        "Máximo por dosis",
        f"{fmt_num(result['max_single_value'],2)} {unit}"
        if result.get("max_single_value") is not None
        else "No cargado",
    )
    st.caption(f"Trazabilidad: {result['formula']}")
    if rule.get("frecuencia_texto"):
        st.info(rule["frecuencia_texto"])
    if rule.get("duracion"):
        st.info(f"Duración: {rule['duracion']}")
    if rule.get("notas"):
        st.info(rule["notas"])
    if rule.get("nota_renal"):
        st.warning("Función renal: " + rule["nota_renal"])

    if str(rule.get("permite_conversion_volumen") or "SI").upper() == "SI":
        st.subheader("Conversión a volumen")
        with st.form("ped_volume_sql", border=True):
            q1, q2 = st.columns(2)
            default_amount = 100000.0 if unit.upper().startswith("U") else 100.0
            label_value = q1.number_input(
                f"Cantidad de fármaco ({unit})",
                min_value=0.0001,
                value=default_amount,
                step=1.0,
            )
            label_ml = q2.number_input(
                "Volumen correspondiente (mL)",
                min_value=0.01,
                value=5.0,
                step=0.5,
            )
            cv = st.form_submit_button(f"Convertir {unit} → mL", use_container_width=True)
        if cv:
            st.session_state["ped_sql_volume"] = quantity_to_ml(
                result["min_value"], result["max_value"], label_value, label_ml
            )
        vol = st.session_state.get("ped_sql_volume")
        if vol:
            v1, v2 = st.columns(2)
            v1.metric("Concentración", f"{fmt_num(vol['unit_per_ml'],3)} {unit}/mL")
            v2.metric("Volumen por dosis", fmt_range(vol["min_ml"], vol["max_ml"], "mL"))

    source_block(rule.get("fuente"), rule.get("url_fuente"), rule.get("fecha_revision"), rule.get("pagina_fuente"))


def page_renal():
    header(
        "Ajuste renal adulto",
        "Dosificación directa con eGFR CKD-EPI 2021 o eGFR conocido. No requiere peso ni talla.",
    )
    st.info(
        "Para adultos, MedCalc usa edad + sexo + creatinina para estimar eGFR con CKD-EPI 2021. "
        "También puede ingresar directamente un eGFR conocido. La creatinina aislada no se interpreta sin edad y sexo."
    )

    med = medication_picker("renal", "Medicamento")
    if not med:
        return
    auto_rules = [r for r in db.renal_rules(med["med_id"]) if r.get("automatizable") == "SI"]
    refs = db.renal_biblio(med["med_id"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Catálogo Supabase", f"{COUNTS['medications']} medicamentos")
    c2.metric("Reglas automáticas", len(auto_rules))
    c3.metric("Referencias renales", len(refs))

    mode = st.radio(
        "Cómo obtener la función renal",
        ["Calcular CKD-EPI 2021", "Ingresar eGFR conocido", "Solo conozco el estadio KDIGO"],
        horizontal=True,
        key="renal_mode_v72",
    )
    hd = st.checkbox("Paciente en hemodiálisis", key="renal_hd_v72")
    egfr = None
    exact_available = False

    if mode == "Calcular CKD-EPI 2021":
        with st.form("renal_ckdepi_v72", border=True):
            a, b, c = st.columns(3)
            age = a.number_input("Edad (años)", min_value=18, max_value=120, value=60, step=1)
            sex = b.selectbox("Sexo para la ecuación", ["Hombre", "Mujer"])
            creat = c.number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
            submit = st.form_submit_button("Calcular eGFR y ajuste", type="primary", use_container_width=True)
        if submit:
            egfr = ckdepi_2021(age, sex, creat)
            st.session_state["renal_v72"] = {"egfr": egfr, "age": age, "sex": sex, "creat": creat, "source": "CKD-EPI 2021"}
        snap = st.session_state.get("renal_v72")
        if snap:
            egfr = snap.get("egfr")
            exact_available = egfr is not None

    elif mode == "Ingresar eGFR conocido":
        with st.form("renal_known_egfr_v72", border=True):
            egfr_in = st.number_input("eGFR (mL/min/1,73 m²)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)
            submit = st.form_submit_button("Usar eGFR y obtener ajuste", type="primary", use_container_width=True)
        if submit:
            st.session_state["renal_v72"] = {"egfr": egfr_in, "source": "eGFR ingresado"}
        snap = st.session_state.get("renal_v72")
        if snap:
            egfr = snap.get("egfr")
            exact_available = egfr is not None

    else:
        stage = st.selectbox("Estadio KDIGO", ["G1", "G2", "G3a", "G3b", "G4", "G5"], key="renal_stage_manual_v72")
        stage_band, stage_error = stage_to_dosing_band(stage)
        stage_desc = {"G1":"normal o alto","G2":"levemente disminuido","G3a":"leve-moderadamente disminuido","G3b":"moderada-severamente disminuido","G4":"severamente disminuido","G5":"falla renal"}.get(stage)
        st.markdown(f'<div class="renal-stage"><strong>{stage}</strong> · {stage_desc}</div>', unsafe_allow_html=True)
        if stage_error:
            st.warning(stage_error)
            band_key = None
        else:
            band_key = stage_band

    if exact_available:
        stage, stage_desc = ckd_g_stage(egfr)
        band_key, band_label = dosing_band_from_egfr(egfr)
        m1, m2, m3 = st.columns(3)
        m1.metric("eGFR", f"{fmt_num(egfr,1)} mL/min/1,73 m²")
        m2.metric("Estadio KDIGO", stage)
        m3.metric("Banda de dosificación FR-001", band_label)
        st.caption(f"{stage}: {stage_desc}. La categoría KDIGO describe función renal; la dosis final depende de la fuente específica del medicamento.")
    elif mode != "Solo conozco el estadio KDIGO":
        band_key = None

    if not auto_rules and not refs:
        st.warning(
            f"**{med['principio_activo']}** todavía no tiene regla renal publicada ni referencia enlazada. "
            "Permanece visible en el catálogo, pero no se inventa un ajuste."
        )
        return

    st.markdown("### Dosis/ajuste renal")
    tab1, tab2 = st.tabs(["Recomendación directa", "Ver bibliografía y reglas"])

    with tab1:
        if hd:
            if refs:
                ref = refs[0]
                st.markdown(f"#### {ref.get('principio_activo') or med['principio_activo']}")
                st.success(ref.get("suplemento_hd") or "La fuente no consigna una pauta específica de hemodiálisis.")
                st.caption("Se reproduce la columna de hemodiálisis de la bibliografía renal enlazada.")
            else:
                dialysis_rules = [r for r in auto_rules if (r.get("tipo_regla") or "").upper() == "DIALISIS"]
                if dialysis_rules:
                    st.success(dialysis_rules[0].get("regimen_ajustado") or "—")
                else:
                    st.warning("Sin pauta de hemodiálisis publicada para este medicamento.")
        elif band_key and refs:
            labels=[f"Tabla {r['table']} · pág. {r['page']} · {r['principio_activo']}" for r in refs]
            pick=st.selectbox("Referencia renal",labels,key="renal_direct_ref_v72")
            ref=refs[labels.index(pick)]
            st.markdown(f"**Dosis con función renal normal:** {ref.get('dosis_fr_normal') or '—'}")
            recommendation=ref.get(band_key) or "—"
            st.success(f"**Ajuste correspondiente:** {recommendation}")
            if ref.get("notas"):
                st.info(ref["notas"])
            source_block(ref.get("fuente"),ref.get("url_fuente"),ref.get("fecha_fuente"))
        elif band_key and auto_rules:
            compatible=[r for r in auto_rules if "EGFR" in str(r.get("metrica_renal") or "").upper() or "1_73" in str(r.get("metrica_renal") or "").upper()]
            if compatible and exact_available:
                indications=sorted({r.get("indicacion") or "Sin indicación" for r in compatible})
                ind=st.selectbox("Indicación / régimen",indications,key="renal_direct_ind_v72")
                rules=[r for r in compatible if (r.get("indicacion") or "Sin indicación")==ind]
                selected,_=select_renal_rule(rules,None,egfr,egfr,False)
                if selected:
                    st.success(selected.get("regimen_ajustado") or "—")
                    if selected.get("notas"): st.info(selected["notas"])
                    source_block(selected.get("fuente"),selected.get("url_fuente"),selected.get("fecha_revision"))
                else:
                    st.warning("No hay una banda eGFR compatible para este valor.")
            else:
                st.warning(
                    "Las reglas automáticas existentes para este fármaco están codificadas con CrCl/Cockcroft-Gault. "
                    "Como esta versión no solicita peso, no las convierte silenciosamente a eGFR. Se muestran en la pestaña de referencia hasta normalizarlas individualmente."
                )
        else:
            st.caption("Ingrese un eGFR exacto o un estadio que permita identificar de forma inequívoca la banda de dosificación.")

    with tab2:
        if refs:
            st.markdown("#### Nefrología al Día 2025")
            for idx, ref in enumerate(refs):
                with st.expander(f"{ref.get('principio_activo')} · Tabla {ref.get('table')} · pág. {ref.get('page')}", expanded=idx==0):
                    st.write(f"**Dosis función renal normal:** {ref.get('dosis_fr_normal') or '—'}")
                    st.write(f"**Método:** {ref.get('metodo') or '—'}")
                    st.write(f"**≥50:** {ref.get('crcl_100_50') or '—'}")
                    st.write(f"**10–49:** {ref.get('crcl_50_10') or '—'}")
                    st.write(f"**<10:** {ref.get('crcl_lt10') or '—'}")
                    st.write(f"**HD:** {ref.get('suplemento_hd') or '—'}")
                    img=resolve_renal_image(ref.get("imagen"),ref.get("table"))
                    if img:
                        st.image(str(img),use_container_width=True)
                    source_block(ref.get("fuente"),ref.get("url_fuente"),ref.get("fecha_fuente"))
        if auto_rules:
            st.markdown("#### Reglas automáticas estructuradas")
            for r in auto_rules:
                with st.expander(f"{r.get('indicacion') or 'Sin indicación'} · {r.get('rango') or 'banda'}"):
                    st.write(f"**Métrica original:** {r.get('metrica_renal') or '—'}")
                    st.write(f"**Régimen:** {r.get('regimen_ajustado') or '—'}")
                    if r.get("notas"): st.info(r["notas"])
                    source_block(r.get("fuente"),r.get("url_fuente"),r.get("fecha_revision"))

def page_toxicology():
    header("Toxicología", "Síntomas detallados por medicamento, dosis bibliográfica original y capa clínica revisada.")
    if st.button("← Volver al inicio", key="tox_back_home"):
        go_to_module("Inicio", st.session_state.get("selected_med_id"))

    definition = db.metadata("general_symptoms_definition")
    if definition:
        with st.expander("¿Qué significa ‘síntomas generales’ en la base antigua?"):
            st.info(definition)
            st.caption("En V7.3 esta expresión deja de ser la respuesta principal: cada ficha farmacológica muestra un bloque específico de manifestaciones de intoxicación.")

    tab1, tab2, tab3 = st.tabs(["Medicamentos", "Drogas/plaguicidas/metales", "Antídotos"])
    with tab1:
        med = medication_picker("tox", "Medicamento")
        if not med:
            return
        tox = db.toxicology(med["med_id"])
        if not tox:
            st.warning("Sin ficha toxicológica enlazada.")
            return

        st.markdown(f"### {med['principio_activo']} · {med['med_id']}")
        detail = tox.get("sintomas_intoxicacion_detallados") or tox.get("manifestaciones_clave") or "Sin detalle disponible."
        st.markdown("#### 🚨 Manifestaciones detalladas de intoxicación")
        st.markdown(f'<div class="result-box"><strong>{detail}</strong></div>', unsafe_allow_html=True)
        status = tox.get("estado_sintomas_detallados") or "—"
        if status in {"LIMITED_OVERDOSE_DATA", "CLASS_BASED", "NEEDS_NAME_VERIFICATION"}:
            st.warning(f"Nivel de caracterización: **{status}**. Cuando la sobredosis humana específica es limitada, la descripción se basa en farmacología, efectos adversos graves conocidos o evidencia por clase y se señala como tal.")
        else:
            st.success(f"Nivel de caracterización: **{status}**")
        if tox.get("fuente_sintomas_detallados"):
            st.caption("Fuente/criterio de los síntomas detallados: " + str(tox["fuente_sintomas_detallados"]))

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Evaluación toxicológica revisada")
            st.write(f"**Criterio / umbral:** {tox.get('dosis_toxica_corregida') or 'SDTE / sin umbral numérico validado'}")
            st.write(f"**Tipo de umbral:** {tox.get('tipo_umbral') or '—'}")
            st.write(f"**Manejo:** {tox.get('manejo_corregido') or '—'}")
            st.write(f"**Terapia específica:** {tox.get('antidoto_especifico') or 'No hay antídoto específico cargado'}")
        with c2:
            st.markdown("#### 📚 Trazabilidad de la base original")
            st.write(f"**Dosis tóxica registrada originalmente:** {tox.get('dosis_toxica_base') or '—'}")
            original_symptoms = tox.get('sintomas_base') or '—'
            if str(original_symptoms).strip().lower() in {'sintomas generales','síntomas generales'}:
                st.write("**Texto original de síntomas:** ‘SÍNTOMAS GENERALES’ (conservado solo para trazabilidad; sustituido arriba por descripción detallada).")
            else:
                st.write(f"**Texto original de síntomas:** {original_symptoms}")
            st.write(f"**Manejo/antídoto original:** {tox.get('antidoto_manejo_base') or '—'}")

        st.caption(f"Estado de revisión global: {tox.get('estado_revision') or '—'} · Nivel de evidencia: {tox.get('nivel_evidencia') or '—'}")
        threshold = as_float(tox.get("umbral_mgkg_automatizable"))
        if str(tox.get("permitir_comparacion_automatica") or "").upper() == "SI" and threshold is not None:
            st.subheader("Calculadora de exposición")
            with st.form("tox_exp_sql"):
                a, b = st.columns(2)
                total = a.number_input("Cantidad total ingerida (mg)", min_value=0.0, value=500.0, step=50.0)
                weight = b.number_input("Peso (kg)", min_value=0.1, value=20.0, step=0.1)
                submit = st.form_submit_button("Calcular mg/kg", use_container_width=True)
            if submit:
                exposure, ratio = calculate_exposure_mgkg(total, weight, threshold)
                st.metric("Exposición", f"{fmt_num(exposure,2)} mg/kg")
                if ratio is not None:
                    st.caption(f"Relación con la referencia cargada: {fmt_num(ratio,2)}×. Este cociente no sustituye la evaluación toxicológica.")
                st.info(tox.get("etiqueta_umbral") or f"Referencia cargada: {threshold:g} mg/kg")
        if tox.get("fuente_principal"):
            st.link_button("Abrir fuente principal", tox["fuente_principal"])

    with tab2:
        q = st.text_input("Buscar tóxico no farmacológico", key="other_tox_q")
        hits = db.search_other_tox(q)
        if hits:
            names = [r.get("toxico") or "—" for r in hits]
            pick = st.selectbox("Tóxico", names, key="other_tox_sel")
            r = hits[names.index(pick)]
            st.write(f"**Síntomas:** {r.get('sintomas_base') or '—'}")
            st.write(f"**Antídoto/tratamiento:** {r.get('antidoto_tratamiento_base') or '—'}")
            st.warning(f"Estado: {r.get('estado_validacion') or '—'}")
    with tab3:
        q = st.text_input("Buscar tóxico, síndrome o antídoto", key="antidote_q")
        hits = db.search_antidotes(q)
        if hits:
            labels = [f"{r.get('toxico_sindrome') or '—'} → {r.get('antidoto_base') or '—'}" for r in hits]
            pick = st.selectbox("Resultado", labels, key="antidote_sel")
            r = hits[labels.index(pick)]
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


# Estado de navegación independiente del widget.
# Si un botón solicitó cambio de módulo en el run anterior, lo aplicamos
# ANTES de instanciar el radio de la barra lateral.
if "nav_widget" not in st.session_state:
    st.session_state["nav_widget"] = "Inicio"

_pending_page = st.session_state.pop("pending_nav_page", None)
if _pending_page in PAGES:
    st.session_state["nav_widget"] = _pending_page

with st.sidebar:
    st.markdown(
        f'<div class="side-brand"><div class="side-brand-title">🩺 MedCalc Clínico</div>'
        f'<div class="side-brand-sub">Farmacología clínica · acceso abierto</div>'
        f'<span class="side-badge">{APP_VERSION}</span></div>',
        unsafe_allow_html=True,
    )
    page=st.radio(
        "Navegación",
        PAGES,
        key="nav_widget",
        format_func=lambda x: {
            "Inicio":"⌂  Inicio",
            "Dosis pediátrica":"👶  Dosis pediátrica",
            "Ajuste renal":"🧮  Ajuste renal",
            "Toxicología":"☠️  Toxicología",
            "Base y fuentes":"📚  Base y fuentes",
        }.get(x,x),
    )
    st.divider()
    st.caption(f"● Supabase conectado · {COUNTS['medications']} MED-ID")
    st.caption("Herramienta de apoyo clínico. Verifique siempre indicación, fuente y contexto del paciente.")
    st.link_button("☎️ CITUC Chile",CITUC_URL,use_container_width=True)

if page=="Inicio": page_home()
elif page=="Dosis pediátrica": page_pediatric()
elif page=="Ajuste renal": page_renal()
elif page=="Toxicología": page_toxicology()
else: page_sources()

st.divider()
st.markdown(
    f'<div class="mc-footer">MedCalc Clínico · {APP_VERSION} · Supabase {SCHEMA_VERSION} · revisión {REVIEW_DATE}<br>'
    'Herramienta de apoyo clínico; no sustituye juicio profesional, ficha técnica ni protocolo institucional.</div>',
    unsafe_allow_html=True,
)
