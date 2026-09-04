from pathlib import Path
import math
import re

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

APP_VERSION = "V7.6.1 · TOXICOLOGÍA + CALCULADORA"
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
      div[data-testid="stMetricValue"] {
        color:#172532;
        font-weight:800;
        letter-spacing:-.02em;
        overflow:visible !important;
        width:100% !important;
      }
      div[data-testid="stMetricValue"] > div,
      div[data-testid="stMetricValue"] [data-testid="stMetricValue"] {
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:clip !important;
        overflow-wrap:anywhere !important;
        word-break:normal !important;
        line-height:1.08 !important;
      }
      div[data-testid="stMetric"] {
        overflow:visible !important;
      }

      /* Tarjetas clínicas: nunca truncar texto con puntos suspensivos. */
      .clinical-grid {
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:14px;
        margin:.45rem 0 .8rem;
      }
      .clinical-card {
        border:1px solid var(--mc-line);
        border-radius:17px;
        background:#fff;
        padding:14px 16px;
        min-width:0;
        box-shadow:0 6px 22px rgba(31,56,72,.045);
      }
      .clinical-label {
        color:#6a7b87;
        font-size:.78rem;
        font-weight:700;
        margin-bottom:.38rem;
      }
      .clinical-value {
        color:#172532;
        font-size:1.45rem;
        line-height:1.15;
        font-weight:850;
        white-space:normal;
        overflow:visible;
        text-overflow:clip;
        overflow-wrap:anywhere;
        word-break:normal;
        -webkit-line-clamp:unset !important;
        max-width:none !important;
      }
      .kpi-grid {
        display:grid;
        grid-template-columns:repeat(4,minmax(0,1fr));
        gap:14px;
        margin:.3rem 0 1rem;
      }
      .kpi-card {
        border:1px solid var(--mc-line);
        border-radius:17px;
        background:#fff;
        padding:14px 16px;
        min-width:0;
        box-shadow:0 6px 22px rgba(31,56,72,.045);
      }
      .kpi-label {color:#6a7b87;font-size:.78rem;font-weight:700;margin-bottom:.3rem;}
      .kpi-value {
        color:#172532;
        line-height:1.05;
        font-weight:850;
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:clip !important;
        overflow-wrap:normal !important;
        word-break:normal !important;
        -webkit-line-clamp:unset !important;
        max-width:none !important;
      }
      .kpi-number {
        display:block;
        font-size:2rem;
        line-height:1;
        font-weight:900;
        white-space:nowrap !important;
        overflow:visible !important;
        text-overflow:clip !important;
      }
      .kpi-unit {
        display:block;
        margin-top:.28rem;
        font-size:.95rem;
        line-height:1.15;
        font-weight:750;
        color:#435565;
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:clip !important;
        -webkit-line-clamp:unset !important;
      }
      @media (max-width: 900px) {
        .clinical-grid {grid-template-columns:1fr;}
        .kpi-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
        .clinical-value {font-size:1.25rem;}
        .kpi-number {font-size:1.75rem;}
        .kpi-unit {font-size:.9rem;}
      }

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
        "Verifique Streamlit Secrets, que el proyecto Supabase esté activo y que las políticas de lectura permitan consultar el módulo de pediatría."
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
    ped_text = f"PEDIATRÍA · {ped_pub} validadas"
    if ped_pending:
        ped_text += f" + {ped_pending} referencias"
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


def _esc(value):
    import html as _html
    return _html.escape(str(value if value not in (None, "") else "—"))


def render_kpi_cards(items):
    """KPI robustos: número y descriptor se renderizan en bloques separados para evitar cualquier ellipsis del navegador."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        if len(item) == 3:
            label, number, unit = item
        else:
            label, number = item
            unit = ""
        with col:
            with st.container(border=True):
                st.caption(str(label))
                st.markdown(f"<div style='font-size:2.15rem;line-height:1;font-weight:900;color:#172532;white-space:normal;overflow:visible;text-overflow:clip'>{_esc(number)}</div>", unsafe_allow_html=True)
                if unit:
                    st.markdown(f"<div style='margin-top:.35rem;font-size:.95rem;line-height:1.2;font-weight:750;color:#435565;white-space:normal;overflow:visible;text-overflow:clip;overflow-wrap:anywhere'>{_esc(unit)}</div>", unsafe_allow_html=True)


def render_clinical_cards(items):
    """Tarjetas de texto clínico largo, sin truncamiento."""
    cards = []
    for label, value in items:
        if value in (None, "", "No consignado"):
            continue
        cards.append(
            f'<div class="clinical-card"><div class="clinical-label">{_esc(label)}</div>'
            f'<div class="clinical-value">{_esc(value)}</div></div>'
        )
    if cards:
        st.markdown('<div class="clinical-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def page_home():
    header("MedCalc Clínico", "Buscador central con navegación directa a pediatría, función renal y toxicología.")
    st.markdown(
        f'<div class="hero"><div class="hero-title">Base clínica central · Supabase</div>'
        f'<div class="hero-copy">Consulta un medicamento una sola vez y navega entre sus pautas pediátricas, ajuste renal y toxicología. '
        f'Actualmente hay <strong>{COUNTS["medications"]} MED-ID</strong> activos en la base clínica.</div></div>',
        unsafe_allow_html=True,
    )

    render_kpi_cards([
        ("Catálogo maestro", COUNTS['medications'], "medicamentos"),
        ("Pediatría", COUNTS['pediatric_rules'], "reglas"),
        ("Renal", COUNTS['renal_rules'], f"auto · {COUNTS['renal_biblio']} ref."),
        ("Toxicología", COUNTS['toxicology'], "fichas"),
    ])

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


def _pediatric_interval_hours(rule):
    """Obtiene el intervalo estructurado o lo recupera de una pauta textual simple tipo 'cada 12 h'."""
    interval = as_float(rule.get("intervalo_h"))
    if interval is not None and interval > 0:
        return interval
    raw = str(rule.get("frecuencia_texto") or "")
    m = re.search(r"cada\s+(\d+(?:[\.,]\d+)?)\s*(?:h|hora|horas)\b", raw, flags=re.IGNORECASE)
    if m:
        try:
            value = float(m.group(1).replace(",", "."))
            return value if value > 0 else None
        except Exception:
            return None
    return None


def _pediatric_divisions_per_day(rule):
    div = as_float(rule.get("divisiones_dia"))
    if div is not None and div > 0:
        return div
    interval = _pediatric_interval_hours(rule)
    if interval:
        return 24.0 / interval
    raw = str(rule.get("frecuencia_texto") or "")
    m = re.search(r"(?:dividid[oa]\s+en|en)\s+(\d+(?:[\.,]\d+)?)\s+(?:dosis|tomas)", raw, flags=re.IGNORECASE)
    if m:
        try:
            value = float(m.group(1).replace(",", "."))
            return value if value > 0 else None
        except Exception:
            return None
    return None


def pediatric_rule_can_calculate(rule):
    """Indica si la pauta visible contiene datos numéricos suficientes para calcularla.

    El cálculo depende de la estructura de la pauta, no de su estado editorial.
    """
    kind = str(rule.get("tipo_dosis") or "").upper().strip()
    dose = as_float(rule.get("dosis_valor"))
    fixed = as_float(rule.get("dosis_fija_valor"))
    if kind.startswith("FIJA"):
        return fixed is not None
    if dose is None:
        return False
    markers = (
        "KG_DOSIS", "KG_DIA", "KG_HORA", "KG_PERIODO",
        "M2_DOSIS", "M2_DIA", "M²_DOSIS", "M²_DIA",
    )
    return any(marker in kind for marker in markers)

def _range_text(lo, hi, unit):
    if lo is None:
        return None
    if hi is None:
        hi = lo
    return fmt_range(lo, hi, unit)


def calculate_loaded_pediatric_rule(rule, weight_kg, height_cm=None):
    """Calcula cualquier pauta numérica estructurada, independientemente de su estado editorial."""
    if weight_kg is None or float(weight_kg) <= 0:
        raise ValueError("Ingrese un peso mayor que cero.")
    weight_kg = float(weight_kg)
    kind = str(rule.get("tipo_dosis") or "").upper().strip()
    unit = str(rule.get("unidad_dosis") or "mg")
    dose = as_float(rule.get("dosis_valor"))
    dose_max = as_float(rule.get("dosis_valor_max"))
    fixed = as_float(rule.get("dosis_fija_valor"))
    fixed_max = as_float(rule.get("dosis_fija_valor_max"))
    interval = _pediatric_interval_hours(rule)
    divisions = _pediatric_divisions_per_day(rule)

    max_single = as_float(rule.get("max_dosis_valor"))
    max_single_kg = as_float(rule.get("max_dosis_valorkg"))
    max_daily = as_float(rule.get("max_dia_valor"))
    max_daily_kg = as_float(rule.get("max_dia_valorkg"))
    single_caps = [x for x in [max_single, max_single_kg * weight_kg if max_single_kg is not None else None] if x is not None]
    daily_caps = [x for x in [max_daily, max_daily_kg * weight_kg if max_daily_kg is not None else None] if x is not None]
    single_cap = min(single_caps) if single_caps else None
    daily_cap = min(daily_caps) if daily_caps else None

    result = {
        "unit": unit,
        "kind": kind,
        "interval_h": interval,
        "per_dose_min": None,
        "per_dose_max": None,
        "daily_min": None,
        "daily_max": None,
        "rate_min": None,
        "rate_max": None,
        "period_min": None,
        "period_max": None,
        "bsa": None,
        "formula": "",
        "caps": [],
    }

    def upper_rate(base, upper):
        return upper if upper is not None else base

    # Dosis fija (tabletas, gotas, puff, mg, UI, etc.)
    if kind in {"FIJA", "FIJA_RANGE"}:
        if fixed is None:
            raise ValueError("La pauta no contiene una dosis fija numérica.")
        lo = fixed
        hi = fixed_max if kind.endswith("RANGE") and fixed_max is not None else fixed
        result["per_dose_min"], result["per_dose_max"] = lo, hi
        result["formula"] = f"Dosis fija de la pauta: {_range_text(lo, hi, unit)}"
        if divisions:
            result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
        elif interval:
            nday = 24.0 / interval
            result["daily_min"], result["daily_max"] = lo * nday, hi * nday

    # Cantidad por kg por dosis
    elif kind.endswith("KG_DOSIS") or kind.endswith("KG_DOSIS_RANGE"):
        if dose is None:
            raise ValueError("La pauta no contiene una dosis por kg numérica.")
        rate_hi = upper_rate(dose, dose_max)
        lo, hi = dose * weight_kg, rate_hi * weight_kg
        result["per_dose_min"], result["per_dose_max"] = lo, hi
        result["formula"] = f"{weight_kg:g} kg × {_range_text(dose, rate_hi, unit + '/kg/dosis')}"
        if divisions:
            result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
        elif interval:
            nday = 24.0 / interval
            result["daily_min"], result["daily_max"] = lo * nday, hi * nday

    # Cantidad por kg por día
    elif kind.endswith("KG_DIA") or kind.endswith("KG_DIA_RANGE"):
        if dose is None:
            raise ValueError("La pauta no contiene una dosis por kg/día numérica.")
        rate_hi = upper_rate(dose, dose_max)
        daily_lo, daily_hi = dose * weight_kg, rate_hi * weight_kg
        result["daily_min"], result["daily_max"] = daily_lo, daily_hi
        result["formula"] = f"{weight_kg:g} kg × {_range_text(dose, rate_hi, unit + '/kg/día')}"
        div = divisions or (24.0 / interval if interval else None)
        if div:
            result["per_dose_min"], result["per_dose_max"] = daily_lo / div, daily_hi / div
            result["interval_h"] = 24.0 / div
            result["formula"] += f" ÷ {div:g} dosis/día"

    # Velocidad por kg/h
    elif "KG_HORA" in kind:
        if dose is None:
            raise ValueError("La pauta no contiene una velocidad por kg/h numérica.")
        rate_hi = upper_rate(dose, dose_max)
        result["rate_min"], result["rate_max"] = dose * weight_kg, rate_hi * weight_kg
        result["formula"] = f"{weight_kg:g} kg × {_range_text(dose, rate_hi, unit + '/kg/h')}"

    # Cantidad por kg administrada durante un periodo definido por la pauta
    elif "KG_PERIODO" in kind:
        if dose is None:
            raise ValueError("La pauta no contiene una cantidad por kg numérica.")
        rate_hi = upper_rate(dose, dose_max)
        result["period_min"], result["period_max"] = dose * weight_kg, rate_hi * weight_kg
        result["formula"] = f"{weight_kg:g} kg × {_range_text(dose, rate_hi, unit + '/kg')}"

    # Superficie corporal
    elif "M2_" in kind or "M²_" in kind:
        if height_cm is None or float(height_cm) <= 0:
            raise ValueError("Esta pauta requiere talla para calcular superficie corporal.")
        bsa = bsa_mosteller(float(height_cm), weight_kg)
        result["bsa"] = bsa
        rate_hi = upper_rate(dose, dose_max)
        if "DOSIS" in kind:
            lo, hi = dose * bsa, rate_hi * bsa
            result["per_dose_min"], result["per_dose_max"] = lo, hi
            result["formula"] = f"SC {bsa:.3f} m² × {_range_text(dose, rate_hi, unit + '/m²/dosis')}"
            if divisions:
                result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
            elif interval:
                nday = 24.0 / interval
                result["daily_min"], result["daily_max"] = lo * nday, hi * nday
        else:
            daily_lo, daily_hi = dose * bsa, rate_hi * bsa
            result["daily_min"], result["daily_max"] = daily_lo, daily_hi
            result["formula"] = f"SC {bsa:.3f} m² × {_range_text(dose, rate_hi, unit + '/m²/día')}"
            div = divisions or (24.0 / interval if interval else None)
            if div:
                result["per_dose_min"], result["per_dose_max"] = daily_lo / div, daily_hi / div
                result["interval_h"] = 24.0 / div
    else:
        raise ValueError("La estructura de esta pauta no permite cálculo numérico automático.")

    # Aplicar máximos cuando el significado es compatible con una dosis o exposición diaria.
    if result["per_dose_min"] is not None and single_cap is not None:
        old = (result["per_dose_min"], result["per_dose_max"])
        result["per_dose_min"] = min(result["per_dose_min"], single_cap)
        result["per_dose_max"] = min(result["per_dose_max"], single_cap)
        if old != (result["per_dose_min"], result["per_dose_max"]):
            result["caps"].append(f"máximo por dosis {fmt_num(single_cap, 2)} {unit}")

    if result["daily_min"] is not None and daily_cap is not None:
        old_daily = (result["daily_min"], result["daily_max"])
        result["daily_min"] = min(result["daily_min"], daily_cap)
        result["daily_max"] = min(result["daily_max"], daily_cap)
        if old_daily != (result["daily_min"], result["daily_max"]):
            result["caps"].append(f"máximo diario {fmt_num(daily_cap, 2)} {unit}")
            if result["per_dose_min"] is not None:
                div = divisions or (24.0 / result["interval_h"] if result["interval_h"] else None)
                if div:
                    result["per_dose_min"] = min(result["per_dose_min"], daily_cap / div)
                    result["per_dose_max"] = min(result["per_dose_max"], daily_cap / div)

    return result


def _render_rule_calculator(rule):
    """Calculadora embebida en una pauta concreta, sin depender de su estado editorial."""
    if not pediatric_rule_can_calculate(rule):
        return

    rule_id = str(rule.get("rule_id") or abs(hash((rule.get("indicacion"), rule.get("poblacion"), rule.get("via")))))
    safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", rule_id)
    needs_height = "M2_" in str(rule.get("tipo_dosis") or "").upper() or "M²_" in str(rule.get("tipo_dosis") or "").upper()

    st.markdown("#### 🧮 Calcular esta pauta")
    with st.form(f"ped_rule_calc_form_{safe_key}", border=True):
        c1, c2, c3 = st.columns(3)
        age_value = c1.number_input(
            "Edad", min_value=0.0, max_value=216.0, value=5.0, step=0.5,
            key=f"ped_age_{safe_key}",
        )
        age_unit = c1.selectbox(
            "Unidad", ["años", "meses", "días"], key=f"ped_age_unit_{safe_key}"
        )
        weight = c2.number_input(
            "Peso (kg)", min_value=0.1, max_value=250.0, value=20.0, step=0.1,
            key=f"ped_weight_{safe_key}",
        )
        height = None
        if needs_height:
            height = c3.number_input(
                "Talla (cm)", min_value=20.0, max_value=230.0, value=110.0, step=0.5,
                key=f"ped_height_{safe_key}",
            )
        else:
            c3.write(f"**Vía:** {rule.get('via') or '—'}")
            c3.write(f"**Pauta:** {pediatric_rule_dose_text(rule)}")
        submitted = st.form_submit_button("Calcular dosis", type="primary", use_container_width=True)

    result_key = f"ped_rule_calc_result_{safe_key}"
    if submitted:
        age_mo = age_to_months(age_value, age_unit)
        if not rule_applies_demographics(rule, age_mo, weight):
            st.session_state.pop(result_key, None)
            st.error("La edad o el peso ingresados no corresponden al rango de esta pauta.")
        else:
            try:
                result = calculate_loaded_pediatric_rule(rule, weight, height)
                st.session_state[result_key] = {
                    "result": result,
                    "weight": weight,
                    "height": height,
                    "age_value": age_value,
                    "age_unit": age_unit,
                }
            except ValueError as exc:
                st.session_state.pop(result_key, None)
                st.error(str(exc))

    saved = st.session_state.get(result_key)
    if not saved:
        return

    result = saved["result"]
    unit = result["unit"]
    cards = []
    if result.get("per_dose_min") is not None:
        cards.append(("Dosis por administración", _range_text(result["per_dose_min"], result["per_dose_max"], unit)))
    if result.get("daily_min") is not None:
        cards.append(("Dosis total diaria", _range_text(result["daily_min"], result["daily_max"], unit + "/día")))
    if result.get("rate_min") is not None:
        cards.append(("Velocidad / hora", _range_text(result["rate_min"], result["rate_max"], unit + "/h")))
    if result.get("period_min") is not None:
        cards.append(("Cantidad para el periodo", _range_text(result["period_min"], result["period_max"], unit)))
    if result.get("interval_h"):
        cards.append(("Intervalo", f"cada {fmt_num(result['interval_h'], 1)} h"))
    if result.get("bsa"):
        cards.append(("Superficie corporal", f"{result['bsa']:.3f} m²"))
    render_clinical_cards(cards)
    st.caption("Cálculo: " + result["formula"])
    if result.get("caps"):
        st.info("Máximo aplicado: " + " · ".join(result["caps"]))

    if result.get("per_dose_min") is not None and str(rule.get("permite_conversion_volumen") or "NO").upper() == "SI":
        with st.expander("Convertir la dosis a mL", expanded=False):
            with st.form(f"ped_rule_volume_form_{safe_key}", border=True):
                q1, q2 = st.columns(2)
                default_amount = 100000.0 if unit.upper().startswith("U") else 100.0
                label_value = q1.number_input(
                    f"Cantidad indicada en la presentación ({unit})",
                    min_value=0.0001, value=default_amount, step=1.0,
                    key=f"ped_label_value_{safe_key}",
                )
                label_ml = q2.number_input(
                    "Volumen de esa presentación (mL)", min_value=0.01, value=5.0, step=0.5,
                    key=f"ped_label_ml_{safe_key}",
                )
                cv = st.form_submit_button("Calcular volumen", use_container_width=True)
            volume_key = f"ped_rule_volume_result_{safe_key}"
            if cv:
                st.session_state[volume_key] = quantity_to_ml(
                    result["per_dose_min"], result["per_dose_max"], label_value, label_ml
                )
            vol = st.session_state.get(volume_key)
            if vol:
                render_clinical_cards([
                    ("Concentración", f"{fmt_num(vol['unit_per_ml'],3)} {unit}/mL"),
                    ("Volumen por administración", fmt_range(vol["min_ml"], vol["max_ml"], "mL")),
                ])


def show_pediatric_rules(rules):
    if not rules:
        return

    st.markdown("### Pautas de dosis")
    for idx, rule in enumerate(rules):
        title = (
            f"{rule.get('indicacion') or 'Sin indicación'} · "
            f"{rule.get('poblacion') or 'Pediatría'} · "
            f"{rule.get('via') or 'vía no consignada'}"
        )
        with st.expander(title, expanded=(idx == 0)):
            dose_text = pediatric_rule_dose_text(rule)
            interval = _pediatric_interval_hours(rule)
            freq_text = rule.get("frecuencia_texto") or (
                f"cada {fmt_num(interval,1)} h" if interval is not None else None
            )
            max_text = pediatric_rule_limits_text(rule)

            cards = [("Dosis registrada", dose_text)]
            if freq_text:
                cards.append(("Intervalo / frecuencia", freq_text))
            if max_text and str(max_text).strip().lower() not in {"no consignado", "—", "-"}:
                cards.append(("Máximos", max_text))
            render_clinical_cards(cards)

            d1, d2 = st.columns(2)
            d1.write(f"**Población:** {rule.get('poblacion') or '—'}")
            d2.write(f"**Vía:** {rule.get('via') or '—'}")
            if rule.get("duracion"):
                st.write(f"**Duración:** {rule['duracion']}")
            if rule.get("notas"):
                st.info(rule["notas"])
            if rule.get("nota_renal"):
                st.warning("Función renal: " + str(rule["nota_renal"]))

            _render_rule_calculator(rule)
            source_block(rule.get("fuente"), rule.get("url_fuente"), rule.get("fecha_revision"), rule.get("pagina_fuente"))

def page_pediatric():
    header(
        "Dosis pediátrica",
        "Seleccione un medicamento y abra la pauta correspondiente. Cada pauta con datos numéricos incluye su propia calculadora de dosis.",
    )
    if st.button("← Volver al inicio", key="ped_back_home"):
        go_to_module("Inicio", st.session_state.get("selected_med_id"))

    render_kpi_cards([
        ("Catálogo", COUNTS['medications'], "medicamentos"),
        ("Con pauta", COUNTS.get('pediatric_meds', 0), "medicamentos"),
        ("Pautas cargadas", COUNTS.get('pediatric_rules', 0), "pautas"),
    ])

    med = medication_picker("ped", "Medicamento")
    if not med:
        return

    rules = db.pediatric_rules(med["med_id"])
    if not rules:
        st.warning(f"**{med['principio_activo']}: SIN PAUTA PEDIÁTRICA CARGADA.**")
        return

    show_pediatric_rules(rules)

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
    header("Toxicología", "Manifestaciones, dosis o umbral de referencia, manejo y tratamiento específico.")
    if st.button("← Volver al inicio", key="tox_back_home"):
        go_to_module("Inicio", st.session_state.get("selected_med_id"))

    def _tox_clean(value):
        """Oculta códigos internos y marcadores sin valor clínico de la base."""
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        norm = normalize_text(s).replace("_", " ").strip()
        placeholders = {
            "sdte", "sin dato toxicologico establecido", "sin dato toxico establecido",
            "sin umbral numerico", "sin umbral numerico validado", "no disponible",
            "pendiente de fuente original", "pendiente", "na", "n a", "none", "null", "—", "-"
        }
        if norm in placeholders:
            return None
        return s

    def _tox_is_generic_symptoms(value):
        s = normalize_text(value or "")
        return s in {"sintomas generales", "síntomas generales", "pendiente de fuente original"}

    def _tox_numeric_references(value):
        """Extrae referencias cuantitativas explícitas sin convertirlas en diagnóstico clínico."""
        raw = str(value or "").replace(",", ".")
        if not raw.strip():
            return {"mgkg": [], "absolute_mg": [], "use_lower": False}

        mgkg = []
        for op, number in re.findall(r"([><≥≤]?)\s*(\d+(?:\.\d+)?)\s*mg\s*/\s*kg", raw, flags=re.I):
            try:
                mgkg.append((op or "", float(number)))
            except ValueError:
                pass

        absolute_mg = []
        # gramos absolutos: evita capturar el 'g' de mg.
        for op, number in re.findall(r"([><≥≤]?)\s*(\d+(?:\.\d+)?)\s*g\b", raw, flags=re.I):
            try:
                absolute_mg.append((op or "", float(number) * 1000.0, f"{number} g"))
            except ValueError:
                pass

        # mg absolutos, excluyendo mg/kg.
        for m in re.finditer(r"([><≥≤]?)\s*(\d+(?:\.\d+)?)\s*mg\b(?!\s*/\s*kg)", raw, flags=re.I):
            try:
                absolute_mg.append((m.group(1) or "", float(m.group(2)), f"{m.group(2)} mg"))
            except ValueError:
                pass

        def dedupe(items, key_index=1):
            seen=set(); out=[]
            for item in items:
                key=round(float(item[key_index]), 9)
                if key not in seen:
                    seen.add(key); out.append(item)
            return out

        norm = normalize_text(raw)
        use_lower = any(phrase in norm for phrase in [
            "lo que sea menor", "el que sea menor", "el menor", "lo que resulte menor",
            "whichever is less", "whichever is lower"
        ])
        return {
            "mgkg": dedupe(mgkg),
            "absolute_mg": dedupe(absolute_mg),
            "use_lower": use_lower,
        }

    tab1, tab2, tab3 = st.tabs(["Medicamentos", "Drogas/plaguicidas/metales", "Antídotos"])
    with tab1:
        med = medication_picker("tox", "Medicamento")
        if not med:
            return
        tox = db.toxicology(med["med_id"])
        if not tox:
            st.info("No hay una ficha toxicológica enlazada para este medicamento.")
            return

        st.markdown(f"### {med['principio_activo']} · {med['med_id']}")

        # Manifestaciones: siempre privilegiar la descripción clínica específica.
        detail = _tox_clean(tox.get("sintomas_intoxicacion_detallados")) or _tox_clean(tox.get("manifestaciones_clave"))
        if detail and not _tox_is_generic_symptoms(detail):
            st.markdown("#### 🚨 Manifestaciones de sobredosis / toxicidad")
            st.markdown(f'<div class="result-box"><strong>{_esc(detail)}</strong></div>', unsafe_allow_html=True)

        # Dosis / umbral: primero la capa revisada; si no existe, conservar el dato bibliográfico original.
        reviewed_threshold = _tox_clean(tox.get("dosis_toxica_corregida"))
        original_threshold = _tox_clean(tox.get("dosis_toxica_base"))
        if reviewed_threshold:
            dose_label = "Dosis / umbral de referencia"
            dose_value = reviewed_threshold
        elif original_threshold:
            dose_label = "Dosis tóxica registrada en la bibliografía"
            dose_value = original_threshold
        else:
            dose_label = "Dosis / umbral tóxico"
            dose_value = "No existe una dosis tóxica específica establecida."

        render_clinical_cards([(dose_label, dose_value)])

        # Manejo y terapia específica en lenguaje clínico, sin estados/códigos internos.
        management = _tox_clean(tox.get("manejo_corregido"))
        treatment = _tox_clean(tox.get("antidoto_especifico"))
        st.markdown("#### Manejo")
        if management:
            st.write(management)
        else:
            st.write("Manejo de soporte según el cuadro clínico y la exposición.")

        st.markdown("#### Tratamiento específico / antídoto")
        if treatment:
            st.write(treatment)
        else:
            st.write("No existe un antídoto específico registrado para esta ficha.")

        # Calculadora de exposición: usa el umbral estructurado cuando existe y,
        # si no, intenta utilizar únicamente referencias numéricas explícitas del texto cargado.
        threshold = as_float(tox.get("umbral_mgkg_automatizable"))
        reference_text = reviewed_threshold or original_threshold or ""
        parsed_refs = _tox_numeric_references(reference_text)
        has_text_reference = bool(parsed_refs["mgkg"] or parsed_refs["absolute_mg"])

        if threshold is not None or has_text_reference:
            st.markdown("#### 🧮 Calculadora de exposición")
            med_key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(med.get("med_id") or "tox"))
            entry_mode = st.radio(
                "Forma de ingresar la exposición",
                ["Cantidad total en mg", "Comprimidos / cápsulas", "Líquido"],
                horizontal=True,
                key=f"tox_mode_{med_key}",
            )

            with st.form(f"tox_exp_{med_key}"):
                weight = st.number_input("Peso del paciente (kg)", min_value=0.1, value=20.0, step=0.1)
                if entry_mode == "Cantidad total en mg":
                    total_mg = st.number_input("Cantidad total administrada/ingerida (mg)", min_value=0.0, value=500.0, step=50.0)
                elif entry_mode == "Comprimidos / cápsulas":
                    c1, c2 = st.columns(2)
                    units = c1.number_input("Número de unidades", min_value=0.0, value=1.0, step=1.0)
                    mg_per_unit = c2.number_input("mg por unidad", min_value=0.0, value=500.0, step=50.0)
                    total_mg = units * mg_per_unit
                else:
                    c1, c2 = st.columns(2)
                    volume_ml = c1.number_input("Volumen total (mL)", min_value=0.0, value=10.0, step=1.0)
                    concentration = c2.number_input("Concentración (mg/mL)", min_value=0.0, value=100.0, step=10.0)
                    total_mg = volume_ml * concentration

                submit = st.form_submit_button("Calcular exposición", use_container_width=True)

            if submit:
                exposure, _ = calculate_exposure_mgkg(total_mg, weight, None)
                cards = [
                    ("Cantidad total", f"{fmt_num(total_mg,2)} mg"),
                    ("Exposición calculada", f"{fmt_num(exposure,2)} mg/kg"),
                ]

                ratio = None
                reference_label = None

                if threshold is not None:
                    ratio = exposure / threshold if threshold > 0 else None
                    reference_label = tox.get("etiqueta_umbral") or f"{threshold:g} mg/kg"
                else:
                    mgkg_refs = parsed_refs["mgkg"]
                    abs_refs = parsed_refs["absolute_mg"]

                    # Si el propio texto indica 'lo que sea menor', calcula ese límite combinado.
                    if parsed_refs["use_lower"] and len(mgkg_refs) == 1 and len(abs_refs) == 1:
                        op_kg, kg_ref = mgkg_refs[0]
                        op_abs, abs_mg, abs_label = abs_refs[0]
                        by_weight_mg = kg_ref * weight
                        applicable_mg = min(by_weight_mg, abs_mg)
                        applicable_mgkg = applicable_mg / weight
                        reference_label = (
                            f"{op_kg or '≥'}{kg_ref:g} mg/kg o {op_abs or '≥'}{abs_label}; "
                            f"para {fmt_num(weight,2)} kg corresponde a {fmt_num(applicable_mg,2)} mg "
                            f"({fmt_num(applicable_mgkg,2)} mg/kg), usando el menor"
                        )
                        ratio = exposure / applicable_mgkg if applicable_mgkg > 0 else None
                    elif len(mgkg_refs) == 1:
                        op, kg_ref = mgkg_refs[0]
                        reference_label = f"{op}{kg_ref:g} mg/kg"
                        ratio = exposure / kg_ref if kg_ref > 0 else None
                    elif len(abs_refs) == 1:
                        op, abs_mg, abs_label = abs_refs[0]
                        reference_label = f"{op}{abs_label} total"
                        ratio = total_mg / abs_mg if abs_mg > 0 else None
                    else:
                        refs=[]
                        refs += [f"{op}{v:g} mg/kg" for op,v in mgkg_refs]
                        refs += [f"{op}{label} total" for op,_,label in abs_refs]
                        if refs:
                            reference_label = " · ".join(refs)

                if reference_label:
                    cards.append(("Referencia del registro", reference_label))
                render_clinical_cards(cards)

                if ratio is not None:
                    st.caption(f"Relación matemática exposición/referencia: {fmt_num(ratio,2)}×. La interpretación clínica depende del escenario, tiempo y formulación.")
                elif reference_label:
                    st.caption("La ficha contiene varias referencias numéricas; se muestran sin elegir automáticamente una como umbral clínico.")

        # Trazabilidad original solo cuando contiene información útil y no redundante.
        original_symptoms = _tox_clean(tox.get("sintomas_base"))
        original_management = _tox_clean(tox.get("antidoto_manejo_base"))
        original_items = []
        if original_threshold and normalize_text(original_threshold) != normalize_text(reviewed_threshold or ""):
            original_items.append(("Dosis registrada originalmente", original_threshold))
        if original_symptoms and not _tox_is_generic_symptoms(original_symptoms) and normalize_text(original_symptoms) != normalize_text(detail or ""):
            original_items.append(("Manifestaciones registradas originalmente", original_symptoms))
        if original_management and normalize_text(original_management) != normalize_text(management or ""):
            original_items.append(("Manejo/antídoto registrado originalmente", original_management))

        if original_items:
            with st.expander("Ver información bibliográfica original"):
                for label, value in original_items:
                    st.write(f"**{label}:** {value}")

        # Fuentes al final, evitando exponer códigos de validación internos.
        symptom_source = _tox_clean(tox.get("fuente_sintomas_detallados"))
        main_source = _tox_clean(tox.get("fuente_principal"))
        if symptom_source:
            st.caption("Fuente de manifestaciones: " + symptom_source)
        if main_source:
            if str(main_source).startswith(("http://", "https://")):
                st.link_button("Abrir fuente", main_source)
            elif not symptom_source or normalize_text(main_source) != normalize_text(symptom_source):
                st.caption("Fuente principal: " + main_source)

    with tab2:
        q = st.text_input("Buscar tóxico no farmacológico", key="other_tox_q")
        hits = db.search_other_tox(q)
        if hits:
            names = [r.get("toxico") or "—" for r in hits]
            pick = st.selectbox("Tóxico", names, key="other_tox_sel")
            r = hits[names.index(pick)]
            st.markdown("#### Manifestaciones")
            st.write(r.get("sintomas_base") or "Sin manifestaciones registradas.")
            st.markdown("#### Tratamiento / antídoto")
            st.write(r.get("antidoto_tratamiento_base") or "No hay tratamiento específico registrado.")

    with tab3:
        q = st.text_input("Buscar tóxico, síndrome o antídoto", key="antidote_q")
        hits = db.search_antidotes(q)
        if hits:
            labels = [f"{r.get('toxico_sindrome') or '—'} → {r.get('antidoto_base') or '—'}" for r in hits]
            pick = st.selectbox("Resultado", labels, key="antidote_sel")
            r = hits[labels.index(pick)]
            render_clinical_cards([
                ("Dosis registrada", r.get("dosis_base") or "No consignada"),
                ("Observaciones", r.get("observaciones_base") or "Sin observaciones adicionales"),
            ])

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
