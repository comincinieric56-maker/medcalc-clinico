from pathlib import Path
import math
import re

import streamlit as st

from supabase_repository import SupabaseRepository, SCHEMA_VERSION, normalize_text
import medcalc_engine as _medcalc_engine
from electrolyte_engine import (
    component_amounts_from_solution_volume,
    component_amounts_from_volume,
    evaluate_administration_options,
    evaluate_condition as evaluate_electrolyte_condition,
    evaluate_rules as evaluate_electrolyte_rules,
    merge_electrolyte_loads,
    prepare_infusion,
    prepare_premixed_infusion,
    product_units_for_mmol,
    product_volume_for_mmol,
    validate_administration_limit,
    total_body_water_l,
    free_water_deficit_l,
    predicted_delta_na_after_infusate,
    corrected_sodium_for_hyperglycemia,
    calculated_serum_osmolality_mosm_kg,
    effective_osmolality_mosm_kg,
    corrected_calcium_mmol_l,
    anion_gap_mmol_l,
    albumin_corrected_anion_gap_mmol_l,
    delta_ratio,
    interpret_acid_base,
    delta_gap_mmol_l,
    delta_bicarbonate_mmol_l,
    corrected_bicarbonate_from_delta_gap,
    interpret_delta_ratio_value,
    henderson_hasselbalch_hco3_mmol_l,
    comprehensive_acid_base_interpretation,
    barometric_pressure_from_altitude_mm_hg,
    alveolar_oxygen_pressure_mm_hg,
    aa_gradient_mm_hg,
    expected_aa_gradient_mm_hg,
    pf_ratio_mm_hg,
    arterial_oxygen_content_ml_dl,
    osmolar_gap_mosm_kg,
    urine_anion_gap_mmol_l,
    stewart_sida_meq_l,
    stewart_side_meq_l,
    strong_ion_gap_meq_l,
    supported_laboratory_units,
    laboratory_value_to_mmol_l,
    mmol_l_to_laboratory_value,
    glucose_to_mmol_l,
    albumin_to_g_l,
)


# -----------------------------------------------------------------------------
# CAPA DE COMPATIBILIDAD DEL MOTOR
# -----------------------------------------------------------------------------
# MedCalc ha tenido varias revisiones de medcalc_engine.py. La interfaz no debe
# dejar de arrancar si Streamlit conserva temporalmente una versión anterior.
# Se usan las funciones del motor cuando existen y, para auxiliares puramente
# deterministas, se aporta un fallback local equivalente.

def _engine_attr(name, fallback):
    return getattr(_medcalc_engine, name, fallback)


def _fallback_as_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def _fallback_age_to_months(value, unit):
    value = float(value)
    if unit == "días":
        return value / 30.4375
    if unit == "meses":
        return value
    return value * 12.0


def _fallback_rule_applies_demographics(rule, age_months, weight_kg):
    amin = _fallback_as_float(rule.get("edad_min_meses"))
    amax = _fallback_as_float(rule.get("edad_max_meses"))
    wmin = _fallback_as_float(rule.get("peso_min_kg"))
    wmax = _fallback_as_float(rule.get("peso_max_kg"))
    min_excl = str(rule.get("peso_min_exclusivo") or "").strip().upper() in {"SI", "SÍ", "TRUE", "1"}
    max_excl = str(rule.get("peso_max_exclusivo") or "").strip().upper() in {"SI", "SÍ", "TRUE", "1"}
    if amin is not None and age_months < amin:
        return False
    if amax is not None and age_months >= amax:
        return False
    if wmin is not None and ((weight_kg <= wmin) if min_excl else (weight_kg < wmin)):
        return False
    if wmax is not None and ((weight_kg >= wmax) if max_excl else (weight_kg > wmax)):
        return False
    return True


def _fallback_ckdepi_2021(age_years, sex, creatinine_mg_dl):
    if age_years < 18 or creatinine_mg_dl <= 0:
        return None
    if sex == "Mujer":
        k, alpha, factor = 0.7, -0.241, 1.012
    else:
        k, alpha, factor = 0.9, -0.302, 1.0
    ratio = creatinine_mg_dl / k
    return 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938 ** age_years * factor


def _fallback_cockcroft_gault(age_years, sex, weight_kg, creatinine_mg_dl):
    if age_years <= 0 or weight_kg <= 0 or creatinine_mg_dl <= 0:
        return None
    value = ((140 - age_years) * weight_kg) / (72 * creatinine_mg_dl)
    return value * 0.85 if sex == "Mujer" else value


def _fallback_bedside_schwartz(height_cm, creatinine_mg_dl):
    if height_cm <= 0 or creatinine_mg_dl <= 0:
        return None
    return 0.413 * height_cm / creatinine_mg_dl


def _fallback_bsa_mosteller(height_cm, weight_kg):
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def _fallback_normalize_crcl_to_173(crcl_ml_min, bsa_m2):
    if crcl_ml_min is None or bsa_m2 is None or bsa_m2 <= 0:
        return None
    return crcl_ml_min * 1.73 / bsa_m2


def _fallback_quantity_to_ml(min_value, max_value, label_value, label_ml):
    if label_value <= 0 or label_ml <= 0:
        raise ValueError("La concentración debe ser mayor que cero.")
    concentration = label_value / label_ml
    return {
        "unit_per_ml": concentration,
        "min_ml": min_value / concentration,
        "max_ml": max_value / concentration,
    }


def _fallback_ckd_g_stage(egfr):
    value = _fallback_as_float(egfr)
    if value is None:
        return "—", "eGFR no disponible"
    if value >= 90:
        return "G1", "normal o alto"
    if value >= 60:
        return "G2", "levemente disminuido"
    if value >= 45:
        return "G3a", "leve-moderadamente disminuido"
    if value >= 30:
        return "G3b", "moderada-severamente disminuido"
    if value >= 15:
        return "G4", "severamente disminuido"
    return "G5", "falla renal"


def _fallback_dosing_band_from_egfr(egfr):
    value = _fallback_as_float(egfr)
    if value is None:
        return None, "eGFR no disponible"
    # Seguridad: las columnas históricas FR-001 están expresadas como CrCl.
    # No se selecciona una columna CrCl usando eGFR CKD-EPI.
    return None, "No inferida: eGFR no se intercambia con CrCl"


def _fallback_stage_to_dosing_band(stage):
    stage = str(stage or "").strip()
    if stage not in {"G1", "G2", "G3a", "G3b", "G4", "G5"}:
        return None, "Estadio KDIGO no reconocido."
    return None, (
        "El estadio KDIGO no se convierte automáticamente en una banda de dosificación CrCl. "
        "Use el valor y la métrica renal exigidos por la ficha o regla específica."
    )


def _fallback_renal_biblio_band(crcl_ml_min):
    value = _fallback_as_float(crcl_ml_min)
    if value is None:
        return None
    if value >= 50:
        return "crcl_100_50"
    if value >= 10:
        return "crcl_50_10"
    return "crcl_lt10"


def _fallback_calculate_exposure_mgkg(total_mg, weight_kg, threshold_mgkg=None):
    if weight_kg <= 0:
        raise ValueError("El peso debe ser mayor que cero.")
    exposure = total_mg / weight_kg
    ratio = None
    if threshold_mgkg is not None and threshold_mgkg > 0:
        ratio = exposure / threshold_mgkg
    return exposure, ratio


def _fallback_select_renal_rule(rules, crcl, crcl_normalized, egfr, hemodialysis=False):
    if hemodialysis:
        dialysis = [r for r in rules if str(r.get("tipo_regla") or "").upper() == "DIALISIS"]
        if dialysis:
            return dialysis[0], None
    for rule in rules:
        if str(rule.get("tipo_regla") or "").upper() == "DIALISIS":
            continue
        metric = rule.get("metrica_renal")
        if metric == "CrCl_CG_mL_min":
            value = crcl
        elif metric in {"CrCl_mL_min_1_73m2", "CrCl_normalizado_mL_min_1_73m2"}:
            value = crcl_normalized
        elif metric == "eGFR_CKDEPI_mL_min_1_73m2":
            value = egfr
        else:
            value = None
        lo = _fallback_as_float(rule.get("limite_inferior"))
        hi = _fallback_as_float(rule.get("limite_superior"))
        li = str(rule.get("inferior_inclusivo") or "").upper() in {"SI", "SÍ", "TRUE", "1"}
        ui = str(rule.get("superior_inclusivo") or "").upper() in {"SI", "SÍ", "TRUE", "1"}
        if lo is None and hi is None:
            return rule, value
        if value is None:
            continue
        if lo is not None and ((value < lo) if li else (value <= lo)):
            continue
        if hi is not None and ((value > hi) if ui else (value >= hi)):
            continue
        return rule, value
    return None, None


age_to_months = _engine_attr("age_to_months", _fallback_age_to_months)
as_float = _engine_attr("as_float", _fallback_as_float)
bedside_schwartz = _engine_attr("bedside_schwartz", _fallback_bedside_schwartz)
bsa_mosteller = _engine_attr("bsa_mosteller", _fallback_bsa_mosteller)
calculate_exposure_mgkg = _engine_attr("calculate_exposure_mgkg", _fallback_calculate_exposure_mgkg)
calculate_pediatric_dose = getattr(_medcalc_engine, "calculate_pediatric_dose", None)
ckdepi_2021 = _engine_attr("ckdepi_2021", _fallback_ckdepi_2021)
cockcroft_gault = _engine_attr("cockcroft_gault", _fallback_cockcroft_gault)
normalize_crcl_to_173 = _engine_attr("normalize_crcl_to_173", _fallback_normalize_crcl_to_173)
quantity_to_ml = _engine_attr("quantity_to_ml", _fallback_quantity_to_ml)
renal_biblio_band = _engine_attr("renal_biblio_band", _fallback_renal_biblio_band)
ckd_g_stage = _engine_attr("ckd_g_stage", _fallback_ckd_g_stage)
dosing_band_from_egfr = _fallback_dosing_band_from_egfr
stage_to_dosing_band = _fallback_stage_to_dosing_band
rule_applies_demographics = _engine_attr("rule_applies_demographics", _fallback_rule_applies_demographics)
select_renal_rule = _engine_attr("select_renal_rule", _fallback_select_renal_rule)

APP_VERSION = "V8.1.4 · HIDROELECTROLITOS · PANEL INTEGRAL + GASES ARTERIALES"
REVIEW_DATE = "2026-09-05"
ROOT = Path(__file__).parent
FALLBACK_DB_PATH = ROOT / "medcalc.db"
CITUC_URL = "https://cituc.uc.cl/"
PAGES = ["Inicio", "Dosis pediátrica", "Ajuste renal", "Toxicología", "Hidroelectrolitos", "Base y fuentes"]

st.set_page_config(
    page_title="MedCalc Clínico",
    page_icon="💊",
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


st.markdown(
    """
    <style>
      /* V7.8.0 · PEDIATRÍA COMPACTA + MICROINTERACCIONES */
      details {
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease !important;
        animation:mcFadeUp .34s ease both;
      }
      details:hover {
        transform:translateY(-2px);
        box-shadow:0 12px 30px rgba(31,72,108,.08);
        border-color:#cbdce8 !important;
      }
      details[open] {box-shadow:0 14px 34px rgba(31,72,108,.075);}
      details > summary {transition:background .16s ease,color .16s ease;}
      details > summary:hover {background:rgba(228,240,247,.55);}

      .ped-rule-summary {
        display:flex;flex-wrap:wrap;gap:7px;margin:.15rem 0 .55rem;
      }
      .ped-rule-summary span {
        display:inline-flex;align-items:center;min-height:34px;padding:7px 10px;border-radius:11px;
        background:linear-gradient(135deg,#f5f9fc,#eef6f8);border:1px solid #dce7ee;
        color:#29465b;font-size:.78rem;line-height:1.25;
      }
      .ped-mini-meta {
        margin-top:1.75rem;padding:8px 10px;border-radius:11px;background:#f5f8fb;border:1px solid #e1e8ef;
        color:#5e7183;font-size:.75rem;line-height:1.45;
      }
      .ped-result-pop {
        display:flex;align-items:center;justify-content:space-between;gap:12px;
        margin:.55rem 0 .3rem;padding:12px 14px;border-radius:14px;
        background:linear-gradient(135deg,#eaf8f2,#e7f6f7);border:1px solid #cde9df;
        box-shadow:0 8px 22px rgba(37,126,98,.08);animation:mcResultPop .28s ease both;
      }
      .ped-result-pop span {font-size:.67rem;font-weight:900;letter-spacing:.08em;color:#34725d;}
      .ped-result-pop strong {font-size:1.22rem;line-height:1.1;color:#126b4b;text-align:right;}

      div[data-testid="stForm"] {
        transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease;
      }
      div[data-testid="stForm"]:hover {
        border-color:#cbdde7 !important;box-shadow:0 8px 24px rgba(31,72,108,.05);
      }
      div.stButton > button, div[data-testid="stLinkButton"] > a {
        transition:transform .16s ease,box-shadow .16s ease,filter .16s ease !important;
      }
      div.stButton > button:hover, div[data-testid="stLinkButton"] > a:hover {
        transform:translateY(-2px) !important;filter:saturate(1.06);
        box-shadow:0 10px 22px rgba(32,91,120,.13) !important;
      }
      div.stButton > button:active {transform:translateY(0) scale(.99) !important;}
      [data-testid="stAlert"] {animation:mcAlertIn .25s ease both;}

      @keyframes mcResultPop {
        from {opacity:0;transform:scale(.985) translateY(5px);}
        to {opacity:1;transform:scale(1) translateY(0);}
      }
      @keyframes mcAlertIn {
        from {opacity:0;transform:translateY(4px);}
        to {opacity:1;transform:translateY(0);}
      }

      @media (max-width: 900px) {
        .ped-result-pop {align-items:flex-start;flex-direction:column;}
        .ped-result-pop strong {text-align:left;}
      }
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {animation-duration:.001ms !important;animation-iteration-count:1 !important;transition-duration:.001ms !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# V7.7.0 · CAPA VISUAL — BUSCADOR CENTRAL
# Se añade como override para preservar todas las clases y comportamientos previos.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --mc-bg:#f3f7fb;
        --mc-card:#ffffff;
        --mc-ink:#10243a;
        --mc-muted:#66788c;
        --mc-line:#dfe8f1;
        --mc-primary:#166d78;
        --mc-primary-2:#247fbc;
        --mc-cyan:#15a8b5;
        --mc-blue:#3478c8;
        --mc-green:#2f8b68;
        --mc-amber:#b07119;
        --mc-red:#ad4a50;
        --mc-soft:#eef7f9;
        --mc-shadow:0 18px 55px rgba(28,58,91,.10);
      }

      html {scroll-behavior:smooth;}
      .stApp {
        background:
          radial-gradient(circle at 88% 6%, rgba(57,151,204,.13), transparent 25%),
          radial-gradient(circle at 8% 0%, rgba(38,157,161,.10), transparent 22%),
          linear-gradient(180deg,#f9fbfe 0%,#f3f7fb 58%,#eef4f8 100%);
      }
      .block-container {max-width:1340px;padding-top:1.15rem;padding-bottom:3.3rem;}

      .home-brand {
        display:flex;align-items:center;justify-content:space-between;gap:18px;
        margin:.2rem 0 1rem;animation:mcFadeUp .45s ease both;
      }
      .home-brand-main {display:flex;align-items:center;gap:12px;min-width:0;}
      .home-logo {
        width:46px;height:46px;border-radius:15px;display:flex;align-items:center;justify-content:center;
        background:linear-gradient(145deg,#196f7c,#3b82c4);color:white;font-size:1.35rem;
        box-shadow:0 11px 28px rgba(31,116,145,.22);
      }
      .home-brand-title {font-size:1.42rem;font-weight:900;letter-spacing:-.03em;color:#10243a;line-height:1.05;}
      .home-brand-sub {font-size:.78rem;color:#718397;margin-top:3px;}
      .live-pill {
        display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border-radius:999px;
        background:rgba(255,255,255,.78);border:1px solid #dce7ef;color:#466177;
        font-size:.72rem;font-weight:800;box-shadow:0 7px 20px rgba(35,66,94,.06);white-space:nowrap;
      }
      .live-dot {width:8px;height:8px;border-radius:50%;background:#35a879;box-shadow:0 0 0 0 rgba(53,168,121,.45);animation:mcPulse 2s infinite;}

      .search-hero {
        position:relative;overflow:hidden;border:1px solid rgba(201,220,232,.95);border-radius:28px;
        background:linear-gradient(135deg,rgba(255,255,255,.96) 0%,rgba(239,248,251,.94) 60%,rgba(239,245,253,.92) 100%);
        padding:26px 28px 21px;margin:.25rem 0 .5rem;box-shadow:var(--mc-shadow);animation:mcFadeUp .52s ease both;
      }
      .search-hero:before {
        content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-85px;top:-125px;
        background:radial-gradient(circle,rgba(37,129,183,.19),rgba(37,129,183,0) 70%);animation:mcFloat 8s ease-in-out infinite;
      }
      .search-hero:after {
        content:"";position:absolute;width:180px;height:180px;border-radius:50%;left:-80px;bottom:-115px;
        background:radial-gradient(circle,rgba(23,151,153,.15),rgba(23,151,153,0) 72%);
      }
      .search-kicker {
        display:inline-flex;align-items:center;gap:7px;padding:6px 10px;border-radius:999px;background:#e9f5f7;
        border:1px solid #d5eaed;color:#216570;font-size:.69rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;margin-bottom:.8rem;
      }
      .search-title {font-size:2.35rem;font-weight:930;letter-spacing:-.045em;line-height:1.04;color:#10243a;margin:0 0 .55rem;max-width:850px;}
      .search-copy {font-size:.98rem;line-height:1.55;color:#63778c;max-width:870px;margin-bottom:.65rem;}
      .search-chips {display:flex;flex-wrap:wrap;gap:7px;margin-top:.72rem;}
      .search-chip {padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.72);border:1px solid #dfe9f0;color:#536b7f;font-size:.74rem;font-weight:750;}

      [data-testid="stTextInput"] label p,[data-testid="stSelectbox"] label p {font-weight:800;color:#344b61;}
      [data-testid="stTextInput"] input {min-height:58px !important;font-size:1.06rem !important;padding-left:17px !important;border-radius:16px !important;}
      [data-testid="stTextInput"] div[data-baseweb="input"] > div {
        border-radius:16px !important;border:1px solid #d7e4ed !important;background:rgba(255,255,255,.97) !important;
        box-shadow:0 9px 28px rgba(25,57,88,.07) !important;transition:border-color .18s ease,box-shadow .18s ease,transform .18s ease;
      }
      [data-testid="stTextInput"] div[data-baseweb="input"] > div:focus-within {
        border-color:#5ca8b2 !important;box-shadow:0 0 0 4px rgba(35,137,151,.10),0 12px 32px rgba(25,57,88,.09) !important;transform:translateY(-1px);
      }

      /* Buscador principal XL: es el elemento funcional dominante del Inicio */
      .st-key-home_med_query {margin-top:.35rem;margin-bottom:.45rem;}
      .st-key-home_med_query label p {
        font-size:1.08rem !important;font-weight:900 !important;color:#18354d !important;
        margin-bottom:.42rem !important;letter-spacing:-.01em;
      }
      .st-key-home_med_query div[data-baseweb="input"] > div {
        min-height:78px !important;border-radius:20px !important;border:1.5px solid #c8dce8 !important;
        background:rgba(255,255,255,.99) !important;
        box-shadow:0 15px 38px rgba(25,57,88,.10) !important;
      }
      .st-key-home_med_query div[data-baseweb="input"] > div:focus-within {
        border-color:#4196a4 !important;
        box-shadow:0 0 0 5px rgba(35,137,151,.12),0 18px 42px rgba(25,57,88,.12) !important;
        transform:translateY(-1px);
      }
      .st-key-home_med_query input {
        min-height:78px !important;font-size:1.24rem !important;font-weight:650 !important;
        padding:0 22px !important;border-radius:20px !important;color:#122c43 !important;
      }
      .st-key-home_med_query input::placeholder {
        color:#7890a4 !important;opacity:1 !important;font-size:1.13rem !important;font-weight:500 !important;
      }
      .st-key-home_med_select label p {
        font-size:1rem !important;font-weight:850 !important;color:#344b61 !important;
      }
      .st-key-home_med_select div[data-baseweb="select"] > div {
        min-height:62px !important;border-radius:18px !important;
        box-shadow:0 10px 28px rgba(25,57,88,.07) !important;
      }
      [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height:54px !important;border-radius:16px !important;border:1px solid #dbe6ee !important;
        background:rgba(255,255,255,.95) !important;box-shadow:0 7px 22px rgba(25,57,88,.05) !important;
      }

      .selected-med {
        position:relative;border:1px solid #dce8ef;border-radius:22px;background:rgba(255,255,255,.90);
        padding:17px 19px;margin:.95rem 0 .8rem;box-shadow:0 10px 30px rgba(27,59,88,.065);animation:mcFadeUp .4s ease both;
      }
      .selected-med-top {display:flex;align-items:flex-start;justify-content:space-between;gap:12px;flex-wrap:wrap;}
      .selected-med-label {font-size:.68rem;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:#78899a;margin-bottom:4px;}
      .selected-med-name {font-size:1.62rem;font-weight:920;letter-spacing:-.03em;color:#10243a;line-height:1.08;}
      .med-id-pill {padding:6px 10px;border-radius:999px;background:#eef5fb;border:1px solid #dce8f3;color:#47677f;font-size:.72rem;font-weight:850;white-space:nowrap;}

      .home-section-title {font-size:1.08rem;font-weight:900;color:#173049;margin:1.05rem 0 .15rem;}
      .home-section-copy {font-size:.82rem;color:#718397;margin-bottom:.72rem;}
      .module-card {
        border:1px solid #dfe8ef;border-radius:21px;padding:18px 18px 15px;background:rgba(255,255,255,.94);
        min-height:180px;box-shadow:0 9px 28px rgba(31,56,72,.055);
        transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease;animation:mcFadeUp .5s ease both;
      }
      .module-card:hover {transform:translateY(-4px);box-shadow:0 17px 42px rgba(29,61,91,.10);border-color:#cddfe9;}
      .module-icon {display:inline-flex;width:40px;height:40px;align-items:center;justify-content:center;border-radius:13px;background:linear-gradient(145deg,#edf7f8,#edf3fb);font-size:1.22rem;margin-bottom:.65rem;}
      .module-title {font-size:1.04rem;font-weight:900;color:#173049;margin-bottom:.28rem;}
      .module-count {font-size:.78rem;color:#718397;margin-bottom:.7rem;}
      .chip {background:#f1f6f9;border-color:#e0e9ef;color:#4c6679;}

      .kpi-grid {margin:.45rem 0 1rem;}
      .kpi-card {transition:transform .16s ease,box-shadow .16s ease;background:rgba(255,255,255,.83);}
      .kpi-card:hover {transform:translateY(-2px);box-shadow:0 12px 32px rgba(31,56,72,.075);}

      div.stButton > button {border-radius:14px !important;font-weight:820 !important;min-height:2.75rem;border-color:#d5e2ea !important;transition:transform .15s ease,box-shadow .15s ease,border-color .15s ease !important;}
      div.stButton > button:hover {transform:translateY(-2px);box-shadow:0 9px 23px rgba(26,58,87,.09);border-color:#a8cbd2 !important;}
      [data-testid="stSidebar"] {background:linear-gradient(180deg,#fbfdff 0%,#f4f8fb 100%);}
      .side-brand {border-radius:18px;box-shadow:0 8px 25px rgba(32,61,88,.055);}

      @keyframes mcFadeUp {from {opacity:0;transform:translateY(10px);}to {opacity:1;transform:translateY(0);}}
      @keyframes mcFloat {0%,100% {transform:translate(0,0) scale(1);}50% {transform:translate(-9px,11px) scale(1.04);}}
      @keyframes mcPulse {0% {box-shadow:0 0 0 0 rgba(53,168,121,.42);}70% {box-shadow:0 0 0 7px rgba(53,168,121,0);}100% {box-shadow:0 0 0 0 rgba(53,168,121,0);}}

      @media (max-width:768px) {
        .home-brand {align-items:flex-start;}.home-brand-title {font-size:1.2rem;}.live-pill {display:none;}
        .search-hero {padding:21px 18px 17px;border-radius:22px;}.search-title {font-size:1.85rem;}.search-copy {font-size:.91rem;}.selected-med-name {font-size:1.35rem;}
        .st-key-home_med_query div[data-baseweb="input"] > div,.st-key-home_med_query input {min-height:68px !important;}
        .st-key-home_med_query input {font-size:1.08rem !important;padding:0 16px !important;}
        .st-key-home_med_query input::placeholder {font-size:1rem !important;}
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


def renal_reference_rules_safe(med_id):
    """Obtiene referencias renales PUBLISHED sin exigir una versión concreta del repositorio.

    V7.7.4: compatibilidad hacia atrás. Si SupabaseRepository expone
    renal_reference_rules(), se usa. Si no, se derivan de renal_rules()
    filtrando las reglas no automatizables. Nunca las convierte en automáticas.
    """
    method = getattr(db, "renal_reference_rules", None)
    if callable(method):
        try:
            return method(med_id) or []
        except Exception:
            pass
    try:
        rules = db.renal_rules(med_id) or []
    except Exception:
        return []
    return [r for r in rules if str(r.get("automatizable") or "").upper() != "SI"]


def fmt_num(value, digits=1):
    if value is None:
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")



def _renal_mapping_options(regimen):
    """Extrae mapeos del tipo 150→75 mg/día de una regla renal estructurada."""
    raw = " ".join(str(regimen or "").split())
    if not raw:
        return {}
    pattern = re.compile(
        r"(?P<base>\d+(?:[.,]\d+)?)\s*[→>]\s*"
        r"(?P<target>\d+(?:[.,]\d+)?(?:\s*[–-]\s*\d+(?:[.,]\d+)?)?)\s*mg/d[ií]a",
        re.IGNORECASE,
    )
    out = {}
    for m in pattern.finditer(raw):
        base = m.group("base").replace(",", ".")
        target = re.sub(r"\s+", "", m.group("target").replace(",", "."))
        out[base] = target
    return out


def _renal_frequency_suffix(regimen):
    raw = " ".join(str(regimen or "").split())
    m = re.search(r"Administrar\s+([^.;]+)", raw, flags=re.IGNORECASE)
    if not m:
        return ""
    freq = m.group(1).strip().upper()
    # Mantener la tarjeta breve.
    if len(freq) > 32:
        return ""
    return f" · {freq}"


def renal_direct_instruction(regimen, range_text=None, hemodialysis=False, rule=None):
    """Devuelve una conducta renal breve y accionable para la tarjeta principal.

    La selección de banda ya ocurrió antes de llamar a esta función. Para decidir
    si una banda corresponde a función renal normal se usan los límites numéricos
    estructurados de la regla, no texto libre. El texto íntegro queda en auditoría.
    """
    raw = " ".join(str(regimen or "").split())
    if not raw or raw in {"—", "-"}:
        return "PAUTA RENAL NO CONSIGNADA"

    norm = normalize_text(raw)
    rule = rule or {}
    lower = as_float(rule.get("limite_inferior"))
    upper = as_float(rule.get("limite_superior"))
    rtype = str(rule.get("tipo_regla") or "").upper()

    # Conductas inequívocas de no ajuste.
    no_adjust_phrases = (
        "no requiere ajuste", "no se requiere ajuste", "sin ajuste renal",
        "no es necesario ajustar", "no precisa ajuste", "no dosage adjustment",
        "dosage adjustment is not necessary", "no dose adjustment",
    )
    if any(x in norm for x in no_adjust_phrases):
        return "NO REQUIERE AJUSTE RENAL · USAR DOSIS HABITUAL"

    # Si la regla seleccionada es la banda basal/normal y remite a la dosis normal,
    # la conducta clínica es inequívoca aunque el texto tenga símbolos/espacios distintos.
    normal_dose_phrases = (
        "dosis diaria total correspondiente a la indicacion en funcion renal normal",
        "dosis habitual segun indicacion", "dosis normal segun indicacion",
        "dosis de funcion renal normal", "dosis con funcion renal normal",
        "dosis diaria normal",
    )
    normal_structured_band = lower is not None and lower >= 60 and upper is None
    if any(x in norm for x in normal_dose_phrases) and normal_structured_band:
        return "NO REQUIERE AJUSTE RENAL · USAR DOSIS HABITUAL SEGÚN INDICACIÓN"

    # Restricciones claras.
    if "contraindicado" in norm or "contraindicada" in norm:
        return "CONTRAINDICADO EN ESTA FUNCIÓN RENAL"
    if any(x in norm for x in ("evitar", "no recomendado", "no se recomienda")):
        return "EVITAR / NO RECOMENDADO EN ESTA FUNCIÓN RENAL"

    # Hemodiálisis: conducta breve.
    if hemodialysis or rtype == "DIALISIS":
        if any(x in norm for x in ("no requiere dosis suplementaria", "sin dosis suplementaria", "no supplemental dose")):
            return "NO REQUIERE DOSIS SUPLEMENTARIA POST-HEMODIÁLISIS"
        if "suplement" in norm and not re.search(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|ui|u)\b", norm):
            return "ADMINISTRAR DOSIS SUPLEMENTARIA POST-HEMODIÁLISIS"
        first = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0].strip()
        if first and len(first) <= 120:
            return first.upper()
        return "USAR PAUTA ESPECÍFICA POST-HEMODIÁLISIS"

    # Si ya hay una pauta corta y concreta, conservarla.
    if len(raw) <= 110:
        return raw.upper()

    # Extraer una dosis prescriptiva simple si aparece en texto largo.
    compact = re.search(
        r"(\d+(?:[.,]\d+)?(?:\s*[–-]\s*\d+(?:[.,]\d+)?)?\s*"
        r"(?:mg|mcg|g|ml|ui|u)(?:/d[ií]a)?(?:\s+(?:cada\s+\d+(?:[.,]\d+)?\s*h|qd|bid|tid))?)",
        raw,
        flags=re.IGNORECASE,
    )
    if compact and "→" not in raw:
        return compact.group(1).upper()

    # No devolver frases vacías del tipo "ajustar según función renal".
    # Si la regla depende de una dosis objetivo/indicación y no puede reducirse a
    # una única dosis sin ese dato, se dice explícitamente qué falta.
    if any(x in norm for x in ("segun la indicacion", "segun indicacion", "dosis diaria objetivo", "dosis diaria normal", "mapeo desde")):
        return "SELECCIONE LA DOSIS HABITUAL OBJETIVO PARA OBTENER LA DOSIS RENAL"

    first = re.split(r"(?<=[.!?])\s+", raw, maxsplit=1)[0].strip()
    if first and len(first) <= 120:
        return first.upper()
    return "REQUIERE AJUSTE RENAL SEGÚN LA REGLA VALIDADA"

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
        "Hidroelectrolitos y reposición": "🧪",
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

def medication_picker(prefix, title="Medicamento", help_text=None, search_label=None, result_label=None):
    """Buscador explícito sobre todo el catálogo maestro Supabase.

    `search_label` y `result_label` permiten dar más protagonismo al buscador
    de Inicio sin alterar los selectores de los demás módulos.
    """
    query = st.text_input(
        search_label or f"Buscar {title.lower()}",
        placeholder="Escriba el nombre del medicamento…  Ej.: amoxi, aciclovir, gabapentina",
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
        result_label or title,
        labels,
        index=index,
        key=f"{prefix}_med_select",
        help=help_text or f"El selector proviene de la tabla maestra Supabase ({COUNTS.get('medications', '—')} MED-ID).",
    )
    row = hits[labels.index(picked)]
    st.session_state["selected_med_id"] = row["med_id"]
    return row


def status_badges(summary):
    c1, c2, c3, c4 = st.columns(4)
    ped_n = int(summary.get("pediatric_rule_count") or 0)
    ped_pub = int(summary.get("pediatric_published_count") or 0)
    ped_pending = int(summary.get("pediatric_pending_count") or 0)
    ren_n = int(summary.get("renal_rule_count") or 0)
    structured_ref_n = int(summary.get("renal_reference_rule_count") or 0)
    # Repositorios anteriores a V7.7.2 no exponen este contador en medication().
    # Se calcula de forma segura para que Inicio no oculte referencias renales.
    if structured_ref_n == 0 and summary.get("med_id"):
        structured_ref_n = len(renal_reference_rules_safe(summary["med_id"]))
    biblio_ref_n = int(summary.get("renal_biblio_count") or 0)
    ref_n = structured_ref_n + biblio_ref_n
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
    # La búsqueda es deliberadamente el primer elemento funcional de la pantalla.
    st.markdown(
        f'''<div class="home-brand">
              <div class="home-brand-main">
                <div class="home-logo">💊</div>
                <div>
                  <div class="home-brand-title">MedCalc Clínico</div>
                  <div class="home-brand-sub">Farmacología clínica estructurada · {APP_VERSION}</div>
                </div>
              </div>
              <div class="live-pill"><span class="live-dot"></span> Supabase conectado · {COUNTS['medications']} MED-ID</div>
            </div>''',
        unsafe_allow_html=True,
    )

    st.markdown(
        '''<div class="search-hero">
             <div class="search-kicker">🔎 Búsqueda central</div>
             <div class="search-title">¿Qué medicamento necesita consultar?</div>
             <div class="search-copy">Escriba el nombre genérico o una parte. La selección queda activa al pasar a Pediatría, Ajuste renal, Toxicología o Hidroelectrolitos.</div>
             <div class="search-chips">
               <span class="search-chip">👶 Dosis pediátrica</span>
               <span class="search-chip">🧮 Función renal</span>
               <span class="search-chip">☠️ Toxicología</span>
               <span class="search-chip">🧪 Hidroelectrolitos</span>
               <span class="search-chip">📚 Fuentes clínicas</span>
             </div>
           </div>''',
        unsafe_allow_html=True,
    )

    med = medication_picker(
        "home",
        "Medicamento",
        search_label="Buscar medicamento",
        result_label="Resultado",
        help_text=f"Catálogo maestro Supabase: {COUNTS.get('medications', '—')} MED-ID activos.",
    )
    if not med:
        return

    summary = db.medication(med["med_id"])
    if not summary:
        st.warning("No fue posible cargar el registro clínico seleccionado.")
        return

    st.markdown(
        f'''<div class="selected-med">
              <div class="selected-med-top">
                <div>
                  <div class="selected-med-label">Medicamento seleccionado</div>
                  <div class="selected-med-name">{_esc(summary['principio_activo'])}</div>
                </div>
                <div class="med-id-pill">{_esc(summary['med_id'])}</div>
              </div>
            </div>''',
        unsafe_allow_html=True,
    )
    status_badges(summary)

    ped_inds = db.pediatric_indications(summary["med_id"])
    renal_inds = db.renal_indications(summary["med_id"])
    renal_structured_refs = renal_reference_rules_safe(summary["med_id"])
    renal_refs = db.renal_biblio(summary["med_id"])
    tox = db.toxicology(summary["med_id"])

    st.markdown('<div class="home-section-title">Abrir módulo clínico</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-section-copy">La selección actual se mantiene automáticamente al cambiar de módulo.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="module-card"><div class="module-icon">👶</div><div class="module-title">Pediatría</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="module-count">{len(ped_inds)} indicación(es) con pauta cargada</div>', unsafe_allow_html=True)
        if ped_inds:
            for r in ped_inds[:5]:
                st.markdown(f'<span class="chip">{_esc(r["indicacion"])}</span>', unsafe_allow_html=True)
            if len(ped_inds) > 5:
                st.caption(f"+ {len(ped_inds)-5} escenarios adicionales")
        else:
            st.caption("Sin pauta pediátrica cargada.")
        if st.button("Abrir Pediatría →", key="home_open_ped", use_container_width=True, type="primary"):
            go_to_module("Dosis pediátrica", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="module-card"><div class="module-icon">🧮</div><div class="module-title">Ajuste renal</div>', unsafe_allow_html=True)
        total_renal_refs = len(renal_structured_refs) + len(renal_refs)
        st.markdown(f'<div class="module-count">{len(renal_inds)} regla(s) automáticas · {total_renal_refs} referencia(s)</div>', unsafe_allow_html=True)
        if renal_inds:
            for r in renal_inds[:4]:
                st.markdown(f'<span class="chip">{_esc(r["indicacion"])}</span>', unsafe_allow_html=True)
        elif renal_structured_refs:
            for r in renal_structured_refs[:4]:
                st.markdown(f'<span class="chip">REF · {_esc(r.get("indicacion") or "Ajuste renal")}</span>', unsafe_allow_html=True)
            st.caption("Referencia renal PUBLISHED validada; no se fuerza cálculo automático.")
        elif renal_refs:
            st.caption("Hay bibliografía renal enlazada aunque no exista regla automática.")
        else:
            st.caption("Sin recomendación renal cargada.")
        if st.button("Abrir Ajuste renal →", key="home_open_renal", use_container_width=True):
            go_to_module("Ajuste renal", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="module-card"><div class="module-icon">☠️</div><div class="module-title">Toxicología</div>', unsafe_allow_html=True)
        if tox:
            st.markdown('<div class="module-count">Ficha toxicológica disponible</div>', unsafe_allow_html=True)
            reviewed = str(tox.get("dosis_toxica_corregida") or "").strip()
            original = str(tox.get("dosis_toxica_base") or "").strip()
            placeholders = {
                "sdte", "sin dato toxicologico establecido", "sin dato toxico establecido",
                "sin umbral numerico", "sin umbral numerico validado", "pendiente",
                "pendiente de fuente original", "na", "n a", "none", "null", "—", "-"
            }
            if normalize_text(reviewed).replace("_", " ").strip() in placeholders:
                reviewed = ""
            if normalize_text(original).replace("_", " ").strip() in placeholders:
                original = ""
            if reviewed:
                st.write(f"**Referencia toxicológica:** {reviewed}")
            elif original:
                st.write(f"**Dosis registrada en bibliografía:** {original}")
            else:
                st.caption("Sin umbral numérico específico establecido.")
        else:
            st.caption("Sin ficha toxicológica cargada.")
        if st.button("Abrir Toxicología →", key="home_open_tox", use_container_width=True):
            go_to_module("Toxicología", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="module-card"><div class="module-icon">🧪</div><div class="module-title">Hidroelectrolitos</div>', unsafe_allow_html=True)
        erules = int(COUNTS.get("electrolyte_rules") or 0)
        eprotos = int(COUNTS.get("electrolyte_protocols") or 0)
        st.markdown(f'<div class="module-count">{erules} regla(s) · {eprotos} protocolo(s)</div>', unsafe_allow_html=True)
        st.caption("Na · K · Mg · Ca · fósforo · Cl · agua libre · ácido-base · reposición conjunta.")
        if st.button("Abrir Hidroelectrolitos →", key="home_open_electrolytes", use_container_width=True):
            go_to_module("Hidroelectrolitos", summary["med_id"])
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="home-section-title">Cobertura de la base clínica</div>', unsafe_allow_html=True)
    st.markdown('<div class="home-section-copy">Resumen global de los módulos conectados a Supabase.</div>', unsafe_allow_html=True)
    render_kpi_cards([
        ("Catálogo maestro", COUNTS['medications'], "medicamentos"),
        ("Pediatría", COUNTS['pediatric_rules'], "reglas"),
        ("Renal", COUNTS['renal_rules'], f"{COUNTS.get('renal_auto_rules', 0)} auto · {COUNTS.get('renal_reference_rules', 0) + COUNTS['renal_biblio']} ref."),
        ("Toxicología", COUNTS['toxicology'], "fichas"),
        ("Hidroelectrolitos", COUNTS.get('electrolyte_rules', 0), "reglas publicadas"),
    ])


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


def _pediatric_is_daily_dose_rule(rule):
    kind = str(rule.get("tipo_dosis") or "").upper().strip()
    return (
        kind.endswith("KG_DIA")
        or kind.endswith("KG_DIA_RANGE")
        or "M2_DIA" in kind
        or "M²_DIA" in kind
    )


def _pediatric_interval_options(rule):
    """Intervalos útiles para redistribuir una DOSIS TOTAL DIARIA.

    El intervalo original de la fuente siempre se conserva y aparece primero.
    Las alternativas son una herramienta matemática para dividir el total diario;
    no se presentan como recomendaciones de la fuente.
    """
    source = _pediatric_interval_hours(rule)
    div = _pediatric_divisions_per_day(rule)
    if source is None and div:
        source = 24.0 / div

    common = [24.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0]
    values = []
    if source and source > 0:
        values.append(float(source))
    for x in common:
        if not any(abs(x-y) < 1e-9 for y in values):
            values.append(x)
    return values, source


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


def calculate_loaded_pediatric_rule(rule, weight_kg, height_cm=None, interval_override_h=None):
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
    source_interval = _pediatric_interval_hours(rule)
    interval = source_interval
    divisions = _pediatric_divisions_per_day(rule)
    if interval_override_h is not None:
        interval_override_h = float(interval_override_h)
        if interval_override_h <= 0:
            raise ValueError("El intervalo de administración debe ser mayor que cero.")
        if not _pediatric_is_daily_dose_rule(rule):
            raise ValueError("Solo se puede redistribuir el intervalo en pautas expresadas como dosis total diaria.")
        interval = interval_override_h
        divisions = 24.0 / interval_override_h

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
        "source_interval_h": source_interval,
        "interval_override_applied": bool(
            interval_override_h is not None
            and (source_interval is None or abs(float(interval_override_h) - float(source_interval)) > 1e-9)
        ),
        "doses_per_day": None,
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
            result["doses_per_day"] = divisions
            result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
        elif interval:
            nday = 24.0 / interval
            result["doses_per_day"] = nday
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
            result["doses_per_day"] = divisions
            result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
        elif interval:
            nday = 24.0 / interval
            result["doses_per_day"] = nday
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
            result["doses_per_day"] = div
            result["per_dose_min"], result["per_dose_max"] = daily_lo / div, daily_hi / div
            result["interval_h"] = 24.0 / div
            result["formula"] += f" ÷ {div:g} dosis/día (cada {24.0/div:g} h)"

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
                result["doses_per_day"] = divisions
                result["daily_min"], result["daily_max"] = lo * divisions, hi * divisions
            elif interval:
                nday = 24.0 / interval
                result["doses_per_day"] = nday
                result["daily_min"], result["daily_max"] = lo * nday, hi * nday
        else:
            daily_lo, daily_hi = dose * bsa, rate_hi * bsa
            result["daily_min"], result["daily_max"] = daily_lo, daily_hi
            result["formula"] = f"SC {bsa:.3f} m² × {_range_text(dose, rate_hi, unit + '/m²/día')}"
            div = divisions or (24.0 / interval if interval else None)
            if div:
                result["doses_per_day"] = div
                result["per_dose_min"], result["per_dose_max"] = daily_lo / div, daily_hi / div
                result["interval_h"] = 24.0 / div
                result["formula"] += f" ÷ {div:g} dosis/día (cada {24.0/div:g} h)"
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


def _render_rule_calculator(rule, compact=False):
    """Calculadora embebida en una pauta concreta. En modo compacto reduce altura para vista en dos columnas."""
    if not pediatric_rule_can_calculate(rule):
        return

    rule_id = str(rule.get("rule_id") or abs(hash((rule.get("indicacion"), rule.get("poblacion"), rule.get("via")))))
    safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", rule_id)
    needs_height = "M2_" in str(rule.get("tipo_dosis") or "").upper() or "M²_" in str(rule.get("tipo_dosis") or "").upper()
    daily_rule = _pediatric_is_daily_dose_rule(rule)
    interval_options, source_interval = _pediatric_interval_options(rule) if daily_rule else ([], None)

    st.markdown("##### 🧮 Calcular esta pauta" if compact else "#### 🧮 Calcular esta pauta")

    with st.form(f"ped_rule_calc_form_{safe_key}", border=True):
        # En escritorio: edad, unidad, peso e intervalo/talla en una sola fila.
        c1, c2, c3, c4 = st.columns([1.05, 1.0, 1.05, 1.55])
        age_value = c1.number_input(
            "Edad", min_value=0.0, max_value=216.0, value=5.0, step=0.5,
            key=f"ped_age_{safe_key}",
        )
        age_unit = c2.selectbox(
            "Unidad", ["años", "meses", "días"], key=f"ped_age_unit_{safe_key}"
        )
        weight = c3.number_input(
            "Peso (kg)", min_value=0.1, max_value=250.0, value=20.0, step=0.1,
            key=f"ped_weight_{safe_key}",
        )

        height = None
        selected_interval = None
        if needs_height:
            height = c4.number_input(
                "Talla (cm)", min_value=20.0, max_value=230.0, value=110.0, step=0.5,
                key=f"ped_height_{safe_key}",
            )
        elif daily_rule:
            selected_interval = c4.selectbox(
                "Administrar cada",
                interval_options,
                index=0,
                format_func=lambda h: (
                    f"{fmt_num(h,1)} h · {fmt_num(24.0/h,2)} dosis/día"
                    + (" · FUENTE" if source_interval is not None and abs(h-source_interval) < 1e-9 else "")
                ),
                key=f"ped_interval_{safe_key}",
                help=(
                    "Redistribuye el mismo total diario entre las administraciones. "
                    "El intervalo marcado como FUENTE es el cargado en la bibliografía."
                ),
            )
        else:
            c4.markdown(
                f"<div class='ped-mini-meta'><b>Vía:</b> {rule.get('via') or '—'}<br>"
                f"<b>Pauta:</b> {pediatric_rule_dose_text(rule)}</div>",
                unsafe_allow_html=True,
            )

        if needs_height and daily_rule:
            selected_interval = st.selectbox(
                "Administrar la dosis calculada cada",
                interval_options,
                index=0,
                format_func=lambda h: (
                    f"cada {fmt_num(h,1)} h · {fmt_num(24.0/h,2)} dosis/día"
                    + (" · INTERVALO DE LA FUENTE" if source_interval is not None and abs(h-source_interval) < 1e-9 else "")
                ),
                key=f"ped_interval_{safe_key}",
            )

        submitted = st.form_submit_button("Calcular dosis", type="primary", use_container_width=True)

    result_key = f"ped_rule_calc_result_{safe_key}"
    if submitted:
        age_mo = age_to_months(age_value, age_unit)
        if not rule_applies_demographics(rule, age_mo, weight):
            st.session_state.pop(result_key, None)
            st.error("La edad o el peso ingresados no corresponden al rango de esta pauta.")
        else:
            try:
                result = calculate_loaded_pediatric_rule(rule, weight, height, selected_interval if daily_rule else None)
                st.session_state[result_key] = {
                    "result": result,
                    "weight": weight,
                    "height": height,
                    "age_value": age_value,
                    "age_unit": age_unit,
                    "selected_interval": selected_interval if daily_rule else None,
                }
            except ValueError as exc:
                st.session_state.pop(result_key, None)
                st.error(str(exc))

    saved = st.session_state.get(result_key)
    if not saved:
        return

    result = saved["result"]
    unit = result["unit"]

    # Resultado principal compacto: lo más útil aparece primero y con poca altura.
    if compact:
        primary = None
        if result.get("per_dose_min") is not None:
            primary = _range_text(result["per_dose_min"], result["per_dose_max"], unit)
            st.markdown(
                f"<div class='ped-result-pop'><span>DOSIS POR ADMINISTRACIÓN</span><strong>{primary}</strong></div>",
                unsafe_allow_html=True,
            )
        elif result.get("period_min") is not None:
            primary = _range_text(result["period_min"], result["period_max"], unit)
            st.markdown(
                f"<div class='ped-result-pop'><span>CANTIDAD PARA EL PERIODO</span><strong>{primary}</strong></div>",
                unsafe_allow_html=True,
            )
        elif result.get("rate_min") is not None:
            primary = _range_text(result["rate_min"], result["rate_max"], unit + "/h")
            st.markdown(
                f"<div class='ped-result-pop'><span>VELOCIDAD / HORA</span><strong>{primary}</strong></div>",
                unsafe_allow_html=True,
            )

        meta = []
        if result.get("daily_min") is not None:
            meta.append(f"Total diario: {_range_text(result['daily_min'], result['daily_max'], unit + '/día')}")
        if result.get("interval_h"):
            meta.append(f"Cada {fmt_num(result['interval_h'],1)} h")
        if result.get("doses_per_day"):
            meta.append(f"{fmt_num(result['doses_per_day'],2)} dosis/día")
        if result.get("bsa"):
            meta.append(f"SC {result['bsa']:.3f} m²")
        if meta:
            st.caption(" · ".join(meta))
    else:
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
        if result.get("doses_per_day"):
            cards.append(("Administraciones/día", fmt_num(result["doses_per_day"], 2)))
        if result.get("bsa"):
            cards.append(("Superficie corporal", f"{result['bsa']:.3f} m²"))
        render_clinical_cards(cards)
        st.caption("Cálculo: " + result["formula"])

    if result.get("interval_override_applied"):
        src_h = result.get("source_interval_h")
        src_txt = f"cada {fmt_num(src_h,1)} h" if src_h else "sin intervalo estructurado"
        st.warning(
            f"Intervalo modificado manualmente: fuente {src_txt}; cálculo mostrado cada {fmt_num(result.get('interval_h'),1)} h. "
            "Verifique que el intervalo elegido sea clínicamente válido para esta indicación."
        )
    if result.get("caps"):
        st.info("Máximo aplicado: " + " · ".join(result["caps"]))

    if result.get("per_dose_min") is not None and str(rule.get("permite_conversion_volumen") or "NO").upper() == "SI":
        with st.expander("💧 Convertir a mL", expanded=False):
            with st.form(f"ped_rule_volume_form_{safe_key}", border=True):
                q1, q2 = st.columns(2)
                default_amount = 100000.0 if unit.upper().startswith("U") else 100.0
                label_value = q1.number_input(
                    f"Cantidad en la presentación ({unit})",
                    min_value=0.0001, value=default_amount, step=1.0,
                    key=f"ped_label_value_{safe_key}",
                )
                label_ml = q2.number_input(
                    "Volumen (mL)", min_value=0.01, value=5.0, step=0.5,
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
                st.success(
                    f"**{fmt_range(vol['min_ml'], vol['max_ml'], 'mL')} por administración** · "
                    f"{fmt_num(vol['unit_per_ml'],3)} {unit}/mL"
                )


def show_pediatric_rules(rules):
    if not rules:
        return

    st.markdown("### Pautas de dosis")
    st.caption("Vista compacta: dos pautas por fila en escritorio. En móvil se apilan automáticamente.")

    # Grid de dos columnas: reduce de forma marcada el desplazamiento vertical.
    cols = st.columns(2, gap="medium")
    for idx, rule in enumerate(rules):
        target = cols[idx % 2]
        title = (
            f"{rule.get('indicacion') or 'Sin indicación'} · "
            f"{rule.get('poblacion') or 'Pediatría'} · "
            f"{rule.get('via') or 'vía no consignada'}"
        )
        with target:
            with st.expander(title, expanded=(idx < 2)):
                dose_text = pediatric_rule_dose_text(rule)
                interval = _pediatric_interval_hours(rule)
                freq_text = rule.get("frecuencia_texto") or (
                    f"cada {fmt_num(interval,1)} h" if interval is not None else None
                )
                max_text = pediatric_rule_limits_text(rule)

                chips = [f"<b>Dosis:</b> {dose_text}"]
                if freq_text:
                    chips.append(f"<b>Frecuencia:</b> {freq_text}")
                if max_text and str(max_text).strip().lower() not in {"no consignado", "—", "-"}:
                    chips.append(f"<b>Máximo:</b> {max_text}")
                st.markdown(
                    "<div class='ped-rule-summary'>" + "".join(f"<span>{x}</span>" for x in chips) + "</div>",
                    unsafe_allow_html=True,
                )

                meta = [f"{rule.get('poblacion') or 'Pediatría'}", f"Vía {rule.get('via') or '—'}"]
                if rule.get("duracion"):
                    meta.append(str(rule["duracion"]))
                st.caption(" · ".join(meta))

                if rule.get("notas"):
                    st.info(rule["notas"])
                if rule.get("nota_renal"):
                    st.warning("Función renal: " + str(rule["nota_renal"]))

                _render_rule_calculator(rule, compact=True)

                source = rule.get("fuente") or "Fuente no consignada"
                revision = rule.get("fecha_revision")
                page = rule.get("pagina_fuente")
                src_txt = source
                if page not in (None, ""):
                    src_txt += f" · pág. {page}"
                if revision:
                    src_txt += f" · revisión {revision}"
                st.caption(src_txt)
                if rule.get("url_fuente"):
                    st.link_button("Abrir fuente", rule["url_fuente"], use_container_width=True)

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
        "Seleccione el medicamento, ingrese la función renal y MedCalc muestra inmediatamente la pauta correspondiente usando la métrica original de la regla.",
    )
    st.info(
        "MedCalc no intercambia CrCl y eGFR. Si la ficha usa Cockcroft–Gault, calcula o solicita CrCl; "
        "si usa eGFR, utiliza eGFR. Las referencias CURRENT_REFERENCE pueden seleccionarse por su banda renal "
        "y mostrarse directamente, pero conservan su condición de referencia clínica cuando dependen de indicación, dosis basal u otras variables."
    )

    med = medication_picker("renal", "Medicamento")
    if not med:
        return

    all_rules = db.renal_rules(med["med_id"])
    auto_rules = [r for r in all_rules if r.get("automatizable") == "SI"]
    structured_refs = [r for r in all_rules if r.get("automatizable") != "SI"]
    refs = db.renal_biblio(med["med_id"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Catálogo Supabase", f"{COUNTS['medications']} medicamentos")
    c2.metric("Reglas automáticas", len(auto_rules))
    c3.metric("Referencias estructuradas", len(structured_refs))
    c4.metric("Bibliografía renal", len(refs))

    metrics = sorted({
        str(r.get("metrica_renal") or "").strip()
        for r in all_rules
        if str(r.get("metrica_renal") or "").strip()
    })
    metrics_upper = [m.upper() for m in metrics]
    needs_crcl = any("CRCL" in m and "1_73" not in m and "NORMALIZ" not in m for m in metrics_upper) or bool(refs)
    needs_norm_crcl = any("CRCL" in m and ("1_73" in m or "NORMALIZ" in m) for m in metrics_upper)
    needs_egfr = any("EGFR" in m for m in metrics_upper)

    if metrics:
        st.caption("Métrica(s) renal(es) de las reglas cargadas: " + ", ".join(metrics))

    mode_options = ["Calcular función renal completa"]
    if needs_crcl:
        mode_options.append("Ingresar CrCl conocido")
    if needs_egfr or not needs_crcl:
        mode_options.append("Ingresar eGFR conocido")
    mode_options.append("Solo conozco el estadio KDIGO")

    mode = st.radio(
        "Cómo obtener la función renal",
        mode_options,
        horizontal=True,
        key="renal_mode_v776",
    )
    hd = st.checkbox("Paciente en hemodiálisis", key="renal_hd_v776")

    egfr = None
    crcl = None
    crcl_norm = None
    bsa = None
    age = None
    sex = None
    weight = None
    creat = None
    band_key = None

    if mode == "Calcular función renal completa":
        with st.form("renal_full_v776", border=True):
            cols = st.columns(5 if needs_norm_crcl else 4)
            age = cols[0].number_input("Edad (años)", min_value=18, max_value=120, value=60, step=1)
            sex = cols[1].selectbox("Sexo para las ecuaciones", ["Hombre", "Mujer"])
            weight = cols[2].number_input("Peso (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
            creat = cols[3].number_input("Creatinina sérica (mg/dL)", min_value=0.1, max_value=20.0, value=1.0, step=0.1)
            height = None
            if needs_norm_crcl:
                height = cols[4].number_input("Talla (cm)", min_value=80.0, max_value=230.0, value=170.0, step=1.0)
            submit = st.form_submit_button("Calcular función renal y mostrar dosis", type="primary", use_container_width=True)
        if submit:
            egfr = ckdepi_2021(age, sex, creat)
            crcl = cockcroft_gault(age, sex, weight, creat)
            if needs_norm_crcl and height:
                bsa = bsa_mosteller(height, weight)
                crcl_norm = normalize_crcl_to_173(crcl, bsa)
            st.session_state["renal_v776"] = {
                "egfr": egfr, "crcl": crcl, "crcl_norm": crcl_norm, "bsa": bsa,
                "age": age, "sex": sex, "weight": weight, "creat": creat,
                "source": "CKD-EPI 2021 + Cockcroft–Gault", "mode": mode,
            }

    elif mode == "Ingresar CrCl conocido":
        with st.form("renal_known_crcl_v776", border=True):
            crcl_in = st.number_input("CrCl conocido (mL/min)", min_value=1.0, max_value=250.0, value=60.0, step=1.0)
            submit = st.form_submit_button("Usar CrCl y mostrar dosis", type="primary", use_container_width=True)
        if submit:
            st.session_state["renal_v776"] = {"crcl": crcl_in, "source": "CrCl ingresado", "mode": mode}

    elif mode == "Ingresar eGFR conocido":
        with st.form("renal_known_egfr_v776", border=True):
            egfr_in = st.number_input("eGFR (mL/min/1,73 m²)", min_value=1.0, max_value=200.0, value=60.0, step=1.0)
            submit = st.form_submit_button("Usar eGFR y mostrar dosis", type="primary", use_container_width=True)
        if submit:
            st.session_state["renal_v776"] = {"egfr": egfr_in, "source": "eGFR ingresado", "mode": mode}

    else:
        stage_manual = st.selectbox("Estadio KDIGO", ["G1", "G2", "G3a", "G3b", "G4", "G5"], key="renal_stage_manual_v776")
        stage_desc = {
            "G1":"normal o alto", "G2":"levemente disminuido", "G3a":"leve-moderadamente disminuido",
            "G3b":"moderada-severamente disminuido", "G4":"severamente disminuido", "G5":"falla renal"
        }.get(stage_manual)
        st.markdown(f'<div class="renal-stage"><strong>{stage_manual}</strong> · {stage_desc}</div>', unsafe_allow_html=True)
        st.warning(
            "El estadio KDIGO por sí solo no identifica de forma segura una banda CrCl ni una dosis específica. "
            "Para una recomendación directa ingrese la métrica exacta exigida por la ficha."
        )

    if mode != "Solo conozco el estadio KDIGO":
        snap = st.session_state.get("renal_v776") or {}
        if snap.get("mode") != mode:
            snap = {}
        egfr = snap.get("egfr")
        crcl = snap.get("crcl")
        crcl_norm = snap.get("crcl_norm")
        bsa = snap.get("bsa")

    # Mostrar resultados de función renal sin mezclar métricas.
    metrics_cards = []
    if egfr is not None:
        stage, stage_desc = ckd_g_stage(egfr)
        metrics_cards.append(("eGFR CKD-EPI", f"{fmt_num(egfr,1)} mL/min/1,73 m²"))
        metrics_cards.append(("Estadio KDIGO", stage))
    if crcl is not None:
        metrics_cards.append(("CrCl Cockcroft–Gault", f"{fmt_num(crcl,1)} mL/min"))
    if crcl_norm is not None:
        metrics_cards.append(("CrCl normalizado", f"{fmt_num(crcl_norm,1)} mL/min/1,73 m²"))
    if metrics_cards:
        render_clinical_cards(metrics_cards)
        if egfr is not None:
            st.caption(f"{stage}: {stage_desc}. eGFR y CrCl se muestran por separado y no se sustituyen entre sí.")

    if not all_rules and not refs:
        st.warning(
            f"**{med['principio_activo']}** todavía no tiene regla renal publicada ni referencia enlazada. "
            "Permanece visible en el catálogo, pero no se inventa un ajuste."
        )
        return

    st.markdown("### Dosis/ajuste renal")

    # ------------------------------------------------------------------
    # RECOMENDACIÓN DIRECTA: primero reglas estructuradas (AUTO o REFERENCE)
    # ------------------------------------------------------------------
    if hd:
        dialysis_rules = [
            r for r in all_rules
            if (r.get("tipo_regla") or "").upper() == "DIALISIS"
            or "HEMOD" in str(r.get("rango") or "").upper()
        ]
        if dialysis_rules:
            indications = sorted({r.get("indicacion") or "Sin indicación" for r in dialysis_rules})
            chosen_ind = indications[0] if len(indications) == 1 else st.selectbox(
                "Indicación / régimen", indications, key="renal_direct_hd_ind_v776"
            )
            chosen = next(r for r in dialysis_rules if (r.get("indicacion") or "Sin indicación") == chosen_ind)
            direct_text = renal_direct_instruction(
                chosen.get("regimen_ajustado"), chosen.get("rango"), hemodialysis=True, rule=chosen
            )
            st.success(f"**{direct_text}**")
        elif refs:
            ref = refs[0]
            direct_text = renal_direct_instruction(ref.get("suplemento_hd"), hemodialysis=True)
            st.success(f"**{direct_text}**")
        else:
            st.warning("No existe pauta de hemodiálisis publicada para este medicamento.")

    else:
        selected = None
        selected_value = None
        selected_indication = None

        if all_rules:
            indications = sorted({r.get("indicacion") or "Sin indicación" for r in all_rules})
            selected_indication = indications[0] if len(indications) == 1 else st.selectbox(
                "Indicación / régimen", indications, key="renal_direct_ind_v776"
            )
            candidate_rules = [r for r in all_rules if (r.get("indicacion") or "Sin indicación") == selected_indication]
            selected, selected_value = select_renal_rule(candidate_rules, crcl, crcl_norm, egfr, False)

        if selected:
            mappings = _renal_mapping_options(selected.get("regimen_ajustado"))
            lower_sel = as_float(selected.get("limite_inferior"))
            upper_sel = as_float(selected.get("limite_superior"))
            normal_band = lower_sel is not None and lower_sel >= 60 and upper_sel is None

            if mappings and not normal_band:
                base_options = sorted(mappings.keys(), key=lambda x: float(x))
                base = st.selectbox(
                    "Dosis habitual objetivo antes del ajuste renal (mg/día)",
                    base_options,
                    format_func=lambda x: f"{x:g} mg/día" if isinstance(x, float) else f"{x} mg/día",
                    key="renal_base_daily_target_v778",
                )
                target = mappings[base]
                suffix = _renal_frequency_suffix(selected.get("regimen_ajustado"))
                direct_text = f"{target.upper()} MG/DÍA{suffix}"
            else:
                direct_text = renal_direct_instruction(
                    selected.get("regimen_ajustado"), selected.get("rango"), hemodialysis=False, rule=selected
                )
            st.success(f"**{direct_text}**")

        elif refs and crcl is not None:
            band_key = renal_biblio_band(crcl)
            if band_key:
                labels = [f"Tabla {r['table']} · pág. {r['page']} · {r['principio_activo']}" for r in refs]
                ref = refs[0] if len(refs) == 1 else refs[labels.index(st.selectbox("Referencia renal", labels, key="renal_direct_ref_v776"))]
                recommendation = ref.get(band_key) or "—"
                direct_text = renal_direct_instruction(recommendation, band_key, hemodialysis=False)
                st.success(f"**{direct_text}**")

        else:
            # Explicar exactamente qué dato falta para poder mostrar la pauta.
            if all_rules:
                candidate_metrics = sorted({
                    str(r.get("metrica_renal") or "").strip()
                    for r in all_rules
                    if str(r.get("metrica_renal") or "").strip()
                })
                missing = []
                if any("CRCL_CG" in m.upper() for m in candidate_metrics) and crcl is None:
                    missing.append("CrCl por Cockcroft–Gault")
                if any("EGFR" in m.upper() for m in candidate_metrics) and egfr is None:
                    missing.append("eGFR")
                if any("CRCL" in m.upper() and ("1_73" in m.upper() or "NORMALIZ" in m.upper()) for m in candidate_metrics) and crcl_norm is None:
                    missing.append("CrCl normalizado")
                if missing:
                    st.warning("Para mostrar la dosis inmediatamente falta: **" + ", ".join(missing) + "**.")
                else:
                    st.warning("No hay una banda estructurada que coincida con los datos ingresados. Revise la fuente antes de prescribir.")
            elif refs:
                st.warning("Esta bibliografía está organizada por CrCl. Ingrese o calcule CrCl para mostrar la dosis correspondiente.")

    # La bibliografía queda debajo, como auditoría, no como paso obligatorio.
    with st.expander("Ver bibliografía y todas las reglas", expanded=False):
        if structured_refs:
            st.markdown("#### Referencias renales estructuradas PUBLISHED")
            for r in structured_refs:
                st.write(
                    f"**{r.get('indicacion') or 'Sin indicación'} · {r.get('rango') or 'banda'}**  \n"
                    f"Métrica: {r.get('metrica_renal') or '—'}  \n"
                    f"Pauta: {r.get('regimen_ajustado') or '—'}"
                )
                if r.get("notas"):
                    st.caption(str(r["notas"]))
                st.divider()
        if auto_rules:
            st.markdown("#### Reglas automáticas estructuradas")
            for r in auto_rules:
                st.write(
                    f"**{r.get('indicacion') or 'Sin indicación'} · {r.get('rango') or 'banda'}**  \n"
                    f"Métrica: {r.get('metrica_renal') or '—'}  \n"
                    f"Pauta: {r.get('regimen_ajustado') or '—'}"
                )
                st.divider()
        if refs:
            st.markdown("#### Bibliografía renal enlazada")
            for ref in refs:
                st.write(f"**{ref.get('principio_activo') or med['principio_activo']} · Tabla {ref.get('table')} · pág. {ref.get('page')}**")
                st.write(f"Función renal normal: {ref.get('dosis_fr_normal') or '—'}")
                st.write(f"≥50 mL/min: {ref.get('crcl_100_50') or '—'}")
                st.write(f"10–49 mL/min: {ref.get('crcl_50_10') or '—'}")
                st.write(f"<10 mL/min: {ref.get('crcl_lt10') or '—'}")
                st.write(f"HD: {ref.get('suplemento_hd') or '—'}")
                source_block(ref.get("fuente"), ref.get("url_fuente"), ref.get("fecha_fuente"))
                st.divider()

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

    def _tox_age_scope(value):
        """Devuelve un rango etario SOLO cuando la propia referencia lo declara.

        No añade contexto de ingesta ni extrapola umbrales entre poblaciones.
        """
        raw = str(value or "")
        norm = normalize_text(raw)
        if not norm:
            return None

        # Referencias que excluyen expresamente adolescentes/adultos (p. ej. tablas RCH
        # de ingestas pediátricas accidentales). En MedCalc se limitan a <12 años
        # para no trasladarlas automáticamente a adolescentes.
        excludes_adol_adult = (
            ("no aplic" in norm or "no corresponde" in norm)
            and ("adolescent" in norm and "adult" in norm)
        )
        if excludes_adol_adult and ("pediatr" in norm or "niñ" in norm or "nino" in norm):
            return {"min": 0.0, "max_exclusive": 12.0, "label": "pediatría <12 años"}

        # Solo usar población genérica cuando la referencia es inequívoca.
        if "pediatr" in norm and "adult" not in norm and "adolescent" not in norm:
            return {"min": 0.0, "max_exclusive": 18.0, "label": "pediatría <18 años"}
        if "adult" in norm and "pediatr" not in norm and "niñ" not in norm and "nino" not in norm:
            return {"min": 18.0, "max_exclusive": None, "label": "adultos ≥18 años"}
        return None

    def _tox_age_applies(age_years, scope):
        if not scope:
            return None
        if scope.get("min") is not None and age_years < float(scope["min"]):
            return False
        if scope.get("max_exclusive") is not None and age_years >= float(scope["max_exclusive"]):
            return False
        return True

    def _tox_age_specific_reference(value, age_years):
        """Selecciona una referencia cuantitativa cuando el texto trae bandas por edad.

        Ej.: '<12 años, 1 mg; ≥12 años, >5 mg'. No interpreta texto sin una
        condición etaria explícita.
        """
        raw = str(value or "").replace(",", ".")
        if not raw.strip():
            return None
        clauses = [c.strip() for c in re.split(r"[;\n]+", raw) if c.strip()]
        for clause in clauses:
            norm = normalize_text(clause)
            applies = None
            label = None
            m = re.search(r"<\s*(\d+(?:\.\d+)?)\s*(?:años|anos)", norm)
            if m:
                cutoff=float(m.group(1)); applies=age_years < cutoff; label=f"<{cutoff:g} años"
            if applies is None:
                m = re.search(r"(?:>=|≥)\s*(\d+(?:\.\d+)?)\s*(?:años|anos)", norm)
                if m:
                    cutoff=float(m.group(1)); applies=age_years >= cutoff; label=f"≥{cutoff:g} años"
            if applies is None:
                m = re.search(r">\s*(\d+(?:\.\d+)?)\s*(?:años|anos)", norm)
                if m:
                    cutoff=float(m.group(1)); applies=age_years > cutoff; label=f">{cutoff:g} años"
            if applies is None:
                m = re.search(r"(?:hasta|≤|<=)\s*(\d+(?:\.\d+)?)\s*(?:años|anos)", norm)
                if m:
                    cutoff=float(m.group(1)); applies=age_years <= cutoff; label=f"≤{cutoff:g} años"
            if applies is None or not applies:
                continue
            refs = _tox_numeric_references(clause)
            if len(refs["mgkg"]) == 1 and not refs["absolute_mg"]:
                op, val = refs["mgkg"][0]
                return {"kind": "mgkg", "op": op or ">=", "value": val, "age_label": label, "clause": clause}
            if len(refs["absolute_mg"]) == 1 and not refs["mgkg"]:
                op, val, display = refs["absolute_mg"][0]
                return {"kind": "absolute_mg", "op": op or ">=", "value": val, "display": display, "age_label": label, "clause": clause}
        return None

    def _tox_source_name(source):
        src = str(source or "").strip()
        low = src.lower()
        if "rch.org.au" in low:
            return "Royal Children’s Hospital Melbourne (RCH)"
        if "pubmed.ncbi.nlm.nih.gov" in low:
            return "PubMed / publicación científica enlazada"
        if "dailymed.nlm.nih.gov" in low:
            return "DailyMed / ficha regulatoria"
        return src or "Fuente no consignada"

    tab1, tab2, tab3 = st.tabs(["Medicamentos", "Tóxicos externos", "Antídotos"])
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

        # Calculadora de exposición. La clasificación "intoxicación sí/no" solo se
        # emite cuando la ficha permite comparación automática o existe un umbral
        # estructurado explícitamente automatizable. Una cifra bibliográfica histórica
        # puede compararse matemáticamente, pero no se convierte por sí sola en diagnóstico.
        threshold = as_float(tox.get("umbral_mgkg_automatizable"))
        compare_flag = str(tox.get("permitir_comparacion_automatica") or "").strip().upper()
        allow_compare = compare_flag in {"SI", "SÍ", "TRUE", "1", "YES"}
        # Compatibilidad con registros antiguos: un umbral en el campo explícitamente
        # automatizable se considera apto salvo que la ficha lo niegue de forma expresa.
        if threshold is not None and compare_flag not in {"NO", "FALSE", "0"}:
            allow_compare = True

        reference_text = reviewed_threshold or original_threshold or ""
        parsed_refs = _tox_numeric_references(reference_text)
        has_text_reference = bool(parsed_refs["mgkg"] or parsed_refs["absolute_mg"])

        def _threshold_positive(value, cutoff, op=""):
            if cutoff is None or cutoff <= 0 or value is None:
                return None
            op = (op or "").strip()
            if op == ">":
                return value > cutoff
            if op in {">=", "≥", ""}:
                return value >= cutoff
            # Un límite expresado como < o ≤ no se interpreta automáticamente como
            # umbral de toxicidad superior.
            return None

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
                age_col, weight_col = st.columns(2)
                age_years = age_col.number_input(
                    "Edad del paciente (años)", min_value=0.0, max_value=120.0, value=10.0, step=0.5
                )
                weight = weight_col.number_input("Peso del paciente (kg)", min_value=0.1, value=20.0, step=0.1)
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
                    ("Edad", f"{fmt_num(age_years,1)} años"),
                ]

                ratio = None
                reference_label = None
                reference_card_label = "Umbral / referencia"
                criterion_positive = None
                criterion_validated = False
                comparison_source = None
                age_eval_positive = None
                age_eval_active = False

                age_scope = _tox_age_scope(reference_text)
                age_applicable = _tox_age_applies(age_years, age_scope)
                age_specific = _tox_age_specific_reference(reference_text, age_years)
                validation_status = str(tox.get("estado_revision") or "").upper()
                reviewed_specific = "VALIDADO_ESPECIFICO" in validation_status

                # Si existe una banda cuantitativa explícita para la edad ingresada,
                # esa banda tiene prioridad sobre la extracción genérica.
                if age_specific is not None:
                    if age_specific["kind"] == "mgkg":
                        cutoff = age_specific["value"]
                        op = age_specific["op"]
                        reference_label = f"{op}{cutoff:g} mg/kg · {age_specific['age_label']}"
                        ratio = exposure / cutoff if cutoff > 0 else None
                        if allow_compare:
                            criterion_positive = _threshold_positive(exposure, cutoff, op)
                            criterion_validated = criterion_positive is not None
                            comparison_source = "umbral toxicológico validado para esta edad"
                        elif reviewed_specific:
                            age_eval_positive = _threshold_positive(exposure, cutoff, op)
                            age_eval_active = age_eval_positive is not None
                    else:
                        cutoff = age_specific["value"]
                        op = age_specific["op"]
                        reference_label = f"{op}{age_specific.get('display') or f'{cutoff:g} mg'} · {age_specific['age_label']}"
                        ratio = total_mg / cutoff if cutoff > 0 else None
                        if allow_compare:
                            criterion_positive = _threshold_positive(total_mg, cutoff, op)
                            criterion_validated = criterion_positive is not None
                            comparison_source = "umbral toxicológico validado para esta edad"
                        elif reviewed_specific:
                            age_eval_positive = _threshold_positive(total_mg, cutoff, op)
                            age_eval_active = age_eval_positive is not None
                    reference_card_label = "Umbral aplicable a esta edad"

                elif age_applicable is False:
                    # La referencia existe, pero la propia fuente la limita a otra población etaria.
                    reference_label = f"NO APLICABLE · {age_scope.get('label') if age_scope else 'otra edad'}"
                    reference_card_label = "Aplicabilidad del umbral"
                    ratio = None

                elif threshold is not None:
                    ratio = exposure / threshold if threshold > 0 else None
                    reference_label = tox.get("etiqueta_umbral") or f"{threshold:g} mg/kg"
                    m_op = re.search(r"(>=|>|≥|≤|<)\s*\d", str(reference_label))
                    op = m_op.group(1) if m_op else ">="
                    if allow_compare and age_applicable is not False:
                        criterion_positive = _threshold_positive(exposure, threshold, op)
                        criterion_validated = criterion_positive is not None
                        comparison_source = "umbral toxicológico estructurado"
                    reference_card_label = "Umbral toxicológico" if allow_compare else "Referencia cuantitativa"

                else:
                    mgkg_refs = parsed_refs["mgkg"]
                    abs_refs = parsed_refs["absolute_mg"]

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
                        if allow_compare and age_applicable is not False:
                            criterion_positive = _threshold_positive(exposure, applicable_mgkg, op_kg or op_abs or ">=")
                            criterion_validated = criterion_positive is not None
                            comparison_source = "referencia combinada validada"
                    elif len(mgkg_refs) == 1:
                        op, kg_ref = mgkg_refs[0]
                        reference_label = f"{op}{kg_ref:g} mg/kg"
                        ratio = exposure / kg_ref if kg_ref > 0 else None
                        if allow_compare and age_applicable is not False:
                            criterion_positive = _threshold_positive(exposure, kg_ref, op or ">=")
                            criterion_validated = criterion_positive is not None
                            comparison_source = "referencia mg/kg validada"
                    elif len(abs_refs) == 1:
                        op, abs_mg, abs_label = abs_refs[0]
                        reference_label = f"{op}{abs_label} total"
                        ratio = total_mg / abs_mg if abs_mg > 0 else None
                        if allow_compare and age_applicable is not False:
                            criterion_positive = _threshold_positive(total_mg, abs_mg, op or ">=")
                            criterion_validated = criterion_positive is not None
                            comparison_source = "referencia absoluta validada"
                    else:
                        refs=[]
                        refs += [f"{op}{v:g} mg/kg" for op,v in mgkg_refs]
                        refs += [f"{op}{label} total" for op,_,label in abs_refs]
                        if refs:
                            reference_label = " · ".join(refs)
                    reference_card_label = "Referencia bibliográfica" if not allow_compare else "Umbral / referencia validada"

                if reference_label:
                    cards.append((reference_card_label, reference_label))
                render_clinical_cards(cards)

                # Fuente inmediatamente junto a la referencia: el usuario no debe
                # tener que buscarla al final de la ficha.
                source_value = _tox_clean(tox.get("fuente_principal"))
                if reference_label and source_value:
                    st.caption(f"Fuente del umbral/referencia: {_tox_source_name(source_value)}")

                if age_applicable is False:
                    st.warning(
                        f"🟡 **UMBRAL NO APLICABLE A ESTA EDAD.** La referencia cargada corresponde a "
                        f"{age_scope.get('label') if age_scope else 'otra población etaria'}; no se usa para clasificar a un paciente de {fmt_num(age_years,1)} años."
                    )
                elif criterion_validated:
                    if criterion_positive:
                        st.error("🔴 **CRITERIO DOSIMÉTRICO DE INTOXICACIÓN: POSITIVO** · La exposición alcanza o supera el umbral toxicológico validado para esta edad.")
                    else:
                        st.success("🟢 **CRITERIO DOSIMÉTRICO DE INTOXICACIÓN: NEGATIVO** · La exposición calculada no alcanza el umbral toxicológico validado para esta edad.")
                    if comparison_source:
                        st.caption(f"Clasificación basada en {comparison_source}. La conducta final depende además de la evaluación clínica.")
                elif age_eval_active:
                    if age_eval_positive:
                        st.warning("🟡 **SUPERA EL UMBRAL DE EVALUACIÓN PARA ESTA EDAD.** Requiere valoración según la guía fuente; este corte de evaluación no equivale por sí solo a un diagnóstico clínico de intoxicación.")
                    else:
                        st.info("🔵 **NO SUPERA EL UMBRAL DE EVALUACIÓN PARA ESTA EDAD.** La dosis por sí sola no excluye toxicidad si existen manifestaciones clínicas.")
                elif ratio is not None and reference_label:
                    if ratio >= 1:
                        st.warning("🟡 **SUPERA LA REFERENCIA BIBLIOGRÁFICA REGISTRADA**, pero esta ficha no permite clasificar automáticamente intoxicación solo con esa cifra.")
                    else:
                        st.info("🔵 **POR DEBAJO DE LA REFERENCIA BIBLIOGRÁFICA REGISTRADA**. Esta ficha no permite descartar intoxicación automáticamente solo por la dosis.")
                    st.caption(f"Relación matemática exposición/referencia: {fmt_num(ratio,2)}×. La referencia se conserva para trazabilidad y no se promueve a umbral clínico sin validación.")
                elif reference_label:
                    st.warning("🟡 **CRITERIO DOSIMÉTRICO: NO CLASIFICABLE AUTOMÁTICAMENTE.** La ficha contiene una referencia, pero no existe un umbral validado aplicable automáticamente a esta edad.")

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
        all_external = db.search_other_tox("")
        ancillary_status = {}
        if hasattr(db, "toxicology_ancillary_status"):
            try:
                ancillary_status = db.toxicology_ancillary_status() or {}
            except Exception:
                ancillary_status = {}

        categories = sorted({
            str(r.get("categoria") or "BASE ORIGINAL").strip()
            for r in all_external
            if str(r.get("categoria") or "").strip()
        })

        f1, f2 = st.columns([0.9, 2.1])
        with f1:
            category = st.selectbox(
                "Categoría",
                ["Todas"] + categories,
                key="other_tox_category",
            )
        with f2:
            q = st.text_input(
                "Buscar tóxico, animal, veneno, planta, metal o químico",
                placeholder="Ej.: mercurio, araña de rincón, Bothrops, paraquat, metanol, brionia...",
                key="other_tox_q",
            )

        hits = db.search_other_tox(q)
        if category != "Todas":
            hits = [r for r in hits if str(r.get("categoria") or "BASE ORIGINAL").strip() == category]

        reviewed_n = sum(
            1 for r in all_external
            if str(r.get("estado_revision") or "").startswith(("VALIDADO_", "REVISADO_"))
        )
        st.caption(
            f"{len(all_external)} fichas externas disponibles · {reviewed_n} con capa clínica revisada. "
            "La información del libro de Toxicología Clínica 2011 se muestra como referencia bibliográfica y no se automatiza como umbral por sí sola."
        )

        if not all_external:
            st.error(
                "NO SE CARGÓ LA BASE DE TÓXICOS EXTERNOS. En GitHub deben estar, junto a app.py, "
                "`supabase_repository.py` V7.9.2. La base externa está integrada como fallback interno."
            )
            if ancillary_status:
                st.code(str(ancillary_status))
        elif not hits:
            st.warning("Sin coincidencias para la búsqueda/filtro seleccionado.")
            if q.strip():
                st.caption(
                    "Si un tóxico conocido no aparece, revise que `toxicos_externos_revisados_v2.csv` haya sido subido a la raíz del repositorio."
                )
        else:
            labels = []
            for item in hits:
                name = item.get("toxico") or "—"
                cat = item.get("categoria") or "BASE ORIGINAL"
                labels.append(f"{name} · {cat}")
            pick = st.selectbox("Tóxico", labels, key="other_tox_sel")
            r = hits[labels.index(pick)]

            st.markdown(f"### {r.get('toxico') or 'Tóxico externo'}")
            meta = []
            if r.get("categoria"):
                meta.append(f"**Categoría:** {r.get('categoria')}")
            if r.get("region_relevancia"):
                meta.append(f"**Región:** {r.get('region_relevancia')}")
            if r.get("via_exposicion"):
                meta.append(f"**Vía:** {r.get('via_exposicion')}")
            if meta:
                st.markdown(" · ".join(meta))
            if r.get("alias"):
                st.caption("También puede encontrarse como: " + str(r.get("alias")))
            if r.get("toxico_canonico") and normalize_text(r.get("toxico_canonico")) != normalize_text(r.get("toxico")):
                st.caption("Revisión clínica basada en la categoría: " + str(r.get("toxico_canonico")))

            symptoms = r.get("sintomas_base") or "Sin manifestaciones específicas registradas."
            st.markdown("#### 🚨 Manifestaciones clínicas")
            st.markdown(
                f'<div class="result-box"><strong>{_esc(symptoms)}</strong></div>',
                unsafe_allow_html=True,
            )

            dose_ref = str(r.get("dosis_toxica_referencia") or "").strip()
            dose_state = str(r.get("estado_dosis") or "").strip()
            if not dose_ref:
                dose_ref = "SDTE — no se dispone de una dosis tóxica humana única suficientemente defendible para esta ficha."
            dose_label = "Dosis tóxica / referencia bibliográfica"
            render_clinical_cards([(dose_label, dose_ref)])
            if dose_state and "NO_AUTOMAT" in dose_state.upper():
                st.caption("La cifra/referencia anterior NO se usa como umbral automático; debe interpretarse con clínica, vía, formulación y fuente.")

            if r.get("signos_gravedad"):
                st.error("**SIGNOS DE GRAVEDAD:** " + str(r.get("signos_gravedad")))

            m1, m2 = st.columns(2)
            with m1:
                st.markdown("#### Manejo inicial")
                st.write(r.get("antidoto_tratamiento_base") or "Manejo de soporte según exposición y cuadro clínico.")
                if r.get("descontaminacion"):
                    st.markdown("**Descontaminación / reducción de exposición**")
                    st.write(r.get("descontaminacion"))
            with m2:
                st.markdown("#### Tratamiento específico / antídoto")
                specific = r.get("tratamiento_especifico") or r.get("antidoto")
                st.write(specific or "No hay tratamiento específico registrado.")
                if r.get("antidoto") and normalize_text(r.get("antidoto")) != normalize_text(specific):
                    st.write("**Antídoto/antiveneno:**", r.get("antidoto"))
                if r.get("dosis_antidoto_referencia"):
                    st.info("**Dosis de antídoto de referencia:** " + str(r.get("dosis_antidoto_referencia")))

            c1, c2 = st.columns(2)
            with c1:
                if r.get("monitorizacion"):
                    st.markdown("#### Monitorización")
                    st.write(r.get("monitorizacion"))
            with c2:
                if r.get("criterios_observacion_uci"):
                    st.markdown("#### Observación / UCI")
                    st.write(r.get("criterios_observacion_uci"))

            if r.get("notas"):
                st.info(str(r.get("notas")))

            st.markdown("#### Fuentes")
            source_name = r.get("fuente")
            source_url = r.get("url_fuente")
            if source_name:
                st.write(f"**Fuente clínica abierta/actual:** {source_name}")
            if source_url and str(source_url).startswith(("http://", "https://")):
                st.link_button("Abrir fuente clínica", source_url)
            if r.get("fuente_libro"):
                st.write("**Referencia bibliográfica complementaria:** " + str(r.get("fuente_libro")))
                if r.get("paginas_libro"):
                    st.caption("Ubicación en el libro: " + str(r.get("paginas_libro")))
            if r.get("fecha_revision"):
                st.caption("Revisión MedCalc: " + str(r.get("fecha_revision")))

            original_symptoms = r.get("sintomas_originales")
            original_treatment = r.get("tratamiento_original")
            changed = (
                (original_symptoms and normalize_text(original_symptoms) != normalize_text(r.get("sintomas_base")))
                or (original_treatment and normalize_text(original_treatment) != normalize_text(r.get("antidoto_tratamiento_base")))
            )
            if changed:
                with st.expander("Ver registro original de MedCalc (trazabilidad)"):
                    if original_symptoms:
                        st.write("**Manifestaciones originales:**", original_symptoms)
                    if original_treatment:
                        st.write("**Tratamiento/antídoto original:**", original_treatment)

    with tab3:
        q = st.text_input(
            "Buscar tóxico, síndrome o antídoto",
            placeholder="Ej.: bicarbonato, mercurio, deferoxamina, naloxona, cianuro...",
            key="antidote_q",
        )
        hits = db.search_antidotes(q)
        if not hits:
            st.warning("Sin coincidencias en la base de antídotos.")
            if ancillary_status and ancillary_status.get("antidotes_total", 0) == 0:
                st.error(
                    "No se cargó la base de antídotos. Suba `antidotos_revisados_v2.csv` junto a `app.py` y `supabase_repository.py`."
                )
        else:
            labels = [f"{r.get('toxico_sindrome') or '—'} → {r.get('antidoto_base') or '—'}" for r in hits]
            pick = st.selectbox("Resultado", labels, key="antidote_sel")
            r = hits[labels.index(pick)]
            st.markdown(f"### {r.get('toxico_sindrome') or 'Antídoto'}")
            st.markdown("#### Antídoto / tratamiento específico")
            st.write(r.get("antidoto_base") or "—")

            dose = r.get("dosis_revisada") or r.get("dosis_base") or "No consignada"
            render_clinical_cards([("Dosis de referencia", dose)])

            if r.get("indicacion_clinica"):
                st.markdown("#### Indicación clínica")
                st.write(r.get("indicacion_clinica"))
            if r.get("precauciones_clave"):
                st.warning("**Precauciones:** " + str(r.get("precauciones_clave")))
            if r.get("observaciones_base"):
                with st.expander("Observaciones de la base original"):
                    st.write(r.get("observaciones_base"))
            if r.get("dosis_original") and normalize_text(r.get("dosis_original")) != normalize_text(dose):
                with st.expander("Dosis consignada originalmente (trazabilidad)"):
                    st.write(r.get("dosis_original"))

            if r.get("fuente_libro"):
                st.markdown("#### Fuente bibliográfica")
                st.write(r.get("fuente_libro"))
                if r.get("paginas_libro"):
                    st.caption("Ubicación: " + str(r.get("paginas_libro")))
                st.caption(
                    "Las pautas del libro son referencia bibliográfica de 2011. Cuando exista un protocolo/ficha vigente, este debe prevalecer."
                )
            if r.get("url_fuente_actual") and str(r.get("url_fuente_actual")).startswith(("http://", "https://")):
                st.link_button("Abrir fuente actual", r.get("url_fuente_actual"))


@st.cache_data(ttl=300, show_spinner=False)
def _electrolyte_bundle_cached(analyte_code="K"):
    return db.electrolyte_bundle(analyte_code)


@st.cache_data(ttl=300, show_spinner=False)
def _electrolyte_modifier_rows_cached(med_ids_tuple, analyte_code="K"):
    return db.medication_electrolyte_modifiers(list(med_ids_tuple), analyte_code)


def _component_by_code(product, analyte_code):
    wanted = str(analyte_code or "").upper()
    for comp in product.get("components") or []:
        code = str((comp.get("analyte") or {}).get("code") or "").upper()
        if code == wanted:
            return comp
    return None


def _electrolyte_rule_source(rule):
    links = rule.get("sources") or []
    primary = next((x for x in links if x.get("evidence_role") == "PRIMARY"), None)
    if primary and primary.get("source"):
        return primary.get("source") or {}
    return ((rule.get("protocol") or {}).get("source") or {})


def _severity_es(value):
    """Traduce etiquetas internas de severidad antes de mostrarlas al usuario."""
    raw = str(value or "INFO").strip().upper()
    return {
        "INFO": "INFORMATIVA",
        "NORMAL": "NORMAL",
        "LOW": "LEVE",
        "MILD": "LEVE",
        "MODERATE": "MODERADA",
        "HIGH": "ALTA",
        "SEVERE": "GRAVE",
        "CRITICAL": "CRÍTICA",
    }.get(raw, raw.replace("_", " "))


def _analyte_name_es(code):
    return {
        "NA": "Sodio",
        "K": "Potasio",
        "MG": "Magnesio",
        "CA": "Calcio",
        "P": "Fósforo",
        "CL": "Cloro",
        "AB": "Ácido-base",
    }.get(str(code or "").upper(), str(code or "—"))


def _render_electrolyte_rule(rule):
    severity = str(rule.get("severity") or "INFO").upper()
    text = rule.get("recommendation_text") or rule.get("rule_code") or "Regla clínica"
    if rule.get("hard_stop") or severity == "CRITICAL":
        st.error(text)
    elif severity == "HIGH":
        st.warning(text)
    else:
        st.info(text)
    src = _electrolyte_rule_source(rule)
    org = src.get("organization") or "Fuente no consignada"
    title = src.get("title") or ""
    version = (rule.get("protocol") or {}).get("source_version") or src.get("edition")
    caption = f"{org}"
    if title:
        caption += f" · {title}"
    if version:
        caption += f" · {version}"
    st.caption(caption)
    if src.get("url"):
        st.link_button("Abrir fuente", src.get("url"), key=f"src_{rule.get('rule_code')}")


def _format_modifier_direction(direction):
    return {"RAISE": "↑ favorece hiperK", "LOWER": "↓ favorece hipoK", "VARIABLE": "↕ efecto variable"}.get(str(direction or "").upper(), str(direction or "—"))


def _unique_texts(rules, rule_types=None):
    """Devuelve recomendaciones clínicas únicas conservando el orden.

    Si rule_types se proporciona, solo incluye reglas cuyo rule_type pertenezca
    a ese conjunto. Las filas sin recommendation_text se omiten: nunca se
    muestra al usuario un rule_code interno como sustituto clínico.
    """
    allowed = {str(x).upper() for x in rule_types} if rule_types else None
    out = []
    seen = set()
    for rule in rules or []:
        if allowed is not None and str(rule.get("rule_type") or "").upper() not in allowed:
            continue
        text = str(rule.get("recommendation_text") or "").strip()
        if not text:
            continue
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def page_electrolytes():
    header(
        "Hidroelectrolitos y reposición",
        "POTASIO · introduzca el valor y el contexto que realmente cambia la conducta; MedCalc devuelve un plan único y una preparación concreta.",
    )
    if st.button("← Volver a Inicio", key="electrolytes_back_home"):
        go_to_module("Inicio", st.session_state.get("selected_med_id"))

    bundle = _electrolyte_bundle_cached("K")
    if not bundle.get("analyte") or not bundle.get("protocols") or not bundle.get("rules"):
        st.error("El módulo de Hidroelectrolitos no está cargado correctamente en Supabase.")
        return

    def _es_line(line):
        return {"PERIPHERAL": "periférica", "CENTRAL": "central", "IO": "intraósea"}.get(str(line or "").upper(), str(line or "—").lower())

    def _es_diluent(name):
        raw = str(name or "")
        repl = {
            "Sodium Chloride 0.9% Injection USP": "Cloruro de sodio 0,9%",
            "Dextrose 5% Injection USP": "Glucosa 5%",
            "Dextrose 10% Injection USP": "Glucosa 10%",
        }
        for a, b in repl.items():
            raw = raw.replace(a, b)
        return raw.replace("—", "–")

    def _action_value(rules, key, default=None):
        for rule in rules:
            action = rule.get("action_json") or {}
            if action.get(key) is not None:
                return action.get(key)
        return default

    # ------------------------------------------------------------------
    # Entrada clínica compacta: dos datos y UN selector de contexto.
    # ------------------------------------------------------------------
    with st.container(border=True):
        st.markdown("### Datos para calcular")
        c1, c2, c3, c4 = st.columns([0.8, 0.9, 1, 1])
        with c1:
            age_years = st.number_input(
                "Edad (años)",
                min_value=0.0, max_value=120.0, value=50.0, step=1.0,
                key="el_auto_age",
                help="La versión actual de POTASIO utiliza protocolos de adultos. La edad queda incorporada al contexto para futuras reglas pediátricas.",
            )
        with c2:
            weight_kg = st.number_input(
                "Peso (kg)",
                min_value=2.0, max_value=350.0, value=70.0, step=0.5,
                key="el_auto_weight",
                help="El peso no se usa para inventar un déficit de K. Se usa para expresar la reposición en mmol/kg, mmol/kg/h y la carga de volumen en mL/kg.",
            )
        with c3:
            k = st.number_input(
                "Potasio plasmático (mmol/L = mEq/L)",
                min_value=0.5, max_value=12.0, value=4.0, step=0.1,
                key="el_auto_k",
            )
        with c4:
            mg_raw = st.text_input(
                "Magnesio (mmol/L) · opcional",
                value="", placeholder="Ej. 0,45", key="el_auto_mg",
            )

        options = [
            "Insuficiencia cardiaca / congestión / restricción de volumen",
            "Deterioro renal (AKI/ERC) con diuresis",
            "Oliguria o anuria",
            "Hemodiálisis",
            "Pérdidas digestivas (vómitos/diarrea/drenajes)",
            "Redistribución (insulina, beta-agonista, alcalosis)",
            "Cetoacidosis diabética / crisis hiperglucémica",
            "Síntomas importantes o cambios ECG atribuibles a K",
            "Debilidad extrema / parálisis",
            "K refractario a corrección previa",
            "No puede recibir vía oral",
            "Ya dispone de vía venosa central",
        ]
        if k >= 5.2:
            options.extend([
                "Sospecha de pseudohiperpotasemia / muestra hemolizada",
                "Acidosis metabólica",
                "Ascenso rápido de K / paciente inestable",
            ])

        clinical_tags = st.multiselect(
            "Contexto clínico · marque solo lo que aplique",
            options,
            key="el_auto_context",
            help="Un solo selector reemplaza los múltiples campos anteriores. Si no marca nada se usa el escenario adulto general.",
        )

        with st.expander("Medicamentos actuales · opcional"):
            all_meds = db.search_medications("", limit=max(COUNTS.get("medications", 1000), 1000))
            med_labels = [f"{m['principio_activo']} · {m['med_id']}" for m in all_meds]
            label_to_id = {f"{m['principio_activo']} · {m['med_id']}": m["med_id"] for m in all_meds}
            selected_labels = st.multiselect(
                "Medicamentos del paciente",
                med_labels,
                default=[],
                key="el_auto_medications",
                help="Solo identifica fármacos que pueden favorecer hipo/hiperpotasemia; no es un interaction checker.",
            )

    # La arquitectura ya guarda population en electrolyte_protocols. Mientras
    # no exista un protocolo pediátrico PUBLISHED para K, no reutilizamos reglas
    # adultas en menores. Esto evita que el peso convierta inadvertidamente una
    # pauta adulta fija en una falsa pauta pediátrica por kg.
    populations = {str((p.get("population") or "")).upper() for p in bundle.get("protocols") or []}
    pediatric_available = any(x in populations for x in {"PEDIATRIC", "PEDIATRIA", "PEDIÁTRICO", "PEDIATRICO", "ALL"})
    if float(age_years) < 18 and not pediatric_available:
        st.error(
            "**POTASIO pediátrico todavía no está publicado en este módulo.** "
            "La versión actual usa protocolos adultos y no transformará una pauta adulta en mmol/kg de forma automática. "
            "Puede consultar el módulo de Pediatría mientras se incorpora el algoritmo hidroelectrolítico pediátrico específico."
        )
        return

    def _tag(name):
        return name in clinical_tags

    mg_value = _fallback_as_float(mg_raw)
    hf = _tag("Insuficiencia cardiaca / congestión / restricción de volumen")
    renal_with_urine = _tag("Deterioro renal (AKI/ERC) con diuresis")
    olig_anuria = _tag("Oliguria o anuria")
    dialysis = _tag("Hemodiálisis")
    gi_losses = _tag("Pérdidas digestivas (vómitos/diarrea/drenajes)")
    redistribution = _tag("Redistribución (insulina, beta-agonista, alcalosis)")
    dka = _tag("Cetoacidosis diabética / crisis hiperglucémica")
    symptoms_ecg = _tag("Síntomas importantes o cambios ECG atribuibles a K")
    paralysis = _tag("Debilidad extrema / parálisis")
    refractory = _tag("K refractario a corrección previa")
    no_oral = _tag("No puede recibir vía oral")
    central_already = _tag("Ya dispone de vía venosa central")
    pseudo = _tag("Sospecha de pseudohiperpotasemia / muestra hemolizada")
    acidosis = _tag("Acidosis metabólica")
    unstable = _tag("Ascenso rápido de K / paciente inestable")

    selected_ids = tuple(label_to_id[x] for x in selected_labels) if selected_labels else tuple()
    modifiers = _electrolyte_modifier_rows_cached(selected_ids, "K") if selected_ids else []
    medication_mechanism_classes = sorted({str(m.get("mechanism_class")) for m in modifiers if m.get("mechanism_class")})
    medication_effect_codes = sorted({str(m.get("effect_code")) for m in modifiers if m.get("effect_code")})

    context = {
        "patient": {"age_years": float(age_years), "weight_kg": float(weight_kg)},
        "serum": {"k_mmol_l": float(k)},
        "magnesium": {"value_mmol_l": float(mg_value) if mg_value is not None else None, "low": False},
        "renal": {
            "impairment": bool(renal_with_urine or olig_anuria or dialysis),
            "failure": bool(olig_anuria or dialysis),
            "dialysis": bool(dialysis),
            "hemodialysis": bool(dialysis),
            "oliguria": bool(olig_anuria),
            "anuria": bool(olig_anuria),
        },
        "clinical": {
            "heart_failure_or_congestion": bool(hf),
            "gi_diuretic_losses": bool(gi_losses),
            "redistribution_context": bool(redistribution),
            "dka": bool(dka),
            "refractory_to_k": bool(refractory),
            "unwell": bool(unstable),
            "rapid_k_rise_expected": bool(unstable),
            "suspected_hypokalemic_arrest": False,
            "suspected_hyperkalemic_arrest": False,
        },
        "symptoms": {"present": bool(symptoms_ecg or paralysis), "muscle_paralysis": bool(paralysis)},
        "ecg": {
            "hypokalemia_changes": bool(k < 3.5 and symptoms_ecg),
            "hyperkalemia_changes": bool(k >= 5.2 and symptoms_ecg),
        },
        "sample": {"pseudohyperkalemia_suspected": bool(pseudo)},
        "acid_base": {"metabolic_acidosis": bool(acidosis)},
        "glucose": {"pretreatment_mmol_l": None},
        "treatment": {"insulin_glucose_given": False},
        "access": {
            "oral_available": not no_oral,
            "iv_used": bool(k < 3.1 or k >= 6.0),
            "peripheral_available": True,
            "central_available": bool(central_already),
        },
        "infusion": {"pump_available": True, "uses_burette": True, "large_vein": bool(central_already)},
        "medications": {
            "mechanism_classes": medication_mechanism_classes,
            "effect_codes": medication_effect_codes,
        },
        "resuscitation": {"cardiac_arrest": False, "peri_or_cardiac_arrest": False},
    }

    matched = evaluate_electrolyte_rules(bundle.get("rules") or [], context)
    insulin_rule_active = any((r.get("action_json") or {}).get("insulin_soluble_units") is not None for r in matched)
    if insulin_rule_active:
        context_after_ig = {**context, "treatment": {"insulin_glucose_given": True}}
        after = evaluate_electrolyte_rules(bundle.get("rules") or [], context_after_ig)
        seen = {r.get("id") or r.get("rule_code") for r in matched}
        matched.extend(r for r in after if (r.get("id") or r.get("rule_code")) not in seen)

    preferred_protocol_ids = {p.get("id") for p in bundle.get("protocols") or [] if p.get("preferred_for_app")}
    classifications = [r for r in matched if r.get("rule_type") == "CLASSIFICATION" and r.get("protocol_id") in preferred_protocol_ids]
    hard_stops = [r for r in matched if r.get("hard_stop")]
    main_rules = [r for r in matched if r.get("protocol_id") in preferred_protocol_ids and r.get("rule_type") != "CLASSIFICATION" and not r.get("hard_stop")]
    comparator_rules = [r for r in matched if r.get("protocol_id") not in preferred_protocol_ids and r.get("rule_type") != "CLASSIFICATION" and not r.get("hard_stop")]

    relevant_directions = {"LOWER"} if k < 3.5 else ({"RAISE"} if k >= 5.2 else set())
    relevant_modifiers = [m for m in modifiers if not relevant_directions or m.get("direction") in relevant_directions or m.get("direction") == "VARIABLE"]

    dep_matches = [d for d in bundle.get("dependencies") or [] if evaluate_electrolyte_condition(d.get("condition_json") or {}, context)]

    severity_map = {"INFO": "INFORMATIVA", "MILD": "LEVE", "MODERATE": "MODERADA", "SEVERE": "GRAVE", "LOW": "LEVE", "HIGH": "ALTA", "CRITICAL": "CRÍTICA"}
    primary_cls = classifications[0] if classifications else None

    with st.container(border=True):
        st.markdown("### Resultado y plan de corrección")
        st.caption(f"Paciente: {fmt_num(age_years,0)} años · {fmt_num(weight_kg,1)} kg")

        if primary_cls:
            raw_cls = (primary_cls.get("action_json") or {}).get("classification") or primary_cls.get("severity") or ""
            cls = severity_map.get(str(raw_cls).upper(), str(raw_cls).upper())
            disorder = "HIPOPOTASEMIA" if primary_cls.get("disorder") == "HYPO" else "HIPERPOTASEMIA"
            msg = f"**{disorder} {cls} · K {fmt_num(k,1)} mmol/L**"
            if str(raw_cls).upper() == "SEVERE" or str(primary_cls.get("severity") or "").upper() == "CRITICAL":
                st.error(msg)
            elif str(raw_cls).upper() == "MODERATE":
                st.warning(msg)
            else:
                st.info(msg)
        elif 3.5 <= k < 5.2:
            st.success(f"**K {fmt_num(k,1)} mmol/L:** no activa una alteración de potasio en el protocolo principal cargado.")
        else:
            st.warning(f"**K {fmt_num(k,1)} mmol/L:** revisar contexto; no se obtuvo una clasificación principal única.")

        if hf:
            st.info("**Contexto ICC/congestión:** MedCalc prioriza la menor carga de volumen posible y contabiliza el sodio/cloro del diluyente. No convertirá automáticamente la reposición en una bolsa de 500–1000 mL.")
        if renal_with_urine:
            st.warning("**Contexto renal con diuresis:** la reposición se hará por unidades pequeñas y se reevaluará antes de acumular nuevas dosis.")
        if olig_anuria or dialysis:
            st.error("**Contexto renal de alto riesgo:** oliguria/anuria o hemodiálisis requieren manejo individualizado; no se genera reposición automática general.")
        if dka:
            st.warning("**Contexto DKA/HHS:** prevalece el algoritmo específico de crisis hiperglucémica; no se usa como si fuera una hipopotasemia general.")

        if hard_stops:
            for text in _unique_texts(hard_stops):
                st.error(f"**BLOQUEO:** {text}")

        if selected_labels:
            if relevant_modifiers:
                st.markdown("**Medicamentos integrados en la interpretación**")
                st.caption(f"MedCalc revisó automáticamente {len(selected_labels)} medicamento(s) contra la base de modificadores de K.")
                for m in relevant_modifiers:
                    mech_label = m.get("mechanism_label")
                    mechanism = m.get("mechanism")
                    implication = m.get("clinical_implication") or m.get("suggested_action")
                    if mech_label:
                        st.write(f"- **{m.get('generic_name')} · {mech_label}**")
                    else:
                        st.write(f"- **{m.get('generic_name')}**")
                    if mechanism:
                        st.caption(f"Mecanismo: {mechanism}")
                    if implication:
                        st.write(implication)
            else:
                st.caption(f"**Revisión farmacológica activa:** se revisaron {len(selected_labels)} medicamento(s) y no se detectaron modificadores publicados que expliquen o empeoren esta alteración de K.")

        etiology_rules = [r for r in main_rules if str(r.get("rule_type") or "").upper() == "ETIOLOGY"]
        if etiology_rules:
            labels = []
            subtypes = []
            for r in etiology_rules:
                action = r.get("action_json") or {}
                if action.get("etiology_label") and action.get("etiology_label") not in labels:
                    labels.append(action.get("etiology_label"))
                if action.get("etiology_subtype") and action.get("etiology_subtype") not in subtypes:
                    subtypes.append(action.get("etiology_subtype"))
            if len(labels) > 1:
                st.warning("**Mecanismo probable: PATRÓN MIXTO.** " + " + ".join(labels))
            elif labels:
                suffix = f" · {', '.join(subtypes)}" if subtypes else ""
                st.info(f"**Mecanismo probable: {labels[0]}{suffix}.**")
            for text in _unique_texts(etiology_rules):
                st.write(f"- {text}")

        if dep_matches:
            for dep in dep_matches:
                action = dep.get("action_json") or {}
                st.warning(f"**K–Mg:** {dep.get('clinical_rationale') or 'Existe una dependencia K–Mg relevante.'}")
                if action.get("mg_guidance"):
                    st.write(action.get("mg_guidance"))

        urgent = _unique_texts(main_rules, {"TRIAGE", "MEMBRANE_STABILIZATION", "SHIFT", "ELIMINATION"})
        replacement = _unique_texts(main_rules, {"REPLACEMENT", "DEPENDENCY"})
        monitoring = _unique_texts(main_rules, {"MONITORING", "LAB_REPEAT"})
        safety = _unique_texts(main_rules, {"ALERT", "DIAGNOSTIC_CONTEXT", "OTHER"})

        if urgent:
            st.markdown("**Conducta inmediata**")
            for text in urgent:
                st.markdown(f"- {text}")
        if replacement:
            st.markdown("**Corrección indicada por las reglas**")
            for text in replacement:
                st.markdown(f"- {text}")
        if monitoring:
            st.markdown("**Control**")
            for text in monitoring:
                st.markdown(f"- {text}")
        if safety:
            with st.expander("Contexto y seguridad"):
                for text in safety:
                    st.markdown(f"- {text}")

        # --------------------------------------------------------------
        # VÍA ORAL: automática cuando corresponde.
        # --------------------------------------------------------------
        mild_oral = next((r for r in main_rules if r.get("rule_type") == "REPLACEMENT" and (r.get("action_json") or {}).get("route") == "PO"), None)
        if mild_oral and not no_oral and not hard_stops:
            oral_action = mild_oral.get("action_json") or {}
            options_oral = oral_action.get("options") or []
            exact_opt = oral_action.get("auto_regimen") or next((o for o in options_oral if o.get("dose_mmol") is not None), None)
            oral_products = [p for p in bundle.get("products") or [] if str(p.get("route") or "").upper() == "PO"]
            oral_products.sort(key=lambda p: (0 if str(p.get("market") or "").upper() == "CL" else 1, p.get("generic_product_name") or ""))
            if exact_opt and oral_products:
                prod = oral_products[0]
                kcomp = _component_by_code(prod, "K")
                mmol_unit = (kcomp or {}).get("mmol_per_unit")
                if mmol_unit:
                    target = float(exact_opt.get("dose_mmol"))
                    units = product_units_for_mmol(target, float(mmol_unit))
                    freq_day = _fallback_as_float(exact_opt.get("frequency_per_day"))
                    max_daily = _fallback_as_float(exact_opt.get("max_daily_mmol"))
                    interval_h = _fallback_as_float(exact_opt.get("interval_hours"))
                    if interval_h is None and freq_day and freq_day > 0:
                        interval_h = 24.0 / freq_day
                    if freq_day is None and interval_h and interval_h > 0:
                        freq_day = 24.0 / interval_h
                    daily_mmol = target * freq_day if freq_day else None
                    dose_mmol_kg = target / float(weight_kg) if float(weight_kg) > 0 else None
                    daily_mmol_kg = daily_mmol / float(weight_kg) if daily_mmol is not None and float(weight_kg) > 0 else None
                    units_text = fmt_num(units,0) if abs(units-round(units)) < 1e-9 else fmt_num(units,2)
                    st.markdown("**Cómo administrarlo por vía oral**")
                    if interval_h:
                        interval_text = fmt_num(interval_h,0) if abs(interval_h-round(interval_h)) < 1e-9 else fmt_num(interval_h,1)
                        st.success(
                            f"**PAUTA AUTOMÁTICA: {units_text} comprimidos VO cada {interval_text} horas** de "
                            f"{prod.get('generic_product_name')} ({prod.get('concentration_label')}). "
                            f"Aporta **{target:g} mmol de K por dosis**"
                            + (f" y **{fmt_num(daily_mmol,0)} mmol/día**." if daily_mmol is not None else ".")
                        )
                        if dose_mmol_kg is not None:
                            weight_line = f"Equivale a **{fmt_num(dose_mmol_kg,2)} mmol/kg por dosis**"
                            if daily_mmol_kg is not None:
                                weight_line += f" y **{fmt_num(daily_mmol_kg,2)} mmol/kg/día**."
                            else:
                                weight_line += "."
                            st.caption(weight_line)
                        if max_daily:
                            max_units = product_units_for_mmol(float(max_daily), float(mmol_unit))
                            max_units_text = fmt_num(max_units,0) if abs(max_units-round(max_units)) < 1e-9 else fmt_num(max_units,2)
                            st.caption(
                                f"Máximo de la regla: {fmt_num(max_daily,0)} mmol/día en dosis divididas "
                                f"(≈ {max_units_text} comprimidos/día de esta presentación)."
                            )
                        st.caption("La guía no fija una duración única para la hipopotasemia leve; ajustar continuidad según causa, pérdidas en curso y control de K.")
                    else:
                        st.success(f"{target:g} mmol de K = **{units_text} comprimidos** de {prod.get('generic_product_name')} ({prod.get('concentration_label')}).")

        # --------------------------------------------------------------
        # IV AUTOMÁTICA: ya NO pregunta cuántos mmol quiere el usuario.
        # Se genera una unidad inicial conservadora desde action_json Supabase.
        # --------------------------------------------------------------
        modsev = next((r for r in main_rules if r.get("rule_type") == "REPLACEMENT" and (r.get("action_json") or {}).get("route_strategy")), None)
        dka_low_rule = next((r for r in main_rules if (r.get("action_json") or {}).get("dka_specific") and (r.get("action_json") or {}).get("k_replacement_rate_mmol_h") is not None), None)

        if modsev and not hard_stops and k < 3.1:
            cfg = modsev.get("action_json") or {}
            target_iv = float(cfg.get("auto_initial_unit_mmol") or 10)
            rate_mmol_h = float(cfg.get("auto_default_rate_mmol_h") or 10)
            if dka_low_rule:
                rate_mmol_h = float((dka_low_rule.get("action_json") or {}).get("k_replacement_rate_mmol_h") or rate_mmol_h)

            # En ICC/congestión se prioriza volumen pequeño y vía central. En el
            # escenario general se usa exactamente la concentración periférica máxima
            # publicada por Queensland: 10 mmol / 250 mL = 40 mmol/L.
            volume_sensitive = bool(hf)
            recommended_line = "CENTRAL" if volume_sensitive or central_already else "PERIPHERAL"
            if k < 2.5 or symptoms_ecg or paralysis:
                recommended_line = "CENTRAL"
            final_volume_ml = float(cfg.get("auto_central_final_volume_ml") or 100) if recommended_line == "CENTRAL" else float(cfg.get("auto_peripheral_final_volume_ml") or 250)
            duration_h = target_iv / rate_mmol_h if rate_mmol_h > 0 else 1.0
            target_mmol_kg = target_iv / float(weight_kg) if float(weight_kg) > 0 else None
            rate_mmol_kg_h = rate_mmol_h / float(weight_kg) if float(weight_kg) > 0 else None
            planned_volume_ml_kg = final_volume_ml / float(weight_kg) if float(weight_kg) > 0 else None

            products = [p for p in bundle.get("products") or [] if str(p.get("route") or "").upper() == "IV" and str(p.get("preparation_type") or "").upper() != "PREMIXED_READY_TO_USE"]
            products.sort(key=lambda p: (
                0 if str(p.get("market") or "").upper() == "CL" else 1,
                0 if "10%" in str(p.get("concentration_label") or "") else 1,
                p.get("generic_product_name") or "",
            ))

            prepared = None
            for product in products:
                kcomp = _component_by_code(product, "K")
                if not kcomp or not kcomp.get("mmol_per_ml"):
                    continue
                compat = [x for x in db.electrolyte_compatibilities(product.get("id")) if x.get("compatible")]
                compat_ids = {x.get("diluent_id") for x in compat}
                diluent = next((d for d in bundle.get("diluents") or [] if d.get("id") in compat_ids and abs(float(d.get("container_volume_ml") or 0) - final_volume_ml) < 0.01 and str(d.get("diluent_code") or "").startswith("NS09")), None)
                if not diluent:
                    continue
                try:
                    prep = prepare_infusion(
                        target_mmol=target_iv,
                        product_mmol_per_ml=kcomp.get("mmol_per_ml"),
                        valence=(bundle.get("analyte") or {}).get("valence") or 1,
                        container_volume_ml=final_volume_ml,
                        duration_h=duration_h,
                        preparation_mode="REPLACE_EQUAL_VOLUME",
                        ampoule_volume_ml=product.get("container_volume_ml"),
                    )
                except Exception:
                    continue
                route_eval = evaluate_administration_options(
                    prep, bundle.get("limits") or [], context,
                    available_line_types=(recommended_line,),
                    product_id=product.get("id"),
                    preferred_line_type=recommended_line,
                )
                if route_eval.get("suggested_line"):
                    prepared = (product, diluent, prep, recommended_line)
                    break

            st.markdown("**Preparación IV automática · unidad inicial**")
            if prepared:
                product, diluent, prep, line = prepared
                conc = _component_by_code(product, "K") or {}
                mmol_ml = float(conc.get("mmol_per_ml") or 0)
                st.success(
                    f"**Administrar {fmt_num(target_iv,0)} mmol (= mEq) de K** como unidad inicial, a **{fmt_num(rate_mmol_h,0)} mmol/h**."
                )
                if target_mmol_kg is not None and rate_mmol_kg_h is not None:
                    st.caption(
                        f"Para {fmt_num(weight_kg,1)} kg: **{fmt_num(target_mmol_kg,2)} mmol/kg** en esta unidad "
                        f"y **{fmt_num(rate_mmol_kg_h,2)} mmol/kg/h**."
                    )
                st.write(
                    f"**1. Producto:** {product.get('generic_product_name')} · {product.get('concentration_label')} "
                    f"(**{fmt_num(mmol_ml,3)} mmol/mL**)."
                )
                st.write(
                    f"**2. Extraer:** **{fmt_num(prep.concentrate_volume_ml,2)} mL** de KCl. "
                    f"{('Abrir ' + str(prep.ampoules.get('whole_ampoules_to_open')) + ' ampolla(s).') if prep.ampoules else ''}"
                )
                st.write(
                    f"**3. Preparar:** retirar del envase de **{_es_diluent(diluent.get('name'))}** el mismo volumen "
                    f"({fmt_num(prep.concentrate_volume_ml,2)} mL), añadir el KCl y dejar **volumen final {fmt_num(prep.final_volume_ml,0)} mL**."
                )
                st.write(
                    f"**4. Administrar:** vía **{_es_line(line)}**, con bomba, a **{fmt_num(prep.rate_ml_h,0)} mL/h** durante **{fmt_num(duration_h,1)} h** "
                    f"= **{fmt_num(prep.rate_mmol_h,1)} mmol/h**. Concentración final: **{fmt_num(prep.final_concentration_mmol_l,1)} mmol/L**."
                )
                if float(weight_kg) > 0:
                    st.caption(
                        f"Carga de volumen de esta unidad: **{fmt_num(prep.final_volume_ml / float(weight_kg),2)} mL/kg**; "
                        f"velocidad de volumen: **{fmt_num(prep.rate_ml_h / float(weight_kg),2)} mL/kg/h**."
                    )
                if line == "CENTRAL" and not central_already:
                    st.warning("Esta preparación compacta requiere **vía venosa central**. Si no dispone de ella, no aumente automáticamente a una bolsa de 500–1000 mL; priorice vía oral cuando sea posible o utilice la alternativa periférica validada según protocolo local.")
                if hf:
                    st.info(
                        f"**Carga de volumen de esta unidad: {fmt_num(prep.final_volume_ml,0)} mL "
                        f"({fmt_num(prep.final_volume_ml / float(weight_kg),2)} mL/kg)**. "
                        "En ICC/congestión se contabiliza dentro del balance total y se prioriza una preparación compacta validada."
                    )
                if renal_with_urine:
                    st.info("Por el contexto renal, MedCalc limita la automatización a **una unidad inicial** y exige reevaluar K, función renal y diuresis antes de repetir.")
                st.caption(
                    f"El objetivo de 60–80 mmol adicionales en 24 h NO se administra como una sola bolsa. Esta es una unidad inicial; repetir K aproximadamente a las {cfg.get('auto_reassess_after_unit_hours') or 4} h y ajustar las siguientes unidades según respuesta."
                )
            else:
                st.error("No se encontró una preparación compacta que cumpla simultáneamente producto, diluyente y límites clínicos publicados. MedCalc no aumentará automáticamente a 500–1000 mL para forzar una solución.")

        if dka_low_rule:
            act = dka_low_rule.get("action_json") or {}
            st.error(
                f"**DKA/HHS:** con K <{fmt_num(act.get('hold_insulin_until_k_gt_mmol_l'),1)} mmol/L, retrasar insulina hasta superar ese valor y reponer K a {fmt_num(act.get('k_replacement_rate_mmol_h'),0)} mmol/h."
            )

        source_objs = []
        seen_src = set()
        for r in classifications + main_rules + hard_stops:
            src = _electrolyte_rule_source(r)
            key = src.get("url") or src.get("title")
            if key and key not in seen_src:
                seen_src.add(key)
                source_objs.append(src)
        if source_objs:
            with st.expander("Fuentes y discrepancias"):
                for src in source_objs:
                    org = src.get("organization") or ""
                    title = src.get("title") or ""
                    st.write(f"**{org}** · {title}")
                    if src.get("edition"):
                        st.caption(str(src.get("edition")))
                    if src.get("url"):
                        st.link_button("Abrir fuente", src.get("url"), key=f"auto_src_{abs(hash(str(src.get('url'))))}")
                if comparator_rules:
                    st.markdown("**Otras fuentes activas / discrepancias**")
                    for text in _unique_texts(comparator_rules):
                        st.caption(f"• {text}")

    st.caption("Las decisiones clínicas y sus fuentes permanecen en Supabase. Edad y peso se incorporan al contexto; el peso expresa la intensidad real de la reposición sin convertir automáticamente el K plasmático en un déficit corporal calculado.")


# Conservar la calculadora de POTASIO V8.0.8 sin reescribirla.
_page_potassium_v808 = page_electrolytes


def _el_v2_optional_float(label, key, placeholder="Opcional"):
    raw = st.text_input(label, value="", placeholder=placeholder, key=key)
    return _fallback_as_float(raw)


def _el_v2_product(bundle, code):
    return next((p for p in (bundle.get("products") or []) if p.get("product_code") == code), None)


def _el_v2_rule_by_strategy(rules, strategy):
    target = str(strategy or "").upper()
    for r in rules or []:
        if str((r.get("action_json") or {}).get("strategy") or "").upper() == target:
            return r
    return None


def _el_v2_rule_by_type(rules, rule_type):
    t = str(rule_type or "").upper()
    return [r for r in (rules or []) if str(r.get("rule_type") or "").upper() == t]


def _el_v2_classification(rules):
    classes = _el_v2_rule_by_type(rules, "CLASSIFICATION")
    if not classes:
        return None
    return sorted(classes, key=lambda r: (r.get("priority") or 100, r.get("rule_code") or ""))[0]


def _el_v2_patient(prefix, include_sex=False):
    cols = st.columns(3 if include_sex else 2)
    with cols[0]:
        age = st.number_input("Edad (años)", min_value=18.0, max_value=120.0, value=50.0, step=1.0, key=f"{prefix}_age")
    with cols[1]:
        weight = st.number_input("Peso (kg)", min_value=25.0, max_value=300.0, value=70.0, step=0.5, key=f"{prefix}_weight")
    sex = None
    if include_sex:
        with cols[2]:
            sex = st.selectbox("Sexo para estimar agua corporal", ["Hombre", "Mujer"], key=f"{prefix}_sex")
    return float(age), float(weight), sex


def _el_v2_medications(prefix, analyte_code):
    with st.expander("Medicamentos actuales · opcional"):
        meds = db.search_medications("", limit=max(COUNTS.get("medications", 1000), 1000))
        labels = [f"{m['principio_activo']} · {m['med_id']}" for m in meds]
        to_id = {f"{m['principio_activo']} · {m['med_id']}": m["med_id"] for m in meds}
        selected = st.multiselect("Medicamentos del paciente", labels, default=[], key=f"{prefix}_meds")
    ids = tuple(to_id[x] for x in selected) if selected else tuple()
    return _electrolyte_modifier_rows_cached(ids, analyte_code) if ids else []


def _el_v2_render_modifiers(modifiers, value_label):
    if not modifiers:
        return
    st.markdown("**Medicamentos que pueden modificar este electrolito**")
    for m in modifiers:
        direction = {"LOWER": "disminuir", "RAISE": "aumentar", "VARIABLE": "modificar"}.get(str(m.get("direction") or "").upper(), "modificar")
        text = m.get("interpretation_text") or m.get("clinical_implication") or ""
        st.write(f"• **{m.get('generic_name')}:** puede {direction} {value_label}. {text}")


def _el_v2_sources(bundle, matched):
    sources = []
    seen = set()
    for r in matched or []:
        src = _electrolyte_rule_source(r)
        key = src.get("url") or src.get("title")
        if key and key not in seen:
            seen.add(key); sources.append(src)
    for proto in bundle.get("protocols") or []:
        src = proto.get("source") or {}
        key = src.get("url") or src.get("title")
        if key and key not in seen:
            seen.add(key); sources.append(src)
    if sources:
        with st.expander("Fuentes y discrepancias"):
            for i, src in enumerate(sources):
                st.write(f"**{src.get('organization') or 'Fuente'}** · {src.get('title') or '—'}")
                if src.get("edition"): st.caption(str(src.get("edition")))
                if src.get("url"): st.link_button("Abrir fuente", src.get("url"), key=f"elv2src_{i}_{abs(hash(str(src.get('url'))))}")


def _page_sodium_v2():
    header("Hidroelectrolitos · Sodio", "Hiponatremia, hipernatremia, agua libre, hiperglucemia y contexto de volumen.")
    age, weight, sex = _el_v2_patient("na_v2", include_sex=True)
    c1, c2 = st.columns([1, 1.4])
    with c1:
        na = st.number_input("Sodio plasmático (mmol/L = mEq/L)", min_value=90.0, max_value=190.0, value=140.0, step=1.0, key="na_v2_value")
    with c2:
        tags = st.multiselect("Contexto que cambia la conducta", [
            "Hipovolemia", "Euvolemia / sospecha SIADH", "Hipervolemia / ICC / cirrosis",
            "Inestabilidad hemodinámica", "Síntomas neurológicos graves (convulsión/coma)",
            "Síntomas neurológicos moderados (cefalea/somnolencia/confusión)", "AKI/ERC", "Hemodiálisis"
        ], key="na_v2_context")
    with st.expander("Glucosa/osmolalidad · opcional"):
        glucose = _el_v2_optional_float("Glucosa (mmol/L)", "na_v2_glucose", "Ej. 22")
        urea = _el_v2_optional_float("Urea (mmol/L)", "na_v2_urea", "Ej. 8")
    mods = _el_v2_medications("na_v2", "NA")
    bundle = _electrolyte_bundle_cached("NA")
    if not bundle.get("rules"):
        st.error("No hay reglas de sodio publicadas. Ejecute el SQL Hidroelectrolitos V2."); return
    ctx = {
        "patient":{"age_years":age,"weight_kg":weight,"sex":sex}, "serum":{"na_mmol_l":float(na)},
        "volume":{"hypovolemic":"Hipovolemia" in tags,"euvolemic":"Euvolemia / sospecha SIADH" in tags,"hypervolemic":"Hipervolemia / ICC / cirrosis" in tags},
        "hemodynamic":{"unstable":"Inestabilidad hemodinámica" in tags},
        "symptoms":{"severe_neurologic":"Síntomas neurológicos graves (convulsión/coma)" in tags,"moderate_neurologic":"Síntomas neurológicos moderados (cefalea/somnolencia/confusión)" in tags},
        "renal":{"impairment":"AKI/ERC" in tags,"dialysis":"Hemodiálisis" in tags},
        "glucose":{"hyperglycemia": glucose is not None},
    }
    matched = evaluate_electrolyte_rules(bundle.get("rules") or [], ctx)
    cls = _el_v2_classification(matched)
    with st.container(border=True):
        st.markdown("### Resultado y plan de corrección")
        if cls:
            st.info(f"**{(cls.get('action_json') or {}).get('severity_label') or cls.get('recommendation_text')} · Na {fmt_num(na,0)} mmol/L**")
        else:
            st.success(f"Na {fmt_num(na,0)} mmol/L: no activa una clasificación hipo/hiper de las reglas publicadas.")
        _el_v2_render_modifiers(mods, "el sodio")
        if glucose is not None:
            try:
                na16 = corrected_sodium_for_hyperglycemia(serum_na=na, glucose_mmol_l=glucose, correction_mmol_per_100mg_dl=1.6)
                na24 = corrected_sodium_for_hyperglycemia(serum_na=na, glucose_mmol_l=glucose, correction_mmol_per_100mg_dl=2.4)
                st.write(f"**Na corregido por hiperglucemia (sensibilidad):** {fmt_num(na16,1)} mmol/L con factor 1,6; {fmt_num(na24,1)} mmol/L con factor 2,4. La discrepancia se muestra, no se oculta.")
                st.caption(f"Tonicidad calculada ≈ {fmt_num(effective_osmolality_mosm_kg(sodium_mmol_l=na, glucose_mmol_l=glucose),0)} mOsm/kg" + (f" · Osmolalidad calculada ≈ {fmt_num(calculated_serum_osmolality_mosm_kg(sodium_mmol_l=na, glucose_mmol_l=glucose, urea_mmol_l=urea),0)} mOsm/kg" if urea is not None else ""))
            except Exception: pass
        shock = _el_v2_rule_by_strategy(matched, "RESTORE_CIRCULATION_FIRST")
        water = _el_v2_rule_by_strategy(matched, "FREE_WATER")
        hypertonic = _el_v2_rule_by_strategy(matched, "HYPERTONIC_3_PERCENT_BOLUS")
        if shock:
            st.error("**Prioridad:** restaurar primero la perfusión con cristaloide isotónico; MedCalc no calcula agua libre hasta estabilizar la circulación.")
        elif water:
            act=water.get("action_json") or {}; target=float(act.get("target_na") or 140); maxdrop=float(act.get("max_drop_24h_mmol_l") or 10)
            try:
                full=float(free_water_deficit_l(weight_kg=weight,serum_na=na,target_na=target,sex=sex))
                target24=max(target,float(na)-maxdrop)
                day=float(free_water_deficit_l(weight_kg=weight,serum_na=na,target_na=target24,sex=sex))
                st.success(f"**Agua libre estimada hasta Na {fmt_num(target,0)}:** {fmt_num(full,2)} L ({fmt_num(full*1000/weight,1)} mL/kg).")
                st.write(f"**Máximo teórico para las primeras 24 h por límite de corrección:** objetivo Na ≈ {fmt_num(target24,0)} mmol/L → {fmt_num(day,2)} L de agua libre estimada, antes de ajustar pérdidas en curso.")
                st.write("Preferir agua oral/enteral si es posible. Si se requiere IV, la regla contempla glucosa 5%; controlar Na aproximadamente cada 4 h en las primeras 24 h y recalcular.")
                st.warning("El déficit calculado NO es una orden de infundir todo ese volumen: debe ajustarse a pérdidas en curso, balance, función renal e ICC/congestión.")
            except Exception as e: st.warning(str(e))
        if hypertonic:
            act=hypertonic.get("action_json") or {}; bolus=float(act.get("bolus_ml") or 150); mins=float(act.get("duration_min") or 20)
            prod=_el_v2_product(bundle, act.get("product_code")); comp=_component_by_code(prod,"NA") if prod else None
            conc=float((comp or {}).get("mmol_per_ml") or 0.513)*1000
            try:
                tbw=float(total_body_water_l(weight,sex=sex)); delta=float(predicted_delta_na_after_infusate(serum_na=na,tbw_l=tbw,infusate_na_mmol_l=conc,volume_ml=bolus))
                st.error(f"**Síntomas neurológicos:** comparador ESE clásico: NaCl 3% **{fmt_num(bolus,0)} mL en {fmt_num(mins,0)} min**. Predicción matemática aproximada para {fmt_num(weight,1)} kg: ΔNa ≈ **+{fmt_num(delta,1)} mmol/L**; medir Na y clínica, no confiar en la fórmula para decidir repeticiones.")
                st.caption("La pauta exacta de 150 mL procede de la guía europea 2014; existe actualización ESE-ERA 2026 anunciada, pero el texto completo de recomendaciones no está incorporado como regla automática. Queensland 2026 es el protocolo principal de límites/seguridad.")
            except Exception: pass
        for r in matched:
            if str(r.get("rule_type") or "").upper() in {"REPLACEMENT","ALERT","MONITORING"} and r not in {shock,water,hypertonic}:
                if r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
        _el_v2_sources(bundle, matched)


def _page_magnesium_v2():
    header("Hidroelectrolitos · Magnesio", "Reposición oral/IV, hipermagnesemia, función renal y dependencia con K/Ca.")
    age, weight, _ = _el_v2_patient("mg_v2")
    c1,c2=st.columns([1,1.4])
    with c1:
        mg_unit=st.selectbox("Unidad de Mg",["mg/dL","mmol/L","mEq/L","µmol/L"],key="mg_v2_unit")
        mg_defaults={"mg/dL":2.0,"mmol/L":0.82,"mEq/L":1.64,"µmol/L":820.0}
        mg_native=st.number_input(f"Magnesio ({mg_unit})",min_value=0.01,max_value=5000.0,value=float(mg_defaults[mg_unit]),step=0.05 if mg_unit!="µmol/L" else 10.0,key=f"mg_v2_value_{mg_unit}")
        mg=float(laboratory_value_to_mmol_l("MG",mg_native,mg_unit))
        if mg_unit!="mmol/L": st.caption(f"≈ {fmt_num(mg,2)} mmol/L para aplicar las reglas")
    with c2: tags=st.multiselect("Contexto que cambia la conducta",["Síntomas/arrítmia/convulsión","AKI/ERC","Oliguria/anuria","Hemodiálisis","No puede recibir vía oral","ICC/congestión"],key="mg_v2_context")
    mods=_el_v2_medications("mg_v2","MG"); bundle=_electrolyte_bundle_cached("MG")
    ctx={"patient":{"age_years":age,"weight_kg":weight},"serum":{"mg_mmol_l":float(mg)},"symptoms":{"present":"Síntomas/arrítmia/convulsión" in tags},"renal":{"impairment":any(x in tags for x in ["AKI/ERC","Oliguria/anuria","Hemodiálisis"]),"failure":any(x in tags for x in ["Oliguria/anuria","Hemodiálisis"]),"dialysis":"Hemodiálisis" in tags},"access":{"oral_available":"No puede recibir vía oral" not in tags},"volume":{"volume_sensitive":"ICC/congestión" in tags}}
    matched=evaluate_electrolyte_rules(bundle.get("rules") or [],ctx); cls=_el_v2_classification(matched)
    with st.container(border=True):
        st.markdown("### Resultado y plan de corrección")
        if cls: st.info(f"**{(cls.get('action_json') or {}).get('severity_label') or cls.get('recommendation_text')} · Mg {fmt_num(mg,2)} mmol/L**")
        else: st.success(f"Mg {fmt_num(mg,2)} mmol/L: sin clasificación automática hipo/hiper activa.")
        _el_v2_render_modifiers(mods,"el magnesio")
        oral=next((r for r in matched if (r.get("action_json") or {}).get("route")=="PO"),None)
        iv=next((r for r in matched if (r.get("action_json") or {}).get("route")=="IV" and r.get("rule_type")=="REPLACEMENT"),None)
        if oral:
            a=oral.get("action_json") or {}; prod=_el_v2_product(bundle,a.get("product_code")); comp=_component_by_code(prod,"MG") if prod else None; per=float((comp or {}).get("mmol_per_unit") or 1.54)
            st.success(f"**Vía oral:** {a.get('units_min')}–{a.get('units_max')} comprimidos cada {fmt_num(a.get('interval_hours'),0)} h = {fmt_num(per*float(a.get('units_min') or 1),2)}–{fmt_num(per*float(a.get('units_max') or 2),2)} mmol de Mg por dosis. Máximo: {a.get('max_units_day')} comprimidos/día.")
        if iv:
            a=iv.get("action_json") or {}; target=float(a.get("auto_initial_mmol") or a.get("target_mmol_min") or 10); prod=_el_v2_product(bundle,a.get("product_code")); comp=_component_by_code(prod,"MG") if prod else None; c=float((comp or {}).get("mmol_per_ml") or 2.0); ml=target/c; vol=float(a.get("final_volume_ml") or 100); dur=float(a.get("duration_h") or 1)
            st.success(f"**Unidad IV inicial automática:** {fmt_num(target,0)} mmol Mg. Extraer **{fmt_num(ml,2)} mL** de sulfato de magnesio, completar a **{fmt_num(vol,0)} mL** con NaCl 0,9% y pasar en **{fmt_num(dur,1)} h** = {fmt_num(vol/dur,0)} mL/h, {fmt_num(target/dur,1)} mmol/h = {fmt_num(target*2/dur,1)} mEq/h.")
            st.caption(f"Para {fmt_num(weight,1)} kg: {fmt_num(target/weight,2)} mmol/kg · volumen {fmt_num(vol/weight,2)} mL/kg. Reevaluar Mg/síntomas en {a.get('lab_repeat_hours_min')}–{a.get('lab_repeat_hours_max')} h antes de repetir.")
            if a.get("target_mmol_max") and float(a.get("target_mmol_max"))>target: st.write(f"La guía permite un rango de {a.get('target_mmol_min')}–{a.get('target_mmol_max')} mmol; MedCalc usa {fmt_num(target,0)} mmol como unidad inicial conservadora y obliga a reevaluar.")
        for r in matched:
            if r.get("rule_type") in {"MEMBRANE_STABILIZATION","ELIMINATION","ALERT"} and r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
        _el_v2_sources(bundle,matched)


def _page_calcium_v2():
    header("Hidroelectrolitos · Calcio", "Calcio ionizado/total, albúmina, Mg concomitante e hipercalcemia.")
    age, weight, _=_el_v2_patient("ca_v2")
    c1,c2,c3=st.columns(3)
    with c1: kind=st.radio("Medición",["Ionizado","Total"],horizontal=True,key="ca_v2_kind")
    with c2:
        ca_unit=st.selectbox("Unidad de Ca",["mg/dL","mmol/L","mEq/L","µmol/L"],key="ca_v2_unit")
        ca_defaults={"mg/dL":8.8,"mmol/L":2.20,"mEq/L":4.40,"µmol/L":2200.0}
        ca_native=st.number_input(f"Calcio ({ca_unit})",min_value=0.01,max_value=10000.0,value=float(ca_defaults[ca_unit]),step=0.05 if ca_unit!="µmol/L" else 10.0,key=f"ca_v2_value_{ca_unit}")
        ca=float(laboratory_value_to_mmol_l("CA",ca_native,ca_unit))
        if ca_unit!="mmol/L": st.caption(f"≈ {fmt_num(ca,2)} mmol/L para aplicar las reglas")
    with c3:
        albumin=_el_v2_optional_float("Albúmina g/L · si calcio total","ca_v2_albumin","Ej. 32") if kind=="Total" else None
    with st.expander("Dato relacionado · opcional"):
        mg=_el_v2_optional_float("Magnesio (mmol/L)","ca_v2_mg","Ej. 0,55")
    tags=st.multiselect("Contexto que cambia la conducta",["Síntomas (tetania/convulsión/parestesias)","AKI/ERC","Oliguria/anuria","Hemodiálisis","ICC/congestión / restricción de volumen","No puede recibir vía oral"],key="ca_v2_context")
    mods=_el_v2_medications("ca_v2","CA"); bundle=_electrolyte_bundle_cached("CA")
    if kind=="Ionizado": interpret=float(ca); ion=float(ca)
    else:
        ion=None
        try: interpret=float(corrected_calcium_mmol_l(total_ca_mmol_l=ca,albumin_g_l=albumin)) if albumin is not None else float(ca)
        except Exception: interpret=float(ca)
    ctx={"patient":{"age_years":age,"weight_kg":weight},"calcium":{"interpretive_mmol_l":interpret,"ionized_mmol_l":ion},"serum":{"mg_mmol_l":mg},"symptoms":{"present":"Síntomas (tetania/convulsión/parestesias)" in tags},"renal":{"impairment":any(x in tags for x in ["AKI/ERC","Oliguria/anuria","Hemodiálisis"]),"failure":any(x in tags for x in ["Oliguria/anuria","Hemodiálisis"]),"dialysis":"Hemodiálisis" in tags},"volume":{"volume_sensitive":"ICC/congestión / restricción de volumen" in tags},"access":{"oral_available":"No puede recibir vía oral" not in tags}}
    matched=evaluate_electrolyte_rules(bundle.get("rules") or [],ctx); cls=_el_v2_classification(matched)
    with st.container(border=True):
        st.markdown("### Resultado y plan de corrección")
        if kind=="Total" and albumin is not None: st.write(f"Calcio total {fmt_num(ca,2)} → **corregido aproximado {fmt_num(interpret,2)} mmol/L**. El calcio ionizado sigue siendo la medida fisiológicamente preferida.")
        elif kind=="Total": st.warning("Sin albúmina, MedCalc usa el calcio total solo como orientación. Queensland 2026 advierte que incluso el calcio corregido es menos fiable que el ionizado.")
        if cls: st.info(f"**{(cls.get('action_json') or {}).get('severity_label') or cls.get('recommendation_text')}**")
        _el_v2_render_modifiers(mods,"el calcio")
        oral=next((r for r in matched if (r.get("action_json") or {}).get("route")=="PO"),None)
        bolus=next((r for r in matched if r.get("rule_code")=="CA_QLD_IV_BOLUS"),None)
        cont=next((r for r in matched if r.get("rule_code")=="CA_QLD_CONT_INF"),None)
        if oral:
            a=oral.get("action_json") or {}; st.success(f"**Vía oral:** calcio 600 mg, **{a.get('units_min')}–{a.get('units_max')} comprimidos al día** = 15–30 mmol Ca/día, con alimentos.")
        if bolus:
            a=bolus.get("action_json") or {}; target=float(a.get("target_mmol") or 4.4); amp=int(a.get("ampoules") or 2); vol=float(a.get("final_volume_ml") or 100); mins=float(a.get("duration_min") or 20); conc_ml=amp*10
            st.error(f"**Corrección IV inicial:** {amp} ampollas de gluconato de calcio 10% = **{fmt_num(target,1)} mmol Ca**. Para mantener volumen final {fmt_num(vol,0)} mL, retirar {fmt_num(conc_ml,0)} mL de una bolsa de NaCl 0,9% de {fmt_num(vol,0)} mL, añadir las {amp} ampollas y pasar en **{fmt_num(mins,0)} min** ≈ {fmt_num(vol/(mins/60),0)} mL/h.")
            st.caption(f"Carga inicial: {fmt_num(target/weight,3)} mmol/kg y {fmt_num(vol/weight,2)} mL/kg. Repetir calcio aproximadamente a las {a.get('lab_repeat_hours')} h.")
        if cont:
            a=cont.get("action_json") or {}; st.write(f"**Si persiste necesidad y NO hay restricción de volumen:** después del bolo, la guía describe {a.get('ampoules')} ampollas ({a.get('target_mmol')} mmol) + {a.get('diluent_volume_ml')} mL NaCl 0,9% a **{a.get('rate_ml_h')} mL/h** durante 1–2 días, ajustando por calcio. En ICC/ERC sensible a volumen esta fase no se automatiza.")
        for r in matched:
            if r.get("rule_type") in {"DEPENDENCY","ALERT"} and r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
            if r.get("rule_code")=="CA_QLD_HYPER_REHYD": st.warning(r.get("recommendation_text"))
        _el_v2_sources(bundle,matched)


def _page_phosphate_v2():
    header("Hidroelectrolitos · Fósforo", "Hipofosfatemia, hiperfosfatemia, carga adicional de Na/K y contextos de realimentación.")
    age, weight, _=_el_v2_patient("p_v2")
    c1,c2=st.columns([1,1.4])
    with c1:
        p_unit=st.selectbox("Unidad de fósforo",["mg/dL","mmol/L","µmol/L"],key="p_v2_unit")
        p_defaults={"mg/dL":3.5,"mmol/L":1.13,"µmol/L":1130.0}
        p_native=st.number_input(f"Fósforo ({p_unit})",min_value=0.01,max_value=10000.0,value=float(p_defaults[p_unit]),step=0.05 if p_unit!="µmol/L" else 10.0,key=f"p_v2_value_{p_unit}")
        pval=float(laboratory_value_to_mmol_l("P",p_native,p_unit))
        if p_unit!="mmol/L": st.caption(f"≈ {fmt_num(pval,2)} mmol/L para aplicar las reglas")
    with c2: tags=st.multiselect("Contexto que cambia la conducta",["Malnutrición/alcohol/realimentación/TPN","Recuperación de DKA o falla respiratoria","Paciente crítico","AKI/ERC","Oliguria/anuria","Hemodiálisis","No puede recibir vía oral","Laboratorio informa fósforo alto"],key="p_v2_context")
    mods=_el_v2_medications("p_v2","P"); bundle=_electrolyte_bundle_cached("P")
    ctx={"patient":{"age_years":age,"weight_kg":weight},"serum":{"p_mmol_l":float(pval)},"phosphate":{"high_risk_context":any(x in tags for x in ["Malnutrición/alcohol/realimentación/TPN","Recuperación de DKA o falla respiratoria"]),"above_lab_range":"Laboratorio informa fósforo alto" in tags},"clinical":{"critically_ill":"Paciente crítico" in tags},"renal":{"impairment":any(x in tags for x in ["AKI/ERC","Oliguria/anuria","Hemodiálisis"]),"failure":any(x in tags for x in ["Oliguria/anuria","Hemodiálisis"]),"dialysis":"Hemodiálisis" in tags},"access":{"oral_available":"No puede recibir vía oral" not in tags}}
    matched=evaluate_electrolyte_rules(bundle.get("rules") or [],ctx); cls=_el_v2_classification(matched)
    with st.container(border=True):
        st.markdown("### Resultado y plan de corrección")
        if cls: st.info(f"**{(cls.get('action_json') or {}).get('severity_label') or cls.get('recommendation_text')} · P {fmt_num(pval,2)} mmol/L**")
        _el_v2_render_modifiers(mods,"el fósforo")
        for r in matched:
            if r.get("rule_type")=="DIAGNOSTIC_CONTEXT" and r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
        oral=next((r for r in matched if (r.get("action_json") or {}).get("route")=="PO"),None)
        critical=next((r for r in matched if r.get("rule_code")=="P_QLD_CRITICAL"),None)
        iv=critical or next((r for r in matched if r.get("rule_code")=="P_QLD_IV"),None)
        if oral:
            a=oral.get("action_json") or {}; prod=_el_v2_product(bundle,a.get("product_code")); pcomp=_component_by_code(prod,"P") if prod else None; nacomp=_component_by_code(prod,"NA") if prod else None; kcomp=_component_by_code(prod,"K") if prod else None
            per=float((pcomp or {}).get("mmol_per_unit") or 16.1)
            st.success(f"**Vía oral:** {a.get('units_min')}–{a.get('units_max')} comprimidos efervescentes por dosis = {fmt_num(per*float(a.get('units_min') or 1),1)}–{fmt_num(per*float(a.get('units_max') or 2),1)} mmol de P, hasta **{a.get('frequency_max_per_day')} veces al día** según respuesta/tolerancia.")
            if nacomp or kcomp: st.caption(f"Cada comprimido añade aproximadamente **{fmt_num((nacomp or {}).get('mmol_per_unit') or 0,1)} mmol Na** y **{fmt_num((kcomp or {}).get('mmol_per_unit') or 0,1)} mmol K**; MedCalc los contabiliza para reposición conjunta.")
        if iv:
            a=iv.get("action_json") or {}; target=float(a.get("target_mmol") or 10); vol=float(a.get("final_volume_ml") or 250); dur=float(a.get("duration_h") or a.get("default_duration_h") or 4); prod=_el_v2_product(bundle,a.get("product_code")); pcomp=_component_by_code(prod,"P") if prod else None; nacomp=_component_by_code(prod,"NA") if prod else None; c=float((pcomp or {}).get("mmol_per_ml") or 1); ml=target/c
            st.error(f"**Preparación IV:** {fmt_num(target,0)} mmol P. Extraer **{fmt_num(ml,2)} mL** de {prod.get('generic_product_name') if prod else 'fosfato sódico'}, retirar el mismo volumen de una bolsa de NaCl 0,9% de {fmt_num(vol,0)} mL, añadir el concentrado y pasar volumen final **{fmt_num(vol,0)} mL en {fmt_num(dur,1)} h** = {fmt_num(vol/dur,0)} mL/h, {fmt_num(target/dur,1)} mmol P/h.")
            st.caption(f"Para {fmt_num(weight,1)} kg: {fmt_num(target/weight,2)} mmol/kg; volumen {fmt_num(vol/weight,2)} mL/kg. Aporte adicional de Na ≈ {fmt_num(float((nacomp or {}).get('mmol_per_ml') or 1)*ml,1)} mmol.")
            if critical: st.warning("Paciente crítico: esta pauta concentrada es preferentemente central y requiere control de P, Ca y función renal cada 1–2 h. No mezclar automáticamente calcio y fosfato en la misma bolsa/línea.")
        for r in matched:
            if r.get("rule_type")=="MONITORING" and r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
        _el_v2_sources(bundle,matched)


def _page_chloride_ab_v2():
    header("Hidroelectrolitos · Cloro y ácido-base", "Anion gap, compensación respiratoria y patrones cloro-responsivos/hiperclorémicos.")
    c1,c2,c3=st.columns(3)
    with c1: na=st.number_input("Na (mmol/L = mEq/L)",90.0,190.0,140.0,1.0,key="ab_v2_na")
    with c2: cl=st.number_input("Cl (mmol/L = mEq/L)",60.0,140.0,104.0,1.0,key="ab_v2_cl")
    with c3: hco3=st.number_input("HCO₃⁻ (mmol/L = mEq/L)",3.0,50.0,24.0,1.0,key="ab_v2_hco3")
    c4,c5,c6=st.columns(3)
    with c4: ph=st.number_input("pH",6.80,7.80,7.40,0.01,key="ab_v2_ph")
    with c5: pco2=st.number_input("pCO₂ (mmHg)",10.0,100.0,40.0,1.0,key="ab_v2_pco2")
    with c6: cl_lab=st.selectbox("Cl según rango del laboratorio",["Normal","Bajo","Alto"],key="ab_v2_cllab")
    with st.expander("Albúmina y K · opcional"):
        albumin=_el_v2_optional_float("Albúmina (g/L)","ab_v2_albumin","Ej. 30")
        kval=_el_v2_optional_float("K (mmol/L)","ab_v2_k","Ej. 3,2")
    tags=st.multiselect("Contexto",["Hipovolemia","Carga reciente importante de NaCl 0,9%","AKI/ERC","Diarrea/pérdidas GI"],key="ab_v2_context")
    try:
        ag=float(anion_gap_mmol_l(sodium_mmol_l=na,chloride_mmol_l=cl,bicarbonate_mmol_l=hco3,potassium_mmol_l=None))
        agcorr=float(albumin_corrected_anion_gap_mmol_l(anion_gap=ag,albumin_g_l=albumin)) if albumin is not None else None
        ab=interpret_acid_base(ph=ph,pco2_mm_hg=pco2,bicarbonate_mmol_l=hco3)
        dr=delta_ratio(anion_gap=agcorr if agcorr is not None else ag,bicarbonate_mmol_l=hco3)
    except Exception as e:
        st.error(str(e)); return
    high_ag=(agcorr if agcorr is not None else ag) > 12
    clbundle=_electrolyte_bundle_cached("CL"); abbundle=_electrolyte_bundle_cached("AB")
    ctx={"chloride":{"below_lab_range":cl_lab=="Bajo","above_lab_range":cl_lab=="Alto"},"acid_base":{"metabolic_alkalosis":ab.get("primary")=="ALCALOSIS_METABOLICA","metabolic_acidosis":ab.get("primary")=="ACIDOSIS_METABOLICA","high_anion_gap":high_ag},"volume":{"hypovolemic":"Hipovolemia" in tags},"renal":{"impairment":"AKI/ERC" in tags}}
    matched=evaluate_electrolyte_rules(clbundle.get("rules") or [],ctx)+evaluate_electrolyte_rules(abbundle.get("rules") or [],ctx)
    with st.container(border=True):
        st.markdown("### Resultado e interpretación")
        labels={"ACIDOSIS_METABOLICA":"ACIDOSIS METABÓLICA","ALCALOSIS_METABOLICA":"ALCALOSIS METABÓLICA","ACIDOSIS_RESPIRATORIA":"ACIDOSIS RESPIRATORIA","ALCALOSIS_RESPIRATORIA":"ALCALOSIS RESPIRATORIA","SIN_TRASTORNO_MAYOR_EVIDENTE":"SIN TRASTORNO MAYOR EVIDENTE"}
        st.info(f"**{labels.get(ab.get('primary'),ab.get('primary'))}** · {ab.get('detail')}")
        st.write(f"**Anion gap:** {fmt_num(ag,1)} mmol/L" + (f" · **corregido por albúmina:** {fmt_num(agcorr,1)} mmol/L" if agcorr is not None else ""))
        if dr is not None: st.write(f"**Delta ratio:** {fmt_num(dr,2)}")
        comp=ab.get("compensation")
        if comp: st.write(f"**pCO₂ esperada por compensación:** {fmt_num(comp.get('expected'),1)} mmHg (aprox. {fmt_num(comp.get('lower'),1)}–{fmt_num(comp.get('upper'),1)}).")
        for r in matched:
            if r.get("recommendation_text"): st.write(f"• {r.get('recommendation_text')}")
        if "Carga reciente importante de NaCl 0,9%" in tags and ab.get("primary")=="ACIDOSIS_METABOLICA": st.warning("La carga de cloruro es un modificador relevante: revisar si una solución balanceada es apropiada para las siguientes necesidades de fluido.")
        st.caption("MedCalc no prescribe bicarbonato IV de forma universal desde pH/HCO₃ aislados; primero define el trastorno y exige causa, ventilación, función renal y contexto crítico.")
        _el_v2_sources(clbundle,matched)


def _page_joint_v2():
    header("Hidroelectrolitos · Reposición conjunta", "Prioriza alteraciones simultáneas y evita sumar volumen/contraiones sin verlo.")
    age, weight, _=_el_v2_patient("joint_v2")
    st.caption("Deje en blanco lo que no tenga. El panel prioriza; las preparaciones exactas se muestran en la pestaña de cada electrolito.")
    c1,c2,c3,c4=st.columns(4)
    with c1: na=_el_v2_optional_float("Na","joint_na","mmol/L")
    with c2: k=_el_v2_optional_float("K","joint_k","mmol/L")
    with c3: mg=_el_v2_optional_float("Mg","joint_mg","mmol/L")
    with c4: ca=_el_v2_optional_float("Ca total/corregido","joint_ca","mmol/L")
    c5,c6,c7=st.columns(3)
    with c5: pval=_el_v2_optional_float("P","joint_p","mmol/L")
    with c6: cl=_el_v2_optional_float("Cl","joint_cl","mmol/L")
    with c7: hco3=_el_v2_optional_float("HCO₃⁻","joint_hco3","mmol/L")
    tags=st.multiselect("Contexto global",["Paciente crítico","ICC/congestión / restricción de volumen","AKI/ERC","Oliguria/anuria","Hemodiálisis","No puede recibir vía oral","Síntomas neurológicos graves por Na"],key="joint_v2_context")
    with st.expander("Medicamentos actuales · opcional"):
        meds=db.search_medications("",limit=max(COUNTS.get("medications",1000),1000))
        labels=[f"{m['principio_activo']} · {m['med_id']}" for m in meds]
        to_id={f"{m['principio_activo']} · {m['med_id']}":m['med_id'] for m in meds}
        selected=st.multiselect("Medicamentos del paciente",labels,default=[],key="joint_v2_meds")
    selected_ids=tuple(to_id[x] for x in selected) if selected else tuple()
    severity_order={"CRITICAL":0,"HIGH":1,"MODERATE":2,"LOW":3,"INFO":4,None:5}
    findings=[]; combined={"patient":{"age_years":age,"weight_kg":weight},"serum":{},"calcium":{},"clinical":{"critically_ill":"Paciente crítico" in tags},"renal":{"impairment":any(x in tags for x in ["AKI/ERC","Oliguria/anuria","Hemodiálisis"]),"failure":any(x in tags for x in ["Oliguria/anuria","Hemodiálisis"]),"dialysis":"Hemodiálisis" in tags},"access":{"oral_available":"No puede recibir vía oral" not in tags},"volume":{"volume_sensitive":"ICC/congestión / restricción de volumen" in tags},"symptoms":{"severe_neurologic":"Síntomas neurológicos graves por Na" in tags}}
    vals={"NA":na,"K":k,"MG":mg,"CA":ca,"P":pval}
    fieldmap={"NA":"na_mmol_l","K":"k_mmol_l","MG":"mg_mmol_l","P":"p_mmol_l"}
    for code,val in vals.items():
        if val is None: continue
        if code=="CA": combined["calcium"]["interpretive_mmol_l"]=val
        else: combined["serum"][fieldmap[code]]=val
        bundle=_electrolyte_bundle_cached(code)
        local=evaluate_electrolyte_rules(bundle.get("rules") or [],combined)
        cls=_el_v2_classification(local)
        if cls: findings.append((severity_order.get(cls.get("severity"),5),code,cls.get("severity"),cls.get("recommendation_text"),local,bundle))
    findings.sort(key=lambda x:x[0])
    with st.container(border=True):
        st.markdown("### Prioridades")
        if not findings: st.success("No se activaron clasificaciones con los valores aportados.")
        for i,(_,code,sev,text,_,_) in enumerate(findings,1): st.write(f"**{i}. {_analyte_name_es(code)} · {_severity_es(sev)}:** {text}")
        if selected_ids:
            med_hits=[]
            for code,val in vals.items():
                if val is None: continue
                med_hits.extend(_electrolyte_modifier_rows_cached(selected_ids,code))
            if med_hits:
                st.markdown("**Modificadores farmacológicos detectados**")
                seenm=set()
                for m in med_hits:
                    key=(m.get('med_id'),m.get('analyte_id'),m.get('effect_code'))
                    if key in seenm: continue
                    seenm.add(key)
                    st.write(f"• **{m.get('generic_name')}:** {m.get('interpretation_text') or m.get('clinical_implication') or 'puede modificar el electrolito.'}")
            else:
                st.caption(f"Revisión farmacológica activa: {len(selected_ids)} medicamento(s) revisados; sin modificadores publicados para los electrolitos introducidos.")
        if k is not None and mg is not None and k<3.5 and mg<0.71: st.warning("**Dependencia K–Mg:** corregir Mg en paralelo porque la hipomagnesemia puede hacer refractaria la corrección de K.")
        if ca is not None and mg is not None and ca<2.15 and mg<0.71: st.warning("**Dependencia Ca–Mg:** corregir Mg concomitante si la hipocalcemia es persistente/refractaria.")
        if pval is not None and ca is not None and pval<0.6 and ca<2.15: st.warning("**Ca + fosfato:** no mezclar automáticamente en la misma bolsa o línea sin compatibilidad confirmada.")
        # Volumen y contraiones de la PRIMERA unidad IV de cada plan, siempre desde action_json/productos Supabase.
        initial_vol=0.0; plans=[]; loads={"NA":0.0,"K":0.0,"CL":0.0,"MG":0.0,"CA":0.0,"P":0.0}
        for _,code,_,_,local,bun in findings:
            chosen=None
            # Para K, la configuración automática vive en KQLD_MODSEV_STRATEGY.
            if code=="K":
                cfg=next((r for r in local if r.get("rule_code")=="KQLD_MODSEV_STRATEGY"),None)
                if cfg:
                    a=cfg.get("action_json") or {}; target=float(a.get("auto_initial_unit_mmol") or 10)
                    v=float(a.get("auto_central_final_volume_ml") or 100) if (k is not None and k<2.5) or "ICC/congestión / restricción de volumen" in tags else float(a.get("auto_peripheral_final_volume_ml") or 250)
                    initial_vol+=v; plans.append(f"K: {fmt_num(v,0)} mL"); loads["K"]+=target; loads["CL"]+=target
                    # Cl/Na del NaCl 0,9% aproximado, descontando concentrado solo si se conoce; se reporta como estimación.
                    loads["NA"]+=154*v/1000; loads["CL"]+=154*v/1000
                    continue
            for r in local:
                a=r.get("action_json") or {}; vol=a.get("final_volume_ml")
                if vol is not None and str(a.get("route") or "").upper()=="IV": chosen=(r,a,float(vol)); break
            if not chosen: continue
            r,a,v=chosen; initial_vol+=v; plans.append(f"{code}: {fmt_num(v,0)} mL")
            target=float(a.get("target_mmol") or a.get("auto_initial_mmol") or a.get("target_mmol_ca") or 0)
            pcode=a.get("product_code"); prod=_el_v2_product(bun,pcode) if pcode else None
            concentrate_ml=0.0
            if prod and target>0:
                primary=_component_by_code(prod,code)
                c=float((primary or {}).get("mmol_per_ml") or 0)
                if c>0: concentrate_ml=target/c
                for comp in prod.get("components") or []:
                    ac=(comp.get("analyte") or {}).get("code")
                    if ac in loads:
                        if comp.get("mmol_per_ml") is not None: loads[ac]+=float(comp.get("mmol_per_ml"))*concentrate_ml
                        elif comp.get("mmol_per_unit") is not None and a.get("ampoules"): loads[ac]+=float(comp.get("mmol_per_unit"))*float(a.get("ampoules"))
            if "NaCl" in str(a.get("diluent") or "") or "sodium chloride" in str(a.get("diluent") or "").lower():
                diluent_ml=max(0.0,v-concentrate_ml); loads["NA"]+=154*diluent_ml/1000; loads["CL"]+=154*diluent_ml/1000
        if initial_vol:
            st.info(f"**Volumen IV inicial visible de las primeras unidades propuestas:** {fmt_num(initial_vol,0)} mL = {fmt_num(initial_vol/weight,2)} mL/kg ({' + '.join(plans)}). No significa administrarlas simultáneamente: Queensland recomienda tratamiento escalonado y verificar compatibilidades.")
            loadtxt=[f"{ion} {fmt_num(val,1)} mmol" for ion,val in loads.items() if val>0.05]
            if loadtxt: st.caption("**Carga iónica estimada de esas primeras unidades:** "+" · ".join(loadtxt)+". El valor se recalcula cuando se cambia a la preparación individual definitiva.")
        if "ICC/congestión / restricción de volumen" in tags and initial_vol: st.warning("Contexto sensible a volumen: priorizar las alteraciones de mayor riesgo y preparaciones concentradas/centralizadas validadas en vez de sumar automáticamente todas las bolsas periféricas.")
        st.caption("Este panel prioriza y contabiliza. Abra el electrolito correspondiente para obtener la preparación completa paso a paso.")



# -----------------------------------------------------------------------------
# V2.1 · UNIDADES DE LABORATORIO + PANEL INTEGRAL
# -----------------------------------------------------------------------------

def _el_v3_optional_lab_input(analyte_code, label, key, default_unit=None, placeholder=""):
    units = list(supported_laboratory_units(analyte_code))
    if default_unit not in units:
        default_unit = units[0]
    idx = units.index(default_unit)
    c1, c2 = st.columns([1.55, 0.75])
    with c1:
        raw = st.text_input(label, value="", placeholder=placeholder or "Opcional", key=f"{key}_value")
    with c2:
        unit = st.selectbox("Unidad", units, index=idx, key=f"{key}_unit", label_visibility="collapsed")
    value = _fallback_as_float(raw)
    if value is None:
        return None, None, unit
    try:
        mmol = float(laboratory_value_to_mmol_l(analyte_code, value, unit))
    except Exception as e:
        st.error(f"{label}: {e}")
        return None, value, unit
    return mmol, float(value), unit


def _el_v3_lab_display(code, mmol_value, native_value, native_unit, digits=2):
    if mmol_value is None:
        return "—"
    if native_value is None or native_unit == "mmol/L":
        return f"{fmt_num(mmol_value,digits)} mmol/L"
    return f"{fmt_num(native_value,digits)} {native_unit} (≈ {fmt_num(mmol_value,digits)} mmol/L)"


def _el_v3_optional_scalar(label, key, units, default_unit, placeholder=""):
    c1,c2=st.columns([1.55,0.75])
    with c1:
        raw=st.text_input(label,value="",placeholder=placeholder or "Opcional",key=f"{key}_value")
    with c2:
        unit=st.selectbox("Unidad",units,index=units.index(default_unit),key=f"{key}_unit",label_visibility="collapsed")
    val=_fallback_as_float(raw)
    return val,unit


def _el_v3_es_diluent(name):
    raw=str(name or "")
    repl={
        "Sodium Chloride 0.9% Injection USP":"Cloruro de sodio 0,9%",
        "Dextrose 5% Injection USP":"Glucosa 5%",
        "Dextrose 10% Injection USP":"Glucosa 10%",
    }
    for a,b in repl.items(): raw=raw.replace(a,b)
    return raw


def _el_v3_med_selector(prefix):
    with st.expander("Medicamentos actuales · opcional"):
        meds=db.search_medications("",limit=max(COUNTS.get("medications",1000),1000))
        labels=[f"{m['principio_activo']} · {m['med_id']}" for m in meds]
        to_id={f"{m['principio_activo']} · {m['med_id']}":m['med_id'] for m in meds}
        selected=st.multiselect("Medicamentos del paciente",labels,default=[],key=f"{prefix}_meds")
    return tuple(to_id[x] for x in selected) if selected else tuple()


def _el_v3_product_component(bundle, product_code, analyte_code):
    prod=_el_v2_product(bundle,product_code) if product_code else None
    return prod,_component_by_code(prod,analyte_code) if prod else None


def _page_integral_v3():
    header(
        "Hidroelectrolitos · Panel integral",
        "Introduzca en una sola pantalla los electrolitos disponibles. MedCalc normaliza las unidades, prioriza las alteraciones y genera el plan de reposición conjunto sin obligarle a abrir cada electrolito por separado.",
    )
    age,weight,sex=_el_v2_patient("integral_v3",include_sex=True)
    st.caption("Ingrese solo los resultados disponibles. Los valores vacíos NO se interpretan como normales.")

    st.markdown("### Electrolitos")
    r1=st.columns(4)
    with r1[0]: na,na_raw,na_u=_el_v3_optional_lab_input("NA","Sodio","int_na","mEq/L","Ej. 128")
    with r1[1]: k,k_raw,k_u=_el_v3_optional_lab_input("K","Potasio","int_k","mEq/L","Ej. 2,9")
    with r1[2]: mg,mg_raw,mg_u=_el_v3_optional_lab_input("MG","Magnesio","int_mg","mg/dL","Ej. 1,2")
    with r1[3]: ca,ca_raw,ca_u=_el_v3_optional_lab_input("CA","Calcio total","int_ca","mg/dL","Ej. 7,4")
    r2=st.columns(3)
    with r2[0]: pval,p_raw,p_u=_el_v3_optional_lab_input("P","Fósforo","int_p","mg/dL","Ej. 1,5")
    with r2[1]: cl,cl_raw,cl_u=_el_v3_optional_lab_input("CL","Cloro","int_cl","mEq/L","Ej. 112")
    with r2[2]: hco3,hco3_raw,hco3_u=_el_v3_optional_lab_input("HCO3","Bicarbonato / CO₂ total","int_hco3","mEq/L","Ej. 17")

    with st.expander("Datos complementarios que cambian la interpretación · opcional"):
        cc1,cc2,cc3=st.columns(3)
        with cc1:
            albumin_raw,albumin_u=_el_v3_optional_scalar("Albúmina","int_albumin",["g/dL","g/L"],"g/dL","Ej. 2,8")
            albumin=float(albumin_to_g_l(albumin_raw,albumin_u)) if albumin_raw is not None else None
        with cc2:
            glucose_raw,glucose_u=_el_v3_optional_scalar("Glucosa","int_glucose",["mg/dL","mmol/L"],"mg/dL","Ej. 420")
            glucose=float(glucose_to_mmol_l(glucose_raw,glucose_u)) if glucose_raw is not None else None
        with cc3:
            ionca,ionca_raw,ionca_u=_el_v3_optional_lab_input("CA","Calcio ionizado","int_ica","mmol/L","Ej. 0,95")
        aa1,aa2,aa3=st.columns(3)
        with aa1: ph=_el_v2_optional_float("pH","int_ph","Ej. 7,24")
        with aa2: pco2=_el_v2_optional_float("pCO₂ (mmHg)","int_pco2","Ej. 28")
        with aa3:
            urea_raw,urea_unit=_el_v3_optional_scalar("Urea / BUN","int_urea",["mmol/L","urea mg/dL","BUN mg/dL"],"mmol/L","Ej. 8")
            if urea_raw is None: urea=None
            elif urea_unit=="mmol/L": urea=float(urea_raw)
            elif urea_unit=="urea mg/dL": urea=float(urea_raw)/6.006
            else: urea=float(urea_raw)/2.801

    with st.expander("Contexto clínico · solo lo que cambie la conducta",expanded=False):
        c1,c2,c3=st.columns(3)
        with c1:
            volume_status=st.selectbox("Estado de volumen",["Sin dato / general","Hipovolemia","Euvolemia / sospecha SIADH","Hipervolemia / ICC / cirrosis"],key="int_volume")
        with c2:
            renal_status=st.selectbox("Función renal / diuresis",["Sin deterioro relevante conocido","AKI/ERC con diuresis","Oliguria/anuria","Hemodiálisis"],key="int_renal")
        with c3:
            symptom_status=st.selectbox("Síntomas",["Sin síntomas de alarma","Síntomas importantes / arritmia / tetania","Síntomas neurológicos graves / convulsión / coma"],key="int_symptoms")
        flags=st.multiselect("Otros contextos",[
            "Paciente crítico o inestable","Pérdidas digestivas","Redistribución (insulina/beta-agonista/alcalosis)",
            "DKA/HHS","Malnutrición/realimentación/TPN","No puede recibir vía oral","Ya dispone de vía venosa central",
            "Sospecha de pseudohiperpotasemia / muestra hemolizada","Carga reciente importante de NaCl 0,9%",
        ],key="int_flags")

    selected_ids=_el_v3_med_selector("integral_v3")
    values={"NA":na,"K":k,"MG":mg,"CA":ionca if ionca is not None else ca,"P":pval}
    if not any(v is not None for v in values.values()) and cl is None and hco3 is None:
        st.info("Introduzca al menos un electrolito para generar el análisis integral.")
        return

    renal_imp=renal_status!="Sin deterioro relevante conocido"
    renal_failure=renal_status in {"Oliguria/anuria","Hemodiálisis"}
    dialysis=renal_status=="Hemodiálisis"
    oral_available="No puede recibir vía oral" not in flags
    volume_sensitive=volume_status=="Hipervolemia / ICC / cirrosis"
    symptoms_present=symptom_status!="Sin síntomas de alarma"
    severe_neuro=symptom_status=="Síntomas neurológicos graves / convulsión / coma"
    critical="Paciente crítico o inestable" in flags
    central="Ya dispone de vía venosa central" in flags

    # Calcio interpretativo: ionizado si está disponible; de lo contrario total corregido por albúmina cuando sea posible.
    ca_interpret=None
    if ionca is not None:
        ca_interpret=ionca
    elif ca is not None:
        try: ca_interpret=float(corrected_calcium_mmol_l(total_ca_mmol_l=ca,albumin_g_l=albumin)) if albumin is not None else ca
        except Exception: ca_interpret=ca

    modifiers={code:(_electrolyte_modifier_rows_cached(selected_ids,code) if selected_ids else []) for code in ["NA","K","MG","CA","P"]}
    k_mechanisms=sorted({str(m.get("mechanism_class")) for m in modifiers["K"] if m.get("mechanism_class")})
    k_effects=sorted({str(m.get("effect_code")) for m in modifiers["K"] if m.get("effect_code")})

    bundles={code:_electrolyte_bundle_cached(code) for code in ["NA","K","MG","CA","P","CL","AB"]}
    matched={}
    if na is not None:
        ctx={"patient":{"age_years":age,"weight_kg":weight,"sex":sex},"serum":{"na_mmol_l":na},
             "volume":{"hypovolemic":volume_status=="Hipovolemia","euvolemic":volume_status=="Euvolemia / sospecha SIADH","hypervolemic":volume_sensitive},
             "hemodynamic":{"unstable":critical},"symptoms":{"severe_neurologic":severe_neuro,"moderate_neurologic":symptoms_present and not severe_neuro},
             "renal":{"impairment":renal_imp,"dialysis":dialysis},"glucose":{"hyperglycemia":glucose is not None}}
        matched["NA"]=evaluate_electrolyte_rules(bundles["NA"].get("rules") or [],ctx)
    if k is not None:
        ctx={"patient":{"age_years":age,"weight_kg":weight},"serum":{"k_mmol_l":k},"magnesium":{"value_mmol_l":mg,"low":False},
             "renal":{"impairment":renal_imp,"failure":renal_failure,"dialysis":dialysis,"hemodialysis":dialysis,"oliguria":renal_status=="Oliguria/anuria","anuria":renal_status=="Oliguria/anuria"},
             "clinical":{"heart_failure_or_congestion":volume_sensitive,"gi_diuretic_losses":"Pérdidas digestivas" in flags,"redistribution_context":"Redistribución (insulina/beta-agonista/alcalosis)" in flags,"dka":"DKA/HHS" in flags,"refractory_to_k":False,"unwell":critical,"rapid_k_rise_expected":critical,"suspected_hypokalemic_arrest":False,"suspected_hyperkalemic_arrest":False},
             "symptoms":{"present":symptoms_present,"muscle_paralysis":False},"ecg":{"hypokalemia_changes":symptoms_present and k<3.5,"hyperkalemia_changes":symptoms_present and k>=5.2},
             "sample":{"pseudohyperkalemia_suspected":"Sospecha de pseudohiperpotasemia / muestra hemolizada" in flags},"acid_base":{"metabolic_acidosis":bool(hco3 is not None and hco3<22)},
             "glucose":{"pretreatment_mmol_l":glucose},"treatment":{"insulin_glucose_given":False},"access":{"oral_available":oral_available,"iv_used":k<3.1 or k>=6,"peripheral_available":True,"central_available":central},
             "infusion":{"pump_available":True,"uses_burette":True,"large_vein":central},"medications":{"mechanism_classes":k_mechanisms,"effect_codes":k_effects},"resuscitation":{"cardiac_arrest":False,"peri_or_cardiac_arrest":False}}
        matched["K"]=evaluate_electrolyte_rules(bundles["K"].get("rules") or [],ctx)
    if mg is not None:
        ctx={"patient":{"age_years":age,"weight_kg":weight},"serum":{"mg_mmol_l":mg},"symptoms":{"present":symptoms_present},"renal":{"impairment":renal_imp,"failure":renal_failure,"dialysis":dialysis},"access":{"oral_available":oral_available},"volume":{"volume_sensitive":volume_sensitive}}
        matched["MG"]=evaluate_electrolyte_rules(bundles["MG"].get("rules") or [],ctx)
    if ca_interpret is not None:
        ctx={"patient":{"age_years":age,"weight_kg":weight},"calcium":{"interpretive_mmol_l":ca_interpret,"ionized_mmol_l":ionca},"serum":{"mg_mmol_l":mg},"symptoms":{"present":symptoms_present},"renal":{"impairment":renal_imp,"failure":renal_failure,"dialysis":dialysis},"volume":{"volume_sensitive":volume_sensitive},"access":{"oral_available":oral_available}}
        matched["CA"]=evaluate_electrolyte_rules(bundles["CA"].get("rules") or [],ctx)
    if pval is not None:
        ctx={"patient":{"age_years":age,"weight_kg":weight},"serum":{"p_mmol_l":pval},"phosphate":{"high_risk_context":any(x in flags for x in ["Malnutrición/realimentación/TPN","DKA/HHS"]),"above_lab_range":False},"clinical":{"critically_ill":critical},"renal":{"impairment":renal_imp,"failure":renal_failure,"dialysis":dialysis},"access":{"oral_available":oral_available}}
        matched["P"]=evaluate_electrolyte_rules(bundles["P"].get("rules") or [],ctx)

    ab_result=None; ag=None; agcorr=None; dr=None; abmatched=[]
    if na is not None and cl is not None and hco3 is not None:
        try:
            ag=float(anion_gap_mmol_l(sodium_mmol_l=na,chloride_mmol_l=cl,bicarbonate_mmol_l=hco3,potassium_mmol_l=None))
            agcorr=float(albumin_corrected_anion_gap_mmol_l(anion_gap=ag,albumin_g_l=albumin)) if albumin is not None else None
            if ph is not None and pco2 is not None:
                ab_result=interpret_acid_base(ph=ph,pco2_mm_hg=pco2,bicarbonate_mmol_l=hco3)
                dr=delta_ratio(anion_gap=agcorr if agcorr is not None else ag,bicarbonate_mmol_l=hco3)
            ctx={"chloride":{"below_lab_range":False,"above_lab_range":False},"acid_base":{"metabolic_alkalosis":bool(ab_result and ab_result.get("primary")=="ALCALOSIS_METABOLICA"),"metabolic_acidosis":bool(ab_result and ab_result.get("primary")=="ACIDOSIS_METABOLICA"),"high_anion_gap":(agcorr if agcorr is not None else ag)>12},"volume":{"hypovolemic":volume_status=="Hipovolemia"},"renal":{"impairment":renal_imp}}
            abmatched=evaluate_electrolyte_rules(bundles["CL"].get("rules") or [],ctx)+evaluate_electrolyte_rules(bundles["AB"].get("rules") or [],ctx)
        except Exception: pass

    severity_order={"CRITICAL":0,"HIGH":1,"MODERATE":2,"LOW":3,"INFO":4,None:5}
    findings=[]
    for code,rules in matched.items():
        cls=_el_v2_classification(rules)
        if cls: findings.append((severity_order.get(cls.get("severity"),5),code,cls))
    findings.sort(key=lambda x:x[0])

    native_map={"NA":(na,na_raw,na_u,0),"K":(k,k_raw,k_u,1),"MG":(mg,mg_raw,mg_u,2),"CA":(ionca if ionca is not None else ca,ionca_raw if ionca is not None else ca_raw,ionca_u if ionca is not None else ca_u,2),"P":(pval,p_raw,p_u,2)}
    with st.container(border=True):
        st.markdown("### Análisis integral y plan de corrección")
        if findings:
            st.markdown("**Prioridades detectadas**")
            for i,(_,code,cls) in enumerate(findings,1):
                vv,rr,uu,dg=native_map[code]
                name=_analyte_name_es(code)
                st.write(f"**{i}. {name} · {_severity_es(cls.get('severity'))}:** {_el_v3_lab_display(code,vv,rr,uu,dg)} — {cls.get('recommendation_text') or ''}")
        else:
            st.success("Con los valores introducidos no se activaron clasificaciones hipo/hiper publicadas para Na/K/Mg/Ca/P.")

        if na is not None and glucose is not None:
            try:
                na16=corrected_sodium_for_hyperglycemia(serum_na=na,glucose_mmol_l=glucose,correction_mmol_per_100mg_dl=1.6)
                na24=corrected_sodium_for_hyperglycemia(serum_na=na,glucose_mmol_l=glucose,correction_mmol_per_100mg_dl=2.4)
                st.info(f"**Na corregido por hiperglucemia:** {fmt_num(na16,1)}–{fmt_num(na24,1)} mmol/L según coeficiente 1,6–2,4. Tonicidad ≈ {fmt_num(effective_osmolality_mosm_kg(sodium_mmol_l=na,glucose_mmol_l=glucose),0)} mOsm/kg.")
            except Exception: pass
        if ca is not None and albumin is not None and ionca is None and ca_interpret is not None:
            st.info(f"**Calcio total corregido por albúmina:** {fmt_num(ca_interpret,2)} mmol/L (≈ {fmt_num(float(mmol_l_to_laboratory_value('CA',ca_interpret,'mg/dL')),2)} mg/dL). Si hay calcio ionizado, este tiene prioridad interpretativa.")
        if ag is not None:
            st.write(f"**Anion gap:** {fmt_num(ag,1)} mmol/L"+(f" · corregido por albúmina {fmt_num(agcorr,1)} mmol/L" if agcorr is not None else ""))
            if ab_result:
                labels={"ACIDOSIS_METABOLICA":"ACIDOSIS METABÓLICA","ALCALOSIS_METABOLICA":"ALCALOSIS METABÓLICA","ACIDOSIS_RESPIRATORIA":"ACIDOSIS RESPIRATORIA","ALCALOSIS_RESPIRATORIA":"ALCALOSIS RESPIRATORIA","SIN_TRASTORNO_MAYOR_EVIDENTE":"SIN TRASTORNO MAYOR EVIDENTE"}
                st.write(f"**Ácido-base:** {labels.get(ab_result.get('primary'),ab_result.get('primary'))} · {ab_result.get('detail')}")
                if dr is not None: st.caption(f"Delta ratio: {fmt_num(dr,2)}")

        if k is not None and mg is not None and k<3.5 and mg<0.71: st.warning("**K–Mg:** corregir Mg en paralelo; la hipomagnesemia puede hacer refractaria la corrección de K.")
        if ca_interpret is not None and mg is not None and ca_interpret<2.15 and mg<0.71: st.warning("**Ca–Mg:** corregir Mg concomitante si la hipocalcemia persiste o es refractaria.")
        if pval is not None and ca_interpret is not None and pval<0.6 and ca_interpret<2.15: st.warning("**Ca + fosfato:** no mezclar automáticamente en la misma bolsa/línea sin compatibilidad confirmada.")

        # Farmacología una sola vez, integrada por electrolito.
        if selected_ids:
            hits=[]
            for code,rows in modifiers.items():
                for m in rows: hits.append((code,m))
            if hits:
                with st.expander("Medicamentos que pueden contribuir",expanded=True):
                    seen=set()
                    for code,m in hits:
                        key=(code,m.get("med_id"),m.get("effect_code"))
                        if key in seen: continue
                        seen.add(key)
                        st.write(f"• **{m.get('generic_name')} · {code}:** {m.get('interpretation_text') or m.get('clinical_implication') or 'puede modificar este electrolito.'}")
            else: st.caption(f"Revisión farmacológica activa: {len(selected_ids)} medicamento(s) revisados; sin modificadores publicados para los electrolitos introducidos.")

        st.markdown("---")
        st.markdown("### Plan integral único")
        st.caption(
            "MedCalc integra las alteraciones en una sola secuencia terapéutica. "
            "Una sola solución clínica no significa mezclar todos los electrolitos en la misma bolsa: "
            "cuando la compatibilidad no está validada, el plan los ordena de forma secuencial."
        )

        plan_steps=[]
        plan_notes=[]
        total_iv=0.0
        loads={"NA":0.0,"K":0.0,"CL":0.0,"MG":0.0,"CA":0.0,"P":0.0}

        def _add_step(priority, title, instruction, *, route=None, duration_h=None, volume_ml=0.0,
                      monitoring=None, rationale=None, ionic_load=None, kind="TREATMENT"):
            nonlocal total_iv
            step={
                "priority":priority,"title":title,"instruction":instruction,"route":route,
                "duration_h":duration_h,"volume_ml":float(volume_ml or 0),"monitoring":monitoring,
                "rationale":rationale,"kind":kind,
            }
            plan_steps.append(step)
            if route=="IV" and float(volume_ml or 0)>0:
                total_iv += float(volume_ml)
            for ion,val in (ionic_load or {}).items():
                if ion in loads and val is not None:
                    loads[ion]+=float(val)

        # --------------------------------------------------------------
        # SODIO: solo se incorpora al tratamiento cuando existe una acción
        # concreta. Los límites de corrección pasan a SEGURIDAD, no se
        # presentan como si fueran una reposición.
        # --------------------------------------------------------------
        if "NA" in matched:
            rules=matched["NA"]
            shock=_el_v2_rule_by_strategy(rules,"RESTORE_CIRCULATION_FIRST")
            water=_el_v2_rule_by_strategy(rules,"FREE_WATER")
            hyper=_el_v2_rule_by_strategy(rules,"HYPERTONIC_3_PERCENT_BOLUS")
            isotonic=_el_v2_rule_by_strategy(rules,"ISOTONIC_SALINE")
            fluid_restrict=_el_v2_rule_by_strategy(rules,"FLUID_RESTRICTION_CAUSE_SPECIFIC")

            if hyper:
                a=hyper.get("action_json") or {}
                bol=float(a.get("bolus_ml") or 150); mins=float(a.get("duration_min") or 20)
                _add_step(
                    5,"Hiponatremia sintomática: corregir primero el riesgo neurológico",
                    f"Administrar NaCl 3% **{fmt_num(bol,0)} mL IV en {fmt_num(mins,0)} min**. "
                    "Medir Na y reevaluar la clínica inmediatamente al terminar; decidir cualquier repetición solo con ese nuevo control.",
                    route="IV",duration_h=mins/60,volume_ml=bol,
                    monitoring="Na y estado neurológico al terminar el bolo.",
                    rationale="Los síntomas neurológicos graves/moderados activaron la regla de solución hipertónica.",
                    ionic_load={"NA":0.513*bol,"CL":0.513*bol},
                )
            elif shock:
                _add_step(
                    6,"Sodio + inestabilidad: restaurar perfusión antes del agua libre",
                    "Administrar cristaloide isotónico para restaurar la circulación. **MedCalc no fija un volumen universal** porque la regla exige titularlo a perfusión, presión arterial, congestión y respuesta clínica; después se recalcula el componente de agua libre.",
                    route="IV",monitoring="Reevaluar perfusión/hemodinamia antes de calcular agua libre.",kind="REQUIRES_TITRATION",
                )
            elif water:
                a=water.get("action_json") or {}
                target=float(a.get("target_na") or 140); maxdrop=float(a.get("max_drop_24h_mmol_l") or 10)
                try:
                    full=float(free_water_deficit_l(weight_kg=weight,serum_na=na,target_na=target,sex=sex))
                    target24=max(target,float(na)-maxdrop)
                    day=max(0.0,float(free_water_deficit_l(weight_kg=weight,serum_na=na,target_na=target24,sex=sex)))
                    if oral_available:
                        _add_step(
                            30,"Hipernatremia: reponer agua libre",
                            f"Objetivo de las primeras 24 h: **{fmt_num(day,2)} L de agua libre** (≈ {fmt_num(day*1000/24,0)} mL/h si se distribuye uniformemente), preferentemente por vía oral/enteral. "
                            f"El déficit estimado hasta Na {fmt_num(target,0)} es {fmt_num(full,2)} L; no se administra completo de una vez.",
                            route="PO",monitoring="Controlar Na aproximadamente cada 4 h y recalcular el volumen restante.",
                            rationale=f"El plan limita el descenso a ≤{fmt_num(maxdrop,0)} mmol/L en 24 h.",
                        )
                    else:
                        _add_step(
                            30,"Hipernatremia: reponer agua libre por vía IV",
                            f"Administrar **glucosa 5% {fmt_num(day*1000,0)} mL en 24 h** como punto de partida calculado (≈ **{fmt_num(day*1000/24,0)} mL/h**), antes de sumar pérdidas en curso. "
                            f"Déficit estimado total hasta Na {fmt_num(target,0)}: {fmt_num(full,2)} L.",
                            route="IV",duration_h=24,volume_ml=day*1000,
                            monitoring="Controlar Na aproximadamente cada 4 h; ajustar la velocidad con cada control y con el balance.",
                            rationale=f"El cálculo limita el descenso a ≤{fmt_num(maxdrop,0)} mmol/L en 24 h.",
                        )
                except Exception as e:
                    plan_notes.append(f"Sodio: no fue posible calcular agua libre ({e}).")
            elif isotonic:
                _add_step(
                    35,"Hiponatremia hipovolémica: corregir volumen, no un 'déficit de sodio'",
                    "Usar **NaCl 0,9%** para restaurar volumen intravascular. El volumen debe titularse a la respuesta hemodinámica y a comorbilidades; no existe una dosis fija segura derivable solo del Na plasmático.",
                    route="IV",kind="REQUIRES_TITRATION",
                    monitoring="Reevaluar estado de volumen y Na durante la reposición; evitar sobrecorrección.",
                )
            elif fluid_restrict:
                _add_step(
                    40,"Hiponatremia euvolémica/hipervolémica: no reponer sodio de rutina",
                    "**No administrar NaCl 3% ni NaCl 0,9% únicamente para subir el Na** si no existe otra indicación. Tratar la causa (p. ej. SIADH/ICC/cirrosis/fármacos) y aplicar restricción hídrica individualizada según el contexto clínico.",
                    route="NO_IV",monitoring="Control seriado de Na; el límite de corrección se muestra al final del plan.",
                )

            # Los límites y alertas van a una sola sección de seguridad.
            for r in rules:
                if str(r.get("rule_type") or "").upper() in {"ALERT","MONITORING"} and r.get("recommendation_text"):
                    txt=str(r.get("recommendation_text")).strip()
                    if txt and txt not in plan_notes: plan_notes.append(txt)

        # --------------------------------------------------------------
        # POTASIO. En hiperK se genera una secuencia urgente; en hipoK se
        # elige automáticamente VO o IV según la regla activa.
        # --------------------------------------------------------------
        if "K" in matched:
            rules=matched["K"]
            hard=[r for r in rules if r.get("hard_stop") or ((r.get("action_json") or {}).get("hard_stop"))]
            if hard:
                for r in hard:
                    if r.get("recommendation_text"):
                        plan_notes.append("Potasio — BLOQUEO: "+r.get("recommendation_text"))

            if k is not None and k>=6.0:
                membrane=sorted([r for r in rules if str(r.get("rule_type") or "").upper()=="MEMBRANE_STABILIZATION"],key=lambda r:r.get("priority") or 99)
                shift=sorted([r for r in rules if str(r.get("rule_type") or "").upper()=="SHIFT"],key=lambda r:r.get("priority") or 99)
                elimination=sorted([r for r in rules if str(r.get("rule_type") or "").upper()=="ELIMINATION"],key=lambda r:r.get("priority") or 99)
                for r in membrane:
                    if r.get("recommendation_text"):
                        _add_step(1,"Hiperpotasemia: estabilizar membrana",r.get("recommendation_text"),route="IV",kind="URGENT")
                # Insulina/glucosa es la medida principal de redistribución; salbutamol se añade como coadyuvante.
                ig=next((r for r in shift if (r.get("action_json") or {}).get("insulin_soluble_units") is not None),None)
                salb=next((r for r in shift if str((r.get("action_json") or {}).get("drug") or "").lower()=="salbutamol"),None)
                bicarb=next((r for r in shift if "bicarbon" in str((r.get("action_json") or {}).get("drug") or "").lower()),None)
                if ig:
                    _add_step(2,"Hiperpotasemia: desplazar K al intracelular",ig.get("recommendation_text"),route="IV",kind="URGENT",monitoring="Monitorizar glucemia y K según protocolo de hiperpotasemia.")
                if salb:
                    _add_step(3,"Hiperpotasemia: coadyuvante",salb.get("recommendation_text"),route="NEB",kind="URGENT")
                if bicarb:
                    _add_step(4,"Hiperpotasemia + acidosis metabólica",bicarb.get("recommendation_text"),route="IV",kind="URGENT")
                hd=next((r for r in elimination if (r.get("action_json") or {}).get("urgent_dialysis")),None)
                szc=next((r for r in elimination if "zircon" in str((r.get("action_json") or {}).get("drug") or "").lower()),None)
                if hd:
                    _add_step(4,"Hiperpotasemia en hemodiálisis",hd.get("recommendation_text"),kind="URGENT")
                elif szc:
                    _add_step(15,"Hiperpotasemia: eliminación de K",szc.get("recommendation_text"),route="PO")
            elif k is not None and k<3.5 and not hard:
                oral=next((r for r in rules if r.get("rule_type")=="REPLACEMENT" and (r.get("action_json") or {}).get("route")=="PO"),None)
                modsev=next((r for r in rules if r.get("rule_type")=="REPLACEMENT" and (r.get("action_json") or {}).get("route_strategy")),None)
                if modsev and k<3.1:
                    a=modsev.get("action_json") or {}
                    target=float(a.get("auto_initial_unit_mmol") or 10); rate=float(a.get("auto_default_rate_mmol_h") or 10)
                    line="CENTRAL" if volume_sensitive or central or k<2.5 or symptoms_present else "PERIPHERAL"
                    vol=float(a.get("auto_central_final_volume_ml") or 100) if line=="CENTRAL" else float(a.get("auto_peripheral_final_volume_ml") or 250)
                    dur=target/rate
                    prods=[p for p in bundles["K"].get("products") or [] if str(p.get("route") or "").upper()=="IV" and str(p.get("preparation_type") or "").upper()!="PREMIXED_READY_TO_USE"]
                    prods.sort(key=lambda p:(0 if str(p.get("market") or "").upper()=="CL" else 1,0 if "10%" in str(p.get("concentration_label") or "") else 1))
                    prod=next((p for p in prods if (_component_by_code(p,"K") or {}).get("mmol_per_ml")),None)
                    if prod:
                        c=float((_component_by_code(prod,"K") or {}).get("mmol_per_ml")); ml=target/c
                        prio=18 if mg is not None and mg<0.71 else 16
                        _add_step(
                            prio,"Hipopotasemia: reposición IV de K",
                            f"Preparar **{fmt_num(target,0)} mmol de K**: extraer **{fmt_num(ml,2)} mL** de {prod.get('generic_product_name')}, "
                            f"completar a **{fmt_num(vol,0)} mL con NaCl 0,9%** y administrar por vía **{'central' if line=='CENTRAL' else 'periférica'}** "
                            f"a **{fmt_num(vol/dur,0)} mL/h durante {fmt_num(dur,1)} h** (= {fmt_num(rate,1)} mmol/h).",
                            route="IV",duration_h=dur,volume_ml=vol,
                            monitoring="Reevaluar K antes de programar la siguiente unidad; la pauta de 24 h no se convierte en una sola bolsa.",
                            rationale=("Mg bajo: la reposición de Mg se prioriza antes o en paralelo por línea separada." if mg is not None and mg<0.71 else None),
                            ionic_load={"K":target,"CL":target+154*max(vol-ml,0)/1000,"NA":154*max(vol-ml,0)/1000},
                        )
                elif oral and oral_available:
                    a=oral.get("action_json") or {}; opt=a.get("auto_regimen") or next((o for o in (a.get("options") or []) if o.get("dose_mmol") is not None),None)
                    prods=[p for p in bundles["K"].get("products") or [] if str(p.get("route") or "").upper()=="PO"]
                    prods.sort(key=lambda p:(0 if str(p.get("market") or "").upper()=="CL" else 1,p.get("generic_product_name") or ""))
                    if opt and prods:
                        comp=_component_by_code(prods[0],"K"); per=float((comp or {}).get("mmol_per_unit") or 0)
                        if per>0:
                            target=float(opt.get("dose_mmol")); units=product_units_for_mmol(target,per)
                            freq=float(opt.get("frequency_per_day") or (24/float(opt.get("interval_hours"))) if opt.get("interval_hours") else 0)
                            interval=float(opt.get("interval_hours") or (24/freq if freq else 0)); daily=target*freq if freq else None
                            _add_step(
                                35,"Hipopotasemia: reposición oral de K",
                                f"Administrar **{fmt_num(units,0)} comprimidos cada {fmt_num(interval,0)} h** de {prods[0].get('generic_product_name')} "
                                f"= **{fmt_num(target,0)} mmol por dosis**"+(f" (**{fmt_num(daily,0)} mmol/día**)." if daily else "."),
                                route="PO",monitoring="Recontrol de K según la regla y el contexto clínico.",
                            )

        # --------------------------------------------------------------
        # MAGNESIO. Si K está bajo, Mg se adelanta para evitar corrección
        # refractaria. No se mezcla automáticamente con K en la misma bolsa.
        # --------------------------------------------------------------
        if "MG" in matched:
            rules=matched["MG"]
            oral=next((r for r in rules if (r.get("action_json") or {}).get("route")=="PO"),None)
            iv=next((r for r in rules if (r.get("action_json") or {}).get("route")=="IV" and r.get("rule_type")=="REPLACEMENT"),None)
            if iv:
                a=iv.get("action_json") or {}; target=float(a.get("auto_initial_mmol") or a.get("target_mmol_min") or 10)
                prod,comp=_el_v3_product_component(bundles["MG"],a.get("product_code"),"MG")
                c=float((comp or {}).get("mmol_per_ml") or 2); ml=target/c; vol=float(a.get("final_volume_ml") or 100); dur=float(a.get("duration_h") or 1)
                _add_step(
                    15 if k is not None and k<3.5 else 22,"Hipomagnesemia: reposición IV de Mg",
                    f"Preparar **{fmt_num(target,0)} mmol de Mg**: extraer **{fmt_num(ml,2)} mL** de sulfato de magnesio, completar a **{fmt_num(vol,0)} mL con NaCl 0,9%** "
                    f"y administrar en **{fmt_num(dur,1)} h a {fmt_num(vol/dur,0)} mL/h** (= {fmt_num(target/dur,1)} mmol/h; {fmt_num(target*2/dur,1)} mEq/h).",
                    route="IV",duration_h=dur,volume_ml=vol,
                    monitoring=f"Reevaluar Mg/síntomas en {a.get('lab_repeat_hours_min')}–{a.get('lab_repeat_hours_max')} h antes de repetir.",
                    rationale=("Se prioriza por hipopotasemia concomitante." if k is not None and k<3.5 else None),
                    ionic_load={"MG":target,"NA":154*max(vol-ml,0)/1000,"CL":154*max(vol-ml,0)/1000},
                )
            elif oral and oral_available:
                a=oral.get("action_json") or {}; prod,comp=_el_v3_product_component(bundles["MG"],a.get("product_code"),"MG"); per=float((comp or {}).get("mmol_per_unit") or 1.54)
                umin=float(a.get("units_min") or 1); umax=float(a.get("units_max") or 2); interval=float(a.get("interval_hours") or 12)
                _add_step(
                    28 if k is not None and k<3.5 else 38,"Hipomagnesemia: reposición oral de Mg",
                    f"Administrar **{fmt_num(umin,0)}–{fmt_num(umax,0)} comprimidos cada {fmt_num(interval,0)} h** "
                    f"= **{fmt_num(per*umin,2)}–{fmt_num(per*umax,2)} mmol de Mg por dosis**.",
                    route="PO",monitoring="Ajustar continuidad según Mg, síntomas, tolerancia y función renal.",
                    rationale=("Corregir junto con K porque Mg bajo puede volver refractaria la reposición de K." if k is not None and k<3.5 else None),
                )

        # --------------------------------------------------------------
        # CALCIO
        # --------------------------------------------------------------
        if "CA" in matched:
            rules=matched["CA"]
            oral=next((r for r in rules if (r.get("action_json") or {}).get("route")=="PO"),None)
            bol=next((r for r in rules if r.get("rule_code")=="CA_QLD_IV_BOLUS"),None)
            if bol:
                a=bol.get("action_json") or {}; target=float(a.get("target_mmol") or 4.4); amp=int(a.get("ampoules") or 2); vol=float(a.get("final_volume_ml") or 100); mins=float(a.get("duration_min") or 20); ml=amp*10
                _add_step(
                    8 if symptoms_present else 20,"Hipocalcemia: reposición IV de calcio",
                    f"Preparar **{amp} ampollas de gluconato de calcio 10%** (= {fmt_num(target,1)} mmol Ca): retirar **{fmt_num(ml,0)} mL** de una bolsa de NaCl 0,9% de {fmt_num(vol,0)} mL, "
                    f"añadir el calcio y administrar volumen final **{fmt_num(vol,0)} mL en {fmt_num(mins,0)} min** (≈ {fmt_num(vol/(mins/60),0)} mL/h).",
                    route="IV",duration_h=mins/60,volume_ml=vol,
                    monitoring="Reevaluar síntomas y calcio; si Mg está bajo, corregirlo también.",
                    ionic_load={"CA":target,"NA":154*max(vol-ml,0)/1000,"CL":154*max(vol-ml,0)/1000},
                )
            elif oral:
                a=oral.get("action_json") or {}
                _add_step(
                    42,"Hipocalcemia: reposición oral de calcio",
                    f"Administrar **calcio 600 mg: {a.get('units_min')}–{a.get('units_max')} comprimidos al día con alimentos**, según la regla activa.",
                    route="PO",monitoring="Controlar calcio y Mg; preferir calcio ionizado si está disponible.",
                )

        # --------------------------------------------------------------
        # FÓSFORO
        # --------------------------------------------------------------
        if "P" in matched:
            rules=matched["P"]
            oral=next((r for r in rules if (r.get("action_json") or {}).get("route")=="PO"),None)
            critical_rule=next((r for r in rules if r.get("rule_code")=="P_QLD_CRITICAL"),None)
            iv=critical_rule or next((r for r in rules if r.get("rule_code")=="P_QLD_IV"),None)
            if iv:
                a=iv.get("action_json") or {}; target=float(a.get("target_mmol") or 10); vol=float(a.get("final_volume_ml") or 250); dur=float(a.get("duration_h") or a.get("default_duration_h") or 4)
                prod,pc=_el_v3_product_component(bundles["P"],a.get("product_code"),"P"); nac=_component_by_code(prod,"NA") if prod else None; c=float((pc or {}).get("mmol_per_ml") or 1); ml=target/c
                _add_step(
                    24 if critical_rule else 36,"Hipofosfatemia: reposición IV de fósforo",
                    f"Preparar **{fmt_num(target,0)} mmol de P**: extraer **{fmt_num(ml,2)} mL** de {prod.get('generic_product_name') if prod else 'fosfato'}, "
                    f"completar a **{fmt_num(vol,0)} mL con NaCl 0,9%** y administrar en **{fmt_num(dur,1)} h a {fmt_num(vol/dur,0)} mL/h** (= {fmt_num(target/dur,1)} mmol P/h).",
                    route="IV",duration_h=dur,volume_ml=vol,
                    monitoring=("Controlar P y Ca en 1–2 h." if critical_rule else "Controlar P, Ca y función renal en 12–24 h."),
                    rationale=("No administrar en la misma bolsa/línea que calcio sin compatibilidad confirmada." if ca_interpret is not None and ca_interpret<2.15 else None),
                    ionic_load={"P":target,"NA":float((nac or {}).get("mmol_per_ml") or 0)*ml+154*max(vol-ml,0)/1000,"CL":154*max(vol-ml,0)/1000},
                )
            elif oral:
                a=oral.get("action_json") or {}; prod,pc=_el_v3_product_component(bundles["P"],a.get("product_code"),"P"); per=float((pc or {}).get("mmol_per_unit") or 16.1)
                _add_step(
                    45,"Hipofosfatemia: reposición oral de fósforo",
                    f"Administrar **{a.get('units_min')}–{a.get('units_max')} comprimidos efervescentes por dosis** "
                    f"= **{fmt_num(per*float(a.get('units_min') or 1),1)}–{fmt_num(per*float(a.get('units_max') or 2),1)} mmol P por dosis**, "
                    f"hasta {a.get('frequency_max_per_day')} veces/día según respuesta y tolerancia.",
                    route="PO",monitoring="Controlar P, Ca y función renal según la regla activa.",
                )

        # Ácido-base no genera una terapia genérica. Solo se incluye si una
        # regla concreta ya produjo una intervención (p. ej., bicarbonato en
        # hiperK + acidosis). El resto queda en interpretación, no como orden.
        if abmatched:
            specific=[r for r in abmatched if (r.get("action_json") or {}).get("drug") or (r.get("action_json") or {}).get("product")]
            for r in specific:
                if r.get("recommendation_text"):
                    _add_step(26,"Ácido-base: intervención específica",r.get("recommendation_text"),kind="CONTEXTUAL")

        # --------------------------------------------------------------
        # ORDENACIÓN Y SECUENCIACIÓN. Vía oral puede iniciarse a tiempo cero.
        # Las infusiones IV se muestran secuenciales por defecto para no
        # asumir líneas múltiples ni compatibilidad no documentada.
        # --------------------------------------------------------------
        plan_steps.sort(key=lambda x:(x["priority"],x["title"]))
        iv_clock_h=0.0
        if plan_steps:
            st.markdown("**Conducta integrada propuesta**")
            number=0
            for step in plan_steps:
                number+=1
                prefix=""
                if step.get("route")=="IV" and step.get("duration_h"):
                    start_h=iv_clock_h; end_h=iv_clock_h+float(step["duration_h"])
                    if end_h<=1:
                        prefix=f"**{fmt_num(start_h*60,0)}–{fmt_num(end_h*60,0)} min:** "
                    else:
                        prefix=f"**{fmt_num(start_h,1)}–{fmt_num(end_h,1)} h:** "
                    iv_clock_h=end_h
                elif step.get("route") in {"PO","NEB"}:
                    prefix="**Desde ahora:** "
                elif step.get("kind")=="REQUIRES_TITRATION":
                    prefix="**Primero:** "
                st.markdown(f"**{number}. {step['title']}**")
                st.write(prefix+step["instruction"])
                if step.get("rationale"): st.caption("Motivo: "+step["rationale"])
                if step.get("monitoring"): st.caption("Control: "+step["monitoring"])
        else:
            st.success("**No se generó una reposición farmacológica/IV automática** con las alteraciones y el contexto introducidos. El panel mantiene la interpretación y monitorización, pero no inventa una terapia cuando la regla no define una.")

        if total_iv>0:
            st.markdown("**Carga total del plan IV mostrado**")
            st.write(
                f"Volumen IV acumulado si se completan todas las unidades propuestas: **{fmt_num(total_iv,0)} mL** "
                f"= **{fmt_num(total_iv/weight,2)} mL/kg**. "
                "El cronograma anterior es secuencial por defecto; no presupone compatibilidad ni varias líneas venosas."
            )
            lt=[f"{ion} {fmt_num(v,1)} mmol" for ion,v in loads.items() if v>0.05]
            if lt: st.caption("Carga iónica aproximada aportada por las preparaciones mostradas: "+" · ".join(lt)+".")
            if volume_sensitive:
                st.warning("**Paciente sensible a volumen:** el volumen total debe entrar al balance. Si la carga resulta excesiva, no se administran automáticamente todas las unidades: se prioriza la alteración de mayor riesgo y se reevalúa antes de continuar.")

        if plan_notes:
            with st.expander("Límites de seguridad y controles",expanded=True):
                seen_notes=set()
                for note in plan_notes:
                    if note in seen_notes: continue
                    seen_notes.add(note)
                    st.write("• "+note)

        if pval is not None and ca_interpret is not None and pval<0.6 and ca_interpret<2.15:
            st.warning("**Compatibilidad:** calcio y fosfato quedan programados secuencialmente; no mezclar en la misma bolsa/línea sin compatibilidad confirmada.")
        if k is not None and mg is not None and k<3.5 and mg<0.71:
            st.info("**Dependencia K–Mg integrada:** Mg queda antes o en paralelo por una línea distinta; no se deja la corrección de Mg como una observación separada.")

        # Una sola sección de fuentes para todo el panel.
        srcs=[]; seen=set()
        for code,rules in matched.items():
            for r in rules:
                src=_electrolyte_rule_source(r); key=src.get("url") or src.get("title")
                if key and key not in seen: seen.add(key); srcs.append(src)
        for r in abmatched:
            src=_electrolyte_rule_source(r); key=src.get("url") or src.get("title")
            if key and key not in seen: seen.add(key); srcs.append(src)
        if srcs:
            with st.expander("Fuentes del plan integral"):
                for i,src in enumerate(srcs):
                    st.write(f"**{src.get('organization') or 'Fuente'}** · {src.get('title') or '—'}")
                    if src.get("url"): st.link_button("Abrir fuente",src.get("url"),key=f"intsrc_{i}_{abs(hash(str(src.get('url'))))}")

    st.caption("Las unidades de entrada son flexibles; el motor las normaliza a mmol/L para aplicar las reglas de Supabase. mmol/L y mEq/L son numéricamente equivalentes solo para iones monovalentes como Na⁺, K⁺, Cl⁻ y HCO₃⁻; no se generaliza esa equivalencia a Mg²⁺ o Ca²⁺.")



def _page_abg_v813():
    header(
        "Hidroelectrolitos · Gases arteriales",
        "Interpretación ácido-base completa, trastornos mixtos, gap/delta gap, oxigenación y análisis avanzado opcional.",
    )
    st.caption("Ingrese el gas arterial. Los estudios adicionales quedan en desplegables para no sobrecargar la pantalla.")

    c1,c2,c3,c4=st.columns(4)
    with c1: ph=st.number_input("pH",6.80,7.80,7.40,0.01,key="abg813_ph")
    with c2: pco2=st.number_input("PaCO₂ (mmHg)",5.0,150.0,40.0,1.0,key="abg813_pco2")
    with c3: pao2=st.number_input("PaO₂ (mmHg)",10.0,700.0,90.0,1.0,key="abg813_pao2")
    with c4: hco3=st.number_input("HCO₃⁻ del gas (mmol/L)",2.0,60.0,24.0,0.5,key="abg813_hco3")

    c5,c6,c7=st.columns(3)
    with c5: age=st.number_input("Edad (años)",0.0,120.0,50.0,1.0,key="abg813_age")
    with c6: fio2_pct=st.number_input("FiO₂ (%)",21.0,100.0,21.0,1.0,key="abg813_fio2")
    with c7: altitude=st.number_input("Altitud (m)",0.0,5000.0,0.0,50.0,key="abg813_altitude",help="Se usa para ajustar la presión barométrica en la ecuación alveolar. Si no desea ajustarla, deje 0 m.")

    serum_hco3=None; na=None; kval=None; cl=None; albumin=None; lactate=None
    glucose=None; bun=None; measured_osm=None; ethanol=None; phosphate=None; mg=None; ca=None
    with st.expander("Gap, lactato y osmolaridad · opcional"):
        a1,a2,a3,a4=st.columns(4)
        with a1: na=_el_v2_optional_float("Na (mmol/L)","abg813_na","Ej. 140")
        with a2: kval=_el_v2_optional_float("K (mmol/L)","abg813_k","Ej. 4,0")
        with a3: cl=_el_v2_optional_float("Cl (mmol/L)","abg813_cl","Ej. 104")
        with a4: serum_hco3=_el_v2_optional_float("HCO₃⁻ sérico/CO₂ total (mmol/L)","abg813_shco3","Ej. 18")
        b1,b2,b3,b4=st.columns(4)
        with b1: albumin=_el_v2_optional_float("Albúmina (g/L)","abg813_albumin","Ej. 30")
        with b2: lactate=_el_v2_optional_float("Lactato (mmol/L)","abg813_lactate","Ej. 4,2")
        with b3: glucose=_el_v2_optional_float("Glucosa (mmol/L)","abg813_glucose","Ej. 8")
        with b4: bun=_el_v2_optional_float("BUN (mg/dL)","abg813_bun","Ej. 20")
        o1,o2,o3=st.columns(3)
        with o1: measured_osm=_el_v2_optional_float("Osmolalidad medida (mOsm/kg)","abg813_osm","Ej. 310")
        with o2: ethanol=_el_v2_optional_float("Etanol (mg/dL)","abg813_etoh","Ej. 0")
        with o3: phosphate=_el_v2_optional_float("Fósforo (mmol/L)","abg813_p","Ej. 1,0")

    hb=None; sao2=None
    with st.expander("Oxigenación avanzada · opcional"):
        o1,o2=st.columns(2)
        with o1: hb=_el_v2_optional_float("Hemoglobina (g/dL)","abg813_hb","Ej. 13")
        with o2: sao2=_el_v2_optional_float("SaO₂ arterial (%)","abg813_sao2","Ej. 96")

    with st.expander("Stewart / strong ion gap · opcional"):
        s1,s2=st.columns(2)
        with s1: ca=_el_v2_optional_float("Calcio ionizado/total para SID (mmol/L)","abg813_ca","Ej. 1,15")
        with s2: mg=_el_v2_optional_float("Magnesio (mmol/L)","abg813_mg","Ej. 0,8")
        st.caption("El análisis Stewart solo se calcula cuando están disponibles Na, K, Cl, lactato, Ca, Mg, albúmina y fósforo.")

    una=uk=ucl=uhco3=None
    with st.expander("Acidosis metabólica sin gap: anion gap urinario · opcional"):
        u1,u2,u3,u4=st.columns(4)
        with u1: una=_el_v2_optional_float("Na urinario (mmol/L)","abg813_una","Ej. 40")
        with u2: uk=_el_v2_optional_float("K urinario (mmol/L)","abg813_uk","Ej. 25")
        with u3: ucl=_el_v2_optional_float("Cl urinario (mmol/L)","abg813_ucl","Ej. 90")
        with u4: uhco3=_el_v2_optional_float("HCO₃⁻ urinario (mmol/L) · si disponible","abg813_uhco3","Ej. 0")

    try:
        ab=comprehensive_acid_base_interpretation(ph=ph,pco2_mm_hg=pco2,bicarbonate_mmol_l=hco3)
        hh=float(henderson_hasselbalch_hco3_mmol_l(ph=ph,pco2_mm_hg=pco2))
        fio2=float(fio2_pct)/100.0
        pb=float(barometric_pressure_from_altitude_mm_hg(altitude))
        PAO2=float(alveolar_oxygen_pressure_mm_hg(fio2=fio2,pco2_mm_hg=pco2,barometric_pressure_mm_hg=pb))
        aa=float(aa_gradient_mm_hg(pao2_mm_hg=pao2,fio2=fio2,pco2_mm_hg=pco2,barometric_pressure_mm_hg=pb))
        aa_expected=float(expected_aa_gradient_mm_hg(age_years=age))
        pf=float(pf_ratio_mm_hg(pao2_mm_hg=pao2,fio2=fio2))
    except Exception as e:
        st.error(str(e)); return

    labels={
        "ACIDOSIS_METABOLICA":"acidosis metabólica","ALCALOSIS_METABOLICA":"alcalosis metabólica",
        "ACIDOSIS_RESPIRATORIA":"acidosis respiratoria","ALCALOSIS_RESPIRATORIA":"alcalosis respiratoria",
        "ACIDOSIS_RESPIRATORIA_COMPENSADA_O_ALCALOSIS_METABOLICA":"acidosis respiratoria compensada o alcalosis metabólica",
        "ALCALOSIS_RESPIRATORIA_COMPENSADA_O_ACIDOSIS_METABOLICA":"alcalosis respiratoria compensada o acidosis metabólica",
        "SIN_TRASTORNO_MAYOR_EVIDENTE":"sin trastorno ácido-base mayor evidente",
    }
    state_label={"ACIDEMIA":"ACIDEMIA","ALKALEMIA":"ALCALEMIA","PH_EN_RANGO":"pH EN RANGO"}.get(ab.get("state"),ab.get("state"))
    proc=" + ".join(labels.get(x,x) for x in ab.get("processes") or [])

    with st.container(border=True):
        st.markdown("### Interpretación integral del gas arterial")
        st.info(f"**{state_label} · {proc.upper()}**")
        if ab.get("chronicity"): st.write(f"**Patrón respiratorio:** compatible principalmente con **{ab.get('chronicity').lower()}** según la respuesta de HCO₃⁻ esperada.")
        for d in ab.get("details") or []: st.write(f"• {d}")
        comp=ab.get("compensation")
        if comp:
            st.write(f"**Compensación esperada:** PaCO₂ ≈ {fmt_num(comp.get('expected'),1)} mmHg (rango aproximado {fmt_num(comp.get('lower'),1)}–{fmt_num(comp.get('upper'),1)}). Medida: **{fmt_num(pco2,1)} mmHg**.")

        st.markdown("#### Comprobación interna del gas")
        diff=abs(float(hco3)-hh)
        st.write(f"HCO₃⁻ calculado por Henderson–Hasselbalch ≈ **{fmt_num(hh,1)} mmol/L**; informado: **{fmt_num(hco3,1)} mmol/L**; diferencia **{fmt_num(diff,1)} mmol/L**.")
        if diff>3: st.warning("La discrepancia entre HCO₃⁻ informado y el calculado desde pH/PaCO₂ es mayor de 3 mmol/L. Revisar muestra, transcripción y si se está comparando HCO₃⁻ del gas con CO₂ total sérico.")
        if serum_hco3 is not None:
            ds=abs(float(serum_hco3)-float(hco3))
            st.write(f"CO₂ total/HCO₃⁻ sérico: **{fmt_num(serum_hco3,1)} mmol/L** · diferencia frente al gas: **{fmt_num(ds,1)} mmol/L**.")
            if ds>3: st.warning("Discrepancia relevante gas–química: para anion gap, priorice el bicarbonato/CO₂ total de química sérica si la muestra es contemporánea y fiable.")

        if na is not None and cl is not None:
            hco3_gap=float(serum_hco3 if serum_hco3 is not None else hco3)
            ag=float(anion_gap_mmol_l(sodium_mmol_l=na,chloride_mmol_l=cl,bicarbonate_mmol_l=hco3_gap,potassium_mmol_l=None))
            agk=float(anion_gap_mmol_l(sodium_mmol_l=na,chloride_mmol_l=cl,bicarbonate_mmol_l=hco3_gap,potassium_mmol_l=kval)) if kval is not None else None
            agc=float(albumin_corrected_anion_gap_mmol_l(anion_gap=ag,albumin_g_l=albumin)) if albumin is not None else None
            ag_use=agc if agc is not None else ag
            dg=float(delta_gap_mmol_l(anion_gap=ag_use))
            dh=float(delta_bicarbonate_mmol_l(bicarbonate_mmol_l=hco3_gap))
            dr=delta_ratio(anion_gap=ag_use,bicarbonate_mmol_l=hco3_gap)
            corr_hco3=float(corrected_bicarbonate_from_delta_gap(bicarbonate_mmol_l=hco3_gap,delta_gap=dg))
            st.markdown("#### Gap y trastornos metabólicos mixtos")
            st.write(f"**Anion gap sin K:** {fmt_num(ag,1)} mmol/L" + (f" · **con K:** {fmt_num(agk,1)} mmol/L" if agk is not None else ""))
            if agc is not None: st.write(f"**Anion gap corregido por albúmina:** {fmt_num(agc,1)} mmol/L.")
            st.write(f"**Delta gap (ΔAG):** {fmt_num(dg,1)} mmol/L · **ΔHCO₃:** {fmt_num(dh,1)} mmol/L · **HCO₃ corregido por delta:** {fmt_num(corr_hco3,1)} mmol/L.")
            if dr is not None:
                drf=float(dr); dri=interpret_delta_ratio_value(dr)
                drlabels={
                    "HAGMA_MAS_ACIDOSIS_METABOLICA_SIN_GAP":"sugiere acidosis metabólica con gap elevado + componente adicional sin gap",
                    "HAGMA_PREDOMINANTE":"compatible principalmente con acidosis metabólica con anion gap elevado",
                    "HAGMA_MAS_ALCALOSIS_METABOLICA_O_HCO3_PREVIAMENTE_ELEVADO":"sugiere HAGMA + alcalosis metabólica o HCO₃⁻ previamente elevado",
                    "SIN_HAGMA_CLARA_O_VALORES_NO_COMPATIBLES":"no interpretable como HAGMA típica con estos valores",
                }
                st.write(f"**Delta ratio:** {fmt_num(drf,2)} → {drlabels.get(dri,dri)}.")
            if lactate is not None:
                st.write(f"**Lactato:** {fmt_num(lactate,1)} mmol/L" + (f" · AG corregido no explicado por lactato ≈ {fmt_num(ag_use-float(lactate),1)} mmol/L" if ag_use is not None else ""))

        st.markdown("#### Oxigenación y ventilación")
        st.write(f"**PaO₂/FiO₂ (P/F): {fmt_num(pf,0)} mmHg** · FiO₂ {fmt_num(fio2_pct,0)}%.")
        st.write(f"**PAO₂ alveolar estimada:** {fmt_num(PAO2,1)} mmHg · **gradiente A–a:** {fmt_num(aa,1)} mmHg · esperado por edad en aire ambiente ≈ {fmt_num(aa_expected,1)} mmHg.")
        if fio2>0.21: st.caption("El rango A–a esperado por edad es más fiable en aire ambiente; con FiO₂ elevada debe interpretarse con cautela.")
        if pao2<60 and pco2<=45: st.warning("Patrón gasométrico compatible con insuficiencia respiratoria hipoxémica (tipo 1); integrar con FiO₂, clínica y mecanismo de hipoxemia.")
        if pao2<60 and pco2>45: st.warning("Hipoxemia asociada a hipercapnia: patrón compatible con insuficiencia respiratoria ventilatoria/hipercápnica.")
        if aa>aa_expected+5 and fio2<=0.21: st.write("• Gradiente A–a aumentado: favorece alteración V/Q, difusión o shunt sobre hipoventilación aislada/FiO₂ baja.")
        elif fio2<=0.21: st.write("• Gradiente A–a no claramente aumentado: si existe hipoxemia, considerar hipoventilación o baja presión inspirada de O₂ entre las posibilidades.")
        if hb is not None and sao2 is not None:
            cao2=float(arterial_oxygen_content_ml_dl(hemoglobin_g_dl=hb,sao2_percent=sao2,pao2_mm_hg=pao2))
            st.write(f"**Contenido arterial de O₂ (CaO₂): {fmt_num(cao2,1)} mL O₂/dL** con Hb {fmt_num(hb,1)} g/dL y SaO₂ {fmt_num(sao2,1)}%.")

        if measured_osm is not None and na is not None:
            og=osmolar_gap_mosm_kg(measured_osmolality_mosm_kg=measured_osm,sodium_mmol_l=na,glucose_mmol_l=glucose,bun_mg_dl=bun,ethanol_mg_dl=ethanol)
            st.markdown("#### Gap osmolar")
            st.write(f"Osmolalidad calculada ≈ **{fmt_num(og.get('calculated'),1)} mOsm/kg** · medida **{fmt_num(measured_osm,1)}** · **gap osmolar {fmt_num(og.get('gap'),1)} mOsm/kg**.")
            if float(og.get("gap"))>10: st.warning("Gap osmolar >10 mOsm/kg: existen osmoles no explicados por la fórmula; correlacionar con tóxicos/alcoholes, manitol, contraste y contexto clínico. Un gap normal no excluye intoxicación evolucionada.")

        if all(v is not None for v in [una,uk,ucl]):
            uag=float(urine_anion_gap_mmol_l(urine_na_mmol_l=una,urine_k_mmol_l=uk,urine_cl_mmol_l=ucl,urine_hco3_mmol_l=uhco3))
            st.markdown("#### Anion gap urinario")
            st.write(f"**UAG:** {fmt_num(uag,1)} mmol/L.")
            if uag<0: st.write("• UAG negativo: en acidosis metabólica sin gap, sugiere excreción apropiada de NH₄⁺ y favorece pérdida extrarrenal de bicarbonato si el contexto coincide.")
            else: st.write("• UAG positivo/no negativo: en acidosis metabólica sin gap puede sugerir baja excreción de NH₄⁺/causa renal; interpretar con función renal, pH urinario y contexto.")

        if all(v is not None for v in [na,kval,cl,lactate,ca,mg,albumin,phosphate]):
            sida=float(stewart_sida_meq_l(sodium_mmol_l=na,potassium_mmol_l=kval,calcium_mmol_l=ca,magnesium_mmol_l=mg,chloride_mmol_l=cl,lactate_mmol_l=lactate))
            side=float(stewart_side_meq_l(bicarbonate_mmol_l=hco3,albumin_g_l=albumin,phosphate_mmol_l=phosphate,ph=ph))
            sig=float(strong_ion_gap_meq_l(sodium_mmol_l=na,potassium_mmol_l=kval,calcium_mmol_l=ca,magnesium_mmol_l=mg,chloride_mmol_l=cl,lactate_mmol_l=lactate,bicarbonate_mmol_l=hco3,albumin_g_l=albumin,phosphate_mmol_l=phosphate,ph=ph))
            st.markdown("#### Análisis Stewart–Figge")
            st.write(f"**SID aparente:** {fmt_num(sida,1)} mEq/L · **SID efectivo:** {fmt_num(side,1)} mEq/L · **Strong Ion Gap (SIG): {fmt_num(sig,1)} mEq/L**.")
            st.caption("El enfoque Stewart es complementario al análisis convencional; no sustituye la interpretación clínica ni los rangos validados por el laboratorio/localidad.")

        st.markdown("#### Conclusión automática")
        conclusions=[]
        conclusions.append(f"Gas compatible con {proc.lower()} ({state_label.lower()}).")
        if ab.get("mixed"): conclusions.append("Existe evidencia de trastorno ácido-base mixto o compensación no simple.")
        if pf<200: conclusions.append("La oxigenación está marcadamente comprometida por índice P/F; interpretar según FiO₂, soporte ventilatorio y PEEP antes de aplicar criterios sindromáticos.")
        elif pf<300: conclusions.append("Índice P/F reducido; existe deterioro de oxigenación.")
        if na is not None and cl is not None:
            conclusions.append("Se calculó anion gap y análisis delta con los electrolitos disponibles.")
        for x in conclusions: st.write(f"• {x}")

    with st.expander("Qué incluye este análisis"):
        st.write("• pH, PaCO₂, HCO₃⁻ y trastorno primario/múltiple.")
        st.write("• Winter y compensación de alcalosis metabólica.")
        st.write("• Compensación esperada aguda y crónica de acidosis/alcalosis respiratoria.")
        st.write("• Henderson–Hasselbalch y comprobación de consistencia del HCO₃⁻.")
        st.write("• Anion gap con/sin K, corrección por albúmina, delta gap, ΔHCO₃, delta ratio y HCO₃ corregido por delta.")
        st.write("• Lactato y gap osmolar cuando se aportan datos.")
        st.write("• Anion gap urinario para acidosis metabólica sin gap cuando se aportan electrolitos urinarios.")
        st.write("• PaO₂/FiO₂, ecuación alveolar, gradiente A–a ajustado por altitud y contenido arterial de O₂.")
        st.write("• Stewart–Figge: SID aparente, SID efectivo y strong ion gap cuando están todos los componentes.")
    st.caption("Interpretación automatizada de apoyo. Las fórmulas identifican patrones fisiológicos; la etiología y el tratamiento definitivo requieren integración con diagnóstico, ventilación, hemodinamia y evolución clínica.")


def page_electrolytes():
    mode=st.radio("Electrolito / análisis",["Panel integral","Potasio","Sodio","Magnesio","Calcio","Fósforo","Gases arteriales","Cloro / ácido-base","Reposición conjunta"],horizontal=True,key="el_v2_mode")
    if mode=="Panel integral": return _page_integral_v3()
    if mode=="Potasio": return _page_potassium_v808()
    if mode=="Sodio": return _page_sodium_v2()
    if mode=="Magnesio": return _page_magnesium_v2()
    if mode=="Calcio": return _page_calcium_v2()
    if mode=="Fósforo": return _page_phosphate_v2()
    if mode=="Gases arteriales": return _page_abg_v813()
    if mode=="Cloro / ácido-base": return _page_chloride_ab_v2()
    return _page_joint_v2()

def page_sources():
    header("Base clínica y fuentes", "Estructura SQL, cobertura y trazabilidad.")
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("MED-ID",COUNTS["medications"])
    c2.metric("Pediatría",COUNTS["pediatric_rules"])
    c3.metric("Renal automático",COUNTS["renal_rules"])
    c4.metric("Toxicología",COUNTS["toxicology"])
    c5.metric("Hidroelectrolitos",COUNTS.get("electrolyte_rules",0))
    st.markdown("#### Base Supabase")
    st.code("medications 1 ─── N pediatric_rules\nmedications 1 ─── N renal_rules\nmedications 1 ─── N renal_bibliography\nmedications 1 ─── 1 toxicology\nelectrolyte_analytes 1 ─── N electrolyte_protocols ─── N electrolyte_rules\nmedications 1 ─── N medication_electrolyte_modifiers")
    st.caption(f"Schema Supabase: {SCHEMA_VERSION} · Datos: {db.metadata('data_version') or 'sin versión'}")
    st.markdown("#### Fuentes")
    for r in db.sources():
        with st.expander(r.get("fuente") or r.get("codigo") or "Fuente"):
            st.write(f"**Código:** {r.get('codigo') or '—'}")
            st.write(f"**Revisión:** {r.get('fecha_revision') or '—'}")
            if r.get("url"): st.link_button("Abrir fuente",r["url"])
    st.success("El catálogo clínico principal se consulta desde PostgreSQL/Supabase con RLS. Los tóxicos externos y antídotos conservan la base original en CSV y añaden una capa revisada con fuentes abiertas, sin eliminar la trazabilidad histórica.")


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
            "Hidroelectrolitos":"🧪  Hidroelectrolitos",
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
elif page=="Hidroelectrolitos": page_electrolytes()
else: page_sources()

st.divider()
st.markdown(
    f'<div class="mc-footer">MedCalc Clínico · {APP_VERSION} · Supabase {SCHEMA_VERSION} · revisión {REVIEW_DATE}<br>'
    'Herramienta de apoyo clínico; no sustituye juicio profesional, ficha técnica ni protocolo institucional.</div>',
    unsafe_allow_html=True,
)
