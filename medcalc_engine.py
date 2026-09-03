import math
import unicodedata


def as_float(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def as_bool_si(value):
    return str(value or "").strip().upper() == "SI"


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.casefold().split())


def age_to_months(value, unit):
    value = float(value)
    if unit == "días":
        return value / 30.4375
    if unit == "meses":
        return value
    return value * 12.0


def rule_applies_demographics(rule, age_months, weight_kg):
    amin = as_float(rule.get("edad_min_meses"))
    amax = as_float(rule.get("edad_max_meses"))
    wmin = as_float(rule.get("peso_min_kg"))
    wmax = as_float(rule.get("peso_max_kg"))

    if amin is not None and age_months < amin:
        return False
    # Edad máxima se interpreta como límite superior exclusivo.
    if amax is not None and age_months >= amax:
        return False

    if wmin is not None:
        if as_bool_si(rule.get("peso_min_exclusivo")):
            if weight_kg <= wmin:
                return False
        elif weight_kg < wmin:
            return False

    if wmax is not None:
        if as_bool_si(rule.get("peso_max_exclusivo")):
            if weight_kg >= wmax:
                return False
        elif weight_kg > wmax:
            return False
    return True


def _dose_unit(rule):
    return (rule.get("unidad_dosis") or "mg").strip()


def _value(rule, generic_field, legacy_mg_field=None):
    value = as_float(rule.get(generic_field))
    if value is None and legacy_mg_field:
        value = as_float(rule.get(legacy_mg_field))
    return value


def _daily_cap(rule, weight_kg):
    unit = _dose_unit(rule)
    caps = []
    max_day = _value(rule, "max_dia_valor", "max_dia_mg")
    max_day_kg = _value(rule, "max_dia_valorkg", "max_dia_mgkg")
    if max_day is not None:
        caps.append((max_day, f"máximo {max_day:g} {unit}/día"))
    if max_day_kg is not None:
        absolute = max_day_kg * weight_kg
        caps.append((absolute, f"máximo {max_day_kg:g} {unit}/kg/día"))
    if not caps:
        return None, []
    return min(x[0] for x in caps), [x[1] for x in caps]


def _single_cap(rule, weight_kg):
    unit = _dose_unit(rule)
    caps = []
    max_single = _value(rule, "max_dosis_valor", "max_dosis_mg")
    max_single_kg = as_float(rule.get("max_dosis_valorkg"))
    if max_single is not None:
        caps.append((max_single, f"máximo {max_single:g} {unit}/dosis"))
    if max_single_kg is not None:
        absolute = max_single_kg * weight_kg
        caps.append((absolute, f"máximo {max_single_kg:g} {unit}/kg/dosis"))
    if not caps:
        return None, []
    return min(x[0] for x in caps), [x[1] for x in caps]


def calculate_pediatric_dose(rule, weight_kg):
    """Calcula dosis pediátrica con soporte de mg, mcg y unidades.

    Tipos admitidos:
    - MG_KG_DOSIS / MG_KG_DOSIS_RANGE
    - MG_KG_DIA / MG_KG_DIA_RANGE
    - CANT_KG_DOSIS / CANT_KG_DOSIS_RANGE
    - CANT_KG_DIA / CANT_KG_DIA_RANGE
    - FIJA / FIJA_RANGE

    La unidad real se obtiene de ``unidad_dosis``. Los prefijos MG_ se conservan
    por compatibilidad con la base histórica.
    """
    if weight_kg <= 0:
        raise ValueError("El peso debe ser mayor que cero.")

    kind = (rule.get("tipo_dosis") or "").strip().upper()
    unit = _dose_unit(rule)
    dose = as_float(rule.get("dosis_valor"))
    dose_max = as_float(rule.get("dosis_valor_max"))
    fixed = _value(rule, "dosis_fija_valor", "dosis_fija_mg")
    fixed_max = as_float(rule.get("dosis_fija_valor_max"))
    divisions = as_float(rule.get("divisiones_dia"))
    interval = as_float(rule.get("intervalo_h"))

    single_cap, single_cap_labels = _single_cap(rule, weight_kg)
    daily_cap, daily_cap_labels = _daily_cap(rule, weight_kg)

    result = {
        "kind": kind,
        "unit": unit,
        "interval_h": interval,
        "doses_per_day": None,
        "min_value": None,
        "max_value": None,
        "daily_min_value": None,
        "daily_max_value": None,
        "max_single_value": single_cap,
        "single_cap_labels": single_cap_labels,
        "daily_cap_value": daily_cap,
        "daily_cap_labels": daily_cap_labels,
        "caps_applied": [],
        "formula": "",
    }

    # Weight-based per dose
    if kind in {"MG_KG_DOSIS", "MG_KG_DOSIS_RANGE", "CANT_KG_DOSIS", "CANT_KG_DOSIS_RANGE"}:
        if dose is None:
            raise ValueError("La regla no contiene una dosis por kg válida.")
        upper_rate = dose_max if kind.endswith("RANGE") and dose_max is not None else dose
        lo = dose * weight_kg
        hi = upper_rate * weight_kg
        result["formula"] = f"{weight_kg:g} kg × {dose:g} {unit}/kg/dosis"
        nday = 24.0 / interval if interval else None
        result["doses_per_day"] = nday

    # Weight-based per day
    elif kind in {"MG_KG_DIA", "MG_KG_DIA_RANGE", "CANT_KG_DIA", "CANT_KG_DIA_RANGE"}:
        if dose is None:
            raise ValueError("La regla no contiene una dosis por kg/día válida.")
        upper_rate = dose_max if kind.endswith("RANGE") and dose_max is not None else dose
        div = divisions or (24.0 / interval if interval else 1.0)
        daily_lo = dose * weight_kg
        daily_hi = upper_rate * weight_kg
        lo, hi = daily_lo / div, daily_hi / div
        interval = 24.0 / div
        nday = div
        result["interval_h"] = interval
        result["doses_per_day"] = nday
        result["formula"] = f"{weight_kg:g} kg × {dose:g} {unit}/kg/día ÷ {div:g} dosis/día"

    elif kind in {"FIJA", "FIJA_RANGE"}:
        if fixed is None:
            raise ValueError("La regla no contiene dosis fija válida.")
        lo = fixed
        hi = fixed_max if kind == "FIJA_RANGE" and fixed_max is not None else fixed
        nday = 24.0 / interval if interval else None
        result["doses_per_day"] = nday
        result["formula"] = (
            f"Dosis fija cargada: {fixed:g}–{hi:g} {unit}"
            if hi != fixed else f"Dosis fija cargada: {fixed:g} {unit}"
        )

    else:
        raise ValueError(f"Tipo de dosis no soportado: {kind or 'vacío'}")

    # Apply per-dose caps.
    if single_cap is not None:
        old_lo, old_hi = lo, hi
        lo, hi = min(lo, single_cap), min(hi, single_cap)
        if lo != old_lo or hi != old_hi:
            result["caps_applied"].append("máximo por dosis cargado")

    # Apply daily caps when a numeric number of administrations/day exists.
    if daily_cap is not None and nday:
        per_dose_daily_cap = daily_cap / nday
        old_lo, old_hi = lo, hi
        lo, hi = min(lo, per_dose_daily_cap), min(hi, per_dose_daily_cap)
        if lo != old_lo or hi != old_hi:
            result["caps_applied"].append("máximo diario cargado")

    result["min_value"] = lo
    result["max_value"] = hi
    result["daily_min_value"] = lo * nday if nday else None
    result["daily_max_value"] = hi * nday if nday else None

    # Backward-compatible mg keys used elsewhere/tests.
    if unit == "mg":
        result["min_mg"] = lo
        result["max_mg"] = hi
        result["daily_min_mg"] = result["daily_min_value"]
        result["daily_max_mg"] = result["daily_max_value"]
        result["max_single_mg"] = single_cap
        result["daily_cap_mg"] = daily_cap
    else:
        result["min_mg"] = result["max_mg"] = None
        result["daily_min_mg"] = result["daily_max_mg"] = None
        result["max_single_mg"] = result["daily_cap_mg"] = None

    return result


def quantity_to_ml(min_value, max_value, label_value, label_ml):
    if label_value <= 0 or label_ml <= 0:
        raise ValueError("La concentración debe ser mayor que cero.")
    concentration = label_value / label_ml
    return {
        "unit_per_ml": concentration,
        "min_ml": min_value / concentration,
        "max_ml": max_value / concentration,
    }


def mg_to_ml(min_mg, max_mg, label_mg, label_ml):
    """Compatibilidad con versiones previas."""
    out = quantity_to_ml(min_mg, max_mg, label_mg, label_ml)
    return {
        "mg_per_ml": out["unit_per_ml"],
        "min_ml": out["min_ml"],
        "max_ml": out["max_ml"],
    }

def ckdepi_2021(age_years, sex, creatinine_mg_dl):
    if age_years < 18 or creatinine_mg_dl <= 0:
        return None
    if sex == "Mujer":
        k, alpha, factor = 0.7, -0.241, 1.012
    else:
        k, alpha, factor = 0.9, -0.302, 1.0
    ratio = creatinine_mg_dl / k
    return (
        142
        * min(ratio, 1) ** alpha
        * max(ratio, 1) ** -1.200
        * 0.9938 ** age_years
        * factor
    )


def cockcroft_gault(age_years, sex, weight_kg, creatinine_mg_dl):
    if age_years <= 0 or weight_kg <= 0 or creatinine_mg_dl <= 0:
        return None
    value = ((140 - age_years) * weight_kg) / (72 * creatinine_mg_dl)
    return value * 0.85 if sex == "Mujer" else value


def bedside_schwartz(height_cm, creatinine_mg_dl):
    if height_cm <= 0 or creatinine_mg_dl <= 0:
        return None
    return 0.413 * height_cm / creatinine_mg_dl


def bsa_mosteller(height_cm, weight_kg):
    if height_cm <= 0 or weight_kg <= 0:
        return None
    return math.sqrt((height_cm * weight_kg) / 3600.0)


def normalize_crcl_to_173(crcl_ml_min, bsa_m2):
    if crcl_ml_min is None or bsa_m2 is None or bsa_m2 <= 0:
        return None
    return crcl_ml_min * 1.73 / bsa_m2


def renal_value_for_metric(metric, crcl, crcl_normalized, egfr):
    if metric == "CrCl_CG_mL_min":
        return crcl
    if metric in {"CrCl_mL_min_1_73m2", "CrCl_normalizado_mL_min_1_73m2"}:
        return crcl_normalized
    if metric == "eGFR_CKDEPI_mL_min_1_73m2":
        return egfr
    return None


def renal_band_match(rule, value):
    if value is None:
        return False
    lo = as_float(rule.get("limite_inferior"))
    hi = as_float(rule.get("limite_superior"))
    li = as_bool_si(rule.get("inferior_inclusivo"))
    ui = as_bool_si(rule.get("superior_inclusivo"))

    if lo is not None:
        if li and value < lo:
            return False
        if not li and value <= lo:
            return False
    if hi is not None:
        if ui and value > hi:
            return False
        if not ui and value >= hi:
            return False
    return True


def select_renal_rule(rules, crcl, crcl_normalized, egfr, hemodialysis=False):
    if hemodialysis:
        dialysis = [r for r in rules if (r.get("tipo_regla") or "") == "DIALISIS"]
        if dialysis:
            return dialysis[0], None

    for rule in rules:
        if (rule.get("tipo_regla") or "") == "DIALISIS":
            continue
        value = renal_value_for_metric(rule.get("metrica_renal"), crcl, crcl_normalized, egfr)
        no_limits = as_float(rule.get("limite_inferior")) is None and as_float(rule.get("limite_superior")) is None
        if no_limits or renal_band_match(rule, value):
            return rule, value
    return None, None


def calculate_exposure_mgkg(total_mg, weight_kg, threshold_mgkg=None):
    if weight_kg <= 0:
        raise ValueError("El peso debe ser mayor que cero.")
    exposure = total_mg / weight_kg
    ratio = None
    if threshold_mgkg is not None and threshold_mgkg > 0:
        ratio = exposure / threshold_mgkg
    return exposure, ratio


def renal_biblio_band(crcl_ml_min):
    """Return the source-table CrCl band used by Nefrología al Día FR-001.

    The bibliography tables are labeled 100–50, 50–10 and <10 mL/min.
    Values >=50 are mapped to the first band for display; this helper does not
    itself prescribe a dose and must be used only to select the corresponding
    bibliographic cell.
    """
    if crcl_ml_min is None:
        return None
    if crcl_ml_min >= 50:
        return "crcl_100_50"
    if crcl_ml_min >= 10:
        return "crcl_50_10"
    return "crcl_lt10"
