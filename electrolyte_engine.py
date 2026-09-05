"""MedCalc Clínico · motor genérico de conversiones hidroelectrolíticas.

Este módulo NO contiene reglas clínicas (umbrales, dosis recomendadas, velocidades
máximas, etc.). Es deliberadamente determinista: recibe valores ya seleccionados
por la capa de reglas Supabase y ejecuta conversiones/preparaciones.

Las reglas clínicas deben permanecer data-driven en Supabase.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from typing import Mapping, Optional


D = Decimal


class ElectrolyteCalculationError(ValueError):
    """Error de validación en una conversión o preparación."""


def _d(value, name: str) -> Decimal:
    try:
        out = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ElectrolyteCalculationError(f"{name} debe ser numérico.") from exc
    if not out.is_finite():
        raise ElectrolyteCalculationError(f"{name} debe ser finito.")
    return out


def _positive(value, name: str, *, allow_zero: bool = False) -> Decimal:
    out = _d(value, name)
    if allow_zero:
        if out < 0:
            raise ElectrolyteCalculationError(f"{name} no puede ser negativo.")
    elif out <= 0:
        raise ElectrolyteCalculationError(f"{name} debe ser mayor que cero.")
    return out


def percent_wv_to_mg_ml(percent_wv) -> Decimal:
    """Convierte % p/v (g/100 mL) a mg/mL. 1% p/v = 10 mg/mL."""
    p = _positive(percent_wv, "percent_wv", allow_zero=True)
    return p * D("10")


def mg_ml_to_percent_wv(mg_ml) -> Decimal:
    mg = _positive(mg_ml, "mg_ml", allow_zero=True)
    return mg / D("10")




# -----------------------------------------------------------------------------
# Unidades de laboratorio: la interfaz puede aceptar unidades clínicas habituales
# y el motor normaliza siempre a mmol/L para evaluar reglas Supabase.
# Estas son conversiones fisicoquímicas deterministas, no reglas clínicas.
# -----------------------------------------------------------------------------
_LAB_MOLAR_MASS_MG_PER_MMOL = {
    "MG": D("24.305"),
    "CA": D("40.078"),
    "P": D("30.973761998"),  # mg/dL informado como fósforo elemental
}
_LAB_VALENCE = {"NA": D("1"), "K": D("1"), "CL": D("1"), "HCO3": D("1"), "MG": D("2"), "CA": D("2"), "P": D("1")}


def supported_laboratory_units(analyte_code: str) -> tuple[str, ...]:
    code = str(analyte_code or "").upper()
    if code in {"NA", "K", "CL", "HCO3"}:
        return ("mmol/L", "mEq/L")
    if code in {"MG", "CA"}:
        return ("mg/dL", "mmol/L", "mEq/L", "µmol/L")
    if code == "P":
        return ("mg/dL", "mmol/L", "µmol/L")
    raise ElectrolyteCalculationError(f"Analito sin conversión de laboratorio publicada en el motor: {code}")


def laboratory_value_to_mmol_l(analyte_code: str, value, unit: str) -> Decimal:
    code = str(analyte_code or "").upper()
    u = str(unit or "").replace("umol", "µmol").strip()
    x = _positive(value, "laboratory_value", allow_zero=True)
    if u == "mmol/L":
        return x
    if u == "µmol/L":
        return x / D("1000")
    if u == "mEq/L":
        z = _LAB_VALENCE.get(code)
        if z is None:
            raise ElectrolyteCalculationError(f"No se conoce la valencia para {code}.")
        return x / z
    if u == "mg/dL":
        mw = _LAB_MOLAR_MASS_MG_PER_MMOL.get(code)
        if mw is None:
            raise ElectrolyteCalculationError(f"mg/dL no se admite para {code}.")
        # mg/dL × 10 = mg/L; 1 mmol del ion pesa MW mg.
        return (x * D("10")) / mw
    raise ElectrolyteCalculationError(f"Unidad de laboratorio no admitida para {code}: {unit}")


def mmol_l_to_laboratory_value(analyte_code: str, mmol_l, unit: str) -> Decimal:
    code = str(analyte_code or "").upper()
    u = str(unit or "").replace("umol", "µmol").strip()
    x = _positive(mmol_l, "mmol_l", allow_zero=True)
    if u == "mmol/L":
        return x
    if u == "µmol/L":
        return x * D("1000")
    if u == "mEq/L":
        z = _LAB_VALENCE.get(code)
        if z is None:
            raise ElectrolyteCalculationError(f"No se conoce la valencia para {code}.")
        return x * z
    if u == "mg/dL":
        mw = _LAB_MOLAR_MASS_MG_PER_MMOL.get(code)
        if mw is None:
            raise ElectrolyteCalculationError(f"mg/dL no se admite para {code}.")
        return (x * mw) / D("10")
    raise ElectrolyteCalculationError(f"Unidad de laboratorio no admitida para {code}: {unit}")


def glucose_to_mmol_l(value, unit: str) -> Decimal:
    x = _positive(value, "glucose", allow_zero=True)
    u = str(unit or "").strip()
    if u == "mmol/L":
        return x
    if u == "mg/dL":
        return x / D("18")
    raise ElectrolyteCalculationError(f"Unidad de glucosa no admitida: {unit}")


def albumin_to_g_l(value, unit: str) -> Decimal:
    x = _positive(value, "albumin", allow_zero=True)
    u = str(unit or "").strip()
    if u == "g/L":
        return x
    if u == "g/dL":
        return x * D("10")
    raise ElectrolyteCalculationError(f"Unidad de albúmina no admitida: {unit}")

def mg_to_mmol(mass_mg, molar_mass_g_mol) -> Decimal:
    """Convierte mg de una sustancia a mmol usando su masa molar (g/mol)."""
    mass = _positive(mass_mg, "mass_mg", allow_zero=True)
    mw = _positive(molar_mass_g_mol, "molar_mass_g_mol")
    # Numéricamente, 1 mmol pesa MW mg.
    return mass / mw


def mmol_to_mg(mmol, molar_mass_g_mol) -> Decimal:
    n = _positive(mmol, "mmol", allow_zero=True)
    mw = _positive(molar_mass_g_mol, "molar_mass_g_mol")
    return n * mw


def g_to_mmol(mass_g, molar_mass_g_mol) -> Decimal:
    mass = _positive(mass_g, "mass_g", allow_zero=True)
    return mg_to_mmol(mass * D("1000"), molar_mass_g_mol)


def mmol_to_g(mmol, molar_mass_g_mol) -> Decimal:
    return mmol_to_mg(mmol, molar_mass_g_mol) / D("1000")


def mmol_to_meq(mmol, valence) -> Decimal:
    n = _positive(mmol, "mmol", allow_zero=True)
    z = abs(_d(valence, "valence"))
    if z == 0:
        raise ElectrolyteCalculationError("valence no puede ser cero.")
    return n * z


def meq_to_mmol(meq, valence) -> Decimal:
    eq = _positive(meq, "meq", allow_zero=True)
    z = abs(_d(valence, "valence"))
    if z == 0:
        raise ElectrolyteCalculationError("valence no puede ser cero.")
    return eq / z


def mmol_ml_from_salt_mg_ml(
    salt_mg_ml,
    salt_molar_mass_g_mol,
    stoichiometric_coefficient=1,
) -> Decimal:
    """mmol del ion/componente por mL a partir de mg/mL de la sal.

    Ej.: KCl 149 mg/mL / 74.5513 mg/mmol ≈ 1.9986 mmol K/mL.
    Para sales con >1 mol del componente por mol de sal, usar el coeficiente.
    """
    salt = _positive(salt_mg_ml, "salt_mg_ml", allow_zero=True)
    mw = _positive(salt_molar_mass_g_mol, "salt_molar_mass_g_mol")
    coeff = _positive(stoichiometric_coefficient, "stoichiometric_coefficient")
    return (salt / mw) * coeff


def product_volume_for_mmol(target_mmol, product_mmol_per_ml) -> Decimal:
    target = _positive(target_mmol, "target_mmol", allow_zero=True)
    conc = _positive(product_mmol_per_ml, "product_mmol_per_ml")
    return target / conc


def product_volume_for_meq(target_meq, product_meq_per_ml) -> Decimal:
    target = _positive(target_meq, "target_meq", allow_zero=True)
    conc = _positive(product_meq_per_ml, "product_meq_per_ml")
    return target / conc


def product_units_for_mmol(target_mmol, product_mmol_per_unit) -> Decimal:
    """Número de comprimidos/unidades necesario para una cantidad objetivo en mmol."""
    target = _positive(target_mmol, "target_mmol", allow_zero=True)
    per_unit = _positive(product_mmol_per_unit, "product_mmol_per_unit")
    return target / per_unit


def product_units_for_meq(target_meq, product_meq_per_unit) -> Decimal:
    target = _positive(target_meq, "target_meq", allow_zero=True)
    per_unit = _positive(product_meq_per_unit, "product_meq_per_unit")
    return target / per_unit


def final_concentration_mmol_l(amount_mmol, final_volume_ml) -> Decimal:
    amount = _positive(amount_mmol, "amount_mmol", allow_zero=True)
    volume = _positive(final_volume_ml, "final_volume_ml")
    return amount * D("1000") / volume


def final_concentration_meq_l(amount_meq, final_volume_ml) -> Decimal:
    amount = _positive(amount_meq, "amount_meq", allow_zero=True)
    volume = _positive(final_volume_ml, "final_volume_ml")
    return amount * D("1000") / volume


def ml_h_from_volume_duration(volume_ml, duration_h) -> Decimal:
    volume = _positive(volume_ml, "volume_ml", allow_zero=True)
    duration = _positive(duration_h, "duration_h")
    return volume / duration


def mmol_h_from_concentration_rate(concentration_mmol_l, rate_ml_h) -> Decimal:
    conc = _positive(concentration_mmol_l, "concentration_mmol_l", allow_zero=True)
    rate = _positive(rate_ml_h, "rate_ml_h", allow_zero=True)
    return conc * rate / D("1000")


def meq_h_from_concentration_rate(concentration_meq_l, rate_ml_h) -> Decimal:
    conc = _positive(concentration_meq_l, "concentration_meq_l", allow_zero=True)
    rate = _positive(rate_ml_h, "rate_ml_h", allow_zero=True)
    return conc * rate / D("1000")


def duration_h_from_amount_rate(amount, rate_per_h) -> Decimal:
    a = _positive(amount, "amount", allow_zero=True)
    rate = _positive(rate_per_h, "rate_per_h")
    return a / rate


def ampoules_required(required_volume_ml, ampoule_volume_ml) -> dict:
    """Devuelve ampollas enteras a abrir y fracción utilizada de la última."""
    req = _positive(required_volume_ml, "required_volume_ml", allow_zero=True)
    amp = _positive(ampoule_volume_ml, "ampoule_volume_ml")
    if req == 0:
        return {
            "whole_ampoules_to_open": 0,
            "full_ampoules_used": 0,
            "last_ampoule_volume_used_ml": D("0"),
            "last_ampoule_fraction": D("0"),
            "unused_volume_ml": D("0"),
        }
    opened = int((req / amp).to_integral_value(rounding=ROUND_CEILING))
    full = int(req // amp)
    last_used = req - (D(full) * amp)
    if last_used == 0:
        full = opened
        last_fraction = D("0")
        unused = D("0")
    else:
        last_fraction = last_used / amp
        unused = amp - last_used
    return {
        "whole_ampoules_to_open": opened,
        "full_ampoules_used": full,
        "last_ampoule_volume_used_ml": last_used,
        "last_ampoule_fraction": last_fraction,
        "unused_volume_ml": unused,
    }


def component_amounts_from_volume(
    volume_ml,
    components_mmol_per_ml: Mapping[str, object],
) -> dict[str, Decimal]:
    vol = _positive(volume_ml, "volume_ml", allow_zero=True)
    out: dict[str, Decimal] = {}
    for code, concentration in components_mmol_per_ml.items():
        c = _positive(concentration, f"components[{code}]", allow_zero=True)
        out[str(code)] = vol * c
    return out


def validate_product_composition(
    *,
    salt_mg_per_ml,
    salt_molar_mass_g_mol,
    declared_component_mmol_per_ml,
    stoichiometric_coefficient=1,
    relative_tolerance=D("0.02"),
) -> dict:
    """Comprueba coherencia entre etiqueta de masa y mmol declarados.

    No sustituye la ficha oficial: sirve para detectar errores de transcripción.
    """
    theoretical = mmol_ml_from_salt_mg_ml(
        salt_mg_per_ml,
        salt_molar_mass_g_mol,
        stoichiometric_coefficient,
    )
    declared = _positive(
        declared_component_mmol_per_ml,
        "declared_component_mmol_per_ml",
        allow_zero=True,
    )
    tolerance = _positive(relative_tolerance, "relative_tolerance", allow_zero=True)
    if theoretical == 0:
        rel_error = D("0") if declared == 0 else D("Infinity")
    else:
        rel_error = abs(declared - theoretical) / theoretical
    return {
        "theoretical_mmol_per_ml": theoretical,
        "declared_mmol_per_ml": declared,
        "relative_error": rel_error,
        "within_tolerance": rel_error <= tolerance,
    }


@dataclass(frozen=True)
class InfusionPreparation:
    target_mmol: Decimal
    target_meq: Decimal
    concentrate_volume_ml: Decimal
    diluent_volume_ml: Decimal
    final_volume_ml: Decimal
    final_concentration_mmol_l: Decimal
    final_concentration_meq_l: Decimal
    rate_ml_h: Decimal
    rate_mmol_h: Decimal
    rate_meq_h: Decimal
    duration_h: Decimal
    ampoules: Optional[dict]


def prepare_infusion(
    *,
    target_mmol,
    product_mmol_per_ml,
    valence,
    container_volume_ml,
    duration_h,
    preparation_mode: str = "REPLACE_EQUAL_VOLUME",
    ampoule_volume_ml=None,
) -> InfusionPreparation:
    """Calcula una preparación sin aplicar límites clínicos.

    preparation_mode:
      - REPLACE_EQUAL_VOLUME: retirar del diluyente el mismo volumen de concentrado;
        el volumen final permanece igual al volumen nominal del contenedor.
      - ADD_TO_BAG: añadir concentrado sin retirar volumen; volumen final aumenta.
      - FINAL_VOLUME: preparar hasta un volumen final exacto; matemáticamente igual
        a REPLACE_EQUAL_VOLUME, pero semánticamente representa una preparación aforada.

    Las reglas de seguridad (concentración máxima, velocidad máxima, vía, ECG)
    deben validarse contra Supabase fuera de esta función.
    """
    target = _positive(target_mmol, "target_mmol", allow_zero=True)
    prod = _positive(product_mmol_per_ml, "product_mmol_per_ml")
    z = abs(_d(valence, "valence"))
    if z == 0:
        raise ElectrolyteCalculationError("valence no puede ser cero.")
    container = _positive(container_volume_ml, "container_volume_ml")
    duration = _positive(duration_h, "duration_h")

    concentrate = product_volume_for_mmol(target, prod)
    mode = str(preparation_mode or "").strip().upper()
    if mode in {"REPLACE_EQUAL_VOLUME", "FINAL_VOLUME"}:
        if concentrate > container:
            raise ElectrolyteCalculationError(
                "El volumen de concentrado supera el volumen final/nominal indicado."
            )
        diluent = container - concentrate
        final_volume = container
    elif mode == "ADD_TO_BAG":
        diluent = container
        final_volume = container + concentrate
    else:
        raise ElectrolyteCalculationError(
            "preparation_mode debe ser REPLACE_EQUAL_VOLUME, ADD_TO_BAG o FINAL_VOLUME."
        )

    target_meq = mmol_to_meq(target, z)
    conc_mmol_l = final_concentration_mmol_l(target, final_volume)
    conc_meq_l = final_concentration_meq_l(target_meq, final_volume)
    rate_ml_h = ml_h_from_volume_duration(final_volume, duration)
    rate_mmol_h = mmol_h_from_concentration_rate(conc_mmol_l, rate_ml_h)
    rate_meq_h = meq_h_from_concentration_rate(conc_meq_l, rate_ml_h)
    amps = None
    if ampoule_volume_ml is not None:
        amps = ampoules_required(concentrate, ampoule_volume_ml)

    return InfusionPreparation(
        target_mmol=target,
        target_meq=target_meq,
        concentrate_volume_ml=concentrate,
        diluent_volume_ml=diluent,
        final_volume_ml=final_volume,
        final_concentration_mmol_l=conc_mmol_l,
        final_concentration_meq_l=conc_meq_l,
        rate_ml_h=rate_ml_h,
        rate_mmol_h=rate_mmol_h,
        rate_meq_h=rate_meq_h,
        duration_h=duration,
        ampoules=amps,
    )


@dataclass(frozen=True)
class PremixedInfusionPreparation:
    """Preparación para una solución lista para usar, sin dilución adicional."""

    target_mmol: Decimal
    target_meq: Decimal
    product_total_mmol: Decimal
    product_total_meq: Decimal
    volume_to_infuse_ml: Decimal
    container_volume_ml: Decimal
    container_fraction: Decimal
    unused_volume_ml: Decimal
    final_volume_ml: Decimal
    final_concentration_mmol_l: Decimal
    final_concentration_meq_l: Decimal
    rate_ml_h: Decimal
    rate_mmol_h: Decimal
    rate_meq_h: Decimal
    duration_h: Decimal
    partial_use_policy: str


def prepare_premixed_infusion(
    *,
    target_mmol,
    product_mmol_per_ml,
    valence,
    container_volume_ml,
    duration_h,
    partial_use_policy: str = "ALLOW_PARTIAL_DISCARD_REMAINDER",
) -> PremixedInfusionPreparation:
    """Calcula la administración de una bolsa/frasco PREMIXED_READY_TO_USE.

    No añade diluyente ni modifica la concentración del producto. La política de
    uso parcial viene de los metadatos Supabase del producto, no de una regla
    clínica embebida en Python.
    """
    target = _positive(target_mmol, "target_mmol")
    prod = _positive(product_mmol_per_ml, "product_mmol_per_ml")
    z = abs(_d(valence, "valence"))
    if z == 0:
        raise ElectrolyteCalculationError("valence no puede ser cero.")
    container = _positive(container_volume_ml, "container_volume_ml")
    duration = _positive(duration_h, "duration_h")
    total_mmol = prod * container
    total_meq = mmol_to_meq(total_mmol, z)
    policy = str(partial_use_policy or "").strip().upper()

    if target > total_mmol:
        raise ElectrolyteCalculationError(
            "La cantidad solicitada supera el contenido total del envase premezclado."
        )
    if policy == "WHOLE_CONTAINER_ONLY":
        if target != total_mmol:
            raise ElectrolyteCalculationError(
                "Este producto está configurado para administrar el envase completo; "
                "la cantidad solicitada no coincide con su contenido."
            )
        volume = container
    elif policy in {"ALLOW_PARTIAL_DISCARD_REMAINDER", "ALLOW_PARTIAL"}:
        volume = target / prod
    else:
        raise ElectrolyteCalculationError(
            "partial_use_policy debe ser WHOLE_CONTAINER_ONLY o ALLOW_PARTIAL_DISCARD_REMAINDER."
        )

    target_meq = mmol_to_meq(target, z)
    conc_mmol_l = final_concentration_mmol_l(target, volume)
    conc_meq_l = final_concentration_meq_l(target_meq, volume)
    rate_ml_h = ml_h_from_volume_duration(volume, duration)
    rate_mmol_h = target / duration
    rate_meq_h = target_meq / duration
    unused = container - volume

    return PremixedInfusionPreparation(
        target_mmol=target,
        target_meq=target_meq,
        product_total_mmol=total_mmol,
        product_total_meq=total_meq,
        volume_to_infuse_ml=volume,
        container_volume_ml=container,
        container_fraction=volume / container,
        unused_volume_ml=unused,
        final_volume_ml=volume,
        final_concentration_mmol_l=conc_mmol_l,
        final_concentration_meq_l=conc_meq_l,
        rate_ml_h=rate_ml_h,
        rate_mmol_h=rate_mmol_h,
        rate_meq_h=rate_meq_h,
        duration_h=duration,
        partial_use_policy=policy,
    )


def component_amounts_from_solution_volume(volume_ml, components_mmol_per_l: Mapping[str, object]) -> dict:
    """Carga iónica de una solución expresada por litro para un volumen administrado."""
    volume = _positive(volume_ml, "volume_ml", allow_zero=True)
    out = {}
    for code, per_l in (components_mmol_per_l or {}).items():
        if per_l is None:
            continue
        out[str(code)] = _positive(per_l, f"{code}_mmol_per_l", allow_zero=True) * volume / D("1000")
    return out


def merge_electrolyte_loads(*loads: Mapping[str, object]) -> dict:
    """Suma cargas de varios productos/diluyentes sin conocer qué iones son."""
    total = {}
    for load in loads:
        for code, value in (load or {}).items():
            amount = _positive(value, f"{code}_load", allow_zero=True)
            total[str(code)] = total.get(str(code), D("0")) + amount
    return total


def evaluate_administration_options(
    preparation,
    limits,
    context: Mapping[str, object],
    *,
    available_line_types=("PERIPHERAL", "CENTRAL"),
    product_id=None,
    preferred_line_type=None,
) -> dict:
    """Evalúa de forma genérica qué accesos cumplen los límites Supabase.

    Los máximos, condiciones y requisitos proceden exclusivamente de ``limits``.
    Un límite con ``product_id`` solo aplica al producto seleccionado; uno sin
    ``product_id`` se trata como regla general del protocolo.
    """
    options = []
    for line in available_line_types or ():
        line_u = str(line).upper()
        relevant = []
        for lim in limits or []:
            lim_product = lim.get("product_id")
            if lim_product not in (None, "") and product_id not in (lim_product, str(lim_product)):
                continue
            lt = str(lim.get("line_type") or "ANY").upper()
            if lt not in {"ANY", line_u}:
                continue
            if not evaluate_condition(lim.get("condition_json") or {}, context):
                continue
            relevant.append(lim)
        validations = [validate_administration_limit(preparation, lim) for lim in relevant]
        central_block = any(v.get("requires_central_line") for v in validations) and line_u != "CENTRAL"
        verified = bool(relevant)
        ok = verified and not central_block and all(v.get("ok") for v in validations)
        options.append({
            "line_type": line_u,
            "verified": verified,
            "ok": ok,
            "limits": relevant,
            "validations": validations,
            "requires_ecg": any(v.get("requires_ecg") for v in validations),
            "requires_pump": any(v.get("requires_pump") for v in validations),
            "central_block": central_block,
        })

    pref = str(preferred_line_type or "").upper()
    ranked = sorted(
        options,
        key=lambda x: (
            0 if x["ok"] else 1,
            0 if pref and x["line_type"] == pref else 1,
            0 if x["line_type"] == "PERIPHERAL" else 1,
        ),
    )
    suggested = next((x["line_type"] for x in ranked if x["ok"]), None)
    return {"suggested_line": suggested, "options": options}

# -----------------------------------------------------------------------------
# MOTOR GENÉRICO DE CONDICIONES DATA-DRIVEN
# -----------------------------------------------------------------------------

_MISSING = object()


def _resolve_field(context: Mapping[str, object], field: str):
    """Resuelve rutas con punto, p. ej. ``renal.egfr``.

    El motor no conoce nombres clínicos concretos; solo navega el contexto que
    la interfaz/repository construye a partir de los datos del paciente.
    """
    current = context
    for part in str(field or "").split("."):
        if not part:
            return _MISSING
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def _compare(actual, op: str, expected=None) -> bool:
    op = str(op or "").strip().lower()
    if op in {"exists", "not_exists"}:
        exists = actual is not _MISSING and actual is not None
        return exists if op == "exists" else not exists

    if actual is _MISSING:
        return False

    if op in {"truthy", "falsy"}:
        return bool(actual) if op == "truthy" else not bool(actual)

    if op in {"=", "==", "eq"}:
        return actual == expected
    if op in {"!=", "ne"}:
        return actual != expected
    if op in {"in", "not_in"}:
        try:
            result = actual in expected
        except TypeError as exc:
            raise ElectrolyteCalculationError("El operador IN requiere una colección en value.") from exc
        return result if op == "in" else not result
    if op in {"contains", "not_contains"}:
        try:
            result = expected in actual
        except TypeError as exc:
            raise ElectrolyteCalculationError("El operador CONTAINS requiere un valor contenedor.") from exc
        return result if op == "contains" else not result

    if op in {"<", "<=", ">", ">="}:
        # Los datos clínicos opcionales pueden no estar disponibles. En ese caso
        # la condición numérica simplemente NO se cumple; no debe caer toda la app.
        # Ejemplo: Mg no solicitado -> magnesium.value_mmol_l = None.
        if actual is None:
            return False
        if expected is None:
            raise ElectrolyteCalculationError(
                f"Regla numérica inválida: falta el valor esperado para {op}."
            )
        try:
            a = Decimal(str(actual))
            b = Decimal(str(expected))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ElectrolyteCalculationError(
                f"Comparación numérica inválida: {actual!r} {op} {expected!r}."
            ) from exc
        return {
            "<": a < b,
            "<=": a <= b,
            ">": a > b,
            ">=": a >= b,
        }[op]

    raise ElectrolyteCalculationError(f"Operador de condición no soportado: {op!r}.")


def evaluate_condition(condition: Mapping[str, object] | None, context: Mapping[str, object]) -> bool:
    """Evalúa un árbol JSON genérico de condiciones.

    Formatos soportados::

        {"all": [cond1, cond2]}
        {"any": [cond1, cond2]}
        {"not": cond}
        {"field": "serum_k", "op": "<", "value": 2.5}

    Un objeto vacío equivale a TRUE. Los umbrales y decisiones permanecen en
    Supabase; esta función solo interpreta operadores genéricos.
    """
    if condition is None or condition == {}:
        return True
    if not isinstance(condition, Mapping):
        raise ElectrolyteCalculationError("condition debe ser un objeto/mapa.")

    combinators = [k for k in ("all", "any", "not") if k in condition]
    if combinators:
        if len(combinators) != 1:
            raise ElectrolyteCalculationError("Una condición no puede mezclar varios combinadores al mismo nivel.")
        key = combinators[0]
        payload = condition[key]
        if key == "not":
            return not evaluate_condition(payload, context)
        if not isinstance(payload, (list, tuple)):
            raise ElectrolyteCalculationError(f"{key} debe contener una lista de condiciones.")
        # Usar cortocircuito real. Además de ser más eficiente, evita evaluar
        # comparaciones dependientes de un dato opcional cuando una condición
        # previa ya hace imposible que la regla se cumpla.
        if key == "all":
            return all(evaluate_condition(c, context) for c in payload)
        return any(evaluate_condition(c, context) for c in payload)

    if "field" not in condition or "op" not in condition:
        raise ElectrolyteCalculationError("La condición hoja requiere field y op.")
    actual = _resolve_field(context, str(condition["field"]))
    return _compare(actual, str(condition["op"]), condition.get("value"))


def evaluate_rules(
    rules,
    context: Mapping[str, object],
    *,
    status_field: str = "status",
    only_published: bool = True,
):
    """Devuelve reglas coincidentes ordenadas por ``priority`` ascendente.

    Acepta dicts provenientes directamente de Supabase. No interpreta la acción;
    únicamente decide si ``condition_json`` se cumple.
    """
    matched = []
    for rule in rules or []:
        if only_published and str(rule.get(status_field) or "").upper() != "PUBLISHED":
            continue
        condition = rule.get("condition_json") or {}
        if evaluate_condition(condition, context):
            matched.append(rule)
    return sorted(
        matched,
        key=lambda r: (
            int(r.get("priority") if r.get("priority") is not None else 100),
            str(r.get("rule_code") or r.get("id") or ""),
        ),
    )


def validate_administration_limit(preparation: InfusionPreparation, limit: Mapping[str, object]) -> dict:
    """Compara una preparación ya calculada con límites obtenidos de Supabase.

    La función no define ningún máximo; solo compara contra campos del registro
    ``electrolyte_administration_limits`` seleccionado por el motor de reglas.
    """
    checks = []

    def add_check(name, actual: Decimal, maximum_key: str):
        maximum = limit.get(maximum_key)
        if maximum is None:
            return
        max_d = _positive(maximum, maximum_key)
        checks.append({
            "code": name,
            "actual": actual,
            "maximum": max_d,
            "ok": actual <= max_d,
        })

    add_check("CONCENTRATION_MMOL_L", preparation.final_concentration_mmol_l, "max_concentration_mmol_l")
    add_check("CONCENTRATION_MEQ_L", preparation.final_concentration_meq_l, "max_concentration_meq_l")
    add_check("RATE_MMOL_H", preparation.rate_mmol_h, "max_rate_mmol_h")
    add_check("RATE_MEQ_H", preparation.rate_meq_h, "max_rate_meq_h")

    return {
        "ok": all(c["ok"] for c in checks),
        "checks": checks,
        "requires_ecg": bool(limit.get("requires_ecg")),
        "requires_pump": bool(limit.get("requires_pump")),
        "requires_central_line": bool(limit.get("requires_central_line")),
        "monitoring_text": limit.get("monitoring_text"),
    }

# -----------------------------------------------------------------------------
# CÁLCULOS FISIOLÓGICOS AUXILIARES · V2
# -----------------------------------------------------------------------------
# Estas funciones implementan fórmulas deterministas. Los objetivos clínicos,
# límites de corrección y selección de tratamiento permanecen en Supabase y se
# pasan como parámetros desde action_json/protocolos.


def total_body_water_l(weight_kg, *, sex=None, factor=None) -> Decimal:
    """Estima agua corporal total (L).

    Si ``factor`` se proporciona, se usa directamente. De lo contrario utiliza
    0,60 para hombre y 0,50 para mujer. La UI debe mostrar el factor aplicado.
    """
    weight = _positive(weight_kg, "weight_kg")
    if factor is None:
        sex_norm = str(sex or "").strip().upper()
        if sex_norm in {"M", "MALE", "HOMBRE", "MASCULINO"}:
            factor = D("0.60")
        elif sex_norm in {"F", "FEMALE", "MUJER", "FEMENINO"}:
            factor = D("0.50")
        else:
            raise ElectrolyteCalculationError("Se requiere sexo o factor de agua corporal total.")
    f = _positive(factor, "tbw_factor")
    if f > D("1"):
        raise ElectrolyteCalculationError("tbw_factor no puede ser mayor que 1.")
    return weight * f


def free_water_deficit_l(*, weight_kg, serum_na, target_na=140, sex=None, tbw_factor=None) -> Decimal:
    """Déficit estimado de agua libre en hipernatremia.

    Fórmula: ACT × (Na_actual / Na_objetivo - 1). No incluye pérdidas en curso.
    """
    na = _positive(serum_na, "serum_na")
    target = _positive(target_na, "target_na")
    tbw = total_body_water_l(weight_kg, sex=sex, factor=tbw_factor)
    deficit = tbw * ((na / target) - D("1"))
    return deficit if deficit > 0 else D("0")


def sodium_target_for_max_daily_change(serum_na, *, max_change_mmol_l_24h, lower_target=None, direction="DOWN") -> Decimal:
    """Calcula un objetivo de Na a 24 h limitado por la variación máxima permitida."""
    na = _positive(serum_na, "serum_na")
    delta = _positive(max_change_mmol_l_24h, "max_change_mmol_l_24h")
    direction = str(direction or "DOWN").upper()
    if direction == "DOWN":
        target = na - delta
        if lower_target is not None:
            target = max(target, _positive(lower_target, "lower_target"))
        return target
    if direction == "UP":
        target = na + delta
        if lower_target is not None:
            target = min(target, _positive(lower_target, "upper_target"))
        return target
    raise ElectrolyteCalculationError("direction debe ser UP o DOWN.")


def predicted_serum_na_after_infusate(*, serum_na, tbw_l, infusate_na_mmol_l, volume_ml) -> Decimal:
    """Estimación de mezcla simple del Na tras una infusión, sin pérdidas/ganancias renales.

    Se usa como apoyo educativo de magnitud, no como sustituto del control seriado.
    """
    na = _positive(serum_na, "serum_na")
    tbw = _positive(tbw_l, "tbw_l")
    inf_na = _positive(infusate_na_mmol_l, "infusate_na_mmol_l", allow_zero=True)
    vol_l = _positive(volume_ml, "volume_ml", allow_zero=True) / D("1000")
    if vol_l == 0:
        return na
    return ((na * tbw) + (inf_na * vol_l)) / (tbw + vol_l)


def predicted_delta_na_after_infusate(**kwargs) -> Decimal:
    return predicted_serum_na_after_infusate(**kwargs) - _positive(kwargs["serum_na"], "serum_na")


def corrected_sodium_for_hyperglycemia(
    *, serum_na, glucose_mmol_l, correction_mmol_per_100mg_dl=1.6, baseline_glucose_mg_dl=100
) -> Decimal:
    """Corrige Na por hiperglucemia usando un coeficiente configurable.

    La discrepancia 1,6 vs 2,4 mmol/L por cada 100 mg/dL sobre 100 puede
    representarse llamando la función dos veces con coeficientes distintos.
    """
    na = _positive(serum_na, "serum_na")
    glu_mmol = _positive(glucose_mmol_l, "glucose_mmol_l", allow_zero=True)
    coeff = _positive(correction_mmol_per_100mg_dl, "correction_mmol_per_100mg_dl")
    base = _positive(baseline_glucose_mg_dl, "baseline_glucose_mg_dl", allow_zero=True)
    glucose_mg_dl = glu_mmol * D("18")
    excess = glucose_mg_dl - base
    if excess <= 0:
        return na
    return na + coeff * (excess / D("100"))


def calculated_serum_osmolality_mosm_kg(*, sodium_mmol_l, glucose_mmol_l=None, urea_mmol_l=None) -> Decimal:
    """Osmolalidad calculada aproximada: 2×Na + glucosa + urea (mmol/L)."""
    na = _positive(sodium_mmol_l, "sodium_mmol_l")
    glu = D("0") if glucose_mmol_l is None else _positive(glucose_mmol_l, "glucose_mmol_l", allow_zero=True)
    urea = D("0") if urea_mmol_l is None else _positive(urea_mmol_l, "urea_mmol_l", allow_zero=True)
    return D("2") * na + glu + urea


def effective_osmolality_mosm_kg(*, sodium_mmol_l, glucose_mmol_l=None) -> Decimal:
    """Tonicidad aproximada: 2×Na + glucosa (mmol/L)."""
    na = _positive(sodium_mmol_l, "sodium_mmol_l")
    glu = D("0") if glucose_mmol_l is None else _positive(glucose_mmol_l, "glucose_mmol_l", allow_zero=True)
    return D("2") * na + glu


def corrected_calcium_mmol_l(*, total_ca_mmol_l, albumin_g_l, coefficient=0.02, reference_albumin_g_l=40) -> Decimal:
    """Calcio total corregido por albúmina (aproximación secundaria)."""
    ca = _positive(total_ca_mmol_l, "total_ca_mmol_l", allow_zero=True)
    albumin = _positive(albumin_g_l, "albumin_g_l", allow_zero=True)
    coeff = _d(coefficient, "coefficient")
    ref = _positive(reference_albumin_g_l, "reference_albumin_g_l")
    return ca + coeff * (ref - albumin)


def anion_gap_mmol_l(*, sodium_mmol_l, chloride_mmol_l, bicarbonate_mmol_l, potassium_mmol_l=None) -> Decimal:
    na = _positive(sodium_mmol_l, "sodium_mmol_l")
    cl = _positive(chloride_mmol_l, "chloride_mmol_l", allow_zero=True)
    hco3 = _positive(bicarbonate_mmol_l, "bicarbonate_mmol_l", allow_zero=True)
    k = D("0") if potassium_mmol_l is None else _positive(potassium_mmol_l, "potassium_mmol_l", allow_zero=True)
    return na + k - cl - hco3


def albumin_corrected_anion_gap_mmol_l(*, anion_gap, albumin_g_l, correction_per_g_dl=2.5, reference_albumin_g_dl=4.0) -> Decimal:
    ag = _d(anion_gap, "anion_gap")
    alb_g_l = _positive(albumin_g_l, "albumin_g_l", allow_zero=True)
    alb_g_dl = alb_g_l / D("10")
    coeff = _positive(correction_per_g_dl, "correction_per_g_dl")
    ref = _positive(reference_albumin_g_dl, "reference_albumin_g_dl")
    return ag + coeff * (ref - alb_g_dl)


def delta_ratio(*, anion_gap, bicarbonate_mmol_l, normal_ag=12, normal_bicarbonate=24) -> Optional[Decimal]:
    ag = _d(anion_gap, "anion_gap")
    hco3 = _positive(bicarbonate_mmol_l, "bicarbonate_mmol_l", allow_zero=True)
    nag = _d(normal_ag, "normal_ag")
    nhco3 = _d(normal_bicarbonate, "normal_bicarbonate")
    denominator = nhco3 - hco3
    if denominator <= 0:
        return None
    return (ag - nag) / denominator


def winters_expected_pco2_mm_hg(bicarbonate_mmol_l) -> dict:
    hco3 = _positive(bicarbonate_mmol_l, "bicarbonate_mmol_l", allow_zero=True)
    expected = D("1.5") * hco3 + D("8")
    return {"expected": expected, "lower": expected - D("2"), "upper": expected + D("2")}


def metabolic_alkalosis_expected_pco2_mm_hg(bicarbonate_mmol_l) -> dict:
    hco3 = _positive(bicarbonate_mmol_l, "bicarbonate_mmol_l", allow_zero=True)
    expected = D("40") + D("0.7") * (hco3 - D("24"))
    return {"expected": expected, "lower": expected - D("5"), "upper": expected + D("5")}


def interpret_acid_base(*, ph, pco2_mm_hg, bicarbonate_mmol_l) -> dict:
    """Interpretación primaria simplificada con compensación metabólica estándar."""
    ph_d = _positive(ph, "ph")
    pco2 = _positive(pco2_mm_hg, "pco2_mm_hg")
    hco3 = _positive(bicarbonate_mmol_l, "bicarbonate_mmol_l", allow_zero=True)

    if ph_d < D("7.35"):
        state = "ACIDEMIA"
    elif ph_d > D("7.45"):
        state = "ALKALEMIA"
    else:
        state = "PH_CASI_NORMAL"

    primary = "INDETERMINADO"
    compensation = None
    mixed = False
    detail = ""

    if hco3 < D("22"):
        primary = "ACIDOSIS_METABOLICA"
        winter = winters_expected_pco2_mm_hg(hco3)
        compensation = winter
        if pco2 > winter["upper"]:
            mixed = True
            detail = "pCO2 mayor de la esperada: componente de acidosis respiratoria asociado."
        elif pco2 < winter["lower"]:
            mixed = True
            detail = "pCO2 menor de la esperada: componente de alcalosis respiratoria asociado."
        else:
            detail = "Compensación respiratoria dentro del rango esperado por fórmula de Winter."
    elif hco3 > D("26"):
        primary = "ALCALOSIS_METABOLICA"
        exp = metabolic_alkalosis_expected_pco2_mm_hg(hco3)
        compensation = exp
        if pco2 > exp["upper"]:
            mixed = True
            detail = "pCO2 mayor de la esperada: componente de acidosis respiratoria asociado."
        elif pco2 < exp["lower"]:
            mixed = True
            detail = "pCO2 menor de la esperada: componente de alcalosis respiratoria asociado."
        else:
            detail = "Compensación respiratoria compatible con alcalosis metabólica."
    elif pco2 > D("45"):
        primary = "ACIDOSIS_RESPIRATORIA"
        detail = "La cronicidad es necesaria para estimar con precisión la compensación renal esperada."
    elif pco2 < D("35"):
        primary = "ALCALOSIS_RESPIRATORIA"
        detail = "La cronicidad es necesaria para estimar con precisión la compensación renal esperada."
    else:
        primary = "SIN_TRASTORNO_MAYOR_EVIDENTE"
        detail = "pH, pCO2 y bicarbonato no muestran un trastorno ácido-base mayor con estos cortes generales."

    return {
        "state": state,
        "primary": primary,
        "mixed": mixed,
        "detail": detail,
        "compensation": compensation,
    }
