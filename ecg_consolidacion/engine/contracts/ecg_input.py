"""Contrato canónico de entrada ECG para todo el motor MEDCALC EKG.

Toda capa posterior a la ingesta recibe un `CanonicalECG` validado:
  * 12 derivaciones en el orden I, II, III, aVR, aVL, aVF, V1–V6;
  * 500 Hz, 10 s (5000 muestras);
  * unidades físicas mV (float32/float64), sin NaN/Inf;
  * procedencia (fuente + SHA-256 del archivo original).

Este módulo NO convierte nada: solo acepta o rechaza con un motivo explícito.
Los adaptadores (100/200 Hz, normalización p99, mV/2 de la caché multicohorte)
deben vivir en módulos aparte y probarse contra este contrato.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LEAD_ORDER: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF",
                               "V1", "V2", "V3", "V4", "V5", "V6")
FS_HZ = 500
N_SAMPLES = 5000
UNITS = "mV"
# Un ECG clínico rara vez supera ±10 mV; por encima casi siempre son unidades erróneas (µV).
MAX_ABS_MV = 20.0


class ContractError(ValueError):
    """Entrada que no cumple el contrato canónico."""


@dataclass(frozen=True)
class CanonicalECG:
    signal: np.ndarray            # shape (12, 5000), mV
    source: str                   # p. ej. "LUDB:109"
    source_sha256: str            # SHA-256 del archivo original
    fs_hz: int = FS_HZ
    units: str = UNITS
    leads: tuple[str, ...] = LEAD_ORDER
    notes: tuple[str, ...] = field(default_factory=tuple)


def validate(signal, *, fs_hz: int, units: str, leads, source: str, source_sha256: str) -> CanonicalECG:
    if fs_hz != FS_HZ:
        raise ContractError(f"fs_hz={fs_hz}; se exige {FS_HZ} (usar un adaptador de remuestreo probado)")
    if units != UNITS:
        raise ContractError(f"units={units!r}; se exige 'mV'")
    if tuple(leads) != LEAD_ORDER:
        raise ContractError(f"orden de derivaciones {tuple(leads)} != {LEAD_ORDER}")
    arr = np.asarray(signal)
    if arr.dtype.kind != "f":
        raise ContractError(f"dtype {arr.dtype}; se exige coma flotante")
    if arr.shape != (12, N_SAMPLES):
        raise ContractError(f"shape {arr.shape}; se exige (12, {N_SAMPLES})")
    if not np.all(np.isfinite(arr)):
        raise ContractError("la señal contiene NaN o Inf")
    peak = float(np.max(np.abs(arr)))
    if peak > MAX_ABS_MV:
        raise ContractError(f"amplitud máxima {peak:.1f} mV > {MAX_ABS_MV}: ¿unidades en µV?")
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise ContractError("source_sha256 debe ser SHA-256 hex en minúscula")
    if not source:
        raise ContractError("falta la procedencia (source)")
    notes = []
    flat = [LEAD_ORDER[i] for i in range(12) if float(np.ptp(arr[i])) < 1e-3]
    if flat:
        notes.append("derivaciones planas: " + ",".join(flat))
    return CanonicalECG(signal=arr, source=source, source_sha256=source_sha256, notes=tuple(notes))
