# MEDCALC CLÍNICO V7.2 — VALIDACIÓN ESTRUCTURAL

Fecha: 2026-09-03

## Cambios principales

- Navegación directa Inicio → Pediatría / Renal / Toxicología conservando MED-ID.
- Buscador explícito en todos los módulos farmacológicos.
- Ajuste renal adulto sin peso ni talla:
  - CKD-EPI 2021 con edad + sexo + creatinina.
  - eGFR conocido.
  - estadio KDIGO G1–G5 como alternativa de orientación.
- G3a requiere eGFR exacto cuando una pauta usa corte 50 mL/min/1.73 m².
- No se convierten automáticamente reglas CrCl/Cockcroft-Gault a eGFR si la fuente original depende de esa métrica.
- Tabla `medication_module_status` para evitar estados clínicos vacíos.
- Vista `v_medcalc_coverage` para auditar qué falta revisar.
- Primera expansión prioritaria del catálogo: 48 medicamentos ausentes de alta utilidad clínica/urgencia.
- Alias iniciales: ALBUTEROL→SALBUTAMOL, ADRENALINA→EPINEFRINA, NORADRENALINA→NOREPINEFRINA.

## Seguridad de datos clínicos

Un medicamento sin pauta revisada permanece visible pero no genera una dosis. `PENDING_REVIEW` no equivale a contraindicación ni a ausencia demostrada de uso pediátrico/renal.
