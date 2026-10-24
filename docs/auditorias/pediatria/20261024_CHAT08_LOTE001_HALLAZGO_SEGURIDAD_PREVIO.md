# Hallazgo previo a intervención — MEDECAL Pediatría, chat 08, lote 001

Fecha: 2026-10-24. Proyecto Supabase: medcalc-clinico (`wqjgmsgcyocwpuwhmpmy`). Lote existente: `CHAT_PED_08_20261024_LOTE001`. Se utiliza un lote **ya reservado** de 100 UUID, `review_state=IN_REVIEW`, sin selección adicional.

## Comprobaciones realizadas sobre datos vigentes

- Inventario global: 1.122 medicamentos; 895 con alguna regla pediátrica; 227 sin regla; 2.299 filas de `pediatric_rules`; 1.220 reglas en `PENDING_REVIEW`.
- Cohorte propia: 100 UUID únicos, 97 medicamentos con al menos una regla y 3 sin regla numérica; 52 únicamente con reglas no validadas/publicadas; 45 con al menos una regla VALIDATED o PUBLISHED (no necesariamente todas sus indicaciones validadas).
- Cotejo con reservas `AUD_PED_PAR08_20261024` y `AUD_PED_PAR_RENALCHAT_20261024`: **0 UUID compartidos** mediante intersección directa de `medication_id`.
- En la cohorte: 220 filas de reglas pediátricas; 137 `PENDING_REVIEW`; 83 `PUBLISHED`; 0 `VALIDATED`. De las 137 pendientes, 25 ya tienen `calculation_hold_reason` y **112 aún no**. Las 112 tienen tipo numérico reconocido, cantidad inicial positiva, unidad y fuente; cero de ellas tienen `automatizable=true`.
- El frontend `pediatric-core.js` en `medecal-web` revisa inicialmente `source_id`, `calculation_hold_reason`, modalidad, unidad, límites numéricos y condiciones demográficas, pero **no exige status VALIDATED/PUBLISHED ni automatizable=true** en `eligibility`. Esto **no prueba despliegue público de ese motor**, pero revela un riesgo de cálculo bibliográfico de regla editorial pendiente.

## Cambio conservador autorizado por seguridad

Registrar **exclusivamente** en reglas de los 100 UUID de este lote que cumplan:

`status='PENDING_REVIEW'`, `NULLIF(btrim(calculation_hold_reason),'') IS NULL`, y regla numérica vinculada a fuente primaria.

Nuevo texto estandarizado en `calculation_hold_reason`:
`AUD_PED08_20261024: regla en PENDING_REVIEW; no autorizar cálculo automático hasta conciliación clínica individual de indicación, edad, vía, dosis, máximos y fuente primaria.`

Sin cambios a cantidades `dose_min/max`, `fixed_dose_min/max`, límites de edad o peso, vía, fuente, indicación, frecuencia, duración, `status`, `automatizable`, estado del módulo o pautas de terceros. Las reglas con bloqueo previo **no se tocan**. Operación SQL condicional y de alcance exclusivo; verificar conteos y que el bloqueo guardado coincide.

Este es un **bloqueo preventivo reversible**, no una revisión clínica terminada. Su reversión requiere validación individual posterior; no quitar el bloqueo por lote.

## Tres fichas sin regla estructurada: hallazgos primarios

1. **MED-1052 OXAMNIQUINA:** Pediamécum incluye terapia oral pediátrica de `Schistosoma mansoni`, con esquema y duración **dependientes de edad y región de infección**. Por ello es incorrecto describir «sin pauta pediátrica conocida»; permanece sin regla parametrizada en MEDECAL. Monografía: https://www.aeped.es/pediamecum/generatepdf/api?n=83750. No autogenerar un rango universal.
2. **MED-1105 TENIPÓSIDO:** Pediamécum documenta pautas por protocolo oncológico y la ficha FDA VUMON (2011) incluye indicación con otros antineoplásicos en leucemia linfoblástica aguda infantil refractaria. Por tanto, **existe evidencia de uso pediátrico**, pero necesita reconciliación por ciclo, superficie corporal y combinación. Monografía: https://www.aeped.es/pediamecum/generatepdf/api?n=83392; ficha FDA: https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/020119s010s011lbl.pdf.
3. **MED-1104 TEMSIROLIMUS:** DailyMed TORISEL informa estudios pediátricos exploratorios, pero sin eficacia establecida para tumores sólidos avanzados recurrentes/refractarios; AEMPS/CIMA declara no recomendado para las indicaciones pediátricas evaluadas. Existe dato de ensayo, no pauta general aprobada. Fuentes: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=95b7dc92-2180-42f1-8699-3c28f609e674; https://cima.aemps.es/cima/dochtml/ft/07424001/FT_07424001.html.

## Evidencia pendiente

La presencia de un enlace monográfico no valida todas las pautas ni garantiza coincidencia de indicación/presentación. Se requieren revisiones uno a uno para las 137 reglas pendientes y sus variaciones. No se declara clínica ni operativamente terminado el lote; no se activan cálculos.

No correr GitHub Actions; se documentarán UUID afectados, antes/después y verificación SQL en otra ficha de auditoría de esta rama exclusiva.
