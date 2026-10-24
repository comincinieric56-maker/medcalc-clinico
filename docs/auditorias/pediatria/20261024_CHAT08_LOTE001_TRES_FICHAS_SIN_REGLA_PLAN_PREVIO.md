# Plan de intervención previo — 3 fármacos sin regla, cohorte Chat08

Fecha: 2026-10-24. Reserva compartida `CHAT_PED_08_20261024_LOTE001`. Proyecto `wqjgmsgcyocwpuwhmpmy`. **No seleccionar otros UUID**.

## Contador previo observado en tiempo real

Global: **1.122 medicamentos; 1.002 con regla, 120 sin ninguna regla**. En la cohorte de 100 UUID de esta conversación permanecen 3 sin regla: `MED-1052`, `MED-1104`, `MED-1105`. Todos en `IN_REVIEW`, `PENDING_REVIEW`, con monografías previamente asociadas y notas clínicas. Las cifras son volátiles por ediciones de otros chats.

## Decisión por medicamento y base bibliográfica

| MED-ID | Principio | Conclusión clínica-documental | Fuente principal |
|---|---|---|---|
| MED-1052 | Oxamniquina | Existe **pauta pediátrica publicada** para `Schistosoma mansoni`; la dosis y duración varían por grupo etario, área geográfica y resistencia. No existe **pauta universal** segura que pueda hacerse calculable desde un único peso. Mantener revisión especializada y registrar referencia textual no calculable. No confundir con ausencia de uso pediátrico. | AEPed Pediamécum `https://www.aeped.es/pediamecum/generatepdf/api?n=83750` y `https://www.aeped.es/comites/cm/pediamecum/principios-activos/oxamniquina` |
| MED-1104 | Temsirolimus | **SIN PAUTA PEDIÁTRICA ESTABLECIDA** para indicaciones autorizadas generales; CIMA expresa no recomendación pediátrica por falta de eficacia en los tumores evaluados. Los regímenes de ensayos son **datos experimentales, no recomendaciones posológicas**. Registrar constancia de no pauta general no calculable. | AEMPS `https://cima.aemps.es/cima/dochtml/ft/07424001/FT_07424001.html`; DailyMed `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=95b7dc92-2180-42f1-8699-3c28f609e674` |
| MED-1105 | Tenipósido | **Existe uso pediátrico en protocolo de quimioterapia**, y la etiqueta FDA histórica de VUMON lo indica en combinación para LLA infantil refractaria. No hay un régimen único independiente de quimioterapia, superficie corporal y ciclos; registrar referencia textual hospitalaria sin dosificación automática. | FDA 2011 `https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/020119s010s011lbl.pdf`; AEPED `https://www.aeped.es/pediamecum/generatepdf/api?n=83392` |

## Operación permitida, después de este archivo

Insertar exactamente una fila `pediatric_rules` por medicamento, **solo** si en la misma operación se confirma su UUID reservado en `medcalc_evidence_audit_queue_v1`, `IN_REVIEW`, y que no se haya creado otra regla. Usar `REFERENCIA_TEXTO`, `PENDING_REVIEW`, `automatizable=false`, `allow_volume_conversion=false`, `dose_min/max` y `fixed_dose_min/max` NULOS, `interval_hours` y `doses_per_day` NULOS, `calculation_hold_reason` explícito y `source_id` verificado por asociación medicamento-fuente. Documentar formulación/indicación/precauciones en `clinical_notes`, sin extrapolar datos de otros principios.

- No pasar a `VALIDATED`/`PUBLISHED`; no modificar dosis existentes ni cambiar `pediatric_status`.
- El simple hecho de crear referencia textual eleva el contador estructural **pero NO constituye pauta numérica autorizada**. Reportar ambos contadores.
- Confirmar las 3 inserciones y fuentes por SQL, sus campos numéricos nulos, cero solapamiento de UUID y recálculo global. Mantener revisión abierta.
- Si hay cambios concurrentes o bloqueos existentes, abortar antes que duplicar filas.

Nota: se ha revisado el PDF FDA 2011 y la ficha AEMPS actual de temsirolimus. El PDF AEPED de oxamniquina fue consultado mediante extracto del buscador porque el servidor impidió la recuperación directa desde la herramienta; requiere cotejo final de especialista antes de cualquier pauta numérica. Pediamécum no sustituye ficha técnica autorizada.
