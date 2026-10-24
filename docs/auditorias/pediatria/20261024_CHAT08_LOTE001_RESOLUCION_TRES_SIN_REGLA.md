# MEDECAL Pediatría — Lote 001, resolución documental de los tres últimos sin regla

**Fecha:** 2026-10-24. **Proyecto:** `medcalc-clinico` (`wqjgmsgcyocwpuwhmpmy`). **Lote:** `CHAT_PED_08_20261024_LOTE001`. **Rama:** `audit/pediatria-chat08-lote001-20261024`. Sin trabajo sobre UUID de otros chats.

## Conteos y carácter de la intervención

Consulta PREVIA: 1.122 medicamentos, 1.002 con alguna regla, **120 sin regla**. Los tres de nuestra cohorte en `IN_REVIEW` seguían sin ninguna fila `pediatric_rules`.

Se creó **una referencia documental no calculable por medicamento** mediante inserción transaccional condicionada a: UUID de nuestra reserva, `review_state=IN_REVIEW`, `pediatric_status=PENDING_REVIEW`, fuente preexistente asociada por `medication_sources` y ausencia de cualquier regla previa. Las tres filas siguen en `PENDING_REVIEW` y NO se consideran posologías validadas.

| MED-ID | UUID de medicamento | UUID de regla nueva | Tipo de evidencia y decisión |
|---|---|---|---|
| MED-1052 Oxamniquina | `1ce88b1b-9ae3-4054-83ac-1d5d7a619fa1` | `99a0ccd4-0c8c-44d5-aa27-ac236f7cb43c` | **Pauta pediátrica AEPED existente**, pero depende de grupo etario, área geográfica y susceptibilidad de `S. mansoni`; anotada como referencia no calculable, no «contraindicada en niños». |
| MED-1104 Temsirolimus | `8e892849-fdea-45f4-9c72-241348143791` | `0dd5c8d5-8666-4286-9029-dc99ac4c52ce` | **SIN PAUTA PEDIÁTRICA TERAPÉUTICA GENERAL ESTABLECIDA**: CIMA no recomienda su uso pediátrico, y DailyMed no ha establecido eficacia para tumores sólidos refractarios. Las dosis de ensayos no son pautas clínicas. |
| MED-1105 Tenipósido | `77f38b54-04c6-48d0-afb0-da043be308e4` | `b5b10c79-c1a4-445f-956d-1ae3aad8b76a` | Existe **uso pediátrico oncológico protocolizado**: FDA VUMON 2011 indica combinación para LLA infantil refractaria. No existe esquema aislado universal por kg; la pauta depende del protocolo, BSA, combinación y ciclos. |

## Comprobaciones de integridad y seguridad posteriores

- Inserción SQL: **3 filas creadas, 3 UUID distintos, 3 referencias textuales no calculables**; no se modificó ninguna regla previa.
- Todas las nuevas filas: `dose_type='REFERENCIA_TEXTO'`, `status='PENDING_REVIEW'`, `automatizable=false`, `allow_volume_conversion=false`, `calculation_hold_reason` obligatorio.
- Cantidades `dose_min`, `dose_max`, `fixed_dose_min`, `fixed_dose_max` **NULL en los tres**. Frecuencia, duración y precauciones están documentadas como texto contextual, **no parámetros de cálculo**.
- Se verificaron seis vínculos de procedencia en `pediatric_rule_sources`: un `PRIMARY` y un `CONTEXT` por regla. NIH MeSH en oxamniquina es contexto de identidad, **no fuente posológica**.
- `medication_module_status.pediatric_status` continúa `PENDING_REVIEW` en los tres. Ninguna regla se ha publicado ni validado.
- El último control posterior de Supabase devolvió **1.122 medicamentos, 1.012 con al menos una regla, 110 sin ninguna regla**. La disminución global desde 120 también incluye escrituras concurrentes de otros chats: **solo 3 de las nuevas reglas son nuestras**.
- Distinción crítica de métricas: de los 1.122 medicamentos, **678 tienen al menos una regla con cantidad numérica estructurada** y **334 tienen reglas pero ninguna cantidad numérica estructurada**. Una referencia textual disminuye el contador «sin regla» **sin aumentar la cobertura de dosificación numérica** y no debe llamarse validación clínica.

## Fuentes primarias y advertencias

**Oxamniquina:**
- AEPED, Pediamécum: https://www.aeped.es/pediamecum/generatepdf/api?n=83750
- Ficha de índice AEPED: https://www.aeped.es/comites/cm/pediamecum/principios-activos/oxamniquina
- Únicamente vía oral contra `S. mansoni`; se describen diferentes edades, áreas geográficas y tasas de resistencia. No introducir un único cálculo para todas las procedencias. Precaución convulsiones, mareo y azúcar de suspensión.

**Temsirolimus:**
- AEMPS/CIMA actual: https://cima.aemps.es/cima/dochtml/ft/07424001/FT_07424001.html
- DailyMed TORISEL, ficha actualizada diciembre 2025: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=95b7dc92-2180-42f1-8699-3c28f609e674
- CIMA no recomienda uso pediátrico en general y señala ineficacia para neuroblastoma, rabdomiosarcoma y glioma de alto grado evaluados. Dosis experimentales no equivalen a dosis recomendadas. No se ha afirmado contraindicación formal absoluta.

**Tenipósido:**
- Ficha FDA VUMON **histórica de 2011**, que advierte que puede no ser última versión: https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/020119s010s011lbl.pdf
- AEPED, Pediamécum: https://www.aeped.es/pediamecum/generatepdf/api?n=83392
- Requiere protocolo antineoplásico y combinación, vigilancia de mielosupresión grave, reacción anafilactoide y funciones renal/hepática. No usar dosis de monoterapia adulta.

## Continuidad y límites

Este expediente está **documentalmente reconciliado**, pero **NO clínicamente terminado**: quedan revisión farmacológica de las dosis por indicación y aprobación independiente antes de ofrecer cualquier cálculo. No se cambió el estado `IN_REVIEW` del lote ni los 112 bloqueos preventivos anteriores. No se reservó ningún UUID nuevo. Las otras conversaciones trabajan directamente en la misma base: recalcular siempre cifras antes de concluir.

En el lote de esta conversación hay ahora **100/100 medicamentos con al menos una regla estructural**, de las cuales **3 son exclusivamente referencias textuales no posológicas**. La etiqueta «pauta pediátrica documentada» solo aplica cuando la fuente y el esquema clínico están documentados por indicación; no usar este total como sinónimo de seguridad automática.
