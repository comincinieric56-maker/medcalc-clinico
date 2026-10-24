# MEDECAL Pediatría — Chat 08 · Lote 001: actualización clínica y bloqueo preventivo

Fecha: 24 de octubre de 2026. Supabase `medcalc-clinico` (`wqjgmsgcyocwpuwhmpmy`), identificador exclusivo `CHAT_PED_08_20261024_LOTE001`.

**Estado del trabajo: AUDITORÍA ESTRUCTURAL Y MEDIDAS DE SEGURIDAD APLICADAS. NO VALIDACIÓN CLÍNICA COMPLETA.**

## Métricas recalculadas desde Supabase antes de la intervención

- **1.122 medicamentos** totales.
- **895 con alguna regla pediátrica** y **227 sin regla estructurada** (79,77 % de cobertura estructural).
- **2.299 reglas pediátricas**, de las cuales **1.220 permanecen en `PENDING_REVIEW`** (estado global inicial de esta intervención).
- En nuestra cohorte: **100 UUID distintos, 0 solapamientos** con `AUD_PED_PAR08_20261024` y `AUD_PED_PAR_RENALCHAT_20261024`, 100 filas en `IN_REVIEW`, 100 fichas con notas y monografías enlazadas.
- Distribución por presencia de regla: 97 con reglas y tres sin regla; 52 medicamentos tienen solamente reglas pendientes; 45 tienen al menos una regla publicada/validada, que **no garantiza** validación de todas sus otras indicaciones.

## Hallazgo de seguridad y remedio

En los **220 registros de reglas** correspondientes a los 100 medicamentos:

| Estado del lote | Reglas |
|---|---:|
| `PUBLISHED` | 83 |
| `PENDING_REVIEW` | 137 |
| `VALIDATED` | 0 |
| Pendientes con bloqueo de cálculo que ya existía | 25 |
| Pendientes sin bloqueo explícito antes de intervenir | 112 |
| **Nuevos bloqueos preventivos aplicados** | **112** |
| Pendientes sin bloqueo explícito después de intervenir | **0** |

Las 112 reglas afectadas corresponden a **43 medicamentos** y poseían fuente y cantidad numérica. Se insertó exclusivamente el campo `calculation_hold_reason` con el aviso:

> AUD_PED08_20261024: regla en PENDING_REVIEW; no autorizar cálculo automático hasta conciliación clínica individual de indicación, edad, vía, dosis, máximos y fuente primaria.

Cada fila fue seleccionada por pertenencia a los UUID de nuestra reserva, `status='PENDING_REVIEW'`, fuente y unidad no nulas, dosis inicial positiva y `calculation_hold_reason` previamente vacío. Se ejecutó `FOR UPDATE OF r SKIP LOCKED` en una sentencia SQL para no competir por filas bloqueadas.

**Verificación posterior:** 112/112 bloqueos, 43 medicamentos, mismo texto normalizado. El snapshot preintervención y la relectura posterior comprobaron igualdad exacta en **112/112 reglas** de UUID de regla, UUID de medicamento, ID canónico, modalidad, indicación documental, unidades, dosis mínima/máxima y fija, intervalo, dosis/día, máximos, límites de edad, vía, fuente, estado y bandera editorial. No se modificó ninguna dosis ni se habilitó automatización.

La comprobación sintética del módulo `public/medcalc/pediatric-core.js` de `medecal-web` confirmó que una regla numérica `PENDING_REVIEW` con fuente y `automatizable=false` podría ser elegible sin bloqueo, y que al añadir `calculation_hold_reason` el motor la rechaza (`CLINICAL_HOLD`). **No se ha confirmado cuál versión de frontend se encuentra desplegada**.

### Riesgo residual de reglas publicadas

Las 83 reglas `PUBLISHED` no se alteraron. Hay **40 banderas editoriales `automatizable=true`** en ese subconjunto. La revisión estructural detectó **73 sin al menos uno de los dos límites etarios**, **59 sin máximo numérico**, **34 sin frecuencia estructurada** y **5 sin vía**. Estas carencias **no demuestran por sí solas** que las pautas sean incorrectas, pero obligan a reconciliar la fuente, la formulación y el algoritmo de cálculo antes de afirmar seguridad. No se deshabilitaron masivamente reglas publicadas, ni se elevaron sus estados.

## Tres fichas con ausencia de regla en MEDECAL

| ID | Medicamento | Resultado de contraste de fuente primaria | Actuación |
|---|---|---|---|
| `MED-1052` | Oxamniquina | **SÍ hay pauta pediátrica en Pediamécum** para `Schistosoma mansoni`, dependiente de edad y geografía. No es correcto afirmar ausencia absoluta de pauta pediátrica. | Ampliada `pediatric_note`; **sin crear dosis** y sin cálculo. |
| `MED-1104` | Temsirolimus | DailyMed informa ensayos pediátricos y eficacia no establecida en tumores sólidos recurrentes/refractarios; CIMA desaconseja uso pediátrico en las indicaciones específicas señaladas por falta de eficacia. | Ampliada `pediatric_note`; sin convertir dosis de ensayo en recomendación. |
| `MED-1105` | Tenipósido | **SÍ hay pauta pediátrica** por protocolo en Pediamécum y etiquetado FDA histórico en leucemia linfoblástica aguda refractaria infantil, combinado con otros antineoplásicos. | Ampliada `pediatric_note`; sin crear dosis protocolizada general. |

Fuentes y vigencia:
- Oxamniquina AEPED: https://www.aeped.es/pediamecum/generatepdf/api?n=83750
- Tenipósido AEPED: https://www.aeped.es/pediamecum/generatepdf/api?n=83392
- Tenipósido VUMON ficha FDA **histórica 2011**: https://www.accessdata.fda.gov/drugsatfda_docs/label/2011/020119s010s011lbl.pdf
- Temsirolimus TORISEL DailyMed: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=95b7dc92-2180-42f1-8699-3c28f609e674
- Temsirolimus TORISEL AEMPS/CIMA: https://cima.aemps.es/cima/dochtml/ft/07424001/FT_07424001.html

`pediatric_status` permanece en `PENDING_REVIEW` para los tres y todavía no se creó ningún registro en `pediatric_rules` para ellos.

## Archivos generados en esta rama (sin cambios a main)

1. `20261024_CHAT08_LOTE001_HALLAZGO_SEGURIDAD_PREVIO.md` — hallazgo registrado ANTES de la actualización.
2. `20261024_CHAT08_LOTE001_SNAPSHOT_PRE_BLOQUEO.json` — instantánea reversible de las 112 reglas.
3. `20261024_CHAT08_LOTE001_INVENTARIO_100_UUID.csv` — lista individual de 100 UUID, ID canónico, nombres, fuentes, estados, recuentos y advertencias.
4. Este informe — resumen de la conciliación, verificación y riesgos sin resolver.

Se mantuvieron los medicamentos en `IN_REVIEW` y `PENDING_REVIEW` según su nivel original. No se modificaron estados de publicación, indicaciones, dosis ni intervalos. Ninguna regla nueva fue activada para cálculo.

## Criterios clínicos pendientes

- Los **52 medicamentos que solo tienen reglas pendientes** necesitan reconciliación individual de indicación, edad gestacional/posnatal cuando proceda, peso, vía, presentación, concentración, frecuencia, duración, máximos, monitorización y ajustes especiales. El bloqueo se levanta solo regla por regla tras revisión.
- Los **45 con una o más reglas publicadas** pueden incluir reglas adicionales pendientes; una regla publicada no certifica el resto, y las 40 banderas editoriales automáticas preexistentes necesitan auditoría independiente.
- Los **tres sin regla** requieren resolución diferenciada: oxamniquina con datos AEPED variables por zona; tenipósido por protocolos de oncología; temsirolimus sin eficacia establecida en los contextos indicados. El resultado final no puede reducirse a «sin dosis» para los tres.
- El lote tiene **cobertura documental 100/100**, pero **0 fichas nuevas clínicamente validadas por este chat**. Las 100 conservan tareas de verificación farmacológica en diverso grado.

**Resultado operativo del chat: 112 bloqueos de riesgo preventivos y tres notas clínicas basadas en fuentes añadidas; 0 dosis inventadas, 0 automatizaciones nuevas, 0 UUID duplicados con las reservas conocidas.**

No se ejecutó GitHub Actions ni se confirmó despliegue de producción.
