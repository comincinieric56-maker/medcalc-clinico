# MEDECAL Pediatría — L091: continuidad de 100 UUID exclusivos y auditoría de evidencia

**Fecha de corte: 24-10-2026.** Supabase: `wqjgmsgcyocwpuwhmpmy`. Repositorio solicitado: `comincinieric56-maker/medcalc-clinico`.  
**Alcance:** continuar la reserva previa **`PED_LOTE_091_20261024_RESERVED`**, sin reservar un nuevo lote ni tocar medicamentos de otros grupos.

## Selección exclusiva y protección de concurrencia

El lote se había reservado transaccionalmente con `FOR UPDATE OF q SKIP LOCKED` sobre `medcalc_evidence_audit_queue_v1`, limitado a medicamentos sin regla y con fuente ya identificada, usando `issue_codes` como marcador duradero.

Nueva consulta al retomar:
- **100 medicamentos/100 UUID** de `medications.id`, cada uno con al menos una ficha `pediatric_rules`.
- **82** fichas creadas por este hilo, **18** creadas concurrentemente durante la fase anterior por otras conversaciones; no se duplicaron ni sobrescribieron.
- Ninguno de esos 100 contiene en `medication_module_status.pediatric_note` `AUD_PED_PAR08_20261024` ni `AUD_PED_PAR_RENALCHAT_20261024`: **0 solapamientos comprobados** con ambas reservas conocidas.
- Se conservó `review_state='IN_REVIEW'`, `status='PENDING_REVIEW'`, `automatizable=false`, sin dosis numéricas y sin pasar filas a `PUBLISHED`.
- Para las fichas que se cotejaron individualmente se añadió a la propia cola el código `PED_L091_FUENTE_PRIMARIA_COTEJO_PUNTUAL_20261024`, **sin cambiar `review_state` a `REVIEWED` ni sellar validación clínica**.

## Informe de revisión del centenar

| Resultado comprobado | Total de medicamentos |
|---|---:|
| Reservados, UUID únicos | **100** |
| Con ficha documental de referencia | **100** |
| Creados directamente por este hilo | **82** |
| Creados por otros hilos antes del alta | **18** |
| **Cotejo puntual contra sección pediátrica regulatoria** | **37** |
| Con pauta clínica completa validada/autorizada para calculadora | **0** |
| Sin cotejo individual completo de dosis en este lote | **63** |
| Estado clínico `PENDING_REVIEW` | **100** |
| `PUBLISHED` | **0** |
| Automatizables | **0** |
| Conflicto UUID conocido | **0** |

**Interpretación obligatoria:** «Ficha documental» ≠ «Pauta pediátrica acreditada» ≠ «Validación médica final» ≠ «Autorización para dosificación automática». El cierre estructural de cien registros **NO es cierre clínico de cien dosis**.

### Medicamento con pauta PEDIÁTRICA de producto concreto identificada, sin cálculo

- **Oxibato sódico** — [DailyMed XYREM §2.2](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=926eb076-a4a8-45e4-91ef-411f0aa4f3ca): narcolepsia ≥7 años, solución oral en **dos tomas nocturnas** separadas 2,5–4 h y dosis inicial, titulación y máximos por bandas de peso. La frecuencia textual registra las bandas **20–<30, 30–<45 y ≥45 kg**, sin confundir techo inicial con indicación automática. No extrapolar a formulaciones de liberación prolongada; riesgo de depresión respiratoria, sedación y abuso. Bloqueo de cálculo mantenido.

### Sin pauta general establecida para la FORMULACIÓN o indicación regulatoria examinada

Cotejados en ficha primaria:
- **Topotecán IV** — [DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=996430c0-ee26-4ed9-82b8-a8b645becba6): seguridad y eficacia no establecidas en ficha FDA pediátrica; tratamientos infantiles de oncología dependen de protocolos específicos, no del régimen adulto.
- **Paroxetina** — [PAXIL DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ef3b5cbe-f9e1-c1ac-79da-cfe14e3a7e7e): sin eficacia/seguridad pediátrica establecida en el producto; ensayos de depresión mayor sin beneficio confirmado, alerta de suicidabilidad.
- **Molnupiravir** — [LAGEVRIO SmPC](https://www.medicines.org.uk/emc/product/13044/smpc): no se establece pauta para menores de 18 años.
- **Vareniclina** — [DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=e8a06e8e-1938-412c-a98a-28f0a29fe3ac): no recomendada en edades ≤16 años por falta de eficacia demostrada; dosis de ensayo no autorizan pauta.
- **Midostaurina** — [RYDAPT DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=11fa3fc9-6776-49a6-b1c1-653f627c3e58): seguridad/eficacia no establecidas; estudios pediátricos con formulación no aprobada y mielosupresión grado 4 prolongada asociada a combinación.
- **Mirtazapina** — [DailyMed Pediatric Use](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=310d8699-2c46-44e8-8c5f-79f01517a4a6): 258 pacientes en estudios de depresión mayor sin respaldo para indicación aprobada; aumento ponderal relevante y vigilancia de suicidabilidad. Dosis de investigación NO equivalen a pauta.
- **Nebivolol** — [DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=a9bad35e-cd48-48f1-93ca-46490dbf70a8): no establecido por ausencia de caracterización suficiente del desarrollo; datos de ratas jóvenes no autorizan calcular mg/kg infantil.
- **Neratinib** — [NERLYNX DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=d1e41dcf-e82a-47c2-a0ad-6c6eef621834): sin eficacia y seguridad establecidas en pediatría; diarrea/deshidratación y hepatotoxicidad requieren especialista.
- **Osimertinib** — [TAGRISSO DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=5e81b4a7-b971-45e1-9c31-29cea8c87ce7): sin eficacia/seguridad establecidas; vigilar neumonitis, QTc y toxicidad oncológica, sin copiar pauta de cáncer pulmonar adulto.
- **Palbociclib** — [IBRANCE DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=fecbdd7d-b729-41b5-9872-231b8fe104ce): tres estudios infantiles sin demostrar eficacia; no transformar rangos de ensayo en dosis prescritas.
- **Pomalidomida** — [POMALYST DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=2b25ef01-5c9e-11e1-b86c-0800200c9a66): sin eficacia/seguridad pediátricas establecidas en dos cohortes; riesgo teratogénico y mielosupresor.
- **Ranolazina LP** — [DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=5b254a8e-db88-49be-bb48-e6d2847f8ab3): sin evidencia de eficacia/seguridad pediátrica; advertencia QTc e interacciones; formulación de liberación prolongada.
- **Ribociclib** — [Kisqali 200 mg SmPC §4.2, §5.1](https://www.medicines.org.uk/emc/product/8110/smpc): sin pauta <18; combinación experimental con topotecán/temozolomida con toxicidad limitante y sin establecer dosis recomendada. Diferenciar estudio de pauta aprobada.
- **Venetoclax** — [VENCLEXTA DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b118a40d-6b56-cee3-10f6-ded821a97018): sin eficacia/seguridad establecidas; dosis animales no convierten a mg/kg humanos; síndrome de lisis tumoral.
- **Vildagliptina** — [SmPC §4.2](https://www.medicines.org.uk/emc/product/101149/smpc): no recomendada <18 por ausencia de datos; no clasificar como contraindicación formal por edad.
- **Vericiguat** — [VERQUVO DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=17056d73-1b1b-4bf2-9c07-b7a9367f0d6d): eficacia/seguridad no establecidas en menores; no extrapolar titulación de insuficiencia cardiaca adulta.
- **Vortioxetina** — [TRINTELLIX DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=cb1e9df4-dd74-4073-9d78-bb55593dd000): en ensayos con 540 niños y 616 adolescentes no superó placebo, sin indicación pediátrica establecida; advertencia suicidabilidad.


### Tres cotejos regulatorios adicionales en este corte

- **Pentosán polisulfato sódico (ELMIRON 100 mg cápsulas VO)** — [DailyMed Pediatric Use](https://dailymed.nlm.nih.gov/dailymed/fda/fdaDrugXsl.cfm?setid=c7417e03-5812-4f35-8433-6062d68420fc&type=display): seguridad y eficacia **no establecidas bajo 16 años**. La fuente no aporta pauta infantil universal de 16–17 años. No deducir contraindicación formal de ese dato ni copiar la pauta adulta; vigilar potencial sangrado y toxicidad macular en contexto especializado.
- **Prucaloprida (MOTEGRITY VO)** — [DailyMed §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=bca1e4c6-4b2b-4876-8ab3-654629c4ff1b): seguridad/eficacia pediátrica no establecidas; ensayo de estreñimiento funcional de **6 meses a 17 años** detenido por futilidad tras respuestas similares a placebo. No convertir dosis del ensayo ni la pauta de estreñimiento crónico adulto en regla pediátrica.
- **Triazolam (HALCION comprimido 0,125 mg)** — [AEMPS CIMA §4.2](https://cima.aemps.es/cima/dochtml/ft/58118/FT_58118.html): explícitamente **no establecidas seguridad/eficacia en menores de 18 años**. Dosis y duración de la ficha son de adultos; riesgos SNC/respiratorios y dependencia. No confundir ausencia de datos con contraindicación formal universal.

Estos tres expedientes conservan **`REFERENCIA_TEXTO`**, `PENDING_REVIEW`, sin dosis numéricas ni cálculo automático. El cotejo quedó también registrado en `issue_codes` de la cola del lote.

**33 casos con ausencia de pauta general para el producto/indicación cotejados; 3 con pauta textual específica de producto y 1 con restricción de formulación (<12 años) sin pauta individual extraída**. No extrapolar estas conclusiones a todos los usos off-label, presentaciones o jurisdicciones. **Ninguno se convirtió a dosis estructurada automática.**


## Nueva tanda de 16 cotejos regulatorios (continuación L091)

Se contrastaron individualmente doce fuentes regulatorias en un primer subgrupo y cuatro adicionales, incorporadas en `pediatric_rules` y en `medcalc_evidence_audit_queue_v1.issue_codes`. Se preservaron **los mismos UUID** y el estado `IN_REVIEW` / `PENDING_REVIEW`, `automatizable=false`, sin dosis estructuradas nuevas, sin publicar.

| Medicamento | Evidencia del producto y decisión segura |
|---|---|
| **Metocarbamol** | [DailyMed](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=425abc3b-2111-4fac-bf27-b685d1e657af): seguridad/eficacia **no establecidas en menores de 16 años** para comprimidos. No se infirió dosis/kg. |
| **Metoxaleno** | [eMC Methoxsalen G.L. Pharma §4.2](https://www.medicines.org.uk/emc/product/15562/smpc): solución **extracorpórea**, fotoféresis especializada; seguridad y eficacia pediátricas no establecidas. No equiparar a metoxaleno oral. |
| **Misoprostol** | [eMC CYTOTEC §4.2](https://www.medicines.org.uk/emc/product/1642/smpc): uso infantil no evaluado para úlcera péptica ni úlcera asociada a AINE. **No extrapolar esa conclusión a protocolos de salud reproductiva adolescente**. |
| **Nortriptilina** | [eMC 10 mg §4.2, 4.4](https://www.medicines.org.uk/emc/product/13903/smpc): no debe usarse **para depresión** en menores de 18 años según la presentación británica; otros usos off-label requieren evaluación distinta. |
| **Oritavancina** | [DailyMed ORBACTIV §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ff09a726-9f9b-4e30-b509-396781293220): eficacia/seguridad menores de 18 años no establecidas para ORBACTIV. Se preservó la **monografía AEPED** como fuente principal y se incorporó la etiqueta FDA como complemento; no negar protocolos off-label. |
| **Riluzol** | [eMC 50 mg](https://www.medicines.org.uk/emc/product/5185/smpc): no recomendado en niños/adolescentes para enfermedades neurodegenerativas por falta de datos. |
| **Trandolapril** | [eMC 0,5 mg](https://www.medicines.org.uk/emc/product/8173/smpc): no debe administrarse a menores por experiencia insuficiente; no trasladar ajuste renal adulto. |
| **Testosterona undecanoato** | [eMC depot IM 1000 mg/4 mL](https://www.medicines.org.uk/emc/product/100812/smpc): no indicado ni estudiado en varones <18 para **esa presentación depot**. No equivale a prohibición de todas las formulaciones de testosterona en endocrinología pediátrica. |
| **Trametinib** | [eMC MEKINIST comprimidos](https://www.medicines.org.uk/emc/product/7792/smpc): sin seguridad/eficacia pediátrica establecida en **la ficha de comprimidos revisada**. No extrapolar ni invalidar otra solución oral o combinación con su propia autorización pediátrica. |
| **Zofenopril** | [AEMPS ZOFENIL §4.2](https://cima.aemps.es/cima/dochtml/ft/62678/FT_62678.html): no recomendado en menores de 18 años por ausencia de seguridad/eficacia establecidas. |
| **Zopiclona** (catalogada «ZOPLICONA») | [eMC 3,75 mg](https://www.medicines.org.uk/emc/product/101310/smpc): no debe usarse <18 años; riesgo de sedación y dependencia. La grafía del catálogo deberá conciliarse. |
| **Pseudoefedrina clorhidrato** | [eMC Care Decongestant 60 mg §4.2–4.3](https://www.medicines.org.uk/emc/product/4900/smpc): **pauta específica en mayores de 12 años**, 60 mg por vía oral cuatro veces al día; **contraindicado en menores de 12** para esos comprimidos. No extrapolar a otras concentraciones ni combinaciones, ni autorizar calculadora antes de cotejo clínico. |
| **Nimesulida** | [AULIN 100 mg, reproducción de SPC eslovaco](https://www.adc.sk/databazy/produkty/spc/aulin-100-mg-tablety-411261.html): adolescentes **12–18 años**, misma pauta que adulto (100 mg VO dos veces/día), curso **máximo 15 días**; menores de 12 contraindicados. Segunda línea; hepatotoxicidad, hepatopatía o ClCr <30 mL/min contraindicados. **No implica autorización local en Chile**. |
| **Podofilotoxina** | [eMC WARTICON solución](https://www.medicines.org.uk/emc/product/1506/smpc): seguridad y eficacia tópicas no establecidas <18 años para condilomas; se conserva AEPED como referencia adicional, sin dosis extrapolada. |
| **Reboxetina** | [eMC EDRONAX](https://www.medicines.org.uk/emc/product/1578/smpc): no debe utilizarse para depresión <18; suicidabilidad y falta de seguridad de desarrollo a largo plazo. Monografía AEPED conservada. |
| **Trifluoperazina** | [eMC comprimidos 1 mg](https://www.medicines.org.uk/emc/product/101101/smpc): esta presentación **no es adecuada <12 años**; la ficha recomienda líquido para esos niños. No documenta en §4.2 una dosis adolescente diferenciada. No significa que la molécula esté globalmente contraindicada. |

**Distribución acumulada L091: 37 cotejos documentales (3 con pauta textual de producto, 33 con ausencia de pauta general respaldada para esa presentación/indicación y 1 restricción de formulación). Quedan 63 fichas para cotejo individual completo.**

Ningún régimen textual se convirtió en `dose_min`, `fixed_dose_min`, `max_single` o `max_daily`, ni se habilitó `automatizable`. Los casos **nimesulida, pseudoefedrina y oxibato sódico** contienen dosis textuales para productos, edades y vías concretas, pero **no son calculadoras clínicas aprobadas**. La evidencia ajena a Chile exige conciliación regulatoria local.

### Hallazgo adicional: automatizaciones históricas publicadas

Consultadas fuera del lote, **sin modificarlas**:
- Reglas pediátricas `PUBLISHED AND automatizable=true`: **693**.
- De ellas, **322** no tienen `max_single`, `max_daily`, `max_single_per_kg` ni `max_daily_per_kg` estructurados.
- **151** no tienen `age_min_months` ni `age_max_months` estructurados.
- **0** carecen de fuente y **0** carecen de unidad/tipo de dosis o dosis inicial.

**Son señales de auditoría, no errores clínicos demostrados.** Los máximos o límites podrían estar en texto, protocolo, modelo de producto o regímenes donde no aplican. Evaluar antes de cambiar automatización ajena; no bloquear globalmente por ausencia de datos estructurados sin revisar cada medicamento.

## Recuento global de Supabase tras los cambios

| Indicador | Último corte |
|---|---:|
| Medicamentos activos | **1.122** |
| Sin fila de regla pediátrica | **110** |
| Sin ninguna regla `PUBLISHED` | **733** |
| Sin enlaces de fuentes **globales** al medicamento | **21** |

Estos indicadores se modifican durante las siete conversaciones restantes. No atribuir el descenso de faltantes globales a este hilo: el trabajo acumulado desde la reserva **añadió y verificó 37 cotejos regulatorios**, de los cuales 16 se realizaron en la presente continuación, sin crear reglas nuevas. Las 82 nuevas y 18 concurrentes ya estaban registradas al iniciar esta ronda.

## Próximos pasos para el mismo lote

De los 100 UUID, **63 necesitan revisión clínica individual más profunda** para establecer pauta real o ausencia fundamentada por producto. Priorizar moléculas con pauta pediátrica dependiente de indicación/protocolo (milrinona, octreótido, ritonavir, somatropina, sirolimus, vasopresores, antineoplásicos, etc.); no reemplazar las monografías AEPED por registros vacíos o falsas contraindicaciones. Si la fuente solo documenta vía/formulación sin posología infantil, mantenerla como ficha documental explícita y bloqueo de cálculo.

Antes de declarar `REVIEWED` o `VALIDATED`: verificar dosis inicial y máxima, edad, peso, concentración, vía, número de administraciones, duración, seguimiento, restricciones de uso y concordancia regulatoria.

## Inventario reproducible

[`2026-10-24-l091-100-fichas-y-verificacion.csv`](./2026-10-24-l091-100-fichas-y-verificacion.csv): 100 medicamentos, ambos UUID, clasificación de procedencia y cotejo, fuente URL, texto íntegro de ficha, datos estructurados, estado y bloqueo. Los 37 cotejos regulatorios quedan individualizados en el CSV.

No se ejecutaron GitHub Actions. No se tocaron Renal ni Toxicología. No se fusiona automáticamente esta rama con `main`, a fin de preservar el control de integración paralela solicitado.
