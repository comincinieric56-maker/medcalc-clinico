# Pediatría PAR08 — tres cotejos adicionales
Fecha 2026-10-24. Reserva exclusiva: `AUD_PED_PAR08_20261024` (100 UUID).

| ID | Medicamento | Evidencia regulatoria | Decisión |
| --- | --- | --- | --- |
| MED-0004 | Abiraterona | [DailyMed ZYTIGA §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=4e338e89-3cf2-48eb-b6e2-a06c608c6513): seguridad y eficacia pediátricas no establecidas | Sin pauta pediátrica establecida para esta formulación |
| MED-0027 | Ácido fenofíbrico | [DailyMed TRILIPIX §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=3057fc9c-ff77-43eb-b687-776660423742): sin seguridad ni eficacia pediátrica establecidas, monoterapia o asociado a estatinas | Sin pauta pediátrica establecida para este producto |
| MED-0029 | Ácido ibandrónico | [DailyMed BONIVA §8.4](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=d6d7711f-541b-47d1-b937-2c6a021eb6c3): seguridad y eficacia pediátricas no establecidas | Sin pauta pediátrica establecida en esa ficha; no extrapolar usos especializados |

Las tres notas se escribieron en `medication_module_status.pediatric_note` solamente para los medicamentos previamente reservados, condicionadas a la ausencia del marcador de cotejo. SQL posterior confirma **10 de 100** con marcador `FUENTE_PRIMARIA_PED_PAR08_20261024`; quedan **90 por cotejar**. Ninguna dosis añadida ni regla habilitada. Esta evidencia no constituye validación clínica humana. El resto de las cohortes permanecen fuera de alcance.
