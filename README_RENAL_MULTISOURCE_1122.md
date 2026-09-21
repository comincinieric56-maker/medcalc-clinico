# MEDCALC RENAL MULTISOURCE 1122

Sí: esta versión hace **una sola ejecución** para todo el catálogo.

## Fuentes/capas usadas en la misma cadena

1. MEDCALC existente: reglas y bibliografía ya validadas.
2. RxNorm/RxNav: identidad farmacológica.
3. DailyMed SPL: ficha regulatoria actual.
4. openFDA Drug Labeling: segunda ingestión oficial del etiquetado FDA.
5. Drugs@FDA/openFDA harmonization: corroboración de producto/identidad.
6. `renal_biblio_verificada_2025.csv`: soporte local secundario.
7. `ajuste_renal.csv`: soporte local secundario.

Las fuentes locales NO se convierten por sí solas en pauta actual.

## Archivos nuevos a subir

Raíz:
- MEDCALC_RENAL_MASTER_OPENFDA_V4.py
- MEDCALC_RENAL_MULTISOURCE_FINALIZE.py

Si alguno falta también deben existir en raíz:
- MEDCALC_RENAL_MASTER_BUILD.py
- MEDCALC_RENAL_MASTER_VALIDATE_V2.py
- MEDCALC_RENAL_MASTER_IDENTITY_V3.py
- MEDCALC_RENAL_MASTER_CATALOGO_1122.csv

Workflow:
- `.github/workflows/medcalc-renal-multisource-1122.yml`

## Una sola ejecución

GitHub -> Actions -> `MEDCALC Renal MULTISOURCE 1122` -> Run workflow.

El workflow reutiliza V1/V2/V3 si ya existen, por lo que en el estado actual
no repetirá innecesariamente la primera corrida de ~30 minutos. Sí hará la
pasada openFDA de todos los pendientes y luego consolidará todo.

## Resultado final

`generated_renal_multisource/`

- renal_multisource_summary.json
- renal_multisource_matrix_1122.csv
- renal_multisource_resolved.csv
- renal_multisource_pending.csv
- MEDCALC_RENAL_MULTISOURCE_FINAL.sql

NO ejecutar el SQL hasta revisar `renal_multisource_summary.json`.
