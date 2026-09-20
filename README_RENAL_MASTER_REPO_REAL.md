# MEDCALC RENAL MASTER — adaptación al repositorio real

Este paquete está preparado para el repositorio actual `medcalc-clinico`.

## NO reemplaza

- app.py
- supabase_repository.py
- repository.py
- medcalc_engine.py
- requirements.txt
- ajuste_renal.csv
- renal_biblio_ocr_indice.csv
- renal_biblio_verificada_2025.csv
- ninguno de los archivos ECG
- ninguna tabla/imagen renal existente

## Archivos nuevos

En la raíz del repositorio:
- `MEDCALC_RENAL_MASTER_BUILD.py`
- `MEDCALC_RENAL_MASTER_CATALOGO_1122.csv`
- `requirements_renal_master.txt`

Crear además:
- `.github/workflows/medcalc-renal-master.yml`

## Ejecución

GitHub -> Actions -> `MEDCALC Renal Master 1122` -> Run workflow.

La acción genera en:
`generated_renal_master/`

- `renal_master_evidence.csv`
- `renal_current_reference_ready.csv`
- `renal_needs_secondary_source_or_review.csv`
- `renal_master_summary.json`
- `MEDCALC_RENAL_MAESTRO_GENERADO.sql`

## Supabase

NO ejecutar ningún SQL hasta revisar primero:
`generated_renal_master/renal_master_summary.json`

y el CSV:
`generated_renal_needs_secondary_source_or_review.csv`.

Cuando la salida sea coherente, se ejecuta en Supabase:
`generated_renal_master/MEDCALC_RENAL_MAESTRO_GENERADO.sql`

## Seguridad

El proceso:
- conserva las reglas CURRENT_AUTO existentes;
- no borra bibliografía;
- no convierte un fuzzy match en pauta;
- no interpreta falta de evidencia como "sin ajuste";
- las nuevas coincidencias automatizadas se incorporan como CURRENT_REFERENCE,
  no como reglas numéricas automáticas.
