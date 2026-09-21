# MEDCALC RENAL GLOBAL MULTISOURCE V5

Esta es la pasada global pedida: **no trabaja por lotes ni una fuente por vez**.

Parte de la matriz ya generada de 1.122 MED-ID y toma todos los que todavía
necesitan una fuente actual. Para cada pendiente consulta, en la misma corrida:

- Drugs@FDA: identidad/aprobación del producto;
- Health Canada DPD: identidad por ingrediente y producto;
- Health Canada Product Monograph PDF: texto renal actual;
- EMA official Product Information: texto renal cuando el producto está
  disponible en el portal de EMA;
- `renal_biblio_verificada_2025.csv`;
- `ajuste_renal.csv`;
- `renal_biblio_ocr_indice.csv`.

Y consolida con lo ya producido por:
- MEDCALC;
- RxNorm/RxNav;
- DailyMed;
- openFDA.

## Archivos a subir

En la raíz del repo:
- `MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5.py`
- `README_RENAL_GLOBAL_MULTISOURCE_V5.md`

En:
`.github/workflows/`
- `medcalc-renal-global-multisource-v5.yml`

No reemplaza `app.py`, `requirements.txt` ni `supabase_repository.py`.

## Ejecutar

GitHub -> Actions -> **MEDCALC Renal GLOBAL Multisource V5** -> Run workflow.

No requiere clave para Health Canada ni EMA.

`OPENFDA_API_KEY` es opcional. Si ya existe como GitHub Secret se utiliza para
Drugs@FDA; si no existe, el script intenta la API pública sin clave.

## Resultados

`generated_renal_global_v5/`

- `renal_global_summary.json`
- `renal_global_matrix_1122.csv`
- `renal_global_resolved.csv`
- `renal_global_still_pending.csv`
- `MEDCALC_RENAL_GLOBAL_MULTISOURCE_FINAL_V5.sql`
- `MEDCALC_RENAL_GLOBAL_V5_VERIFICAR.sql`

## Regla clínica

Una coincidencia de producto o una mera mención de "renal impairment" NO cierra
el medicamento. Debe haber identidad farmacológica estricta y una conducta renal
explícita: ajuste/no ajuste, umbral CrCl/eGFR/GFR, no recomendado en deterioro
renal o pauta específica de diálisis.

`NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND` significa únicamente que las fuentes
oficiales recuperadas no contienen una pauta renal explícita. NO equivale a
"no requiere ajuste renal".

## SQL

No ejecutar el SQL final hasta revisar primero `renal_global_summary.json`.
