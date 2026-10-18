# Auditoría técnica inicial — Embarazo y Renal (2026-10-17)

## Embarazo: lote clínico PR #160

- Archivo: MEDCALC_PREGNANCY_CLINICAL_GUIDANCE_V1.csv.
- Filas de datos: 106; MED-ID únicos: 106; duplicados: 0.
- Estados contados con anclaje de campo: REVIEWED = 60 (incluye 9 filas originales sin comillas); PENDING_REVIEW = 46.
- Los registros REVIEWED son propuestas clínicas en rama; NO equivalen a validación clínica independiente, publicación Supabase o despliegue.
- Fuente/procedencia: numerosas filas usan URL genérica de DailyMed o campo source_url vacío; se requiere enlace específico verificable antes de promover a producción.
- No fusionar ni publicar automáticamente sin validación de fuentes y pruebas.

## Renal: auditoría de cierre pendiente

- Existe MEDCALC_RENAL_MASTER_CATALOGO_1122.csv y pipelines de múltiples fuentes.
- Verificar que cada uno de los 1.122 MED-ID tenga identidad exacta, cobertura por fuente, y si existe o no recomendación renal explícita; ausencia de etiqueta no equivale a ajuste validado.
- Validar indicaciones, eGFR/CrCl, diálisis, unidades, dosis y contraindicaciones con referencias verificables; no inferir ajustes por analogía.
- Confirmar funcionamiento app.py → supabase_repository.py → tablas reales, migraciones, RLS, tests y publicación efectiva antes de declarar el módulo terminado.

## Puertas de cierre

1. Identidad y cardinalidad del catálogo.
2. Proveniencia específica y accesible por pauta clínica.
3. Separación FDA histórico/PLLR/TGA y ausencia de trimestres inventados.
4. Validación de semántica clínica por indicación, vía y dosis.
5. Pruebas de regresión, CI verde y revisión de despliegue.
6. Supabase DRAFT hasta aprobación; sin cambios automáticos en producción.
