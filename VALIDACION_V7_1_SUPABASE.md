# MEDCALC V7.1 — SUPABASE

## Arquitectura

- 618 medicamentos: Supabase/PostgreSQL.
- Reglas pediátricas: Supabase/PostgreSQL.
- Reglas renales automáticas: Supabase/PostgreSQL.
- Bibliografía renal 2025: Supabase/PostgreSQL.
- Toxicología farmacológica: Supabase/PostgreSQL.
- Fuentes: Supabase/PostgreSQL.
- RLS: la app pública usa Publishable Key y solo lee registros PUBLISHED.

## Respaldo temporal

`medcalc.db` se conserva únicamente para:
- drogas/plaguicidas/metales;
- fichas de antídotos.

Estos dos submódulos no formaron parte del primer esquema Supabase y se migrarán después de su auditoría clínica.

## Seguridad

La app no utiliza ni necesita Secret/Service Role Key. Streamlit usa exclusivamente:

- SUPABASE_URL
- SUPABASE_PUBLISHABLE_KEY

La Secret Key se usó únicamente durante la migración administrativa en Colab.

## Nota toxicología

El primer esquema Supabase no incluyó los campos numéricos `umbral_mgkg_automatizable`, `etiqueta_umbral` y `nivel_evidencia`. V7.1 no intenta inferirlos desde texto. Hasta aplicar el parche toxicológico V1.1, la comparación automática mg/kg permanece deshabilitada en Supabase por seguridad.
