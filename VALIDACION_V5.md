# Validación técnica MedCalc Clínico V5

Fecha: 2026-09-02

## Controles ejecutados

- Compilación Python: `app.py`, `medcalc_engine.py`, `repository.py`.
- 8 pruebas unitarias del motor: aprobadas.
- Catálogo maestro: 618 medicamentos.
- Reglas pediátricas: 73; todos los `med_id` existen en catálogo.
- Reglas renales: 113; todos los `med_id` existen en catálogo.
- Fichas toxicológicas farmacológicas: 618.
- Prueba pediátrica: amoxicilina, 20 kg, 22.5 mg/kg/dosis q12h -> 450 mg/dosis, 900 mg/día.
- Prueba renal: aciclovir HSV no SNC, CrCl 40 mL/min -> banda 25–50 -> 5 mg/kg IV q12h.
- Conversión mg/mL y límites diarios: cubiertos por pruebas unitarias.

## Límites

El entorno de construcción no incluye Streamlit instalado, por lo que se verificó sintaxis, motor, integridad de datos y pruebas unitarias, pero no se abrió el servidor web dentro de este entorno.

Las reglas renales automáticas siguen siendo principalmente adultas. El módulo toxicológico de drogas/plaguicidas/metales y la tabla local de antídotos aún no tienen la misma auditoría registro por registro que los 618 medicamentos farmacológicos.
