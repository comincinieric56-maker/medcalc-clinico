# VALIDACIÓN MEDCALC V6.1 — PEDIATRÍA EXPANDIDA

Fecha de revisión: 2026-09-03

## Cobertura

- Catálogo maestro: 618 medicamentos.
- Medicamentos con al menos una regla pediátrica automatizable: 80.
- Reglas pediátricas estructuradas: 246.
- MED-ID huérfanos: 0.
- Pruebas unitarias del motor: 15/15.

## Cambios del motor

V6.1 soporta:
- mg/kg/dosis y rangos.
- mg/kg/día y rangos.
- mcg/kg/día y rangos.
- unidades/kg/dosis y unidades/kg/día.
- dosis fijas y rangos fijos.
- máximo absoluto por dosis.
- máximo por kg por dosis.
- máximo absoluto diario.
- máximo por kg por día.
- límites de peso inclusivos/exclusivos.
- conversión de cualquier unidad compatible a mL cuando la regla lo permite.

## Nuevos grupos añadidos en V6/V6.1

Antiinfecciosos, alergia/respiratorio, gastroenterología, neurología, psiquiatría,
cardiovascular, endocrinología, hematología/ferropenia y analgesia.

Entre los medicamentos nuevos o ampliados:
clonazepam, clonidina ER, doxiciclina, escitalopram, esomeprazol, fexofenadina,
fluoxetina, furosemida, hidrocortisona, naproxeno, olanzapina, risperidona,
sulfato de magnesio, topiramato, valaciclovir, sulfato ferroso, dipirona y ketorolaco.

## Reglas de seguridad

- La ausencia de regla no significa contraindicación.
- No se extrapolan dosis entre indicaciones.
- Las reglas de especialidad se marcan como ESPECIALISTA/MONITORIZACIÓN o URGENCIA/ESPECIALISTA.
- Sulfato ferroso se calcula como hierro elemental.
- Dipirona conserva la dosis por kg y máximo diario del folleto ISP, pero no inventa un intervalo pediátrico que la fuente no especifica.
- Formulaciones especiales (p. ej., clonidina ER) no se intercambian mg por mg con otras formulaciones.
- La aplicación continúa siendo una herramienta de apoyo y requiere validación local antes de uso asistencial.
