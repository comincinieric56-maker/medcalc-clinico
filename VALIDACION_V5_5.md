# Validación MedCalc Clínico V5.5 — Renal 2025

## Fuente renal incorporada

- García Montemayor V, Sanchez-Agesta Martínez M, Naranjo Muñoz J.
- *Ajuste de Fármacos en la Enfermedad Renal Crónica*.
- Nefrología al Día, ID FR-001.
- Actualizado: 24-05-2025.
- PDF original incluido como `data/FR-001_Nefrologia_al_dia_2025.pdf`.

## Cobertura

- 26 tablas originales extraídas como imágenes fuente.
- 127 filas de las tablas 1–6 transcritas y verificadas visualmente contra el PDF (antibacterianos, antifúngicos, antiparasitarios, tuberculostáticos y antivirales).
- 342 filas candidatas indexadas por OCR en las 26 tablas para navegación; el OCR está marcado explícitamente como NO validado y NO se usa para dosificar.
- 46 de las 127 filas verificadas enlazadas directamente con un `MED-ID` del catálogo maestro actual de 618 medicamentos.
- Se conservan las 113 reglas renales automatizadas previas; la bibliografía 2025 es una capa separada y no reemplaza las reglas indicación-específicas.

## Seguridad de la lógica bibliográfica

- La banda de la tabla se selecciona por Cockcroft–Gault: `>=50`, `10–<50`, `<10 mL/min`.
- La app muestra literalmente la celda bibliográfica de la banda; no convierte porcentajes o intervalos en una prescripción nueva.
- Hemodiálisis se muestra como campo separado de suplemento HD.
- La capa bibliográfica está limitada a adultos; no se extrapola a pediatría.
- Foscarnet permanece como referencia de tabla por usar CCr expresado en mL/min/kg y ajustes separados de inducción/mantenimiento.

## Pruebas

`python -m pytest -q`

Resultado: **10 passed**.
