# Criterios objetivos de aceptación y rechazo

## Rechazo automático

1. Cambia el SHA de algo en `frozen/` o de un split.
2. Abre un conjunto reservado o `CONSUMED` para seleccionar modelo, umbral o hiperparámetro.
3. Produce resultados sin protocolo registrado en un commit anterior.
4. Compara con cobertura distinta sin reportar cobertura, MAE y errores graves juntos.
5. No reproduce (bit a bit o dentro de la tolerancia declarada) en una segunda ejecución.
6. Presenta etiquetas automáticas (MUSE, 12SL, ECGDeli, SCP) como verdad clínica.
7. Compara modelos en LUDB cuando alguno entrenó con los sujetos evaluados (ver `datasets/registry.yaml: ludb.cross_split_leakage`).

## Modelo de medición (QT, QRS, PR, T-end)

Sobre el conjunto preregistrado y los **mismos latidos** que la referencia vigente:

- IC95 (bootstrap por paciente) de la diferencia de MAE con límite superior ≤ 0, o ≤ margen de no inferioridad preespecificado (sugerido: +2 ms QRS, +5 ms QT); **y**
- errores >100 ms no aumentan (abstenciones cuentan como fallo en el denominador); **y**
- la cobertura no baja más de 2 puntos porcentuales; **y**
- ningún subgrupo predefinido (ritmo, QRS ancho, T de baja amplitud) empeora más del doble del margen; **y**
- latidos con referencia discordante (`REFERENCE_DISCORDANT`: marca DII a >60 ms de la mediana manual del resto de derivaciones) se reportan con y sin ellos, nunca solo sin ellos.

## Modelo de clasificación

Mejora AP (no solo AUROC) con IC que excluye 0, no empeora la calibración (Brier/ECE) y no reduce la sensibilidad en etiquetas graves.

## Software determinista

CI en verde, regresión numérica idéntica (o diferencias explicadas) y revisión de una IA distinta del autor.
