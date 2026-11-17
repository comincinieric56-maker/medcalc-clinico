# Checklist para PR que tocan ecg_consolidacion/

Copiar en la descripción del PR.

## Qué cambia

<!-- Una o dos frases. -->

## Autor y revisor

- Autor (IA o persona): 
- Revisor (debe ser una IA distinta del autor): 
- Aprobación final: @comincinieric56-maker

## Tipo de cambio

- [ ] Software determinista (ingesta, DSP, mediciones, reglas, informe)
- [ ] Modelo (nuevo checkpoint, umbral, calibración) → requiere protocolo preregistrado
- [ ] Registro / datos / protocolo → requiere aprobación del responsable
- [ ] UI / visualización

## Lista obligatoria (rechazo automático si falla alguna)

- [ ] Check `MEDCALC EKG consolidación (F0/F1)` en verde (registro, congelados, reservados, preregistro, pruebas).
- [ ] No cambia ningún SHA de `frozen/` ni de un split.
- [ ] No abre ningún conjunto reservado o `CONSUMED` para elegir modelo, umbral o hiperparámetro.
- [ ] Si hay resultados: el protocolo se registró con `tools/prereg.py` en un commit **anterior** (enlace: ).
- [ ] Las comparaciones reportan **cobertura + MAE + errores graves** sobre los **mismos latidos/registros**.
- [ ] Reproducible: segunda ejecución idéntica o dentro de la tolerancia declarada.
- [ ] Ninguna etiqueta automática (MUSE, 12SL, ECGDeli, SCP) se presenta como verdad clínica.
- [ ] Ningún modelo de lenguaje calcula mediciones ni diagnósticos.

## Solo para cambios de modelo de medición (ver docs/ACCEPTANCE_CRITERIA.md)

| Métrica | Referencia vigente | Candidato | Diferencia [IC95 por paciente] |
|---|---|---|---|
| Cobertura | | | |
| MAE (ms) | | | |
| Errores >100 ms | | | |
| Peor subgrupo predefinido | | | |
