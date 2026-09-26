# MEDCALC ECG · Foto/PDF → U-Net → R27

## Flujo visible para el usuario

La app no solicita WFDB ni ECG digital.

```
Foto / PDF
  -> Open ECG Digitizer U-Net
  -> identificación de derivaciones
  -> señal canónica 500 Hz
  -> control de cobertura temporal
  -> 100 Hz derivado sólo si la señal está completa
  -> R27
  -> 35 probabilidades
```

## Digitalizador

MEDCALC incluye una copia vendorizada y fijada de Open ECG Digitizer:

- upstream: Ahus-AIM/Open-ECG-Digitizer
- commit: 97a15087d4abcda843da8c58ee74b1d8f47e6f9a
- licencia: CC BY-SA 4.0
- U-Net de segmentación SHA-256:
  17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b
- U-Net de identificación de derivaciones SHA-256:
  840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2

Los pesos están incluidos en `ecg_digitizer_assets/`; no se descargan en tiempo de ejecución.

## Regla fail-closed para R27

R27 permanece congelado sobre una señal de 10 s y 12 derivaciones.

MEDCALC sólo ejecuta R27 cuando el digitalizador demuestra:

- 12 derivaciones estándar;
- 5000 muestras observadas y finitas por derivación a 500 Hz;
- ninguna porción no impresa inventada, repetida o imputada.

Un impreso convencional 3×4 suele contener sólo una fracción temporal de muchas
derivaciones. En ese caso el ECG se digitaliza, pero el resultado se marca
`DIGITIZED_ONLY` y R27 no recibe una señal artificial.

## Memoria

El U-Net se ejecuta en un subprocess separado. Ese proceso termina antes de arrancar
el subprocess de R27, evitando mantener ambos stacks neuronales simultáneamente en RAM.

## R27

R27 conserva:

- `RESEARCH_PROBABILITY_ONLY_RELEASE`
- 35/35 salidas de probabilidad;
- 0 thresholds desplegables;
- 0 clasificaciones binarias;
- 0 etiquetas diagnósticas;
- `CLINICAL_DEPLOYMENT_BLOCKED`.

## Secret requerido

Como el runtime R27 está guardado en un repositorio privado, Streamlit necesita un
token GitHub de solo lectura limitado a `medcalc-r27-backend`:

```toml
R27_GITHUB_TOKEN = "github_pat_..."
```

No se requiere Render, `ECG_R27_API_URL` ni `ECG_R27_API_TOKEN`.

## Validación CI

`.github/workflows/medcalc-ecg-runtime-smoke.yml` verifica:

1. instalación del runtime científico;
2. SHA y strict-load de ambos U-Net;
3. inicialización del stack real;
4. una fotografía real de ECG atravesando el worker U-Net;
5. forma canónica 5000×12;
6. importación de los módulos MEDCALC.
