# MEDCALC ECG · FOTO/PDF → U-Net → R27

## Arquitectura activa

La entrada pública del módulo ECG es únicamente:

- fotografía: JPG/JPEG/PNG/WEBP;
- PDF: el usuario selecciona la página que contiene el ECG.

No se solicitan archivos WFDB al usuario final.

```
Foto / PDF
  ↓
rasterización y rectificación
  ↓
ECG-Digitiser M3 · nnU-Net 2D
  ↓
WFDB reconstruido a 500 Hz
  ↓
control de integridad temporal y 12 derivaciones
  ↓
resample_poly 500→100 Hz
  ↓
R27 local
  ↓
35 probabilidades
```

## Digitalizador neuronal

Fuente fijada:

- Repositorio: `felixkrones/ECG-Digitiser`
- Commit: `e6f62aa776f105e4c7b04f21669da4d4f0df370b`
- Modelo: M3 / nnU-Net 2D
- Checkpoint SHA-256:
  `8e4bae0b568b91ee26bc29841ba2a1d9eb5571149f19a009459c85342375cffb`
- Tamaño del checkpoint: 474,901,894 bytes
- Licencia upstream: BSD-2-Clause
- Procedencia: solución ganadora del George B. Moody PhysioNet Challenge 2024.

El checkpoint se descarga desde GitHub Media en la primera ejecución de una instancia
y se verifica por tamaño y SHA-256 antes de inferencia.

## Aislamiento de memoria

El U-Net/nnU-Net corre en un subprocess separado. Ese proceso termina antes de
iniciar el subprocess R27. Esto evita mantener simultáneamente en RAM el modelo de
segmentación y la pila completa de R27.

Los hilos BLAS/OMP se limitan a 1 en el worker para reducir presión de memoria.

## Gate temporal fail-closed

R27 fue congelado sobre 10 segundos completos de las 12 derivaciones.

Un ECG impreso 3×4 habitual sólo contiene aproximadamente 2.5 s observados de la
mayoría de las derivaciones, más una tira larga de ritmo. Ningún algoritmo de
digitalización puede recuperar muestras que no están impresas.

Por lo tanto MEDCALC:

- digitaliza el trazado visible;
- calcula cobertura observada por derivación;
- exige las 12 derivaciones;
- exige 5000×12 a 500 Hz;
- exige al menos 90% de cobertura observada en cada derivación antes de llamar R27;
- NO repite, extrapola, imputa ni inventa los segmentos ausentes.

Si la imagen es 3×4 y no cumple cobertura, el resultado queda como
`DIGITIZED_ONLY` y R27 no se ejecuta.

## Adaptador 100 Hz

Cuando el registro cumple la cobertura completa, la representación 100 Hz se genera
desde la señal reconstruida de 500 Hz con:

```
scipy.signal.resample_poly(signal_500, up=1, down=5)
```

Este adaptador foto→señal constituye un dominio nuevo. No equivale a validación
externa del modelo R27.

## R27

R27 permanece sin cambios:

- `RESEARCH_PROBABILITY_ONLY_RELEASE`
- 35/35 salidas de probabilidad
- 0 thresholds desplegables
- 0 clasificaciones binarias
- 0 etiquetas diagnósticas
- `CLINICAL_DEPLOYMENT_BLOCKED`

## Secret requerido en Streamlit

```toml
R27_GITHUB_TOKEN = "github_pat_..."
```

El token debe tener sólo `Contents: Read-only` sobre el repositorio privado
`comincinieric56-maker/medcalc-r27-backend`.

No se requieren `ECG_R27_API_URL` ni `ECG_R27_API_TOKEN`.

## Archivos principales

- `ecg_r27_research_page.py`: interfaz foto/PDF.
- `ecg_unet_r27_bridge.py`: descarga/verificación del modelo y orquestación.
- `ecg_unet_worker.py`: digitalización neuronal aislada.
- `r27_local_runtime.py`: ejecución exacta del runtime R27.
