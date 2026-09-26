# MEDCALC ECG · FOTO/PDF → U-Net → R27

## Flujo público

La entrada del usuario final es exclusivamente:

- fotografía JPG/JPEG/PNG/WEBP;
- PDF del ECG, con selección de página cuando corresponde.

No se solicitan archivos WFDB ni ECG digital.

```
Foto / PDF
  ↓
rasterización local
  ↓
Open ECG Digitizer · U-Net
  ↓
perspectiva + cuadrícula + layout + extracción
  ↓
12 derivaciones canónicas · objetivo 5000 muestras
  ↓
gate de cobertura observada
  ├─ incompleto → DIGITIZED_ONLY
  └─ completo   → WFDB 500 Hz
                       ↓
                 resample_poly 100 Hz
                       ↓
                     R27
                       ↓
               35 probabilidades
```

## Digitalizador U-Net fijado

Fuente vendorizada:

- Repositorio: `https://github.com/Ahus-AIM/Open-ECG-Digitizer`
- Commit: `97a15087d4abcda843da8c58ee74b1d8f47e6f9a`
- Licencia upstream: `CC BY-SA 4.0`

Checkpoints verificados e incluidos en MEDCALC:

### Segmentación ECG
- Archivo: `ecg_digitizer_assets/unet_weights_07072025.pt`
- SHA-256: `17fe7071ef270102631306127262fc08c250d79d4e3aeb572ab1719dd34d320b`
- Tamaño: 90,464,067 bytes

### Identificación de derivaciones
- Archivo: `ecg_digitizer_assets/lead_name_unet_weights_07072025.pt`
- SHA-256: `840bd6bf2433ee6c22db67f57c861d9d427f29e10a32eeb334f0bcf061b175a2`
- Tamaño: 23,296,757 bytes

La fuente necesaria para inferencia está en `ecg_digitizer_vendor/src/`. Se
conserva la licencia y el archivo de procedencia.

## Memoria

La digitalización U-Net ocurre en `ecg_unet_worker.py`, un subprocess aislado.
Cuando termina, ese proceso desaparece antes de cargar R27. Por tanto MEDCALC no
mantiene a la vez en RAM el U-Net y toda la pila de modelos R27.

La inferencia se fuerza a CPU y se limitan OMP/BLAS a un hilo.

## Regla fail-closed para ECG impresos 3×4

R27 fue congelado con un contrato de señal de 10 segundos × 12 derivaciones.

Un ECG estándar 3×4 suele imprimir aproximadamente 2.5 segundos de cada
derivación, con una tira larga de ritmo. Los otros segundos **no existen en el
papel** y no pueden recuperarse legítimamente mediante digitalización.

Open ECG Digitizer conserva las zonas no observadas como NaN en la matriz
canónica. MEDCALC usa esa información para medir cobertura.

R27 sólo se ejecuta cuando:

- existen exactamente las 12 derivaciones estándar;
- la matriz canónica es exactamente 12 × 5000;
- las 5000 muestras de cada una de las 12 derivaciones son observadas y finitas;
- no existe ninguna muestra NaN/no observada.

MEDCALC no:

- repite segmentos;
- rellena 2.5 s hasta 10 s;
- extrapola ondas;
- imputa señal no impresa;
- sustituye derivaciones faltantes.

Cuando el ECG impreso es válido para digitalización pero no cumple los 10 s × 12,
el estado es `DIGITIZED_ONLY` y R27 no se ejecuta.

## Adaptador 500 Hz → 100 Hz

Cuando las 12 derivaciones completas sí están observadas:

1. el digitalizador produce 5000 muestras por derivación;
2. sus unidades µV se convierten a mV;
3. se escribe WFDB a 500 Hz;
4. se genera una segunda representación a 100 Hz mediante:
   `scipy.signal.resample_poly(signal_500, up=1, down=5)`.

Este adaptador pertenece al nuevo dominio foto/PDF y **no equivale a validación
externa de R27**.

## R27 permanece congelado

- `RESEARCH_PROBABILITY_ONLY_RELEASE`
- 35/35 probabilidades
- 0 thresholds desplegables
- 0 clasificaciones binarias
- 0 etiquetas diagnósticas
- `CLINICAL_DEPLOYMENT_BLOCKED`

R27 se materializa desde el repositorio privado
`comincinieric56-maker/medcalc-r27-backend` en el commit congelado
`ffb4980570a4efd4c54cb0d326c94858f3905711`.

## Secret requerido en Streamlit

```toml
R27_GITHUB_TOKEN = "github_pat_..."
```

El token debe tener únicamente `Contents: Read-only` sobre
`comincinieric56-maker/medcalc-r27-backend`.

No se requieren `ECG_R27_API_URL` ni `ECG_R27_API_TOKEN`.

## Archivos de integración

- `ecg_r27_research_page.py`: interfaz pública foto/PDF.
- `ecg_unet_r27_bridge.py`: verificación de checkpoints y orquestación.
- `ecg_unet_worker.py`: inferencia U-Net y construcción de señal.
- `ecg_digitizer_vendor/`: fuente fijada del digitalizador.
- `ecg_digitizer_assets/`: checkpoints + licencia + procedencia.
- `r27_local_runtime.py`: runtime R27 congelado.
