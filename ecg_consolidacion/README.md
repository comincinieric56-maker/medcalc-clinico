# MEDCALC EKG — consolidación (F0/F1)

Carpeta de consolidación del motor ECG de MEDCALC dentro de `medcalc-clinico`. **Investigación únicamente: `CLINICAL_DEPLOYMENT_BLOCKED`.**

Ningún componente está validado externamente. Ningún modelo de lenguaje calcula mediciones ni diagnósticos.

## Qué hay aquí (fases F0 y F1)

| Carpeta | Contenido |
|---|---|
| `models/registry.yaml` | Registro canónico de modelos: SHA-256, datos de entrenamiento/selección/evaluación, métricas con su archivo de origen, estado (RC/EXP/VI/VE/OBS/PE). |
| `models/manifests/` | Manifiestos SHA-256 originales de R27 (backend y release V41.0R27). |
| `datasets/registry.yaml` | Estado de cada conjunto (`FRESH`, `LOCKED_FRESH`, `DEV`, `CONSUMED`), particiones y fugas conocidas. |
| `datasets/splits_*.json` | Particiones LUDB verificadas (TRI-HEAD y V16). |
| `frozen/FROZEN.lock.json` | Candado de todo lo congelado; la CI falla si algo cambia. |
| `protocols/` | Protocolos y ledger de preregistro. |
| `engine/contracts/` | Contrato canónico de entrada (12×5000, 500 Hz, mV). |
| `tools/` | Validadores de la CI y verificador de Drive de solo lectura. |
| `docs/` | Flujo entre IAs, criterios de aceptación, checklist de PR e informe F0/F1. |

Este repositorio es **público**: aquí solo hay hashes, rutas y documentación. Los **pesos y datos no se guardan en Git**: siguen en Google Drive (`MyDrive/MEDCALC`) y se referencian por SHA-256.

## Comandos

```bash
cd ecg_consolidacion
pip install -r requirements-dev.txt
python tools/registry.py            # valida registros
python tools/check_frozen.py        # verifica congelados
python tools/check_reserved_paths.py
python tools/prereg.py verify
python -m pytest -q
```

Verificar los SHA contra Drive (en Colab, solo lectura):

```python
from google.colab import drive; drive.mount('/content/drive')
!git clone --depth 1 https://github.com/comincinieric56-maker/medcalc-clinico /content/medcalc-clinico
!pip -q install pyyaml
!python /content/medcalc-clinico/ecg_consolidacion/tools/verify_drive_artifacts.py --drive-root /content/drive/MyDrive/MEDCALC
```

## Reglas

- Todo PR que toque esta carpeta necesita el check `MEDCALC EKG consolidación (F0/F1)` en verde, revisión de una IA distinta del autor y aprobación del responsable (`.github/CODEOWNERS`). Usar `docs/PR_CHECKLIST.md`.
- Cambios de modelo: protocolo registrado con `tools/prereg.py` **antes** de producir resultados.
- Ver `docs/AI_WORKFLOW.md` y `docs/ACCEPTANCE_CRITERIA.md`.
