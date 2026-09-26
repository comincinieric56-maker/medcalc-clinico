# MEDCALC ECG R27 · ejecución local en Streamlit

## Estado

La aplicación ya no depende de Render para ejecutar R27.

Ruta:

```
Streamlit Community Cloud
  -> Electrocardiograma
  -> ECG digital
  -> r27_local_runtime.py
  -> repositorio privado medcalc-r27-backend @ ffb4980570a4efd4c54cb0d326c94858f3905711
  -> runtime R27 exacto
  -> 35 probabilidades
```

R27 permanece:

- `RESEARCH_PROBABILITY_ONLY_RELEASE`
- 35/35 salidas de probabilidad
- 0 thresholds desplegables
- 0 clasificaciones binarias
- 0 etiquetas diagnósticas
- `CLINICAL_DEPLOYMENT_BLOCKED`

## Secret requerido en Streamlit

Crear un GitHub Fine-grained Personal Access Token con:

- Resource owner: `comincinieric56-maker`
- Repository access: **Only select repositories**
- Repositorio: `medcalc-r27-backend`
- Repository permissions -> **Contents: Read-only**
- Sin permisos de escritura.

Agregarlo en Streamlit Community Cloud -> App -> Settings -> Secrets:

```toml
R27_GITHUB_TOKEN = "github_pat_..."
```

No usar `ECG_R27_API_URL` ni `ECG_R27_API_TOKEN` para la nueva ruta local.

## Materialización

El runtime se descarga sólo cuando se entra al flujo R27 y se solicita verificar/analizar.
La copia queda en el almacenamiento temporal de la instancia Streamlit.

Se fija el commit fuente exacto:

```
ffb4980570a4efd4c54cb0d326c94858f3905711
```

Además se verifican SHA-256 críticos de P6A, P6B, V31, V27/V24, V37 y la política del adaptador.

## Dependencias

`requirements.txt` fija el entorno científico usado en el smoke local y solicita PyTorch CPU-only.

## Navegación

La navegación pública queda unificada como:

```
Electrocardiograma
  |- Foto / PDF
  |    -> reservado para U-Net + digitalizador definitivo
  |
  '- ECG digital
       -> R27 local
```

`ECG V9 Auditor` y `ECG R27 Investigación` dejan de aparecer como páginas separadas.

La pestaña Foto/PDF todavía no envía imágenes a R27. No se inventan muestras ni segmentos ausentes.
