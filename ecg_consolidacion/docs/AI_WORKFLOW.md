# Flujo de trabajo entre ChatGPT/Codex, Claude y Google AI Studio

Regla de oro: **quien escribe un cambio no lo aprueba**, y ninguna IA promueve un modelo.

| Responsable | Hace | No hace |
|---|---|---|
| ChatGPT / Codex | Implementación, integración, DSP, pipelines de datos, scripts de benchmark, pruebas, PRs | Aprobar sus PRs; elegir umbrales; tocar `frozen/`, `datasets/`, `protocols/` |
| Claude | Revisión de cada PR (código y método), auditoría de fuga y particiones, revisión de protocolos antes de ejecutar, análisis de errores, revisión de seguridad | Escribir y revisar el mismo PR |
| Google AI Studio / Gemini | Prototipos de UI, visor de 12 derivaciones, herramientas de anotación, maquetas de informe | Calcular mediciones o diagnósticos en la UI |
| Responsable humano | Merge a `main`, promoción de modelos, apertura de conjuntos reservados, adjudicación médica | — |

## Ramas

- `main`: protegida. Solo merge por PR aprobado.
- `feat/<ia>-<tema>`: p. ej. `feat/codex-qt-benchmark`, `feat/gemini-viewer`.
- `audit/claude-<tema>`: informes de auditoría.
- `proto/<tema>`: solo protocolos (se fusionan antes de producir resultados).

## Ciclo de un cambio de modelo

1. PR `proto/...` con el protocolo YAML + `python tools/prereg.py register ...` → revisión de Claude → merge.
2. PR `feat/...` con código y resultados que citan el hash del protocolo → CI → revisión de una IA distinta → tu aprobación.
3. Promoción: PR que cambia `status`/`frozen` en `models/registry.yaml` y regenera `frozen/FROZEN.lock.json`. Solo el responsable.

## Configuración de GitHub pendiente (requiere al responsable)

En Settings → Branches → `main`: exigir PR, 1 aprobación, revisión de CODEOWNERS y el check `MEDCALC EKG consolidación (F0/F1) / checks` en verde; bloquear push directo.
