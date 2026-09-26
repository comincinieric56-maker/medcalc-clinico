MEDCALC · R27 APP INTEGRATION · CURRENT EXACT APP

BASE APP
========
Source supplied by user:
app(20260926-123045).py

Detected current identity:
V9.5 · EKG AUDITOR + SHADOW LEARNING · V8.4.3 CLÍNICO

Preserved modules include:
- Inicio
- Dosis pediátrica
- Ajuste renal
- Toxicología
- Embarazo
- Hidroelectrolitos
- Electrocardiograma
- ECG V9 Auditor
- Base y fuentes

PATCH
=====
Adds only:
- safe import of ecg_r27_research_page
- page "ECG R27 Investigación"
- sidebar navigation label
- r27_ session namespace reset
- route to page_ecg_r27_research
- visible version label mentioning R27 Research

No existing clinical module was removed.
No ECG model was retrained or modified.

R27 remains:
RESEARCH_PROBABILITY_ONLY_RELEASE
CLINICAL_DEPLOYMENT_BLOCKED

The new page remains inactive until ECG_R27_API_URL is configured.
