# MedCalc Clínico V5

Aplicación Streamlit con una base maestra compartida para:

- Dosis pediátrica por medicamento + indicación + edad/peso + vía.
- Conversión de mg a mL desde la presentación ingresada.
- Cockcroft–Gault, CKD-EPI 2021, Schwartz bedside y superficie corporal.
- Ajuste renal por bandas específicas de cada regla.
- Toxicología farmacológica curada con trazabilidad de la base original.
- Buscador global por medicamento.

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Estructura

```text
medcalc_streamlit_v5_app/
├── app.py
├── medcalc_engine.py
├── repository.py
├── requirements.txt
├── .streamlit/config.toml
├── tests/test_engine.py
└── data/
```

## Seguridad clínica

La app no asigna una dosis genérica a todo el catálogo. Solo calcula cuando existe una regla marcada como automatizable. Los registros sin regla permanecen visibles en el catálogo como **NO HABILITADOS**.

En toxicología, **SDTE = SIN DOSIS TÓXICA ESPECÍFICA** y nunca se convierte automáticamente en un umbral numérico.

Las reglas renales automáticas de esta versión son principalmente adultas y no se extrapolan a pediatría.

## Despliegue en Streamlit Community Cloud

1. Subir esta carpeta a un repositorio de GitHub.
2. En Streamlit Community Cloud seleccionar el repositorio.
3. Main file: `app.py`.
4. No se requieren secrets para esta versión.

Antes de uso asistencial real, validar las reglas contra protocolos institucionales, disponibilidad local de formulaciones y políticas farmacológicas del centro.
