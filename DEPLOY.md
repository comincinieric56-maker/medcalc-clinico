# Deploy en Streamlit Community Cloud

- Main file: `app.py`
- Python recomendado en Community Cloud: `3.12`
- Dependencia: `streamlit==1.63.0`
- No requiere secrets en esta versión.

Ejecutar localmente:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
