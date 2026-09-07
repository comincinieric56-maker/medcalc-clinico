from __future__ import annotations
import json

from ecg_v9_auditor_ui import page_count, render_auditor


def _audit_client(st):
    try:
        from ecg_v9_audit_repository import AuditRestRepository
        url = str(st.secrets.get('SUPABASE_URL') or '').strip()
        # Prefer the modern server key. Keep legacy name only as a fallback.
        key = str(
            st.secrets.get('SUPABASE_SECRET_KEY')
            or st.secrets.get('SUPABASE_SERVICE_ROLE_KEY')
            or ''
        ).strip()
        if not (url and key):
            return None
        return AuditRestRepository(url, key)
    except Exception:
        return None


def _safe_diag(st):
    try:
        from ecg_v9_audit_repository import safe_connection_diagnostic
        url = str(st.secrets.get('SUPABASE_URL') or '').strip()
        key = str(st.secrets.get('SUPABASE_SECRET_KEY') or st.secrets.get('SUPABASE_SERVICE_ROLE_KEY') or '').strip()
        return safe_connection_diagnostic(url, key)
    except Exception:
        return {}


def page_ecg_v9_auditor(st):
    st.header('EKG V9 · Auditor de aprendizaje')
    st.caption(
        'Construye la verdad supervisada del lector V9. Esta pantalla no cambia el modelo clínico en vivo: '
        'guarda tus correcciones para entrenar y validar versiones posteriores.'
    )

    client = _audit_client(st)
    if client is None:
        st.warning('No se pudo crear el cliente administrativo de auditoría. Configura `SUPABASE_SECRET_KEY` en Streamlit Secrets.')
    else:
        try:
            client.healthcheck()
            st.success('Supabase de auditoría: conexión administrativa verificada.', icon='✅')
        except Exception as exc:
            diag = _safe_diag(st)
            st.error(f'La autenticación del auditor falló: {exc}')
            with st.expander('Diagnóstico seguro de conexión'):
                st.json(diag)
            st.info(
                'Usa preferentemente la **Secret key moderna** del mismo proyecto (`sb_secret_...`) y guárdala como '
                '`SUPABASE_SECRET_KEY`. Este diagnóstico no muestra la clave completa.'
            )

    uploaded = st.file_uploader(
        'ECG para auditoría · JPG / PNG / PDF',
        type=['jpg', 'jpeg', 'png', 'pdf'],
        key='v9_audit_upload',
        help='Puedes subir el PDF completo de varios ECG y auditarlo página por página.',
    )
    if uploaded is None:
        st.info('Sube un ECG o PDF para comenzar la auditoría V9.')
        return

    raw = uploaded.getvalue()
    filename = uploaded.name
    try:
        n = page_count(raw, filename)
    except Exception as exc:
        st.error(f'No pude abrir el archivo: {exc}')
        return

    saved_rows = []
    existing_by_page = {}
    if client is not None:
        try:
            from ecg_v9_audit_repository import list_training_annotations_for_source, annotation_row_to_record
            saved_rows = list_training_annotations_for_source(client, raw_source_bytes=raw)
            existing_by_page = {
                int(r.get('page_number')): annotation_row_to_record(r)
                for r in saved_rows if r.get('page_number') is not None
            }
        except Exception as exc:
            st.warning(f'La tabla de auditoría aún no está disponible o no pudo consultarse: {exc}')

    saved_pages = set(existing_by_page)
    if n > 1:
        c1, c2, c3 = st.columns(3)
        c1.metric('Páginas del PDF', n)
        c2.metric('Auditadas', len(saved_pages))
        c3.metric('Pendientes', max(n - len(saved_pages), 0))
        if saved_pages:
            progress = min(1.0, len(saved_pages) / max(n, 1))
            st.progress(progress, text=f'Progreso de auditoría: {len(saved_pages)}/{n}')
        first_pending = next((i for i in range(n) if (i + 1) not in saved_pages), 0)
        page_index = st.selectbox(
            'Página a auditar', list(range(n)), index=first_pending,
            format_func=lambda i: f"{'✅' if (i+1) in saved_pages else '○'} Página {i+1} de {n}",
            key='v9_audit_page',
        )
    else:
        page_index = 0
        if 1 in saved_pages:
            st.success('Este ECG ya tiene una auditoría guardada. Puedes revisarla y sobrescribirla si corriges algo.')

    existing = existing_by_page.get(int(page_index) + 1)
    if existing:
        st.info(
            f"Esta página ya fue auditada. Última actualización: {existing.get('updated_at') or 'registrada'}. "
            'Los campos se precargan con la verdad guardada.'
        )
        with st.expander('Ver anotación guardada'):
            st.json(existing)

    record = render_auditor(
        st, raw_source=raw, filename=filename, page_index=int(page_index),
        prediction=None, existing_annotation=existing,
    )

    st.divider()
    st.markdown('## Guardar auditoría')
    encoded = json.dumps(record, ensure_ascii=False, indent=2).encode('utf-8')
    st.download_button(
        '⬇️ Descargar anotación JSON', data=encoded,
        file_name=f"ecg_v9_annotation_{record['source_sha256'][:10]}_p{record['page_number']:02d}.json",
        mime='application/json', key=f'v9_download_{page_index}',
    )

    if client is None:
        st.warning('Supabase de auditoría todavía no puede escribir. Agrega `SUPABASE_SECRET_KEY` en Streamlit Secrets.')
        return

    try:
        from ecg_v9_audit_repository import save_training_annotation
        button_label = '💾 Actualizar verdad auditada' if existing else '💾 Guardar verdad auditada en Supabase'
        if st.button(button_label, type='primary', key=f'v9_save_{page_index}'):
            row_id = save_training_annotation(
                client,
                raw_source_bytes=raw,
                source_filename=filename,
                page_number=record['page_number'],
                annotation=record,
                annotator_label='medcalc_v9_auditor',
            )
            st.success(f"Auditoría guardada correctamente. ID: {row_id or 'confirmado'}")
            st.rerun()
    except Exception as exc:
        st.error(f'No pude guardar la auditoría en Supabase: {exc}')
