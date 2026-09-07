from __future__ import annotations
import json

from ecg_v9_auditor_ui import page_count, render_auditor, render_source_page


def _audit_client(st):
    try:
        from ecg_v9_audit_repository import AuditRestRepository
        url = str(st.secrets.get('SUPABASE_URL') or '').strip()
        key = str(st.secrets.get('SUPABASE_SECRET_KEY') or st.secrets.get('SUPABASE_SERVICE_ROLE_KEY') or '').strip()
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


def _get_shadow_state(client):
    if client is None:
        return None
    try:
        from ecg_v9_online_learner import MODEL_NAME
        row = client.get_learner_state(MODEL_NAME)
        return (row or {}).get('state') if row else None
    except Exception:
        return None



def _shadow_eval_summary(client):
    try:
        rows = client.list_shadow_evaluations()
        first = [r for r in rows if r.get('first_exposure')]
        if not first:
            return {'n':0,'orientation_accuracy':None,'layout_accuracy':None}
        o=[r for r in first if r.get('orientation_correct') is not None]
        l=[r for r in first if r.get('layout_correct') is not None]
        return {
            'n':len(first),
            'orientation_accuracy': (sum(bool(r.get('orientation_correct')) for r in o)/len(o)) if o else None,
            'layout_accuracy': (sum(bool(r.get('layout_correct')) for r in l)/len(l)) if l else None,
        }
    except Exception:
        return {'n':0,'orientation_accuracy':None,'layout_accuracy':None}

def _render_shadow_summary(st, prediction, eval_summary=None):
    st.markdown('## 🧪 Modo sombra · orientación/layout')
    st.caption('Este aprendiz NO hace diagnóstico clínico. Predice antes de ver tu auditoría y aprende después de guardarla.')
    if not prediction or not prediction.get('available'):
        st.info((prediction or {}).get('reason') or 'Aún no hay modelo sombra disponible.')
        return
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Modelo sombra', f"v{prediction.get('model_version',0)}")
    c2.metric('Casos aprendidos', int(prediction.get('n_cases',0)))
    c3.metric('Orientación', f"{prediction.get('orientation_deg')}°", f"{prediction.get('orientation_confidence',0)*100:.0f}% conf.")
    c4.metric('Layout', prediction.get('layout') or '—', f"{prediction.get('layout_confidence',0)*100:.0f}% conf.")
    with st.expander('Probabilidades y métricas del aprendiz'):
        st.json(prediction)
    if eval_summary and eval_summary.get('n',0)>0:
        st.caption('Evaluación pre-update en primeras exposiciones:')
        e1,e2,e3=st.columns(3)
        e1.metric('Páginas evaluadas antes de aprender', int(eval_summary.get('n',0)))
        oa=eval_summary.get('orientation_accuracy'); la=eval_summary.get('layout_accuracy')
        e2.metric('Acierto orientación', '—' if oa is None else f'{oa*100:.0f}%')
        e3.metric('Acierto layout', '—' if la is None else f'{la*100:.0f}%')
    st.warning('La confianza aquí es interna al modelo sombra. La métrica pre-update es útil para seguimiento, pero no reemplaza una validación externa independiente.')


def _update_shadow_after_save(client, *, image, record):
    from ecg_v9_online_learner import make_training_feature_record, rebuild_state, MODEL_NAME, FEATURE_VERSION
    feature = make_training_feature_record(
        image,
        source_sha256=record['source_sha256'],
        page_number=record['page_number'],
        orientation_deg=record['orientation_deg'],
        layout=record['layout'],
        exclude=record.get('exclude_from_training', False),
    )
    client.upsert_online_feature(feature)
    rows = client.list_online_features()
    prev = client.get_learner_state(MODEL_NAME) or {}
    prev_ver = int(prev.get('model_version') or 0)
    state = rebuild_state(rows, previous_version=prev_ver)
    client.upsert_learner_state({
        'model_name': MODEL_NAME,
        'model_version': state['model_version'],
        'feature_version': FEATURE_VERSION,
        'n_cases': state['n_cases'],
        'state': state,
        'metrics': state.get('metrics') or {},
    })
    return state


def page_ecg_v9_auditor(st):
    st.header('EKG V9 · Auditor + aprendizaje sombra')
    st.caption(
        'La verdad auditada alimenta un aprendiz visual de orientación/layout. El diagnóstico clínico NO se reentrena en línea.'
    )

    client = _audit_client(st)
    auth_ok = False
    if client is None:
        st.warning('No se pudo crear el cliente administrativo de auditoría. Configura `SUPABASE_SECRET_KEY`.')
    else:
        try:
            client.healthcheck(); auth_ok = True
            st.success('Supabase de auditoría: conexión administrativa verificada.', icon='✅')
        except Exception as exc:
            st.error(f'La autenticación del auditor falló: {exc}')
            with st.expander('Diagnóstico seguro de conexión'):
                st.json(_safe_diag(st))

    uploaded = st.file_uploader('ECG para auditoría · JPG / PNG / PDF', type=['jpg','jpeg','png','pdf'], key='v9_audit_upload')
    if uploaded is None:
        st.info('Sube un ECG o PDF para comenzar la auditoría V9.')
        return
    raw = uploaded.getvalue(); filename = uploaded.name
    try:
        n = page_count(raw, filename)
    except Exception as exc:
        st.error(f'No pude abrir el archivo: {exc}'); return

    saved_rows=[]; existing_by_page={}
    if client is not None and auth_ok:
        try:
            from ecg_v9_audit_repository import list_training_annotations_for_source, annotation_row_to_record
            saved_rows = list_training_annotations_for_source(client, raw_source_bytes=raw)
            existing_by_page = {int(r.get('page_number')): annotation_row_to_record(r) for r in saved_rows if r.get('page_number') is not None}
        except Exception as exc:
            st.warning(f'No pude consultar el progreso de auditoría: {exc}')

    saved_pages=set(existing_by_page)
    if n>1:
        c1,c2,c3=st.columns(3); c1.metric('Páginas del PDF',n); c2.metric('Auditadas',len(saved_pages)); c3.metric('Pendientes',max(n-len(saved_pages),0))
        st.progress(min(1.0,len(saved_pages)/max(n,1)), text=f'Progreso de auditoría: {len(saved_pages)}/{n}')
        first_pending=next((i for i in range(n) if i+1 not in saved_pages),0)
        page_index=st.selectbox('Página a auditar',list(range(n)),index=first_pending,format_func=lambda i:f"{'✅' if i+1 in saved_pages else '○'} Página {i+1} de {n}",key='v9_audit_page')
    else:
        page_index=0

    existing=existing_by_page.get(int(page_index)+1)
    base_img = render_source_page(raw, filename, page_index=int(page_index), dpi=150)

    # Shadow prediction is made before the current unsaved truth is seen.
    prediction=None
    if client is not None and auth_ok:
        try:
            from ecg_v9_online_learner import predict_shadow
            prediction=predict_shadow(base_img, _get_shadow_state(client))
        except Exception as exc:
            prediction={'available':False,'reason':f'Aprendiz sombra no disponible: {exc}'}
    _render_shadow_summary(st,prediction, _shadow_eval_summary(client) if client is not None and auth_ok else None)

    if existing:
        st.info(f"Esta página ya fue auditada. Última actualización: {existing.get('updated_at') or 'registrada'}.")
        with st.expander('Ver anotación guardada'):
            st.json(existing)

    record=render_auditor(st,raw_source=raw,filename=filename,page_index=int(page_index),prediction=prediction,existing_annotation=existing)

    st.divider(); st.markdown('## Guardar auditoría')
    encoded=json.dumps(record,ensure_ascii=False,indent=2).encode('utf-8')
    st.download_button('⬇️ Descargar anotación JSON',data=encoded,file_name=f"ecg_v9_annotation_{record['source_sha256'][:10]}_p{record['page_number']:02d}.json",mime='application/json',key=f'v9_download_{page_index}')
    if client is None or not auth_ok:
        st.warning('Supabase de auditoría todavía no puede escribir.'); return

    try:
        from ecg_v9_audit_repository import save_training_annotation
        label='💾 Actualizar verdad auditada' if existing else '💾 Guardar verdad + enseñar al aprendiz sombra'
        if st.button(label,type='primary',key=f'v9_save_{page_index}'):
            row_id=save_training_annotation(client,raw_source_bytes=raw,source_filename=filename,page_number=record['page_number'],annotation=record,annotator_label='medcalc_v9_auditor')
            if record.get('layout') in ('3x4','6x2') and record.get('orientation_deg') in (0,90,180,270):
                try:
                    # On first exposure, log the prediction BEFORE learning this truth.
                    if (not existing) and prediction and prediction.get('available'):
                        client.insert_shadow_evaluation({
                            'source_sha256': record['source_sha256'],
                            'page_number': int(record['page_number']),
                            'model_version': int(prediction.get('model_version') or 0),
                            'n_cases_before': int(prediction.get('n_cases') or 0),
                            'predicted_orientation_deg': prediction.get('orientation_deg'),
                            'orientation_confidence': prediction.get('orientation_confidence'),
                            'predicted_layout': prediction.get('layout'),
                            'layout_confidence': prediction.get('layout_confidence'),
                            'truth_orientation_deg': int(record['orientation_deg']),
                            'truth_layout': record['layout'],
                            'orientation_correct': prediction.get('orientation_deg') == int(record['orientation_deg']),
                            'layout_correct': prediction.get('layout') == record['layout'],
                            'first_exposure': True,
                        })
                    new_state=_update_shadow_after_save(client,image=base_img,record=record)
                    st.success(f"Auditoría guardada. Aprendiz sombra actualizado a v{new_state['model_version']} con {new_state['n_cases']} caso(s). ID: {row_id or 'confirmado'}")
                except Exception as learn_exc:
                    st.success(f"Auditoría guardada. ID: {row_id or 'confirmado'}")
                    st.warning(f'La auditoría se guardó, pero el aprendiz sombra no pudo actualizarse: {learn_exc}')
            else:
                st.success(f"Auditoría guardada. Este caso no se usó para orientación/layout por layout no entrenable. ID: {row_id or 'confirmado'}")
            st.rerun()
    except Exception as exc:
        st.error(f'No pude guardar la auditoría en Supabase: {exc}')
