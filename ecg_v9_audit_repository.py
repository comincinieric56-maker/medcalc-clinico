from __future__ import annotations
import hashlib
import json
import base64
from typing import Any, Dict, Optional, List
from urllib import request, parse, error


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _key_kind(key: str) -> str:
    k = (key or '').strip()
    if k.startswith('sb_secret_'):
        return 'secret_modern'
    if k.startswith('eyJ'):
        return 'legacy_jwt'
    if k.startswith('sb_publishable_'):
        return 'publishable_wrong_for_admin'
    return 'unknown'


def _decode_legacy_claims(key: str) -> Dict[str, Any]:
    if _key_kind(key) != 'legacy_jwt':
        return {}
    try:
        parts = key.split('.')
        payload = parts[1] + '=' * (-len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
    except Exception:
        return {}


def safe_connection_diagnostic(url: str, key: str) -> Dict[str, Any]:
    clean_url = (url or '').strip().rstrip('/')
    clean_key = (key or '').strip()
    host = parse.urlparse(clean_url).hostname or ''
    project_ref = host.split('.')[0] if host.endswith('.supabase.co') else host
    claims = _decode_legacy_claims(clean_key)
    return {
        'project_ref_from_url': project_ref,
        'key_kind': _key_kind(clean_key),
        'key_length': len(clean_key),
        'key_prefix': (clean_key[:10] + '…') if clean_key else '(vacía)',
        'legacy_role': claims.get('role'),
        'legacy_ref': claims.get('ref'),
        'legacy_issuer': claims.get('iss'),
    }


class AuditRestRepository:
    """Minimal server-only PostgREST client for ECG audit data.

    Modern Supabase secret keys are sent as `apikey` only, matching Supabase's
    current backend-key guidance. Legacy service_role JWTs additionally use the
    Authorization Bearer header so PostgREST receives the service_role claim.
    """
    def __init__(self, url: str, key: str, timeout: int = 20):
        self.url = (url or '').strip().rstrip('/')
        self.key = (key or '').strip()
        self.timeout = int(timeout)
        if not self.url or not self.key:
            raise ValueError('Faltan SUPABASE_URL o la clave secreta de auditoría.')
        if _key_kind(self.key) == 'publishable_wrong_for_admin':
            raise ValueError('La clave de auditoría no puede ser una sb_publishable_. Usa la Secret key sb_secret_.')

    def _headers(self, *, write: bool = False, prefer: Optional[str] = None) -> Dict[str, str]:
        headers = {
            'apikey': self.key,
            'Accept': 'application/json',
            'User-Agent': 'MedCalc-Streamlit-ECG-Auditor/9.4.1',
        }
        if _key_kind(self.key) == 'legacy_jwt':
            headers['Authorization'] = f'Bearer {self.key}'
        if write:
            headers['Content-Type'] = 'application/json'
        if prefer:
            headers['Prefer'] = prefer
        return headers

    def _call(self, method: str, path: str, *, payload: Any = None, prefer: Optional[str] = None) -> Any:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
        req = request.Request(
            self.url + path,
            data=data,
            method=method,
            headers=self._headers(write=payload is not None, prefer=prefer),
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode('utf-8')) if raw else []
        except error.HTTPError as exc:
            body = exc.read().decode('utf-8', errors='replace')
            if exc.code == 401:
                diag = safe_connection_diagnostic(self.url, self.key)
                raise RuntimeError(
                    'Supabase rechazó la clave de auditoría (401). '
                    f"Tipo detectado={diag['key_kind']}; proyecto URL={diag['project_ref_from_url']}; "
                    f"prefijo={diag['key_prefix']}; longitud={diag['key_length']}. "
                    'Verifica que la Secret key pertenezca exactamente al mismo proyecto que SUPABASE_URL. '
                    f'Detalle Supabase: {body}'
                ) from exc
            raise RuntimeError(f'HTTP {exc.code} de Supabase: {body}') from exc
        except Exception as exc:
            raise RuntimeError(f'No se pudo conectar con Supabase: {exc}') from exc

    def healthcheck(self) -> bool:
        q = parse.urlencode({'select': 'id', 'limit': '1'})
        self._call('GET', f'/rest/v1/ecg_training_annotations?{q}')
        return True

    def list_for_source(self, source_sha256: str) -> List[Dict[str, Any]]:
        fields = 'id,source_sha256,source_filename,page_number,annotation_version,orientation_deg,layout_code,lead_order,lead_boxes,calibration,quality,clinical_truth,diagnoses,exclude_from_training,exclusion_reason,annotator_label,notes,created_at,updated_at'
        q = parse.urlencode({
            'select': fields,
            'source_sha256': f'eq.{source_sha256}',
            'order': 'page_number.asc',
        })
        rows = self._call('GET', f'/rest/v1/ecg_training_annotations?{q}')
        return rows if isinstance(rows, list) else []

    def get_for_page(self, source_sha256: str, page_number: int) -> Optional[Dict[str, Any]]:
        q = parse.urlencode({
            'select': '*',
            'source_sha256': f'eq.{source_sha256}',
            'page_number': f'eq.{int(page_number)}',
            'limit': '1',
        })
        rows = self._call('GET', f'/rest/v1/ecg_training_annotations?{q}')
        return rows[0] if isinstance(rows, list) and rows else None

    def upsert_annotation(self, payload: Dict[str, Any]) -> Optional[str]:
        q = parse.urlencode({'on_conflict': 'source_sha256,page_number'})
        rows = self._call(
            'POST',
            f'/rest/v1/ecg_training_annotations?{q}',
            payload=payload,
            prefer='resolution=merge-duplicates,return=representation',
        )
        return rows[0].get('id') if isinstance(rows, list) and rows else None


def save_training_annotation(
    client: AuditRestRepository,
    *,
    raw_source_bytes: bytes,
    source_filename: str,
    page_number: int | None,
    annotation: Dict[str, Any],
    annotator_label: str | None = None,
) -> Optional[str]:
    payload = {
        'source_sha256': sha256_bytes(raw_source_bytes),
        'source_filename': source_filename or None,
        'page_number': page_number,
        'annotation_version': str(annotation.get('schema') or 'MEDCALC_ECG_ANNOTATION_V9_4_1'),
        'orientation_deg': annotation.get('orientation_deg'),
        'layout_code': annotation.get('layout'),
        'lead_order': annotation.get('lead_order') or {},
        'lead_boxes': annotation.get('lead_boxes') or {},
        'calibration': annotation.get('calibration') or {},
        'quality': annotation.get('quality') or {},
        'clinical_truth': annotation.get('clinical_truth') or {},
        'diagnoses': annotation.get('diagnoses') or [],
        'exclude_from_training': bool(annotation.get('exclude_from_training', False)),
        'exclusion_reason': annotation.get('exclusion_reason') or None,
        'annotator_label': annotator_label or None,
        'notes': annotation.get('notes') or None,
    }
    return client.upsert_annotation(payload)


def list_training_annotations_for_source(client: AuditRestRepository, *, raw_source_bytes: bytes) -> List[Dict[str, Any]]:
    return client.list_for_source(sha256_bytes(raw_source_bytes))


def get_training_annotation(client: AuditRestRepository, *, raw_source_bytes: bytes, page_number: int) -> Optional[Dict[str, Any]]:
    return client.get_for_page(sha256_bytes(raw_source_bytes), int(page_number))


def annotation_row_to_record(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    return {
        'schema': row.get('annotation_version') or 'MEDCALC_ECG_ANNOTATION_V9_4_1',
        'source_sha256': row.get('source_sha256'),
        'source_filename': row.get('source_filename'),
        'page_number': row.get('page_number'),
        'orientation_deg': row.get('orientation_deg'),
        'layout': row.get('layout_code'),
        'lead_order': row.get('lead_order') or {},
        'lead_boxes': row.get('lead_boxes') or {},
        'calibration': row.get('calibration') or {},
        'quality': row.get('quality') or {},
        'clinical_truth': row.get('clinical_truth') or {},
        'diagnoses': row.get('diagnoses') or [],
        'exclude_from_training': bool(row.get('exclude_from_training', False)),
        'exclusion_reason': row.get('exclusion_reason') or '',
        'notes': row.get('notes') or '',
        'saved_id': row.get('id'),
        'updated_at': row.get('updated_at'),
    }
