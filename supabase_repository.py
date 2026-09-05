from pathlib import Path
import csv
import json
import re
import sqlite3
import unicodedata

from supabase import create_client

SCHEMA_VERSION = "MEDCALC_SUPABASE_V3"


def normalize_text(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Za-z0-9]+", " ", text).casefold()
    return " ".join(text.split())


def _si(value):
    return "SI" if bool(value) else "NO"


def _source_date(value):
    if value is None:
        return None
    return str(value)


class SupabaseRepository:
    """Read-only repository for the public MedCalc Streamlit app.

    Core clinical data are read from Supabase using the publishable key and RLS.
    The optional SQLite path is used only as a temporary fallback for the two
    ancillary toxicology datasets that were not part of the first Supabase
    migration (non-pharmaceutical toxicants and antidote cards).
    """

    def __init__(self, url, publishable_key, fallback_db_path=None):
        if not url or not publishable_key:
            raise ValueError("Faltan SUPABASE_URL o SUPABASE_PUBLISHABLE_KEY.")
        self.client = create_client(url, publishable_key)
        self.fallback_db_path = Path(fallback_db_path) if fallback_db_path else None

        version = self.metadata("schema_version")
        if version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Supabase incompatible. La app requiere {SCHEMA_VERSION} y el proyecto reporta {version or 'sin versión'}."
            )

        self._medications = self._fetch_all("medications", "id,med_id,generic_name,normalized_name,active")
        self._medications = [r for r in self._medications if r.get("active") is not False]
        self._medications.sort(key=lambda r: (normalize_text(r.get("generic_name")), r.get("med_id") or ""))
        self._med_by_med_id = {r["med_id"]: r for r in self._medications}
        self._uuid_by_med_id = {r["med_id"]: r["id"] for r in self._medications}

        alias_rows = self._fetch_all("drug_aliases", "medication_id,alias,normalized_alias")
        self._aliases_by_uuid = {}
        for row in alias_rows:
            self._aliases_by_uuid.setdefault(row.get("medication_id"), []).append(row.get("alias") or "")

        status_rows = self._fetch_all(
            "medication_module_status",
            "medication_id,pediatric_status,renal_status,toxicology_status,clinical_priority,pediatric_note,renal_note,toxicology_note",
        )
        self._status_by_uuid = {r.get("medication_id"): r for r in status_rows}

        source_rows = self._fetch_all(
            "sources",
            "id,title,organization,authors,publication_year,edition,url,page,source_type,last_verified",
        )
        self._sources_by_id = {r["id"]: r for r in source_rows}
        self._sources_cache = source_rows
        self._renal_biblio_cache = None
        self._counts_cache = None

    # ---------- Supabase primitives ----------
    def _fetch_all(self, table, columns="*"):
        # All current MedCalc tables are <1000 rows. Keep a small pagination
        # loop anyway so future growth does not silently truncate results.
        out = []
        start = 0
        page_size = 1000
        while True:
            res = self.client.table(table).select(columns).range(start, start + page_size - 1).execute()
            batch = res.data or []
            out.extend(batch)
            if len(batch) < page_size:
                break
            start += page_size
        return out

    def _published_for_med(self, table, med_id, columns="*"):
        """Devuelve exclusivamente registros PUBLISHED.

        Se conserva para módulos donde la app solo debe trabajar con contenido
        validado/publicado (renal, toxicología, etc.).
        """
        medication_uuid = self._uuid_by_med_id.get(med_id)
        if not medication_uuid:
            return []
        res = (
            self.client.table(table)
            .select(columns)
            .eq("medication_id", medication_uuid)
            .eq("status", "PUBLISHED")
            .execute()
        )
        return res.data or []

    def _visible_pediatric_for_med(self, med_id, columns="*"):
        """Pediatría: hace visibles PUBLISHED y PENDING_REVIEW.

        PENDING_REVIEW se expone como referencia bibliográfica estructurada,
        pero NO adquiere por ello condición de regla validada ni permiso de
        cálculo automático. Esa separación se hace en la interfaz.
        """
        medication_uuid = self._uuid_by_med_id.get(med_id)
        if not medication_uuid:
            return []
        res = (
            self.client.table("pediatric_rules")
            .select(columns)
            .eq("medication_id", medication_uuid)
            .execute()
        )
        rows = res.data or []
        return [
            r for r in rows
            if str(r.get("status") or "").upper() in {"PUBLISHED", "PENDING_REVIEW"}
        ]

    def _source(self, source_id):
        return self._sources_by_id.get(source_id) or {}

    def metadata(self, key):
        res = self.client.table("app_metadata").select("value").eq("key", key).limit(1).execute()
        rows = res.data or []
        return rows[0].get("value") if rows else None

    # ---------- Counts / catalogue ----------
    def counts(self):
        if self._counts_cache is not None:
            return dict(self._counts_cache)

        peds_all = self._fetch_all("pediatric_rules", "medication_id,automatizable,status")
        peds = [
            r for r in peds_all
            if str(r.get("status") or "").upper() in {"PUBLISHED", "PENDING_REVIEW"}
        ]
        peds_published = [r for r in peds if str(r.get("status") or "").upper() == "PUBLISHED"]
        peds_pending = [r for r in peds if str(r.get("status") or "").upper() == "PENDING_REVIEW"]

        renals = self._fetch_all("renal_rules", "medication_id,automatizable,status")
        renals = [r for r in renals if r.get("status") == "PUBLISHED"]
        refs = self._fetch_all("renal_bibliography", "id,status")
        refs = [r for r in refs if r.get("status") == "PUBLISHED"]
        tox = self._fetch_all("toxicology", "id,status")
        tox = [r for r in tox if r.get("status") == "PUBLISHED"]

        self._counts_cache = {
            "medications": len(self._medications),
            # Pediatría visible = PUBLISHED + PENDING_REVIEW.
            "pediatric_rules": len(peds),
            "pediatric_rules_published": len(peds_published),
            "pediatric_rules_pending": len(peds_pending),
            "pediatric_meds": len({r["medication_id"] for r in peds}),
            "pediatric_auto_meds": len({
                r["medication_id"] for r in peds_published if r.get("automatizable")
            }),
            "renal_rules": len(renals),
            "renal_auto_rules": sum(1 for r in renals if r.get("automatizable")),
            "renal_reference_rules": sum(1 for r in renals if not r.get("automatizable")),
            "renal_meds": len({r["medication_id"] for r in renals}),
            "renal_auto_meds": len({r["medication_id"] for r in renals if r.get("automatizable")}),
            "renal_reference_meds": len({r["medication_id"] for r in renals if not r.get("automatizable")}),
            "renal_biblio": len(refs),
            "toxicology": len(tox),
        }
        return dict(self._counts_cache)

    def search_medications(self, query="", limit=2000):
        q = normalize_text(query)
        rows = self._medications
        if q:
            filtered = []
            for r in rows:
                aliases = self._aliases_by_uuid.get(r.get("id"), [])
                if (
                    q in normalize_text(r.get("generic_name"))
                    or q in normalize_text(r.get("med_id"))
                    or any(q in normalize_text(a) for a in aliases)
                ):
                    filtered.append(r)
            rows = filtered
        rows = rows[: int(limit)]
        return [
            {
                "id": r.get("id"),
                "med_id": r.get("med_id"),
                "principio_activo": r.get("generic_name"),
                "search_name": normalize_text(r.get("generic_name")),
            }
            for r in rows
        ]

    def medication(self, med_id):
        med = self._med_by_med_id.get(med_id)
        if not med:
            return None
        peds = self.pediatric_rules(med_id)
        renals = self.renal_rules(med_id)
        refs = self.renal_biblio(med_id)
        tox = self.toxicology(med_id)
        status = self._status_by_uuid.get(med.get("id")) or {}
        return {
            "id": med.get("id"),
            "med_id": med.get("med_id"),
            "principio_activo": med.get("generic_name"),
            "search_name": normalize_text(med.get("generic_name")),
            "pediatric_status": status.get("pediatric_status") or "PENDING_REVIEW",
            "renal_status": status.get("renal_status") or "PENDING_REVIEW",
            "toxicology_status": status.get("toxicology_status") or "PENDING_REVIEW",
            "clinical_priority": status.get("clinical_priority") or 3,
            "pediatric_rule_count": len(peds),
            "pediatric_published_count": sum(
                1 for r in peds if str(r.get("estado") or "").upper() == "PUBLISHED"
            ),
            "pediatric_pending_count": sum(
                1 for r in peds if str(r.get("estado") or "").upper() == "PENDING_REVIEW"
            ),
            "pediatric_auto_count": sum(
                1 for r in peds
                if str(r.get("estado") or "").upper() == "PUBLISHED"
                and r.get("automatizable") == "SI"
            ),
            # Mantener renal_rule_count como conteo automático por compatibilidad.
            "renal_rule_count": sum(1 for r in renals if r.get("automatizable") == "SI"),
            "renal_reference_rule_count": sum(1 for r in renals if r.get("automatizable") != "SI"),
            "renal_total_rule_count": len(renals),
            "renal_biblio_count": len(refs),
            "toxicology_available": 1 if tox else 0,
        }

    def module_status(self, med_id):
        med = self._med_by_med_id.get(med_id)
        if not med:
            return None
        return dict(self._status_by_uuid.get(med.get("id")) or {})

    # ---------- Pediatric ----------
    def _map_pediatric(self, r):
        src = self._source(r.get("source_id"))
        unit = r.get("dose_unit") or "mg"
        fixed = r.get("fixed_dose_min")
        max_single = r.get("max_single")
        max_daily = r.get("max_daily")
        max_daily_kg = r.get("max_daily_per_kg")
        return {
            "id": r.get("id"),
            "rule_id": f"PED-SB-{str(r.get('id') or '')[:8]}",
            "med_id": None,
            "principio_activo": None,
            "indicacion": r.get("indication"),
            "poblacion": r.get("population"),
            "edad_min_meses": r.get("age_min_months"),
            "edad_max_meses": r.get("age_max_months"),
            "peso_min_kg": r.get("weight_min_kg"),
            "peso_max_kg": r.get("weight_max_kg"),
            "peso_min_exclusivo": _si(r.get("weight_min_exclusive")),
            "peso_max_exclusivo": _si(r.get("weight_max_exclusive")),
            "via": r.get("route"),
            "tipo_dosis": r.get("dose_type"),
            "unidad_dosis": unit,
            "dosis_valor": r.get("dose_min"),
            "dosis_valor_max": r.get("dose_max"),
            "intervalo_h": r.get("interval_hours"),
            "divisiones_dia": r.get("doses_per_day"),
            "dosis_fija_valor": fixed,
            "dosis_fija_valor_max": r.get("fixed_dose_max"),
            "dosis_fija_mg": fixed if unit == "mg" else None,
            "max_dosis_valor": max_single,
            "max_dosis_valorkg": r.get("max_single_per_kg"),
            "max_dosis_mg": max_single if unit == "mg" else None,
            "max_dia_valor": max_daily,
            "max_dia_valorkg": max_daily_kg,
            "max_dia_mg": max_daily if unit == "mg" else None,
            "max_dia_mgkg": max_daily_kg if unit == "mg" else None,
            "frecuencia_texto": r.get("frequency_text"),
            "duracion": r.get("duration"),
            "notas": r.get("clinical_notes"),
            "nota_renal": r.get("renal_note"),
            "nivel_uso": r.get("use_level") or "GENERAL",
            "permite_conversion_volumen": _si(r.get("allow_volume_conversion")),
            "automatizable": _si(r.get("automatizable")),
            "estado": r.get("status"),
            "fuente": src.get("title"),
            "pagina_fuente": src.get("page"),
            "url_fuente": src.get("url"),
            "fecha_revision": _source_date(src.get("last_verified")) or _source_date(r.get("reviewed_at")),
        }

    def pediatric_rules(self, med_id):
        # A diferencia del resto de módulos, pediatría muestra también
        # PENDING_REVIEW como referencia bibliográfica claramente etiquetada.
        rows = self._visible_pediatric_for_med(med_id)
        med = self._med_by_med_id.get(med_id) or {}
        out = []
        for r in rows:
            x = self._map_pediatric(r)
            x["med_id"] = med_id
            x["principio_activo"] = med.get("generic_name")
            out.append(x)
        out.sort(
            key=lambda r: (
                0 if str(r.get("estado") or "").upper() == "PUBLISHED" else 1,
                r.get("indicacion") or "",
                r.get("via") or "",
                r.get("rule_id") or "",
            )
        )
        return out

    def pediatric_indications(self, med_id):
        """Resume TODAS las indicaciones visibles, separando estado."""
        grouped = {}
        for r in self.pediatric_rules(med_id):
            ind = r.get("indicacion") or "Sin indicación"
            g = grouped.setdefault(
                ind,
                {
                    "indicacion": ind,
                    "vias": set(),
                    "reglas": 0,
                    "publicadas": 0,
                    "pendientes": 0,
                },
            )
            if r.get("via"):
                g["vias"].add(r["via"])
            g["reglas"] += 1
            status = str(r.get("estado") or "").upper()
            if status == "PUBLISHED":
                g["publicadas"] += 1
            elif status == "PENDING_REVIEW":
                g["pendientes"] += 1
        out = []
        for g in grouped.values():
            out.append(
                {
                    "indicacion": g["indicacion"],
                    "vias": ", ".join(sorted(g["vias"])),
                    "reglas": g["reglas"],
                    "publicadas": g["publicadas"],
                    "pendientes": g["pendientes"],
                }
            )
        return sorted(out, key=lambda r: normalize_text(r["indicacion"]))

    # ---------- Renal ----------
    def _map_renal_rule(self, r):
        src = self._source(r.get("source_id"))
        return {
            "id": r.get("id"),
            "rule_id": f"REN-SB-{str(r.get('id') or '')[:8]}",
            "indicacion": r.get("indication"),
            "poblacion": r.get("population"),
            "via": r.get("route"),
            "metrica_renal": r.get("renal_metric"),
            "rango": r.get("range_text"),
            "limite_inferior": r.get("lower_limit"),
            "limite_superior": r.get("upper_limit"),
            "inferior_inclusivo": _si(r.get("lower_inclusive")),
            "superior_inclusivo": _si(r.get("upper_inclusive")),
            "regimen_ajustado": r.get("adjusted_regimen"),
            "tipo_regla": r.get("rule_type"),
            "notas": r.get("notes"),
            "automatizable": _si(r.get("automatizable")),
            "estado": r.get("status"),
            "fuente": src.get("title"),
            "pagina_fuente": src.get("page"),
            "url_fuente": src.get("url"),
            "fecha_revision": _source_date(src.get("last_verified")) or _source_date(r.get("reviewed_at")),
        }

    def renal_rules(self, med_id):
        rows = self._published_for_med("renal_rules", med_id)
        out = [self._map_renal_rule(r) for r in rows]
        out.sort(key=lambda r: (r.get("indicacion") or "", r.get("rule_id") or ""))
        return out

    def renal_reference_rules(self, med_id):
        """Reglas renales PUBLISHED deliberadamente no automatizables.

        Son referencias clínicas estructuradas y deben ser visibles en la UI,
        pero nunca se convierten en cálculo automático por el solo hecho de
        estar publicadas.
        """
        return [r for r in self.renal_rules(med_id) if r.get("automatizable") != "SI"]

    def renal_indications(self, med_id):
        grouped = {}
        for r in self.renal_rules(med_id):
            if r.get("automatizable") != "SI":
                continue
            ind = r.get("indicacion") or "Sin indicación"
            g = grouped.setdefault(ind, {"indicacion": ind, "vias": set(), "reglas": 0})
            if r.get("via"):
                g["vias"].add(r["via"])
            g["reglas"] += 1
        return [
            {"indicacion": g["indicacion"], "vias": ", ".join(sorted(g["vias"])), "reglas": g["reglas"]}
            for g in sorted(grouped.values(), key=lambda x: normalize_text(x["indicacion"]))
        ]

    def _map_renal_biblio(self, r):
        src = self._source(r.get("source_id"))
        table_num = r.get("table_number")
        page_num = r.get("page_number")
        image = None
        if table_num is not None and page_num is not None:
            image = f"tabla_{int(table_num):02d}_pag_{int(page_num):02d}.png"
        return {
            "id": r.get("id"),
            "ref_id": f"RB-SB-{str(r.get('id') or '')[:8]}",
            "principio_activo": r.get("drug_name_source"),
            "dosis_fr_normal": r.get("normal_dose"),
            "metodo": r.get("adjustment_method"),
            "crcl_100_50": r.get("crcl_100_50"),
            "crcl_50_10": r.get("crcl_50_10"),
            "crcl_lt10": r.get("crcl_lt_10"),
            "suplemento_hd": r.get("hemodialysis"),
            "dosis_hfvvc": r.get("hfvvh"),
            "notas": r.get("recommendations"),
            "table": table_num,
            "page": page_num,
            "estado": "VERIFICADA" if r.get("verified") else "PENDIENTE",
            "imagen": image,
            "fuente": src.get("title"),
            "url_fuente": src.get("url"),
            "fecha_fuente": _source_date(src.get("last_verified")),
        }

    def renal_biblio(self, med_id):
        rows = self._published_for_med("renal_bibliography", med_id)
        out = [self._map_renal_biblio(r) for r in rows]
        out.sort(key=lambda r: ((r.get("table") or 999), (r.get("principio_activo") or "")))
        return out

    def _all_renal_biblio(self):
        if self._renal_biblio_cache is None:
            rows = self._fetch_all("renal_bibliography")
            rows = [r for r in rows if r.get("status") == "PUBLISHED"]
            self._renal_biblio_cache = [self._map_renal_biblio(r) for r in rows]
        return self._renal_biblio_cache

    def search_renal_biblio(self, query=""):
        q = normalize_text(query)
        rows = self._all_renal_biblio()
        if q:
            rows = [r for r in rows if q in normalize_text(r.get("principio_activo"))]
        return sorted(rows, key=lambda r: (normalize_text(r.get("principio_activo")), r.get("table") or 999))

    # ---------- Toxicology ----------
    def toxicology(self, med_id):
        medication_uuid = self._uuid_by_med_id.get(med_id)
        if not medication_uuid:
            return None
        res = (
            self.client.table("toxicology")
            .select("*")
            .eq("medication_id", medication_uuid)
            .eq("status", "PUBLISHED")
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        r = rows[0]
        src = self._source(r.get("source_id"))
        original = {}
        raw = r.get("original_source_text")
        if raw:
            try:
                original = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except Exception:
                original = {}
        return {
            "id_revision": med_id,
            "clase_toxicologica": r.get("toxicological_class"),
            "dosis_toxica_base": r.get("original_toxic_dose"),
            "unidad_medida": original.get("unidad_medida"),
            "concentracion": original.get("concentracion"),
            "unidad_referencia": original.get("unidad_referencia"),
            "sintomas_base": original.get("sintomas_base"),
            "antidoto_manejo_base": original.get("antidoto_manejo_base"),
            "dosis_toxica_corregida": r.get("reviewed_toxic_threshold"),
            "tipo_umbral": r.get("threshold_type"),
            "manifestaciones_clave": r.get("clinical_manifestations"),
            "sintomas_generales_definicion": r.get("general_symptoms_detail"),
            "sintomas_intoxicacion_detallados": r.get("toxicity_symptoms_detailed") or r.get("clinical_manifestations"),
            "fuente_sintomas_detallados": r.get("toxicity_detail_source"),
            "estado_sintomas_detallados": r.get("toxicity_detail_status"),
            "manejo_corregido": r.get("initial_management"),
            "antidoto_especifico": r.get("specific_treatment") or r.get("antidote"),
            "estado_revision": r.get("validation_status"),
            "nivel_evidencia": r.get("evidence_level"),
            # V1 core migration did not include the numeric automation columns.
            # Keep the automatic calculator disabled until the optional V1.1
            # toxicology patch is applied rather than inferring a number from text.
            "umbral_mgkg_automatizable": r.get("threshold_numeric_mgkg"),
            "etiqueta_umbral": r.get("threshold_label"),
            "permitir_comparacion_automatica": _si(
                r.get("automatic_comparison") and r.get("threshold_numeric_mgkg") is not None
            ),
            "fuente_principal": src.get("url") or src.get("title"),
            "fecha_revision": _source_date(r.get("reviewed_at")) or _source_date(src.get("last_verified")),
        }

    # ---------- Toxicología no farmacológica y antídotos ----------
    # Estos dos bloques forman parte de la base original de MedCalc y permanecen
    # como CSV en la raíz del repositorio. La primera migración Supabase no los
    # incluyó; por eso NO deben depender exclusivamente de medcalc.db.
    def _fallback_all(self, table):
        """Compatibilidad con instalaciones antiguas que aún tengan medcalc.db."""
        if not self.fallback_db_path or not self.fallback_db_path.exists():
            return []
        try:
            con = sqlite3.connect(self.fallback_db_path)
            con.row_factory = sqlite3.Row
            rows = [dict(r) for r in con.execute(f"SELECT * FROM {table}").fetchall()]
            con.close()
            return rows
        except Exception:
            return []

    def _original_csv_rows(self, filename):
        """Carga una tabla histórica directamente desde el repositorio.

        Se usa utf-8-sig para aceptar el BOM de los CSV originales. No modifica
        ni 'cura' el contenido: devuelve las columnas tal como están almacenadas.
        """
        candidates = [Path(__file__).resolve().parent / filename]
        if self.fallback_db_path:
            candidates.append(self.fallback_db_path.resolve().parent / filename)

        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open('r', encoding='utf-8-sig', newline='') as fh:
                    return [dict(row) for row in csv.DictReader(fh)]
            except Exception:
                continue
        return []

    def _original_other_tox(self):
        rows = self._original_csv_rows('toxicos_drogas_plaguicidas_metales.csv')
        if not rows:
            rows = self._fallback_all('other_tox')

        # El CSV original conserva una primera fila descriptiva ("Droga /
        # Síntomas de Intoxicación / Antídoto-Tratamiento"); no es un tóxico.
        out = []
        for r in rows:
            name = str(r.get('toxico') or '').strip()
            if not name:
                continue
            if normalize_text(name) in {'droga', 'toxico'} and 'sintomas de intoxicacion' in normalize_text(r.get('sintomas_base')):
                continue
            out.append(r)
        return out

    def _original_antidotes(self):
        rows = self._original_csv_rows('antidotos.csv')
        if not rows:
            rows = self._fallback_all('antidotes')
        return [r for r in rows if str(r.get('toxico_sindrome') or '').strip()]

    def search_other_tox(self, query=''):
        rows = self._original_other_tox()
        q = normalize_text(query)
        if q:
            rows = [
                r for r in rows
                if q in normalize_text(r.get('toxico'))
                or q in normalize_text(r.get('sintomas_base'))
                or q in normalize_text(r.get('antidoto_tratamiento_base'))
            ]
        return sorted(rows, key=lambda r: normalize_text(r.get('toxico')))

    def search_antidotes(self, query=''):
        rows = self._original_antidotes()
        q = normalize_text(query)
        if q:
            rows = [
                r for r in rows
                if q in normalize_text(r.get('toxico_sindrome'))
                or q in normalize_text(r.get('antidoto_base'))
                or q in normalize_text(r.get('dosis_base'))
                or q in normalize_text(r.get('observaciones_base'))
            ]
        return sorted(rows, key=lambda r: (
            normalize_text(r.get('toxico_sindrome')),
            normalize_text(r.get('antidoto_base')),
        ))

    # ---------- Sources ----------
    def sources(self):
        rows = []
        for r in self._sources_cache:
            rows.append({
                "codigo": r.get("source_type"),
                "fuente": r.get("title"),
                "url": r.get("url"),
                "fecha_revision": _source_date(r.get("last_verified")),
                "organizacion": r.get("organization"),
                "autores": r.get("authors"),
            })
        return sorted(rows, key=lambda r: normalize_text(r.get("fuente")))
