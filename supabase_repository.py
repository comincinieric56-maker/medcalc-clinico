from pathlib import Path
import csv
import json
import difflib
import re
import sqlite3
import unicodedata

from supabase import create_client

SCHEMA_VERSION = "MEDCALC_SUPABASE_V3"
REPOSITORY_FEATURE_VERSION = "PREGNANCY_V1_V8_4_1_ELECTROLYTES_V1_TOXCSV_V2_FULLCOVERAGE_V1_RENALGLOBAL_V11"


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
            "medication_id,pediatric_status,renal_status,toxicology_status,pregnancy_status,clinical_priority,pediatric_note,renal_note,toxicology_note,pregnancy_note",
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
        self._local_csv_cache = {}

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

    def _fetch_optional(self, table, columns="*"):
        """Obtiene una tabla auxiliar sin derribar la aplicación si no está disponible.

        Algunas tablas de Hidroelectrolitos fueron añadidas por migraciones sucesivas.
        En instalaciones donde una tabla auxiliar todavía no exista o no sea visible por
        RLS, el módulo debe degradar a una lista vacía en vez de lanzar AttributeError.
        """
        try:
            return self._fetch_all(table, columns)
        except Exception:
            return []


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
        pregnancy = self._fetch_all("pregnancy_safety", "medication_id,status")
        pregnancy = [r for r in pregnancy if r.get("status") == "PUBLISHED"]

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
            "renal_meds": len({r["medication_id"] for r in renals if r.get("automatizable")}),
            "renal_biblio": len(refs),
            "renal_coverage_meds": len(self._medications),
            "renal_specific_meds": len({
                self._uuid_by_med_id.get(str(r.get("med_id") or "").strip())
                for r in self._csv_rows("ajuste_renal.csv")
                if self._uuid_by_med_id.get(str(r.get("med_id") or "").strip())
            } | {r.get("medication_id") for r in renals if r.get("medication_id")}),
            "toxicology": len(tox),
            "toxicology_coverage_meds": len(self._medications),
            "toxicology_specific_meds": len({
                self._uuid_by_med_id.get(str(r.get("id_revision") or "").strip())
                for r in self._csv_rows("toxicos_medicamentos_revisados_v3.csv")
                if self._uuid_by_med_id.get(str(r.get("id_revision") or "").strip())
            } | {
                r.get("medication_id") for r in self._fetch_all("toxicology", "medication_id,status")
                if r.get("status") == "PUBLISHED" and r.get("medication_id")
            }),
            "pregnancy": len(pregnancy),
            "pregnancy_meds": len({r["medication_id"] for r in pregnancy}),
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
        pregnancy = self.pregnancy_safety(med_id)
        status = self._status_by_uuid.get(med.get("id")) or {}
        return {
            "id": med.get("id"),
            "med_id": med.get("med_id"),
            "principio_activo": med.get("generic_name"),
            "search_name": normalize_text(med.get("generic_name")),
            "pediatric_status": status.get("pediatric_status") or "PENDING_REVIEW",
            "renal_status": status.get("renal_status") or "PENDING_REVIEW",
            "toxicology_status": status.get("toxicology_status") or "PENDING_REVIEW",
            "pregnancy_status": status.get("pregnancy_status") or "PENDING_REVIEW",
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
            "renal_rule_count": sum(1 for r in renals if r.get("automatizable") == "SI"),
            "renal_biblio_count": len(refs),
            "toxicology_available": 1 if tox else 0,
            "pregnancy_available": 1 if pregnancy else 0,
            "pregnancy_recommendation": pregnancy.get("recommendation") if pregnancy else None,
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

    def _local_renal_rule_rows(self, med_id):
        """Reglas renales locales validadas usadas solo si Supabase no tiene reglas PUBLISHED."""
        med = self._med_by_med_id.get(med_id) or {}
        med_name = normalize_text(med.get("generic_name"))
        rows = self._csv_rows("ajuste_renal.csv")
        out = []
        for r in rows:
            row_med_id = str(r.get("med_id") or "").strip()
            row_name = normalize_text(r.get("principio_activo"))
            if row_med_id != med_id and (not med_name or row_name != med_name):
                continue
            auto = str(r.get("automatizable") or "").strip().upper() in {"SI", "SÍ", "TRUE", "1", "YES"}
            out.append({
                "id": r.get("rule_id"),
                "rule_id": r.get("rule_id") or f"REN-LOCAL-{med_id}",
                "indicacion": r.get("indicacion"),
                "poblacion": r.get("poblacion"),
                "via": r.get("via"),
                "metrica_renal": r.get("metrica_renal"),
                "rango": r.get("rango"),
                "limite_inferior": r.get("limite_inferior"),
                "limite_superior": r.get("limite_superior"),
                "inferior_inclusivo": r.get("inferior_inclusivo") or "NO",
                "superior_inclusivo": r.get("superior_inclusivo") or "NO",
                "regimen_ajustado": r.get("regimen_ajustado"),
                "tipo_regla": r.get("tipo_regla"),
                "notas": r.get("notas"),
                "automatizable": "SI" if auto else "NO",
                "estado": r.get("estado") or "VALIDADO_LOCAL",
                "validation_class": "CURRENT_AUTO" if auto else "CURRENT_REFERENCE",
                "validation_note": "Fallback local versionado; se usa únicamente cuando Supabase no expone una regla PUBLISHED para este MED-ID.",
                "fuente": r.get("fuente"),
                "pagina_fuente": None,
                "url_fuente": r.get("url_fuente"),
                "fecha_revision": r.get("fecha_revision"),
                "coverage_status": "SPECIFIC_LOCAL_RULE",
            })
        return sorted(out, key=lambda r: (r.get("indicacion") or "", r.get("rango") or "", r.get("rule_id") or ""))

    def _renal_regulatory_reference(self, med_id):
        """Devuelve la mejor evidencia renal regulatoria local disponible.

        Prioridad: AEMPS/CIMA V6 > Health Canada/EMA V5 > openFDA V4 >
        DailyMed V3. Toda esta capa es CURRENT_REFERENCE, nunca automática.
        """
        med = self._med_by_med_id.get(med_id) or {}
        name = med.get("generic_name") or med_id

        def find_row(filename):
            for row in self._csv_rows(filename):
                if str(row.get("med_id") or "").strip() == med_id:
                    return row
            return None

        evidence = None

        # V11 · TGA Australia
        row = find_row("generated_renal_global_v11/renal_global_v11_tga_results.csv")
        if row and str(row.get("tga_status") or "").upper() == "ACCEPT":
            evidence = {
                "source_kind": "TGA_AU_V11",
                "source": "Therapeutic Goods Administration · Product Information",
                "url": row.get("tga_pi_url") or row.get("tga_artg_page"),
                "reason": row.get("tga_reason"),
                "text": row.get("tga_renal_text"),
                "locator": f"ARTG {row.get('tga_artg_id') or '—'}",
                "date": "2026-09-27",
            }

        # V10 · MHRA United Kingdom
        if evidence is None:
            row = find_row("generated_renal_global_v10/renal_global_v10_mhra_results.csv")
        else:
            row = None
        if row and str(row.get("mhra_status") or "").upper() == "ACCEPT":
            evidence = {
                "source_kind": "MHRA_UK_V10",
                "source": "MHRA Products · Summary of Product Characteristics",
                "url": row.get("mhra_spc_url") or row.get("mhra_product_page"),
                "reason": row.get("mhra_reason"),
                "text": row.get("mhra_renal_text"),
                "locator": row.get("mhra_product_page") or "SmPC",
                "date": "2026-09-27",
            }

        # V9 · ANSM / Base de Données Publique des Médicaments (France)
        if evidence is None:
            row = find_row("generated_renal_global_v9/renal_global_v9_ansm_results.csv")
        else:
            row = None
        if row and str(row.get("ansm_status") or "").upper() == "ACCEPT":
            evidence = {
                "source_kind": "ANSM_BDPM_V9",
                "source": "ANSM / Base de Données Publique des Médicaments · RCP",
                "url": row.get("ansm_source_url"),
                "reason": row.get("ansm_reason"),
                "text": row.get("ansm_renal_text"),
                "locator": f"Code CIS {row.get('ansm_cis') or '—'}",
                "date": "2026-09-27",
            }

        # V8 · Medsafe New Zealand
        if evidence is None:
            row = find_row("generated_renal_global_v8/renal_global_v8_medsafe_results.csv")
            if row and str(row.get("medsafe_status") or "").upper() == "ACCEPT":
                evidence = {
                    "source_kind": "MEDSAFE_NZ_V8",
                    "source": "Medsafe New Zealand · Data Sheet",
                    "url": row.get("medsafe_source_url"),
                    "reason": row.get("medsafe_reason"),
                    "text": row.get("medsafe_renal_text"),
                    "locator": row.get("medsafe_product_name") or "Data Sheet",
                    "date": "2026-09-27",
                }

        # V7 · ISP Chile
        if evidence is None:
            row = find_row("generated_renal_global_v7/renal_global_v7_isp_results.csv")
        else:
            row = None
        if row and str(row.get("isp_status") or "").upper() == "ACCEPT":
            evidence = {
                "source_kind": "ISP_CHILE_V7",
                "source": "Instituto de Salud Pública de Chile · Folleto al profesional",
                "url": row.get("isp_document_url") or row.get("isp_index_url"),
                "reason": row.get("isp_reason"),
                "text": row.get("isp_renal_text"),
                "locator": f"Registro sanitario {row.get('isp_registro') or '—'} · {row.get('isp_year') or '—'}",
                "date": str(row.get("isp_year") or "2026"),
            }

        # V6 · AEMPS/CIMA
        if evidence is None:
            row = find_row("generated_renal_global_v6/renal_global_v6_cima_results.csv")
        else:
            row = None
        if row and str(row.get("cima_status") or "").upper() == "ACCEPT":
            evidence = {
                "source_kind": "AEMPS_CIMA_V6",
                "source": "AEMPS CIMA · ficha técnica oficial",
                "url": row.get("cima_source_url"),
                "reason": row.get("cima_reason"),
                "text": row.get("cima_renal_text"),
                "locator": f"Nº registro {row.get('cima_nregistro') or '—'}",
                "date": "2026-09-27",
            }

        # V5 · Health Canada / EMA
        if evidence is None:
            row = find_row("generated_renal_global_v5/renal_global_v5_pending_input_full_results.csv")
            if row:
                primary = str(row.get("v5_primary_source") or "").upper()
                if primary == "HEALTH_CANADA" and str(row.get("hc_status") or "").upper() == "ACCEPT":
                    evidence = {
                        "source_kind": "HEALTH_CANADA_V5",
                        "source": "Health Canada · Product Monograph",
                        "url": row.get("hc_monograph_url"),
                        "reason": row.get("hc_reason"),
                        "text": row.get("hc_renal_text"),
                        "locator": f"DIN {row.get('hc_din') or '—'}",
                        "date": row.get("hc_monograph_date") or "2026-09-27",
                    }
                elif primary == "EMA" and str(row.get("ema_status") or "").upper() == "ACCEPT":
                    evidence = {
                        "source_kind": "EMA_V5",
                        "source": "European Medicines Agency · Product Information",
                        "url": row.get("ema_pdf_url") or row.get("ema_product_page"),
                        "reason": row.get("ema_reason"),
                        "text": row.get("ema_renal_text"),
                        "locator": "Official Product Information",
                        "date": "2026-09-27",
                    }
                elif str(row.get("hc_status") or "").upper() == "ACCEPT":
                    evidence = {
                        "source_kind": "HEALTH_CANADA_V5",
                        "source": "Health Canada · Product Monograph",
                        "url": row.get("hc_monograph_url"),
                        "reason": row.get("hc_reason"),
                        "text": row.get("hc_renal_text"),
                        "locator": f"DIN {row.get('hc_din') or '—'}",
                        "date": row.get("hc_monograph_date") or "2026-09-27",
                    }
                elif str(row.get("ema_status") or "").upper() == "ACCEPT":
                    evidence = {
                        "source_kind": "EMA_V5",
                        "source": "European Medicines Agency · Product Information",
                        "url": row.get("ema_pdf_url") or row.get("ema_product_page"),
                        "reason": row.get("ema_reason"),
                        "text": row.get("ema_renal_text"),
                        "locator": "Official Product Information",
                        "date": "2026-09-27",
                    }

        # V4 · openFDA
        if evidence is None:
            row = find_row("generated_renal_master_v4_openfda/renal_v4_openfda_accepted.csv")
            if row:
                evidence = {
                    "source_kind": "OPENFDA_V4",
                    "source": "U.S. FDA · openFDA Drug Label",
                    "url": row.get("v4_source_url"),
                    "reason": row.get("v4_reason"),
                    "text": row.get("v4_renal_text"),
                    "locator": row.get("v4_renal_fields"),
                    "date": "2026-09-21",
                }

        # V3 · DailyMed
        if evidence is None:
            row = find_row("generated_renal_master_v3/renal_v3_identity_accepted.csv")
            if row:
                evidence = {
                    "source_kind": "DAILYMED_V3",
                    "source": "DailyMed · U.S. National Library of Medicine",
                    "url": row.get("source_url"),
                    "reason": row.get("v2_reason"),
                    "text": row.get("renal_text"),
                    "locator": row.get("renal_section_titles"),
                    "date": "2026-09-21",
                }

        if evidence is None:
            return None

        text = str(evidence.get("text") or "").strip()
        if not text:
            return None

        reason = str(evidence.get("reason") or "").upper()
        if "NO_ADJUSTMENT" in reason:
            rule_type = "NO_AJUSTE"
        elif "NOT_RECOMMENDED" in reason:
            rule_type = "PRECAUCION"
        elif "DIALYSIS" in reason:
            rule_type = "DIALISIS"
        elif "DOSING" in reason or "THRESHOLD" in reason:
            rule_type = "REGIMEN"
        else:
            rule_type = "REFERENCIA"

        display_text = text[:7000]
        return {
            "id": f"REN-REG-{med_id}",
            "rule_id": f"REN-REG-{med_id}",
            "indicacion": f"{name} — referencia renal regulatoria",
            "poblacion": "Adulto / según ficha regulatoria",
            "via": None,
            "metrica_renal": None,
            "rango": "Texto regulatorio actual",
            "limite_inferior": None,
            "limite_superior": None,
            "inferior_inclusivo": "NO",
            "superior_inclusivo": "NO",
            "regimen_ajustado": display_text,
            "tipo_regla": rule_type,
            "notas": (
                f"{evidence.get('source_kind')} · {evidence.get('reason') or 'acción renal explícita'}"
                + (f" · {evidence.get('locator')}" if evidence.get("locator") else "")
                + ". Referencia clínica no automatizable; conservar el texto de la ficha técnica."
            ),
            "automatizable": "NO",
            "estado": "PUBLISHED_LOCAL_REGULATORY",
            "validation_class": "CURRENT_REFERENCE",
            "validation_note": (
                "Identidad farmacológica estricta y recomendación renal explícita recuperada "
                "de una fuente regulatoria actual. No se transforma automáticamente en bandas numéricas."
            ),
            "fuente": evidence.get("source"),
            "pagina_fuente": evidence.get("locator"),
            "url_fuente": evidence.get("url"),
            "fecha_revision": evidence.get("date"),
            "coverage_status": "SPECIFIC_REGULATORY_REFERENCE",
        }

    def _renal_multisource_reference(self, med_id):
        """Mejor referencia renal disponible para un MED-ID sin regla Supabase."""
        regulatory = self._renal_regulatory_reference(med_id)
        if regulatory:
            return regulatory

        rows = self._csv_rows("generated_renal_multisource/renal_multisource_matrix_1122.csv")
        match = next((r for r in rows if str(r.get("med_id") or "").strip() == med_id), None)
        med = self._med_by_med_id.get(med_id) or {}
        name = med.get("generic_name") or med_id

        if match:
            v2 = str(match.get("v2_reason") or "").upper()
            v4 = str(match.get("v4_reason") or "").upper()
            reason = v4 or v2
            resolution = str(match.get("resolution") or "UNRESOLVED").upper()
            next_action = str(match.get("next_action") or "").strip()

            if "NO_ADJUSTMENT" in reason:
                regimen = (
                    "La auditoría regulatoria identificó una señal explícita de que no se requiere "
                    "ajuste renal, pero todavía no existe texto regulatorio local recuperado para "
                    "mostrar como referencia específica."
                )
            elif "NOT_RECOMMENDED" in reason:
                regimen = (
                    "La auditoría regulatoria identificó una restricción/no recomendación relacionada "
                    "con función renal. Revisar la ficha técnica específica antes de prescribir."
                )
            elif "DIALYSIS_DOSING" in reason:
                regimen = (
                    "La auditoría regulatoria identificó información específica para diálisis, pero "
                    "la pauta exacta todavía no está disponible en la capa visible."
                )
            elif "RENAL_DOSING" in reason:
                regimen = (
                    "Existe una señal regulatoria de ajuste renal; la pauta exacta continúa pendiente "
                    "de recuperación/estructuración para esta ficha."
                )
            else:
                regimen = (
                    "No se dispone todavía de una pauta renal específica suficientemente validada. "
                    "No modificar dosis por inferencia; verificar ficha técnica o guía vigente."
                )

            notes = (
                f"Auditoría multisource: {resolution}. "
                f"DailyMed confirmado: {match.get('dailymed_v3_confirmed')}; "
                f"openFDA confirmado: {match.get('openfda_v4_confirmed')}. "
                f"Motivo: {reason or 'sin código específico'}. "
                f"Siguiente acción: {next_action or 'revisión de fuente actual'}."
            )
        else:
            regimen = (
                "No se dispone de una pauta renal específica estructurada para este MED-ID. "
                "No modificar dosis por inferencia; verificar ficha técnica/guía vigente."
            )
            notes = "MED-ID no localizado en la matriz renal multisource 1122."

        return {
            "id": f"REN-COVERAGE-{med_id}",
            "rule_id": f"REN-COVERAGE-{med_id}",
            "indicacion": "Cobertura renal del medicamento",
            "poblacion": "Adulto",
            "via": None,
            "metrica_renal": None,
            "rango": "Referencia no automatizable",
            "limite_inferior": None,
            "limite_superior": None,
            "inferior_inclusivo": "NO",
            "superior_inclusivo": "NO",
            "regimen_ajustado": regimen,
            "tipo_regla": "REFERENCE_COVERAGE",
            "notas": notes,
            "automatizable": "NO",
            "estado": "COVERAGE_REFERENCE",
            "validation_class": "CURRENT_REFERENCE",
            "validation_note": f"{name}: cobertura de seguridad; no equivale a pauta posológica automática.",
            "fuente": "MEDCALC Renal Multisource 1122",
            "pagina_fuente": None,
            "url_fuente": None,
            "fecha_revision": "2026-09-27",
            "coverage_status": "GENERAL_RENAL_COVERAGE",
        }

    def renal_rules(self, med_id):
        rows = self._published_for_med("renal_rules", med_id)
        out = [self._map_renal_rule(r) for r in rows]
        if out:
            out.sort(key=lambda r: (r.get("indicacion") or "", r.get("rule_id") or ""))
            return out

        local = self._local_renal_rule_rows(med_id)
        if local:
            return local

        if med_id in self._med_by_med_id:
            return [self._renal_multisource_reference(med_id)]
        return []

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
        if out:
            out.sort(key=lambda r: ((r.get("table") or 999), (r.get("principio_activo") or "")))
            return out

        med = self._med_by_med_id.get(med_id) or {}
        med_name = normalize_text(med.get("generic_name"))
        local = []
        for r in self._csv_rows("renal_biblio_verificada_2025.csv"):
            row_med_id = str(r.get("med_id") or "").strip()
            names = {
                normalize_text(r.get("catalogo_nombre")),
                normalize_text(r.get("principio_activo")),
            }
            if row_med_id != med_id and (not med_name or med_name not in names):
                continue
            local.append({
                "id": r.get("ref_id"),
                "ref_id": r.get("ref_id"),
                "principio_activo": r.get("principio_activo") or med.get("generic_name"),
                "dosis_fr_normal": r.get("dosis_fr_normal"),
                "metodo": r.get("metodo"),
                "crcl_100_50": r.get("crcl_100_50"),
                "crcl_50_10": r.get("crcl_50_10"),
                "crcl_lt10": r.get("crcl_lt10"),
                "suplemento_hd": r.get("suplemento_hd"),
                "dosis_hfvvc": r.get("dosis_hfvvc"),
                "notas": r.get("notas"),
                "table": r.get("table"),
                "page": r.get("page"),
                "estado": r.get("estado") or "TRANSCRIPCION_VERIFICADA_IMAGEN",
                "imagen": r.get("imagen"),
                "fuente": r.get("fuente"),
                "url_fuente": r.get("url_fuente"),
                "fecha_fuente": r.get("fecha_fuente"),
                "coverage_status": "SPECIFIC_LOCAL_BIBLIO",
            })
        return sorted(local, key=lambda r: ((r.get("table") or "999"), normalize_text(r.get("principio_activo"))))

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
            return self._local_medication_toxicology(med_id)
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

    def _local_medication_toxicology(self, med_id):
        """Fallback toxicológico por MED-ID para evitar medicamentos sin ficha visible."""
        med = self._med_by_med_id.get(med_id) or {}
        med_name = normalize_text(med.get("generic_name"))

        match = None
        for r in self._csv_rows("toxicos_medicamentos_revisados_v3.csv"):
            if str(r.get("id_revision") or "").strip() == med_id:
                match = r
                break
            if med_name and normalize_text(r.get("principio_activo")) == med_name:
                match = r
                break

        if match:
            threshold_numeric = match.get("umbral_mgkg_automatizable")
            compare = str(match.get("permitir_comparacion_automatica") or "").strip().upper()
            return {
                "id_revision": med_id,
                "clase_toxicologica": match.get("clase_toxicologica"),
                "dosis_toxica_base": match.get("dosis_toxica_base"),
                "unidad_medida": match.get("unidad_medida"),
                "concentracion": match.get("concentracion"),
                "unidad_referencia": match.get("unidad_referencia"),
                "sintomas_base": match.get("sintomas_base"),
                "antidoto_manejo_base": match.get("antidoto_manejo_base"),
                "dosis_toxica_corregida": match.get("dosis_toxica_corregida"),
                "tipo_umbral": match.get("tipo_umbral"),
                "manifestaciones_clave": match.get("manifestaciones_clave"),
                "sintomas_generales_definicion": None,
                "sintomas_intoxicacion_detallados": match.get("manifestaciones_clave") or match.get("sintomas_base"),
                "fuente_sintomas_detallados": match.get("fuente_principal"),
                "estado_sintomas_detallados": match.get("estado_revision"),
                "manejo_corregido": match.get("manejo_corregido"),
                "antidoto_especifico": match.get("antidoto_especifico"),
                "estado_revision": match.get("estado_revision"),
                "nivel_evidencia": match.get("nivel_evidencia"),
                "umbral_mgkg_automatizable": threshold_numeric or None,
                "etiqueta_umbral": match.get("etiqueta_umbral"),
                "permitir_comparacion_automatica": "SI" if compare in {"SI", "SÍ", "TRUE", "1", "YES"} and threshold_numeric not in (None, "") else "NO",
                "fuente_principal": match.get("fuente_principal") or match.get("fuente_secundaria"),
                "fecha_revision": match.get("fecha_revision"),
                "coverage_status": "SPECIFIC_LOCAL_TOX_V3",
                "specific_data_available": True,
            }

        name = med.get("generic_name") or med_id
        return {
            "id_revision": med_id,
            "clase_toxicologica": "Cobertura general de seguridad · datos específicos pendientes",
            "dosis_toxica_base": None,
            "unidad_medida": None,
            "concentracion": None,
            "unidad_referencia": None,
            "sintomas_base": None,
            "antidoto_manejo_base": None,
            "dosis_toxica_corregida": "No existe un umbral toxicológico específico validado en la base para este medicamento; no automatizar.",
            "tipo_umbral": "SIN_UMBRAL_NUMERICO_VALIDADO",
            "manifestaciones_clave": (
                f"{name}: no se dispone todavía de una ficha toxicológica específica validada. "
                "La presentación de sobredosis depende del mecanismo farmacológico, dosis, formulación, "
                "comorbilidades y coingestas."
            ),
            "sintomas_generales_definicion": None,
            "sintomas_intoxicacion_detallados": (
                "No usar ausencia de una ficha específica como evidencia de baja toxicidad. "
                "Valorar toxíndrome, estado neurológico, ventilación, hemodinamia y ECG según el contexto."
            ),
            "fuente_sintomas_detallados": "https://cituc.uc.cl/",
            "estado_sintomas_detallados": "COBERTURA_GENERAL_NO_ESPECIFICA",
            "manejo_corregido": (
                "ABCDE y tratamiento de soporte dirigido al cuadro clínico. Considerar ECG, glucemia, "
                "electrolitos, función renal/hepática y otras pruebas según mecanismo y síntomas. "
                "Ingesta intencional, dosis desconocida, formulación de liberación prolongada o paciente "
                "sintomático: evaluación urgente y consulta a toxicología/CIT."
            ),
            "antidoto_especifico": (
                "No asumir ausencia de antídoto por falta de ficha específica. Verificar toxicología clínica/CIT "
                "y la ficha técnica vigente del medicamento."
            ),
            "estado_revision": "COBERTURA_GENERAL_NO_ESPECIFICA",
            "nivel_evidencia": "COBERTURA_DE_SEGURIDAD",
            "umbral_mgkg_automatizable": None,
            "etiqueta_umbral": None,
            "permitir_comparacion_automatica": "NO",
            "fuente_principal": "https://cituc.uc.cl/",
            "fecha_revision": "2026-09-27",
            "coverage_status": "GENERAL_TOX_COVERAGE",
            "specific_data_available": False,
        }

    # ---------- Pregnancy safety ----------
    def pregnancy_safety(self, med_id):
        medication_uuid = self._uuid_by_med_id.get(med_id)
        if not medication_uuid:
            return None
        res = (
            self.client.table("pregnancy_safety")
            .select("*")
            .eq("medication_id", medication_uuid)
            .eq("status", "PUBLISHED")
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        row = dict(rows[0])

        links_res = (
            self.client.table("pregnancy_safety_sources")
            .select("source_id,evidence_role,evidence_note")
            .eq("pregnancy_safety_id", row.get("id"))
            .execute()
        )
        sources = []
        for link in (links_res.data or []):
            src = self._source(link.get("source_id"))
            sources.append({
                "role": link.get("evidence_role"),
                "evidence_note": link.get("evidence_note"),
                "title": src.get("title"),
                "organization": src.get("organization"),
                "url": src.get("url"),
                "source_type": src.get("source_type"),
                "last_verified": _source_date(src.get("last_verified")),
            })
        role_order = {
            "PRIMARY_REGULATORY": 0,
            "PRODUCT_LABEL": 1,
            "GUIDELINE": 2,
            "TERATOLOGY_SERVICE": 3,
            "SUPPORTING": 4,
            "LEGACY_CATEGORY": 5,
        }
        sources.sort(key=lambda x: (role_order.get(x.get("role"), 99), normalize_text(x.get("title"))))
        row["sources"] = sources
        return row

    # ---------- Toxicología externa / antídotos ----------
    def _fallback_all(self, table):
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

    def _csv_rows(self, filename):
        """Lee CSV clínicos versionados incluidos en el deploy y conserva cache local."""
        if filename in self._local_csv_cache:
            return [dict(r) for r in self._local_csv_cache[filename]]

        candidates = [Path(__file__).resolve().parent / filename]
        if self.fallback_db_path:
            candidates.append(self.fallback_db_path.resolve().parent / filename)

        rows = []
        for path in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    rows = [dict(row) for row in csv.DictReader(fh)]
                break
            except Exception:
                continue

        self._local_csv_cache[filename] = rows
        return [dict(r) for r in rows]

    @staticmethod
    def _merge_nonempty(base, overlay):
        row = dict(base or {})
        for field, value in dict(overlay or {}).items():
            if value not in (None, ""):
                row[field] = value
        return row

    def _original_other_tox(self):
        rows = []
        rows.extend(self._csv_rows("toxicos_drogas_plaguicidas_metales.csv"))
        rows.extend(self._fallback_all("other_tox"))

        out = {}
        for r in rows:
            name = str(r.get("toxico") or "").strip()
            if not name:
                continue
            key = normalize_text(name)
            if key in {"droga", "toxico"} and "sintomas de intoxicacion" in normalize_text(r.get("sintomas_base")):
                continue
            if key in out:
                out[key] = self._merge_nonempty(out[key], r)
            else:
                out[key] = dict(r)
        return list(out.values())

    def _reviewed_external_tox(self):
        rows = self._csv_rows("toxicos_externos_revisados_v2.csv")
        if not rows:
            rows = self._csv_rows("toxicos_externos_revisados_v1.csv")
        return [dict(r) for r in rows if str(r.get("toxico") or "").strip()]

    @staticmethod
    def _external_match_keys(row):
        keys = set()
        main = normalize_text((row or {}).get("toxico"))
        if main:
            keys.add(main)
        raw_alias = str((row or {}).get("alias") or "")
        for part in re.split(r"[;|,/]+", raw_alias):
            key = normalize_text(part)
            if len(key) >= 3:
                keys.add(key)
        return keys

    def _merged_other_tox(self):
        original = self._original_other_tox()
        reviewed = self._reviewed_external_tox()

        reviewed_by_key = {}
        for idx, r in enumerate(reviewed):
            for key in self._external_match_keys(r):
                reviewed_by_key.setdefault(key, idx)

        out = []
        seen = set()
        for old in original:
            old_name = str(old.get("toxico") or "").strip()
            key = normalize_text(old_name)
            if not key or key in seen:
                continue
            seen.add(key)

            idx = reviewed_by_key.get(key)
            if idx is None:
                row = dict(old)
                row.setdefault("categoria", "BASE ORIGINAL")
                row.setdefault("estado_revision", "BASE_ORIGINAL")
                row.setdefault("origen_registro", "Base original MedCalc")
                out.append(row)
                continue

            rev = reviewed[idx]
            rev_main = normalize_text(rev.get("toxico"))
            merged = dict(old)
            merged["sintomas_originales"] = old.get("sintomas_base")
            merged["tratamiento_original"] = old.get("antidoto_tratamiento_base")
            merged.update(rev)
            if key != rev_main:
                merged["toxico_canonico"] = rev.get("toxico")
                merged["toxico"] = old_name
                merged["alias"] = f"{old_name}; {rev.get('toxico') or ''}".strip("; ")
            merged["origen_registro"] = "Base original MedCalc + revisión bibliográfica"
            out.append(merged)

        existing = {normalize_text(r.get("toxico")) for r in out}
        for rev in reviewed:
            key = normalize_text(rev.get("toxico"))
            if not key or key in existing:
                continue
            row = dict(rev)
            row["origen_registro"] = "Revisión bibliográfica"
            out.append(row)
            existing.add(key)

        return out

    @staticmethod
    def _fuzzy_tox_match(query, row, searchable):
        q = normalize_text(query)
        if not q:
            return True

        if any(q in normalize_text(row.get(field)) for field in searchable):
            return True
        if len(q) < 4:
            return False

        candidates = set()
        for field in ("toxico", "toxico_canonico", "alias"):
            value = normalize_text(row.get(field))
            if not value:
                continue
            candidates.add(value)
            candidates.update(tok for tok in value.split() if len(tok) >= 4)

        return any(
            difflib.SequenceMatcher(None, q, candidate).ratio() >= 0.64
            for candidate in candidates
        )

    def search_other_tox(self, query=""):
        rows = self._merged_other_tox()
        q = normalize_text(query)
        if q:
            searchable = (
                "toxico", "toxico_canonico", "alias", "categoria",
                "region_relevancia", "via_exposicion", "mecanismo_toxicidad",
                "mecanismo_accion", "sintomas_base", "signos_gravedad",
                "antidoto_tratamiento_base", "tratamiento_especifico",
                "antidoto", "fuente", "sintomas_originales",
                "tratamiento_original",
            )
            literal = [
                r for r in rows
                if any(q in normalize_text(r.get(field)) for field in searchable)
            ]
            rows = literal if literal else [
                r for r in rows if self._fuzzy_tox_match(q, r, searchable)
            ]

        return sorted(
            rows,
            key=lambda r: (
                normalize_text(r.get("categoria") or "ZZZ"),
                normalize_text(r.get("toxico")),
            ),
        )

    def _original_antidotes(self):
        rows = self._csv_rows("antidotos.csv")
        rows.extend(self._fallback_all("antidotes"))
        out = {}
        for r in rows:
            key = (
                normalize_text(r.get("toxico_sindrome")),
                normalize_text(r.get("antidoto_base")),
            )
            if not any(key):
                continue
            if key in out:
                out[key] = self._merge_nonempty(out[key], r)
            else:
                out[key] = dict(r)
        return list(out.values())

    def _reviewed_antidotes(self):
        return [
            dict(r)
            for r in self._csv_rows("antidotos_revisados_v2.csv")
            if str(r.get("toxico_sindrome") or "").strip()
        ]

    def _merged_antidotes(self):
        original = self._original_antidotes()
        reviewed = self._reviewed_antidotes()
        if not reviewed:
            return original

        def key(row):
            return (
                normalize_text((row or {}).get("toxico_sindrome")),
                normalize_text((row or {}).get("antidoto_base")),
            )

        rev_by_key = {key(r): r for r in reviewed}
        out = []
        used = set()

        for old in original:
            k = key(old)
            if k in rev_by_key:
                row = dict(old)
                row["dosis_original"] = old.get("dosis_base")
                row["observaciones_originales"] = old.get("observaciones_base")
                row.update(rev_by_key[k])
                row["origen_registro"] = "Base original MedCalc + revisión bibliográfica"
                out.append(row)
                used.add(k)
            else:
                row = dict(old)
                row["origen_registro"] = "Base original MedCalc"
                out.append(row)

        existing = {key(r) for r in out}
        for r in reviewed:
            k = key(r)
            if k not in used and k not in existing:
                row = dict(r)
                row["origen_registro"] = "Revisión bibliográfica"
                out.append(row)

        return out

    def search_antidotes(self, query=""):
        rows = self._merged_antidotes()
        q = normalize_text(query)
        if q:
            searchable = (
                "toxico_sindrome", "antidoto_base", "dosis_base",
                "observaciones_base", "dosis_revisada", "indicacion_clinica",
                "precauciones_clave", "fuente_libro", "paginas_libro",
            )
            rows = [
                r for r in rows
                if any(q in normalize_text(r.get(field)) for field in searchable)
            ]
        return sorted(
            rows,
            key=lambda r: (
                normalize_text(r.get("toxico_sindrome")),
                normalize_text(r.get("antidoto_base")),
            ),
        )

    def toxicology_ancillary_status(self):
        return {
            "external_original": len(self._original_other_tox()),
            "external_reviewed": len(self._reviewed_external_tox()),
            "external_total": len(self._merged_other_tox()),
            "antidotes_original": len(self._original_antidotes()),
            "antidotes_reviewed": len(self._reviewed_antidotes()),
            "antidotes_total": len(self._merged_antidotes()),
        }

    # ---------- Hidroelectrolitos / reposición ----------
    def electrolyte_analytes(self):
        rows = self._fetch_optional(
            "electrolyte_analytes",
            "id,code,name,symbol,valence,meq_supported,molar_mass_g_mol,reference_unit,display_order,active",
        )
        rows = [r for r in rows if r.get("active") is not False]
        return sorted(rows, key=lambda r: (r.get("display_order") or 999, r.get("code") or ""))

    def _electrolyte_analyte(self, code):
        code = str(code or "").upper().strip()
        for row in self.electrolyte_analytes():
            if str(row.get("code") or "").upper() == code:
                return row
        return None

    def electrolyte_protocols(self, analyte_code="K", disorder=None):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        rows = self._fetch_optional("electrolyte_protocols", "*")
        rows = [
            r for r in rows
            if r.get("analyte_id") == analyte.get("id") and r.get("status") == "PUBLISHED"
        ]
        if disorder:
            d = str(disorder).upper()
            rows = [r for r in rows if str(r.get("disorder") or "").upper() in {d, "BOTH"}]
        for r in rows:
            src = self._source(r.get("source_id"))
            r["source"] = src
        return sorted(rows, key=lambda r: (
            0 if r.get("preferred_for_app") else 1,
            r.get("clinical_setting") or "",
            r.get("code") or "",
        ))

    def electrolyte_rules(self, analyte_code="K", disorder=None, protocol_code=None):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        protocols = {r.get("id"): r for r in self.electrolyte_protocols(analyte_code, disorder)}
        if protocol_code:
            protocols = {k: v for k, v in protocols.items() if v.get("code") == protocol_code}
        rows = self._fetch_optional("electrolyte_rules", "*")
        rows = [
            dict(r) for r in rows
            if r.get("status") == "PUBLISHED"
            and r.get("analyte_id") == analyte.get("id")
            and r.get("protocol_id") in protocols
        ]
        if disorder:
            d = str(disorder).upper()
            rows = [r for r in rows if str(r.get("disorder") or "").upper() in {d, "BOTH"}]
        source_links = self._fetch_optional("electrolyte_rule_sources", "rule_id,source_id,evidence_role,citation_note")
        by_rule = {}
        for link in source_links:
            by_rule.setdefault(link.get("rule_id"), []).append({
                **link,
                "source": self._source(link.get("source_id")),
            })
        for r in rows:
            r["protocol"] = protocols.get(r.get("protocol_id")) or {}
            r["sources"] = by_rule.get(r.get("id"), [])
        return sorted(rows, key=lambda r: (r.get("priority") or 100, r.get("rule_code") or ""))

    def electrolyte_products(self, analyte_code="K", route=None):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        rows = self._fetch_optional("electrolyte_products", "*")
        rows = [dict(r) for r in rows if r.get("status") == "PUBLISHED" and r.get("primary_analyte_id") == analyte.get("id")]
        if route:
            rows = [r for r in rows if str(r.get("route") or "").upper() == str(route).upper()]
        comps = self._fetch_optional("electrolyte_product_components", "*")
        analytes = {r.get("id"): r for r in self.electrolyte_analytes()}
        by_product = {}
        for c in comps:
            c = dict(c)
            c["analyte"] = analytes.get(c.get("analyte_id")) or {}
            by_product.setdefault(c.get("product_id"), []).append(c)
        for r in rows:
            r["components"] = by_product.get(r.get("id"), [])
            r["source"] = self._source(r.get("source_id"))
        return sorted(rows, key=lambda r: (r.get("route") or "", r.get("generic_product_name") or ""))

    def electrolyte_diluents(self):
        rows = [dict(r) for r in self._fetch_optional("electrolyte_diluents", "*") if r.get("status") == "PUBLISHED"]
        comps = self._fetch_optional("electrolyte_diluent_components", "*")
        analytes = {r.get("id"): r for r in self.electrolyte_analytes()}
        by_diluent = {}
        for c in comps:
            c = dict(c)
            c["analyte"] = analytes.get(c.get("analyte_id")) or {}
            by_diluent.setdefault(c.get("diluent_id"), []).append(c)
        for r in rows:
            r["components"] = by_diluent.get(r.get("id"), [])
            r["source"] = self._source(r.get("source_id"))
        return sorted(rows, key=lambda r: (r.get("name") or "", r.get("container_volume_ml") or 0))

    def electrolyte_compatibilities(self, product_id=None):
        rows = [dict(r) for r in self._fetch_optional("electrolyte_product_diluent_compatibility", "*") if r.get("status") == "PUBLISHED"]
        if product_id:
            rows = [r for r in rows if r.get("product_id") == product_id]
        for r in rows:
            r["source"] = self._source(r.get("source_id"))
        return rows

    def electrolyte_administration_limits(self, analyte_code="K", protocol_code=None):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        protocols = {r.get("id"): r for r in self.electrolyte_protocols(analyte_code)}
        if protocol_code:
            protocols = {k: v for k, v in protocols.items() if v.get("code") == protocol_code}
        rows = [
            dict(r) for r in self._fetch_optional("electrolyte_administration_limits", "*")
            if r.get("status") == "PUBLISHED"
            and r.get("analyte_id") == analyte.get("id")
            and (not protocol_code or r.get("protocol_id") in protocols)
        ]
        for r in rows:
            r["protocol"] = protocols.get(r.get("protocol_id")) or {}
            r["source"] = self._source(r.get("source_id"))
        return sorted(rows, key=lambda r: (r.get("protocol_id") or "", r.get("limit_code") or ""))

    def medication_electrolyte_modifiers(self, med_ids, analyte_code="K"):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        wanted = {self._uuid_by_med_id.get(m) for m in (med_ids or [])}
        wanted.discard(None)
        if not wanted:
            return []
        rows = self._fetch_optional("medication_electrolyte_modifiers", "*")
        out = []
        reverse = {v: k for k, v in self._uuid_by_med_id.items()}
        for r in rows:
            if r.get("status") != "PUBLISHED" or r.get("analyte_id") != analyte.get("id") or r.get("medication_id") not in wanted:
                continue
            x = dict(r)
            med_id = reverse.get(x.get("medication_id"))
            med = self._med_by_med_id.get(med_id) or {}
            x["med_id"] = med_id
            x["generic_name"] = med.get("generic_name")
            x["source"] = self._source(x.get("source_id"))
            out.append(x)
        return sorted(out, key=lambda r: (r.get("direction") or "", normalize_text(r.get("generic_name"))))

    def electrolyte_dependencies(self, analyte_code="K"):
        analyte = self._electrolyte_analyte(analyte_code)
        if not analyte:
            return []
        rows = [
            dict(r) for r in self._fetch_optional("electrolyte_dependencies", "*")
            if r.get("status") == "PUBLISHED" and r.get("primary_analyte_id") == analyte.get("id")
        ]
        for r in rows:
            r["source"] = self._source(r.get("source_id"))
        return sorted(rows, key=lambda r: (r.get("priority") or 100, r.get("dependency_code") or ""))

    def electrolyte_bundle(self, analyte_code="K"):
        return {
            "analyte": self._electrolyte_analyte(analyte_code),
            "protocols": self.electrolyte_protocols(analyte_code),
            "rules": self.electrolyte_rules(analyte_code),
            "products": self.electrolyte_products(analyte_code),
            "diluents": self.electrolyte_diluents(),
            "limits": self.electrolyte_administration_limits(analyte_code),
            "dependencies": self.electrolyte_dependencies(analyte_code),
        }



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
