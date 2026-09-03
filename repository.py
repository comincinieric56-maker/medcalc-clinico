import csv
from pathlib import Path

from medcalc_engine import normalize_text


class Repository:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.meds = self._load("toxicos_medicamentos_revisados_v3.csv")
        self.other_tox = self._load("toxicos_drogas_plaguicidas_metales.csv")
        self.antidotes = self._load("antidotos.csv")
        self.general_history = self._load("medidas_generales_historicas.csv")
        self.ped_rules = self._load("dosis_pediatria.csv")
        self.renal_rules = self._load("ajuste_renal.csv")
        self.catalog = self._load("catalogo_calculos.csv")
        self.sources = self._load("fuentes_calculos.csv")
        self.renal_biblio = self._load_optional("renal_biblio_verificada_2025.csv")
        self.renal_ocr_index = self._load_optional("renal_biblio_ocr_indice.csv")

        self.catalog_by_id = {r["med_id"]: r for r in self.catalog}
        self.tox_by_id = {r["id_revision"]: r for r in self.meds}
        self.ped_by_drug = self._group(self.ped_rules, "principio_activo")
        self.renal_by_drug = self._group(self.renal_rules, "principio_activo")
        self.renal_biblio_by_drug = self._group(self.renal_biblio, "principio_activo")
        self.renal_biblio_by_med_id = self._group([r for r in self.renal_biblio if r.get("med_id")], "med_id")

    def _load(self, name):
        with open(self.data_dir / name, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _load_optional(self, name):
        path = self.data_dir / name
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _group(rows, field):
        out = {}
        for row in rows:
            out.setdefault(row[field], []).append(row)
        return out

    def search_catalog(self, query, limit=40):
        needle = normalize_text(query)
        if not needle:
            return self.catalog[:limit]
        hits = []
        for row in self.catalog:
            if needle in normalize_text(row.get("principio_activo")):
                hits.append(row)
                if len(hits) >= limit:
                    break
        return hits

    def search_meds(self, query):
        needle = normalize_text(query)
        if not needle:
            return self.meds
        return [r for r in self.meds if needle in normalize_text(r.get("principio_activo"))]

    def search_other_tox(self, query):
        needle = normalize_text(query)
        if not needle:
            return self.other_tox
        return [r for r in self.other_tox if needle in normalize_text(r.get("toxico"))]

    def search_antidotes(self, query):
        needle = normalize_text(query)
        if not needle:
            return self.antidotes
        return [
            r for r in self.antidotes
            if needle in normalize_text(r.get("toxico_sindrome"))
            or needle in normalize_text(r.get("antidoto_base"))
        ]
