#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEDCALC · RENAL GLOBAL V6 · AEMPS/CIMA
======================================

Segunda pasada regulatoria sobre los MED-ID que continúen pendientes después de
GLOBAL V5. Utiliza exclusivamente AEMPS/CIMA (España), cuya API REST pública
permite buscar por principio activo y recuperar la ficha técnica segmentada.

Seguridad:
- identidad farmacológica estricta;
- integridad de combinaciones;
- exige una conducta renal explícita;
- no crea reglas numéricas automáticas;
- toda evidencia nueva se clasifica CURRENT_REFERENCE;
- "sin recomendación renal explícita" NO significa "no requiere ajuste".
"""

from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from MEDCALC_RENAL_GLOBAL_MULTISOURCE_V5 import (
    exact_component_match,
    is_combo,
    read_csv,
    split_combo,
    write_csv,
)

ROOT = Path(".")
V5 = Path("generated_renal_global_v5")
OUT = Path("generated_renal_global_v6")
OUT.mkdir(parents=True, exist_ok=True)

PENDING_V5 = V5 / "renal_global_still_pending.csv"
MATRIX_V5 = V5 / "renal_global_matrix_1122.csv"

CIMA = "https://cima.aemps.es/cima/rest"

RENAL_ES = re.compile(
    r"\b(insuficiencia renal|funci[oó]n renal|deterioro renal|disfunci[oó]n renal|"
    r"enfermedad renal|aclaramiento de creatinina|clearance de creatinina|"
    r"creatinine clearance|\bcrcl\b|\begfr\b|\btfg\b|filtrado glomerular|"
    r"hemodi[aá]lisis|di[aá]lisis|enfermedad renal terminal)\b",
    re.I,
)

NO_ADJUST_ES = re.compile(
    r"\b(no (?:es )?necesari[oa] (?:un )?ajust(?:e|ar) (?:de )?(?:la )?dosis|"
    r"no se requiere (?:un )?ajuste (?:de )?(?:la )?dosis|"
    r"no requiere (?:un )?ajuste (?:de )?(?:la )?dosis|"
    r"no es preciso ajustar (?:la )?dosis|"
    r"no se recomienda ning[uú]n ajuste (?:de )?(?:la )?dosis)\b",
    re.I,
)

NOT_RECOMMENDED_ES = re.compile(
    r"\b(no (?:se )?recomienda|debe evitarse|evitar (?:su )?uso|"
    r"contraindicad[oa]|no debe utilizarse|no debe administrarse)\b",
    re.I,
)

THRESHOLD_ES = re.compile(
    r"(?:aclaramiento de creatinina|clearance de creatinina|crcl|egfr|tfg|"
    r"filtrado glomerular)"
    r"[^.;:\n]{0,180}(?:<|>|≤|≥|menor|mayor|entre|\d)",
    re.I,
)

DOSE_ACTION_ES = re.compile(
    r"\b(dosis|posolog[ií]a|administrar|administraci[oó]n|reducir|reducci[oó]n|"
    r"ajustar|ajuste|intervalo|cada|una vez al d[ií]a|dos veces al d[ií]a|"
    r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml))\b",
    re.I,
)

DIALYSIS_DOSING_ES = re.compile(
    r"(?:administrar|dosis|suplement|ajust)[^.;:\n]{0,220}(?:hemodi[aá]lisis|di[aá]lisis)|"
    r"(?:hemodi[aá]lisis|di[aá]lisis)[^.;:\n]{0,220}(?:administrar|dosis|suplement|ajust)",
    re.I,
)

ADJUST_ES = re.compile(
    r"\b(ajuste de dosis|ajustar la dosis|reducir la dosis|reducci[oó]n de dosis|"
    r"aumentar (?:el )?intervalo|modificar (?:la )?dosis|modificar (?:el )?intervalo)\b",
    re.I,
)


def clean_text(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def renal_windows_es(text: str, before: int = 650, after: int = 2200) -> str:
    text = clean_text(text)
    windows = []
    for m in RENAL_ES.finditer(text):
        lo = max(0, m.start() - before)
        hi = min(len(text), m.end() + after)
        w = text[lo:hi]
        if w not in windows:
            windows.append(w)
        if len(windows) >= 10:
            break
    return "\n---\n".join(windows)[:14000]


def clinical_action_es(text: str) -> tuple[bool, str]:
    text = str(text or "")
    if not RENAL_ES.search(text):
        return False, "CIMA_NO_RENAL_TEXT"

    if NO_ADJUST_ES.search(text):
        return True, "CIMA_NO_ADJUSTMENT_EXPLICIT"

    if NOT_RECOMMENDED_ES.search(text):
        if THRESHOLD_ES.search(text) or re.search(
            r"(grave|severa|terminal)[^.;]{0,160}(?:insuficiencia|deterioro|enfermedad) renal",
            text,
            re.I,
        ):
            return True, "CIMA_RENAL_NOT_RECOMMENDED_EXPLICIT"

    if THRESHOLD_ES.search(text) and DOSE_ACTION_ES.search(text):
        return True, "CIMA_RENAL_THRESHOLD_DOSING_EXPLICIT"

    if DIALYSIS_DOSING_ES.search(text):
        return True, "CIMA_DIALYSIS_DOSING_EXPLICIT"

    for m in RENAL_ES.finditer(text):
        lo = max(0, m.start() - 300)
        hi = min(len(text), m.end() + 850)
        if ADJUST_ES.search(text[lo:hi]):
            return True, "CIMA_RENAL_DOSING_TEXT_EXPLICIT"

    return False, "CIMA_RENAL_MENTION_WITHOUT_EXPLICIT_DOSING_ACTION"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "MEDCALC-Renal-CIMA-V6/1.0",
        "Accept-Language": "es,en;q=0.8",
    })
    return s


def get_json(s: requests.Session, path: str, params=None, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            r = s.get(f"{CIMA}/{path}", params=params, timeout=40)
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(2.5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.0 + 1.5 * i)
    if last:
        raise last
    return None


def list_items(payload: Any) -> list[dict]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in (
            "resultados", "results", "result", "medicamentos",
            "presentaciones", "items", "data",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if payload.get("nregistro"):
            return [payload]
    return []


def raw_combo_parts(name: str) -> list[str]:
    parts = re.split(r"\s*/\s*|\s+\+\s+|\s+\bY\b\s+", str(name or ""), flags=re.I)
    return [p.strip() for p in parts if p.strip()]


def query_terms(name: str) -> list[str]:
    terms = []
    raw_parts = raw_combo_parts(name)
    if raw_parts:
        terms.append(raw_parts[0])
    terms.append(str(name or "").strip())

    cleaned = re.sub(
        r"^(?:CLORHIDRATO|HIDROCLORURO|ACETATO|FUMARATO|MALEATO|SUCCINATO|"
        r"MESILATO|BESILATO|TARTRATO|CITRATO|SULFATO|FOSFATO|BROMURO)\s+DE\s+",
        "",
        str(name or "").strip(),
        flags=re.I,
    )
    terms.append(cleaned)

    out = []
    for t in terms:
        t = re.sub(r"\s+", " ", t).strip()
        if t and t.lower() not in {x.lower() for x in out}:
            out.append(t)
    return out


def cima_search(s: requests.Session, generic_name: str) -> list[dict]:
    out = []
    seen = set()

    for term in query_terms(generic_name):
        for page in range(1, 6):
            try:
                payload = get_json(
                    s,
                    "medicamentos",
                    params={
                        "practiv1": term,
                        "autorizados": 1,
                        "pagina": page,
                    },
                )
            except Exception:
                break
            items = list_items(payload)
            if not items:
                break
            new_on_page = 0
            for item in items:
                reg = str(item.get("nregistro") or "").strip()
                if not reg or reg in seen:
                    continue
                seen.add(reg)
                out.append(item)
                new_on_page += 1
            if new_on_page == 0:
                break
            if len(out) >= 80:
                return out
    return out


def cima_detail(s: requests.Session, nregistro: str) -> dict:
    payload = get_json(s, "medicamento", params={"nregistro": nregistro})
    items = list_items(payload)
    return items[0] if items else (payload if isinstance(payload, dict) else {})


def detail_ingredients(detail: dict) -> list[str]:
    vals = []
    raw = detail.get("principiosActivos")
    if isinstance(raw, list):
        for x in raw:
            if isinstance(x, dict) and x.get("nombre"):
                vals.append(str(x["nombre"]))
    if not vals:
        p = detail.get("pactivos")
        if p:
            vals.extend(x.strip() for x in re.split(r"[,;/]+", str(p)) if x.strip())
    return vals


def flatten_json_text(payload: Any) -> str:
    pieces = []
    if isinstance(payload, str):
        pieces.append(payload)
    elif isinstance(payload, list):
        for x in payload:
            pieces.append(flatten_json_text(x))
    elif isinstance(payload, dict):
        for k, v in payload.items():
            if str(k).lower() in {"contenido", "texto", "titulo", "seccion"}:
                pieces.append(flatten_json_text(v))
            elif isinstance(v, (dict, list)):
                pieces.append(flatten_json_text(v))
    return clean_text(" ".join(p for p in pieces if p))


def cima_ft_text(s: requests.Session, nregistro: str) -> str:
    url = f"{CIMA}/docSegmentado/contenido/1"
    last = None
    for i in range(4):
        try:
            r = s.get(
                url,
                params={"nregistro": nregistro},
                headers={"Accept": "text/plain"},
                timeout=50,
            )
            if r.status_code == 404:
                return ""
            if r.status_code == 429:
                time.sleep(2.5 * (i + 1))
                continue
            r.raise_for_status()
            text = clean_text(r.text)
            if text:
                return text
        except Exception as exc:
            last = exc
            time.sleep(1.0 + 1.5 * i)

    # JSON fallback for servers ignoring text/plain.
    try:
        payload = get_json(
            s,
            "docSegmentado/contenido/1",
            params={"nregistro": nregistro},
        )
        return flatten_json_text(payload)
    except Exception:
        if last:
            return ""
        return ""


def cima_check(s: requests.Session, generic_name: str) -> dict:
    candidates = cima_search(s, generic_name)
    exact = []

    for item in candidates[:80]:
        reg = str(item.get("nregistro") or "").strip()
        if not reg:
            continue
        try:
            detail = cima_detail(s, reg)
        except Exception:
            continue
        ingredients = detail_ingredients(detail)
        if not ingredients or not exact_component_match(generic_name, ingredients):
            continue
        exact.append((reg, detail, ingredients))

    if not exact:
        return {
            "cima_status": "NO_EXACT_CIMA_PRODUCT",
            "cima_reason": "",
            "cima_nregistro": "",
            "cima_product_name": "",
            "cima_source_url": "",
            "cima_renal_text": "",
        }

    # Prefer marketed products and then process up to the first 12 exact products.
    exact.sort(
        key=lambda x: (
            bool(x[1].get("comerc")),
            str(x[1].get("nombre") or ""),
        ),
        reverse=True,
    )

    first_seen = None
    for reg, detail, ingredients in exact[:12]:
        text = cima_ft_text(s, reg)
        renal = renal_windows_es(text)
        ok, reason = clinical_action_es(renal)
        rec = {
            "cima_status": "ACCEPT" if ok else "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
            "cima_reason": reason,
            "cima_nregistro": reg,
            "cima_product_name": str(detail.get("nombre") or ""),
            "cima_source_url": f"https://cima.aemps.es/cima/dochtml/ft/{quote(reg)}/FT_{quote(reg)}.html",
            "cima_renal_text": renal,
            "cima_ingredients": " | ".join(ingredients),
        }
        if first_seen is None:
            first_seen = rec
        if ok:
            return rec

    return first_seen or {
        "cima_status": "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION",
        "cima_reason": "CIMA_NO_EXPLICIT_RENAL_ACTION",
        "cima_nregistro": exact[0][0],
        "cima_product_name": str(exact[0][1].get("nombre") or ""),
        "cima_source_url": "",
        "cima_renal_text": "",
    }


def process_one(row: dict) -> dict:
    med_id = row.get("med_id") or ""
    name = row.get("generic_name") or ""
    s = session()
    out = {"med_id": med_id, "generic_name": name}
    try:
        out.update(cima_check(s, name))
    except Exception as exc:
        out.update(
            cima_status="ERROR",
            cima_reason="",
            cima_nregistro="",
            cima_product_name="",
            cima_source_url="",
            cima_renal_text="",
            cima_error=repr(exc),
        )

    if out.get("cima_status") == "ACCEPT":
        out["v6_resolution"] = "CURRENT_SINGLE_OFFICIAL_EXACT"
        out["v6_primary_source"] = "AEMPS_CIMA"
        out["v6_reason"] = out.get("cima_reason") or ""
    elif out.get("cima_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION":
        out["v6_resolution"] = "NO_EXPLICIT_RENAL_RECOMMENDATION_FOUND"
        out["v6_primary_source"] = ""
        out["v6_reason"] = out.get("cima_reason") or ""
    else:
        out["v6_resolution"] = "UNRESOLVED"
        out["v6_primary_source"] = ""
        out["v6_reason"] = ""
    return out


def main():
    if not PENDING_V5.exists() or not MATRIX_V5.exists():
        raise SystemExit("Faltan outputs GLOBAL V5; V6 debe ejecutarse después de V5.")

    pending = read_csv(PENDING_V5)
    matrix = read_csv(MATRIX_V5)
    if len(matrix) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V5 esperada 1122; encontrada {len(matrix)}")

    print(f"Pending entering CIMA V6: {len(pending)}")

    results = []
    workers = 5
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, r): r for r in pending}
        done = 0
        for fut in as_completed(futures):
            base = futures[fut]
            done += 1
            try:
                rec = fut.result()
            except Exception as exc:
                rec = {
                    "med_id": base.get("med_id"),
                    "generic_name": base.get("generic_name"),
                    "cima_status": "ERROR",
                    "cima_error": repr(exc),
                    "v6_resolution": "UNRESOLVED",
                }
            results.append(rec)
            print(
                f"[{done:03d}/{len(pending)}] "
                f"{rec.get('med_id')} {rec.get('generic_name')} -> "
                f"{rec.get('v6_resolution')}"
            )
            if done % 25 == 0:
                write_csv(
                    OUT / "renal_global_v6_cima_partial.csv",
                    sorted(results, key=lambda x: x.get("med_id") or ""),
                )

    results.sort(key=lambda x: x.get("med_id") or "")
    write_csv(OUT / "renal_global_v6_cima_results.csv", results)

    by_mid = {r.get("med_id"): r for r in results}
    final = []
    for old in matrix:
        row = dict(old)
        mid = old.get("med_id")
        nr = by_mid.get(mid)
        if nr:
            row.update(nr)
            if nr.get("v6_resolution") == "CURRENT_SINGLE_OFFICIAL_EXACT":
                row["global_resolution_v6"] = nr["v6_resolution"]
                row["global_primary_source_v6"] = "AEMPS_CIMA"
                row["global_current_resolved_v6"] = True
            else:
                row["global_resolution_v6"] = old.get("global_resolution") or nr.get("v6_resolution")
                row["global_primary_source_v6"] = old.get("global_primary_source") or ""
                row["global_current_resolved_v6"] = str(old.get("global_current_resolved")).lower() == "true"
        else:
            row["global_resolution_v6"] = old.get("global_resolution")
            row["global_primary_source_v6"] = old.get("global_primary_source")
            row["global_current_resolved_v6"] = str(old.get("global_current_resolved")).lower() == "true"
        final.append(row)

    if len(final) != 1122:
        raise SystemExit(f"SEGURIDAD: matriz V6 !=1122 ({len(final)})")

    write_csv(OUT / "renal_global_matrix_1122.csv", final)

    resolved = [
        r for r in final
        if r.get("global_current_resolved_v6") is True
        or str(r.get("global_current_resolved_v6")).lower() == "true"
    ]
    still = [r for r in final if r not in resolved]
    write_csv(OUT / "renal_global_resolved.csv", resolved)
    write_csv(OUT / "renal_global_still_pending.csv", still)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_rows": 1122,
        "v5_pending_input": len(pending),
        "cima_accept": sum(r.get("cima_status") == "ACCEPT" for r in results),
        "cima_exact_without_explicit_action": sum(
            r.get("cima_status") == "EXACT_PRODUCT_NO_EXPLICIT_RENAL_ACTION"
            for r in results
        ),
        "cima_no_exact_product": sum(
            r.get("cima_status") == "NO_EXACT_CIMA_PRODUCT"
            for r in results
        ),
        "cima_errors": sum(r.get("cima_status") == "ERROR" for r in results),
        "global_current_resolved_v6": len(resolved),
        "global_still_pending_v6": len(still),
        "total_check": len(resolved) + len(still),
        "source": {
            "name": "AEMPS CIMA",
            "role": "official Spanish Summary of Product Characteristics / ficha técnica",
            "api": "https://cima.aemps.es/cima/rest/",
        },
        "safety": {
            "new_numeric_automatic_rules_created": False,
            "explicit_renal_action_required": True,
            "exact_drug_identity_required": True,
            "combination_integrity_required": True,
            "absence_of_explicit_recommendation_is_not_no_adjustment": True,
        },
    }
    (OUT / "renal_global_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
