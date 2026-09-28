#!/usr/bin/env python3
"""Generate review-only Supabase SQL from MEDCALC pregnancy regulatory audit.

Safety invariant: regulatory evidence NEVER auto-publishes a clinical pregnancy
recommendation. New rows are DRAFT + INSUFFICIENT_DATA until clinician review.
No TGA->FDA mapping and no trimester inference are performed here.
"""
from __future__ import annotations
import csv
from pathlib import Path

IN=Path("generated_pregnancy_global_v1/pregnancy_v1_regulatory_found.csv")
OUT=Path("generated_pregnancy_global_v1/MEDCALC_EMBARAZO_REGULATORY_DRAFT_UPSERTS.sql")

def q(v):
    if v is None or str(v)=="":
        return "null"
    return "'" + str(v).replace("'","''") + "'"

def main():
    rows=list(csv.DictReader(IN.open(encoding="utf-8-sig")))
    sql=[
      "-- GENERATED: regulatory evidence only; clinical review required before PUBLISHED.",
      "-- Never map TGA to FDA. Never infer trimester risk.",
      "begin;",
    ]
    for r in rows:
        med=r.get("med_id","").strip()
        if not med: continue
        txt=r.get("pregnancy_text","").strip()
        cat=r.get("fda_historical_category","").strip()
        url=r.get("source_url","").strip()
        eff=r.get("effective_time","").strip()
        identity=r.get("identity_method","").strip()
        qname=r.get("regulatory_query_name","").strip()
        setid=r.get("set_id","").strip()
        app=r.get("application_number","").strip()
        mfr=r.get("manufacturer_name","").strip()
        title=f"US drug label pregnancy evidence · {qname or r.get('generic_name','')}"
        note=f"Identity={identity}; regulatory_query={qname}; effective_time={eff}; set_id={setid}; application_number={app}; manufacturer={mfr}"
        # Preserve any existing clinical row. Only create a non-published review row if
        # this medication has no pregnancy_safety record at all.
        sql += [
          f"-- {med}",
          "insert into public.pregnancy_safety (medication_id,status,recommendation,risk_summary,evidence_level,legacy_category,legacy_system,reviewed_at)",
          f"select m.id,'DRAFT','INSUFFICIENT_DATA',{q(txt)},'UNKNOWN',{q(cat or None)},{q('FDA_HISTORICAL' if cat else None)},null",
          f"from public.medications m where m.med_id={q(med)}",
          "and not exists (select 1 from public.pregnancy_safety ps where ps.medication_id=m.id);",
          "with s as (",
          "  insert into public.sources (title,organization,url,source_type,last_verified)",
          f"  select {q(title)},'FDA / DailyMed',{q(url)},'PRODUCT_LABEL',current_date",
          f"  where {q(url)} is not null and {q(url)} <> ''",
          f"  and not exists (select 1 from public.sources where url={q(url)})",
          "  returning id",
          "), src as (",
          "  select id from s union all",
          f"  select id from public.sources where url={q(url)} limit 1",
          "), ps as (",
          f"  select p.id from public.pregnancy_safety p join public.medications m on m.id=p.medication_id where m.med_id={q(med)} order by (p.status='PUBLISHED') desc, p.created_at desc limit 1",
          ")",
          "insert into public.pregnancy_safety_sources (pregnancy_safety_id,source_id,evidence_role,evidence_note)",
          f"select ps.id,src.id,{q('LEGACY_CATEGORY' if cat else 'PRODUCT_LABEL')},{q(note)} from ps cross join src",
          "on conflict (pregnancy_safety_id,source_id,evidence_role) do update set evidence_note=excluded.evidence_note;",
        ]
    sql += ["commit;"]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(sql)+"\n",encoding="utf-8")
    print(f"generated {OUT} from {len(rows)} regulatory rows")

if __name__=="__main__": main()
