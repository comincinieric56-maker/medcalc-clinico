#!/usr/bin/env python3
"""Static safety regression for pregnancy ingestion/generation."""
from pathlib import Path

audit=Path("MEDCALC_PREGNANCY_GLOBAL_V1.py").read_text(encoding="utf-8")
upserts=Path("MEDCALC_PREGNANCY_SUPABASE_UPSERTS_V1.py").read_text(encoding="utf-8")
schema=Path("MEDCALC_EMBARAZO_V1_SCHEMA.sql").read_text(encoding="utf-8")

assert '"cefadroxilo":"cefadroxil"' in audit
assert 'norm(name) in REGULATORY_ALIASES else "EXACT_GENERIC"' in audit
assert '"regulatory_query_name"' in audit
assert 'fda_historical_category=category(txt)' in audit
assert '"trimester_specific":False' in audit
assert "TGA" not in audit or "TGA-to-FDA" in audit or "NO_TGA_TO_FDA_MAPPING" in audit

assert "'DRAFT','INSUFFICIENT_DATA'" in upserts
assert "'PUBLISHED'" not in upserts
assert "'COMPATIBLE'" not in upserts
assert "'PREFERRED'" not in upserts
assert "trimester_1" not in upserts and "trimester_2" not in upserts and "trimester_3" not in upserts
assert "not exists (select 1 from public.pregnancy_safety ps where ps.medication_id=m.id)" in upserts

assert "where status='PUBLISHED'" in schema
assert "FDA_HISTORICAL" in schema and "'TGA'" in schema
print("pregnancy safety invariants: OK")
