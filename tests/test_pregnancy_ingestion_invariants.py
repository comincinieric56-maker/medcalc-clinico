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
assert "select m.id,'PUBLISHED'" not in upserts
assert "set status='PUBLISHED'" not in upserts
assert "values ('PUBLISHED'" not in upserts
assert "'COMPATIBLE'" not in upserts
assert "'PREFERRED'" not in upserts
assert "trimester_1" not in upserts and "trimester_2" not in upserts and "trimester_3" not in upserts
assert "not exists (select 1 from public.pregnancy_safety ps where ps.medication_id=m.id)" in upserts

assert "where status='PUBLISHED'" in schema
assert "FDA_HISTORICAL" in schema and "'TGA'" in schema
print("pregnancy safety invariants: OK")

# Existing installations must be reconciled without destructive rewrites.
assert "add column if not exists recommendation" in schema
assert "add column if not exists legacy_category" in schema
assert "drop table" not in schema.lower()

# Regulatory aliases are explicit reviewed identity translations, never fuzzy matching.
pipeline=Path("MEDCALC_PREGNANCY_GLOBAL_V1.py").read_text(encoding="utf-8")
assert '"cefadroxilo":"cefadroxil"' in pipeline
assert '"acetazolamida":"acetazolamide"' in pipeline
assert "fuzzy" not in pipeline.lower() or "never accept a fuzzy identity" in pipeline.lower()
assert "NO_TGA_TO_FDA_MAPPING" in pipeline

# Candidate discovery is triage only and cannot become regulatory evidence.
candidates=Path("MEDCALC_PREGNANCY_CANDIDATES_V1.py").read_text(encoding="utf-8")
assert '"identity_accepted":"false"' in candidates
assert '"decision":"REVIEW_REQUIRED"' in candidates
assert '"autoaccepted":0' in candidates
assert "SIMILARITY_IS_DISCOVERY_ONLY" in candidates
for forbidden in ("fda_historical_category","pregnancy_text","trimester_1","trimester_2","trimester_3","PUBLISHED","COMPATIBLE","PREFERRED"):
    assert forbidden not in candidates

# AEMPS/CIMA-verified multilingual regulatory identity aliases.
for pair in ('"darifenacina":"darifenacin"','"metocarbamol":"methocarbamol"','"gabapentina":"gabapentin"','"isotretinoina":"isotretinoin"'):
    assert pair in pipeline

for pair in ('"ciprofloxacina":"ciprofloxacin"','"itraconazol":"itraconazole"','"voriconazol":"voriconazole"'):
    assert pair in pipeline
