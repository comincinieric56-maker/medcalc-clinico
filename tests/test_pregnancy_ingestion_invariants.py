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

# openFDA labels may carry pregnancy narrative under use_in_specific_populations.
assert '"use_in_specific_populations"' in pipeline
assert 'fda_historical_category=category(txt)' in pipeline

# Newly reviewed aliases must have auditable provenance outside similarity ranking.
alias_registry=Path("MEDCALC_PREGNANCY_IDENTITY_ALIASES_V1.csv").read_text(encoding="utf-8")
assert "verification_authority" in alias_registry and "verification_basis" in alias_registry
assert alias_registry.count(",IDENTITY_VERIFIED_SOURCE_PENDING") >= 7

# Every alias newly recorded in the provenance registry must match the runtime alias map.
import csv, io
rows=list(csv.DictReader(io.StringIO(alias_registry)))
assert rows
for row in rows:
    assert row["status"] in {"VERIFIED","IDENTITY_VERIFIED_SOURCE_PENDING","LEGACY_PROVENANCE_PENDING"}
    if row["status"] == "VERIFIED":
        assert row["source_url"].strip() and row["reviewed_at"].strip()
    assert row["verification_authority"].strip()
    assert row["verification_basis"].strip()
    expected=f'"{row["local_name"]}":"{row["regulatory_name"]}"'
    assert expected in pipeline

# Provenance registry is the gate for all newly added aliases after V1.
# Legacy reviewed aliases remain explicitly grandfathered until individually migrated.
registry_pairs={(r["local_name"],r["regulatory_name"]) for r in rows}
required_new={
("darifenacina","darifenacin"),("metocarbamol","methocarbamol"),
("gabapentina","gabapentin"),("isotretinoina","isotretinoin"),
("ciprofloxacina","ciprofloxacin"),("itraconazol","itraconazole"),
("voriconazol","voriconazole")}
assert required_new <= registry_pairs
assert len(registry_pairs) == len(rows)
assert len(rows)==30
assert sum(r["status"]=="LEGACY_PROVENANCE_PENDING" for r in rows)==23
# Runtime and provenance registry must now contain the same explicit alias identities.
import ast
mod=ast.parse(pipeline)
runtime_aliases=None
for node in mod.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REGULATORY_ALIASES" for t in node.targets):
        runtime_aliases=ast.literal_eval(node.value)
assert runtime_aliases is not None
assert set(runtime_aliases.items())==registry_pairs

# Provenance schema reserves exact official-source URL and review date for migration.
assert {"source_url","reviewed_at"} <= set(rows[0].keys())

resolution=Path("MEDCALC_PREGNANCY_RESOLUTION_V1.py").read_text(encoding="utf-8")
assert "ESTER_REQUIRES_VERIFIED_EQUIVALENCE" in resolution
assert "SALT_REQUIRES_VERIFIED_EQUIVALENCE" in resolution
assert "VERIFY_ACTIVE_MOIETY_AND_ESTER_EQUIVALENCE" in resolution
assert "VERIFY_ACTIVE_MOIETY_AND_SALT_EQUIVALENCE" in resolution

# Functional triage checks: classification must be behaviorally correct, not just present as strings.
import importlib.util
spec=importlib.util.spec_from_file_location("preg_resolution","MEDCALC_PREGNANCY_RESOLUTION_V1.py")
preg_resolution=importlib.util.module_from_spec(spec); spec.loader.exec_module(preg_resolution)
assert preg_resolution.classify({"status":"UNRESOLVED","generic_name":"fármaco clorhidrato"})=="SALT_REQUIRES_VERIFIED_EQUIVALENCE"
assert preg_resolution.candidate_reason({"status":"UNRESOLVED","generic_name":"fármaco clorhidrato"})=="VERIFY_ACTIVE_MOIETY_AND_SALT_EQUIVALENCE"
assert preg_resolution.classify({"status":"UNRESOLVED","generic_name":"fármaco cipionato"})=="ESTER_REQUIRES_VERIFIED_EQUIVALENCE"
assert preg_resolution.candidate_reason({"status":"UNRESOLVED","generic_name":"fármaco cipionato"})=="VERIFY_ACTIVE_MOIETY_AND_ESTER_EQUIVALENCE"
assert preg_resolution.classify({"status":"UNRESOLVED","generic_name":"a + b"})=="COMBINATION_REQUIRES_EXACT_PRODUCT_IDENTITY"
assert preg_resolution.classify({"status":"REGULATORY_TEXT_FOUND","generic_name":"fármaco clorhidrato"})=="FDA_DAILYMED_LABEL_FOUND"

assert preg_resolution.classify({"status":"UNRESOLVED","generic_name":"fármaco acetato"})=="AMBIGUOUS_SALT_ESTER_REQUIRES_PRODUCT_IDENTITY_REVIEW"
assert preg_resolution.candidate_reason({"status":"UNRESOLVED","generic_name":"fármaco acetato"})=="VERIFY_SALT_OR_ESTER_FROM_EXACT_PRODUCT_IDENTITY"
assert preg_resolution.AMBIGUOUS_SALT_ESTER_WORDS == {"acetato"}

# Audit output distinguishes label identity method from alias provenance completeness.
assert '"identity_provenance_status"' in pipeline
assert "load_alias_provenance_status" in pipeline
assert "EXACT_GENERIC_NOT_ALIAS" in pipeline

# Functional provenance lookup must reproduce every registry status exactly.
spec_global=importlib.util.spec_from_file_location("preg_global","MEDCALC_PREGNANCY_GLOBAL_V1.py")
preg_global=importlib.util.module_from_spec(spec_global); spec_global.loader.exec_module(preg_global)
prov=preg_global.load_alias_provenance_status()
assert len(prov)==30
for row in rows:
    assert prov[preg_global.norm(row["local_name"])]==row["status"]
assert all(v in {"VERIFIED","IDENTITY_VERIFIED_SOURCE_PENDING","LEGACY_PROVENANCE_PENDING"} for v in prov.values())
