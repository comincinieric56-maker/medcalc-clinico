-- MEDCALC · EMBARAZO V1 · ESQUEMA REPRODUCIBLE
-- Idempotente. No asigna recomendaciones clínicas ni convierte TGA <-> FDA.
-- Requiere public.medications y public.sources ya instaladas.

do $$
begin
  if to_regclass('public.medications') is null then
    raise exception 'MEDCALC embarazo: falta public.medications';
  end if;
  if to_regclass('public.sources') is null then
    raise exception 'MEDCALC embarazo: falta public.sources';
  end if;
end $$;

create table if not exists public.pregnancy_safety (
  id uuid primary key default gen_random_uuid(),
  medication_id uuid not null references public.medications(id) on delete cascade,
  status text not null default 'DRAFT'
    check (status in ('DRAFT','PUBLISHED','RETIRED')),
  recommendation text not null default 'INSUFFICIENT_DATA'
    check (recommendation in (
      'PREFERRED','COMPATIBLE','USE_WITH_CAUTION','AVOID',
      'CONTRAINDICATED','SPECIALIST_ONLY','INSUFFICIENT_DATA'
    )),
  risk_summary text,
  clinical_considerations text,
  pregnancy_indication_note text,
  fetal_neonatal_risk text,
  evidence_level text not null default 'UNKNOWN'
    check (evidence_level in ('HIGH','MODERATE','LOW','VERY_LOW','UNKNOWN')),
  trimester_1 text,
  trimester_2 text,
  trimester_3 text,
  legacy_category text,
  legacy_system text,
  reviewed_at date,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (
    legacy_category is null
    or legacy_category in ('A','B','B1','B2','B3','C','D','X')
  ),
  check (
    legacy_system is null
    or legacy_system in ('FDA_HISTORICAL','TGA','OTHER')
  )
);

-- Reconciliación no destructiva para instalaciones que ya tenían la tabla.
-- ADD COLUMN IF NOT EXISTS conserva íntegramente las fichas clínicas existentes.
alter table public.pregnancy_safety
  add column if not exists recommendation text not null default 'INSUFFICIENT_DATA',
  add column if not exists risk_summary text,
  add column if not exists clinical_considerations text,
  add column if not exists pregnancy_indication_note text,
  add column if not exists fetal_neonatal_risk text,
  add column if not exists evidence_level text not null default 'UNKNOWN',
  add column if not exists trimester_1 text,
  add column if not exists trimester_2 text,
  add column if not exists trimester_3 text,
  add column if not exists legacy_category text,
  add column if not exists legacy_system text,
  add column if not exists reviewed_at date,
  add column if not exists created_at timestamptz not null default now(),
  add column if not exists updated_at timestamptz not null default now();

alter table public.pregnancy_safety_sources
  add column if not exists evidence_role text,
  add column if not exists evidence_note text,
  add column if not exists created_at timestamptz not null default now();

-- Una sola ficha publicada por medicamento. Se permiten borradores históricos.
create unique index if not exists uq_pregnancy_safety_published_medication
  on public.pregnancy_safety (medication_id)
  where status='PUBLISHED';
create index if not exists ix_pregnancy_safety_medication
  on public.pregnancy_safety (medication_id);
create index if not exists ix_pregnancy_safety_status
  on public.pregnancy_safety (status);

create table if not exists public.pregnancy_safety_sources (
  pregnancy_safety_id uuid not null references public.pregnancy_safety(id) on delete cascade,
  source_id uuid not null references public.sources(id) on delete cascade,
  evidence_role text not null
    check (evidence_role in (
      'PRIMARY_REGULATORY','PRODUCT_LABEL','GUIDELINE',
      'TERATOLOGY_SERVICE','SUPPORTING','LEGACY_CATEGORY'
    )),
  evidence_note text,
  created_at timestamptz not null default now(),
  primary key (pregnancy_safety_id, source_id, evidence_role)
);

create index if not exists ix_pregnancy_safety_sources_safety
  on public.pregnancy_safety_sources (pregnancy_safety_id);
create index if not exists ix_pregnancy_safety_sources_source
  on public.pregnancy_safety_sources (source_id);

alter table public.pregnancy_safety enable row level security;
alter table public.pregnancy_safety_sources enable row level security;

grant select on public.pregnancy_safety to anon, authenticated;
grant select on public.pregnancy_safety_sources to anon, authenticated;

drop policy if exists "public_read_published_pregnancy_safety" on public.pregnancy_safety;
create policy "public_read_published_pregnancy_safety"
on public.pregnancy_safety
for select to anon, authenticated
using (status='PUBLISHED');

drop policy if exists "public_read_pregnancy_safety_sources" on public.pregnancy_safety_sources;
create policy "public_read_pregnancy_safety_sources"
on public.pregnancy_safety_sources
for select to anon, authenticated
using (
  exists (
    select 1 from public.pregnancy_safety ps
    where ps.id=pregnancy_safety_id and ps.status='PUBLISHED'
  )
);

comment on table public.pregnancy_safety is
  'Ficha obstétrica MEDCALC. Recommendation is a clinical conclusion and must not be inferred mechanically from TGA or historical FDA letters.';
comment on column public.pregnancy_safety.legacy_category is
  'Historical/regulatory category only when explicitly verified in the named legacy_system; never cross-map TGA and FDA.';
comment on column public.pregnancy_safety.trimester_1 is
  'Populate only when the source explicitly supports trimester-specific interpretation; NULL means no trimester-specific statement.';
comment on column public.pregnancy_safety.trimester_2 is
  'Populate only when the source explicitly supports trimester-specific interpretation; NULL means no trimester-specific statement.';
comment on column public.pregnancy_safety.trimester_3 is
  'Populate only when the source explicitly supports trimester-specific interpretation; NULL means no trimester-specific statement.';
