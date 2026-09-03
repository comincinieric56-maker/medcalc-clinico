-- MEDCALC SUPABASE V2 · COBERTURA + CATÁLOGO PRIORITARIO
-- Ejecutar una sola vez en SQL Editor.

-- 1) Estado explícito por módulo para que ningún medicamento quede como "vacío".
create table if not exists public.medication_module_status (
    medication_id uuid primary key references public.medications(id) on delete cascade,
    pediatric_status text not null default 'PENDING_REVIEW'
        check (pediatric_status in ('PUBLISHED','PENDING_REVIEW','NO_PEDIATRIC_INDICATION','SPECIALIST_ONLY','NOT_APPLICABLE')),
    renal_status text not null default 'PENDING_REVIEW'
        check (renal_status in ('PUBLISHED','PENDING_REVIEW','NO_ADJUSTMENT_REQUIRED','SPECIALIST_ONLY','NOT_APPLICABLE')),
    toxicology_status text not null default 'PENDING_REVIEW'
        check (toxicology_status in ('PUBLISHED','PENDING_REVIEW','SDTE','NOT_APPLICABLE')),
    pediatric_note text,
    renal_note text,
    toxicology_note text,
    clinical_priority smallint not null default 3 check (clinical_priority between 1 and 5),
    updated_at timestamptz not null default now()
);

alter table public.medication_module_status enable row level security;
grant select on public.medication_module_status to anon, authenticated;

drop policy if exists "public_read_medication_module_status" on public.medication_module_status;
create policy "public_read_medication_module_status"
on public.medication_module_status
for select to anon, authenticated
using (true);

-- 2) Inicializar estado de los medicamentos existentes según datos publicados.
insert into public.medication_module_status (
    medication_id, pediatric_status, renal_status, toxicology_status
)
select
    m.id,
    case when exists (
        select 1 from public.pediatric_rules p
        where p.medication_id=m.id and p.status='PUBLISHED'
    ) then 'PUBLISHED' else 'PENDING_REVIEW' end,
    case when exists (
        select 1 from public.renal_rules r
        where r.medication_id=m.id and r.status='PUBLISHED'
    ) or exists (
        select 1 from public.renal_bibliography rb
        where rb.medication_id=m.id and rb.status='PUBLISHED'
    ) then 'PUBLISHED' else 'PENDING_REVIEW' end,
    case when exists (
        select 1 from public.toxicology t
        where t.medication_id=m.id and t.status='PUBLISHED'
    ) then 'PUBLISHED' else 'PENDING_REVIEW' end
from public.medications m
on conflict (medication_id) do nothing;

-- 3) Agregar primera tanda de medicamentos prioritarios ausentes del catálogo.
insert into public.medications (med_id, generic_name, normalized_name, active)
values
    ('MED-0619', 'CEFTRIAXONA', 'CEFTRIAXONA', true),
    ('MED-0620', 'CEFOTAXIMA', 'CEFOTAXIMA', true),
    ('MED-0621', 'CEFAZOLINA', 'CEFAZOLINA', true),
    ('MED-0622', 'CEFEPIMA', 'CEFEPIMA', true),
    ('MED-0623', 'CEFTAZIDIMA', 'CEFTAZIDIMA', true),
    ('MED-0624', 'GENTAMICINA', 'GENTAMICINA', true),
    ('MED-0625', 'AMIKACINA', 'AMIKACINA', true),
    ('MED-0626', 'MEROPENEM', 'MEROPENEM', true),
    ('MED-0627', 'ERTAPENEM', 'ERTAPENEM', true),
    ('MED-0628', 'IMIPENEM', 'IMIPENEM', true),
    ('MED-0629', 'PIPERACILINA/TAZOBACTAM', 'PIPERACILINA TAZOBACTAM', true),
    ('MED-0630', 'AMOXICILINA/ÁCIDO CLAVULÁNICO', 'AMOXICILINA ACIDO CLAVULANICO', true),
    ('MED-0631', 'AZTREONAM', 'AZTREONAM', true),
    ('MED-0632', 'PENICILINA G', 'PENICILINA G', true),
    ('MED-0633', 'SALBUTAMOL', 'SALBUTAMOL', true),
    ('MED-0634', 'EPINEFRINA', 'EPINEFRINA', true),
    ('MED-0635', 'ATROPINA', 'ATROPINA', true),
    ('MED-0636', 'ADENOSINA', 'ADENOSINA', true),
    ('MED-0637', 'NOREPINEFRINA', 'NOREPINEFRINA', true),
    ('MED-0638', 'DOPAMINA', 'DOPAMINA', true),
    ('MED-0639', 'DOBUTAMINA', 'DOBUTAMINA', true),
    ('MED-0640', 'HIDRALAZINA', 'HIDRALAZINA', true),
    ('MED-0641', 'NITROGLICERINA', 'NITROGLICERINA', true),
    ('MED-0642', 'HEPARINA NO FRACCIONADA', 'HEPARINA NO FRACCIONADA', true),
    ('MED-0643', 'ENOXAPARINA', 'ENOXAPARINA', true),
    ('MED-0644', 'INSULINA REGULAR', 'INSULINA REGULAR', true),
    ('MED-0645', 'INSULINA NPH', 'INSULINA NPH', true),
    ('MED-0646', 'GLUCAGÓN', 'GLUCAGON', true),
    ('MED-0647', 'DEXTROSA', 'DEXTROSA', true),
    ('MED-0648', 'GLUCONATO DE CALCIO', 'GLUCONATO DE CALCIO', true),
    ('MED-0649', 'CLORURO DE CALCIO', 'CLORURO DE CALCIO', true),
    ('MED-0650', 'LIDOCAÍNA', 'LIDOCAINA', true),
    ('MED-0651', 'BUPIVACAÍNA', 'BUPIVACAINA', true),
    ('MED-0652', 'KETAMINA', 'KETAMINA', true),
    ('MED-0653', 'PROPOFOL', 'PROPOFOL', true),
    ('MED-0654', 'MORFINA', 'MORFINA', true),
    ('MED-0655', 'FENTANILO', 'FENTANILO', true),
    ('MED-0656', 'ROCURONIO', 'ROCURONIO', true),
    ('MED-0657', 'SUCCINILCOLINA', 'SUCCINILCOLINA', true),
    ('MED-0658', 'ETOMIDATO', 'ETOMIDATO', true),
    ('MED-0659', 'LACTULOSA', 'LACTULOSA', true),
    ('MED-0660', 'POLIETILENGLICOL 3350', 'POLIETILENGLICOL 3350', true),
    ('MED-0661', 'SENÓSIDOS', 'SENOSIDOS', true),
    ('MED-0662', 'NALOXONA', 'NALOXONA', true),
    ('MED-0663', 'FLUMAZENIL', 'FLUMAZENIL', true),
    ('MED-0664', 'PRALIDOXIMA', 'PRALIDOXIMA', true),
    ('MED-0665', 'HIDROXOCOBALAMINA', 'HIDROXOCOBALAMINA', true),
    ('MED-0666', 'FOMEPIZOL', 'FOMEPIZOL', true)
on conflict (med_id) do nothing;

-- Crear su fila de estado automáticamente como pendiente.
insert into public.medication_module_status (medication_id, clinical_priority)
select m.id, 1
from public.medications m
where m.med_id between 'MED-0619' and 'MED-0666'
on conflict (medication_id) do update set clinical_priority=1, updated_at=now();

-- 4) Alias clínicos frecuentes.
insert into public.drug_aliases (medication_id, alias, normalized_alias)
select id, 'ALBUTEROL', 'ALBUTEROL' from public.medications where generic_name='SALBUTAMOL'
on conflict (medication_id, alias) do nothing;
insert into public.drug_aliases (medication_id, alias, normalized_alias)
select id, 'ADRENALINA', 'ADRENALINA' from public.medications where generic_name='EPINEFRINA'
on conflict (medication_id, alias) do nothing;
insert into public.drug_aliases (medication_id, alias, normalized_alias)
select id, 'NORADRENALINA', 'NORADRENALINA' from public.medications where generic_name='NOREPINEFRINA'
on conflict (medication_id, alias) do nothing;
insert into public.drug_aliases (medication_id, alias, normalized_alias)
select id, 'AMOXI/CLAVULÁNICO', 'AMOXI CLAVULANICO' from public.medications where generic_name='AMOXICILINA/ÁCIDO CLAVULÁNICO'
on conflict (medication_id, alias) do nothing;

-- 5) Enlazar referencias renales 2025 para los nuevos fármacos de coincidencia segura.
update public.renal_bibliography rb
set medication_id = m.id
from public.medications m
where rb.medication_id is null
and lower(trim(rb.drug_name_source)) = lower(trim(m.generic_name));

-- Alias específicos de la tabla renal.
update public.renal_bibliography rb
set medication_id = m.id
from public.medications m
where rb.medication_id is null
  and rb.drug_name_source='Amoxi/Clavulánico'
  and m.generic_name='AMOXICILINA/ÁCIDO CLAVULÁNICO';

update public.renal_bibliography rb
set medication_id = m.id
from public.medications m
where rb.medication_id is null
  and rb.drug_name_source='Piperacilina / Tazobactan'
  and m.generic_name='PIPERACILINA/TAZOBACTAM';

-- 6) Refrescar estado renal después de enlazar referencias.
update public.medication_module_status s
set renal_status='PUBLISHED', updated_at=now()
where exists (
    select 1 from public.renal_bibliography rb
    where rb.medication_id=s.medication_id and rb.status='PUBLISHED'
) or exists (
    select 1 from public.renal_rules rr
    where rr.medication_id=s.medication_id and rr.status='PUBLISHED'
);

-- 7) Vista de cobertura clínica para auditoría.
create or replace view public.v_medcalc_coverage as
select
    m.med_id,
    m.generic_name,
    s.clinical_priority,
    s.pediatric_status,
    count(distinct p.id) filter (where p.status='PUBLISHED') as pediatric_rules,
    s.renal_status,
    count(distinct rr.id) filter (where rr.status='PUBLISHED') as renal_rules,
    count(distinct rb.id) filter (where rb.status='PUBLISHED') as renal_references,
    s.toxicology_status,
    count(distinct t.id) filter (where t.status='PUBLISHED') as toxicology_records
from public.medications m
left join public.medication_module_status s on s.medication_id=m.id
left join public.pediatric_rules p on p.medication_id=m.id
left join public.renal_rules rr on rr.medication_id=m.id
left join public.renal_bibliography rb on rb.medication_id=m.id
left join public.toxicology t on t.medication_id=m.id
where m.active=true
group by m.med_id,m.generic_name,s.clinical_priority,s.pediatric_status,s.renal_status,s.toxicology_status;

grant select on public.v_medcalc_coverage to anon, authenticated;

-- 8) Versión de esquema/datos.
insert into public.app_metadata(key,value) values ('schema_version','MEDCALC_SUPABASE_V2')
on conflict (key) do update set value=excluded.value, updated_at=now();
insert into public.app_metadata(key,value) values ('catalog_expansion','PRIORITY_MEDS_V1')
on conflict (key) do update set value=excluded.value, updated_at=now();
