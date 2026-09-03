-- MEDCALC: medicamentos que requieren revisión por módulo
select * from public.v_medcalc_coverage
where pediatric_status <> 'PUBLISHED'
   or renal_status <> 'PUBLISHED'
   or toxicology_status <> 'PUBLISHED'
order by clinical_priority asc, generic_name;
