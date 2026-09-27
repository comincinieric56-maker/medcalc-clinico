-- MEDCALC RENAL GLOBAL V5 · VERIFICACIÓN POST-CARGA
SELECT
  COUNT(*) FILTER (WHERE renal_status='PUBLISHED') AS renal_published,
  COUNT(*) AS total_module_rows
FROM public.medication_module_status;

SELECT
  rv.validation_class,
  COUNT(DISTINCT rr.medication_id) AS medicamentos
FROM public.renal_rules rr
JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
WHERE rr.status='PUBLISHED'
GROUP BY rv.validation_class
ORDER BY rv.validation_class;

SELECT
  COUNT(DISTINCT m.id) AS activos,
  COUNT(DISTINCT m.id) FILTER (
    WHERE EXISTS (
      SELECT 1 FROM public.renal_rules rr
      JOIN public.renal_rule_validation rv ON rv.rule_id=rr.id
      WHERE rr.medication_id=m.id
        AND rr.status='PUBLISHED'
        AND rv.validation_class IN ('CURRENT_AUTO','CURRENT_REFERENCE','TDM')
    )
  ) AS con_regla_validada
FROM public.medications m
WHERE COALESCE(m.active,TRUE)=TRUE;
