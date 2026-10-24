# Seguimiento lote pediátrico renalchat — 2026-10-24

Marcador: AUD_PED_PAR_RENALCHAT_20261024. 100 UUID asignados; sin nueva reserva.

## Estado previo a la intervención
1122 medicamentos, 895 con reglas, 227 sin reglas.

## Corrección documental en MED-0274 — furazolidona
La nota anterior decía que no había pauta pediátrica contemporánea. Se identificó evidencia que obliga a corregir esa afirmación:
- INVIMA Acta 03/2025: pediatría >1 mes 1,25 mg/kg/dosis VO cada 6 horas; máximo reportado 8,8 mg/kg/día; duración por indicación. https://invima.gov.co/invima_website/static/attachments/medicamentos_sala_especializada_medicamentos_sintesis/acta_no._03_de_2025_sempb_segunda_parte.pdf
- Ficha fabricante Laboratorio Chile, furazolidona infantil 16,7 mg/5 mL: 5–8 mg/kg/día divididos en 3–4 tomas; formulación y contexto distintos. https://www.laboratoriochile.cl/wp-content/uploads/2015/11/Furazolidona-Infantil-167mg-5ml.pdf

La nota en Supabase se corrigió como revisión documental pendiente de conciliación. No se insertó regla dosificadora ni se activó cálculo automático. Diferencias entre presentaciones, restricciones neonatales, G6PD e interacciones exigen revisión clínica adicional.

## Otros pendientes sin regla
- Fenoterol: identificar principio activo/formulación y registro por país.
- Fenoverina: fuente secundaria sugiere no administrar <12 años, requiere ficha oficial; no inventar dosis.

No se declararon fichas clínicamente validadas en esta intervención.
