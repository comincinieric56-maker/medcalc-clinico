"""Synthetic fail-closed regression cases for MEDCALC renal selection.

Does not validate clinical dosing and does not contact production data.
Run: python -m unittest -v tests.test_renal_selector_safety
"""
import math
import unittest
from medcalc_engine import select_renal_rule

def rule(identifier, lower=None, upper=None, metric="CrCl_CG_mL_min",
         rule_type="REGIMEN", li="SI", ui="NO", **overrides):
    result = {
        "id": identifier,
        "tipo_regla": rule_type,
        "automatizable": "SI",
        "metrica_renal": metric,
        "limite_inferior": lower,
        "limite_superior": upper,
        "inferior_inclusivo": li,
        "superior_inclusivo": ui,
        "regimen_ajustado": "100 mg PO cada 24 h (solo prueba sintética)",
    }
    result.update(overrides)
    return result

class RenalFailClosedTests(unittest.TestCase):
    def pick(self, rules, cg=None, normalized=None, egfr=None, hd=False):
        return select_renal_rule(rules, cg, normalized, egfr, hd)

    def test_unique_matched_cg_band(self):
        a = rule("A", 30, 60)
        selected, value = self.pick([a], cg=40)
        self.assertEqual(selected["id"], "A")
        self.assertEqual(value, 40)

    def test_missing_required_metric_returns_none(self):
        self.assertEqual(self.pick([rule("A", 30, 60)], egfr=40), (None, None))

    def test_egfr_not_substituted_for_cockcroft_gault(self):
        a = rule("A", 30, 60, metric="eGFR_CKDEPI_mL_min_1_73m2")
        self.assertEqual(self.pick([a], cg=40), (None, None))
        self.assertEqual(self.pick([a], egfr=40)[0]["id"], "A")

    def test_normalized_crcl_requires_its_own_value(self):
        a = rule("A", 30, 60, metric="CrCl_normalizado_mL_min_1_73m2")
        self.assertEqual(self.pick([a], cg=40), (None, None))
        self.assertEqual(self.pick([a], normalized=40)[0]["id"], "A")

    def test_unknown_metric_not_selectable(self):
        self.assertEqual(self.pick([rule("A", 30, 60, metric="UNRECOGNIZED")], cg=40), (None, None))

    def test_unbounded_rule_never_selected(self):
        self.assertEqual(self.pick([rule("A", None, None)], cg=40), (None, None))

    def test_unbounded_rule_cannot_preempt_valid_band(self):
        selected, _ = self.pick([rule("N", None, None), rule("A", 30, 60)], cg=40)
        self.assertEqual(selected["id"], "A")

    def test_hd_rule_requires_special_workflow(self):
        a = rule("HD", None, None, rule_type="DIALISIS")
        self.assertEqual(self.pick([a], cg=10, hd=True), (None, None))
        self.assertEqual(self.pick([a], cg=10, hd=False), (None, None))

    def test_multiple_matching_regimens_rejected(self):
        a = rule("A", 30, 60, ui="SI")
        b = rule("B", 60, None, li="SI")
        self.assertEqual(self.pick([a, b], cg=60), (None, None))

    def test_same_regimen_list_order_does_not_change_ambiguity(self):
        a = rule("A", 30, 60, ui="SI")
        b = rule("B", 60, None, li="SI")
        self.assertEqual(self.pick([b, a], cg=60), (None, None))

    def test_non_overlap_60_when_upper_exclusive(self):
        a = rule("A", 30, 60, ui="NO")
        b = rule("B", 60, None, li="SI")
        self.assertEqual(self.pick([a, b], cg=60)[0]["id"], "B")

    def test_lower_exclusive_boundary(self):
        a = rule("A", 15, 30, li="NO")
        self.assertEqual(self.pick([a], cg=15), (None, None))
        self.assertEqual(self.pick([a], cg=15.1)[0]["id"], "A")

    def test_nan_is_rejected(self):
        self.assertEqual(self.pick([rule("A", 30, 60)], cg=math.nan), (None, None))

    def test_infinity_is_rejected(self):
        self.assertEqual(self.pick([rule("A", 30, None)], cg=math.inf), (None, None))

    def test_negative_renal_value_is_rejected(self):
        self.assertEqual(self.pick([rule("A", None, 30)], cg=-2), (None, None))

    def test_invalid_reversed_band(self):
        self.assertEqual(self.pick([rule("A", 60, 30)], cg=40), (None, None))

    def test_multiple_nonmatching_and_one_match(self):
        a = rule("A", 0, 30)
        b = rule("B", 30, 60)
        c = rule("C", 60, None)
        self.assertEqual(self.pick([a, b, c], cg=45)[0]["id"], "B")

    def test_no_rules(self):
        self.assertEqual(self.pick([], cg=40), (None, None))

    def test_bad_renal_value_type_rejected(self):
        self.assertEqual(self.pick([rule("A", 20, 60)], cg="forty"), (None, None))



    def test_no_adjustment_not_a_regimen(self):
        a = rule("NA", 30, None, rule_type="NO_AJUSTE", regimen_ajustado="No requiere ajuste renal")
        self.assertEqual(self.pick([a], cg=55), (None, None))

    def test_per_kg_needs_verified_weight_basis(self):
        a = rule("W", 30, None, regimen_ajustado="5 mg/kg IV q8h")
        self.assertEqual(self.pick([a], cg=55), (None, None))

    def test_dialysis_timing_in_numeric_regimen_is_not_automatic(self):
        a = rule("HD", None, 15, regimen_ajustado="200 mg oral cada 96 h. Dar después de la sesión de hemodiálisis.")
        self.assertEqual(self.pick([a], cg=10), (None, None))

    def test_rule_without_automatic_flag_is_reference_only(self):
        a = rule("REF", 30, 60, automatizable="NO")
        self.assertEqual(self.pick([a], cg=45), (None, None))

    def test_ideal_weight_note_blocks_automatic_selection(self):
        a = rule("IDEAL", 30, 60, regimen_ajustado="100 mg/día",
                 notas="Utilizar peso ideal en obesidad")
        self.assertEqual(self.pick([a], cg=45), (None, None))

    def test_mixed_reference_and_numeric_valid_rule(self):
        ref = rule("NA", 30, 60, rule_type="NO_AJUSTE")
        active = rule("R", 30, 60, regimen_ajustado="100 mg PO cada 24 h")
        selected, _ = self.pick([ref, active], cg=45)
        self.assertEqual(selected["id"], "R")


    def test_meropenem_crcl_decimal_gap_is_not_interpolated(self):
        lower = rule("10-25", 10, 25, li="SI", ui="SI")
        higher = rule("26-50", 26, 50, li="SI", ui="SI")
        self.assertEqual(self.pick([lower, higher], cg=25.5), (None, None))

    def test_meropenem_crcl_boundaries_preserve_source(self):
        lower = rule("10-25", 10, 25, li="SI", ui="SI")
        higher = rule("26-50", 26, 50, li="SI", ui="SI")
        self.assertEqual(self.pick([lower, higher], cg=25)[0]["id"], "10-25")
        self.assertEqual(self.pick([lower, higher], cg=26)[0]["id"], "26-50")

    def test_renal_release_gate_is_explicitly_disabled(self):
        import ast
        from pathlib import Path
        code = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        tree = ast.parse(code)
        assignments = [
            a.value for a in tree.body
            if isinstance(a, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "RENAL_DOSE_AUTOMATION_RELEASED"
                    for t in a.targets)
        ]
        self.assertEqual(len(assignments), 1, "No debe haber múltiples puertas de lanzamiento")
        self.assertIsInstance(assignments[0], ast.Constant)
        self.assertIs(assignments[0].value, False)
        self.assertIn("auto_rules = auto_candidates if RENAL_DOSE_AUTOMATION_RELEASED else []", code)

    def test_renal_streamlit_fallback_remains_fail_closed(self):
        import ast
        from pathlib import Path
        code = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        functions = [n for n in ast.parse(code).body
                     if isinstance(n, ast.FunctionDef)
                     and n.name == "_fallback_select_renal_rule"]
        self.assertEqual(len(functions), 1)
        returns = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Return)]
        self.assertEqual(len(returns), 1)
        self.assertIsInstance(returns[0].value, ast.Tuple)
        self.assertTrue(all(isinstance(x, ast.Constant) and x.value is None
                            for x in returns[0].value.elts))


    def test_dosis_habitual_not_selected(self):
        a = rule("HAB", 30, 60, regimen_ajustado="Dosis habitual")
        self.assertEqual(self.pick([a], cg=45), (None, None))

    def test_alternative_strengths_not_selected(self):
        a = rule("ALT", 0, 30, regimen_ajustado="250/125 mg o 500/125 mg PO cada 24 h según gravedad")
        self.assertEqual(self.pick([a], cg=20), (None, None))

    def test_titration_not_selected(self):
        a = rule("TIT", 0, 30, regimen_ajustado="Iniciar 0,5 mg/día, puede titularse según tolerancia")
        self.assertEqual(self.pick([a], cg=20), (None, None))

    def test_source_without_explicit_quantity_not_selected(self):
        a = rule("UNITS", 30, 60, regimen_ajustado="Continuar pauta habitual según foco")
        self.assertEqual(self.pick([a], cg=40), (None, None))

    def test_unambiguous_numeric_text_still_selectable_in_isolation(self):
        a = rule("ONE", 30, 60, regimen_ajustado="200 mg PO cada 24 h")
        selected, value = self.pick([a], cg=40)
        self.assertEqual(selected["id"], "ONE")
        self.assertEqual(value, 40)

    def test_real_catalog_ambiguous_regimen_phrases_stay_reference_only(self):
        # Text patterns observed in MEDCALC: documentation only, no clinical dosing approval.
        samples = [
            "Dosis habitual",
            "Dosis habitual (si no usa atazanavir/ritonavir)",
            "Dosis adulta habitual",
            "Iniciar 0,5 mg VO una vez al día; puede titularse según necesidad/tolerancia",
            "250/125 mg o 500/125 mg PO cada 24 h según gravedad",
            "250/125 mg o 500/125 mg PO cada 12 h según gravedad",
            "400 mg/día = 10 mL/día de suspensión, en una toma o según pauta clínica",
        ]
        for i, regimen in enumerate(samples):
            with self.subTest(index=i):
                specimen = rule(f"CAT-{i}", 0, 60, regimen_ajustado=regimen)
                self.assertEqual(self.pick([specimen], cg=30), (None, None))

if __name__ == "__main__":
    unittest.main()
