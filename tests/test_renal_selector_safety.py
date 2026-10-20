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
        "metrica_renal": metric,
        "limite_inferior": lower,
        "limite_superior": upper,
        "inferior_inclusivo": li,
        "superior_inclusivo": ui,
        "regimen_ajustado": "Texto sintético sin uso clínico",
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


if __name__ == "__main__":
    unittest.main()
