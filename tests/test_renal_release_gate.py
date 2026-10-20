"""Fail-closed release interlock for MEDCALC renal dosing.

Runs without Streamlit, Supabase or patient data.
A future clinical release MUST introduce reviewed approval and update this
gate deliberately; editing a single flag must NOT silently pass CI.
"""
import ast
from pathlib import Path
import unittest

from medcalc_engine import renal_automatic_context_supported

ROOT = Path(__file__).resolve().parents[1]
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE)


def _release_assignments():
    found = []
    for node in ast.walk(APP_TREE):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "RENAL_DOSE_AUTOMATION_RELEASED":
                    found.append(node)
        if isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == "RENAL_DOSE_AUTOMATION_RELEASED":
                found.append(node)
    return found


def _rule(kind="REGIMEN", body="200 mg oral cada 24 h", notes="",
          flag="SI", metric="CrCl_CG_mL_min", low=30, high=60):
    return {
        "tipo_regla": kind, "regimen_ajustado": body, "notas": notes,
        "automatizable": flag, "metrica_renal": metric,
        "limite_inferior": low, "limite_superior": high,
    }


class RenalReleaseInterlockTests(unittest.TestCase):
    def test_exactly_one_release_assignment(self):
        self.assertEqual(len(_release_assignments()), 1)

    def test_release_flag_literal_false(self):
        statement = _release_assignments()[0]
        self.assertIsInstance(statement, ast.Assign)
        self.assertIsInstance(statement.value, ast.Constant)
        self.assertIs(statement.value.value, False)

    def test_ui_automated_rules_behind_gate(self):
        nodes = [
            n for n in ast.walk(APP_TREE) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "auto_rules" for t in n.targets)
        ]
        self.assertEqual(len(nodes), 1)
        exp = nodes[0].value
        self.assertIsInstance(exp, ast.IfExp)
        self.assertIsInstance(exp.test, ast.Name)
        self.assertEqual(exp.test.id, "RENAL_DOSE_AUTOMATION_RELEASED")
        self.assertIsInstance(exp.orelse, ast.List)
        self.assertEqual(len(exp.orelse.elts), 0)

    def test_numeric_regimen_is_structurally_recognized(self):
        self.assertTrue(renal_automatic_context_supported(_rule()))

    def test_no_adjustment_kept_as_reference(self):
        self.assertFalse(renal_automatic_context_supported(_rule(kind="NO_AJUSTE")))

    def test_mgkg_rows_held_even_if_flagged_automatic(self):
        self.assertFalse(renal_automatic_context_supported(_rule(body="5 mg/kg IV q12h")))

    def test_mg_per_kg_with_spaces_held(self):
        self.assertFalse(renal_automatic_context_supported(_rule(body="5 mg / kg cada 8 horas")))

    def test_ideal_weight_note_held(self):
        self.assertFalse(renal_automatic_context_supported(
            _rule(notes="Calcular con peso ideal o peso ajustado")
        ))

    def test_hemodialysis_timing_held_in_notes(self):
        self.assertFalse(renal_automatic_context_supported(
            _rule(notes="Administrar después de la sesión de hemodiálisis")
        ))

    def test_crrt_reference_held(self):
        self.assertFalse(renal_automatic_context_supported(
            _rule(notes="En CRRT considerar depuración extracorpórea")
        ))

    def test_unreviewed_flag_not_automatable(self):
        self.assertFalse(renal_automatic_context_supported(_rule(flag="NO")))

    def test_nonnumeric_rules_never_eligible(self):
        for kind in ("DIALISIS", "CONTRAINDICACION", "PORCENTAJE", "FORMULACION"):
            with self.subTest(rule_type=kind):
                self.assertFalse(renal_automatic_context_supported(_rule(kind=kind)))

    def test_unknown_renal_metric_blocked(self):
        self.assertFalse(renal_automatic_context_supported(_rule(metric="NO_METRIC")))

    def test_missing_or_reversed_band_blocked(self):
        self.assertFalse(renal_automatic_context_supported(_rule(low=None, high=None)))
        self.assertFalse(renal_automatic_context_supported(_rule(low=60, high=30)))


if __name__ == "__main__":
    unittest.main()
