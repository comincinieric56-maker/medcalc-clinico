import unittest
from ecg_trigeminy_pattern import audit_trigeminy

class TestTrigeminy(unittest.TestCase):
    def test_three_ventricular_cycles(self):
        x=audit_trigeminy([{'beat_index':i,'kind':'PVC_COMPATIBLE'} for i in (2,5,8)],10)
        self.assertTrue(x['trigeminy_pattern'])
        self.assertEqual(x['trigeminy_repeated_cycles'],3)
        self.assertFalse(x['diagnostic_claim_allowed'])
    def test_two_cycles_not_enough(self):
        self.assertFalse(audit_trigeminy([{'beat_index':i,'kind':'PVC_COMPATIBLE'} for i in (2,5)],10)['trigeminy_pattern'])
    def test_unrelated_premature_breaks_pattern(self):
        a=[{'beat_index':i,'kind':'PVC_COMPATIBLE'} for i in (2,5,8)]
        a.append({'beat_index':4,'kind':'PAC_OR_NARROW_PREMATURE_BEAT'})
        self.assertFalse(audit_trigeminy(a,10)['trigeminy_pattern'])
    def test_mixed_class_breaks_pattern(self):
        a=[{'beat_index':2,'kind':'PVC_COMPATIBLE'}, {'beat_index':5,'kind':'PAC_OR_NARROW_PREMATURE_BEAT'}, {'beat_index':8,'kind':'PVC_COMPATIBLE'}]
        self.assertFalse(audit_trigeminy(a,10)['trigeminy_pattern'])
    def test_short_not_evaluable(self):
        self.assertFalse(audit_trigeminy([{'beat_index':2,'kind':'PVC_COMPATIBLE'}],5)['trigeminy_evaluable'])
    def test_duplicate_invalid(self):
        a=[{'beat_index':2,'kind':'PVC_COMPATIBLE'}]*2
        self.assertFalse(audit_trigeminy(a,10)['trigeminy_evaluable'])
    def test_fractional_index_invalid(self):
        self.assertFalse(audit_trigeminy([{'beat_index':2.2,'kind':'PVC_COMPATIBLE'}],10)['trigeminy_evaluable'])
    def test_four_cycles(self):
        a=[{'beat_index':i,'kind':'PAC_OR_NARROW_PREMATURE_BEAT'} for i in (2,5,8,11)]
        x=audit_trigeminy(a,13)
        self.assertEqual(x['trigeminy_repeated_cycles'],4)
        self.assertTrue(x['trigeminy_pattern'])
        self.assertEqual(x['trigeminy_pattern_kind'],'PAC_OR_NARROW_PREMATURE_BEAT')

if __name__=='__main__':unittest.main()