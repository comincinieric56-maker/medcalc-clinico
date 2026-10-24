import unittest
from ecg_missing_seven_evidence import audit_missing_seven, MISSING_SEVEN

GATES = {'domains': {x: {'eligible': True} for x in ('AV_CONDUCTION','RHYTHM','ECTOPY','REPOLARIZATION')}}

class TestG26SevenEvidence(unittest.TestCase):
    def test_seven_fail_closed_without_inputs(self):
        a=audit_missing_seven()
        self.assertEqual(set(a['modules']),set(MISSING_SEVEN))
        self.assertEqual(a['clinical_deployment_status'],'BLOCKED')
        self.assertEqual(a['modules']['AVB2']['status'],'NOT_EVALUABLE')
        self.assertEqual(a['modules']['SVT']['status'],'NOT_EVALUABLE')
        self.assertEqual(a['modules']['ST_ELEVATION']['status'],'NOT_EVALUABLE')
    def test_no_neural_or_clinical_output(self):
        a=audit_missing_seven(domain_gates=GATES)
        self.assertFalse(a['diagnostic_claim_allowed'])
        for item in a['modules'].values():
            self.assertFalse(item['trained_model_available'])
            for field in ('score','neural_output','binary_classification','diagnostic_label'):
                self.assertIsNone(item[field])
            self.assertFalse(item['clinical_diagnostic_claim_allowed'])
    def test_avb2_valid_pattern(self):
        a=audit_missing_seven(av_conduction={'evaluable':True,'classification':'MOBITZ_II_COMPATIBLE','atrial_sequence_regular':True,'nonconducted_p_n':1},domain_gates=GATES)
        self.assertEqual(a['modules']['AVB2']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
        self.assertNotEqual(a['modules']['AVB3']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
    def test_avb2_lacks_mechanistic_support(self):
        a=audit_missing_seven(av_conduction={'evaluable':True,'classification':'MOBITZ_II_COMPATIBLE','nonconducted_p_n':0},domain_gates=GATES)
        self.assertEqual(a['modules']['AVB2']['status'],'INCONSISTENT_MECHANISTIC_EVIDENCE')
    def test_avb3_requires_dissociation(self):
        av={'evaluable':True,'classification':'COMPLETE_AV_BLOCK_COMPATIBLE','atrial_sequence_regular':True,'ventricular_sequence_regular':True}
        a=audit_missing_seven(av_conduction=av,domain_gates=GATES)
        self.assertEqual(a['modules']['AVB3']['status'],'INCONSISTENT_MECHANISTIC_EVIDENCE')
        av['av_dissociation_phase']=True
        a=audit_missing_seven(av_conduction=av,domain_gates=GATES)
        self.assertEqual(a['modules']['AVB3']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
    def test_gates_cannot_be_bypassed(self):
        a=audit_missing_seven(av_conduction={'evaluable':True,'classification':'MOBITZ_II_COMPATIBLE','atrial_sequence_regular':True,'nonconducted_p_n':3},domain_gates={'domains':{'AV_CONDUCTION':{'eligible':False}}})
        self.assertEqual(a['modules']['AVB2']['status'],'NOT_EVALUABLE')
    def test_svt_evaluable(self):
        a=audit_missing_seven(atrial_mechanism={'evaluable':True,'mechanism':'OTHER_SVT_COMPATIBLE'},domain_gates=GATES)
        self.assertEqual(a['modules']['SVT']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
    def test_trigeminy_minimum_cycles(self):
        a=audit_missing_seven(ectopy={'evaluable':True,'trigeminy_evaluable':True,'trigeminy_pattern':True,'trigeminy_repeated_cycles':2},domain_gates=GATES)
        self.assertNotEqual(a['modules']['TRIGEMINY']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
        b=audit_missing_seven(ectopy={'evaluable':True,'trigeminy_evaluable':True,'trigeminy_pattern':True,'trigeminy_repeated_cycles':3},domain_gates=GATES)
        self.assertEqual(b['modules']['TRIGEMINY']['status'],'RESEARCH_PATTERN_FOR_REVIEW')
    def test_trigeminy_needs_nine_beats(self):
        a=audit_missing_seven(ectopy={'evaluable':True,'trigeminy_evaluable':False,'trigeminy_pattern':False},domain_gates=GATES)
        self.assertEqual(a['modules']['TRIGEMINY']['status'],'NOT_EVALUABLE')
    def test_st_leads_not_infarction(self):
        a=audit_missing_seven(repolarization={'st_elevation_leads':['V2','V3']},domain_gates=GATES)
        self.assertEqual(a['modules']['ST_ELEVATION']['status'],'MEASURED_MORPHOLOGY_FOR_REVIEW')
        self.assertEqual(a['modules']['ISCHEMIA_GENERIC']['status'],'CLINICAL_CONTEXT_REQUIRED')
        self.assertEqual(a['modules']['MI_HISTORY_Q_SCREEN']['status'],'VALIDATED_Q_CRITERIA_REQUIRED')
    def test_reject_malformed_inputs(self):
        a=audit_missing_seven(ectopy=[],atrial_mechanism='fake',av_conduction=42,domain_gates={'domains':[]},repolarization=4)
        self.assertEqual(a['modules']['TRIGEMINY']['status'],'NOT_EVALUABLE')
    def test_no_mutation(self):
        av={'evaluable':True,'classification':'MOBITZ_II_COMPATIBLE','atrial_sequence_regular':True,'nonconducted_p_n':2}
        copy=av.copy()
        audit_missing_seven(av_conduction=av,domain_gates=GATES)
        self.assertEqual(av,copy)

if __name__=='__main__':unittest.main()