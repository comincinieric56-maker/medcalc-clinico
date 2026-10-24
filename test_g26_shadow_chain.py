import copy
import unittest
from ecg_missing_seven_evidence import audit_missing_seven, MISSING_SEVEN
from ecg_trigeminy_pattern import audit_trigeminy

class G26ShadowChainTests(unittest.TestCase):
    def test_complete_shadow_chain_does_not_change_upstream_evidence(self):
        events=[{'beat_index': i, 'kind': 'PVC_COMPATIBLE'} for i in (2,5,8)]
        ect={'evaluable':True,'events':copy.deepcopy(events),'pvc_compatible_n':3}
        ect.update(audit_trigeminy(events,10))
        av={'evaluable':True,'classification':'TWO_TO_ONE_AV_BLOCK_COMPATIBLE','atrial_sequence_regular':True,'nonconducted_p_n':3}
        atr={'evaluable':True,'mechanism':'OTHER_SVT_COMPATIBLE'}
        rep={'st_elevation_leads':['V2','V3']}
        gates={'domains':{x:{'eligible':True} for x in ['AV_CONDUCTION','RHYTHM','ECTOPY','REPOLARIZATION']}}
        prior=copy.deepcopy([ect,av,atr,rep,gates])
        m=audit_missing_seven(av_conduction=av,atrial_mechanism=atr,ectopy=ect,repolarization=rep,domain_gates=gates)
        self.assertEqual(copy.deepcopy([ect,av,atr,rep,gates]),prior)
        self.assertEqual(len(m['modules']),7)
        for k in ('AVB2','SVT','TRIGEMINY'):
            self.assertEqual(m['modules'][k]['status'],'RESEARCH_PATTERN_FOR_REVIEW')
        self.assertEqual(m['modules']['ST_ELEVATION']['status'],'MEASURED_MORPHOLOGY_FOR_REVIEW')
        self.assertFalse(m['reasoner_mutation_allowed'])
        self.assertFalse(m['diagnostic_claim_allowed'])
        self.assertTrue(all(x['diagnostic_label'] is None for x in m['modules'].values()))
    def test_shadow_missing_gates_never_produces_positive(self):
        ect={'evaluable':True,'trigeminy_evaluable':True,'trigeminy_pattern':True,'trigeminy_repeated_cycles':4}
        a=audit_missing_seven(ectopy=ect)
        self.assertEqual(a['modules']['TRIGEMINY']['status'],'NOT_EVALUABLE')

if __name__=='__main__':unittest.main()