import copy
import json
import unittest
from pathlib import Path

from ecg_g29_seven_audit_ui import (
    CODES, safe_seven_audit_rows, render_seven_audit_ui,
)


def fixture():
    return {
        'diagnostic_summary': {'findings': [{'code': 'SINUS_COMPATIBLE', 'publishable': True}]},
        'measurement_summary': {'qtc_bazett_ms': 430.0},
        'unsupported_seven_evidence': {
            'schema': 'NON_DIAGNOSTIC_SEVEN_MODULE_COVERAGE',
            'version': 'MEDCALC_ECG_G26_MISSING_SEVEN_EVIDENCE_V1',
            'model_status': 'NO_NEW_TRAINED_MODEL',
            'clinical_deployment_status': 'BLOCKED',
            'diagnostic_claim_allowed': False,
            'measurement_mutation_allowed': False,
            'reasoner_mutation_allowed': False,
            'modules': {code: {
                'module': code, 'status': 'NOT_EVALUABLE',
                'trained_model_available': False,
                'clinical_diagnostic_claim_allowed': False,
                'clinical_deployment_allowed': False,
                'score': None, 'neural_output': None,
                'binary_classification': None, 'diagnostic_label': None,
                'reason': 'INTERNAL_TEST_SHOULD_NEVER_BE_VISIBLE',
            } for code in CODES},
        },
    }


class FakeStreamlit:
    def __init__(self):
        self.calls=[]
    class _Expander:
        def __init__(self, owner): self.owner=owner
        def __enter__(self): return self
        def __exit__(self,exc_type,exc,tb): return False
    def expander(self,label,expanded=False):
        self.calls.append(('expander',label,expanded));return self._Expander(self)
    def warning(self,msg): self.calls.append(('warning',msg))
    def dataframe(self,rows,**kwargs): self.calls.append(('dataframe',rows,kwargs))
    def caption(self,msg): self.calls.append(('caption',msg))


class TestSevenResearchUI(unittest.TestCase):
    def test_normal(self):
        x=fixture();before=copy.deepcopy(x)
        z=safe_seven_audit_rows(x)
        self.assertEqual(z['status'],'ready')
        self.assertEqual(len(z['rows']),7)
        self.assertEqual(x,before)
        self.assertNotIn('INTERNAL_TEST',str(z))
        self.assertNotIn('SINUS_COMPATIBLE',str(z))
    def test_older_version(self):
        self.assertEqual(safe_seven_audit_rows({})['status'],'absent')
    def test_malformed(self):
        for bad in (None, [], 17, 'fake', {'unsupported_seven_evidence':[]}, {'unsupported_seven_evidence': {}}):
            self.assertEqual(safe_seven_audit_rows(bad)['status'],'blocked' if bad!={} else 'absent')
    def test_block_probability_and_model_output(self):
        for field,value in [('score',0.95),('neural_output',-2.5),('binary_classification',True),('diagnostic_label','AVB3')]:
            x=fixture();x['unsupported_seven_evidence']['modules']['AVB3'][field]=value
            self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked',field)
    def test_block_missing_score_key(self):
        x=fixture();del x['unsupported_seven_evidence']['modules']['AVB3']['score']
        self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked')
    def test_block_claim_flags(self):
        for key in ('trained_model_available','clinical_diagnostic_claim_allowed','clinical_deployment_allowed'):
            for val in [True, None, 0]:
                x=fixture();x['unsupported_seven_evidence']['modules']['AVB3'][key]=val
                self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked',key)
    def test_block_extra_missing_category(self):
        x=fixture();x['unsupported_seven_evidence']['modules'].pop('AVB3')
        self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked')
        x=fixture();x['unsupported_seven_evidence']['modules']['MALICIOUS']='FAKE'
        self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked')
    def test_block_bad_policy_or_schema(self):
        for k,v in [('schema','UNKNOWN'),('model_status','TRAINED'),('clinical_deployment_status','ACTIVE'),('diagnostic_claim_allowed',0),('reasoner_mutation_allowed',True)]:
            x=fixture();x['unsupported_seven_evidence'][k]=v
            self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked',k)
    def test_block_unknown_status(self):
        x=fixture();x['unsupported_seven_evidence']['modules']['SVT']['status']='CONFIRMED_SVT'
        self.assertEqual(safe_seven_audit_rows(x)['status'],'blocked')
    def test_streamlit_ready_render(self):
        s=FakeStreamlit();x=fixture();b=copy.deepcopy(x)
        self.assertEqual(render_seven_audit_ui(s,x),'ready')
        self.assertEqual(x,b)
        self.assertEqual([r[0] for r in s.calls],['expander','warning','dataframe','caption'])
        self.assertFalse(s.calls[0][2])
        self.assertEqual(len(s.calls[2][1]),7)
        self.assertNotIn('INTERNAL_TEST',str(s.calls))
    def test_streamlit_blocked_no_table(self):
        s=FakeStreamlit();x=fixture();x['unsupported_seven_evidence']['modules']['AVB3']['score']=0.9
        self.assertEqual(render_seven_audit_ui(s,x),'blocked')
        self.assertEqual([r[0] for r in s.calls],['warning'])
    def test_streamlit_absent_no_table(self):
        s=FakeStreamlit();self.assertEqual(render_seven_audit_ui(s,{}),'absent')
        self.assertEqual(s.calls,[])


if __name__=='__main__': unittest.main()