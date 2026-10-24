import copy
import unittest

from ecg_g28_strict_parity import ADDED_FIELDS, G26_MODULES, _walk, compare_pair


def _pair():
    base={"subject":1,"success":True,"fields":{
        "ectopy":{"evaluable":True,"premature_beat_n":1},
        "feature_graph":{"specialist_evidence":{"ectopy":{"evaluable":True,"premature_beat_n":1}}},
        "report":{"ectopy":{"premature_beat_n":1},
            "feature_graph":{"specialist_evidence":{"ectopy":{"premature_beat_n":1}}},
            "digital_measurements_v2":{"ectopy":{"premature_beat_n":1},
                "feature_graph":{"specialist_evidence":{"ectopy":{"premature_beat_n":1}}}},
            "specialist_reasoning":{"diagnostic_summary":{"findings":[]}},
            "global_measurements":{"qrs_ms":111,"qtc_ms":415},
            "formatted":{"conclusion":"SIN CAMBIOS"}},
        "global":{"heart_rate_bpm":75},"rhythm":{"evaluable":True}}}
    feature=copy.deepcopy(base)
    for p in sorted(ADDED_FIELDS):
        obj=feature["fields"]
        for k in p[:-1]:obj=obj[k]
        if p==("report","unsupported_seven_evidence"):
            obj[p[-1]]={"schema":"NON_DIAGNOSTIC_SEVEN_MODULE_COVERAGE",
             "version":"MEDCALC_ECG_G26_MISSING_SEVEN_EVIDENCE_V1",
             "model_status":"NO_NEW_TRAINED_MODEL",
             "diagnostic_claim_allowed":False,
             "clinical_deployment_status":"BLOCKED",
             "measurement_mutation_allowed":False,"reasoner_mutation_allowed":False,
             "modules":{key:{"module":key,"trained_model_available":False,"score":None,
                "neural_output":None,"binary_classification":None,
                "diagnostic_label":None,"clinical_diagnostic_claim_allowed":False,
                "clinical_deployment_allowed":False,"status":"NOT_EVALUABLE"}
                for key in G26_MODULES}}
        else:obj[p[-1]]= (False if p[-1] in ("trigeminy_evaluable","trigeminy_pattern") else
                                0 if p[-1]=="trigeminy_repeated_cycles" else
                                True if p[-1]=="review_only" else None)
    return base,feature


class StrictG28ContractTests(unittest.TestCase):
    def test_valid_additive_contract(self):
        a,b=_pair();self.assertTrue(compare_pair(a,b)["pass"])
        self.assertEqual(compare_pair(a,b)["added_n"],37)
    def test_new_report_field_wont_hide_existing_drift(self):
        a,b=_pair();b["fields"]["report"]["global_measurements"]["qtc_ms"]=430
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_ectopy_count_drift(self):
        a,b=_pair();b["fields"]["ectopy"]["premature_beat_n"]=2
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_reasoner_change(self):
        a,b=_pair();b["fields"]["report"]["specialist_reasoning"]["diagnostic_summary"]["findings"].append({"code":"AF"})
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_ectopy_extra_diagnosis_forbidden(self):
        a,b=_pair();b["fields"]["ectopy"]["diagnostic_label"]="TRIGEMINY"
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_non_contract_nested_addition_forbidden(self):
        a,b=_pair();b["fields"]["report"]["global_measurements"]["trigeminy_pattern"]=True
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_missing_contract_field(self):
        a,b=_pair();del b["fields"]["report"]["unsupported_seven_evidence"]
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_neural_score_forbidden(self):
        a,b=_pair();b["fields"]["report"]["unsupported_seven_evidence"]["modules"]["AVB3"]["score"]=0.9
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_clinical_deployment_forbidden(self):
        a,b=_pair();b["fields"]["report"]["unsupported_seven_evidence"]["clinical_deployment_status"]="ALLOWED"
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_missing_seven_category_forbidden(self):
        a,b=_pair();del b["fields"]["report"]["unsupported_seven_evidence"]["modules"]["AVB2"]
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_reject_forged_g26_status(self):
        a,b=_pair()
        b["fields"]["report"]["unsupported_seven_evidence"]["modules"]["AVB3"]["status"]="CONFIRMED_AVB3"
        self.assertFalse(compare_pair(a,b)["pass"])

    def test_reject_mismatched_g26_module_identity(self):
        a,b=_pair()
        b["fields"]["report"]["unsupported_seven_evidence"]["modules"]["AVB2"]["module"]="AVB3"
        self.assertFalse(compare_pair(a,b)["pass"])

    def test_reject_altered_g26_manifest_contract(self):
        for key,value in (
            ("schema","UNRECOGNIZED"),
            ("version","FAKE_VERSION"),
            ("model_status","TRAINED"),
        ):
            with self.subTest(key=key):
                a,b=_pair()
                b["fields"]["report"]["unsupported_seven_evidence"][key]=value
                self.assertFalse(compare_pair(a,b)["pass"])

    def test_reject_nonboolean_research_flags(self):
        for key in ("diagnostic_claim_allowed","measurement_mutation_allowed","reasoner_mutation_allowed"):
            with self.subTest(key=key):
                a,b=_pair()
                b["fields"]["report"]["unsupported_seven_evidence"][key]=0
                self.assertFalse(compare_pair(a,b)["pass"])

    def test_failclosed_on_failed_inference(self):
        a,b=_pair();b["success"]=False
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_failclosed_on_mismatched_patient(self):
        a,b=_pair();b["subject"]=2
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_type_changes_detected(self):
        a,b=_pair();b["fields"]["global"]["heart_rate_bpm"]="75"
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_list_order_and_element_changes_detected(self):
        a,b=_pair();a["fields"]["report"]["specialist_reasoning"]["diagnostic_summary"]["findings"]=[{"code":"SINUS"}]
        self.assertFalse(compare_pair(a,b)["pass"])
    def test_no_removed_fields(self):
        a,b=_pair();del b["fields"]["rhythm"]
        self.assertFalse(compare_pair(a,b)["pass"])


if __name__=='__main__':unittest.main()