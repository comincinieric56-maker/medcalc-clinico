from __future__ import annotations

from ecg_r27_consensus import compare_r27_with_medcalc
from ecg_unet_r27_bridge import _attach_r27_independent_qa


def payload(**scores):
    modules = {}
    for name, score in scores.items():
        modules[name] = {
            "probability": score,
            "interpretability": "PROBABILITY_ONLY",
        }
    return {"modules": modules, "input_adapter": {"r27_tiled": False}}


def structured(code: str):
    return {
        "specialist_reasoning": {
            "diagnostic_summary": {
                "findings": [{
                    "domain": "RHYTHM",
                    "code": code,
                    "confidence": 0.93,
                    "publishable": True,
                }]
            }
        },
        "repolarization": {},
        "measurement_summary": {},
    }


def main() -> None:
    agree = compare_r27_with_medcalc(structured("AF_COMPATIBLE"), payload(AF=0.91))
    assert agree["cross_engine_support_n"] == 1, agree
    assert agree["comparisons"][0]["status"] == "CROSS_ENGINE_SUPPORT", agree

    disagree = compare_r27_with_medcalc(structured("LBBB_MORPHOLOGY_COMPATIBLE"), payload(LBBB=0.07))
    assert disagree["discordance_review_n"] == 1, disagree
    assert not disagree["diagnostic_mutation_allowed"], disagree

    r27_only = compare_r27_with_medcalc(
        {"specialist_reasoning": {"diagnostic_summary": {"findings": []}}},
        payload(RBBB_COMPLETE=0.94),
    )
    assert r27_only["r27_only_review_signals"][0]["r27_module"] == "RBBB_COMPLETE"
    assert not r27_only["r27_only_review_signals"][0]["diagnostic_claim_allowed"]

    tiled = {
        "modules": {
            "AF": {
                "probability": 0.99,
                "interpretability": "NOT_INTERPRETABLE_R27_TILED",
            }
        },
        "input_adapter": {"r27_tiled": True},
    }
    no_temporal = compare_r27_with_medcalc(structured("AF_COMPATIBLE"), tiled)
    assert no_temporal["comparisons"][0]["status"] == "NOT_COMPARABLE", no_temporal

    structured_report = structured("LBBB_MORPHOLOGY_COMPATIBLE")
    bridge_payload = payload(LBBB=0.91)
    result = {
        "payload": bridge_payload,
        "digitizer": {
            "structured_report": structured_report,
            "signal": {"r27_tiled": False},
        },
    }
    before_report = {
        "specialist_reasoning": structured_report["specialist_reasoning"].copy(),
        "repolarization": dict(structured_report.get("repolarization") or {}),
        "measurement_summary": dict(structured_report.get("measurement_summary") or {}),
    }
    before_probability = bridge_payload["modules"]["LBBB"]["probability"]
    attached = _attach_r27_independent_qa(result)
    qa = attached.get("r27_independent_qa") or {}
    assert qa.get("pipeline_position") == "POST_REASONER_POST_REPORT_AUDIT", qa
    assert qa.get("measurement_mutation_allowed") is False, qa
    assert qa.get("diagnostic_mutation_allowed") is False, qa
    assert (
        attached["digitizer"]["structured_report"]["specialist_reasoning"]
        == before_report["specialist_reasoning"]
    ), attached
    assert attached["payload"]["modules"]["LBBB"]["probability"] == before_probability
    assert attached["digitizer"]["r27_independent_qa"] == qa, attached

    print("MEDCALC_R27_BRIDGE_POST_REASONER_QA_PASS")
    print("MEDCALC_R27_CONSENSUS_SELFTEST_PASS")


if __name__ == "__main__":
    main()
