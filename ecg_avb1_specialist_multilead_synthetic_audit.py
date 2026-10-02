from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ecg_adult_diagnostic_dev_benchmark import _pr_multilead_audit, _published_codes
from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import (
    CONTROL_N,
    DIAGNOSTIC_CASES_EACH,
    DIAGNOSTIC_GROUPS,
    ROLE,
    TOTAL_CASES,
    all_specs,
    canonical,
    make_signal,
)

VERSION="MEDCALC_AVB1_SPECIALIST_MULTILEAD_SYNTHETIC_AUDIT_V1"
CODE="FIRST_DEGREE_AV_DELAY_COMPATIBLE"
STRONG={"1_TO_1_P_QRS","PR_MEDIAN_GT_200MS","PR_STABLE"}


def _pr_only_block(fusion:dict)->bool:
    if bool(fusion.get("publishable")):
        return False
    gate=dict(fusion.get("domain_gate") or {})
    if not bool(gate.get("eligible")) or gate.get("blocked_by_conflicts"):
        return False
    unresolved={str(x) for x in (fusion.get("unresolved_required_measurements") or [])}
    boundary=[dict(x or {}) for x in (fusion.get("boundary_failures") or [])]
    reason=str(fusion.get("fusion_reason") or "")
    if reason=="REQUIRED_MEASUREMENT_UNUSABLE":
        return unresolved=={"pr_ms"}
    if reason=="REQUIRED_THRESHOLD_NOT_CONFIDENTLY_SATISFIED":
        metrics={str(x.get("metric") or "") for x in boundary if str(x.get("metric") or "")}
        return bool(metrics) and metrics=={"pr_ms"} and not unresolved
    return False


def _row(a:dict)->dict:
    cand=dict((((a.get("high_recall_candidates") or {}).get("by_code") or {}).get(CODE)) or {})
    fused=dict((((a.get("evidence_fusion") or {}).get("by_code") or {}).get(CODE)) or {})
    published=CODE in _published_codes(a)
    ev={str(x) for x in (cand.get("evidence") or [])}
    strong=bool(cand and cand.get("specialist_confirmed") and STRONG.issubset(ev))
    blocked=bool(cand and _pr_only_block(fused))
    base_rescue=bool(strong and blocked and not published)
    ml=_pr_multilead_audit(a)
    ge2=bool(ml.get("ge2_pr_gt_200_leads"))
    median=ml.get("usable_lead_median_ms")
    median_gt200=bool(median is not None and float(median)>200.0)
    return {
        "published":published,
        "strong":strong,
        "base_rescue":base_rescue,
        "ge2":ge2,
        "median_gt200":median_gt200,
        "strict_rescue":bool(base_rescue and ge2 and median_gt200),
    }


def _apply(dst:Counter,row:dict)->None:
    dst["n"]+=1
    dst["baseline_final_avb1_n"]+=int(row["published"])
    dst["strong_signature_n"]+=int(row["strong"])
    dst["base_rescue_n"]+=int(row["base_rescue"])
    dst["ge2_long_pr_n"]+=int(row["ge2"])
    dst["multilead_median_gt200_n"]+=int(row["median_gt200"])
    dst["strict_rescue_n"]+=int(row["strict_rescue"])


def run_shard(shard_index:int,shard_count:int)->dict:
    specs=all_specs()
    selected=[spec for i,spec in enumerate(specs) if i%shard_count==shard_index]
    per_target={target:Counter() for target in DIAGNOSTIC_GROUPS}
    controls=Counter()
    control_types={}
    errors=[]
    for pos,spec in enumerate(selected,1):
        try:
            a=analyze_canonical_ecg(canonical(spec,make_signal(spec)))
            row=_row(a)
            if spec.get("kind")=="TARGET":
                _apply(per_target[str(spec["target"])],row)
            else:
                _apply(controls,row)
                ctype=str(spec.get("control_type") or "UNKNOWN")
                _apply(control_types.setdefault(ctype,Counter()),row)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}:{exc}")
        if pos%10==0:
            print(f"MEDCALC_AVB1_MULTILEAD_SYNTHETIC {shard_index}/{shard_count} {pos}/{len(selected)}",flush=True)
    return {
        "version":VERSION,
        "role":ROLE,
        "purpose":"AUDIT_ONLY_AVB1_SPECIALIST_MULTILEAD_TRIGGER",
        "external_validation_claim_allowed":False,
        "shard_index":shard_index,
        "shard_count":shard_count,
        "per_target":{k:dict(v) for k,v in per_target.items()},
        "controls":dict(controls),
        "control_types":{k:dict(v) for k,v in control_types.items()},
        "errors":errors,
        "policy":"SYNTHETIC_1000_ONLY; NO_CLINICAL_OUTPUT_CHANGE; EXISTING_GE2_PR_GT_200_LEADS_CONF_GE_0_50; EXISTING_200MS_THRESHOLD; NO_FAST; NO_FOLD9_10; NO_EXTERNAL",
    }


def aggregate(root:Path)->dict:
    files=sorted(root.glob("**/avb1-multilead-synthetic-shard-*.json"))
    if not files:
        raise RuntimeError("No shard files")
    docs=[json.loads(p.read_text()) for p in files]
    per_target={target:Counter() for target in DIAGNOSTIC_GROUPS}
    controls=Counter(); control_types={}; errors=[]
    for d in docs:
        errors.extend(d.get("errors") or [])
        for target,row in (d.get("per_target") or {}).items():
            per_target[target].update({str(k):int(v) for k,v in row.items()})
        controls.update({str(k):int(v) for k,v in (d.get("controls") or {}).items()})
        for ctype,row in (d.get("control_types") or {}).items():
            control_types.setdefault(ctype,Counter()).update({str(k):int(v) for k,v in row.items()})
    out={
        "version":VERSION+"_AGGREGATE",
        "role":ROLE,
        "purpose":"AUDIT_ONLY_AVB1_SPECIALIST_MULTILEAD_TRIGGER",
        "external_validation_claim_allowed":False,
        "case_count":sum(v["n"] for v in per_target.values())+controls["n"],
        "expected_case_count":TOTAL_CASES,
        "expected_control_n":CONTROL_N,
        "expected_diagnostic_case_n_each":DIAGNOSTIC_CASES_EACH,
        "per_target":{k:dict(v) for k,v in per_target.items()},
        "controls":dict(controls),
        "control_types":{k:dict(v) for k,v in control_types.items()},
        "errors":errors,
        "policy":"SYNTHETIC_1000_ONLY; NO_CLINICAL_OUTPUT_CHANGE; EXISTING_GE2_PR_GT_200_LEADS_CONF_GE_0_50; EXISTING_200MS_THRESHOLD; NO_FAST; NO_FOLD9_10; NO_EXTERNAL",
    }
    if out["case_count"]!=TOTAL_CASES:
        raise RuntimeError(f"case_count={out['case_count']} expected={TOTAL_CASES}")
    if errors:
        raise RuntimeError(f"errors={errors[:5]}")
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-index",type=int)
    ap.add_argument("--shard-count",type=int,default=10)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--aggregate-dir",type=Path)
    args=ap.parse_args()
    if args.aggregate_dir:
        out=aggregate(args.aggregate_dir)
    else:
        if args.shard_index is None:
            raise SystemExit("--shard-index required without --aggregate-dir")
        out=run_shard(args.shard_index,args.shard_count)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
