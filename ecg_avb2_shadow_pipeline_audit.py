from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from typing import Any

from ecg_signal_measurements import analyze_canonical_ecg
from ecg_synthetic_signal_cohort import DIAGNOSTIC_GROUPS, all_specs, canonical, make_signal

VERSION="MEDCALC_AVB2_SHADOW_PIPELINE_AUDIT_V1"

def run_shard(shard_index:int, shard_count:int)->dict[str,Any]:
    groups={k:Counter() for k in DIAGNOSTIC_GROUPS}
    controls=Counter()
    errors=[]
    for i,spec in enumerate(all_specs()):
        if i % shard_count != shard_index:
            continue
        try:
            analysis=analyze_canonical_ecg(canonical(spec,make_signal(spec)))
            ev=analysis.get("avb2_evidence_shadow")
            if not isinstance(ev,dict):
                raise RuntimeError("AVB2_SHADOW_MISSING")
            dst=groups[str(spec["target"])] if spec.get("kind")=="TARGET" else controls
            dst["n"]+=1
            dst["shadow_error_n"]+=int(ev.get("shadow_status")!="OK")
            dst["evaluable_n"]+=int(bool(ev.get("evaluable")))
            dst["compatible_n"]+=int(bool(ev.get("compatible")))
            dst["claim_allowed_n"]+=int(bool(ev.get("diagnostic_claim_allowed")))
        except Exception as exc:
            errors.append(f"{spec.get('case_id')}:{type(exc).__name__}:{exc}")
    return {
        "version":VERSION,
        "shard_index":shard_index,
        "shard_count":shard_count,
        "groups":{k:dict(v) for k,v in groups.items()},
        "controls":dict(controls),
        "errors":errors,
        "policy":"SYNTHETIC_1000_ONLY; SHADOW_ONLY; NO_THRESHOLD_TUNING; NO_FAST_GATE; NO_FOLD9",
    }

def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--shard-index",type=int,required=True)
    ap.add_argument("--shard-count",type=int,default=10)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=run_shard(args.shard_index,args.shard_count)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
