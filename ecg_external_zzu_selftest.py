from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from ecg_external_zzu_prepare import select
from ecg_external_zzu_score import _base_code, _parse_age_days, _split_codes, score


def _header(record: str, nsig: int = 12, fs: int = 500, nsamp: int = 5000) -> str:
    first=f"{record} {nsig} {fs} {nsamp}"
    lines=[first]
    leads=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"][:nsig]
    for lead in leads:
        lines.append(f"{record}.dat 16 1000/mV 16 0 0 0 0 {lead}")
    lines.append("# diagnosis: SHOULD_NOT_BE_READ")
    return "\n".join(lines)+"\n"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="zzu_selftest_") as td:
        root=Path(td)
        headers=root/"headers"
        headers.mkdir()
        for patient in range(1,7):
            for exam in (1,2):
                stem=f"P{patient:05d}_E{exam:02d}"
                (headers/f"{stem}.hea").write_text(_header(stem),encoding="utf-8")
        (headers/"P99999_E01.hea").write_text(_header("P99999_E01",nsig=9),encoding="utf-8")

        selection_dir=root/"selection"
        summary=select(headers,selection_dir,shards=2)
        assert summary["selected_patients"]==6, summary
        assert summary["selected_records"]==6, summary
        assert summary["rejected"]["not_12_lead"]==1, summary
        for h in (selection_dir/"sanitized_headers").rglob("*.hea"):
            text=h.read_text(encoding="utf-8")
            assert "diagnosis" not in text.lower(), text
            assert len(text.splitlines())==13, text

        manifest=pd.read_csv(selection_dir/"selection_manifest.csv",dtype=str)
        pred=[]
        attrs=[]
        for i,row in enumerate(manifest.itertuples(index=False)):
            pred.append({
                "patient_id":row.patient_id,
                "record_id":row.record_id,
                "pred_atrial_fibrillation":i==0,
                "pred_atrial_flutter":False,
                "pred_sinus_bradycardia":False,
                "pred_sinus_tachycardia":False,
                "pred_right_bundle_branch_block":False,
                "pred_left_bundle_branch_block":False,
                "pred_left_anterior_fascicular_block":False,
                "pred_left_posterior_fascicular_block":False,
                "pred_first_degree_av_block":False,
                "pred_second_degree_av_block":False,
                "pred_complete_av_block":False,
                "pred_ventricular_preexcitation":False,
                "publication_allowed":True,
                "abstention_n":0,
                "remeasure_required":False,
                "analysis_error":"",
            })
            attrs.append({
                "FileName":row.record_id,
                "Patient_ID":row.patient_id,
                "Age":f"{572+i*100}d",
                "AHA_Code":"'L145+Modifier362'" if i==0 else "",
            })
        pred_dir=root/"pred"
        pred_dir.mkdir()
        pd.DataFrame(pred).to_csv(pred_dir/"pred.csv",index=False)
        attr_path=root/"AttributesDictionary.csv"
        pd.DataFrame(attrs).to_csv(attr_path,index=False)
        code_path=root/"ECGCode.csv"
        pd.DataFrame([
            {"AHA_Code":"L145","AHA_Statement":"Atrial fibrillation"},
            {"AHA_Code":"SB1","AHA_Statement":"Sinus bradycardia"},
        ]).to_csv(code_path,index=False)

        assert _split_codes("'L145+Modifier362';'J(111+112+113)'")==["L145+Modifier362","J(111+112+113)"]
        assert _base_code("L145+Modifier362")=="L145"
        assert _base_code("J(111+112+113)")=="J(111+112+113)"
        assert _parse_age_days("572d")==572.0

        mapping_path=Path("ecg_external_validation_contracts/ZZU_PECG_SEMANTIC_MAPPING_V1.json")
        out=root/"summary.json"
        result=score(
            pred_dir,attr_path,code_path,mapping_path,
            selection_dir/"selection_summary.json",out
        )
        assert result["records_scored"]==6, result
        assert result["metrics"]["atrial_fibrillation"]["sensitivity"]==1.0, result
        assert result["metrics"]["atrial_fibrillation"]["specificity"]==1.0, result
        assert result["age_parse_rate"]==1.0, result
        assert result["gold_positive_total"]==1, result
        assert result["anti_leakage"]["row_level_gold_prediction_join_persisted"] is False, result
        assert not (root/"joined.csv").exists()

    print("MEDCALC_ZZU_PECG_EXTERNAL_HARNESS_SELFTEST_PASS")


if __name__=="__main__":
    main()
