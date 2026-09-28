from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from ecg_external_mimic_prepare import prepare
from ecg_external_mimic_score import score


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="medcalc_mimic_selftest_") as td:
        root = Path(td)
        records = []
        for subject in range(1000, 1030):
            for k in (1, 2):
                study = subject * 10 + k
                records.append({
                    "subject_id": str(subject),
                    "study_id": str(study),
                    "path": (
                        f"files/p{str(subject)[:4]}/p{subject}/"
                        f"s{study}/{study}"
                    ),
                })
        record_list = root / "record_list.csv"
        pd.DataFrame(records).to_csv(record_list, index=False)

        prep_root = root / "prepared"
        selection = prepare(
            record_list,
            prep_root,
            patient_n=10,
            shards=2,
        )
        assert selection["selection_is_label_blind"] is True
        assert selection["record_n"] == 10
        manifest = pd.read_csv(
            prep_root / "mimic_selection_manifest.csv",
            dtype=str,
        )
        assert len(manifest) == 10
        assert manifest["subject_id"].nunique() == 10

        pred_root = root / "predictions"
        pred_root.mkdir()
        predictions = []
        machine = []
        for row in manifest.itertuples(index=False):
            predictions.append({
                "subject_id": row.subject_id,
                "study_id": row.study_id,
                "engine_hr_bpm": 76.0,
                "engine_pr_ms": 165.0,
                "engine_qrs_ms": 110.0,
                "engine_qt_ms": 420.0,
                "engine_qrs_axis_deg": 35.0,
                "remeasure_required": False,
                "analysis_error": "",
            })
            machine.append({
                "subject_id": row.subject_id,
                "study_id": row.study_id,
                "rr_interval": 800.0,
                "p_onset": 100.0,
                "p_end": 180.0,
                "qrs_onset": 260.0,
                "qrs_end": 360.0,
                "t_end": 660.0,
                "qrs_axis": 30.0,
            })
        pd.DataFrame(predictions).to_csv(
            pred_root / "mimic-pred-00.csv",
            index=False,
        )
        machine_path = root / "machine_measurements.csv"
        pd.DataFrame(machine).to_csv(machine_path, index=False)

        output = root / "summary.json"
        result = score(
            pred_root,
            machine_path,
            prep_root / "mimic_selection_summary.json",
            output,
        )
        assert result["records_scored"] == 10, result
        assert result["analysis_failure_rate"] == 0.0, result
        assert result["metrics"]["heart_rate_bpm"]["mae"] == 1.0, result
        assert result["metrics"]["pr_ms"]["mae"] == 5.0, result
        assert result["metrics"]["qrs_ms"]["mae"] == 10.0, result
        assert result["metrics"]["qt_ms"]["mae"] == 20.0, result
        assert result["metrics"]["qrs_axis_deg"]["mae"] == 5.0, result
        assert result["anti_leakage"]["row_level_joined_output_persisted"] is False
        assert not (root / "predictions_with_machine.csv").exists()

        persisted = json.loads(output.read_text(encoding="utf-8"))
        assert persisted["validation_id"] == "MIMIC_IV_ECG_FROZEN_MEASUREMENT_V1"

    print("MEDCALC_MIMIC_EXTERNAL_HARNESS_SELFTEST_PASS")


if __name__ == "__main__":
    main()
