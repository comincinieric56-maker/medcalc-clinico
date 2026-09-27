from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ecg_validation_harness import validation_metrics


def _run_case(
    *,
    source: Path,
    vendor_root: Path,
    segmentation_model: Path,
    lead_model: Path,
    output_root: Path,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "digitizer_meta.json"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("ecg_unet_worker.py")),
        "--vendor-root", str(vendor_root),
        "--segmentation-model", str(segmentation_model),
        "--lead-model", str(lead_model),
        "--source", str(source),
        "--output-root", str(output_root / "worker"),
        "--meta", str(meta_path),
        "--pdf-page-index", "0",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=2400,
    )
    meta = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["runner_returncode"] = int(proc.returncode)
    meta["runner_stdout_tail"] = proc.stdout[-8000:]
    meta["runner_stderr_tail"] = proc.stderr[-8000:]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run synthetic rendered ECG cases through the same U-Net/digitizer "
            "worker used by MEDCALC and compare recovered measurements with truth."
        )
    )
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--vendor-root", required=True)
    ap.add_argument("--segmentation-model", required=True)
    ap.add_argument("--lead-model", required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--output", default="validation_results.json")
    args = ap.parse_args()

    case_dir = Path(args.case_dir).resolve()
    truth = json.loads(
        (case_dir / "ground_truth.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (case_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if int(args.max_cases) > 0:
        manifest = manifest[: int(args.max_cases)]

    results = []
    for idx, case in enumerate(manifest, start=1):
        source = case_dir / case["file"]
        with tempfile.TemporaryDirectory(prefix=f"ecg_validation_{idx:03d}_") as tmp:
            meta = _run_case(
                source=source,
                vendor_root=Path(args.vendor_root).resolve(),
                segmentation_model=Path(args.segmentation_model).resolve(),
                lead_model=Path(args.lead_model).resolve(),
                output_root=Path(tmp),
            )
        structured = meta.get("structured_report") or {}
        digital = (meta.get("signal") or {}).get("digital_ecg") or {}
        metrics = validation_metrics(
            structured,
            truth,
            recovered_digital_ecg=digital,
        )
        results.append({
            "case": case,
            "status": meta.get("status"),
            "reason": meta.get("reason"),
            "layout": (meta.get("signal") or {}).get("layout_name"),
            "digital_schema": (meta.get("signal") or {}).get(
                "digital_signal_schema"
            ),
            "clinical_source": (meta.get("signal") or {}).get(
                "clinical_measurement_source"
            ),
            "recovered_lead_count": digital.get("recovered_lead_count"),
            "metrics": metrics,
            "runner_returncode": meta.get("runner_returncode"),
        })

    def mean_metric(name: str):
        vals = [
            float((row.get("metrics") or {})[name])
            for row in results
            if (row.get("metrics") or {}).get(name) is not None
        ]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "schema": "MEDCALC_ECG_VALIDATION_RESULTS_V1",
        "case_count": len(results),
        "pipeline_success_count": sum(
            1 for row in results
            if row.get("runner_returncode") == 0
            and row.get("digital_schema") == "MEDCALC_DIGITAL_ECG_V2"
        ),
        "mean_MAE_QRS_MS": mean_metric("MAE_QRS_MS"),
        "mean_MAE_PR_MS": mean_metric("MAE_PR_MS"),
        "mean_MAE_QT_MS": mean_metric("MAE_QT_MS"),
        "mean_MAE_ST_MV": mean_metric("MAE_ST_MV"),
        "mean_ERROR_RR_MS": mean_metric("ERROR_RR_MS"),
        "mean_ERROR_FC_BPM": mean_metric("ERROR_FC_BPM"),
        "mean_MAE_AMPLITUDE_MV": mean_metric("MAE_AMPLITUDE_MV"),
        "mean_RECOVERED_LEAD_PERCENT": mean_metric(
            "RECOVERED_LEAD_PERCENT"
        ),
        "results": results,
    }

    Path(args.output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
