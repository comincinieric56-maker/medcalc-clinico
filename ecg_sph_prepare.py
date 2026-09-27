from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


SALT = "MEDCALC_SPH_EXTERNAL_V1"


def bucket(value: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{SALT}|{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(modulus)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--records-tar", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--patient-modulus", type=int, default=20)
    ap.add_argument("--patient-remainder", type=int, default=0)
    ap.add_argument("--shards", type=int, default=8)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    selected = [
        row for row in rows
        if bucket(str(row["Patient_ID"]), args.patient_modulus)
        == int(args.patient_remainder)
    ]
    if not selected:
        raise RuntimeError("Frozen patient-hash selection produced zero records.")

    selected_ids = {str(row["ECG_ID"]) for row in selected}
    patient_ids = {str(row["Patient_ID"]) for row in selected}
    shards = {ecg_id: bucket(ecg_id, args.shards) for ecg_id in selected_ids}

    gold_dir = args.output_dir / "gold"
    gold_dir.mkdir(exist_ok=True)
    gold_fields = ["ECG_ID","Patient_ID","AHA_Code","Age","Sex","N","Date"]
    with (gold_dir / "gold.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gold_fields)
        w.writeheader()
        for row in sorted(selected, key=lambda x: str(x["ECG_ID"])):
            w.writerow({k: row.get(k, "") for k in gold_fields})
    shutil.copy2(args.code, gold_dir / "code.csv")

    manifest_handles = {}
    artifact_dirs = {}
    for shard in range(args.shards):
        d = args.output_dir / f"artifact-{shard:02d}"
        d.mkdir(exist_ok=True)
        artifact_dirs[shard] = d
        fh = (d / "manifest.csv").open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(fh, fieldnames=["ECG_ID","N"])
        writer.writeheader()
        manifest_handles[shard] = (fh, writer)

    for row in sorted(selected, key=lambda x: str(x["ECG_ID"])):
        sid = shards[str(row["ECG_ID"])]
        manifest_handles[sid][1].writerow({
            "ECG_ID": row["ECG_ID"],
            "N": row.get("N",""),
        })
    for fh, _ in manifest_handles.values():
        fh.close()

    output_tars = {
        shard: tarfile.open(artifact_dirs[shard] / "records.tar", mode="w")
        for shard in range(args.shards)
    }
    found = set()
    try:
        with tarfile.open(args.records_tar, mode="r:*") as src:
            for member in src:
                if not member.isfile():
                    continue
                stem = Path(member.name).stem
                if stem not in selected_ids:
                    continue
                fileobj = src.extractfile(member)
                if fileobj is None:
                    continue
                out_member = tarfile.TarInfo(name=f"{stem}.h5")
                out_member.size = int(member.size)
                out_member.mode = 0o644
                output_tars[shards[stem]].addfile(out_member, fileobj)
                found.add(stem)
    finally:
        for tf in output_tars.values():
            tf.close()

    missing = sorted(selected_ids - found)
    if missing:
        raise RuntimeError(
            f"Selected SPH records missing from archive: {missing[:20]} "
            f"(missing_n={len(missing)})"
        )

    summary = {
        "selection_salt": SALT,
        "patient_modulus": int(args.patient_modulus),
        "patient_remainder": int(args.patient_remainder),
        "selected_patient_n": len(patient_ids),
        "selected_record_n": len(selected_ids),
        "shards": int(args.shards),
        "labels_used_for_selection": False,
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
