from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path

import pandas as pd

NAMESPACE_RECORD = "MEDCALC_ZZU_PECG_RECORD_V1"
NAMESPACE_SHARD = "MEDCALC_ZZU_PECG_SHARD_V1"


def _hash(text: str, namespace: str) -> str:
    return hashlib.sha256(f"{namespace}|{text}".encode("utf-8")).hexdigest()


def _patient_id_from_stem(stem: str) -> str:
    m = re.match(r"^(P\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return stem.split("_", 1)[0]


def _parse_header_minimal(path: Path) -> tuple[int, float, int, list[str]]:
    # Read only record/signal specification lines. Never read trailing comments,
    # where clinical annotations may reside.
    with path.open("r", encoding="utf-8", errors="strict") as fh:
        first = fh.readline().rstrip("\r\n")
        fields = first.split()
        if len(fields) < 4:
            raise ValueError(f"{path}: malformed WFDB first line")
        nsig = int(fields[1])
        fs_token = fields[2].split("/", 1)[0]
        fs = float(fs_token)
        nsamp = int(fields[3])
        signal_lines = []
        for _ in range(nsig):
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated WFDB signal specification")
            signal_lines.append(line.rstrip("\r\n"))
    return nsig, fs, nsamp, [first, *signal_lines]


def select(headers_root: Path, output_dir: Path, shards: int) -> dict:
    headers = sorted(headers_root.rglob("*.hea"))
    if not headers:
        raise FileNotFoundError(f"No .hea files under {headers_root}")

    rows = []
    rejected = {"not_12_lead": 0, "not_500_hz": 0, "malformed": 0}
    for path in headers:
        try:
            nsig, fs, nsamp, sanitized_lines = _parse_header_minimal(path)
        except Exception:
            rejected["malformed"] += 1
            continue
        if nsig != 12:
            rejected["not_12_lead"] += 1
            continue
        if abs(fs - 500.0) > 1e-6:
            rejected["not_500_hz"] += 1
            continue
        stem = path.stem
        patient_id = _patient_id_from_stem(stem)
        rel = path.relative_to(headers_root)
        rows.append({
            "patient_id": patient_id,
            "record_id": stem,
            "header_relpath": rel.as_posix(),
            "dat_relpath": rel.with_suffix(".dat").as_posix(),
            "sample_count": int(nsamp),
            "duration_s": float(nsamp / fs),
            "_record_hash": _hash(f"{patient_id}|{stem}", NAMESPACE_RECORD),
            "_sanitized_lines": sanitized_lines,
        })

    if not rows:
        raise ValueError("No eligible 12-lead 500 Hz ZZU records found")

    df = pd.DataFrame([{k:v for k,v in r.items() if k != "_sanitized_lines"} for r in rows])
    line_map = {r["header_relpath"]: r["_sanitized_lines"] for r in rows}
    chosen = (
        df.sort_values(["patient_id", "_record_hash", "record_id"])
        .groupby("patient_id", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    chosen["_shard"] = chosen["record_id"].map(
        lambda x: int(_hash(str(x), NAMESPACE_SHARD)[:8], 16) % int(shards)
    )
    chosen = chosen.sort_values(["_shard", "patient_id", "record_id"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    sanitized_root = output_dir / "sanitized_headers"
    sanitized_root.mkdir(parents=True, exist_ok=True)

    for row in chosen.itertuples(index=False):
        rel = Path(row.header_relpath)
        out = sanitized_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(line_map[row.header_relpath]) + "\n", encoding="utf-8")

    chosen.drop(columns=["_record_hash"]).to_csv(
        output_dir / "selection_manifest.csv",
        index=False,
    )
    (output_dir / "selected_dat_paths.txt").write_text(
        "\n".join(chosen["dat_relpath"].astype(str)) + "\n",
        encoding="utf-8",
    )

    shard_counts = {}
    manifest_root = output_dir / "manifests"
    manifest_root.mkdir(exist_ok=True)
    for shard in range(int(shards)):
        part = chosen[chosen["_shard"] == shard].drop(columns=["_record_hash", "_shard"])
        part.to_csv(manifest_root / f"zzu-shard-{shard:02d}.csv", index=False)
        shard_counts[f"{shard:02d}"] = int(len(part))

    summary = {
        "version": "MEDCALC_ZZU_PECG_SELECTION_V1",
        "headers_seen": int(len(headers)),
        "eligible_12lead_500hz_records": int(len(df)),
        "selected_patients": int(chosen["patient_id"].nunique()),
        "selected_records": int(len(chosen)),
        "records_per_patient": 1,
        "selection_uses_labels": False,
        "attribute_dictionary_opened": False,
        "diagnosis_dictionary_opened": False,
        "header_comments_opened": False,
        "header_content_retained_for_inference": "WFDB_RECORD_AND_SIGNAL_SPEC_LINES_ONLY",
        "shards": int(shards),
        "shard_counts": shard_counts,
        "rejected": rejected,
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def package(selection_dir: Path, dat_root: Path, output_dir: Path) -> dict:
    manifest = pd.read_csv(selection_dir / "selection_manifest.csv", dtype=str)
    if manifest.empty:
        raise ValueError("Empty ZZU selection manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_values = sorted(int(x) for x in manifest["_shard"].unique())
    counts = {}
    for shard in shard_values:
        rows = manifest[pd.to_numeric(manifest["_shard"]) == shard].copy()
        with tarfile.open(output_dir / f"zzu-shard-{shard:02d}.tar", "w") as tf:
            temp = output_dir / f".work-{shard:02d}"
            if temp.exists():
                shutil.rmtree(temp)
            (temp / "records").mkdir(parents=True)
            rows.drop(columns=["_shard"]).to_csv(temp / "manifest.csv", index=False)

            for row in rows.itertuples(index=False):
                hrel = Path(row.header_relpath)
                drel = Path(row.dat_relpath)
                src_h = selection_dir / "sanitized_headers" / hrel
                src_d = dat_root / drel
                if not src_h.is_file():
                    raise FileNotFoundError(src_h)
                if not src_d.is_file():
                    raise FileNotFoundError(src_d)
                dst_h = temp / "records" / hrel
                dst_d = temp / "records" / drel
                dst_h.parent.mkdir(parents=True, exist_ok=True)
                dst_d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_h, dst_h)
                shutil.copy2(src_d, dst_d)

            tf.add(temp / "manifest.csv", arcname="manifest.csv")
            tf.add(temp / "records", arcname="records")
            shutil.rmtree(temp)
        counts[f"{shard:02d}"] = int(len(rows))

    summary = {"shards": len(shard_values), "shard_counts": counts}
    (output_dir / "package_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select")
    s.add_argument("--headers-root", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--shards", type=int, default=16)

    p = sub.add_parser("package")
    p.add_argument("--selection-dir", type=Path, required=True)
    p.add_argument("--dat-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "select":
        result = select(args.headers_root, args.output_dir, args.shards)
    else:
        result = package(args.selection_dir, args.dat_root, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
