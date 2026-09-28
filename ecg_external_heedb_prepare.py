from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

PATIENT_NAMESPACE = "MEDCALC_HEEDB_PATIENT_V1"
RECORD_NAMESPACE = "MEDCALC_HEEDB_RECORD_V1"
SITES = ("I0001", "I0006")


def _h(value: str, ns: str) -> str:
    return hashlib.sha256(f"{ns}|{value}".encode("utf-8")).hexdigest()


def _load_site(path: Path, site: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, usecols=lambda c: c in {"BDSPPatientID", "FileName"})
    required = {"BDSPPatientID", "FileName"}
    if not required.issubset(df.columns):
        raise ValueError(f"{site}: metadata missing {required - set(df.columns)}")
    out = df[["BDSPPatientID", "FileName"]].dropna().copy()
    out["BDSPPatientID"] = out["BDSPPatientID"].astype(str).str.strip()
    out["FileName"] = out["FileName"].astype(str).str.strip()
    out = out[(out["BDSPPatientID"] != "") & (out["FileName"] != "")]
    out["site"] = site
    if out.empty:
        raise ValueError(f"{site}: no usable metadata rows")
    return out


def _select_site(df: pd.DataFrame, site: str, n: int) -> pd.DataFrame:
    patients = (
        df[["BDSPPatientID"]]
        .drop_duplicates()
        .assign(
            _ph=lambda x: x["BDSPPatientID"].map(
                lambda v: _h(f"{site}|{v}", PATIENT_NAMESPACE)
            )
        )
        .sort_values(["_ph", "BDSPPatientID"])
    )
    if len(patients) < n:
        raise ValueError(f"{site}: requested {n} patients but only {len(patients)} available")
    chosen = set(patients.head(n)["BDSPPatientID"])
    cand = df[df["BDSPPatientID"].isin(chosen)].copy()
    cand["_rh"] = cand.apply(
        lambda r: _h(
            f"{site}|{r['BDSPPatientID']}|{r['FileName']}",
            RECORD_NAMESPACE,
        ),
        axis=1,
    )
    picked = (
        cand.sort_values(["BDSPPatientID", "_rh", "FileName"])
        .groupby("BDSPPatientID", as_index=False)
        .first()
    )
    if len(picked) != n:
        raise AssertionError(f"{site}: one-record-per-patient selection failed")
    return picked[["site", "BDSPPatientID", "FileName"]]


def prepare(i0001: Path, i0006: Path, output_dir: Path, per_site: int = 5000) -> dict:
    parts = []
    for site, path in (("I0001", i0001), ("I0006", i0006)):
        parts.append(_select_site(_load_site(path, site), site, int(per_site)))
    cohort = pd.concat(parts, ignore_index=True)
    if cohort["BDSPPatientID"].astype(str).duplicated().any():
        # IDs are expected to be site-local. Duplicate IDs across sites are allowed
        # only after site is included in the key, so check composite uniqueness.
        if cohort[["site", "BDSPPatientID"]].duplicated().any():
            raise AssertionError("Duplicate site+patient in frozen cohort")
    if cohort["FileName"].duplicated().any():
        raise AssertionError("Duplicate FileName in frozen cohort")
    cohort = cohort.sort_values(["site", "BDSPPatientID", "FileName"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "heedb_frozen_manifest.csv"
    cohort.to_csv(manifest, index=False)

    summary = {
        "version": "MEDCALC_HEEDB_SELECTION_V1",
        "validation_id": "HEEDB_DIAGNOSTIC_V1",
        "records": int(len(cohort)),
        "patients": int(len(cohort)),
        "site_counts": cohort["site"].value_counts().sort_index().to_dict(),
        "records_per_patient": 1,
        "selection_label_blind": True,
        "diagnoses_acquisition_opened": False,
        "diagnoses_dictionary_opened": False,
        "software_12sl_labels_used": False,
        "physician_overreads_used": False,
        "patient_namespace": PATIENT_NAMESPACE,
        "record_namespace": RECORD_NAMESPACE,
    }
    (output_dir / "heedb_selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--i0001-metadata", type=Path, required=True)
    ap.add_argument("--i0006-metadata", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--per-site", type=int, default=5000)
    args = ap.parse_args()
    print(json.dumps(
        prepare(args.i0001_metadata, args.i0006_metadata, args.output_dir, args.per_site),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
