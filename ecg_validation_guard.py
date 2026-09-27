from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_PROVENANCE = Path(__file__).with_name("ecg_dataset_provenance.json")


class ProvenanceError(RuntimeError):
    pass


def load_registry(path: Path = DEFAULT_PROVENANCE) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _index(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = []
    rows.extend(registry.get("development_contaminated") or [])
    rows.extend(registry.get("provisional_external_locked") or [])
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dataset_id = str(row.get("id") or "").strip()
        if not dataset_id:
            raise ProvenanceError("Dataset without id in provenance registry.")
        if dataset_id in out:
            raise ProvenanceError(f"Duplicate dataset id: {dataset_id}")
        out[dataset_id] = row
    return out


def validate_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    idx = _index(registry)
    errors: list[str] = []

    contaminated = {
        str(row.get("id"))
        for row in registry.get("development_contaminated") or []
    }
    external = {
        str(row.get("id"))
        for row in registry.get("provisional_external_locked") or []
    }

    overlap = sorted(contaminated & external)
    if overlap:
        errors.append("Dataset ids present in both development and external: " + ", ".join(overlap))

    for dataset_id in contaminated:
        row = idx[dataset_id]
        if bool(row.get("external_validation_allowed")):
            errors.append(f"{dataset_id}: contaminated dataset cannot allow external validation")

    for dataset_id in external:
        row = idx[dataset_id]
        if not bool(row.get("frozen")):
            errors.append(f"{dataset_id}: external dataset must be frozen")
        if bool(row.get("allow_tuning")):
            errors.append(f"{dataset_id}: external dataset cannot allow tuning")
        if bool(row.get("allow_threshold_selection")):
            errors.append(f"{dataset_id}: external dataset cannot allow threshold selection")
        if not bool(row.get("external_validation_allowed")):
            errors.append(f"{dataset_id}: external dataset must allow external validation")

    for row in registry.get("development_contaminated") or []:
        for child in row.get("contains_or_overlaps") or []:
            if child not in idx:
                errors.append(f"{row.get('id')}: unknown overlap dataset {child}")

    if errors:
        raise ProvenanceError("\n".join(errors))

    return {
        "version": registry.get("version"),
        "development_contaminated_n": len(contaminated),
        "external_locked_n": len(external),
        "development_ids": sorted(contaminated),
        "external_ids": sorted(external),
        "status": "PASS",
    }


def assert_external_dataset(dataset_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    validate_registry(registry)
    idx = _index(registry)
    if dataset_id not in idx:
        raise ProvenanceError(
            f"Unknown dataset '{dataset_id}'. Add it to ecg_dataset_provenance.json before use."
        )
    row = idx[dataset_id]
    if row.get("status") != "PROVISIONAL_EXTERNAL_LOCKED":
        raise ProvenanceError(
            f"{dataset_id} is not eligible for external validation: status={row.get('status')}"
        )
    if not bool(row.get("external_validation_allowed")):
        raise ProvenanceError(f"{dataset_id} external validation is disabled.")
    if bool(row.get("allow_tuning")) or bool(row.get("allow_threshold_selection")):
        raise ProvenanceError(f"{dataset_id} violates frozen external-test policy.")
    return row


def assert_development_dataset(dataset_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    validate_registry(registry)
    idx = _index(registry)
    if dataset_id not in idx:
        raise ProvenanceError(f"Unknown dataset '{dataset_id}'.")
    row = idx[dataset_id]
    if row.get("status") != "DEVELOPMENT_CONTAMINATED":
        raise ProvenanceError(
            f"{dataset_id} is not registered as development-contaminated."
        )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", type=Path, default=DEFAULT_PROVENANCE)
    ap.add_argument("--external", default=None)
    ap.add_argument("--development", default=None)
    args = ap.parse_args()

    registry = load_registry(args.registry)
    summary = validate_registry(registry)
    if args.external:
        row = assert_external_dataset(args.external, registry)
        summary["selected_external"] = row["id"]
    if args.development:
        row = assert_development_dataset(args.development, registry)
        summary["selected_development"] = row["id"]

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
