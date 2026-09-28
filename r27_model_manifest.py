from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import joblib


MANIFEST_VERSION = "MEDCALC_R27_MODEL_MANIFEST_V1"


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _estimator_tree(obj: Any, depth: int = 0) -> Dict[str, Any]:
    if depth > 5:
        return {"class": type(obj).__name__, "module": type(obj).__module__, "truncated": True}

    row: Dict[str, Any] = {
        "class": type(obj).__name__,
        "module": type(obj).__module__,
    }

    if hasattr(obj, "n_features_in_"):
        try:
            row["n_features_in"] = int(obj.n_features_in_)
        except Exception:
            pass

    if hasattr(obj, "classes_"):
        try:
            row["classes"] = [str(x) for x in list(obj.classes_)]
        except Exception:
            pass

    if hasattr(obj, "n_estimators"):
        try:
            row["n_estimators"] = int(obj.n_estimators)
        except Exception:
            pass

    for attr in ("max_depth", "criterion", "max_features", "class_weight"):
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                row[attr] = value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
            except Exception:
                pass

    if hasattr(obj, "steps"):
        try:
            row["steps"] = [
                {"name": str(name), "estimator": _estimator_tree(est, depth + 1)}
                for name, est in list(obj.steps)
            ]
        except Exception:
            pass

    if hasattr(obj, "estimators") and not hasattr(obj, "steps"):
        try:
            estimators = getattr(obj, "estimators")
            if isinstance(estimators, list):
                row["named_estimators"] = [
                    {
                        "name": str(name),
                        "estimator": _estimator_tree(est, depth + 1),
                    }
                    for name, est in estimators
                    if isinstance(name, str)
                ]
        except Exception:
            pass

    return row


def _module_guess(path: Path) -> str | None:
    stem = path.stem.upper()
    suffixes = (
        "_EXTRATREES",
        "_RANDOMFOREST",
        "_RANDOM_FOREST",
        "_LOGREG",
        "_RF",
        "_ET",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem or None


def inspect_r27_models(runtime_root: Path) -> Dict[str, Any]:
    runtime_root = runtime_root.resolve()
    roots = [
        runtime_root / "compat" / "MEDCALC",
        runtime_root,
    ]
    search_root = next((p for p in roots if p.is_dir()), runtime_root)

    rows = []
    for path in sorted(search_root.rglob("*.joblib")):
        try:
            obj = joblib.load(path)
            tree = _estimator_tree(obj)
            load_error = None
        except Exception as exc:
            tree = None
            load_error = f"{type(exc).__name__}:{exc}"

        rows.append({
            "relative_path": str(path.relative_to(runtime_root)),
            "filename": path.name,
            "module_guess": _module_guess(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "load_pass": load_error is None,
            "load_error": load_error,
            "estimator": tree,
        })

    class_counts: Dict[str, int] = {}
    for row in rows:
        est = row.get("estimator") or {}
        cls = str(est.get("class") or "UNLOADED")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    return {
        "version": MANIFEST_VERSION,
        "runtime_root": str(runtime_root),
        "joblib_n": len(rows),
        "top_level_class_counts": dict(sorted(class_counts.items())),
        "models": rows,
        "training_performed": False,
        "fit_called": False,
        "threshold_tuning_performed": False,
        "policy": "INTROSPECTION_ONLY_NO_MODEL_MUTATION",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runtime_root", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    result = inspect_r27_models(args.runtime_root)
    payload = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
