from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ARTICLE_ID = 27078763
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"


def fetch_inventory() -> dict:
    req = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "MEDCALC-ECG-ZZU-external-validation/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        article = json.load(response)

    files = []
    for item in article.get("files") or []:
        files.append({
            "id": item.get("id"),
            "name": item.get("name"),
            "size": item.get("size"),
            "is_link_only": bool(item.get("is_link_only", False)),
        })

    return {
        "inventory_version": "MEDCALC_ZZU_PECG_FIGSHARE_INVENTORY_V1",
        "article_id": ARTICLE_ID,
        "title": article.get("title"),
        "license_name": (article.get("license") or {}).get("name"),
        "license_url": (article.get("license") or {}).get("url"),
        "file_count": len(files),
        "total_size_bytes": int(sum(int(x.get("size") or 0) for x in files)),
        "files": files,
        "clinical_files_opened": False,
        "attribute_dictionary_opened": False,
        "diagnosis_dictionary_opened": False,
        "signal_files_downloaded": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = fetch_inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
