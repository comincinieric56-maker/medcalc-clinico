#!/usr/bin/env python3
"""Download/verify the official openFDA drug-label bulk corpus.

Explicit operation: this is intentionally not run on every pull request.
Manifest: https://api.fda.gov/download.json
"""
from __future__ import annotations
import argparse, json, urllib.request
from pathlib import Path
from urllib.parse import urlparse

MANIFEST="https://api.fda.gov/download.json"
UA={"User-Agent":"MEDCALC-clinico pregnancy bulk audit/2.0"}

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:
        return json.load(r)

def label_manifest():
    data=get_json(MANIFEST)
    node=data["results"]["drug"]["label"]
    parts=node["partitions"]
    if not parts:
        raise RuntimeError("openFDA drug/label manifest has no partitions")
    return data["meta"].get("last_updated"),node.get("export_date"),node.get("total_records"),parts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir",default=".cache/openfda-drug-label")
    ap.add_argument("--download",action="store_true")
    args=ap.parse_args()
    dest=Path(args.dir); dest.mkdir(parents=True,exist_ok=True)
    updated,export_date,total,parts=label_manifest()
    if len(parts) < 1 or not total:
        raise RuntimeError("invalid openFDA drug/label manifest")
    manifest={"manifest_last_updated":updated,"export_date":export_date,"total_records":total,
              "partition_count":len(parts),"partitions":parts}
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"openFDA drug/label: {len(parts)} partitions, {total} records, export={export_date}, manifest={updated}")
    missing=[]
    for i,p in enumerate(parts,1):
        url=p["file"]; name=Path(urlparse(url).path).name; target=dest/name
        if target.exists() and target.stat().st_size>0:
            print(f"[{i}/{len(parts)}] cached {name}")
            continue
        if not args.download:
            missing.append(name); continue
        print(f"[{i}/{len(parts)}] downloading {name}",flush=True)
        req=urllib.request.Request(url,headers=UA)
        with urllib.request.urlopen(req,timeout=300) as r, target.open("wb") as f:
            while True:
                chunk=r.read(1024*1024)
                if not chunk: break
                f.write(chunk)
    if missing:
        raise SystemExit("Missing partitions (rerun with --download): "+", ".join(missing))
    print("openFDA bulk corpus complete")

if __name__=="__main__": main()
