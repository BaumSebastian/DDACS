#!/usr/bin/env python3
"""Publish the DDACS teaser to Zenodo.

publish/zenodo/metadata.json is the single source of truth for the record
metadata — it is a snapshot of the curated deposit (legacy deposit-API schema,
including the `custom` Software section) and is PUT verbatim on every upload.

First run creates a brand-new record; after it is published, write the printed
record id into metadata.json ("record_id") so every later run creates a NEW
VERSION of the same record (published Zenodo records are immutable; versions
share one concept DOI).

Files come from publish/.staging (run publish/stage_teaser.sh first, or let
this script do it). Zenodo has no folders, so the bundle is flattened; the
Croissant manifest is uploaded as croissant.json. The teaser README is not
bundled here: the record description and the PDF preview cover it on Zenodo.

Env:
  ZENODO_API_TOKEN   required (scopes: deposit:write + deposit:actions)
  ZENODO_URL         default https://zenodo.org

Usage:
  python publish/zenodo/upload.py                    # draft only — review on the site
  python publish/zenodo/upload.py --publish          # draft + publish
  python publish/zenodo/upload.py --publish --version 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
STAGE = PUB / ".staging"
CONFIG_FILE = HERE / "metadata.json"

BASE = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
TOKEN = os.environ.get("ZENODO_API_TOKEN") or sys.exit("Set ZENODO_API_TOKEN")
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# staged path -> filename on Zenodo (flat)
FILES = {
    "ddacs_documentation.pdf": "ddacs_documentation.pdf",
    "data/metadata.json": "croissant.json",
    "data/process_parameters.csv": "process_parameters.csv",
    "data/h5/258864.zip": "258864.zip",
}


def _check(r: requests.Response) -> dict:
    if r.status_code >= 400:
        sys.exit(f"Zenodo API error {r.status_code} on {r.request.method} {r.url}: {r.text[:500]}")
    return r.json() if r.text else {}


def new_draft(record_id: str | None) -> dict:
    """A fresh deposition, or a new-version draft of the existing record."""
    if record_id is None:
        print("creating a NEW record (no record_id in metadata.json yet)")
        return _check(requests.post(f"{BASE}/api/deposit/depositions", json={}, headers=AUTH))
    print(f"record {record_id}: creating new-version draft")
    draft_url = _check(
        requests.post(
            f"{BASE}/api/deposit/depositions/{record_id}/actions/newversion", headers=AUTH
        )
    )["links"]["latest_draft"]
    draft = _check(requests.get(draft_url, headers=AUTH))
    # a new-version draft inherits the old files — drop them for a clean re-upload
    for f in draft.get("files", []):
        _check(requests.delete(f"{draft_url}/files/{f['id']}", headers=AUTH))
    return draft


def set_default_preview(draft_id: str) -> None:
    """Pin the preview file explicitly (an RDM-API-only property; without it
    Zenodo auto-picks an arbitrary previewable file next to the file list).
    The documentation PDF renders well in Zenodo's preview box; the markdown
    README does not (Zenodo's renderer has no tables) and stays bundle-only."""
    rdm = {**AUTH, "Accept": "application/vnd.inveniordm.v1+json"}
    url = f"{BASE}/api/records/{draft_id}/draft"
    d = _check(requests.get(url, headers=rdm))
    d["files"]["default_preview"] = "ddacs_documentation.pdf"
    _check(requests.put(url, json=d, headers={**rdm, "Content-Type": "application/json"}))
    print("default preview set to ddacs_documentation.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--publish", action="store_true", help="publish after upload (else leave as draft)"
    )
    p.add_argument("--version", default=None, help="version string for the record metadata")
    args = p.parse_args()

    if not (STAGE / "data" / "h5" / "258864.zip").is_file():
        subprocess.run([str(PUB / "stage_teaser.sh")], check=True)

    config = json.loads(CONFIG_FILE.read_text())
    metadata = config["metadata"]
    if args.version:
        metadata["version"] = args.version

    draft = new_draft(config.get("record_id"))
    bucket = draft["links"]["bucket"]
    dep_url = f"{BASE}/api/deposit/depositions/{draft['id']}"

    for rel, name in FILES.items():
        src = STAGE / rel
        if not src.is_file():
            print(f"  ! {rel} not staged — skipping {name}", file=sys.stderr)
            continue
        with open(src, "rb") as fh:
            _check(requests.put(f"{bucket}/{name}", data=fh, headers=AUTH))
        print(f"  uploaded {name} ({src.stat().st_size/1e6:.1f} MB)")

    _check(requests.put(dep_url, json={"metadata": metadata}, headers=AUTH))
    print("metadata set")
    set_default_preview(draft["id"])

    if args.publish:
        rec = _check(requests.post(f"{dep_url}/actions/publish", headers=AUTH))
        print(f"PUBLISHED: {rec['links'].get('record_html', '?')}  (doi: {rec.get('doi', '?')})")
        print(f"record id: {rec['id']}")
    else:
        print(f"draft ready for review: {BASE}/uploads/{draft['id']}")
        print("publish on the site, or re-run with --publish")

    if config.get("record_id") is None:
        print(
            '\nNOTE: after the first publish, set "record_id" in '
            "publish/zenodo/metadata.json so future runs create new versions."
        )


if __name__ == "__main__":
    main()
