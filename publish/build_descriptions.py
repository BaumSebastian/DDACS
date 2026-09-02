#!/usr/bin/env python3
"""Fill the Kaggle and Zenodo description fields from publish/teaser/README.md.

Single source: the two marked blocks in the teaser README (intro, sample) plus
the platform tails below. Run by stage_teaser.sh; the JSONs are regenerated in
place, so editing a description means editing the README (or a tail here).
"""

import json
import re
from pathlib import Path

PUB = Path(__file__).resolve().parent
README = (PUB / "teaser/README.md").read_text()

TAIL_MD = (
    "The full release (32,466 simulations, ~640 GB of lossless HDF5, predefined "
    "25,973/3,246/3,247 train/val/test split) is hosted on DaRUS with a citable DOI: "
    "[doi:10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801)."
)
LINKS_MD = (
    "Quickstart: `pip install ddacs` - Docs: [ddacs.readthedocs.io](https://ddacs.readthedocs.io) - "
    "Source: [github.com/BaumSebastian/DDACS](https://github.com/BaumSebastian/DDACS)"
)
LICENSE_MD = "License: data CC BY 4.0, package code MIT. Please cite the dataset DOI and the paper (doi:10.1051/matecconf/202540801090)."
ZENODO_NOTE_MD = (
    "*Layout note: to use this bundle with the `ddacs` package, place `258864.zip` under "
    "`data/h5/` next to `process_parameters.csv` and the manifest. This is the layout that "
    "`ddacs download --small` creates. Zenodo stores files flat, so the folder cannot be preserved here.*"
)


def block(name: str) -> str:
    m = re.search(rf"<!-- desc:{name}:start -->\n(.*?)\n<!-- desc:{name}:end -->", README, re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip()


def to_text(md: str) -> str:
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", md)
    return md.replace("**", "").replace("`", "").replace("*", "")


def to_html(md: str) -> str:
    md = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', md)
    md = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", md)
    md = re.sub(r"`([^`]+)`", r"<code>\1</code>", md)
    md = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", md)
    return md


paras = [block("sample"), block("intro"), TAIL_MD, LINKS_MD, LICENSE_MD]

kaggle = PUB / "kaggle/dataset-metadata.json"
m = json.loads(kaggle.read_text())
m["description"] = "\n\n".join(to_text(p) for p in paras)
kaggle.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")

zen = PUB / "zenodo/metadata.json"
z = json.loads(zen.read_text())
z["metadata"]["description"] = "".join(f"<p>{to_html(p)}</p>" for p in paras + [ZENODO_NOTE_MD])
zen.write_text(json.dumps(z, indent=2, ensure_ascii=False) + "\n")
print("descriptions rebuilt from teaser/README.md")
