#!/usr/bin/env python3
"""Generate Kaggle Notebook (kernel) versions of the six DDACS tutorials.

The repo notebooks in notebooks/ are the single source of truth; the only
adaptation for Kaggle is a setup cell that `pip install`s ddacs and fetches the
22 MB sample with `ddacs download --small` (Kaggle auto-extracts uploaded zips,
so reading the attached dataset would need reshaping — downloading is simpler
and gives the exact layout ddacs expects). The dataset is still attached so the
kernels appear on the dataset's Code tab.

Output: publish/.staging/kernels/<slug>/{notebook, kernel-metadata.json}.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_DIR = REPO / "notebooks"
OUT = REPO / "publish" / ".staging" / "kernels"

OWNER = "baumsebastian"
DATASET = f"{OWNER}/ddacs-teaser"
DATA_DIR = "/kaggle/working/data"
LOOSE_DIR = "/kaggle/working/ddacs_loose"


def _slug(title: str) -> str:
    # Kaggle derives the kernel slug from the title, and the id slug MUST equal
    # it ("your kernel title does not resolve to a specific id" otherwise).
    s = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return re.sub(r"-+", "-", s).strip("-")


# file, title, pip target, kind
NOTEBOOKS = [
    ("01_getting_started.ipynb", "DDACS 01 Getting Started", "ddacs", "std"),
    ("02_views.ipynb", "DDACS 02 Build Your Own View", "ddacs", "std"),
    ("03_pytorch.ipynb", "DDACS 03 PyTorch Training", "ddacs[torch]", "std"),
    ("04_visualization.ipynb", "DDACS 04 Visualization", "ddacs", "std"),
    ("05_loose_h5.ipynb", "DDACS 05 Loose HDF5", "ddacs", "loose"),
    ("06_streaming.ipynb", "DDACS 06 Streaming and Numpy Export", "ddacs", "std"),
]


_CTA = """\
---

## ⬇️ Get the full dataset — 600+ GB

This notebook ran on a **22 MB sample** (a single simulation). The complete
**DDACS** dataset — **32,466 simulations, 600+ GB of lossless HDF5**, with the
predefined train / validation / test split — is hosted on DaRUS with a citable DOI:

### ➡️ [doi.org/10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801)

Everything above scales to the full release unchanged — just fetch it with the package:

```bash
pip install ddacs
ddacs download        # full 600+ GB release
```

Docs: https://ddacs.readthedocs.io · Package: https://pypi.org/project/ddacs · Source: https://github.com/BaumSebastian/DDACS
"""


def _code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def adapt(nb: dict, pip_target: str, kind: str) -> dict:
    """Kernel = two quiet setup cells (pip install, small-data download) on top
    of the unmodified repo notebook (paths mapped to /kaggle/working)."""
    for c in nb["cells"]:
        c["source"] = [
            ln.replace("Path('../data')", f"Path('{DATA_DIR}')")
            .replace("Path('./data')", f"Path('{DATA_DIR}')")
            .replace("./data_loose", LOOSE_DIR)
            .replace("/tmp/ddacs_loose", LOOSE_DIR)
            for ln in c["source"]
            # the repo-root/notebook alternative comment is meaningless on Kaggle
            if not ln.lstrip().startswith("# DATA_DIR = ")
        ]

    if kind == "loose":
        fetch = (
            "from pathlib import Path\n"
            "\n"
            f"LOOSE_DIR = Path('{LOOSE_DIR}')\n"
            "\n"
            "# Fetch the 22 MB sample once, extracted to loose .h5 files\n"
            "if not (LOOSE_DIR / 'metadata.json').exists():\n"
            "    !ddacs download --small --extract --remove-zip -y --quiet --out {LOOSE_DIR}"
        )
    else:
        fetch = (
            "from pathlib import Path\n"
            "\n"
            f"DATA_DIR = Path('{DATA_DIR}')\n"
            "\n"
            "# Fetch the 22 MB sample once: ddacs download --small\n"
            "if not (DATA_DIR / 'metadata.json').exists():\n"
            "    !ddacs download --small -y --quiet --out {DATA_DIR}"
        )

    setup = [
        _md(
            "## Kaggle setup\n\nInstall `ddacs` and fetch the 22 MB sample — both quiet. "
            "Auto-generated from the repo notebook — "
            "https://github.com/BaumSebastian/DDACS ."
        ),
        _code(f"!pip install -q {pip_target}"),
        _code(fetch),
    ]
    nb["cells"] = setup + nb["cells"] + [_md(_CTA)]
    return nb


def kernel_metadata(slug: str, title: str, code_file: str) -> dict:
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": False,
        "enable_internet": True,  # needed for pip install + ddacs download
        "dataset_sources": [DATASET],  # links the kernel to the dataset's Code tab
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)  # drop stale slugs from earlier builds
    OUT.mkdir(parents=True, exist_ok=True)
    for fname, title, pip_target, kind in NOTEBOOKS:
        slug = _slug(title)
        nb = adapt(json.loads((NB_DIR / fname).read_text()), pip_target, kind)
        dest = OUT / slug
        dest.mkdir(parents=True, exist_ok=True)
        (dest / fname).write_text(json.dumps(nb, indent=1))
        (dest / "kernel-metadata.json").write_text(
            json.dumps(kernel_metadata(slug, title, fname), indent=2)
        )
        print(f"built {slug}  ->  {dest}")


if __name__ == "__main__":
    main()
