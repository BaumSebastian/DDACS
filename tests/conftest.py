"""Test fixtures for DDACS.

Three tiers of data:

* ``synthetic_data_dir`` — hand-built ~few-KB Croissant dataset constructed
  programmatically. Offline, isolated, used for unit tests of the new public
  surface (`load`, `add_view`, `open_h5`, `inspect_h5`, `DDACSDataset`).
* ``small_data_dir`` — `ddacs download --small` against DaRUS, run once per
  pytest session. Picks up ``DARUS_API_TOKEN`` from ``.env`` if present so
  tests work against the current draft; falls back to the public version
  otherwise. Skipped when the download fails.
* ``rddac_data_dir`` — a local path expected to contain ``h5/rddac.zip``
  (the unpacked corner block). Defaults to ``/tmp/ddacs_real`` and can be
  overridden via ``DDACS_TEST_DATA_DIR``. Skipped when the file is missing.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Load .env so DARUS_API_TOKEN is picked up automatically.
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text().splitlines():
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Tier A — synthetic dataset
# ---------------------------------------------------------------------------

_SYNTHETIC_SIM_IDS = [1, 2, 3]


def _make_synthetic_h5(sim_id: int) -> bytes:
    """Build a tiny in-memory h5 with a couple of OP10/blank fields."""
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        rng = np.random.default_rng(sim_id)
        f.attrs["index"] = sim_id
        g = f.create_group("OP10/blank")
        g.create_dataset("node_displacement", data=rng.random((4, 5, 3)))
        g.create_dataset("element_shell_thickness", data=rng.random((4, 7)))
    return buf.getvalue()


def _make_synthetic_manifest(sim_ids: list[int]) -> dict:
    """A minimal Croissant 1.1 manifest exercising the parts we care about."""
    csv_id = "process_parameters_csv"
    zip_ids = [f"{sid}.zip" for sid in sim_ids]
    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "containedIn": "cr:containedIn",
            "cr": "http://mlcommons.org/croissant/",
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "extract": "cr:extract",
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "includes": "cr:includes",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "recordSet": "cr:recordSet",
            "regex": "cr:regex",
            "source": "cr:source",
            "transform": "cr:transform",
            "sc": "https://schema.org/",
        },
        "@type": "sc:Dataset",
        "name": "synthetic",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "description": "synthetic ddacs test dataset",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": csv_id,
                "name": "process_parameters.csv",
                "contentUrl": "process_parameters.csv",
                "encodingFormat": "text/csv",
                "md5": "0" * 32,
            },
        ]
        + [
            {
                "@type": "cr:FileObject",
                "@id": zid,
                "name": zid,
                "contentUrl": zid,
                "encodingFormat": "application/zip",
                "md5": "0" * 32,
            }
            for zid in zip_ids
        ]
        + [
            {
                "@type": "cr:FileSet",
                "@id": "all_h5_files",
                "name": "all_h5_files",
                "containedIn": [{"@id": zid} for zid in zip_ids],
                "encodingFormat": "application/x-hdf5",
                "includes": ["**/*.h5"],
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "process-parameters",
                "name": "process-parameters",
                "key": {"@id": "process-parameters/index"},
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "process-parameters/index",
                        "name": "index",
                        "dataType": "sc:Integer",
                        "source": {
                            "fileObject": {"@id": csv_id},
                            "extract": {"column": "index"},
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "process-parameters/geometry",
                        "name": "geometry",
                        "dataType": "sc:Text",
                        "source": {
                            "fileObject": {"@id": csv_id},
                            "extract": {"column": "geometry"},
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "field-map",
                "name": "field-map",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "field-map/op10_blank_node_displacement",
                        "name": "op10_blank_node_displacement",
                        "dataType": "sc:Float",
                        "source": {
                            "fileSet": {"@id": "all_h5_files"},
                            "extract": {"fileProperty": "content"},
                            "transform": [{"regex": "OP10/blank/node_displacement"}],
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "field-map/op10_blank_element_shell_thickness",
                        "name": "op10_blank_element_shell_thickness",
                        "dataType": "sc:Float",
                        "source": {
                            "fileSet": {"@id": "all_h5_files"},
                            "extract": {"fileProperty": "content"},
                            "transform": [{"regex": "OP10/blank/element_shell_thickness"}],
                        },
                    },
                ],
            },
            {
                "@type": "cr:RecordSet",
                "@id": "springback-minimal",
                "name": "springback-minimal",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "springback-minimal/forming",
                        "name": "forming",
                        "dataType": "sc:Float",
                        "source": {
                            "field": {"@id": "field-map/op10_blank_node_displacement"},
                            "transform": [{"jsonPath": "$[2]"}],
                        },
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "springback-minimal/springback",
                        "name": "springback",
                        "dataType": "sc:Float",
                        "source": {
                            "field": {"@id": "field-map/op10_blank_node_displacement"},
                            "transform": [{"jsonPath": "$[3]"}],
                        },
                    },
                ],
            },
        ],
    }


@pytest.fixture(scope="session")
def synthetic_data_dir(tmp_path_factory) -> Path:
    """Self-contained dataset: one CSV + one zip per sim id + manifest."""
    out = tmp_path_factory.mktemp("ddacs_synth")
    (out / "h5").mkdir()

    # process_parameters.csv
    rows = ["index,geometry"] + [f"{sid},rectangular" for sid in _SYNTHETIC_SIM_IDS]
    (out / "process_parameters.csv").write_text("\n".join(rows) + "\n")

    # One zip per sim containing <sim>.h5
    for sid in _SYNTHETIC_SIM_IDS:
        zip_path = out / "h5" / f"{sid}.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(f"{sid}.h5", _make_synthetic_h5(sid))

    # metadata.json — Croissant manifest
    (out / "metadata.json").write_text(json.dumps(_make_synthetic_manifest(_SYNTHETIC_SIM_IDS)))

    return out


# ---------------------------------------------------------------------------
# Tier B — `ddacs download --small`
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def small_data_dir(tmp_path_factory) -> Path:
    """Fresh `ddacs download --small` into a session-scoped tmp dir."""
    out = tmp_path_factory.mktemp("ddacs_small")
    cmd = ["ddacs"]
    token = os.environ.get("DARUS_API_TOKEN")
    if token:
        cmd += ["--token", token]
    cmd += ["download", "--small", "-y", "--out", str(out)]
    if token:
        cmd.append(":draft")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        pytest.skip("ddacs download --small timed out")
    if res.returncode != 0:
        pytest.skip(f"ddacs download --small failed: {res.stderr[-200:]}")
    return out


# ---------------------------------------------------------------------------
# Tier C — local rddac fixture
# ---------------------------------------------------------------------------

_RDDAC_DATA_DIR = Path(os.environ.get("DDACS_TEST_DATA_DIR", "/tmp/ddacs_real"))


@pytest.fixture(scope="session")
def rddac_data_dir() -> Path:
    """Local data dir expected to contain ``h5/rddac.zip``."""
    rddac = _RDDAC_DATA_DIR / "h5" / "rddac.zip"
    if not rddac.is_file():
        pytest.skip(
            f"rddac.zip not at {rddac}; set DDACS_TEST_DATA_DIR or unpack "
            f"rddac.zip.zip into {_RDDAC_DATA_DIR}/h5/"
        )
    return _RDDAC_DATA_DIR
