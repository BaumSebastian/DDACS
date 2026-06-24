"""Croissant manifest access via ``mlcroissant``.

Single source of truth for field/column descriptions, dataset name, etc. —
nothing is duplicated in the Python package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlcroissant as mlc

from .config import DARUS_BASE_URL, DATASET_DOI, METADATA_FILE

# Permanent DaRUS download URL for the published metadata.json.
METADATA_URL = (
    f"{DARUS_BASE_URL}/api/access/datafile/:persistentId"
    f"?persistentId={DATASET_DOI}/{METADATA_FILE}"
)


def resolve_source(source: str | Path | None = None, data_dir: str | Path | None = None) -> str:
    """Return the source string (path or URL) that ``load_dataset`` would use.

    Resolution order:
        1. ``source`` if given (local path or HTTP(S) URL).
        2. ``<data_dir>/metadata.json`` if it exists locally.
        3. The permanent DaRUS URL ``METADATA_URL``.
    """
    if source is not None:
        return str(source)
    if data_dir is not None:
        local = Path(data_dir) / METADATA_FILE
        if local.is_file():
            return str(local)
    return METADATA_URL


def load_dataset(
    source: str | Path | None = None, data_dir: str | Path | None = None
) -> mlc.Dataset:
    """Return an ``mlcroissant.Dataset`` for the DDACS manifest.

    Local-first, URL-fallback resolution — see :func:`resolve_source`.
    """
    return mlc.Dataset(jsonld=resolve_source(source, data_dir))


def _record_set(dataset: mlc.Dataset, rs_id: str):
    for rs in dataset.metadata.record_sets:
        if rs.id == rs_id:
            return rs
    return None


def process_parameters_descriptions(dataset: mlc.Dataset) -> dict[str, str]:
    """Map each process_parameters column to its human-readable description.

    Pulled directly from the ``process-parameters`` RecordSet in the manifest.
    """
    rs = _record_set(dataset, "process-parameters")
    if rs is None:
        return {}
    return {f.name: (f.description or "") for f in rs.fields}


def field_map(dataset: mlc.Dataset) -> dict[str, Any]:
    """Map each HDF5 field name in the ``field-map`` RecordSet to its mlc Field."""
    rs = _record_set(dataset, "field-map")
    if rs is None:
        return {}
    return {f.name: f for f in rs.fields}


def dataset_name(dataset: mlc.Dataset) -> str:
    return dataset.metadata.name or ""
