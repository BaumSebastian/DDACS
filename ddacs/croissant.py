"""Croissant manifest access via ``mlcroissant``.

Single source of truth for field/column descriptions, dataset name, etc. —
nothing is duplicated in the Python package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlcroissant as mlc

from .config import DARUS_BASE_URL, DATASET_DOI, DEFAULT_DATA_DIR, METADATA_FILE

# Permanent DaRUS download URL for the published metadata.json.
METADATA_URL = (
    f"{DARUS_BASE_URL}/api/access/datafile/:persistentId"
    f"?persistentId={DATASET_DOI}/{METADATA_FILE}"
)


def resolve_source(source: str | Path | None = None, data_dir: str | Path | None = None) -> str:
    """Return the source string (path or URL) that :func:`load` would use.

    Resolution order:
        1. ``source`` if given (local path or HTTP(S) URL).
        2. ``<data_dir>/metadata.json`` if it exists locally.
        3. The permanent DaRUS URL :data:`METADATA_URL`.
    """
    if source is not None:
        return str(source)
    if data_dir is not None:
        local = Path(data_dir) / METADATA_FILE
        if local.is_file():
            return str(local)
    return METADATA_URL


def _build_mapping(jsonld: str, data_dir: str | Path) -> dict[str, str] | None:
    """Map each ``FileObject`` UUID to a local file inside ``data_dir``.

    `mlcroissant` accepts a ``mapping`` dict at :class:`mlc.Dataset` construction
    time and uses the listed local paths instead of refetching via the
    manifest's ``contentUrl``. We populate it from whatever files the user has
    already downloaded under ``data_dir`` (typically via ``ddacs download``).

    Returns ``None`` if ``data_dir`` doesn't exist on disk so the caller can
    skip the ``mapping`` argument entirely and let `mlcroissant` fall back to
    its own cache.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        return None
    # Discover candidate local files once. Keep the first match per filename
    # to avoid ambiguity if a user has multiple copies under data_dir.
    candidates: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name not in candidates:
            candidates[path.name] = path

    # Walk the manifest once to find FileObject UUIDs and match by filename.
    # The manifest is parsed twice (here and inside mlc.Dataset) — cheap, and
    # avoids leaking mlcroissant internals into the caller.
    probe = mlc.Dataset(jsonld=jsonld)
    mapping: dict[str, str] = {}
    for node in probe.metadata.distribution:
        if not isinstance(node, mlc.FileObject):
            continue
        local = candidates.get(node.name)
        if local is not None:
            mapping[node.uuid] = str(local)
    return mapping or None


def load(
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
) -> mlc.Dataset:
    """Return an :class:`mlcroissant.Dataset` for the DDACS manifest.

    Local-first, URL-fallback resolution — see :func:`resolve_source`. When
    ``data_dir`` points at a directory that contains files referenced by the
    manifest (e.g. zips written by ``ddacs download``), `mlcroissant` is told
    to use those local copies instead of refetching from DaRUS.

    Pass ``data_dir=None`` to opt out of local-file discovery and force
    `mlcroissant` to download via its own cache.
    """
    jsonld = resolve_source(source, data_dir)
    mapping = _build_mapping(jsonld, data_dir) if data_dir is not None else None
    return mlc.Dataset(jsonld=jsonld, mapping=mapping)


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
