"""Croissant manifest access via ``mlcroissant``.

Single source of truth for field/column descriptions, dataset name, etc. —
nothing is duplicated in the Python package.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

import mlcroissant as mlc

from .config import (
    DARUS_BASE_URL,
    DATASET_DOI,
    DEFAULT_DATA_DIR,
    DEFAULT_FIELD_DATA_TYPE,
    FIELD_MAP_RECORD_SET,
    METADATA_FILE,
)

# A field-spec value in `add_view(fields=...)`. See `add_view` docstring.
FieldSpec = str | tuple
TimestepSpec = None | int | list

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
    """Map each HDF5 field name in the field-map RecordSet to its mlc Field."""
    rs = _record_set(dataset, FIELD_MAP_RECORD_SET)
    if rs is None:
        return {}
    return {f.name: f for f in rs.fields}


def dataset_name(dataset: mlc.Dataset) -> str:
    return dataset.metadata.name or ""


# ---------------------------------------------------------------------------
# add_view — extend a loaded dataset with a custom RecordSet
# ---------------------------------------------------------------------------


def _load_jsonld_dict(source: str) -> dict[str, Any]:
    """Return the manifest's JSON-LD as a plain dict.

    Used by :func:`add_view`. We don't go through ``ds.metadata.to_json()``
    because that round-trip drops the original ``value`` payload on some
    fields and the rebuilt dataset fails validation.
    """
    if str(source).startswith(("http://", "https://")):
        with urllib.request.urlopen(str(source)) as r:
            return json.load(r)
    with open(source) as f:
        return json.load(f)


def _slicing_to_jsonpath(slicing: TimestepSpec) -> str | None:
    """Convert a timestep spec to a JSONPath transform expression.

    ``None`` -> ``None`` (no transform, whole field is read).
    ``int``  -> ``$[N]``.
    ``list`` -> ``$[a,b,c,...]``.
    """
    if slicing is None:
        return None
    if isinstance(slicing, bool):
        # bool is a subclass of int — reject explicitly so True/False don't
        # silently become "$[1]" / "$[0]".
        raise TypeError("timestep slicing must be None, int, or list[int]")
    if isinstance(slicing, int):
        return f"$[{slicing}]"
    if isinstance(slicing, list):
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in slicing):
            raise TypeError("timestep slicing list must contain only ints")
        return "$[" + ",".join(str(i) for i in slicing) + "]"
    raise TypeError(f"unsupported timestep slicing: {slicing!r}")


def _normalize_field_spec(spec: FieldSpec) -> tuple[str, TimestepSpec]:
    """Return ``(field_id, slicing)`` from one entry of the ``fields`` dict.

    Bare string -> whole field.
    ``(field_id, None | int | list[int])`` -> explicit slicing.
    """
    if isinstance(spec, str):
        return spec, None
    if isinstance(spec, tuple) and len(spec) == 2:
        field_id, slicing = spec
        if not isinstance(field_id, str):
            raise TypeError(f"field_id must be a string, got {field_id!r}")
        return field_id, slicing
    raise TypeError(f"fields value must be a string or (field_id, slicing) tuple, got {spec!r}")


def _resolve_field_id(field_id: str) -> str:
    """Return a fully qualified ``<record_set>/<field>`` id.

    Bare ids ("op10_blank_node_displacement") are assumed to live under
    ``field-map`` for backward compatibility. Already-qualified ids
    ("process-parameters/sheet_metal_thickness") are returned unchanged.
    """
    if "/" in field_id:
        return field_id
    return f"{FIELD_MAP_RECORD_SET}/{field_id}"


def _lookup_data_type(jsonld: dict[str, Any], qualified_id: str) -> str:
    """Return the source field's `dataType`, searching across all RecordSets."""
    rs_id, _, field_name = qualified_id.partition("/")
    for rs in jsonld.get("recordSet", []):
        if rs.get("@id") != rs_id:
            continue
        for f in rs.get("field", []):
            if f.get("name") == field_name:
                dt = f.get("dataType")
                if isinstance(dt, list):
                    return dt[0]
                return dt or DEFAULT_FIELD_DATA_TYPE
    return DEFAULT_FIELD_DATA_TYPE


def _build_record_set(
    jsonld: dict[str, Any], name: str, fields: dict[str, FieldSpec]
) -> dict[str, Any]:
    """Build the JSON-LD fragment for a new RecordSet (mixed sources allowed)."""
    field_entries: list[dict[str, Any]] = []
    field_map_prefix = f"{FIELD_MAP_RECORD_SET}/"
    has_field_map = False
    has_process_params = False
    first_field_map_entry: dict[str, Any] | None = None

    for alias, spec in fields.items():
        field_id, slicing = _normalize_field_spec(spec)
        qualified = _resolve_field_id(field_id)
        is_field_map = qualified.startswith(field_map_prefix)
        is_process_params = qualified.startswith("process-parameters/")
        if slicing is not None and not is_field_map:
            raise ValueError(
                f"timestep slicing is only supported on {FIELD_MAP_RECORD_SET!r} "
                f"sources; field {alias!r} -> {field_id!r}"
            )
        source: dict[str, Any] = {"field": {"@id": qualified}}
        jsonpath = _slicing_to_jsonpath(slicing)
        if jsonpath is not None:
            source["transform"] = [{"jsonPath": jsonpath}]
        entry = {
            "@type": "cr:Field",
            "@id": f"{name}/{alias}",
            "name": alias,
            "dataType": _lookup_data_type(jsonld, qualified),
            "source": source,
        }
        field_entries.append(entry)
        if is_field_map:
            has_field_map = True
            if first_field_map_entry is None:
                first_field_map_entry = entry
        if is_process_params:
            has_process_params = True

    # mlcroissant requires an explicit cross-source join when a RecordSet pulls
    # from more than one source. Declare it once on the first field-map field;
    # streaming.iter_view ignores it and joins via the sim_id internally.
    if has_field_map and has_process_params and first_field_map_entry is not None:
        first_field_map_entry["references"] = {"field": {"@id": "process-parameters/index"}}

    return {
        "@type": "cr:RecordSet",
        "@id": name,
        "name": name,
        "field": field_entries,
    }


def add_view(
    ds: mlc.Dataset,
    name: str,
    fields: dict[str, FieldSpec],
) -> mlc.Dataset:
    """Add a custom RecordSet to a loaded dataset and rewire it in place.

    Each ``fields`` entry references a manifest field by name. Bare ids
    resolve to ``field-map`` (the h5 source); qualified ids of the form
    ``"<record-set>/<field>"`` can pull from any other RecordSet, e.g.
    ``"process-parameters/sheet_metal_thickness"``:

    - ``"name": "field_id"``                            — whole field-map field
    - ``"name": ("field_id", None | int | list[int])``   — explicit, with optional timestep slicing
    - ``"name": "process-parameters/<column>"``          — CSV column from the index

    Timestep slicing is only valid on field-map sources. The published
    manifest on DaRUS is untouched — only the in-memory representation
    grows. ``ds`` is mutated in place and returned for optional chaining.
    """
    jsonld = _load_jsonld_dict(str(ds.jsonld))
    jsonld.setdefault("recordSet", []).append(_build_record_set(jsonld, name, fields))
    rebuilt = mlc.Dataset(jsonld=jsonld, mapping=ds.mapping)
    ds.metadata = rebuilt.metadata
    ds.operations = rebuilt.operations
    return ds
