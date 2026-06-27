"""Streaming iteration over a Croissant view, independent of PyTorch.

`iter_view` is the plain-Python counterpart to `DDACSDataset.__iter__`. It
yields one `dict[str, numpy.ndarray]` per simulation, exactly the same shape
the PyTorch adapter does, but with no torch dependency and no DataLoader
sharding. Use it when:

* you need an offline iterator (notebook, script, data conversion),
* you want to avoid the ``mlcroissant.Dataset.records(view)`` setup walk that
  opens every zip's central directory before yielding the first record, or
* your data already lives as **loose `.h5` files** (after
  ``ddacs download --extract --remove-zip``) and you can't use the
  ``DDACSDataset`` zip-only path.

Layout discovery is unified: the local index built at construction time
recognises both loose ``data_dir/h5/<sim_id>.h5`` files and the zipped
``data_dir/h5/*.zip`` archives, preferring loose files when both exist
(direct h5py reads are faster than going through ``BytesIO``).
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from . import croissant as _croissant
from .config import (
    DEFAULT_DATA_DIR,
    FIELD_MAP_RECORD_SET,
    ID_COLUMN,
    PROCESS_PARAMETERS_FILE,
)

_JSONPATH_RE = re.compile(r"^\$\[(.+)\]$")


def iter_view(
    view: str,
    *,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    sim_ids: list[int] | None = None,
    where: Callable[[pd.Series], bool] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield one record per simulation for a Croissant view.

    Args:
        view: Name of the RecordSet to stream. Must exist on the loaded
            ``mlcroissant.Dataset`` (either as a published RecordSet or a
            view added via :func:`ddacs.add_view`).
        source: Override the Croissant manifest URL/path. Ignored when
            ``dataset=`` is given.
        data_dir: Directory holding ``metadata.json``,
            ``process_parameters.csv`` and either loose ``h5/<sim_id>.h5``
            files or zipped ``h5/*.zip`` archives.
        dataset: A pre-loaded ``mlcroissant.Dataset`` (e.g. one mutated by
            :func:`ddacs.add_view`). When supplied, the manifest is not
            re-parsed and the caller's object is used as-is.
        sim_ids: Optional allowlist of simulation ids; iteration order is
            preserved from the CSV when ``sim_ids`` is not supplied,
            otherwise from the allowlist.
        where: Predicate applied to each ``process_parameters.csv`` row
            before any HDF5 file is touched. Combined with ``sim_ids`` if
            both are given.

    Yields:
        A ``dict[str, np.ndarray]`` per simulation, keyed by the view-field
        aliases. Sliced fields come out with the requested timesteps
        applied; whole fields come out at their full shape.
    """
    ds = dataset if dataset is not None else _croissant.load(source=source, data_dir=data_dir)
    field_specs = _build_field_specs(ds, view)
    h5_index = _build_unified_index(Path(data_dir) if data_dir is not None else None)
    final_ids = _resolve_sim_ids(
        Path(data_dir) if data_dir is not None else None,
        h5_index,
        sim_ids,
        where,
    )

    last_zip_path: str | None = None
    last_zf: zipfile.ZipFile | None = None
    try:
        for sim_id in final_ids:
            path = h5_index.get(int(sim_id))
            if path is None:
                continue

            if path.endswith(".h5"):
                # Loose file: open directly, no zip indirection.
                with h5py.File(path, "r") as f:
                    yield _extract_record(f, field_specs)
            elif path.endswith(".zip"):
                # Cache the zip handle across consecutive sims that land in
                # the same archive; corner-block zips group ~2200 sims each.
                if path != last_zip_path:
                    if last_zf is not None:
                        last_zf.close()
                    last_zf = zipfile.ZipFile(path)
                    last_zip_path = path
                try:
                    data = last_zf.read(f"{sim_id}.h5")
                except KeyError:
                    continue
                with h5py.File(io.BytesIO(data), "r") as f:
                    yield _extract_record(f, field_specs)
    finally:
        if last_zf is not None:
            last_zf.close()


# ---------------------------------------------------------------------------
# Internals (also consumed by ddacs.pytorch.DDACSDataset once refactored).
# ---------------------------------------------------------------------------


def _build_unified_index(data_dir: Path | None) -> dict[int, str]:
    """Map ``sim_id`` -> path of a loose ``.h5`` file or a ``.zip`` archive.

    Loose ``.h5`` files take precedence over zips when both contain the
    same simulation id, because direct h5py reads avoid the per-record
    BytesIO round trip. ``data_dir/h5/`` is preferred over ``data_dir``
    itself; both are scanned to support either layout.
    """
    if data_dir is None:
        return {}

    candidates = []
    h5_subdir = data_dir / "h5"
    if h5_subdir.is_dir():
        candidates.append(h5_subdir)
    candidates.append(data_dir)

    index: dict[int, str] = {}

    # Pass 1: loose .h5 files (preferred).
    for d in candidates:
        for p in d.glob("*.h5"):
            try:
                sim_id = int(p.stem)
            except ValueError:
                continue
            index.setdefault(sim_id, str(p))

    # Pass 2: zip archives, only for sim_ids we haven't already indexed loose.
    for d in candidates:
        for p in d.glob("*.zip"):
            try:
                with zipfile.ZipFile(p) as zf:
                    for name in zf.namelist():
                        if not name.endswith(".h5"):
                            continue
                        try:
                            sim_id = int(Path(name).stem)
                        except ValueError:
                            continue
                        index.setdefault(sim_id, str(p))
            except zipfile.BadZipFile:
                continue

    return index


def _build_field_specs(ds, view: str) -> dict[str, tuple[str, Any]]:
    """For each view-field, return ``(h5_path, slicing)``.

    ``slicing`` is ``None``, an ``int`` (single timestep), or a
    ``list[int]`` (multiple timesteps), parsed from the view-field's
    JSONPath transform.
    """
    view_rs = next((r for r in ds.metadata.record_sets if r.id == view), None)
    if view_rs is None:
        raise ValueError(f"view {view!r} not found in manifest")
    fm_rs = next(
        (r for r in ds.metadata.record_sets if r.id == FIELD_MAP_RECORD_SET),
        None,
    )
    if fm_rs is None:
        raise ValueError(f"{FIELD_MAP_RECORD_SET!r} RecordSet missing - manifest is malformed")
    fm = {f.name: f for f in fm_rs.fields}

    specs: dict[str, tuple[str, Any]] = {}
    for f in view_rs.fields:
        source_field_id = f.source.uuid.split("/", 1)[-1]
        if source_field_id not in fm:
            raise ValueError(f"view field {f.name!r} sources unknown field {source_field_id!r}")
        h5_path = fm[source_field_id].source.transforms[0].regex
        slicing = None
        if f.source.transforms:
            slicing = _parse_jsonpath(f.source.transforms[0].json_path)
        specs[f.name] = (h5_path, slicing)
    return specs


def _parse_jsonpath(expr: str | None) -> Any:
    """Parse ``$[N]`` -> int, ``$[a,b,c]`` -> list[int]. Anything else -> None."""
    if not expr:
        return None
    m = _JSONPATH_RE.match(expr)
    if not m:
        return None
    inner = m.group(1)
    if "," in inner:
        return [int(s) for s in inner.split(",")]
    try:
        return int(inner)
    except ValueError:
        return None


def _resolve_sim_ids(
    data_dir: Path | None,
    h5_index: dict[int, str],
    sim_ids_arg: list[int] | None,
    where: Callable[[pd.Series], bool] | None,
) -> list[int]:
    """Apply ``sim_ids`` + ``where`` to produce the final ordered list of sim ids."""
    if data_dir is not None:
        csv_path = data_dir / PROCESS_PARAMETERS_FILE
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            if ID_COLUMN not in df.columns:
                raise ValueError(f"{csv_path} missing required {ID_COLUMN!r} column")
            if sim_ids_arg is not None:
                df = df[df[ID_COLUMN].isin(set(sim_ids_arg))]
            if where is not None:
                df = df[df.apply(where, axis=1)]
            return [int(x) for x in df[ID_COLUMN].tolist()]

    if where is not None:
        raise ValueError(f"`where` filter requires {PROCESS_PARAMETERS_FILE} under data_dir")
    if sim_ids_arg is not None:
        return [int(x) for x in sim_ids_arg]
    return sorted(h5_index.keys())


def _extract_record(f: h5py.File, field_specs: dict[str, tuple[str, Any]]) -> dict[str, np.ndarray]:
    """Read the view-fields from a single h5py.File. Caches full reads so
    multiple aliases sharing one source field only read the array once.
    """
    cache: dict[str, np.ndarray] = {}
    rec: dict[str, np.ndarray] = {}
    for alias, (h5_path, slicing) in field_specs.items():
        arr = cache.get(h5_path)
        if arr is None:
            arr = f[h5_path][...]
            cache[h5_path] = arr
        rec[alias] = arr[slicing] if slicing is not None else arr
    return rec
