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


class _LoadedExport:
    """Lazy reader for a directory of ``.npy`` memmap shards.

    Returns plain ``numpy.ndarray`` rows and uses only the standard Python
    data model (``__len__`` / ``__getitem__`` / ``__iter__``). The class is
    private; users construct instances through :func:`load_export`.

    The implementation is deliberately framework-agnostic: it satisfies
    PyTorch's map-style ``Dataset`` protocol (so ``DataLoader(instance, ...)``
    works), can drive ``tf.data.Dataset.from_generator``, and is fine with
    JAX or plain Python loops -- there is no ``torch`` import inside.
    """

    def __init__(self, directory: str | Path, fields: list[str] | None = None):
        self.directory = Path(directory)
        sim_ids_path = self.directory / "sim_ids.npy"
        if not sim_ids_path.is_file():
            raise FileNotFoundError(
                f"{sim_ids_path} not found. Run `ddacs.streaming.export_to_numpy` first."
            )
        self.sim_ids = np.load(sim_ids_path)

        available = sorted(p.stem for p in self.directory.glob("*.npy") if p.stem != "sim_ids")
        if not available:
            raise FileNotFoundError(
                f"No data shards found in {self.directory} besides sim_ids.npy."
            )

        if fields is None:
            wanted = available
        else:
            wanted = list(fields)
            unknown = sorted(set(wanted) - set(available))
            if unknown:
                raise ValueError(
                    f"unknown field(s) {unknown}. Available fields in "
                    f"{self.directory}: {available}"
                )

        self._fields: dict[str, np.ndarray] = {
            alias: np.load(self.directory / f"{alias}.npy", mmap_mode="r") for alias in wanted
        }
        self._n = int(self.sim_ids.shape[0])

    @property
    def fields(self) -> tuple[str, ...]:
        """Tuple of field aliases available on each record."""
        return tuple(self._fields.keys())

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        if not 0 <= idx < self._n:
            raise IndexError(f"index {idx} out of range for export with {self._n} records")
        return {alias: arr[idx] for alias, arr in self._fields.items()}

    def __iter__(self):
        for i in range(self._n):
            yield self[i]

    def by_sim_id(self, sim_id: int) -> dict[str, np.ndarray]:
        """Return the record for a specific simulation id."""
        match = np.where(self.sim_ids == int(sim_id))[0]
        if match.size == 0:
            raise KeyError(f"sim_id {sim_id} not present in export")
        return self[int(match[0])]

    def __repr__(self) -> str:
        return (
            f"<LoadedExport directory={str(self.directory)!r} "
            f"records={self._n} fields={self.fields}>"
        )


def load_export(directory: str | Path, fields: list[str] | None = None) -> _LoadedExport:
    """Open a directory of ``.npy`` shards produced by :func:`export_to_numpy`.

    Returns a lazy reader that exposes the standard Python data model
    (``len(reader)``, ``reader[i]``, ``for r in reader``, ``reader.by_sim_id(sid)``).
    Each record is a plain ``dict[str, numpy.ndarray]`` backed by
    ``mmap_mode='r'``, so the full release fits even when it exceeds RAM --
    only the rows actually accessed page in.

    The returned object is framework-agnostic. Pass it directly into
    ``torch.utils.data.DataLoader(loader, batch_size=...)`` because it
    satisfies the map-style ``Dataset`` protocol. The same instance also
    works with ``tf.data.Dataset.from_generator(lambda: iter(loader), ...)``,
    with JAX, or with plain Python loops -- no adapter required and no
    ``torch`` import inside.

    Args:
        directory: Path produced by :func:`export_to_numpy`. Must contain a
            ``sim_ids.npy`` and at least one other ``.npy`` shard.
        fields: Optional subset of field aliases to load. ``None`` loads
            every shard in the directory. Unknown names raise ``ValueError``
            so typos surface immediately, listing the available fields.

    Returns:
        A reader instance. The concrete type is private; rely on the
        documented protocol rather than the class name.

    Raises:
        FileNotFoundError: ``directory`` is empty or missing ``sim_ids.npy``.
        ValueError: ``fields`` references a name that is not on disk.

    Example:
        >>> export = ddacs.streaming.load_export("./data/tutorial_export")
        >>> len(export), export.fields                # (1, ('delta', 'forming', ...))
        >>> export[0]                                 # {alias: ndarray}
        >>> export.by_sim_id(258864)                  # lookup by simulation id
        >>> from torch.utils.data import DataLoader
        >>> DataLoader(export, batch_size=16, shuffle=True)
    """
    return _LoadedExport(directory, fields=fields)


def export_to_numpy(
    view: str,
    out_dir: str | Path,
    *,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    sim_ids: list[int] | None = None,
    where: Callable[[pd.Series], bool] | None = None,
    transforms: dict[str, Callable[[Any], Any]] | None = None,
    record_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    show_progress: bool = False,
) -> dict[str, Path]:
    """Materialize a Croissant view as flat ``.npy`` files on disk.

    Iterates the view once (via :func:`iter_view`), applies optional
    per-field transforms, and writes each output alias into its own
    pre-allocated ``numpy.memmap`` of shape ``(n_sims, *field_shape)``.
    Also writes ``sim_ids.npy`` so the row order is recoverable.

    After this runs once, the per-record cost drops from "open zip + h5py
    + decompress" to a single mmap'd numpy slice, which is the lever the
    Croissant + HDF5 layer cannot match on its own.

    Args:
        view: Name of the RecordSet to export.
        out_dir: Output directory. Created if missing.
        source: Override the manifest URL/path. Ignored when ``dataset=`` is
            given.
        data_dir: Local data directory used to discover the simulations.
        dataset: Pre-loaded ``mlcroissant.Dataset`` (carries any ``add_view``
            mutations).
        sim_ids: Optional allowlist of simulation ids.
        where: Optional predicate on ``process_parameters.csv`` rows.
        transforms: Per-field encoder callable, ``{alias: fn(raw_value)}``.
            Applied before the record is written. Use for non-numeric columns
            (e.g. ``"geometry"`` -> integer label) or for dtype downcasts.
            Identity is assumed for any alias not listed.
        record_transform: Optional full-record callable, runs after the
            per-field transforms. Can add, drop or rename fields. The
            returned dict determines which ``.npy`` files end up on disk.
        show_progress: If ``True`` and ``tqdm`` is importable, display a
            progress bar.

    Returns:
        A dict ``{alias: Path}`` mapping each output field's name to its
        ``.npy`` file. ``sim_ids`` is included under the same name.

    Raises:
        ValueError: If the view yields no records, or if a transformed
            value cannot be coerced to a numpy array.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transforms = transforms or {}

    # Build the manifest + index once so iter_view doesn't redo the work.
    ds = dataset if dataset is not None else _croissant.load(source=source, data_dir=data_dir)

    records = iter_view(
        view=view,
        source=source,
        data_dir=data_dir,
        dataset=ds,
        sim_ids=sim_ids,
        where=where,
    )

    # Pre-resolve the sim id order so we can both size the memmaps and
    # record which row of each output array corresponds to which sim.
    final_ids = _resolve_sim_ids(
        Path(data_dir) if data_dir is not None else None,
        _build_unified_index(Path(data_dir) if data_dir is not None else None),
        sim_ids,
        where,
    )
    if not final_ids:
        raise ValueError("no simulations matched; nothing to export")

    progress = _progress_iter(records, total=len(final_ids), enabled=show_progress)

    try:
        first = next(progress)
    except StopIteration as exc:
        raise ValueError("view yielded no records") from exc

    first = _apply_transforms(first, transforms, record_transform)
    if not first:
        raise ValueError("record_transform returned an empty dict; nothing to write")

    memmaps: dict[str, np.memmap] = {}
    paths: dict[str, Path] = {}
    for alias, val in first.items():
        arr = _as_array(alias, val)
        path = out_dir / f"{alias}.npy"
        mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=arr.dtype, shape=(len(final_ids), *arr.shape)
        )
        mm[0] = arr
        memmaps[alias] = mm
        paths[alias] = path

    # Stream the rest.
    for i, rec in enumerate(progress, start=1):
        rec = _apply_transforms(rec, transforms, record_transform)
        for alias, val in rec.items():
            if alias not in memmaps:
                raise ValueError(
                    f"record {i} produced new alias {alias!r} not present in record 0; "
                    "all records must share the same output keys"
                )
            memmaps[alias][i] = _as_array(alias, val)

    for mm in memmaps.values():
        mm.flush()

    sim_ids_path = out_dir / "sim_ids.npy"
    np.save(sim_ids_path, np.asarray(final_ids, dtype=np.int64))
    paths["sim_ids"] = sim_ids_path

    return paths


# ---------------------------------------------------------------------------
# Internals (also consumed by ddacs.pytorch.DDACSDataset once refactored).
# ---------------------------------------------------------------------------


def _apply_transforms(
    record: dict[str, Any],
    transforms: dict[str, Callable[[Any], Any]],
    record_transform: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Apply per-field transforms first, then the optional whole-record callback."""
    out = {
        alias: transforms[alias](val) if alias in transforms else val
        for alias, val in record.items()
    }
    if record_transform is not None:
        out = record_transform(out)
    return out


def _as_array(alias: str, value: Any) -> np.ndarray:
    """Convert a transformed value to a numpy array, with a useful error message."""
    try:
        return np.asarray(value)
    except Exception as exc:
        raise ValueError(
            f"field {alias!r} could not be converted to a numpy array "
            f"({type(value).__name__}: {value!r}). "
            "Pass a `transforms={alias: encoder}` callable to map it to a numeric type."
        ) from exc


def _progress_iter(iterable, total: int, enabled: bool):
    """Wrap with tqdm if available and enabled; identity otherwise."""
    if not enabled:
        return iter(iterable)
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iter(iterable)
    return iter(tqdm(iterable, total=total, desc="export_to_numpy", unit="sim"))


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
