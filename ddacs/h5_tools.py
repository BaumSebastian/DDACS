"""HDF5 access helpers — open by simulation id, inspect the file hierarchy.

`open_h5` resolves the Croissant manifest, locates the zip member matching
the requested simulation, reads it into a `BytesIO` and returns an
`h5py.File`. `inspect_h5` prints the group/dataset hierarchy of any
`h5py.File` or path.

Both are re-exported as `ddacs.open_h5` and `ddacs.inspect_h5`. There is no
`ddacs.h5.*` namespace — the trailing `_h5` self-documents the role.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h5py

from . import croissant as _croissant
from .spec import DDACS_SPEC, DatasetSpec

DEFAULT_DATA_DIR = DDACS_SPEC.default_data_dir


def open_h5(
    sim_id: int,
    source: str | Path | None = None,
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    dataset=None,
    spec: DatasetSpec = DDACS_SPEC,
) -> h5py.File:
    """Return an `h5py.File` for the requested simulation.

    Looks the manifest up, walks the locally mapped zips and reads the h5
    member matching `<sim_id>.h5` into a `BytesIO`. The returned object is
    read-only, supports the `with` idiom and can be indexed like any other
    `h5py.File`.

    Args:
        sim_id: The simulation id (matches the h5 filename inside the zip).
        source: Override the Croissant manifest URL / path. Falls back to
            `ddacs.croissant.resolve_source` resolution.
        data_dir: Override the directory searched for already-downloaded
            zips. Defaults to `ddacs.spec.DDACS_SPEC.default_data_dir`. Pass `None`
            to skip the local lookup entirely.
        dataset: A pre-loaded `mlcroissant.Dataset` (e.g. from `ddacs.load`).
            When given, `source` and `data_dir` are ignored: the manifest is
            not re-parsed and the caller's object is used. Useful when the
            caller has applied `ddacs.add_view` and wants the mutation to
            stay in scope.

    Raises:
        FileNotFoundError: No locally mapped zip contained the requested h5.
    """
    ds = (
        dataset
        if dataset is not None
        else _croissant.load(source=source, data_dir=data_dir, spec=spec)
    )
    h5_name = f"{spec.id_format.format(int(sim_id))}.h5"

    for zip_path in (ds.mapping or {}).values():
        zip_path = str(zip_path)
        if not zip_path.endswith(".zip"):
            continue
        try:
            with zipfile.ZipFile(zip_path) as zf:
                if h5_name in zf.namelist():
                    return h5py.File(io.BytesIO(zf.read(h5_name)), "r")
        except zipfile.BadZipFile:
            continue

    hint = (
        " (data_dir lookup was skipped — pass data_dir=...)"
        if data_dir is None
        else f" under {data_dir!r}. Run `{spec.prog} download` to fetch the dataset first."
    )
    raise FileNotFoundError(f"{h5_name} not found in any locally mapped zip{hint}")


def inspect_h5(file_or_path: h5py.File | str | Path) -> None:
    """Pretty-print the group/dataset hierarchy of an HDF5 file.

    Accepts either an open `h5py.File` (e.g. from `ddacs.open_h5`) or a path
    on disk. Prints to stdout and returns nothing.
    """
    if isinstance(file_or_path, h5py.File):
        _print_tree(file_or_path)
    else:
        with h5py.File(file_or_path, "r") as f:
            _print_tree(f)


def _print_tree(f: h5py.File) -> None:
    name = Path(f.filename).name if f.filename else "<in-memory>"
    print(name)
    _print_children(f, prefix="")


def _print_children(obj, prefix: str) -> None:
    """Print attrs (as `@key = value`) then group/dataset members."""
    attrs = list(obj.attrs.items())
    members = list(obj.items()) if isinstance(obj, h5py.Group) else []

    total = len(attrs) + len(members)
    i = 0

    for k, v in attrs:
        i += 1
        last = i == total
        branch = "└── " if last else "├── "
        print(f"{prefix}{branch}@{k} = {v}")

    for k, child in members:
        i += 1
        last = i == total
        branch = "└── " if last else "├── "
        child_prefix = prefix + ("    " if last else "│   ")
        if isinstance(child, h5py.Group):
            print(f"{prefix}{branch}{k}/")
            _print_children(child, child_prefix)
        elif isinstance(child, h5py.Dataset):
            shape = str(child.shape) if child.shape else "scalar"
            print(f"{prefix}{branch}{k}  {shape} {child.dtype}")
            _print_children(child, child_prefix)
