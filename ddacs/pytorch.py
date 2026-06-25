"""PyTorch IterableDataset adapter for DDACS.

Streams records of a Croissant view by walking the locally mapped zips and
applying the view's field selection per simulation. Auto-shards across
DataLoader workers and DDP ranks; supports `where` / `sim_ids` filters and
per-shard seeded shuffle.
"""

from __future__ import annotations

import io
import logging
import random
import re
import zipfile
from collections.abc import Callable
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

try:
    from torch.utils.data import IterableDataset, get_worker_info
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for DDACSDataset. Install with `pip install ddacs[torch]` "
        "or install a flavour from https://pytorch.org/get-started/locally/."
    ) from exc

logger = logging.getLogger(__name__)

_JSONPATH_RE = re.compile(r"^\$\[(.+)\]$")


class DDACSDataset(IterableDataset):
    """Streaming PyTorch dataset for a single Croissant view.

    Yields a `dict[str, numpy.ndarray]` per simulation. Field selection
    (which dataset path, optional timestep slicing) is derived from the
    Croissant view + field-map; no manual extraction code lives here.

    Sharding is decided inside `__iter__` via `get_worker_info()` and
    `torch.distributed`, so the same instance plays under
    `num_workers=0`, `num_workers=N` and DDP without constructor changes.

    Args:
        view: Name of the RecordSet to stream (e.g. "springback-minimal").
        source: Override the manifest URL / path. Defaults to the resolution
            chain in `ddacs.croissant.resolve_source`.
        data_dir: Override the local data directory. Defaults to
            `ddacs.config.DEFAULT_DATA_DIR`. Pass `None` to skip local-file
            discovery.
        sim_ids: Explicit allowlist of simulation ids to stream (default:
            every sim in `process_parameters.csv`).
        where: Predicate applied to each `process_parameters.csv` row before
            any zip is opened. Combined with `sim_ids` if both are given.
        shuffle: If true, each shard shuffles its own sim_id list with a
            seed derived from `seed + epoch + shard_id`. Call `set_epoch`
            between epochs to get a fresh permutation.
        seed: Base seed for the per-shard shuffle.
    """

    def __init__(
        self,
        view: str,
        source: str | Path | None = None,
        data_dir: str | Path | None = DEFAULT_DATA_DIR,
        sim_ids: list[int] | None = None,
        where: Callable[[pd.Series], bool] | None = None,
        shuffle: bool = False,
        seed: int = 0,
    ):
        super().__init__()
        self.view = view
        self.source = source
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.where = where
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        ds = _croissant.load(source=source, data_dir=data_dir)
        self._field_specs = self._build_field_specs(ds)
        self._h5_index = self._build_h5_index(ds.mapping or {})
        self._sim_ids = self._resolve_sim_ids(sim_ids)

    # --- public ------------------------------------------------------------

    def set_epoch(self, epoch: int) -> None:
        """Update the epoch used in shuffle seeding.

        Call once per epoch (analogous to `DistributedSampler.set_epoch`)
        to get a different per-shard permutation each pass.
        """
        self._epoch = int(epoch)

    def __iter__(self):
        shard_id, total_shards = self._shard_position()
        my_ids = self._sim_ids[shard_id::total_shards]

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch * 1_000_003 + shard_id)
            my_ids = list(my_ids)
            rng.shuffle(my_ids)

        # Reuse the open zip across consecutive sims that land in it. Corner
        # blocks group ~2200 sims per zip, so the cache hits nearly every time.
        last_path: str | None = None
        last_zf: zipfile.ZipFile | None = None
        try:
            for sim_id in my_ids:
                zip_path = self._h5_index.get(int(sim_id))
                if zip_path is None:
                    continue
                if zip_path != last_path:
                    if last_zf is not None:
                        last_zf.close()
                    last_zf = zipfile.ZipFile(zip_path)
                    last_path = zip_path
                try:
                    data = last_zf.read(f"{sim_id}.h5")
                except KeyError:
                    continue
                with h5py.File(io.BytesIO(data), "r") as f:
                    yield self._extract_record(f)
        finally:
            if last_zf is not None:
                last_zf.close()

    # --- internals ---------------------------------------------------------

    def _shard_position(self) -> tuple[int, int]:
        """Return `(shard_id, total_shards)` for the current worker × rank."""
        worker = get_worker_info()
        num_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        rank, world = self._ddp_info()
        return rank * num_workers + worker_id, world * num_workers

    @staticmethod
    def _ddp_info() -> tuple[int, int]:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except (ImportError, RuntimeError):
            pass
        return 0, 1

    def _build_field_specs(self, ds) -> dict[str, tuple[str, Any]]:
        """For each view-field, return `(h5_path, slicing)`.

        `slicing` is `None`, an `int` (single timestep), or a `list[int]`
        (multiple timesteps), parsed from the view-field's JSONPath transform.
        """
        view_rs = next((r for r in ds.metadata.record_sets if r.id == self.view), None)
        if view_rs is None:
            raise ValueError(f"view {self.view!r} not found in manifest")
        fm_rs = next(
            (r for r in ds.metadata.record_sets if r.id == FIELD_MAP_RECORD_SET),
            None,
        )
        if fm_rs is None:
            raise ValueError(f"{FIELD_MAP_RECORD_SET!r} RecordSet missing — manifest is malformed")
        fm = {f.name: f for f in fm_rs.fields}

        specs: dict[str, tuple[str, Any]] = {}
        for f in view_rs.fields:
            source_field_id = f.source.uuid.split("/", 1)[-1]
            if source_field_id not in fm:
                raise ValueError(f"view field {f.name!r} sources unknown field {source_field_id!r}")
            h5_path = fm[source_field_id].source.transforms[0].regex
            slicing = None
            if f.source.transforms:
                slicing = self._parse_jsonpath(f.source.transforms[0].json_path)
            specs[f.name] = (h5_path, slicing)
        return specs

    @staticmethod
    def _parse_jsonpath(expr: str | None) -> Any:
        """Parse `$[N]` -> int, `$[a,b,c]` -> list[int]. Anything else -> None."""
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

    @staticmethod
    def _build_h5_index(mapping: dict[str, str]) -> dict[int, str]:
        """Map sim_id (int) -> absolute path of the zip containing `<sim_id>.h5`."""
        index: dict[int, str] = {}
        for path in mapping.values():
            path_str = str(path)
            if not path_str.endswith(".zip"):
                continue
            try:
                with zipfile.ZipFile(path_str) as zf:
                    for name in zf.namelist():
                        if not name.endswith(".h5"):
                            continue
                        try:
                            sim_id = int(Path(name).stem)
                        except ValueError:
                            continue
                        index[sim_id] = path_str
            except zipfile.BadZipFile:
                continue
        return index

    def _resolve_sim_ids(self, sim_ids_arg: list[int] | None) -> list[int]:
        """Apply sim_ids + where to produce the final ordered list of sim ids.

        The list is built once at construction time (before any worker fork)
        so every shard sees the same ordering.
        """
        if self.data_dir is not None:
            csv_path = self.data_dir / PROCESS_PARAMETERS_FILE
            if csv_path.is_file():
                df = pd.read_csv(csv_path)
                if ID_COLUMN not in df.columns:
                    raise ValueError(f"{csv_path} missing required {ID_COLUMN!r} column")
                if sim_ids_arg is not None:
                    df = df[df[ID_COLUMN].isin(set(sim_ids_arg))]
                if self.where is not None:
                    df = df[df.apply(self.where, axis=1)]
                return [int(x) for x in df[ID_COLUMN].tolist()]

        if self.where is not None:
            raise ValueError(f"`where` filter requires {PROCESS_PARAMETERS_FILE} under data_dir")
        if sim_ids_arg is not None:
            return [int(x) for x in sim_ids_arg]
        return sorted(self._h5_index.keys())

    def _extract_record(self, f: h5py.File) -> dict[str, np.ndarray]:
        rec: dict[str, np.ndarray] = {}
        for alias, (h5_path, slicing) in self._field_specs.items():
            arr = f[h5_path][...]
            if slicing is not None:
                arr = arr[slicing]
            rec[alias] = arr
        return rec
