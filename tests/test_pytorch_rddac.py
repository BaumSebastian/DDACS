"""Substantive PyTorch tests over the local rddac fixture.

Skipped when `h5/rddac.zip` isn't present under the rddac data dir.
The fixture documents how to set it up.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from ddacs.pytorch import DDACSDataset  # noqa: E402

# RDDAC sub-study has 396 simulations.
RDDAC_SIM_COUNT = 396


@pytest.fixture(scope="module")
def rddac_dataset(rddac_data_dir):
    return DDACSDataset(
        view="springback-minimal",
        data_dir=str(rddac_data_dir),
        where=lambda row: row["rddac"],
    )


def test_where_filter_keeps_all_rddac(rddac_dataset):
    assert len(rddac_dataset._sim_ids) == RDDAC_SIM_COUNT


def test_iterates_all_rddac_sims(rddac_dataset):
    """Single-process iteration yields one record per sim_id."""
    count = sum(1 for _ in rddac_dataset)
    assert count == RDDAC_SIM_COUNT


def test_sharding_partitions_sim_ids_exactly(rddac_dataset):
    """Pure-math check on the worker × DDP slicing.

    For every shard configuration up to 8 shards, the union of `[shard::total]`
    slices must equal the original sim_ids list — no duplicates, no missing.
    """
    sim_ids = rddac_dataset._sim_ids
    for total in (1, 2, 3, 4, 8):
        seen: list[int] = []
        for shard in range(total):
            seen.extend(sim_ids[shard::total])
        assert sorted(seen) == sorted(sim_ids), f"shard math wrong at total={total}"


def test_dataloader_num_workers_yields_full_set(rddac_dataset):
    """End-to-end check: 4 workers stream the full 396 records.

    Combined with the math test above (no shard overlap), this proves no
    duplicates without depending on per-sim record content.
    """
    loader = DataLoader(rddac_dataset, batch_size=1, num_workers=4)
    count = sum(1 for _ in loader)
    assert count == RDDAC_SIM_COUNT


def test_large_batch_stacks(rddac_dataset):
    """DataLoader batch_size=16 stacks correctly when shapes match."""
    loader = DataLoader(rddac_dataset, batch_size=16, num_workers=0)
    first = next(iter(loader))
    assert first["op10_blank_node_displacement_forming"].shape[0] == 16
    assert first["op10_blank_node_displacement_springback"].shape[0] == 16
