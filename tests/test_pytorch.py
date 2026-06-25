"""Tests for `ddacs.pytorch.DDACSDataset`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader  # noqa: E402

from ddacs.pytorch import DDACSDataset  # noqa: E402


class TestBasicIteration:
    def test_field_specs_resolved(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        assert ds._field_specs == {
            "forming": ("OP10/blank/node_displacement", 2),
            "springback": ("OP10/blank/node_displacement", 3),
        }

    def test_sim_ids_loaded_from_csv(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        assert ds._sim_ids == [1, 2, 3]

    def test_h5_index_built(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        assert set(ds._h5_index.keys()) == {1, 2, 3}

    def test_iterates_one_record_per_sim(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        records = list(ds)
        assert len(records) == 3
        for rec in records:
            assert set(rec.keys()) == {"forming", "springback"}
            assert rec["forming"].shape == (5, 3)
            assert rec["springback"].shape == (5, 3)

    def test_invalid_view_raises(self, synthetic_data_dir):
        with pytest.raises(ValueError):
            DDACSDataset(view="nonexistent", data_dir=str(synthetic_data_dir))


class TestFilters:
    def test_sim_ids_allowlist(self, synthetic_data_dir):
        ds = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            sim_ids=[1, 3],
        )
        assert ds._sim_ids == [1, 3]
        assert len(list(ds)) == 2

    def test_where_predicate(self, synthetic_data_dir):
        # All synthetic rows have geometry=rectangular, so both filters
        # produce the expected counts.
        kept = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            where=lambda row: row["geometry"] == "rectangular",
        )
        assert kept._sim_ids == [1, 2, 3]

        dropped = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            where=lambda row: row["geometry"] == "concave",
        )
        assert dropped._sim_ids == []


class TestDataLoader:
    def test_batch_size_two(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        loader = DataLoader(ds, batch_size=2, num_workers=0)
        batches = list(loader)
        # 3 records, batch_size=2 -> [2, 1]
        assert [tuple(b["forming"].shape) for b in batches] == [
            (2, 5, 3),
            (1, 5, 3),
        ]

    def test_num_workers_sharding_no_duplicates(self, synthetic_data_dir):
        ds = DDACSDataset(view="springback-minimal", data_dir=str(synthetic_data_dir))
        loader = DataLoader(ds, batch_size=1, num_workers=3)
        seen = set()
        for batch in loader:
            arr = batch["forming"].numpy()
            # Each batch carries one record; use the first element as a fingerprint.
            seen.add(float(arr[0, 0, 0]))
        assert len(seen) == 3, "expected exactly one record per sim_id, no duplicates"


class TestShuffle:
    def test_seeded_shuffle_is_reproducible(self, synthetic_data_dir):
        a = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        b = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        # Same seed + same epoch -> same shard order
        order_a = [rec["forming"][0, 0].item() for rec in a]
        order_b = [rec["forming"][0, 0].item() for rec in b]
        assert order_a == order_b

    def test_set_epoch_changes_order(self, synthetic_data_dir):
        ds = DDACSDataset(
            view="springback-minimal",
            data_dir=str(synthetic_data_dir),
            shuffle=True,
            seed=42,
        )
        ds.set_epoch(0)
        epoch0 = [rec["forming"][0, 0].item() for rec in ds]
        ds.set_epoch(1)
        epoch1 = [rec["forming"][0, 0].item() for rec in ds]
        # Either order may match by chance, but they should differ for at
        # least one of the synthetic ids (with seed=42 across 3 records).
        assert epoch0 != epoch1
