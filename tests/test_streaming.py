"""Tests for `ddacs.streaming.iter_view` and `ddacs.streaming.export_to_numpy`."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest

import ddacs


class TestIterViewZippedLayout:
    """The default synthetic fixture stores one `.h5` per zip under `h5/`."""

    def test_yields_one_record_per_sim(self, synthetic_data_dir):
        records = list(
            ddacs.streaming.iter_view("springback-minimal", data_dir=str(synthetic_data_dir))
        )
        assert len(records) >= 1
        for rec in records:
            assert set(rec.keys()) == {
                "forming",
                "springback",
                "_sim_id",  # private scratch key; stripped by export_to_numpy
            }
            assert rec["forming"].shape == (5, 3)
            assert rec["springback"].shape == (5, 3)

    def test_sim_ids_filter(self, synthetic_data_dir):
        records = list(
            ddacs.streaming.iter_view(
                "springback-minimal",
                data_dir=str(synthetic_data_dir),
                sim_ids=[1, 3],
            )
        )
        assert len(records) == 2

    def test_where_filter(self, synthetic_data_dir):
        records = list(
            ddacs.streaming.iter_view(
                "springback-minimal",
                data_dir=str(synthetic_data_dir),
                where=lambda row: row["index"] == 1,
            )
        )
        assert len(records) == 1


class TestIterViewLooseLayout:
    """`ddacs download --extract --remove-zip` produces loose `.h5` files."""

    @pytest.fixture
    def loose_data_dir(self, synthetic_data_dir, tmp_path):
        """Mirror the synthetic dataset with the h5 files extracted from their zips."""
        out = tmp_path / "ddacs_loose"
        (out / "h5").mkdir(parents=True)
        # Copy manifest + csv unchanged.
        (out / "metadata.json").write_text((synthetic_data_dir / "metadata.json").read_text())
        (out / "process_parameters.csv").write_text(
            (synthetic_data_dir / "process_parameters.csv").read_text()
        )
        # Unpack each zip into h5/<sim>.h5.
        for zp in (synthetic_data_dir / "h5").glob("*.zip"):
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(out / "h5")
        return out

    def test_yields_one_record_per_sim(self, loose_data_dir):
        records = list(
            ddacs.streaming.iter_view("springback-minimal", data_dir=str(loose_data_dir))
        )
        assert len(records) >= 1
        for rec in records:
            assert rec["forming"].shape == (5, 3)

    def test_index_prefers_loose_over_zip(self, loose_data_dir, synthetic_data_dir):
        """When both layouts exist side by side, loose files win and zips are ignored."""
        # Copy the original zips into the loose dir so both formats coexist.
        for zp in (synthetic_data_dir / "h5").glob("*.zip"):
            (loose_data_dir / "h5" / zp.name).write_bytes(zp.read_bytes())

        index = ddacs.streaming._build_unified_index(loose_data_dir)
        # Every indexed sim should resolve to a `.h5` file, not a `.zip`.
        for sim_id, path in index.items():
            assert path.endswith(
                ".h5"
            ), f"sim {sim_id} resolved to {path!r}; loose layout should win"


class TestIterViewDatasetKwarg:
    """`add_view` -> `iter_view(dataset=ds)` is the non-PyTorch equivalent of
    `DDACSDataset(dataset=ds)`."""

    def test_custom_view_via_dataset_kwarg(self, synthetic_data_dir):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ddacs.add_view(
            ds,
            "trajectory-only",
            fields={"trajectory": ("op10_blank_node_displacement", None)},
        )

        # Without dataset=, the manifest is re-parsed and the custom view is invisible.
        with pytest.raises(ValueError):
            list(ddacs.streaming.iter_view("trajectory-only", data_dir=str(synthetic_data_dir)))

        # With dataset=, the in-memory mutation carries through.
        records = list(
            ddacs.streaming.iter_view(
                "trajectory-only",
                data_dir=str(synthetic_data_dir),
                dataset=ds,
            )
        )
        assert len(records) >= 1
        for rec in records:
            assert "trajectory" in rec
            assert rec["trajectory"].ndim == 3  # (timesteps, n_nodes, 3)


class TestIterViewInvalidView:
    def test_unknown_view_raises(self, synthetic_data_dir):
        with pytest.raises(ValueError):
            list(ddacs.streaming.iter_view("nonexistent", data_dir=str(synthetic_data_dir)))


class TestExportToNumpy:
    """`export_to_numpy` materialises a view as flat .npy memmap files."""

    def test_basic_round_trip(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards"
        paths = ddacs.streaming.export_to_numpy(
            "springback-minimal",
            out,
            data_dir=str(synthetic_data_dir),
        )
        assert set(paths.keys()) == {"forming", "springback", "sim_ids"}
        for p in paths.values():
            assert p.is_file()

        streamed = list(
            ddacs.streaming.iter_view("springback-minimal", data_dir=str(synthetic_data_dir))
        )
        sim_ids = np.load(paths["sim_ids"])
        forming = np.load(paths["forming"], mmap_mode="r")
        assert forming.shape == (len(streamed), *streamed[0]["forming"].shape)
        for i, rec in enumerate(streamed):
            np.testing.assert_array_equal(forming[i], rec["forming"])
        assert sim_ids.dtype == np.int64
        assert len(sim_ids) == len(streamed)

    def test_per_field_transforms(self, synthetic_data_dir, tmp_path):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ddacs.add_view(
            ds,
            "geom-view",
            fields={"forming": ("op10_blank_node_displacement", 2)},
        )
        out = tmp_path / "shards-tx"
        paths = ddacs.streaming.export_to_numpy(
            "geom-view",
            out,
            data_dir=str(synthetic_data_dir),
            dataset=ds,
            transforms={"forming": lambda arr: arr.astype(np.float32)},
        )
        loaded = np.load(paths["forming"], mmap_mode="r")
        assert loaded.dtype == np.float32

    def test_record_transform_combines_fields(self, synthetic_data_dir, tmp_path):
        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ddacs.add_view(
            ds,
            "combo-view",
            fields={
                "a": ("op10_blank_node_displacement", 2),
                "b": ("op10_blank_node_displacement", 3),
            },
        )
        out = tmp_path / "shards-combo"
        paths = ddacs.streaming.export_to_numpy(
            "combo-view",
            out,
            data_dir=str(synthetic_data_dir),
            dataset=ds,
            record_transform=lambda rec: {"delta": rec["b"] - rec["a"]},
        )
        assert set(paths.keys()) == {"delta", "sim_ids"}
        delta = np.load(paths["delta"], mmap_mode="r")
        assert delta.shape[0] == np.load(paths["sim_ids"]).shape[0]

    def test_sim_ids_filter_subset(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards-subset"
        paths = ddacs.streaming.export_to_numpy(
            "springback-minimal",
            out,
            data_dir=str(synthetic_data_dir),
            sim_ids=[1, 3],
        )
        assert np.load(paths["sim_ids"]).tolist() == [1, 3]
        assert np.load(paths["forming"], mmap_mode="r").shape[0] == 2

    def test_empty_export_raises(self, synthetic_data_dir, tmp_path):
        with pytest.raises(ValueError):
            ddacs.streaming.export_to_numpy(
                "springback-minimal",
                tmp_path / "shards-empty",
                data_dir=str(synthetic_data_dir),
                sim_ids=[999_999_999],
            )


class TestLoadExport:
    """`load_export` is the lazy reader counterpart to `export_to_numpy`."""

    @pytest.fixture
    def shard_dir(self, synthetic_data_dir, tmp_path):
        out = tmp_path / "shards"
        ddacs.streaming.export_to_numpy("springback-minimal", out, data_dir=str(synthetic_data_dir))
        return out

    def test_basic_attributes(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir)
        assert len(export) >= 1
        assert set(export.fields) == {"forming", "springback"}

    def test_getitem_returns_dict(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir)
        rec = export[0]
        assert set(rec.keys()) == set(export.fields)
        for v in rec.values():
            assert isinstance(v, np.ndarray)

    def test_iteration_matches_indexing(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir)
        iterated = list(export)
        assert len(iterated) == len(export)
        for i, rec in enumerate(iterated):
            for alias in export.fields:
                np.testing.assert_array_equal(rec[alias], export[i][alias])

    def test_by_sim_id(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir)
        first_sim_id = int(export.sim_ids[0])
        rec_by_id = export.by_sim_id(first_sim_id)
        rec_by_idx = export[0]
        for alias in export.fields:
            np.testing.assert_array_equal(rec_by_id[alias], rec_by_idx[alias])
        with pytest.raises(KeyError):
            export.by_sim_id(999_999_999)

    def test_index_out_of_range(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir)
        with pytest.raises(IndexError):
            _ = export[len(export)]

    def test_missing_sim_ids_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ddacs.streaming.load_export(tmp_path)

    def test_fields_subset(self, shard_dir):
        export = ddacs.streaming.load_export(shard_dir, fields=["forming"])
        assert export.fields == ("forming",)
        assert set(export[0].keys()) == {"forming"}

    def test_unknown_field_raises(self, shard_dir):
        with pytest.raises(ValueError, match="unknown field"):
            ddacs.streaming.load_export(shard_dir, fields=["forming", "typo"])

    def test_works_with_torch_dataloader(self, shard_dir):
        pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        export = ddacs.streaming.load_export(shard_dir)
        loader = DataLoader(export, batch_size=min(2, len(export)), shuffle=False)
        batch = next(iter(loader))
        assert set(batch.keys()) == set(export.fields)
        for v in batch.values():
            assert v.shape[0] <= 2


class TestMissingDataWarning:
    """Explicitly requested sim_ids that cannot be served must warn, suppressibly."""

    def test_warns_on_id_missing_from_csv(self, synthetic_data_dir):
        import ddacs

        with pytest.warns(ddacs.MissingDataWarning, match="not present in process_parameters"):
            records = list(
                ddacs.streaming.iter_view(
                    "springback-minimal", data_dir=str(synthetic_data_dir), sim_ids=[1, 999]
                )
            )
        assert [r["_sim_id"] for r in records] == [1]

    def test_no_warning_when_all_ids_available(self, synthetic_data_dir):
        import warnings as _warnings

        import ddacs

        with _warnings.catch_warnings():
            _warnings.simplefilter("error", ddacs.MissingDataWarning)
            records = list(
                ddacs.streaming.iter_view(
                    "springback-minimal", data_dir=str(synthetic_data_dir), sim_ids=[1, 3]
                )
            )
        assert [r["_sim_id"] for r in records] == [1, 3]

    def test_warning_is_suppressible(self, synthetic_data_dir):
        import warnings as _warnings

        import ddacs

        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            _warnings.filterwarnings("ignore", category=ddacs.MissingDataWarning)
            list(
                ddacs.streaming.iter_view(
                    "springback-minimal", data_dir=str(synthetic_data_dir), sim_ids=[1, 999]
                )
            )
        assert not [w for w in caught if issubclass(w.category, ddacs.MissingDataWarning)]

    def test_mixed_view_streams_csv_and_h5(self, synthetic_data_dir):
        """Regression for the fixture @context gap: mixed h5+csv views validate."""
        import ddacs

        ds = ddacs.load(data_dir=str(synthetic_data_dir))
        ddacs.add_view(
            ds,
            "mixed",
            fields={
                "disp": "op10_blank_node_displacement",
                "geometry": "process-parameters/geometry",
            },
        )
        records = list(
            ddacs.streaming.iter_view(
                "mixed", data_dir=str(synthetic_data_dir), dataset=ds, sim_ids=[1, 3]
            )
        )
        assert len(records) == 2
        assert all(r["geometry"] == "rectangular" for r in records)
        assert records[0]["disp"].shape == (4, 5, 3)
        assert {r["_sim_id"] for r in records} == {1, 3}
