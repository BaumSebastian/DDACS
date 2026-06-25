"""Tests for `ddacs.open_h5` and `ddacs.inspect_h5`."""

from __future__ import annotations

import h5py
import pytest

import ddacs


class TestOpenH5:
    def test_returns_h5py_file(self, synthetic_data_dir):
        with ddacs.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            assert isinstance(f, h5py.File)
            assert "OP10/blank/node_displacement" in f

    def test_attrs_round_trip(self, synthetic_data_dir):
        with ddacs.open_h5(2, data_dir=str(synthetic_data_dir)) as f:
            assert f.attrs["index"] == 2

    def test_read_only_mode(self, synthetic_data_dir):
        with ddacs.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            assert f.mode == "r"

    def test_missing_sim_raises_filenotfound(self, synthetic_data_dir):
        with pytest.raises(FileNotFoundError):
            ddacs.open_h5(99999, data_dir=str(synthetic_data_dir))

    def test_iterating_multiple_sims(self, synthetic_data_dir):
        for sid in (1, 2, 3):
            with ddacs.open_h5(sid, data_dir=str(synthetic_data_dir)) as f:
                assert f.attrs["index"] == sid


class TestInspectH5:
    def test_accepts_open_file(self, synthetic_data_dir, capsys):
        with ddacs.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            ddacs.inspect_h5(f)
        out = capsys.readouterr().out
        assert "OP10/" in out
        assert "@index = 1" in out

    def test_accepts_path(self, synthetic_data_dir, tmp_path, capsys):
        # Extract one h5 to disk to test the path variant.
        import zipfile

        zip_path = synthetic_data_dir / "h5" / "1.zip"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        ddacs.inspect_h5(tmp_path / "1.h5")
        out = capsys.readouterr().out
        assert "OP10/" in out

    def test_tree_characters_present(self, synthetic_data_dir, capsys):
        with ddacs.open_h5(1, data_dir=str(synthetic_data_dir)) as f:
            ddacs.inspect_h5(f)
        out = capsys.readouterr().out
        assert "├──" in out or "└──" in out
