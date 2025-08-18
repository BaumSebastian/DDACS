"""Tests for ddacs.pytorch module with real dataset."""

import pytest
from pathlib import Path


class TestDDACSDataset:
    """Test cases for DDACSDataset with real data."""

    def test_pytorch_import(self):
        """Test that PyTorch components can be imported."""
        try:
            from ddacs.pytorch import DDACSDataset
            from torch.utils.data import DataLoader
        except ImportError as e:
            pytest.skip(f"PyTorch not available: {e}")

    def test_dataset_init(self, real_data_dir):
        """Test dataset initialization."""
        try:
            from ddacs.pytorch import DDACSDataset
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)
        assert len(dataset) > 0
        print(f"Dataset contains {len(dataset)} samples")

    def test_dataset_getitem(self, real_data_dir):
        """Test dataset item access."""
        try:
            from ddacs.pytorch import DDACSDataset
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)

        # Test first item
        sim_id, metadata_vals, h5_path = dataset[0]
        assert isinstance(sim_id, int)
        assert Path(h5_path).exists()  # h5_path is returned as string

        # Test another item if available
        if len(dataset) > 1:
            sim_id2, metadata_vals2, h5_path2 = dataset[1]
            assert isinstance(sim_id2, int)
            assert Path(h5_path2).exists()  # h5_path is returned as string

    def test_dataloader_integration(self, real_data_dir):
        """Test integration with PyTorch DataLoader."""
        try:
            from ddacs.pytorch import DDACSDataset
            from torch.utils.data import DataLoader
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)

        # Skip DataLoader test if metadata contains mixed types
        # (PyTorch DataLoader has issues with object arrays containing strings)
        sim_id, metadata_vals, h5_path = dataset[0]
        if metadata_vals.dtype == object:
            pytest.skip(
                "DataLoader not compatible with mixed-type metadata (contains strings)"
            )

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Test first batch
        batch = next(iter(dataloader))
        sim_ids, metadata_vals, h5_paths = batch

        assert len(sim_ids) <= 2  # batch_size
        assert len(metadata_vals) <= 2
        assert len(h5_paths) <= 2

    def test_dataset_string_representation(self, real_data_dir):
        """Test dataset string representation."""
        try:
            from ddacs.pytorch import DDACSDataset
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)
        str_repr = str(dataset)

        assert "DDACS Dataset (PyTorch)" in str_repr
        assert "Samples:" in str_repr
        print(str_repr)

    def test_get_metadata_columns(self, real_data_dir):
        """Test get_metadata_columns method."""
        try:
            from ddacs.pytorch import DDACSDataset
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)
        columns = dataset.get_metadata_columns()

        # Verify it returns a list
        assert isinstance(columns, list)
        # Verify ID is not included
        assert "ID" not in columns
        # Verify it's not empty (assuming real data has metadata)
        assert len(columns) > 0
        print(f"Metadata columns: {columns}")

    def test_get_metadata_descriptions(self, real_data_dir):
        """Test get_metadata_descriptions method."""
        try:
            from ddacs.pytorch import DDACSDataset
        except ImportError:
            pytest.skip("PyTorch not available")

        dataset = DDACSDataset(real_data_dir)
        descriptions = dataset.get_metadata_descriptions()

        # Verify it returns a dictionary
        assert isinstance(descriptions, dict)
        # Verify all values are strings
        assert all(isinstance(desc, str) for desc in descriptions.values())
        # Verify no empty descriptions
        assert all(len(desc) > 0 for desc in descriptions.values())

        print("Parameter descriptions:")
        for param, desc in descriptions.items():
            print(f"  {param}: {desc}")

    def test_metadata_vals_memory_efficiency(self, real_data_dir):
        """Test that metadata_vals uses memory-efficient np.asarray approach."""
        try:
            from ddacs.pytorch import DDACSDataset
            import numpy as np
        except ImportError:
            pytest.skip("PyTorch or NumPy not available")

        dataset = DDACSDataset(real_data_dir)
        sim_id, metadata_vals, h5_path = dataset[0]

        # Verify it's explicitly np.ndarray (shows np.asarray was used)
        assert isinstance(metadata_vals, np.ndarray)
        assert type(metadata_vals).__name__ == "ndarray"

        # Verify it contains expected number of values (columns - ID)
        expected_length = len(dataset.get_metadata_columns())
        assert len(metadata_vals) == expected_length
