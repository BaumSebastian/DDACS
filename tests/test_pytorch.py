"""Tests for ddacs.pytorch module with real dataset."""

import pytest


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
        assert h5_path.exists()
        
        # Test another item if available
        if len(dataset) > 1:
            sim_id2, metadata_vals2, h5_path2 = dataset[1]
            assert isinstance(sim_id2, int)
            assert h5_path2.exists()
    
    def test_dataloader_integration(self, real_data_dir):
        """Test integration with PyTorch DataLoader."""
        try:
            from ddacs.pytorch import DDACSDataset
            from torch.utils.data import DataLoader
        except ImportError:
            pytest.skip("PyTorch not available")
            
        dataset = DDACSDataset(real_data_dir)
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