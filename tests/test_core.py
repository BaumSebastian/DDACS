"""Tests for ddacs.core module with real dataset."""

import pytest
from ddacs.core import DDACSIterator


class TestDDACSIterator:
    """Test cases for DDACSIterator with real data."""
    
    def test_init_with_real_data(self, real_data_dir):
        """Test initialization with real dataset."""
        iterator = DDACSIterator(real_data_dir)
        assert len(iterator) > 0
        print(f"Found {len(iterator)} simulations")
    
    def test_iteration_sample(self, real_data_dir):
        """Test iteration with first few samples."""
        iterator = DDACSIterator(real_data_dir)
        
        # Test first 5 samples
        count = 0
        for sim_id, metadata_vals, h5_path in iterator:
            assert isinstance(sim_id, int)
            assert h5_path.exists()
            assert h5_path.suffix == '.h5'
            count += 1
            if count >= 5:  # Only test first 5
                break
    
    def test_get_by_id_real(self, real_data_dir):
        """Test getting simulation by ID with real data."""
        iterator = DDACSIterator(real_data_dir)
        
        # Get first available ID
        first_sim = next(iter(iterator))
        sim_id = first_sim[0]
        
        # Test retrieving it
        result = iterator.get_by_id(sim_id)
        assert result is not None
        assert result[0] == sim_id
    
    def test_sample_real(self, real_data_dir):
        """Test sampling with real data."""
        iterator = DDACSIterator(real_data_dir)
        
        # Sample 3 simulations
        samples = list(iterator.sample(3))
        assert len(samples) == 3
        
        for sim_id, metadata_vals, h5_path in samples:
            assert h5_path.exists()
    
    def test_string_representation_real(self, real_data_dir):
        """Test string representation with real data."""
        iterator = DDACSIterator(real_data_dir)
        str_repr = str(iterator)
        
        assert "DDACS Iterator" in str_repr
        assert "Available simulations:" in str_repr
        print(str_repr)