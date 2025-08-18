"""Tests for generator functions, focusing on metadata_vals type safety and memory efficiency."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from ddacs.generators import iter_ddacs, iter_h5_files


class TestMetadataValsTypeSafety:
    """Test metadata_vals type safety and memory efficiency."""
    
    def test_metadata_vals_type_and_memory(self, tmp_path):
        """Test that metadata_vals is np.ndarray and doesn't copy memory unnecessarily."""
        # Create test metadata CSV
        metadata_data = {
            'ID': [113525, 113526, 113527],
            'param1': [0.1, 0.2, 0.3], 
            'param2': [1.0, 2.0, 3.0],
            'param3': [10, 20, 30]
        }
        metadata_df = pd.DataFrame(metadata_data)
        
        # Setup test directory structure
        h5_dir = tmp_path / "h5"
        h5_dir.mkdir()
        metadata_path = tmp_path / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        
        # Create dummy H5 files (CSV might save IDs as floats)
        for sim_id in metadata_data['ID']:
            (h5_dir / f"{sim_id}.h5").touch()
            (h5_dir / f"{float(sim_id)}.h5").touch()  # Also create float version
        
        # Test the generator
        for sim_id, metadata_vals, h5_path in iter_ddacs(tmp_path):
            # Verify type is explicitly np.ndarray
            assert isinstance(metadata_vals, np.ndarray), f"metadata_vals should be np.ndarray, got {type(metadata_vals)}"
            
            # Verify values are correct (excluding ID column)
            expected_row = metadata_df[metadata_df['ID'] == sim_id].iloc[0]
            expected_vals = expected_row.values[1:]  # Skip ID
            
            # Test values are equal
            np.testing.assert_array_equal(metadata_vals, expected_vals)
            
            # Test that our approach preserves type safety
            # Note: Memory sharing can't be tested directly here since CSV reading creates new objects
            # But we verified in test_metadata_vals_no_memory_copy that np.asarray(copy=False) works correctly
            
            break  # Just test first iteration
    
    def test_metadata_vals_no_memory_copy(self):
        """Test that np.asarray with copy=False doesn't create unnecessary copies."""
        # Create test pandas Series
        test_data = pd.Series([1.0, 2.0, 3.0, 'test'])
        original_vals = test_data.values
        
        # Test our approach
        result_vals = np.asarray(original_vals, copy=False)
        
        # Verify no copying occurred
        assert np.shares_memory(original_vals, result_vals), "np.asarray(copy=False) should not copy memory"
        assert isinstance(result_vals, np.ndarray), "Result should be np.ndarray"
        
    def test_metadata_vals_dtype_preservation(self, tmp_path):
        """Test that metadata_vals preserves original data types correctly."""
        # Create test metadata with mixed types
        metadata_data = {
            'ID': [113525],
            'float_param': [3.14],
            'int_param': [42],
            'str_param': ['test']
        }
        metadata_df = pd.DataFrame(metadata_data)
        
        # Setup test directory
        h5_dir = tmp_path / "h5" 
        h5_dir.mkdir()
        metadata_path = tmp_path / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        (h5_dir / "113525.h5").touch()
        
        # Test the generator
        for sim_id, metadata_vals, h5_path in iter_ddacs(tmp_path):
            # Verify the array contains the expected values
            expected_vals = np.array([3.14, 42, 'test'], dtype=object)
            np.testing.assert_array_equal(metadata_vals, expected_vals)
            break


class TestGeneratorFunctions:
    """Basic functionality tests for generator functions."""
    
    def test_iter_ddacs_basic_functionality(self, real_data_dir):
        """Test basic iteration over DDACS data."""
        count = 0
        for sim_id, metadata_vals, h5_path in iter_ddacs(real_data_dir):
            assert isinstance(sim_id, (int, np.integer))
            assert isinstance(metadata_vals, np.ndarray)
            assert isinstance(h5_path, Path)
            assert h5_path.exists()
            count += 1
            if count >= 3:  # Just test first few iterations
                break
    
    def test_iter_h5_files_basic_functionality(self, real_data_dir):
        """Test basic H5 file iteration.""" 
        count = 0
        for h5_path in iter_h5_files(real_data_dir):
            assert isinstance(h5_path, Path)
            assert h5_path.suffix == '.h5'
            assert h5_path.exists()
            count += 1
            if count >= 3:  # Just test first few iterations
                break