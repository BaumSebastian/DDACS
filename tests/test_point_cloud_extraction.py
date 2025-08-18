"""
Test suite for point cloud extraction functionality.

Tests to verify that the extract_point_cloud function correctly handles
timesteps and that points actually move between different timesteps.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from ddacs.utils import extract_point_cloud, display_structure


class TestPointCloudExtraction:
    """Test class for point cloud extraction functionality."""
    
    @pytest.fixture
    def sample_h5_files(self):
        """Fixture to provide sample H5 files for testing."""
        data_dir = Path("data/h5")
        if not data_dir.exists():
            pytest.skip("No H5 data directory found")
        
        h5_files = list(data_dir.glob("*.h5"))
        if not h5_files:
            pytest.skip("No H5 files found in data directory")
        
        # Return first few valid files for testing
        return h5_files[:3]
    
    def test_extract_point_cloud_basic(self, sample_h5_files):
        """Test basic point cloud extraction functionality."""
        for h5_path in sample_h5_files:
            try:
                coords = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                
                # Basic shape and type checks
                assert isinstance(coords, np.ndarray)
                assert coords.shape[1] == 3  # Should have 3 coordinates (x, y, z)
                assert coords.dtype.kind == 'f'  # Should be floating point
                assert coords.shape[0] > 0  # Should have some nodes
                
                # Check that coordinates are reasonable (not all zeros or NaN)
                assert not np.all(coords == 0)
                assert not np.any(np.isnan(coords))
                
            except Exception as e:
                pytest.fail(f"Basic extraction failed for {h5_path}: {e}")
    
    def test_timestep_differences(self, sample_h5_files):
        """Test that different timesteps actually produce different point clouds."""
        for h5_path in sample_h5_files:
            try:
                # Check if the file has displacement data
                with h5py.File(h5_path, 'r') as f:
                    comp_group = f['OP10/blank']
                    if 'node_displacement' not in comp_group:
                        pytest.skip(f"No displacement data in {h5_path}")
                    
                    disp = comp_group['node_displacement']
                    if len(disp.shape) != 3:
                        pytest.skip(f"Displacement data is not 3D in {h5_path}")
                    
                    num_timesteps = disp.shape[0]
                    if num_timesteps < 2:
                        pytest.skip(f"Not enough timesteps in {h5_path}")
                
                # Extract point clouds for different timesteps
                coords_t0 = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                coords_t1 = extract_point_cloud(str(h5_path), 'blank', timestep=1)
                
                # Check that shapes match
                assert coords_t0.shape == coords_t1.shape, f"Shape mismatch between timesteps in {h5_path}"
                
                # Calculate differences
                diff = coords_t1 - coords_t0
                diff_magnitudes = np.linalg.norm(diff, axis=1)
                mean_diff = np.mean(diff_magnitudes)
                max_diff = np.max(diff_magnitudes)
                
                # Check that there is actually movement between timesteps
                assert mean_diff > 1e-10, f"No movement detected between timesteps 0 and 1 in {h5_path}. Mean diff: {mean_diff}"
                assert max_diff > 1e-10, f"No movement detected between timesteps 0 and 1 in {h5_path}. Max diff: {max_diff}"
                
                # The movement should be physically reasonable (not too large)
                # Assuming coordinates are in mm, movements should be less than 1000mm typically
                assert max_diff < 1000.0, f"Suspiciously large movement detected in {h5_path}: {max_diff}"
                
            except Exception as e:
                pytest.fail(f"Timestep difference test failed for {h5_path}: {e}")
    
    def test_last_timestep_extraction(self, sample_h5_files):
        """Test extraction of last timestep using -1 index."""
        for h5_path in sample_h5_files:
            try:
                # Test that -1 timestep works
                coords_last = extract_point_cloud(str(h5_path), 'blank', timestep=-1)
                coords_first = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                
                # Should have same number of points
                assert coords_last.shape == coords_first.shape
                
                # Should be different from first timestep
                diff = coords_last - coords_first
                diff_magnitudes = np.linalg.norm(diff, axis=1)
                mean_diff = np.mean(diff_magnitudes)
                
                # Last timestep should typically be significantly different from first
                assert mean_diff > 1e-6, f"Last timestep appears identical to first in {h5_path}"
                
            except Exception as e:
                pytest.fail(f"Last timestep test failed for {h5_path}: {e}")
    
    def test_timestep_consistency(self, sample_h5_files):
        """Test that extracting the same timestep twice gives identical results."""
        for h5_path in sample_h5_files:
            try:
                coords1 = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                coords2 = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                
                # Should be exactly identical
                np.testing.assert_array_equal(coords1, coords2)
                
            except Exception as e:
                pytest.fail(f"Consistency test failed for {h5_path}: {e}")
    
    def test_displacement_application_bug(self, sample_h5_files):
        """
        Specific test for the suspected bug in displacement application.
        
        This test checks if the function correctly applies displacements
        from the displacement array to the base coordinates.
        """
        for h5_path in sample_h5_files:
            try:
                # Manually check the displacement logic
                with h5py.File(h5_path, 'r') as f:
                    comp_group = f['OP10/blank']
                    if 'node_displacement' not in comp_group:
                        continue
                    
                    coords = np.array(comp_group['node_coordinates'])
                    disp = np.array(comp_group['node_displacement'])
                    
                    if len(disp.shape) != 3:  # Not a 3D displacement array
                        continue
                    
                    # Manual calculation for timestep 0 and 1
                    expected_t0 = coords + disp[0]
                    expected_t1 = coords + disp[1]
                    
                    # Compare with function output
                    actual_t0 = extract_point_cloud(str(h5_path), 'blank', timestep=0)
                    actual_t1 = extract_point_cloud(str(h5_path), 'blank', timestep=1)
                    
                    # Check if function output matches expected
                    np.testing.assert_allclose(actual_t0, expected_t0, rtol=1e-10,
                                             err_msg=f"Timestep 0 displacement not applied correctly in {h5_path}")
                    np.testing.assert_allclose(actual_t1, expected_t1, rtol=1e-10,
                                             err_msg=f"Timestep 1 displacement not applied correctly in {h5_path}")
                    
                    # Check that the displacements are actually different
                    disp_diff = np.linalg.norm(disp[1] - disp[0], axis=1)
                    mean_disp_diff = np.mean(disp_diff)
                    
                    assert mean_disp_diff > 1e-10, f"Displacements are identical between timesteps 0 and 1 in {h5_path}"
                    
            except Exception as e:
                pytest.fail(f"Displacement application test failed for {h5_path}: {e}")
    
    def test_multiple_components(self, sample_h5_files):
        """Test extraction from different components."""
        components = ['blank', 'die', 'punch', 'binder']
        
        for h5_path in sample_h5_files:
            try:
                with h5py.File(h5_path, 'r') as f:
                    available_components = list(f['OP10'].keys())
                
                for component in components:
                    if component in available_components:
                        try:
                            coords = extract_point_cloud(str(h5_path), component, timestep=0)
                            assert coords.shape[1] == 3
                            assert coords.shape[0] > 0
                        except Exception as e:
                            pytest.fail(f"Failed to extract {component} from {h5_path}: {e}")
                            
            except Exception as e:
                pytest.fail(f"Component extraction test failed for {h5_path}: {e}")


def test_file_not_found():
    """Test that FileNotFoundError is properly handled."""
    with pytest.raises(FileNotFoundError):
        extract_point_cloud("nonexistent_file.h5", "blank", timestep=0)


def test_invalid_component():
    """Test handling of invalid component names."""
    # This test requires a valid H5 file
    data_dir = Path("data/h5")
    if not data_dir.exists():
        pytest.skip("No H5 data directory found")
    
    h5_files = list(data_dir.glob("*.h5"))
    if not h5_files:
        pytest.skip("No H5 files found")
    
    with pytest.raises(KeyError):
        extract_point_cloud(str(h5_files[0]), "invalid_component", timestep=0)


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__, "-v"])