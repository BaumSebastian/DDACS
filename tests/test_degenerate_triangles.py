"""
Test suite for identifying degenerate triangles functionality.

Tests to verify that the no_degenerate_mask function correctly handles incorrect triangles.
"""

import pytest
import numpy as np
import h5py
from pathlib import Path
from ddacs.pytorch import DDACSDataset
from ddacs.utils import extract_mesh, extract_element_thickness, non_degenerate_mask


class TestDegenerateTrianlges:
    """Test class for degenerate triangles functionality."""

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

    def test_basic_extract_mesh(self, sample_h5_files):
        """Test basic data read in."""
        for h5_path in sample_h5_files:
            try:
                verts, triangles = extract_mesh(str(h5_path), "blank", timestep=0)

                # Basic shape and type checks
                assert isinstance(verts, np.ndarray)
                assert verts.shape[1] == 3  # Should have 3 coordinates (x, y, z)
                assert verts.dtype.kind == "f"  # Should be floating point
                assert verts.shape[0] > 0  # Should have some nodes

                assert triangles.shape[1] == 3  # Should have 3 coordinates (x, y, z)
                assert triangles.dtype.kind == "i"  # Should be floating point
                assert triangles.shape[0] > 0  # Should have some nodes

                # Check that coordinates are reasonable (not all zeros or NaN)
                assert not np.all(triangles == 0)
                assert not np.any(np.isnan(triangles))
                assert not np.all(verts == 0)
                assert not np.any(np.isnan(verts))

            except Exception as e:
                pytest.fail(f"Basic extraction failed for {h5_path}: {e}")

    def test_op10_degenerate_triangles(self, sample_h5_files):
        """Test that different timesteps actually produce different point clouds."""
        for h5_path in sample_h5_files:
            try:
                # OP10 should contain all triangles
                _, triangles = extract_mesh(h5_path, "blank", timestep=-1, operation=10)
                mask = non_degenerate_mask(triangles)
                assert False not in mask  # All triangles should be valid

            except Exception as e:
                pytest.fail(f"OP10 contains degenerated triangles {h5_path}: {e}")

    def test_op20_degenerate_triangles(self, sample_h5_files):
        """Test that different timesteps actually produce different point clouds."""
        for h5_path in sample_h5_files:
            try:
                # OP10 should contain all triangles
                _, triangles = extract_mesh(h5_path, "blank", timestep=-1, operation=20)
                thickness = extract_element_thickness(
                    h5_path, timestep=-1, operation=20
                )
                assert (
                    triangles.shape[0] == thickness.shape[0]
                )  # Check if the structure is the same

                mask = non_degenerate_mask(triangles)
                assert False in mask  # triangles should be degenerated

                # it should remove all degenerate triangles
                triangles = triangles[mask]
                thickness = thickness[mask]

                mask = non_degenerate_mask(triangles)
                assert False not in mask  # All degenerated should be removed
                assert (
                    triangles.shape[0] == thickness.shape[0]
                )  # Check if the structure is the same

            except Exception as e:
                pytest.fail(f"OP10 contains degenerated triangles {h5_path}: {e}")


if __name__ == "__main__":
    # Allow running the tests directly
    pytest.main([__file__, "-v"])
