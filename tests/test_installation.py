"""Test package installation and basic functionality."""

import pytest


class TestInstallation:
    """Test package installation and imports."""
    
    def test_package_import(self):
        """Test that the package can be imported."""
        import ddacs
        assert ddacs is not None
    
    def test_generators_functions_import(self):
        """Test generator functions import."""
        from ddacs import iter_ddacs, get_simulation_by_id, sample_simulations, count_available_simulations
        assert iter_ddacs is not None
        assert get_simulation_by_id is not None
        assert sample_simulations is not None
        assert count_available_simulations is not None
    
    def test_pytorch_import_handling(self):
        """Test PyTorch import handling."""
        try:
            from ddacs.pytorch import DDACSDataset
            # If successful, PyTorch is available
            assert DDACSDataset is not None
        except ImportError as e:
            # Should give clear error message about PyTorch
            assert "PyTorch is required" in str(e)
    
    def test_generators_import(self):
        """Test generators module import."""
        from ddacs import generators
        assert generators is not None
    
    def test_package_structure(self):
        """Test that package has expected structure."""
        import ddacs
        import ddacs.generators
        
        # Check that functions are available
        from ddacs.generators import iter_ddacs, get_simulation_by_id, sample_simulations
        assert callable(iter_ddacs)
        assert callable(get_simulation_by_id)
        assert callable(sample_simulations)