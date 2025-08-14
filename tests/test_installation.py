"""Test package installation and basic functionality."""

import pytest


class TestInstallation:
    """Test package installation and imports."""
    
    def test_package_import(self):
        """Test that the package can be imported."""
        import ddacs
        assert ddacs is not None
    
    def test_core_import(self):
        """Test core module import."""
        from ddacs.core import DDACSIterator
        assert DDACSIterator is not None
    
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
        import ddacs.core
        import ddacs.generators
        
        # Check that classes are available
        from ddacs.core import DDACSIterator
        assert hasattr(DDACSIterator, '__init__')
        assert hasattr(DDACSIterator, '__iter__')
        assert hasattr(DDACSIterator, '__len__')