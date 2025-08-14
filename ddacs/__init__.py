# Core functionality - always available (no external ML dependencies)
from .core import DDACSIterator
from .generators import iter_ddacs, iter_h5_files

# PyTorch integration - only available if PyTorch is installed
try:
    from .pytorch import DDACSDataset
    _PYTORCH_AVAILABLE = True
except ImportError:
    _PYTORCH_AVAILABLE = False
    DDACSDataset = None

# Public API
__all__ = [
    'DDACSIterator',
    'iter_ddacs', 
    'iter_h5_files',
]

# Add PyTorch classes to public API if available
if _PYTORCH_AVAILABLE:
    __all__.append('DDACSDataset')

# Version info
__version__ = '1.0.0'
