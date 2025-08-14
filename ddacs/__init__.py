from .core import DDACSIterator
from .generators import iter_ddacs, iter_h5_files

# Optional PyTorch dataset (only if PyTorch is available)
try:
    from .pytorch import DDACSDataset
except ImportError:
    pass
