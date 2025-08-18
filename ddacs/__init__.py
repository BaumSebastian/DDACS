from .generators import (
    iter_ddacs, 
    iter_h5_files, 
    get_simulation_by_id, 
    sample_simulations, 
    count_available_simulations
)

# Optional PyTorch dataset (only if PyTorch is available)
try:
    from .pytorch import DDACSDataset
except ImportError:
    pass

# Optional visualization module (only if matplotlib is available)
try:
    from . import visualization
except ImportError:
    pass

# Utils module (optional import based on h5py availability)
try:
    from . import utils
except ImportError:
    pass
