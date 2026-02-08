"""
DDACS - Deep Drawing and Cutting Simulations Dataset.

A Python package for accessing and processing the DDACS dataset containing
FEM simulations of sheet metal forming processes.

Examples:
    >>> from ddacs import iter_ddacs, count_available_simulations
    >>> count = count_available_simulations('./data')
    >>> for sim_id, metadata, h5_path in iter_ddacs('./data'):
    ...     print(f"Simulation {sim_id}")

    >>> from ddacs.utils import extract_mesh, extract_element_thickness
    >>> vertices, triangles = extract_mesh('simulation.h5', 'blank')
"""

__version__ = "2.0.2"

from .generators import (
    count_available_simulations,
    get_simulation_by_id,
    iter_ddacs,
    iter_h5_files,
    sample_simulations,
)
from .utils import (
    compute_von_mises,
    display_structure,
    extract_element_strain,
    extract_element_stress,
    extract_element_thickness,
    extract_mesh,
    extract_point_cloud,
    extract_point_springback,
    non_degenerate_mask,
)

# Optional PyTorch dataset (only if PyTorch is available)
try:
    from .pytorch import DDACSDataset
except ImportError:
    pass

# Visualization
from .visualization import (
    COMPONENT_COLORS,
    plot_2d_projection,
    plot_mesh,
    plot_point_cloud,
    plot_vectors,
)

__all__ = [
    # Version
    "__version__",
    # Generators
    "iter_ddacs",
    "iter_h5_files",
    "get_simulation_by_id",
    "sample_simulations",
    "count_available_simulations",
    # Utils
    "extract_point_cloud",
    "extract_mesh",
    "extract_element_thickness",
    "extract_element_stress",
    "extract_element_strain",
    "extract_point_springback",
    "compute_von_mises",
    "non_degenerate_mask",
    "display_structure",
    # PyTorch (optional)
    "DDACSDataset",
    # Visualization
    "plot_mesh",
    "plot_point_cloud",
    "plot_vectors",
    "plot_2d_projection",
    "COMPONENT_COLORS",
]
