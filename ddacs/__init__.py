"""DDACS — Deep Drawing and Cutting Simulations Dataset.

Python interface for the DDACS dataset (FEM simulations of sheet metal
forming). Built on a Croissant 1.1 manifest: `ddacs.load()` returns an
`mlcroissant.Dataset` whose `records(view)` streams the data; `add_view`,
`open_h5`, `inspect_h5` are convenience helpers around the same manifest.

Examples:
    >>> import ddacs
    >>> ds = ddacs.load(data_dir="./data")
    >>> for record in ds.records("springback-minimal"):
    ...     ...

    >>> with ddacs.open_h5(258864, data_dir="./data") as f:
    ...     ddacs.inspect_h5(f)
"""

__version__ = "3.0.0"

from .croissant import add_view, load
from .h5_tools import inspect_h5, open_h5
from .visualization import (
    COMPONENT_COLORS,
    plot_2d_projection,
    plot_mesh,
    plot_point_cloud,
    plot_vectors,
)

try:
    from .pytorch import DDACSDataset
except ImportError:
    pass

__all__ = [
    "__version__",
    # Croissant entry point + helpers
    "load",
    "add_view",
    # HDF5 helpers
    "open_h5",
    "inspect_h5",
    # PyTorch (optional — only available if torch is installed)
    "DDACSDataset",
    # Visualization
    "plot_mesh",
    "plot_point_cloud",
    "plot_vectors",
    "plot_2d_projection",
    "COMPONENT_COLORS",
]
