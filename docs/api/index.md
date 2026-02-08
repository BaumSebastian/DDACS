# API Reference

The DDACS package provides four main modules:

## Modules

- **[Generators](generators.md)** - Iterator functions for streaming data
- **[Utils](utils.md)** - Utility functions for extracting mesh, stress, strain data
- **[Visualization](visualization.md)** - Plotting utilities for meshes and point clouds
- **[PyTorch](pytorch.md)** - PyTorch Dataset integration (optional)

## Quick Import

```python
# Generator functions
from ddacs import iter_ddacs, iter_h5_files, sample_simulations

# Utility functions
from ddacs import extract_mesh, extract_point_cloud, extract_element_thickness

# Visualization
from ddacs import plot_mesh, plot_point_cloud

# PyTorch (requires torch extra)
from ddacs.pytorch import DDACSDataset
```
