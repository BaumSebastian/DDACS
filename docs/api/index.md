# API Reference

The DDACS package exposes a small set of public functions, all importable from the top level `ddacs` module.

## Modules

- **[Croissant](croissant.md)**: `load`, `add_view`. Loads the Croissant manifest and adds custom RecordSets.
- **[HDF5](h5.md)**: `open_h5`, `inspect_h5`. Reads a single simulation by id and prints the HDF5 hierarchy.
- **[Visualization](visualization.md)**: `plot_mesh`, `plot_point_cloud`, `plot_vectors`, `plot_2d_projection`. Matplotlib plotting helpers.
- **[PyTorch](pytorch.md)**: `DDACSDataset`. Streaming `IterableDataset` over a Croissant view. Requires the `[torch]` extra.

## Quick import

```python
import ddacs

# Croissant entry points
ds = ddacs.load(data_dir="./data")
ddacs.add_view(ds, "my-view", fields={"forming": ("op10_blank_node_displacement", 2)})

# Single-simulation HDF5 access
with ddacs.open_h5(258864) as f:
    ddacs.inspect_h5(f)

# Visualization
ax, cbar = ddacs.plot_mesh(vertices, faces, values=thickness)

# PyTorch (requires `pip install ddacs[torch]`)
from ddacs.pytorch import DDACSDataset
loader = DDACSDataset(view="springback-minimal")
```
