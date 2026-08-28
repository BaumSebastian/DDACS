# API Reference

The DDACS package exposes a small set of public functions, all importable from the top level `ddacs` module.

## Modules

- **[Croissant](croissant.md)**: `load`, `add_view`. Loads the Croissant manifest and adds custom RecordSets.
- **[HDF5](h5.md)**: `open_h5`, `inspect_h5`. Reads a single simulation by id and prints the HDF5 hierarchy.
- **[Streaming](streaming.md)**: `iter_view`, `export_to_numpy`, `export_to_numpy_per_sim`, `load_export`. Torch-free iteration and numpy materialisation.
- **[Visualization](visualization.md)**: `plot_mesh`, `plot_point_cloud`, `plot_vectors`, `plot_2d_projection`. Matplotlib plotting helpers.
- **[PyTorch](pytorch.md)**: `DDACSDataset`. Streaming `IterableDataset` over a Croissant view. Requires the `[torch]` extra.
- **[Dataset spec](spec.md)**: `DatasetSpec`, `DDACS_SPEC`, `MissingDataWarning`. The identity knobs shared with sibling datasets and the warning raised for unavailable simulations.

## Quick import

```python
import ddacs

# Croissant entry points
ds = ddacs.load(data_dir="./data")
ddacs.add_view(ds, "my-view", fields={"forming": ("op10_blank_node_displacement", 2)})

# Single-simulation HDF5 access
with ddacs.open_h5(258864) as f:
    ddacs.inspect_h5(f)

# Streaming (no PyTorch required)
for rec in ddacs.streaming.iter_view("springback-minimal", data_dir="./data", dataset=ds):
    ...

# Visualization
ax, cbar = ddacs.plot_mesh(vertices, faces, values=thickness)

# PyTorch (requires `pip install ddacs[torch]`)
from ddacs.pytorch import DDACSDataset
dataset = DDACSDataset(view="springback-minimal")
```
