# Tutorials

These tutorials walk through the typical workflows on top of the `ddacs` Python surface, from a first download to a PyTorch training loop. Each page is short and code centric: read top to bottom or jump to the section that matches the task at hand.

1. [Getting Started](getting-started.md): Install, download the small bundle, load the manifest, open one simulation, render the first plot.
2. [Build your own view](views.md): Use `ddacs.add_view` to compose a custom Croissant RecordSet from the field map.
3. [PyTorch training](pytorch.md): Stream a view through `DDACSDataset`, batching with `DataLoader`, multi worker sharding, DDP, and a reproducibility note. Includes the per simulation read benchmark.
4. [Visualization](visualization.md): Mesh, point cloud, vector field, and projection plots on top of `ddacs.open_h5`.
5. [Loose HDF5 recipe](loose-h5.md): Iterate over loose `.h5` files after `ddacs download --extract --remove-zip`.
6. [Streaming and numpy export](streaming.md): Iterate any view with `ddacs.streaming.iter_view` (no PyTorch, no mlcroissant FileSet walk), and materialise it as flat `.npy` shards with `streaming.export_to_numpy` for ~1000x faster training reads.

## Prerequisites

```bash
pip install ddacs
ddacs download --small
```

## Notebooks

The companion Jupyter notebooks live in the [`notebooks/`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks) directory of the repository. Each tutorial page has a matching notebook that reproduces the code blocks end to end.
