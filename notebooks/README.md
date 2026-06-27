# DDACS notebooks

Five end-to-end Jupyter notebooks that companion the [online tutorials](https://ddacs.readthedocs.io/en/latest/tutorials/). Each notebook is self-contained: it opens with a Walkthrough list and the Assumptions it relies on, then walks through the topic step by step. Reading top to bottom is the intended flow.

## Prerequisites

```bash
# Install the package + the PyTorch extra (needed for 03_pytorch.ipynb).
pip install 'ddacs[torch]'

# Fetch the small sample bundle once. Writes metadata.json,
# process_parameters.csv, and one sample simulation (258864.zip)
# into ./data.
ddacs download --small -y
```

Notebooks 04 and 05 work off the same `./data` directory. Notebook 05 additionally writes a throwaway loose-HDF5 layout into `/tmp/ddacs_loose` so the project-local `./data` stays untouched.

## Run

From the repository root:

```bash
jupyter lab notebooks/
```

Inside each notebook the data directory is hard-coded as `DATA_DIR = '../data'` because the notebooks live in `notebooks/` while `./data` sits at the repo root.

## Index

| Notebook | Companion tutorial | What it covers |
|----------|-------------------|----------------|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | [Getting Started](https://ddacs.readthedocs.io/en/latest/tutorials/getting-started/) | Install, download, `ddacs.load`, iterate one record, open one HDF5 file with `ddacs.open_h5`, render the formed blank coloured by sheet thickness. |
| [`02_views.ipynb`](02_views.ipynb) | [Build your own view](https://ddacs.readthedocs.io/en/latest/tutorials/views/) | Append a custom `RecordSet` with `ddacs.add_view`, inspect the resolved field map and timestep transforms, iterate the new view, and read the published `process-parameters` and `simulation-provenance` (SIM-KAx) RecordSets. |
| [`03_pytorch.ipynb`](03_pytorch.ipynb) | [PyTorch training](https://ddacs.readthedocs.io/en/latest/tutorials/pytorch/) | Build a `DDACSDataset`, iterate one batch through a `DataLoader`, filter by Croissant manifest columns, build the canonical train/val/test splits, shuffle with `set_epoch`, and contrast the graceful skip vs. the `records()` abort on partial downloads. |
| [`04_visualization.ipynb`](04_visualization.ipynb) | [Visualization](https://ddacs.readthedocs.io/en/latest/tutorials/visualization/) | Thickness mesh with the false-colour cmap, 2 x 2 component grid, springback magnitude on the OP20 mesh, pure arrow field, and a composed mesh + arrows section. |
| [`05_loose_h5.ipynb`](05_loose_h5.ipynb) | [Loose HDF5 recipe](https://ddacs.readthedocs.io/en/latest/tutorials/loose-h5/) | The post-`--extract --remove-zip` workflow: read `process_parameters.csv` with pandas, filter rows, then open each loose `.h5` directly with `h5py`. |
| [`06_streaming.ipynb`](06_streaming.ipynb) | [Streaming and numpy export](https://ddacs.readthedocs.io/en/latest/tutorials/streaming/) | `ddacs.streaming.iter_view` for offline iteration without PyTorch; `streaming.export_to_numpy` with a `record_transform` that geometrically normalises the point cloud to `[-1, 1]` and stores `center_mm` / `scale_mm` for de-normalisation; back-to-back timing showing ~1000x speedup after the one-time export. |

If any of these fail to run end to end after `ddacs download --small`, please open an [issue](https://github.com/BaumSebastian/DDACS/issues) so we can fix the discrepancy with the docs.
