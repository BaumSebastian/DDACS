<div align="center">
  <img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/icon/DDACS.png" width="150"/>
  <h1>Deep Drawing and Cutting Simulations (DDACS) Dataset</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/ddacs/badge/?version=latest)](https://ddacs.readthedocs.io)
[![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801)
[![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--4801-blue.svg)](https://doi.org/10.18419/DARUS-4801)
[![Paper](https://img.shields.io/badge/paper-MATEC%20Web%20Conf.-red.svg)](https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html)

A Croissant-native Python package for accessing the [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801). One CLI for the download, one Python module for access, and an optional PyTorch `IterableDataset` for training.

**[Read the full documentation](https://ddacs.readthedocs.io)**

<div align="center">

![Simulation overview](https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/simulation_overview.gif)

*Simulation with the tool geometries showing sheet metal thinning, stress, and strain.*

</div>

## Table of Contents

- [What's new in v3](#whats-new-in-v30)
- [Installation](#installation)
- [Download the dataset](#download-the-dataset)
- [Basic usage](#basic-usage)
- [PyTorch integration](#pytorch-integration)
- [Tutorials](#tutorials)
- [Version compatibility](#version-compatibility)
- [Citation](#citation)
- [Development](#development)

## What's new in v3

v3 is a major release because the dataset itself now ships with a [Croissant 1.1](https://docs.mlcommons.org/croissant/) manifest (`metadata.json`). The manifest is the single source of truth for the dataset schema: every HDF5 field, every CSV column, the SIM-KAx simulation provenance, and a set of task-specific views are declared once and consumed by both the package and external Croissant tools.

The Python surface was rewritten around it:

- `ddacs.load(data_dir)` parses the manifest and exposes published RecordSets (`process-parameters`, `field-map`, `simulation-provenance`, plus task views such as `springback-minimal`, `forming-snapshot`, `cutting-view`).
- `ddacs.open_h5(sim_id, data_dir)` reads any simulation by id without needing the zip extracted.
- `ddacs.add_view(ds, name, fields)` appends a custom view to the in-memory dataset.
- `DDACSDataset(view=...)` streams records of any view (published or custom) with worker-shard / DDP-safe partitioning, manifest-driven filtering, and graceful skip on partial downloads.
- The CLI default flipped to keep zips intact (`mlcroissant` reads HDF5 members in place); `--extract` and `--remove-zip` opt in to the loose-HDF5 layout.

The v2 helpers (`iter_ddacs`, `count_available_simulations`, the data extraction utilities) are removed. See the [tutorials](#tutorials) for the migration path.

## Installation

```bash
pip install ddacs
```

The PyTorch adapter is an optional extra. For hardware-specific PyTorch builds (CUDA, ROCm, MPS), install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/), then install the extra:

```bash
pip install 'ddacs[torch]'
```

## Download the dataset

```bash
# Small sample bundle (22 MB): manifest, CSV, and one simulation.
ddacs download --small -y

# Full release.
ddacs download

# Show available versions on DaRUS.
ddacs info
```

Files land in `./data` by default. The same path is the default for `ddacs.load(data_dir=...)` and `DDACSDataset(data_dir=...)`, so no further configuration is needed.

**CLI flags**

| Flag | Description |
|------|-------------|
| `VERSION` | Dataset version to download (default: `3.0`). |
| `--small` | Download the small sample bundle instead of the full release. |
| `--files FILE...` | Download only the listed files. |
| `--out PATH` | Output directory (default: `./data`). |
| `--extract` | Extract zip files in place after download. |
| `--remove-zip` | Delete the zip file after a successful extraction (requires `--extract`). |
| `-y, --yes` | Skip the confirmation prompt. |
| `--token TOKEN` | DaRUS API token (used to access draft versions). |

By default zip files are kept on disk and are *not* extracted; `mlcroissant` reads HDF5 members in place. Pass `--extract --remove-zip` to switch to a loose-HDF5 layout instead; see the [Loose HDF5 recipe](https://ddacs.readthedocs.io/en/latest/tutorials/loose-h5/) for the matching iteration pattern.

## Basic usage

`ddacs.load` parses the Croissant manifest; `ddacs.open_h5` opens a single simulation in memory and returns an `h5py.File`.

```python
import ddacs

# Load the dataset manifest. Lists every published RecordSet.
ds = ddacs.load(data_dir="./data")
print([rs.id for rs in ds.metadata.record_sets])

# Open one simulation. OP10 carries the blank and the three tools.
with ddacs.open_h5(258864, data_dir="./data") as f:
    blank_thickness = f["OP10/blank/element_shell_thickness"][-1]
    print("final-timestep thickness:", blank_thickness.shape)
```

For custom RecordSets and slicing the field map, see [Build your own view](https://ddacs.readthedocs.io/en/latest/tutorials/views/). For mesh / point-cloud / vector plotting, see [Visualization](https://ddacs.readthedocs.io/en/latest/tutorials/visualization/).

## PyTorch integration

`DDACSDataset` is a `torch.utils.data.IterableDataset` over a Croissant view. It builds a `sim_id -> local zip` index at construction time and silently skips simulations whose zip is missing, so partial downloads stream cleanly.

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

ds = DDACSDataset(view="springback-minimal", data_dir="./data")
loader = DataLoader(ds, batch_size=16, num_workers=0)

for batch in loader:
    forming    = batch["op10_blank_node_displacement_forming"]
    springback = batch["op10_blank_node_displacement_springback"]
    # ... training step ...
    break
```

For filtering, train / val / test splits, shuffling, and the partial-download story, see [PyTorch training](https://ddacs.readthedocs.io/en/latest/tutorials/pytorch/).

## Tutorials

Five tutorials walk through the package end to end. Each one is published on Read the Docs as a [tutorial page](https://ddacs.readthedocs.io/en/latest/tutorials/) and shipped as an executable notebook under [`notebooks/`](./notebooks/) that reproduces every cell:

1. [Getting started](https://ddacs.readthedocs.io/en/latest/tutorials/getting-started/) - [`01_getting_started.ipynb`](./notebooks/01_getting_started.ipynb): install, download, first plot.
2. [Build your own view](https://ddacs.readthedocs.io/en/latest/tutorials/views/) - [`02_views.ipynb`](./notebooks/02_views.ipynb): `ddacs.add_view`, manifest inspection, SIM-KAx provenance.
3. [PyTorch training](https://ddacs.readthedocs.io/en/latest/tutorials/pytorch/) - [`03_pytorch.ipynb`](./notebooks/03_pytorch.ipynb): `DDACSDataset`, filters, train/val/test splits.
4. [Visualization](https://ddacs.readthedocs.io/en/latest/tutorials/visualization/) - [`04_visualization.ipynb`](./notebooks/04_visualization.ipynb): thickness, components, springback, vectors.
5. [Loose HDF5 recipe](https://ddacs.readthedocs.io/en/latest/tutorials/loose-h5/) - [`05_loose_h5.ipynb`](./notebooks/05_loose_h5.ipynb): pandas + `h5py` after `--extract --remove-zip`.

See [`notebooks/README.md`](./notebooks/README.md) for prerequisites and run instructions.

## Version compatibility

The `ddacs` package major version tracks the DaRUS dataset major version. The pairing is enforced by the Croissant manifest bundled with each release: a mismatched package version will fail to resolve the field map.

| Package | DaRUS dataset |
|---------|---------------|
| `ddacs 3.x` | [v3.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=3.0) and any future v3.x updates (current) |
| `ddacs 2.x` | [v1.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=1.0) and [v2.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=2.0) |

Pin the package major to the dataset major you target, for example `pip install 'ddacs~=3.0'` to stay on the v3 line.

## Citation

If you use this dataset or code in your research, please cite both the dataset and the paper:

```bibtex
@dataset{baum2025ddacs,
  title={Deep Drawing and Cutting Simulations Dataset},
  subtitle={FEM Simulations of a deep drawn and cut dual phase steel part},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2025},
  version={3.0},
  publisher={DaRUS},
  doi={10.18419/DARUS-4801},
  license={CC BY 4.0},
  url={https://doi.org/10.18419/DARUS-4801}
}

@article{heinzelmann2025benchmark,
  title={A Comprehensive Benchmark Dataset for Sheet Metal Forming: Advancing Machine Learning and Surrogate Modelling in Process Simulations},
  author={Heinzelmann, Pascal and Baum, Sebastian and Riedmueller, Kim Rouven and Liewald, Mathias and Weyrich, Michael},
  journal={MATEC Web of Conferences},
  volume={408},
  year={2025},
  pages={01090},
  doi={10.1051/matecconf/202540801090},
  url={https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html}
}
```

## Development

```bash
git clone https://github.com/BaumSebastian/DDACS.git
cd DDACS
pip install -e ".[dev]"
pre-commit install   # set up code formatting hooks
pytest               # run the full test suite
```
