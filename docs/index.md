# DDACS

<div align="center">
  <img src="images/icon/DDACS.png" width="150"/>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801)
[![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--4801-blue.svg)](https://doi.org/10.18419/DARUS-4801)

The [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801) contains {{ simulation_count() }} sheet metal forming simulations exported from LS DYNA. The `ddacs` package provides a Croissant native interface: one CLI for the download, one Python module for access, and an optional PyTorch `IterableDataset` for training.

<div align="center">
  <img src="images/simulation_overview.gif" alt="Simulation Overview"/>
  <p><em>Simulation with tool geometries showing sheet metal thinning, stress and strain.</em></p>
</div>

## Get the data

```bash
pip install ddacs               # installation of package
ddacs download --small          # ~22 MB sample bundle into ./data
```

The package major version tracks the DaRUS dataset major version: `ddacs 3.x` reads dataset v3.0. See [Version compatibility](dataset.md#version-compatibility) for the full pairing.

## A first read

`ddacs.open_h5(sim_id)` opens one simulation and returns an `h5py.File`. The OP10 group carries the blank and the three tools (binder, die, punch); each has a node displacement tensor across the simulation timesteps.

```python
import ddacs

sim_id = 258864                  # one simulation in the small sample bundle

with ddacs.open_h5(sim_id) as f:
    blank  = f["OP10/blank/node_displacement"][-1]    # (n_nodes, 3) at last timestep
    binder = f["OP10/binder/node_displacement"][-1]
    die    = f["OP10/die/node_displacement"][-1]
    punch  = f["OP10/punch/node_displacement"][-1]

print("nodes:", blank.shape[0], binder.shape[0], die.shape[0], punch.shape[0])
# nodes: 11236 146 2047 1104
```

For training the [PyTorch tutorial](tutorials/pytorch.md) wraps the same data in `DDACSDataset`. For a guided tour start with [Getting Started](tutorials/getting-started.md).

## Citation

```bibtex
@dataset{baum2025ddacs,
  title={Deep Drawing and Cutting Simulations Dataset},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2025},
  publisher={DaRUS},
  doi={10.18419/DARUS-4801}
}

@article{heinzelmann2025benchmark,
  title={A Comprehensive Benchmark Dataset for Sheet Metal Forming},
  author={Heinzelmann, Pascal and Baum, Sebastian and others},
  journal={MATEC Web of Conferences},
  volume={408},
  year={2025},
  doi={10.1051/matecconf/202540801090}
}
```
