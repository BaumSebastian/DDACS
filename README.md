<div align="center">
  <img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/icon/DDACS_small.png" width="150"/>
  <h1>Deep Drawing and Cutting Simulations (DDACS) Dataset</h1>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://readthedocs.org/projects/ddacs/badge/?version=latest)](https://ddacs.readthedocs.io)
[![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801)
[![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--4801-blue.svg)](https://doi.org/10.18419/DARUS-4801)
[![Paper](https://img.shields.io/badge/paper-MATEC%20Web%20Conf.-red.svg)](https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html)

A Python package for accessing and processing the [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801).
It includes a CLI for downloading datasets from DaRUS and a Python API for accessing simulation data with metadata.

**[Read the full documentation](https://ddacs.readthedocs.io)**

<div align="center">

![Thickness Distribution Example](https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/simulation_overview.gif)

*Simulation with the tool geometries and various additional information like sheet metal thinning, stress and strain.*

</div>

## Table of Contents

- [Installation](#installation)
- [Download Dataset](#download-dataset)
- [Basic Usage](#basic-usage)
- [PyTorch Integration](#pytorch-integration)
- [Citation](#citation)
- [Development](#development)

## Installation

```bash
pip install ddacs
```

For PyTorch integration, install PyTorch first following the [official instructions](https://pytorch.org/get-started/locally/) for your CUDA version.

## Download Dataset

Download the dataset using the `ddacs` CLI:

```bash
# Download full dataset (requires ~1TB storage)
ddacs download

# Download small test set for quick demos (requires ~50GB storage)
ddacs download --small

# Show dataset info and available versions
ddacs info
```

**Important:** The full dataset is approximately 1TB in size. Ensure you have sufficient storage space. The download may take several hours depending on your internet connection.

**Options:**
| Flag | Description |
| ------ | ------------- |
| `VERSION` | Dataset version to download (default: `2.0`) |
| `--small` | Download small test set for demos |
| `--out ./path` | Custom output directory (default: `./data`) |
| `--no-extract` | Skip extraction of zip files |
| `--keep-zip` | Keep zip files after extraction |
| `-y, --yes` | Skip confirmation prompt |

## Basic Usage

```python
from ddacs import iter_ddacs, count_available_simulations
import h5py
import numpy as np

# Path to DDACS dataset
data_dir = "./data"

# Count available simulations
count = count_available_simulations(data_dir)
print(f"Available simulations: {count}")

# Iterate over samples (skip_missing=True for partial downloads)
for sim_id, metadata, h5_path in iter_ddacs(data_dir, skip_missing=True):
    with h5py.File(h5_path, "r") as f:
        displacement = np.array(f["OP10"]["blank"]["node_displacement"])
        print(f"ID={sim_id}, shape={displacement.shape}")
    break
```

## PyTorch Integration

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

# Path to DDACS dataset
data_dir = "./data"

dataset = DDACSDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for sim_ids, metadata_batch, h5_paths in dataloader:
    # Your training code here
    break
```

For more examples, see the [full documentation](https://ddacs.readthedocs.io) and [`notebooks/dataset_demo.ipynb`](./notebooks/dataset_demo.ipynb).

## Citation

If you use this dataset or code in your research, please cite both the dataset and the paper:

```bibtex
@dataset{baum2025ddacs,
  title={Deep Drawing and Cutting Simulations Dataset},
  subtitle={FEM Simulations of a deep drawn and cut dual phase steel part},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2025},
  version={2.0},
  publisher={DaRUS},
  doi={10.18419/DARUS-4801},
  license={CC BY 4.0},
  url={https://doi.org/10.18419/DARUS-4801}
}

@article{heinzelmann2025benchmark,
  title={A Comprehensive Benchmark Dataset for Sheet Metal Forming: Advancing Machine Learning and Surrogate Modelling in Process Simulations},
  author={Heinzelmann, Pascal and Baum, Sebastian and Riedmüller, Kim Rouven and Liewald, Mathias and Weyrich, Michael},
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
pre-commit install  # Setup code formatting hooks
pytest              # Run tests
```
