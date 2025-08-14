# Deep Drawing and Cutting Simulations (DDACS) Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801)
[![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--4801-blue.svg)](https://doi.org/10.18419/DARUS-4801)
[![Paper](https://img.shields.io/badge/paper-MATEC%20Web%20Conf.-red.svg)](https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html)

A python example for accessing and processing the [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801).
It includes functionality for downloading datasets with the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) CLI and accessing simulation data with metadata.

## Table of Contents
- [Deep Drawing and Cutting Simulations (DDACS) Dataset](#deep-drawing-and-cutting-simulations-ddacs-dataset)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Core Installation](#core-installation)
    - [Pytorch Installation](#pytorch-installation)
    - [Examples Installation](#examples-installation)
    - [Full Installation (Development)](#full-installation-development)
  - [Download Dataset](#download-dataset)
  - [Basic Usage](#basic-usage)
    - [Core Usage](#core-usage)
    - [PyTorch Usage](#pytorch-usage)
  - [Citation](#citation)
    - [Dataset Citation](#dataset-citation)
    - [Paper Citation](#paper-citation)
  - [Development Installation](#development-installation)

## Installation

**Note:** I recommend using [uv](https://docs.astral.sh/uv/) as a fast Python package installer and resolver. Simply replace `pip` with `uv pip` in the commands below.

### Core Installation

For basic dataset access without high weight module dependencies:

```bash
pip install git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git
```

### Pytorch Installation

For examples with PyTorch and visualization capabilities:

```bash
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[torch]"
```

### Examples Installation

For examples with PyTorch and visualization capabilities:

```bash
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[examples]"
```

### Full Installation (Development)

For all features including development and testing tools:

```bash
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[full]"
```

## Download Dataset

Download the dataset using the `darus-download` CLI command:

```bash
darus-download --url "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801" --path "./data"
```

**Important:** The dataset is approximately 1TB in size. Specify the `--path` parameter to choose a directory with sufficient storage space. The download may take several hours depending on your internet connection.

**Note:** For more download options and advanced usage, see the [`darus` package documentation](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction).

## Basic Usage

### Core Usage

For basic dataset iteration:

```python
import h5py
import numpy as np
from ddacs.core import DDACSIterator

# Initialize iterator
iterator = DDACSIterator("./data")
print(iterator)

# Iterate over first few samples
for i, (sim_id, metadata, h5_file_path) in enumerate(iterator):
    print(f"Sample {i+1}: ID={sim_id}, Path={h5_file_path}")
    
    # Access simulation data
    with h5py.File(h5_file_path, "r") as f:
        data = np.array(f["OP10"]["blank"]["node_displacement"])
        print(f"Data shape: {data.shape}")
    
    if i >= 2:  # Show first 3 samples
        break
```

### PyTorch Usage

For PyTorch-compatible dataset with DataLoader support:

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = DDACSDataset("./data")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Use in training loop
for batch_idx, (sim_ids, metadata_batch, h5_paths) in enumerate(dataloader):
    print(f"Batch {batch_idx}: {len(sim_ids)} samples")
    # Your training code here
    if batch_idx >= 2:  # Show first 3 batches
        break
```

See [`examples/`](./examples/) for comprehensive tutorials including visualization and advanced usage patterns.

## Citation

If you use this dataset or code in your research, please cite both the dataset and the paper:

### Dataset Citation

```bibtex
@dataset{baum2025ddacs,
  title={Deep Drawing and Cutting Simulations Dataset},
  subtitle={FEM Simulations of a deep drawn and cut dual phase steel part},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2025},
  version={1.0},
  publisher={DaRUS},
  doi={10.18419/DARUS-4801},
  license={CC BY 4.0},
  url={https://doi.org/10.18419/DARUS-4801}
}
```

### Paper Citation

```bibtex
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

## Development Installation

For developers who want to contribute to this project:

```bash
git clone https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git DDACS
cd DDACS

# Install in editable mode with development dependencies
pip install -e ".[dev]"  # or pip install -e ".[dev]"
```

This automatically installs the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) which provides the `darus-download` CLI command.
