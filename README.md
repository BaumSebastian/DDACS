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
    - [Core Installation (Lightweight)](#core-installation-lightweight)
    - [PyTorch Installation](#pytorch-installation)
    - [Examples Installation](#examples-installation)
    - [Full Installation](#full-installation)
    - [Adding to Your Project](#adding-to-your-project)
  - [Download Dataset](#download-dataset)
  - [Basic Usage](#basic-usage)
  - [Citation](#citation)
    - [Dataset Citation](#dataset-citation)
    - [Paper Citation](#paper-citation)
  - [Development Installation](#development-installation)

## Installation

**Note:** We recommend using [uv](https://docs.astral.sh/uv/) as a fast Python package installer and resolver. Simply replace `pip` with `uv pip` in the commands below.

### Core Installation (Lightweight)
For basic dataset access without machine learning dependencies:
```bash
pip install git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git
```

### Examples Installation (Recommended)
For interactive examples with PyTorch and visualization capabilities:
```bash
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[examples]"
```

### Full Installation (Development)
For all features including development and testing tools:
```bash
pip install "git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[full]"
```

### Adding to Your Project

Add DDACS to your existing project's `requirements.txt`:

```txt
# requirements.txt
# Your other dependencies...
numpy>=1.21.0
torch>=2.0.0

# Add DDACS dataset package
git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[pytorch]
```

Or for pip-tools users, add to `requirements.in`:
```txt
# requirements.in  
git+https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git[pytorch]
```

Then run:
```bash
pip-compile requirements.in  # generates requirements.txt
pip install -r requirements.txt
```

## Download Dataset
Download the dataset using the `darus-download` CLI command:

```bash
darus-download --url "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801" --path "./data"
```

**Important:** The dataset is approximately 1TB in size. Specify the `--path` parameter to choose a directory with sufficient storage space. The download may take several hours depending on your internet connection.

**Note:** For more download options and advanced usage, see the [`darus` package documentation](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction).

## Basic Usage

After installing the package, you can easily import and use the dataset classes:

```python
from ddacs import SimulationDataset

def main():

    data_dir = "./data" # The root directory of the dataset.
    h5_subdir = "h5" # The sub directory that contains the h5 files.

    simulation_dataset = SimulationDataset(data_dir, h5_subdir)
    print(simulation_dataset)

    # Fetch a sample from the dataset
    sim_id, metadata, h5_file_path = next(iter(simulation_dataset))
    
    print(
        "\n".join(
            [
                "Sample data entry.",
                f" - ID: {sim_id}",
                f" - Metadata: {metadata}",
                f" - h5 file path: {h5_file_path}",
            ]
        )
    )

    # Access the individual entry based on h5 structure.
    with h5py.File(h5_file_path, "r") as f:
        data = np.array(f["OP10"]["blank"]["node_displacement"])
    print(
        f"Example of pointcloud of 'blank' geometry for all ({data.shape[0]}) timesteps {data.shape}"
    )

if __name__ == "__main__":
    main()
```
With the h5py package, you can access all simulation data based on the file path. See [`main.py`](./main.py) for a simple example and [`examples/`](./examples/) for comprehensive tutorials including visualization and different access patterns.

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
uv pip install -e ".[dev]"  # or pip install -e ".[dev]"
```

This automatically installs the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) which provides the `darus-download` CLI command.
