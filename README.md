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
    - [Using pip](#using-pip)
    - [Using uv](#using-uv)
  - [Download Dataset](#download-dataset)
  - [Basic Usage](#basic-usage)
  - [Citation](#citation)
    - [Dataset Citation](#dataset-citation)
    - [Paper Citation](#paper-citation)
  - [License](#license)

## Installation
Clone the repository and navigate into it:
```bash
git clone https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git DDACS & cd DDACS
```

### Using pip
```bash
pip install -r requirements.txt
```

### Using uv
```bash
uv pip install -r requirements.txt
```

This will automatically install the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) which provides the `darus-download` CLI command for downloading datasets.

## Download Dataset
Download the dataset using the `darus-download` CLI command:

```bash
darus-download --url "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801" --path "./data"
```

**Important:** The dataset is approximately 1TB in size. Specify the `--path` parameter to choose a directory with sufficient storage space. The download may take several hours depending on your internet connection.

**Note:** For more download options and advanced usage, see the [`darus` package documentation](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction).

## Basic Usage

To interact with the dataset, you can use the [`SimulationDataset`](src/simulation_dataset.py) class. Here’s an example of how to use the class in [`main.py`](main.py):

```python
from src import SimulationDataset

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
With the h5py package, you can access all simulation data based on the file path. See [`main.py`](./main.py) for an example to access the 'blank' geometry.

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

## License

This project is licensed under the MIT License.
