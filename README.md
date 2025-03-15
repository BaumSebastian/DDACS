# Deep Drawing and Cutting Simulations (DDACS) Dataset
A python example for accessing and processing the [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801).
It includes functionality for downloading datasets wiht [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) and accessing simulation data with metadata.

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Download Dataset](#download-dataset)
- [License](#license)

## Installation
Clone the repository and navigate into. Install the requirements into your environment.
```bash
git clone https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git simulation_dataset
cd simulation_dataset
pip install -r requirements.txt
```
As pytorch relies on your hardware, please install it by yourself based on [pytorch installation guide](https://pytorch.org/get-started/locally/).

## Configuration
The configuration is stored in [`./config/config_template.yaml`](./config/config_template.yaml) and should be adjusted before executing [`main.py`](./main.py).
```yaml
data_dir: "./data"  # Root directory of the dataset
h5_subdir: "h5"  # Data directory inside the root directory
download_dataset: True  # Indicates if the dataset should be downloaded
dataset_url: "https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801"  # URL of the dataset needed if download = True
```
As the dataset is quite large (~ 1 TB) make sure to choose an appropriate `data_dir` with enough storage and write permissions. If you want to download the dataset not via [`main.py`](./main.py) but by yourself please read [section below](#download-dataset) or visit the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) for more information.

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
                "Sampel data entry.",
                f" - ID: {sim_id}",
                f" - Metadata: {metadata}",
                f" - h5 file path: {h5_file_path}",
            ]
        )
    )

    # Access the indvidual entry based on h5 structure.
    with h5py.File(h5_file_path, "r") as f:
        data = np.array(f["OP10"]["blank"]["node_displacement"])
    print(
        f"Example of pointcloud of 'blank' geometry for all ({data.shape[0]}) timesteps {data.shape}"
    )

if __name__ == "__main__":
    main()
```
With the h5py package, you can access all simulation data based on the file path. See [`main.py`](./main.py) for an example to access the 'blank' geoemtry.

### Download Dataset

Use the [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) to download the dataset from DaRUS (See [utils.py](src/utils.py)):

```python
from darus import Dataset as DarusDataset


def download_dataset(dataset_url: str, data_dir: str) -> None:
    """
    Download a dataset from a given URL and save it to a local directory.

    :param dataset_url: The url to the dataset on DaRUS.
    :type dataset_url: str
    :param data_dir: The root directory to save the dataset.
    :type data_dir: str
    """
    ds = DarusDataset(dataset_url)
    ds.summary()
    ds.download(data_dir)
```

## License

This project is licensed under the MIT License.
