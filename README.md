# Deep Drawing and Cutting Simulations (DDACS) Dataset
A python example for accessing and processing the [Deep Drawing and Cutting Simulations (DDACS) Dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801).
It includes functionality for downloading datasets wiht [`darus` package](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction) and accessing simulation data with metadata.

## Installation
Clone the repository and navigate into. Install the requirements and execute main.py for a basic example. As the dataset is kinda big (~ 1 TB), the download flag is set to `False` (See [config example](./config/config_template.yaml)).
```bash
git clone https://github.com/BaumSebastian/Deep-Drawing-and-Cutting-Simulations-Dataset.git simulation_dataset
cd simulation_dataset
pip install -r requirements.txt
```

## Basic Usage

To interact with the dataset, you can use the [`SimulationDataset`](src/simulation_dataset.py) class. Here’s an example of how to use the [`SimulationDataset`](src/simulation_dataset.py) class in [`main.py`](main.py):

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

if __name__ == "__main__":
    main()
```
With the h5py package, you can access all simulation data based on the file path. See [main.py](./main.py) for an example to access the 'blank' geoemtry.

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
