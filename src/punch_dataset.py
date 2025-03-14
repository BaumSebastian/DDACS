import numpy as np
import yaml
import h5py
import pandas as pd
from typing import Tuple

from base_fem_dataset import BaseFEMDataset


class PunchDataset(BaseFEMDataset):
    def __init__(
        self,
        root,
        data_dir="data",
        metadata_file="metadata.csv",
        transform=None,
        label_file="label.csv",
    ):
        """
        Initializes FEM Dataset. The dataset is a basic implementation for getting metadata and hdf5 files.

        :param root: The root directory of the dataset.
        :type root: str
        :param data_dir: The directory in the root directory that contains the  h5py files. [Default 'data']
        :type data_dir: str
        :param metadata_file: The name of the metadata file in the root directory. [Default: 'metadata.csv']
        :type metadata_file: str
        :param transform: The transformation applied the data. [Default: None]

        :raise FileNotFoundError: If the metadata file does not exist.
        :raise FileNotFoundError: If the sub directory does not exist.
        """
        super().__init__(root, data_dir, metadata_file, transform)

        self._label_file_path = self.root / label_file
        self._label = pd.read_csv(self._label_file_path)

    def __getitem__(self, idx: int) -> Tuple[np.array, Tuple[np.array, np.array]]:
        """
        Retrieves a data point and its corresponding label by index.

        :param idx: The index of the dataentry.
        :type idx: int
        :return: A tuple of the metadata parameter and the punch geometry as point cloud.
        :rtype: Tuple[np.int64, np.array, str]
        """
        (id, parameter, file) = super().__getitem__(idx)
        label = self._label[self._label["ID"] == id].values[0, 1:]

        with h5py.File(file, "r") as f:
            data = np.array(f["OP10"]["punch"]["node_coordinates"])

        if self.transform:
            data = self.transform(data)

        return (label, (parameter, data))


if __name__ == "__main__":

    config_file_path = r"../config/config.yaml"
    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    root = config["root"]
    data_dir = config["data_dir"]

    ds = PunchDataset(root, data_dir)

    print(ds)
    y, (p, X) = next(iter(ds))
    print(
        "\nSampel data entry.\n"
        + f"Label: {y}\n"
        + f"Metadata: {p}\n"
        + f"Point Cloud Data Shape: {X.shape}"
    )
