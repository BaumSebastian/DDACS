from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
import warnings


class DDACSDataset(Dataset):
    def __init__(
        self, data_dir, h5_subdir="h5", metadata_file="metadata.csv", transform=None
    ):
        """
        Initializes FEM Dataset. The dataset is a basic implementation for getting metadata and hdf5 files.

        :param data_dir: The root directory of the dataset.
        :type data_dir: str
        :param h5_subdir: The relative directory in data_dir that contains the  h5 files. [Default 'data']
        :type h5_subdir: str
        :param metadata_file: The name of the metadata file in data_dir. [Default: 'metadata.csv']
        :type metadata_file: str
        :param transform: The transformation applied to the data. [Default: None]

        :raise FileNotFoundError: If the metadata file does not exist.
        :raise FileNotFoundError: If the sub directory does not exist.
        """

        self.data_dir = Path(data_dir)
        self._metadata_file_path = self.data_dir / metadata_file
        self._h5_subdir = self.data_dir / h5_subdir
        self.transform = transform

        if not self._h5_subdir.is_dir():
            raise FileNotFoundError(
                f"The data sub directory '{self._h5_subdir}' does not exist:"
            )

        self._metadata = pd.read_csv(self._metadata_file_path)

        # Check if all files exist in order to mitigate exception while execution
        self._metadata = self.__clean_metadata()
        self._length = len(self._metadata)

    def __clean_metadata(self) -> pd.DataFrame:
        """
        Checks if the files are available in the data directory.

        :return: A panda datafrme only with entries that exist.
        """
        # Filter for existing h5 files.
        mask = self._metadata["ID"].apply(
            lambda id: (self._h5_subdir / f"{id}.h5").is_file()
        )
        clean_metadata = self._metadata[mask]
        n_origin = len(self._metadata)
        n_clean = len(clean_metadata)

        if n_origin != n_clean:
            warnings.warn(
                f"Expected {n_origin} simulations but found {n_clean} in {self._h5_subdir}. Continues with {n_clean} simulations.",
                RuntimeWarning,
            )

        return clean_metadata

    def __len__(self) -> int:
        """Returns the number of data points in the dataset."""
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.array, str]:
        """
        Retrieves a data point and its corresponding label by index.

        :param idx: The index of the dataentry.
        :type idx: int
        :return: A tuple of the simulation id, the metadata parameter and the hdf5 file path.
        :rtype: Tuple[np.int64, np.array, str]
        """
        id = self._metadata["ID"].iat[idx]
        hdf5_file_path = self._h5_subdir / f"{id}.h5"
        parameter = self._metadata.iloc[idx].values[
            1:
        ]  # The first entry is the simulation ID.

        return (id, parameter, hdf5_file_path)

    def __str__(self) -> str:
        """Return a nice string representation of the dataset with tree view."""
        lines = [
            f"Deep Drawing and Cutting Simulations (DDACS) Dataset.",
            f"  Root directory: {self.data_dir}",
            f"  Metadata file: {self._metadata_file_path}",
            f"  Dataset length: {self._length} samples",
        ]

        lines.append("  Metadata:")
        for key in list(self._metadata)[1:]:  # Show all metadata beside id
            lines.append(f"    - {key}")

        return "\n".join(lines)
