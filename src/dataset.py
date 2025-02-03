import sys
import os
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from lasso.dyna import ArrayType, D3plot, FilterType
from typing import Union, Dict
from enum import Enum


class LABEL(Enum):
    """The type of label that this dataset will use."""

    FULL = ("label_per_element",)
    TEST = ("label_per_part",)
    TEST2 = "label_per_part_halfed"


class dataset(Dataset):
    def __init__(
        self,
        sim_dir,
        label_dir,
        train: bool,
        transform,
        return_id: bool = False,
        features: Union[ArrayType] = None,
        geometries: Union[str] = None,
    ):
        """"""
        self.root = sim_dir
        self.metadata_dir = label_dir

        self.__validate_inputs()

        self.metadata = self.__load_metadata()
        self.return_id = return_id
        self.transform = transform

    def __validate_inputs(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Base directory '{self.root}' does not exist.")

    def __load_metadata(self):
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'.")

        metadata = pd.read_csv(metadata_path, compression="gzip")

        if "Simulation_ID" not in metadata.columns:
            raise ValueError("Metadata file must contain an 'id' column.")

        return metadata

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.metadata)

    def __get_label(self, id) -> np.array:
        # Load labels
        label_path = os.path.join(self.metadata_dir, f"{id}.csv.gz")
        occuring_labels = pd.read_csv(label_path)

        # Get all existing columns
        all_labels = self.__all_columns()

        # Create and fill all labels
        label = np.zeros((len(all_labels), len(occuring_labels)))

        # Fill label
        for l_idx, l in enumerate(all_labels):
            if l in occuring_labels:
                label[l_idx] = occuring_labels[l]

        return label

    def __all_columns(self):
        return (
            "Element Shell ID",
            "Major Strain",
            "Minor Strain",
            "Inadequate stretch",
            "Safe",
            "Risk of cracks",
            "Cracks",
            "Wrinkling tendency",
            "Wrinkles",
        )

    def __getitem__(self, idx):
        """"""
        metadata = self.metadata.iloc[idx].values

        # Get the Id and process/geometric parameters
        id = metadata[0]

        parameter_idx = 0 if self.return_id else 1
        parameters = metadata[parameter_idx:]

        # Get the label and data
        label = self.__get_label(id)
        data = self.__get_data(id)

        # Process data and label
        label = np.sum(label[3:], axis=1)

        return (label, (parameters, data))

    def __get_data(self, id):

        state_filter = None  # {0, -1}

        # The filter which arrays to load. Can increase execution speed.
        state_array_filter = [
            ArrayType.node_displacement,  # Position of the nodes in each simulationstep
        ]  # If Nothing then set to None

        path = os.path.join(self.root, str(id), "SPP2422_OP10.d3plot")
        d3plot = self._get_d3plot(path, state_array_filter, state_filter)
        return self.extract_point_clouds(d3plot, 0)['Punch']
        
    def _get_part_mapping(self, d3plot: D3plot) -> Dict:
        """
        Get the mapping of part names to their corresponding ids.

        :d3plot: the d3plot with the names and their ids.
        :return: a dictionary mapping part names to their corresponding part ids.
        """
        if not isinstance(d3plot, D3plot):
            raise TypeError()

        titles = d3plot.arrays[ArrayType.part_titles]
        ids = d3plot.arrays[ArrayType.part_titles_ids]

        # Clean the name and map the name to their ids
        clean_titles = map(lambda title: title.decode("utf8").strip(), titles)
        mappings = {title: id for title, id in zip(clean_titles, ids)}

        return mappings

    def _get_d3plot(self, file: str, state_array_filter, state_filter) -> D3plot:
        """
        Get the D3plot object from the file.

        :file: the path to the d3plot file.
        :return: the D3plot object.
        """

        if not os.path.exists(file):
            raise FileNotFoundError(file)

        return D3plot(
            str(file),
            state_array_filter=state_array_filter,
            state_filter=state_filter,
            buffered_reading=True,
        )
    def extract_point_clouds(self, d3plot: D3plot, timestep : int) -> Dict[str, np.ndarray]:
        """
        Extract point clouds from d3plot file.
        
        :param d3plot_path: Path to the d3plot file
        :return: Dictionary of point clouds for different element types
        """

        # Mapping of parts to ids:
        part_mapping = self._get_part_mapping(d3plot)
        
        # Get the nodes of the element
        point_clouds = {}

        for title in part_mapping:
            id = part_mapping[title]
            
            node_mask = d3plot.get_part_filter(FilterType.NODE, id, for_state_array=False)
            nodes = d3plot.arrays[ArrayType.node_displacement][:,node_mask][timestep]

            point_clouds[title] = nodes

        return point_clouds

if __name__ == "__main__":

    # Change the working directory to main directory in order to import the modules.
    os.chdir("../")

    # Pass the base directory as cmd argument
    sim_dir = "/mnt/nas/uncompressed_data/d3plot/"
    label_dir = "/mnt/nas/uncompressed_data/Heinzelmann/output_FLD/"
    metadata_path = "/home/sebastian/software/iddrg/data/metadata.csv.gz"

    ds = dataset(sim_dir, label_dir, False, metadata_path, None, True)
    label, (pc, parameter) = ds[0]