from pathlib import Path
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from lasso.dyna import ArrayType, D3plot, FilterType
from typing import Union, Dict, Tuple

class dataset(Dataset):
    def __init__(
        self,
        root = None,
        sub_dir='data',
        partition = 'train',
        split=[.75, .1, .15],
    ):
        self.root = Path(root)
        self.metadata_path = root / Path('metadata.csv')
        self.data_dir = self.root / sub_dir
        
        if sum(split) != 1:
            raise ValueError("The sum over split needs to be 1.")

        if not os.path.isdir(root):
            raise ValueError(f"The directory '{self.root}' does not exist")

        metadata = pd.read_csv(self.metadata_path)

        if "ID" not in metadata.columns:
            raise ValueError("Metadata file must contain an 'id' column.")

        return metadata

        self.test_data, self.test_data, self.val_data = self.__split_data(split)

        if partition == 'train':
            self.metadata = self.trailen_dataset
        elif partition == 'test':
            self.metadata = self.test_data
        else:
            self.metadata = self.test_data

    def __split_data(self, fraction):
        # Shuffle the dataset
        shuffled_metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the dataset into train and test sets
        train_size = int(len(shuffled_metadata) * fraction)
        trailen_dataset = shuffled_metadata.iloc[:train_size]
        test_data = shuffled_metadata.iloc[train_size:]
        return trailen_dataset, test_data


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
    
    def __part_mapping(self):
        return {
            'Blank': 1,
            'Die': 2,
            'Punch': 3, 
            'Binder': 4
        }

    def __getitem__(self, idx):
        """"""
        metadata = self.metadata.iloc[idx].values

        # Get the Id and process/geometric parameters
        id = metadata[0]

        parameter_idx = 0 if self.return_id else 1
        parameters = metadata[-4:]

        # Get the label and data
        label = self.__get_label(id)
        data = self.__get_data(id)

        # Process data and label
        label = np.sum(label[3:], axis=1)
        label /= np.sum(label)

        return (label, (parameters, data))

    def __get_data(self, id):

        state_filter = {0}

        # The filter which arrays to load. Can increase execution speed.
        state_array_filter = [
            ArrayType.node_displacement,  # Position of the nodes in each simulationstep
        ]  # If Nothing then set to None

        path = os.path.join(self.root, str(id), "SPP2422_OP10.d3plot")
        d3plot = self._get_d3plot(path, state_array_filter, state_filter)
        return self.extract_point_clouds(self.__part_mapping()['Punch'], d3plot, 0)
        
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
    def extract_point_clouds(self, id, d3plot: D3plot, timestep : int) -> Dict[str, np.ndarray]:
        """
        Extract point clouds from d3plot file.
        
        :param d3plot_path: Path to the d3plot file
        :return: Dictionary of point clouds for different element types
        """
        
        # Get the nodes of the element            
        node_mask = d3plot.get_part_filter(FilterType.NODE, id, for_state_array=True)
        return d3plot.arrays[ArrayType.node_displacement][:,node_mask][timestep]

    def __str__(self) -> str:
        # To get a good fitting intendation.
        l_root_parent= len(f'Root location: {os.path.dirname(self.root.rstrip("/"))}')
        return (
            "Simulation Dataset\n"
            + f"    Number of datapoints: {self.__len__()}\n"
            + f"    Partition: {self.partition}\n"
            + f"    Split (for training): {self.fraction}\n"
            + f"    Root location: {self.root}\n"
            + f"    {' ' * l_root_parent}├── metadata.csv.gz\n"
            + f"    {' ' * l_root_parent}└── label.csv.gz\n"
        )

class fem_dataset(Dataset):
    def __init__(
        self,
        root=None,
        sub_dir="data",
        partition="train",
        split=[0.75, 0.1, 0.15],
    ):
        self.root = Path(root)
        self.metadata_path = root / Path('metadata.csv')
        self.data_dir = self.root / sub_dir

        if sum(split) != 1:
            raise ValueError("The sum over split needs to be 1.")

        if not os.path.isdir(root):
            raise ValueError(f"The directory '{self.root}' does not exist")

        self.metadata = pd.read_csv(self.metadata_path)

        if "ID" not in self.metadata.columns:
            raise ValueError("Metadata file must contain an 'id' column.")

        self.test_data, self.test_data, self.val_data = self.__split_data(split)

        if partition == 'train':
            self.metadata = self.trailen_dataset
        elif partition == 'test':
            self.metadata = self.test_data
        else:
            self.metadata = self.test_data

    def __split_data(self, fraction):
        # Shuffle the dataset
        shuffled_metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the dataset into train and test sets
        train_size = int(len(shuffled_metadata) * fraction)
        trailen_dataset = shuffled_metadata.iloc[:train_size]
        test_data = shuffled_metadata.iloc[train_size:]
        return trailen_dataset, test_data

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

    def __getitem__(self, idx):
        """"""
        metadata = self.metadata.iloc[idx].values

        # Get the Id and process/geometric parameters
        id = metadata[0]

        parameter_idx = 0 if self.return_id else 1
        parameters = metadata[-4:]

        # Get the label and data
        label = self.__get_label(id)
        data = self.__get_data(id)

        # Process data and label
        label = np.sum(label[3:], axis=1)
        label /= np.sum(label)

        return (label, (parameters, data))

    def __str__(self) -> str:
        # To get a good fitting intendation.
        l_root_parent= len(f'Root location: {os.path.dirname(self.root.rstrip("/"))}')
        return (
            "Simulation Dataset\n"
            + f"    Number of datapoints: {self.__len__()}\n"
            + f"    Partition: {self.partition}\n"
            + f"    Split (for training): {self.fraction}\n"
            + f"    Root location: {self.root}\n"
            + f"    {' ' * l_root_parent}├── metadata.csv.gz\n"
            + f"    {' ' * l_root_parent}└── label.csv.gz\n"
        )

if __name__ == "__main__":

    # Change the working directory to main directory in order to import the modules.
    os.chdir("../")
    np.random.seed(42)

#   # Pass the base directory as cmd argument
#   root = "/mnt/sim_data/darus/"

#   train_ds = dataset(root, sub_dir='hdf_fld', partition='train')
#   print(train_ds)

#   # Get first data entry
#   y, X = train_ds[0]
#   p, x = X
#   print("\nSampel data entry (type - shape).")
#   print(f"Label: {type(y)} - {y.shape}")
#   print(f"Parameter: {type(p)} - {p.shape}")
#   print(f"Data: {type(x)} - {x.shape}") 

    trai_ds = fem_dataset("/mnt/sim_data/darus/", sub_dir="hdf5_fld")
