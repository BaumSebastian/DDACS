from pathlib import Path
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Union, Optional
import warnings


def iter_simulations(data_dir: Union[str, Path], h5_subdir: str = "h5", 
                    metadata_file: str = "metadata.csv") -> Generator[Tuple[np.int64, np.ndarray, Path], None, None]:
    """
    Fast generator that yields simulation data without PyTorch dependency.
    
    :param data_dir: The root directory of the dataset.
    :param h5_subdir: The sub directory that contains the h5 files.
    :param metadata_file: The name of the metadata file in data_dir.
    :yield: Tuple of (simulation_id, metadata_parameters, h5_file_path)
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir
    
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"The data sub directory '{h5_dir}' does not exist")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"The metadata file '{metadata_path}' does not exist")
    
    metadata = pd.read_csv(metadata_path)
    
    for _, row in metadata.iterrows():
        sim_id = row["ID"]
        h5_file_path = h5_dir / f"{sim_id}.h5"
        
        if h5_file_path.is_file():
            parameters = row.values[1:]  # Exclude ID column
            yield sim_id, parameters, h5_file_path
        else:
            warnings.warn(f"H5 file not found: {h5_file_path}", RuntimeWarning)


def iter_h5_files(data_dir: Union[str, Path], h5_subdir: str = "h5") -> Generator[Path, None, None]:
    """
    Ultra-fast generator that just yields H5 file paths without metadata loading.
    
    :param data_dir: The root directory of the dataset.
    :param h5_subdir: The sub directory that contains the h5 files.
    :yield: Path to H5 file
    """
    data_dir = Path(data_dir)
    h5_dir = data_dir / h5_subdir
    
    if not h5_dir.is_dir():
        raise FileNotFoundError(f"The data sub directory '{h5_dir}' does not exist")
    
    for h5_file in h5_dir.glob("*.h5"):
        yield h5_file


class SimulationIterator:
    """
    Lightweight iterator for simulation data without PyTorch dependency.
    Provides similar interface to SimulationDataset but with generator-based access.
    """
    
    def __init__(self, data_dir: Union[str, Path], h5_subdir: str = "h5", 
                 metadata_file: str = "metadata.csv"):
        """
        Initialize the simulation iterator.
        
        :param data_dir: The root directory of the dataset.
        :param h5_subdir: The sub directory that contains the h5 files.
        :param metadata_file: The name of the metadata file in data_dir.
        """
        self.data_dir = Path(data_dir)
        self.h5_subdir = h5_subdir
        self.metadata_file = metadata_file
        self._metadata_path = self.data_dir / metadata_file
        self._h5_dir = self.data_dir / h5_subdir
        
        if not self._h5_dir.is_dir():
            raise FileNotFoundError(f"The data sub directory '{self._h5_dir}' does not exist")
        
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"The metadata file '{self._metadata_path}' does not exist")
        
        # Load metadata once for length calculation
        self._metadata = pd.read_csv(self._metadata_path)
        # Filter for existing files
        mask = self._metadata["ID"].apply(
            lambda id: (self._h5_dir / f"{id}.h5").is_file()
        )
        self._metadata = self._metadata[mask]
        
        n_origin = len(pd.read_csv(self._metadata_path))
        n_clean = len(self._metadata)
        
        if n_origin != n_clean:
            warnings.warn(
                f"Expected {n_origin} simulations but found {n_clean} in {self._h5_dir}. "
                f"Continues with {n_clean} simulations.",
                RuntimeWarning,
            )
    
    def __len__(self) -> int:
        """Return the number of available simulations."""
        return len(self._metadata)
    
    def __iter__(self) -> Generator[Tuple[np.int64, np.ndarray, Path], None, None]:
        """Iterate over all simulations."""
        for _, row in self._metadata.iterrows():
            sim_id = row["ID"]
            h5_file_path = self._h5_dir / f"{sim_id}.h5"
            parameters = row.values[1:]  # Exclude ID column
            yield sim_id, parameters, h5_file_path
    
    def get_by_id(self, sim_id: int) -> Optional[Tuple[np.int64, np.ndarray, Path]]:
        """Get a specific simulation by ID."""
        row = self._metadata[self._metadata["ID"] == sim_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        h5_file_path = self._h5_dir / f"{sim_id}.h5"
        parameters = row.values[1:]  # Exclude ID column
        return sim_id, parameters, h5_file_path
    
    def sample(self, n: int = 1) -> Generator[Tuple[np.int64, np.ndarray, Path], None, None]:
        """Randomly sample n simulations."""
        sampled = self._metadata.sample(n=min(n, len(self._metadata)))
        for _, row in sampled.iterrows():
            sim_id = row["ID"]
            h5_file_path = self._h5_dir / f"{sim_id}.h5"
            parameters = row.values[1:]  # Exclude ID column
            yield sim_id, parameters, h5_file_path
    
    def __str__(self) -> str:
        """Return a string representation of the iterator."""
        lines = [
            f"Simulation Iterator (Lightweight)",
            f"  Root directory: {self.data_dir}",
            f"  Metadata file: {self._metadata_path}",
            f"  Available simulations: {len(self)} samples",
        ]
        
        if len(self._metadata) > 0:
            lines.append("  Metadata columns:")
            for col in list(self._metadata.columns)[1:]:  # Skip ID column
                lines.append(f"    - {col}")
        
        return "\n".join(lines)