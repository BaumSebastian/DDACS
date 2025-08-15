"""
Core DDACS dataset access functionality.

This module provides the main DDACSIterator class for accessing DDACS simulation
data in a pure Python environment without ML framework dependencies.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Generator
import warnings


class DDACSIterator:
    """
    Lightweight iterator for DDACS simulation data.
    Pure Python implementation with no external ML dependencies.
    
    Examples:
        >>> iterator = DDACSIterator('/path/to/dataset')
        >>> print(f"Found {len(iterator)} simulations")
        >>> for sim_id, metadata, h5_path in iterator:
        ...     print(f"Processing simulation {sim_id}")
        
        >>> # Get specific simulation
        >>> sim_data = iterator.get_by_id(12345)
        >>> if sim_data:
        ...     sim_id, metadata, h5_path = sim_data
        
        >>> # Random sampling
        >>> for sim_id, metadata, h5_path in iterator.sample(5):
        ...     print(f"Sampled simulation {sim_id}")
    """
    
    def __init__(self, data_dir: Union[str, Path], h5_subdir: str = "h5", 
                 metadata_file: str = "metadata.csv"):
        """
        Initialize DDACS iterator.
        
        Args:
            data_dir: Root directory of the dataset
            h5_subdir: Subdirectory containing H5 files (default: "h5")
            metadata_file: Name of metadata CSV file (default: "metadata.csv")
            
        Examples:
            >>> iterator = DDACSIterator('/data/ddacs_dataset')
            >>> iterator = DDACSIterator('/data/ddacs_dataset', h5_subdir='simulations')
            
        Raises:
            FileNotFoundError: If the H5 directory or metadata file doesn't exist.
        """
        self.data_dir = Path(data_dir)
        self._h5_dir = self.data_dir / h5_subdir
        self._metadata_path = self.data_dir / metadata_file
        
        # Validate paths
        if not self._h5_dir.is_dir():
            raise FileNotFoundError(f"H5 directory not found: {self._h5_dir}")
        if not self._metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self._metadata_path}")
        
        # Load and filter metadata for existing files
        self._metadata = pd.read_csv(self._metadata_path)
        self._metadata = self._filter_existing_files()
        
    def _filter_existing_files(self) -> pd.DataFrame:
        """Filter metadata to only include entries with existing H5 files.
        
        Returns:
            pd.DataFrame: Filtered metadata containing only simulations with existing H5 files.
            
        Warns:
            UserWarning: If some simulations in metadata don't have corresponding H5 files.
        """
        mask = self._metadata["ID"].apply(
            lambda sim_id: (self._h5_dir / f"{int(sim_id)}.h5").exists()
        )
        filtered = self._metadata[mask]
        
        n_original = len(self._metadata)
        n_filtered = len(filtered)
        if n_original != n_filtered:
            warnings.warn(
                f"Found {n_filtered}/{n_original} simulations with existing H5 files",
                UserWarning
            )
        
        return filtered
    
    def __len__(self) -> int:
        """Return the number of available simulations.
        
        Returns:
            int: Number of simulations with existing H5 files.
        """
        return len(self._metadata)
    
    def __iter__(self) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
        """Iterate over all available simulations.
        
        Yields:
            Tuple[int, np.ndarray, Path]: Simulation ID, metadata values, and H5 file path.
        """
        for _, row in self._metadata.iterrows():
            sim_id = int(row["ID"])
            h5_path = self._h5_dir / f"{sim_id}.h5"
            metadata_vals = row.values[1:]  # Exclude ID column
            yield sim_id, metadata_vals, h5_path
    
    def get_by_id(self, sim_id: int) -> Optional[Tuple[int, np.ndarray, Path]]:
        """Retrieve a specific simulation by its ID.
        
        Args:
            sim_id: The simulation ID to retrieve.
            
        Returns:
            Optional[Tuple[int, np.ndarray, Path]]: Simulation data if found, None otherwise.
                Tuple contains (simulation_id, metadata_values, h5_file_path).
        """
        row = self._metadata[self._metadata["ID"] == sim_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        sim_id_int = int(row["ID"])
        h5_path = self._h5_dir / f"{sim_id_int}.h5"
        metadata_vals = row.values[1:]  # Exclude ID column
        return sim_id_int, metadata_vals, h5_path
    
    def sample(self, n: int = 1) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
        """Randomly sample simulations from the dataset.
        
        Args:
            n: Number of simulations to sample (default: 1).
               If n exceeds available simulations, returns all available.
               
        Yields:
            Tuple[int, np.ndarray, Path]: Simulation ID, metadata values, and H5 file path.
        """
        sampled = self._metadata.sample(n=min(n, len(self._metadata)))
        for _, row in sampled.iterrows():
            sim_id = int(row["ID"])
            h5_path = self._h5_dir / f"{sim_id}.h5"
            metadata_vals = row.values[1:]  # Exclude ID column
            yield sim_id, metadata_vals, h5_path
    
    def __str__(self) -> str:
        """Return a formatted string representation of the iterator.
        
        Returns:
            str: Multi-line string showing dataset directory, available simulations,
                and metadata column names.
        """
        lines = [
            f"DDACS Iterator",
            f"  Directory: {self.data_dir}",
            f"  Available simulations: {len(self)}",
        ]
        
        if len(self._metadata) > 0:
            lines.append("  Metadata columns:")
            for col in self._metadata.columns[1:]:  # Skip ID
                lines.append(f"    - {col}")
        
        return "\n".join(lines)