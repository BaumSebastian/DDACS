from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Generator
import warnings


class DDACSIterator:
    """
    Lightweight iterator for DDACS simulation data.
    Pure Python implementation with no external ML dependencies.
    """
    
    def __init__(self, data_dir: Union[str, Path], h5_subdir: str = "h5", 
                 metadata_file: str = "metadata.csv"):
        """
        Initialize DDACS iterator.
        
        Args:
            data_dir: Root directory of the dataset
            h5_subdir: Subdirectory containing H5 files (default: "h5")
            metadata_file: Name of metadata CSV file (default: "metadata.csv")
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
        """Filter metadata to only include entries with existing H5 files."""
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
        """Number of available simulations."""
        return len(self._metadata)
    
    def __iter__(self) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
        """Iterate over all simulations."""
        for _, row in self._metadata.iterrows():
            sim_id = int(row["ID"])
            h5_path = self._h5_dir / f"{sim_id}.h5"
            metadata_vals = row.values[1:]  # Exclude ID column
            yield sim_id, metadata_vals, h5_path
    
    def get_by_id(self, sim_id: int) -> Optional[Tuple[int, np.ndarray, Path]]:
        """Get simulation by ID."""
        row = self._metadata[self._metadata["ID"] == sim_id]
        if row.empty:
            return None
        
        row = row.iloc[0]
        sim_id_int = int(row["ID"])
        h5_path = self._h5_dir / f"{sim_id_int}.h5"
        metadata_vals = row.values[1:]  # Exclude ID column
        return sim_id_int, metadata_vals, h5_path
    
    def sample(self, n: int = 1) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
        """Randomly sample n simulations."""
        sampled = self._metadata.sample(n=min(n, len(self._metadata)))
        for _, row in sampled.iterrows():
            sim_id = int(row["ID"])
            h5_path = self._h5_dir / f"{sim_id}.h5"
            metadata_vals = row.values[1:]  # Exclude ID column
            yield sim_id, metadata_vals, h5_path
    
    def __str__(self) -> str:
        """String representation."""
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