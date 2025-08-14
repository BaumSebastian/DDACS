from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional
import warnings

try:
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError(
        "PyTorch is required for DDACSDataset. "
        "Install with: pip install torch or use DDACSIterator for PyTorch-free access."
    )


class DDACSDataset(Dataset):
    """
    PyTorch-compatible DDACS dataset for machine learning workflows.
    
    This class inherits from torch.utils.data.Dataset and is designed
    for use with PyTorch DataLoaders and training loops.
    """
    
    def __init__(self, data_dir: Union[str, Path], h5_subdir: str = "h5", 
                 metadata_file: str = "metadata.csv", transform=None):
        """
        Initialize PyTorch-compatible DDACS dataset.
        
        Args:
            data_dir: Root directory of the dataset
            h5_subdir: Subdirectory containing H5 files (default: "h5")
            metadata_file: Name of metadata CSV file (default: "metadata.csv")
            transform: Optional transform to apply to data
        """
        self.data_dir = Path(data_dir)
        self._h5_dir = self.data_dir / h5_subdir
        self._metadata_path = self.data_dir / metadata_file
        self.transform = transform
        
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
        """Number of samples in the dataset."""
        return len(self._metadata)
    
    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, Path]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (simulation_id, metadata_values, h5_file_path)
        """
        row = self._metadata.iloc[idx]
        sim_id = int(row["ID"])
        h5_path = self._h5_dir / f"{sim_id}.h5"
        metadata_vals = row.values[1:]  # Exclude ID column
        
        if self.transform:
            metadata_vals = self.transform(metadata_vals)
            
        return sim_id, metadata_vals, h5_path
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"DDACS Dataset (PyTorch)",
            f"  Directory: {self.data_dir}",
            f"  Samples: {len(self)}",
        ]
        
        if len(self._metadata) > 0:
            lines.append("  Metadata columns:")
            for col in self._metadata.columns[1:]:  # Skip ID
                lines.append(f"    - {col}")
        
        return "\n".join(lines)