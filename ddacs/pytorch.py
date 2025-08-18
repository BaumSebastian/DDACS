"""
PyTorch integration for DDACS dataset.

This module provides PyTorch-compatible Dataset class for machine learning
workflows with DDACS simulation data.
"""

from pathlib import Path
from typing import Tuple, Union
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

try:
    from torch.utils.data import Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for DDACSDataset. Install with: pip install torch or use DDACSIterator for PyTorch-free access."
    ) from exc


class DDACSDataset(Dataset):
    """
    PyTorch-compatible DDACS dataset for machine learning workflows.

    Raises:
        FileNotFoundError: If the H5 directory or metadata file don't exist.
        ImportError: If PyTorch is not installed.

    Examples:
        >>> dataset = DDACSDataset('/data/ddacs')
        >>> print(len(dataset))

        >>> # Use with PyTorch DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        h5_subdir: str = "h5",
        metadata_file: str = "metadata.csv",
        transform=None,
    ):
        """
        Initialize PyTorch-compatible DDACS dataset.

        Args:
            data_dir: Root directory of the dataset.
            h5_subdir: Subdirectory containing H5 files (default: "h5").
            metadata_file: Name of metadata CSV file (default: "metadata.csv").
            transform: Optional transform to apply to metadata.

        Raises:
            FileNotFoundError: If the H5 directory or metadata file don't exist.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> print(f"Dataset has {len(dataset)} samples")

            >>> # Custom subdirectory and transform
            >>> dataset = DDACSDataset('/data/ddacs', h5_subdir='results', transform=my_transform)
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
        """Filter metadata to only include entries with existing H5 files.

        Returns:
            pd.DataFrame: Filtered metadata containing only simulations with existing H5 files.

        Warns:
            UserWarning: If some simulations in metadata don't have corresponding H5 files.

        Examples:
            >>> # Called automatically during initialization
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> # Warns if some H5 files are missing
        """
        mask = self._metadata["ID"].apply(
            lambda sim_id: (self._h5_dir / f"{int(sim_id)}.h5").exists()
        )
        filtered = self._metadata[mask]

        n_original = len(self._metadata)
        n_filtered = len(filtered)
        if n_original != n_filtered:
            logger.warning(
                f"WARNING: Found {n_filtered}/{n_original} simulations with existing H5 files"
            )

        return filtered

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of available simulation samples.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> print(len(dataset))
        """
        return len(self._metadata)

    def get_metadata_columns(self) -> list[str]:
        """Get list of metadata column names (excluding ID).

        Returns:
            list[str]: Column names from metadata CSV, excluding the ID column.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> columns = dataset.get_metadata_columns()
            >>> print(f"Available parameters: {columns}")
        """
        return self._metadata.columns[1:].tolist()

    def get_metadata_descriptions(self) -> dict[str, str]:
        """Get explanations for abbreviated metadata column names.

        Returns:
            dict[str, str]: Mapping of column abbreviations to their descriptions.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> descriptions = dataset.get_metadata_descriptions()
            >>> for col, desc in descriptions.items():
            ...     print(f"{col}: {desc}")
        """
        descriptions = {
            "GEO_R": "Rectangular geometry (1=yes, 0=no)",
            "GEO_V": "Concave geometry (1=yes, 0=no)", 
            "GEO_X": "Convex geometry (1=yes, 0=no)",
            "RAD": "Characteristic radius [30-150 mm]",
            "MAT": "Material scaling factor [0.9-1.1]",
            "FC": "Friction coefficient [0.05-0.15]",
            "SHTK": "Sheet thickness [0.95-1.0 mm]",
            "BF": "Blank holder force [100,000-500,000 N]",
        }

        available_columns = self.get_metadata_columns()
        return {
            col: descriptions.get(col, "Unknown parameter") for col in available_columns
        }

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray, str]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Tuple[int, np.ndarray, str]: Simulation ID, metadata values array,
                and path to corresponding H5 file.

        Raises:
            IndexError: If idx is out of range.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> sim_id, metadata, h5_path = dataset[0]
            >>> print(f"Simulation {sim_id}: {h5_path}")
        """
        row = self._metadata.iloc[idx]
        sim_id = int(row["ID"])
        h5_path = self._h5_dir / f"{sim_id}.h5"
        metadata_vals = np.asarray(row.values[1:], copy=False)  # Skip ID, no copy

        if self.transform:
            metadata_vals = self.transform(metadata_vals)

        return sim_id, metadata_vals, str(h5_path)

    def __str__(self) -> str:
        """Return a formatted string representation of the dataset.

        Returns:
            str: Multi-line string showing dataset directory, number of samples,
                and metadata column names.

        Examples:
            >>> dataset = DDACSDataset('/data/ddacs')
            >>> print(dataset)
        """
        lines = [
            "DDACS Dataset (PyTorch)",
            f"  Directory: {self.data_dir}",
            f"  Samples: {len(self)}",
        ]

        if len(self._metadata) > 0:
            lines.append("  Metadata columns:")
            for col in self._metadata.columns[1:]:  # Skip ID
                lines.append(f"    - {col}")

        return "\n".join(lines)
