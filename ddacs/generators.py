"""
Simple generator functions for DDACS data streaming.

This module provides lightweight generator functions for iterating over
DDACS simulation data without class overhead.
"""

from pathlib import Path
from typing import Generator, Tuple, Union, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def iter_ddacs(
    data_dir: Union[str, Path],
    h5_subdir: str = "h5",
    metadata_file: str = "metadata.csv",
) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Ultra-simple generator for streaming DDACS data.

    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        metadata_file: Name of the metadata CSV file (default: "metadata.csv").

    Yields:
        Tuple[int, np.ndarray, Path]: Simulation ID, metadata values array,
            and path to corresponding H5 file.

    Raises:
        FileNotFoundError: If the H5 directory or individual H5 files don't exist.

    Examples:
        >>> for sim_id, metadata, h5_path in iter_ddacs('/data/ddacs'):
        ...     print(f"Simulation {sim_id}: {h5_path}")

        >>> # Custom subdirectory
        >>> for sim_id, metadata, h5_path in iter_ddacs('/data/ddacs', h5_subdir='results'):
        ...     print(f"Processing {sim_id}")
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir

    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    metadata = pd.read_csv(metadata_path)

    for _, row in metadata.iterrows():
        sim_id = int(row["ID"])
        h5_path = h5_dir / f"{sim_id}.h5"

        if h5_path.exists():
            metadata_vals = np.asarray(row.values[1:], copy=False)  # Skip ID, no copy
            yield sim_id, metadata_vals, h5_path
        else:
            raise FileNotFoundError(f"H5 file not found: {h5_path}")


def iter_h5_files(
    data_dir: Union[str, Path], h5_subdir: str = "h5"
) -> Generator[Path, None, None]:
    """
    Minimal generator for H5 file paths only.

    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").

    Yields:
        Path: Absolute path to each H5 file found in the specified directory.

    Raises:
        FileNotFoundError: If the H5 directory doesn't exist.

    Examples:
        >>> for h5_path in iter_h5_files('/data/ddacs'):
        ...     print(f"Found H5 file: {h5_path.name}")

        >>> # Count all H5 files
        >>> h5_count = sum(1 for _ in iter_h5_files('/data/ddacs'))
        >>> print(f"Total H5 files: {h5_count}")

    Note:
        Yields all .h5 files found in the directory, regardless of metadata.
    """
    h5_dir = Path(data_dir) / h5_subdir

    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")

    for h5_file in h5_dir.glob("*.h5"):
        yield h5_file


def get_simulation_by_id(
    sim_id: int,
    data_dir: Union[str, Path],
    h5_subdir: str = "h5",
    metadata_file: str = "metadata.csv",
) -> Optional[Tuple[int, np.ndarray, Path]]:
    """
    Get a specific simulation by its ID.

    Args:
        sim_id: The simulation ID to retrieve.
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        metadata_file: Name of the metadata CSV file (default: "metadata.csv").

    Returns:
        Optional[Tuple[int, np.ndarray, Path]]: Simulation data if found, None otherwise.
            Tuple contains (simulation_id, metadata_values, h5_file_path).

    Raises:
        FileNotFoundError: If the H5 directory or metadata file don't exist.

    Examples:
        >>> sim_data = get_simulation_by_id(113525, '/data/ddacs')
        >>> if sim_data:
        ...     sim_id, metadata, h5_path = sim_data
        ...     print(f"Found simulation {sim_id}")

        >>> # Check if simulation exists
        >>> if get_simulation_by_id(999999, '/data/ddacs') is None:
        ...     print("Simulation not found")
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir

    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    row = metadata[metadata["ID"] == sim_id]

    if row.empty:
        return None

    row = row.iloc[0]
    h5_path = h5_dir / f"{sim_id}.h5"

    if not h5_path.exists():
        return None

    metadata_vals = np.asarray(row.values[1:], copy=False)  # Skip ID, no copy
    return sim_id, metadata_vals, h5_path


def sample_simulations(
    n: int,
    data_dir: Union[str, Path],
    h5_subdir: str = "h5",
    metadata_file: str = "metadata.csv",
) -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Randomly sample simulations from the dataset.

    Args:
        n: Number of simulations to sample.
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        metadata_file: Name of the metadata CSV file (default: "metadata.csv").

    Yields:
        Tuple[int, np.ndarray, Path]: Simulation ID, metadata values array,
            and path to corresponding H5 file.

    Raises:
        FileNotFoundError: If the H5 directory or metadata file don't exist.

    Examples:
        >>> # Sample 5 random simulations
        >>> for sim_id, metadata, h5_path in sample_simulations(5, '/data/ddacs'):
        ...     print(f"Sampled simulation {sim_id}")

        >>> # Convert to list for further processing
        >>> samples = list(sample_simulations(10, '/data/ddacs'))
        >>> print(f"Got {len(samples)} samples")
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir

    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Filter to only existing H5 files
    mask = metadata["ID"].apply(lambda sim_id: (h5_dir / f"{int(sim_id)}.h5").exists())
    available_metadata = metadata[mask]

    if len(available_metadata) == 0:
        logger.warning("No simulations with existing H5 files found")
        return

    # Sample the requested number (or all available if less)
    n_sample = min(n, len(available_metadata))
    sampled = available_metadata.sample(n=n_sample)

    for _, row in sampled.iterrows():
        sim_id = int(row["ID"])
        h5_path = h5_dir / f"{sim_id}.h5"
        metadata_vals = np.asarray(row.values[1:], copy=False)  # Skip ID, no copy
        yield sim_id, metadata_vals, h5_path


def count_available_simulations(
    data_dir: Union[str, Path],
    h5_subdir: str = "h5",
    metadata_file: str = "metadata.csv",
) -> int:
    """
    Count available simulations (with existing H5 files).

    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        metadata_file: Name of the metadata CSV file (default: "metadata.csv").

    Returns:
        int: Number of simulations with existing H5 files.

    Raises:
        FileNotFoundError: If the H5 directory or metadata file don't exist.

    Examples:
        >>> count = count_available_simulations('/data/ddacs')
        >>> print(f"Dataset contains {count} available simulations")
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir

    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory not found: {h5_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    mask = metadata["ID"].apply(lambda sim_id: (h5_dir / f"{int(sim_id)}.h5").exists())
    return mask.sum()


if __name__ == "__main__":
    enumerate = iter_ddacs("data")
    print("run")
