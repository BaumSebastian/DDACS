"""
Simple generator functions for DDACS data streaming.

This module provides lightweight generator functions for iterating over
DDACS simulation data without class overhead.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Union


def iter_ddacs(data_dir: Union[str, Path], h5_subdir: str = "h5", 
               metadata_file: str = "metadata.csv") -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Ultra-simple generator for streaming DDACS data.
    
    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        metadata_file: Name of the metadata CSV file (default: "metadata.csv").
        
    Yields:
        Tuple[int, np.ndarray, Path]: Simulation ID, metadata values array,
            and path to corresponding H5 file.
            
    Examples:
        >>> for sim_id, metadata, h5_path in iter_ddacs('/data/ddacs'):
        ...     print(f"Simulation {sim_id}: {h5_path}")
        
        >>> # Custom subdirectory
        >>> for sim_id, metadata, h5_path in iter_ddacs('/data/ddacs', h5_subdir='results'):
        ...     print(f"Processing {sim_id}")
            
    Note:
        Only yields simulations that have existing H5 files. Missing files
        are silently skipped.
    """
    data_dir = Path(data_dir)
    metadata_path = data_dir / metadata_file
    h5_dir = data_dir / h5_subdir
    
    metadata = pd.read_csv(metadata_path)
    
    for _, row in metadata.iterrows():
        sim_id = row["ID"]
        h5_path = h5_dir / f"{sim_id}.h5"
        
        if h5_path.exists():
            metadata_vals = row.values[1:]  # Skip ID
            yield sim_id, metadata_vals, h5_path


def iter_h5_files(data_dir: Union[str, Path], h5_subdir: str = "h5") -> Generator[Path, None, None]:
    """
    Minimal generator for H5 file paths only.
    
    Args:
        data_dir: Root directory of the dataset.
        h5_subdir: Subdirectory containing H5 files (default: "h5").
        
    Yields:
        Path: Absolute path to each H5 file found in the specified directory.
        
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
    for h5_file in h5_dir.glob("*.h5"):
        yield h5_file