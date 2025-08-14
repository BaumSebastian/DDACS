from pathlib import Path
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Union


def iter_ddacs(data_dir: Union[str, Path], h5_subdir: str = "h5", 
               metadata_file: str = "metadata.csv") -> Generator[Tuple[int, np.ndarray, Path], None, None]:
    """
    Ultra-simple generator for streaming DDACS data.
    
    Args:
        data_dir: Root directory of the dataset
        h5_subdir: Subdirectory with H5 files
        metadata_file: Metadata CSV filename
        
    Yields:
        Tuple of (simulation_id, metadata_values, h5_file_path)
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
        data_dir: Root directory of the dataset
        h5_subdir: Subdirectory with H5 files
        
    Yields:
        Path to each H5 file
    """
    h5_dir = Path(data_dir) / h5_subdir
    for h5_file in h5_dir.glob("*.h5"):
        yield h5_file