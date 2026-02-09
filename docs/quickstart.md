# Quick Start

## Download the Dataset

```bash
# Download full dataset (~1TB)
ddacs download

# Download small test set (~50GB)
ddacs download --small

# Show dataset info
ddacs info
```

## Basic Usage

```python
from ddacs import iter_ddacs, count_available_simulations
import h5py
import numpy as np

# Path to DDACS dataset
data_dir = "./data"

# Count available simulations
count = count_available_simulations(data_dir)
print(f"Available simulations: {count}")

# Iterate over samples (skip_missing=True for partial downloads)
for sim_id, metadata, h5_path in iter_ddacs(data_dir, skip_missing=True):
    with h5py.File(h5_path, "r") as f:
        displacement = np.array(f["OP10"]["blank"]["node_displacement"])
        print(f"ID={sim_id}, shape={displacement.shape}")
    break
```

## PyTorch Integration

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

# Path to DDACS dataset
data_dir = "./data"

dataset = DDACSDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for sim_ids, metadata_batch, h5_paths in dataloader:
    # Your training code here
    break
```

## Extract Mesh Data

```python
from ddacs.utils import extract_mesh, extract_element_thickness

# Path to DDACS dataset
data_dir = "./data"

# Get mesh vertices and faces (using simulation ID 16336)
vertices, faces = extract_mesh(f"{data_dir}/h5/16336.h5", "blank", timestep=-1)

# Get thickness values per element
thickness = extract_element_thickness(f"{data_dir}/h5/16336.h5", timestep=-1)
```
