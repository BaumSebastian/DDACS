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

# Count available simulations
count = count_available_simulations("./data")
print(f"Available simulations: {count}")

# Iterate over samples (skip_missing=True for partial downloads)
for sim_id, metadata, h5_path in iter_ddacs("./data", skip_missing=True):
    with h5py.File(h5_path, "r") as f:
        displacement = np.array(f["OP10"]["blank"]["node_displacement"])
        print(f"ID={sim_id}, shape={displacement.shape}")
    break
```

## PyTorch Integration

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

dataset = DDACSDataset("./data")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for sim_ids, metadata_batch, h5_paths in dataloader:
    # Your training code here
    break
```

## Extract Mesh Data

```python
from ddacs import extract_mesh, extract_element_thickness

# Get mesh vertices and faces
vertices, faces = extract_mesh("./data/h5/123456.h5", timestep=0)

# Get thickness values per element
thickness = extract_element_thickness("./data/h5/123456.h5", timestep=0)
```
