# Getting Started

This tutorial covers the basics of loading and exploring DDACS simulation data.

## Setup

First, install DDACS and download the dataset:

```bash
pip install ddacs
ddacs download --small
```

## Counting Simulations

Check how many simulations are available in your dataset:

```python
from ddacs import count_available_simulations

data_dir = "./data"
count = count_available_simulations(data_dir)
print(f"Available simulations: {count}")
```

Output:
```
Available simulations: 32071
```

## Iterating Over Simulations

Use `iter_ddacs` to loop through all simulations:

```python
from ddacs import iter_ddacs

for sim_id, metadata, h5_path in iter_ddacs("./data"):
    print(f"Simulation {sim_id}: {h5_path.name}")
    print(f"Metadata: {metadata}")
    break  # Just show first one
```

Output:
```
Simulation 16336: 16336.h5
Metadata: [1. 0. 0. 30. 0.9 0.05 0.95 100000.]
```

### Handling Partial Downloads

If you haven't downloaded all files, use `skip_missing=True`:

```python
for sim_id, metadata, h5_path in iter_ddacs("./data", skip_missing=True):
    print(f"Processing simulation {sim_id}")
```

!!! warning
    With `skip_missing=False` (default), a `FileNotFoundError` is raised
    if any H5 file referenced in metadata is missing.

## Exploring HDF5 Structure

Each simulation is stored as an HDF5 file. Use `display_structure` to see its contents:

```python
from ddacs import iter_ddacs
from ddacs.utils import display_structure

sim_id, metadata, h5_path = next(iter_ddacs("./data", skip_missing=True))
display_structure(h5_path, max_depth=2)
```

Output:
```
/
├── OP10/
│   ├── blank/
│   ├── die/
│   ├── punch/
│   ├── binder/
│   └── general/
└── OP20/
    └── blank/
```

## Accessing Specific Simulations

Get a specific simulation by ID:

```python
from ddacs import get_simulation_by_id

result = get_simulation_by_id(16336, "./data")
if result:
    sim_id, metadata, h5_path = result
    print(f"Found simulation {sim_id}")
else:
    print("Simulation not found")
```

Output:
```
Found simulation 16336
```

## Random Sampling

Sample random simulations for testing or validation:

```python
from ddacs import sample_simulations

for sim_id, metadata, h5_path in sample_simulations(5, "./data"):
    print(f"Sampled: {sim_id}")
```

Output:
```
Sampled: 24891
Sampled: 18432
Sampled: 31205
Sampled: 12847
Sampled: 29156
```

## Reading Simulation Data

Access the actual simulation data using h5py:

```python
import h5py
import numpy as np
from ddacs import iter_ddacs

for sim_id, metadata, h5_path in iter_ddacs("./data", skip_missing=True):
    with h5py.File(h5_path, "r") as f:
        displacement = np.array(f["OP10"]["blank"]["node_displacement"])
        print(f"Displacement shape: {displacement.shape}")
        # Shape: (timesteps, nodes, 3)
    break
```

Output:
```
Displacement shape: (4, 11041, 3)
```

## Understanding Metadata

The metadata array contains process parameters (excluding the ID column):

| Index | Parameter | Description | Range |
|-------|-----------|-------------|-------|
| 0-2 | GEO_R/V/X | Geometry type (one-hot) | 0 or 1 |
| 3 | RAD | Characteristic radius | 30-150 mm |
| 4 | MAT | Material scaling factor | 0.9-1.1 |
| 5 | FC | Friction coefficient | 0.05-0.15 |
| 6 | SHTK | Sheet thickness | 0.95-1.0 mm |
| 7 | BF | Blank holder force | 100k-500k N |

```python
from ddacs import iter_ddacs

for sim_id, metadata, h5_path in iter_ddacs("./data", skip_missing=True):
    print(f"Simulation {sim_id}")
    print(f"  Geometry: R={metadata[0]}, V={metadata[1]}, X={metadata[2]}")
    print(f"  Radius: {metadata[3]} mm")
    print(f"  Material: {metadata[4]}")
    print(f"  Friction: {metadata[5]}")
    print(f"  Thickness: {metadata[6]} mm")
    print(f"  Holder Force: {metadata[7]} N")
    break
```

Output:
```
Simulation 16336
  Geometry: R=1.0, V=0.0, X=0.0
  Radius: 30.0 mm
  Material: 0.9
  Friction: 0.05
  Thickness: 0.95 mm
  Holder Force: 100000.0 N
```

## Next Steps

- Learn about [Visualization](visualization.md) to create plots
- See the [Dataset Overview](../dataset.md) for detailed physics background
- Check the [HDF5 Structure](../hdf5-structure.md) for all available fields
