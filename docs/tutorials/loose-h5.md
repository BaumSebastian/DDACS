# Loose HDF5 recipe

`ddacs download --extract --remove-zip` unpacks each zip in place and deletes the archive on success. The Croissant manifest path (`ddacs.load`, `ddacs.open_h5`, `DDACSDataset`) requires the zips, so once they are gone the only access path is `pandas` for the index and raw `h5py` for the simulations. That is what this recipe walks through.

The companion notebook at [`notebooks/05_loose_h5.ipynb`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks/05_loose_h5.ipynb) reproduces every cell below.

## 1. Download with extract + remove

From a shell, fetch the small bundle into a throwaway directory so the project-local `./data` stays intact:

```bash
ddacs download --small --extract --remove-zip --out /tmp/ddacs_loose -y
```

`--extract` unzips each archive in place; `--remove-zip` deletes the zip afterwards. See the [CLI reference](../cli.md#ddacs-download) for the full Rich-rendered output and the rest of the flags.

## 2. Inspect the loose layout

`metadata.json` and `process_parameters.csv` sit at the bundle root; the simulations live in `h5/<sim_id>.h5`.

```python
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

DATA_DIR = Path('/tmp/ddacs_loose')

for entry in sorted(DATA_DIR.iterdir()):
    if entry.is_dir():
        print(f'{entry.name}/')
        for sub in sorted(entry.iterdir())[:5]:
            print(f'  {sub.name}')
    else:
        print(entry.name)
```

Output:

```
h5/
  258864.h5
metadata.json
process_parameters.csv
```

## 3. Read `process_parameters.csv` with pandas

The CSV is the simulation index: one row per `sim_id`, with every process parameter exposed as a named column. Filter it in pandas before touching any HDF5 file : IO scales with the surviving rows, not with the full 32 466.

```python
params = pd.read_csv(DATA_DIR / 'process_parameters.csv')
print(f'rows: {len(params)}, columns: {list(params.columns)}')
print()
print(params.head(3).to_string())
```

Output:

```
rows: 32466, columns: ['index', 'geometry', 'curvature_radius', 'bottom_radius', 'wall_angle', 'material_scaling_factor', 'sheet_metal_thickness', 'friction_coefficient', 'blankholder_force', 'split', 'rddac']

   index     geometry  curvature_radius  bottom_radius  wall_angle  material_scaling_factor  sheet_metal_thickness  friction_coefficient  blankholder_force  split  rddac
0  16039  rectangular              30.0            5.0        10.0                      0.9                   0.95                  0.05           100000.0  train  False
1  16040  rectangular              30.0            5.0        10.0                      0.9                   0.95                  0.06           100000.0  train  False
2  16041  rectangular              30.0            5.0        10.0                      0.9                   0.95                  0.07           100000.0  train  False
```

## 4. Iterate the loose files with `h5py`

Walk the (filtered) rows, build the path, skip simulations whose `.h5` is missing locally, and open each one with `h5py.File`. Below: take the concave subset, then read final-timestep blank thickness from every loose file that landed on disk. With the small bundle only `258864.h5` exists, so the loop yields one line.

```python
concave = params.query("geometry == 'concave'")
print(f'concave sims in CSV: {len(concave):>6d} of {len(params)}')

h5_dir = DATA_DIR / 'h5'
found = 0
for _, row in concave.iterrows():
    h5_path = h5_dir / f"{row['index']}.h5"
    if not h5_path.is_file():
        continue
    with h5py.File(h5_path, 'r') as f:
        thickness = f['OP10/blank/element_shell_thickness'][-1]
    print(f"  sim {row['index']:>6d}  thickness in range of {thickness.min():.3f} mm - {thickness.max():.3f} mm  (number of samples: {len(thickness)})")
    found += 1
print(f'\nopened {found} loose h5 file(s)')
```

Output:

```
concave sims in CSV:  10888 of 32466
  sim 258864  thickness in range of 0.886 mm - 1.161 mm  (number of samples: 11025)

opened 1 loose h5 file(s)
```

## Where to go next

- [Build your own view](views.md) and [PyTorch training](pytorch.md) cover the zipped-bundle path that the Croissant API supports.
- [HDF5 structure reference](../hdf5-structure.md) lists every dataset and attribute available inside each loose `.h5`.
