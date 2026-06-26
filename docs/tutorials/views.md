# Build your own view

`ddacs.add_view` appends a custom `RecordSet` to the in memory dataset. The published manifest on DaRUS stays unchanged; only your in process copy grows.

The companion notebook at [`notebooks/02_views.ipynb`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks/02_views.ipynb) reproduces every cell below.

## 1. Load the dataset

```python
DATA_DIR = './data'      # repository root, or '../data' from notebooks/
sim_id   = 258864

import ddacs

ds = ddacs.load(data_dir=DATA_DIR)
print([rs.id for rs in ds.metadata.record_sets])
```

Output:

```
['process-parameters', 'field-map', 'simulation-provenance',
 'springback-minimal', 'springback-prediction',
 'forming-snapshot', 'cutting-view']
```

The dataset already ships several published RecordSets (`springback-minimal`, `forming-snapshot`, ...). When none of them fits a particular task, build your own.

## 2. Append a custom RecordSet

```python
ddacs.add_view(
    ds,
    "my-view",
    fields={
        "trajectory": ("op10_blank_node_displacement", None),       # all 4 timesteps
        "forming":    ("op10_blank_node_displacement", 2),          # one timestep
        "springback": ("op10_blank_node_displacement", [2, 3]),     # subset of timesteps
        "thickness":  "op10_blank_element_shell_thickness",         # whole field, shortcut form
    },
)
```

The first argument is the loaded dataset, the second is the new RecordSet's identifier, and `fields` maps an alias to a field-map field plus an optional timestep selection.

Field IDs (such as `op10_blank_node_displacement`) come from the field-map RecordSet declared in the Croissant manifest. The complete list lives in the [HDF5 structure reference](../hdf5-structure.md#complete-field-reference).

## `fields` value shapes

| Value | Result |
|-------|--------|
| `"field_id"` | whole field (shortcut) |
| `("field_id", None)` | whole field (explicit) |
| `("field_id", int)` | one timestep |
| `("field_id", [int, int, ...])` | subset of timesteps |

Behind the scenes a JSONPath transform is attached to each field that requires slicing: `("...", 2)` becomes `$[2]`, `("...", [2, 3])` becomes `$[2,3]`. The transform is applied at iteration time, so memory and IO scale with the timesteps actually requested, not with the full HDF5 array.

## 3. Inspect the new view's fields

```python
view = next(rs for rs in ds.metadata.record_sets if rs.id == "my-view")
for f in view.fields:
    transforms = [t.json_path for t in (f.source.transforms or [])]
    print(f"  {f.id:30s} <- {f.source.uuid:60s} transforms={transforms}")
```

Output:

```
my-view/trajectory  <- field-map/op10_blank_node_displacement       transforms=[]
my-view/forming     <- field-map/op10_blank_node_displacement       transforms=['$[2]']
my-view/springback  <- field-map/op10_blank_node_displacement       transforms=['$[2,3]']
my-view/thickness   <- field-map/op10_blank_element_shell_thickness transforms=[]
```

`f.source.uuid` is the field-map field that supplies the data, `transforms` records the timestep selection.

## 4. Iterate records

Each record is a dict keyed by the aliases declared in `add_view`, with values already sliced according to the JSONPath transform. `forming` comes out as a 2D `(n_nodes, 3)` array (the `$[2]` transform dropped the timestep axis) while `trajectory` keeps the full `(4, n_nodes, 3)` shape.

```python
import numpy as np

for rec in ds.records("my-view"):
    for k, v in rec.items():
        arr = np.asarray(v)
        print(f"  {k:20s} shape={arr.shape} dtype={arr.dtype}")
    break
```

Output (with the small bundle, the first missing zip aborts the iteration):

```
records stopped: GenerationError: An error occured during the sequential
generation of the dataset, more specifically during the operation
Download(106921_109120.zip)
```

!!! note
    `mlcroissant` walks every zip referenced by the FileSet and aborts at the first missing one. With the small bundle only `258864.zip` is on disk, so the snippet above will raise a `GenerationError` before yielding anything. The iteration works in full only after `ddacs download` (without `--small`) has pulled the complete release.

    For partial downloads, [`DDACSDataset`](pytorch.md) builds a `sim_id -> local zip` index at construction time and silently skips simulations whose zip is missing.

## 5. Read other RecordSets

Two RecordSets ship with the published manifest that are useful even outside of view building:

- **`process-parameters`** is the simulation index : one record per simulation, with every column of `process_parameters.csv` exposed as a named field. It is sourced from the CSV (not from the h5 zips), so the records iterator works regardless of which zips are downloaded locally.
- **`simulation-provenance`** is a dataset-wide [SIM-KAx](https://doi.org/10.1007/s11740-026-01441-7) record describing the LS-DYNA setup (program, material model, mesh settings, contact, ...). Because its fields carry constant values directly in the manifest, they are read off `ds.metadata` without iterating records at all.

### Iterating process-parameters

The pattern is identical to step 4: ask for `ds.records(<RecordSet id>)` and consume the iterator. For tabular work, `pandas.DataFrame(list(ds.records("process-parameters")))` is one line away, but the canonical access pattern is the iterator.

```python
for n, rec in enumerate(ds.records("process-parameters"), start=1):
    if n == 1:
        for k, v in rec.items():
            print(f"  {k:42s} = {v}")
    if n >= 1:
        break
```

Output (first row):

```
process-parameters/index                   = 16039
process-parameters/geometry                = b'rectangular'
process-parameters/curvature_radius        = 30.0
process-parameters/bottom_radius           = 5.0
process-parameters/wall_angle              = 10.0
process-parameters/material_scaling_factor = 0.9
process-parameters/sheet_metal_thickness   = 0.95
process-parameters/friction_coefficient    = 0.05
process-parameters/blankholder_force       = 100000.0
process-parameters/split                   = b'train'
process-parameters/rddac                   = False
```

### Simulation provenance (SIM-KAx)

`simulation-provenance` fields use the `value` form of `mlcroissant`'s source spec, so the data lives directly in the manifest : no record iteration required. Read the field list off `ds.metadata.record_sets`:

```python
sp = next(rs for rs in ds.metadata.record_sets if rs.id == "simulation-provenance")
for f in sp.fields:
    print(f"  {f.name:50s} = {f.value!r}")
```

Output (abridged):

```
description           = 'Deep drawing of a modified quadratic cup (210 x 210 mm, 30 mm drawing depth) from DP600 dual-phase steel.'
program               = 'LS-DYNA'
tool_deformability    = 'rigid'
tool_movement         = 'kinematic'
model_symmetries      = 'none'
element_size          = 1.0
contact_tolerance     = 0.0
contact_type          = '*CONTACT_FORMING_ONE_WAY_TO_SURFACE'
material_model        = '*MAT_125 (Kinematic Hardening Transversely Anisotropic, Yoshida-Uemori, calibrated for DP600)'
schema_reference      = 'SIM-KAx, doi:10.1007/s11740-026-01441-7'
```

## Where to go next

- [PyTorch training](pytorch.md) wraps a view (custom or published) in `DDACSDataset` for batched training.
- [Visualization](visualization.md) plots arrays pulled from a record.
- [Croissant manifest](../croissant.md) explains the schema and the field-map RecordSet that underpins `add_view`.
