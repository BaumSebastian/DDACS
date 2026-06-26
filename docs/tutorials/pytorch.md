# PyTorch training

`DDACSDataset` is a `torch.utils.data.IterableDataset` over a Croissant view. It is the iteration path to reach for when training a model: the underlying `mlcroissant.Dataset.records(...)` aborts on the first missing zip, while `DDACSDataset` builds a `sim_id -> local zip` index at construction time and silently skips simulations whose zip is not present. Full constructor signature in the [API reference](../api/pytorch.md).

The companion notebook at [`notebooks/03_pytorch.ipynb`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks/03_pytorch.ipynb) reproduces every cell below.

## 1. Construct a `DDACSDataset`

`DDACSDataset(view, data_dir, ...)` loads the manifest the same way `ddacs.load` does, resolves the view's fields to HDF5 paths plus JSONPath transforms, walks `data_dir` to build the local zip index, and reads `process_parameters.csv` to apply optional filters. After construction `ds._sim_ids` is the list of simulation ids the dataset will attempt to iterate, and `ds._h5_index` is the subset for which a zip is actually on disk.

```python
DATA_DIR = './data'      # repository root, or '../data' from notebooks/
sim_id   = 258864

import ddacs
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

ds = DDACSDataset(view='springback-minimal', data_dir=DATA_DIR)
print('view:           ', ds.view)
print('field specs:    ', ds._field_specs)
print('total sim_ids:  ', len(ds._sim_ids), '(every sim in process_parameters.csv)')
print('locally indexed:', len(ds._h5_index), '(only these will actually stream)')
```

Output (with the small bundle on disk):

```
view:            springback-minimal
field specs:     {'op10_blank_node_displacement_forming':    ('OP10/blank/node_displacement', 2),
                  'op10_blank_node_displacement_springback': ('OP10/blank/node_displacement', 3)}
total sim_ids:   32466 (every sim in process_parameters.csv)
locally indexed: 1    (only these will actually stream)
```

## 2. Iterate one batch through a `DataLoader`

`DDACSDataset` plugs straight into a `DataLoader`. The default `collate_fn` stacks each field along a new leading batch axis. With the small bundle only `258864.zip` is on disk so the loader yields a single-element "batch"; with the full release `batch_size=16` would fill normally.

```python
loader = DataLoader(ds, batch_size=16, num_workers=0)
for batch in loader:
    for k, v in batch.items():
        print(f'  {k:50s} shape={tuple(v.shape)} dtype={v.dtype}')
    break
```

Output:

```
op10_blank_node_displacement_forming     shape=(1, 11236, 3) dtype=torch.float64
op10_blank_node_displacement_springback  shape=(1, 11236, 3) dtype=torch.float64
```

## 3. Filter via the Croissant manifest

Both filters run against `process_parameters.csv` **before** any zip is opened, so IO scales with the surviving simulations rather than with the full 32 466.

The row keys are not magic: they come straight from the Croissant manifest. `metadata.json` declares `process_parameters.csv` as a `FileObject` and exposes its columns as the `process-parameters` `RecordSet`. The same fields are what `ds.records('process-parameters')` yields when iterating through `mlcroissant` directly. `DDACSDataset` simply consumes those rows at construction time and applies the predicate before any zip is touched.

### What `where` receives

`DDACSDataset` reads `process_parameters.csv` with `pandas` and runs `where` once per row via `df.apply(where, axis=1)`. The `row` argument is a `pandas.Series` whose index is the CSV column names, so you can read columns with either `row['split']` or `row.split`. Values are native Python types:

| Column | Type | Example |
|--------|------|---------|
| `index` | `int` | `16039` |
| `geometry` | `str` | `'rectangular'`, `'concave'`, `'convex'` |
| `curvature_radius`, `bottom_radius`, `wall_angle`, ... | `float` | `30.0` |
| `split` | `str` | `'train'`, `'val'`, `'test'` |
| `rddac` | `bool` | `False` |

- `where=<callable>`: any function `pd.Series -> bool`. Available column names match the manifest's `process-parameters` fields.
- `sim_ids=[...]`: explicit allowlist of integers, applied before `where`.

Both can be combined; the predicate is applied after the allowlist.

```python
# Inspect the manifest's process-parameters columns once.
raw = ddacs.load(data_dir=DATA_DIR)
sample = next(iter(raw.records('process-parameters')))
print('manifest columns:', [k.split('/')[-1] for k in sample.keys()])

rect = DDACSDataset(
    view='springback-minimal',
    data_dir=DATA_DIR,
    where=lambda row: row['geometry'] == 'rectangular',
)
print(f'rectangular-only sim_ids: {len(rect._sim_ids):>6d} (of 32466)')

ids_only = DDACSDataset(
    view='springback-minimal',
    data_dir=DATA_DIR,
    sim_ids=[sim_id],
)
print(f'sim_ids=[sim_id]:         {len(ids_only._sim_ids):>6d}')
```

Output:

```
manifest columns: ['index', 'geometry', 'curvature_radius', 'bottom_radius', 'wall_angle', 'material_scaling_factor', 'sheet_metal_thickness', 'friction_coefficient', 'blankholder_force', 'split', 'rddac']
rectangular-only sim_ids:  10689 (of 32466)
sim_ids=[sim_id]:              1
```

### `where` only sees CSV columns

The predicate runs before any HDF5 is opened, so it cannot read HDF5 attributes. In practice every per-simulation root attribute of the v3 HDF5 files is already mirrored in `process_parameters.csv` (`geometry`, `sheet_metal_thickness`, `blankholder_force`, ...), so the CSV path covers all per-simulation filtering today. For an HDF5-only attribute, scan the archives once to build a sim-id allowlist and pass it through `sim_ids=...`.

## 4. Train / val / test splits

`process_parameters.csv` ships with a `split` column whose canonical values are `'train'`, `'val'`, and `'test'`. Because the column is part of the Croissant manifest, the same `where=` predicate that filtered by geometry above works on it. Three `DDACSDataset` instances, one per split, is the fastest way to wire up a training loop without writing any custom partitioning code.

Shuffle the train split for SGD; leave `val`/`test` deterministic for reproducible evaluation.

```python
splits = {}
for name in ('train', 'val', 'test'):
    splits[name] = DDACSDataset(
        view='springback-minimal',
        data_dir=DATA_DIR,
        where=lambda row, n=name: row['split'] == n,
        shuffle=(name == 'train'),
        seed=42,
    )

for name, split_ds in splits.items():
    streamable = sum(1 for sid in split_ds._sim_ids if sid in split_ds._h5_index)
    print(f"{name:>5s}: {len(split_ds._sim_ids):>6d} sim_ids (of 32466), "
          f"{streamable:>3d} streamable now (zip on disk)")
```

Output (with the small bundle on disk; `258864` happens to sit in the `val` split):

```
train:  25973 sim_ids (of 32466),   0 streamable now (zip on disk)
  val:   3246 sim_ids (of 32466),   1 streamable now (zip on disk)
 test:   3247 sim_ids (of 32466),   0 streamable now (zip on disk)
```

`len(_sim_ids)` is the CSV-side count after the predicate; the second column counts only the rows whose zip is on disk and will actually yield a record. With the full release on disk, the second column matches `len(_sim_ids)`.

## 5. Shuffle + `set_epoch`

`shuffle=True` permutes each shard with a seed derived from `seed + epoch + shard_id`. Worker shards stay disjoint, so two workers do not produce the same simulation. Call `set_epoch(n)` once per epoch to get a different permutation each time. Without it, every epoch sees the same order.

```python
ds_shuf = DDACSDataset(
    view='springback-minimal',
    data_dir=DATA_DIR,
    shuffle=True,
    seed=42,
)
for epoch in range(2):
    ds_shuf.set_epoch(epoch)
    print(f'epoch {epoch}: first sim_id this shard would visit -> '
          f'{ds_shuf._sim_ids[0] if not ds_shuf.shuffle else "(shuffled per shard)"}')
```

Output:

```
epoch 0: first sim_id this shard would visit -> (shuffled per shard)
epoch 1: first sim_id this shard would visit -> (shuffled per shard)
```

## 6. Sharding

Workers and DDP ranks are detected from `torch.utils.data.get_worker_info()` and `torch.distributed`. The same `DDACSDataset` instance works under `num_workers=0`, `num_workers=N`, and DDP without constructor changes: each shard slices `ds._sim_ids` by `(rank * num_workers + worker_id)` modulo `(world_size * num_workers)`, so the partition is exhaustive and disjoint.

## 7. Partial download: graceful skip vs `records()` abort

`DDACSDataset` is the iteration path to use when only a subset of zips is on disk. `mlcroissant.Dataset.records(...)` walks every zip in the FileSet alphabetically and aborts at the first missing one; `DDACSDataset` checks the local index per simulation and silently skips.

```python
import numpy as np

# 1) mlcroissant records : expected to abort on the first missing zip.
raw = ddacs.load(data_dir=DATA_DIR)
yielded = 0
try:
    for rec in raw.records('springback-minimal'):
        yielded += 1
        break
except Exception as e:
    print(f'records(): yielded {yielded} record(s) then {type(e).__name__}: {str(e)[:120]}')

# 2) DDACSDataset : graceful skip.
yielded = sum(1 for _ in DDACSDataset(view='springback-minimal', data_dir=DATA_DIR))
print(f'DDACSDataset: yielded {yielded} record(s) without error')
```

Output (with the small bundle on disk):

```
records(): yielded 0 record(s) then GenerationError: An error occured during
the sequential generation of the dataset, more specifically during the
operation Download(106921_109120.zip)
DDACSDataset: yielded 1 record(s) without error
```

## Performance

`DDACSDataset.__iter__` matches a hand rolled `zipfile + h5py.File` loop computing the same springback delta. Both paths produce identical numerical output (verified by the `sum |delta|` check in the benchmark). On the 396 simulation RDDAC subset:

| Path | Time | Per sim |
|------|------|---------|
| Hand rolled `zipfile + h5py.File` loop | 9.4 s | 23.8 ms |
| `DDACSDataset.__iter__` | 9.2 s | 23.3 ms |

Three repeated runs land within ~1 % of each other; `DDACSDataset` is consistently at parity or slightly ahead thanks to the cached zip handle across consecutive sims in the same archive.

## Custom collate

Records with variable mesh sizes need a custom `collate_fn`. Use `DataLoader(..., collate_fn=...)` and stack only the dimensions that match across the batch.

## Where to go next

- [Build your own view](views.md) explains how the field map and JSONPath transforms drive `DDACSDataset` field selection.
- [Visualization](visualization.md) plots arrays pulled from a single record.
- [API reference](../api/pytorch.md) lists every constructor argument.
