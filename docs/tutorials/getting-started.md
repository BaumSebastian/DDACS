# Getting Started

This tutorial walks from installation to a first plot. It uses only the public surface that ships with v3.0.0: `ddacs.load`, `ddacs.open_h5`, `ddacs.inspect_h5`, and the visualization helpers.

## 1. Install

```bash
pip install ddacs
```

The PyTorch adapter is optional. Install it explicitly if a model is to be trained:

```bash
pip install ddacs[torch]
```

For hardware specific PyTorch builds (CUDA, ROCm, MPS) see [pytorch.org](https://pytorch.org/get-started/locally/) and install PyTorch before `ddacs`.

## 2. Download the sample bundle

The full dataset is large. For a first walkthrough, download the small bundle ({{ small_download_size() }}):

```bash
ddacs download --small
```

The command writes `metadata.json`, `process_parameters.csv`, and one sample simulation zip (`258864.zip`) into `./data/`. The zips are not extracted; `mlcroissant` reads them in place.

## 3. Load the dataset

`ddacs.load()` returns an `mlcroissant.Dataset`. The default `data_dir` matches the `ddacs download` output, so no further argument is needed:

```python
import ddacs

ds = ddacs.load()
print(ds.metadata.name)
print([rs.id for rs in ds.metadata.record_sets])
```

Output:

```
DDACS
['process-parameters', 'field-map', 'simulation-provenance',
 'springback-minimal', 'springback-prediction',
 'forming-snapshot', 'cutting-view']
```

The dataset is composed of named RecordSets. Each RecordSet defines a fixed selection of fields suitable for one task (e.g. springback prediction, forming snapshot).

## 4. Iterate records

The `process-parameters` RecordSet is the simulation index. It iterates {{ simulation_count() }} rows, one per simulation:

```python
for n, rec in enumerate(ds.records('process-parameters'), start=1):
    if n == 1:
        for k, v in rec.items():
            print(f"{k:42s} = {v}")
    if n >= 1:
        break
```

The `springback-minimal` RecordSet pulls real HDF5 arrays from the local zips. With the small bundle only the sample simulation is iterable; with the full download it produces {{ simulation_count() }} records.

## 5. Inspect one simulation

`ddacs.open_h5(sim_id)` resolves the manifest, finds the right zip, reads the HDF5 member into memory and returns an `h5py.File`. It is read only and supports the `with` idiom:

```python
with ddacs.open_h5(258864) as f:
    print(list(f['OP10/blank'].keys()))
```

`ddacs.inspect_h5` prints the group and dataset hierarchy of an open file or a path on disk:

```python
with ddacs.open_h5(258864) as f:
    ddacs.inspect_h5(f)
```

Each line that starts with `@` is an HDF5 attribute. Groups end in `/`; datasets show their shape and dtype.

## 6. First plot

Combine `open_h5` with the visualization helpers to render the blank mesh coloured by thickness at the final forming timestep:

```python
import ddacs

with ddacs.open_h5(258864) as f:
    nodes = f['OP10/blank/node_coordinates'][:]
    faces = f['OP10/blank/element_shell_node_indexes'][:]
    thickness = f['OP10/blank/element_shell_thickness'][-1]

ax, cbar = ddacs.plot_mesh(nodes, faces, values=thickness, cmap='viridis')
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/first_plot.png" width="700">

## Where to go next

- [Build your own view](views.md) explains `ddacs.add_view` and the JSONPath transforms behind it.
- [PyTorch training](pytorch.md) covers `DDACSDataset`, multi worker `DataLoader`, DDP, and the per simulation read benchmark.
- [Visualization](visualization.md) covers mesh, point cloud, and vector field plotting.
- [Loose HDF5 recipe](loose-h5.md) shows the CSV plus `h5py.File` iteration loop for users who run `ddacs download --extract --remove-zip`.
