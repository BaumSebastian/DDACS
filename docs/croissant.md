# Croissant Manifest

DDACS ships a [Croissant 1.1](https://github.com/mlcommons/croissant) manifest (`metadata.json`) describing the dataset in a machine readable form. The same file is read by [Kaggle](https://www.kaggle.com/), [Hugging Face](https://huggingface.co/), and Google Dataset Search, so any tool that already speaks Croissant can consume DDACS without DDACS specific code.

## Why Croissant?

The Croissant manifest is the single source of truth for what is in DDACS and where to find it. Three concrete benefits follow.

**One schema across tooling.** Field names, shapes, units and descriptions live in the manifest only. The `ddacs` Python package reads them from there; Kaggle, Hugging Face, Google Dataset Search and any other Croissant aware tool read the same file. Renaming a field, adding a new use case RecordSet, or updating a description happens once and propagates everywhere.

**No extraction step.** `mlcroissant` opens the zip archives via the [zipfile](https://docs.python.org/3/library/zipfile.html) standard library and streams individual HDF5 members on demand. With the small bundle that is one ~20 MB zip; with the full release it is 22 zips totalling {{ total_size() }}. The user never has to expand the dataset on disk; `ddacs download` keeps the zips as they were uploaded.

**Streaming matches a hand rolled loop.** `DDACSDataset` uses the manifest only to discover field paths and timestep transforms; the per record read is a zip member fetch into a `BytesIO` and an `h5py.File` over it, the same primitives any hand rolled code would use. On the 396 RDDAC simulations, computing the same springback delta:

| Path | Time | Per sim |
|------|------|---------|
| Hand rolled `zipfile + h5py.File` loop | 9.4 s | 23.8 ms |
| `DDACSDataset.__iter__` (Croissant driven) | 9.2 s | 23.3 ms |

The ~2 % spread is within run to run noise; across three repeated runs the two paths consistently land within a percent of each other, sometimes with DDACSDataset slightly ahead because of the cached zip handle. See the [PyTorch training](tutorials/pytorch.md#performance) tutorial for the methodology.

## Contents

- **Distribution**: every file on DaRUS plus a FileSet for the HDF5 contents inside the zips.
- **`process-parameters`**: rows of `process_parameters.csv` with descriptions and types.
- **`field-map`**: every HDF5 dataset (path, shape, named dimensions, unit) declared once.
- **`simulation-provenance`**: dataset wide constants following the [SIM-KAx schema](https://doi.org/10.1007/s11740-026-01441-7).
- **Use case RecordSets**: `springback-minimal`, `springback-prediction`, `forming-snapshot`, `cutting-view`.

## With the `ddacs` package

```python
import ddacs
from ddacs import croissant

ds = ddacs.load()                            # default URL + data_dir="./data"
ds = ddacs.load(data_dir="/mnt/big-disk")    # files were downloaded elsewhere
ds = ddacs.load(source="./metadata.json")    # local manifest copy

print(croissant.dataset_name(ds))
for col, desc in croissant.process_parameters_descriptions(ds).items():
    print(col, ":", desc)
```

`croissant.METADATA_URL` is the permanent DaRUS download URL for `metadata.json`.

## Without the `ddacs` package

```python
import mlcroissant as mlc

ds = mlc.Dataset(jsonld="{{ metadata_url() }}")
for record in ds.records("springback-minimal"):
    ...
```
