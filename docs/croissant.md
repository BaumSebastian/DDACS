# Croissant Manifest

DDACS ships with a [Croissant 1.1](https://github.com/mlcommons/croissant) manifest
(`metadata.json`) that describes the dataset in a machine-readable form. The
same manifest is what [Kaggle](https://www.kaggle.com/),
[Hugging Face](https://huggingface.co/) and Google Dataset Search read to
present a dataset to ML pipelines, so DDACS is consumable out of the box by
tooling that already speaks the standard.

## What is in the manifest

- **Distribution**: every uploaded file on DaRUS plus a FileSet for the HDF5 contents inside the zip packages.
- **`process-parameters` RecordSet**: the columns of `process_parameters.csv` with descriptions and types.
- **`field-map` RecordSet**: every HDF5 dataset (path, shape, named dimensions, unit) declared once.
- **`simulation-provenance` RecordSet**: dataset-wide constants following the [SIM-KAₓ schema](https://doi.org/10.1007/s11740-026-01441-7) — LS-DYNA setup, material model, mesh, contact, …
- **Use-case RecordSets**: `springback-minimal`, `springback-prediction`, `forming-snapshot`, `cutting-view` — pre-defined slices of the field-map for the most common ML targets.

## Reading the manifest from Python

The package exposes a small helper around the official `mlcroissant` library.
By default it loads the local copy from `data/metadata.json` if present and
falls back to the live DaRUS URL otherwise.

```python
from ddacs import metadata as md

ds = md.load_dataset(data_dir="./data")        # local copy when available
ds = md.load_dataset()                          # remote (DaRUS) otherwise
ds = md.load_dataset(source=md.METADATA_URL)   # explicit DaRUS URL

print(md.dataset_name(ds))
for col, desc in md.process_parameters_descriptions(ds).items():
    print(col, "—", desc)
```

`md.METADATA_URL` is the permanent DaRUS download URL — useful for build pipelines, CI and Kaggle notebooks.

## Reading the manifest directly with `mlcroissant`

If you only need Croissant access (no DDACS package), the standard library works against the same URL:

```python
import mlcroissant as mlc

ds = mlc.Dataset(jsonld="{{ metadata_url() }}")
for record in ds.records("springback-minimal"):
    ...  # one record per simulation
```

## When to refresh `metadata.json`

The DaRUS-hosted manifest is the source of truth. The `data/metadata.json`
copy in this repository is only a convenience for local docs builds and tests.
If you update the published dataset, regenerate the manifest with
`.helper/v3/build_croissant.py` and re-upload it; the package needs no changes
because `ddacs.metadata` resolves the URL through `ddacs.config`.
