# DDACS — Deep Drawing and Cutting Simulations Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Documentation](https://img.shields.io/badge/docs-readthedocs.io-blue.svg)](https://ddacs.readthedocs.io) [![DaRUS Repository](https://img.shields.io/badge/repository-DaRUS-green.svg)](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801) [![DOI](https://img.shields.io/badge/DOI-10.18419%2FDARUS--4801-blue.svg)](https://doi.org/10.18419/DARUS-4801) [![Paper](https://img.shields.io/badge/paper-MATEC%20Web%20Conf.-red.svg)](https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html)

![Simulation overview](https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/simulation_overview.gif)

*Simulation with the tool geometries showing sheet metal thinning, stress, and strain.*

**A large-scale dataset and benchmark for training AI models that replace computationally expensive FEA simulations in industrial sheet metal manufacturing.** Each simulation models a two-stage stamping process (deep drawing in OP10 and trimming with elastic recovery in OP20) for a cup geometry parameterised by 8 input dimensions. Train ML surrogates that predict mesh deformation, stress, strain, and springback in seconds instead of the minutes-to-hours a CAE solver would take.

|  |  |
|---|---|
| **Simulations** | 32,466 |
| **Total size** | ~640 GB (HDF5, lossless) |
| **Process steps per sim** | 2 (OP10 deep drawing, OP20 trimming) |
| **Input parameters** | 8 (4 geometric + 4 process) |
| **Train / val / test** | 25,973 / 3,246 / 3,247 (predefined) |
| **Mesh-node states** | ~2.1 B across all sims, timesteps, components |

**[Documentation](https://ddacs.readthedocs.io)** · **[Dataset DOI](https://doi.org/10.18419/DARUS-4801)** · **[Paper](https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html)**

## About this sample

This is a **22 MB teaser** of DDACS — one full simulation plus the Croissant 1.1
manifest, the complete process-parameter table, and the six tutorial notebooks —
so you can explore the schema and run every tutorial in seconds before committing
to the full download.

```
data/
  metadata.json             Croissant 1.1 manifest (the dataset schema)
  process_parameters.csv    8 input parameters for all 32,466 simulations
  h5/258864.zip             one full simulation (OP10 + OP20, all components)
ddacs_documentation.pdf     dataset documentation
notebooks/                  six end-to-end tutorials (see notebooks/README.md)
```

**Croissant manifest.** `data/metadata.json` is the
[Croissant 1.1](https://mlcommons.org/croissant/) manifest — the machine-readable
schema (every HDF5 field and CSV column) that `ddacs.load()` and any
Croissant-aware tool consume. It is the same manifest published with the full
dataset on DaRUS ([doi:10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801)).

## Installation

```bash
pip install ddacs          # add the PyTorch adapter with: pip install 'ddacs[torch]'
```

## Basic usage

`ddacs.load` parses the Croissant manifest; `ddacs.open_h5` opens a single
simulation in memory and returns an `h5py.File`.

```python
import ddacs

# Load the dataset manifest bundled with this sample.
ds = ddacs.load(data_dir="data")
print([rs.id for rs in ds.metadata.record_sets])

# Open the included simulation. OP10 carries the blank and the three tools.
with ddacs.open_h5(258864, data_dir="data") as f:
    blank_thickness = f["OP10/blank/element_shell_thickness"][-1]
    print("final-timestep thickness:", blank_thickness.shape)
```

## PyTorch integration

`DDACSDataset` is a `torch.utils.data.IterableDataset` over a Croissant view. It
auto-shards across DataLoader workers and DDP ranks, and silently skips
simulations whose zip is missing — so partial downloads (like this teaser) stream
cleanly.

```python
from ddacs.pytorch import DDACSDataset
from torch.utils.data import DataLoader

ds = DDACSDataset(view="springback-minimal", data_dir="data")
for batch in DataLoader(ds, batch_size=1, num_workers=0):
    forming    = batch["op10_blank_node_displacement_forming"]
    springback = batch["op10_blank_node_displacement_springback"]
    break
```

## Tutorials

Six end-to-end notebooks ship in `notebooks/` and are published on
[Read the Docs](https://ddacs.readthedocs.io/en/latest/tutorials/):

1. **Getting started** — install, load, first plot.
2. **Build your own view** — `ddacs.add_view`, manifest inspection, SIM-KAx provenance.
3. **PyTorch training** — `DDACSDataset`, filters, train/val/test splits.
4. **Visualization** — thickness, components, springback, vectors.
5. **Loose HDF5 recipe** — pandas + `h5py` after `--extract --remove-zip`.
6. **Streaming & numpy export** — `iter_view`, `export_to_numpy`, ~1000× speedup.

## Version compatibility

The `ddacs` package major version tracks the DaRUS dataset major version, enforced
by the bundled Croissant manifest.

| Package | DaRUS dataset |
|---------|---------------|
| `ddacs 3.x` | v3.0 and any future v3.x updates (current) |
| `ddacs 2.x` | v1.0 and v2.0 |

Pin the major to the dataset you target, e.g. `pip install 'ddacs~=3.0'`.

## ⬇️ Get the full dataset

**This sample contains a single simulation.** The complete DDACS dataset —
**32,466 simulations, ~640 GB of lossless HDF5**, with the predefined
25,973 / 3,246 / 3,247 train/val/test split — is hosted on DaRUS with a citable DOI:

### ➡️ https://doi.org/10.18419/DARUS-4801

Everything you ran here scales to the full release unchanged — just point the same
code at the full download, or let the package fetch it:

```bash
pip install ddacs
ddacs download            # full release  (ddacs download --small for this 22 MB sample)
```

## Citation

If you use this dataset or code in your research, please cite both the dataset and the paper:

```bibtex
@dataset{baum2025ddacs,
  title={Deep Drawing and Cutting Simulations Dataset},
  subtitle={FEM Simulations of a deep drawn and cut dual phase steel part},
  author={Baum, Sebastian and Heinzelmann, Pascal},
  year={2025}, version={3.0}, publisher={DaRUS},
  doi={10.18419/DARUS-4801}, license={CC BY 4.0},
  url={https://doi.org/10.18419/DARUS-4801}
}

@article{heinzelmann2025benchmark,
  title={A Comprehensive Benchmark Dataset for Sheet Metal Forming: Advancing
         Machine Learning and Surrogate Modelling in Process Simulations},
  author={Heinzelmann, Pascal and Baum, Sebastian and Riedmueller, Kim Rouven
          and Liewald, Mathias and Weyrich, Michael},
  journal={MATEC Web of Conferences}, volume={408}, year={2025}, pages={01090},
  doi={10.1051/matecconf/202540801090},
  url={https://www.matec-conferences.org/articles/matecconf/abs/2025/02/matecconf_iddrg2025_01090/matecconf_iddrg2025_01090.html}
}
```

## License

Data: **CC BY 4.0**. Package code: MIT.
