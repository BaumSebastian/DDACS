# Dataset Overview

DDACS contains **{{ simulation_count() }} LS-DYNA simulations** of a dual phase steel sheet metal part formed and trimmed in a two stage tooling setup. Each simulation captures deep drawing (OP10) followed by cutting (OP20) and the springback after both. The full release is {{ total_size() }} ({{ per_sim_size() }} per simulation, HDF5 + gzip).

| | |
|---|---|
| DOI | [10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801) |
| Records | {{ simulation_count() }} simulations |
| Total size | {{ total_size() }} |
| File size | {{ per_sim_size() }} (gzip + shuffle) |
| Format | HDF5 |

## Version compatibility

The `ddacs` Python package major version tracks the DaRUS dataset major version. The pairing is enforced by the Croissant manifest bundled with each release: a mismatched package version will fail to resolve the field map.

| Package | DaRUS dataset |
|---------|---------------|
| `ddacs 3.x` | [v3.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=3.0) and any future v3.x updates (current) |
| `ddacs 2.x` | [v1.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=1.0) and [v2.0](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-4801&version=2.0) |

Pin the package major to the dataset major you target, for example `pip install 'ddacs~=3.0'` to stay on the v3 line.

## Operations

Each simulation runs two operations end to end. OP10 is the deep drawing step: the blank holder presses the sheet against the die, the punch travels down, and the tools are released. OP20 then trims the formed cup and lets the part relax once more.

The blank is stored across multiple timesteps in each operation, capturing the transient state of the workpiece during forming and after springback.

## Components

Four LS-DYNA parts are tracked per simulation: the **blank** workpiece and the three tools (**binder**, **die**, **punch**). Tools are rigid and recorded at three OP10 timesteps; the blank is recorded across all four (initial, mid forming, max forming, after springback).

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_tools.png" width="700">

## Timesteps

| Component | Timesteps | What each one is |
|-----------|-----------|------------------|
| OP10 blank | 4 | initial, 50 % forming, max forming, after springback |
| OP10 tools | 3 | initial, 50 % forming, max forming |
| OP20 blank | 2 | after cutting, after springback |

!!! warning "Index `-1` is not the same state everywhere"
    `-1` selects the last timestep, which is **after springback** for the OP10 blank and OP20 blank, but **max forming** for the OP10 tools.

## RDDAC sub study

{{ simulation_count_rddac() }} of the {{ simulation_count() }} simulations form the RDDAC sub study, an experimental investigation of the deviation between LS-DYNA simulation and physical reality. The accompanying paper, [*Statistical Analysis of Simulation to Reality Deviation in Deep Drawing with a Benchmark Dataset*](https://doi.org/10.1007/s12666-026-03870-5), characterises the epistemic uncertainty and proposes mitigations; the analysis code lives at [github.com/BaumSebastian/RDDAC](https://github.com/BaumSebastian/RDDAC). The `rddac` column of `process_parameters.csv` flags membership; `DDACSDataset(where=lambda row: row["rddac"])` streams only the sub study.

## Excluded simulations

Eleven simulations from the original grid are not part of the v3 release. Six were dropped during the v3 repackaging because of corrupted gzip chunks in their OP20 datasets; five never finished the original FE run on the HLRS cluster. None of them appear in `process_parameters.csv`, so iterating the CSV yields only the simulations that successfully shipped.

| Sim IDs | Geometry | Reason |
|---------|----------|--------|
| 16095, 17061, 19044 | rectangular | Corrupted gzip chunk in OP20 |
| 259953, 259968 | concave | Corrupted gzip chunk in OP20 |
| 398073 | convex | Corrupted gzip chunk in OP20 |
| 5 further IDs | rectangular | FE solver did not produce output |

## Further reading

- [Process parameters](process-parameters.md): geometry families and the columns of `process_parameters.csv`.
- [HDF5 structure](hdf5-structure.md): every field, shape, and dtype inside one simulation.
- [Croissant manifest](croissant.md): the machine readable schema the package and external tools read.
