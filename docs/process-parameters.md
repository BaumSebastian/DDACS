# Process Parameters

Each row in `process_parameters.csv` describes one simulation. The `index` column is the same number as the HDF5 filename inside the zips, so the CSV is the single point of entry for filtering before any HDF5 is opened.

## Columns

{{ process_parameters_table() }}

## Geometry

The `geometry` column takes three values, one per family of corner shapes. The `curvature_radius` column samples two extremes per family.

| `geometry` | Description | `curvature_radius` |
|------------|-------------|--------------------|
| `rectangular` | Rectangular corners | 30 / 40 mm |
| `concave` | Inward corners | 50 / 150 mm |
| `convex` | Outward corners | 100 / 150 mm |

`bottom_radius` is sampled at 5 / 7.5 / 10 mm and `wall_angle` at 10 / 20 / 30 deg across all three families.

**OP10 (after forming)**

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_geometries_op10.png" width="700">

**OP20 (after cutting)**

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_geometries_op20.png" width="700">

## Split

The `split` column carries the recommended `train` / `val` / `test` partition. The partition was chosen so that each split sees every geometry family and a representative range of material and process parameters.

## RDDAC membership

The boolean `rddac` column flags whether a simulation belongs to the [RDDAC sub study](dataset.md#rddac-sub-study). 396 of the {{ simulation_count() }} simulations are flagged. `DDACSDataset(where=lambda row: row["rddac"])` streams only the sub study.

## Sample

```
index,geometry,curvature_radius,bottom_radius,wall_angle,material_scaling_factor,sheet_metal_thickness,friction_coefficient,blankholder_force,split,rddac
16039,rectangular,30,5,10,0.9,0.95,0.05,100000.0,train,False
16040,rectangular,30,5,10,0.9,0.95,0.06,100000.0,train,False
```

## Filtering recipe

```python
import pandas as pd

df = pd.read_csv("./data/process_parameters.csv")

rectangular  = df[df["geometry"] == "rectangular"]
thin_sheet   = df[df["sheet_metal_thickness"] < 0.97]
rddac_only   = df[df["rddac"]]
```

The same predicate is accepted by `DDACSDataset(where=...)`, which keeps IO scaled to the surviving rows rather than the full {{ simulation_count() }}.
