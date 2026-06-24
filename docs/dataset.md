# Dataset Overview

The DDACS (Deep Drawing and Cutting Simulations) dataset contains **{{ simulation_count() }} finite element simulations** of sheet metal forming processes for a dual-phase steel part. Each simulation captures two distinct operations: deep drawing (OP10) followed by cutting (OP20), both including springback analysis.

- **Total Size**: {{ total_size() }}
- **Simulations**: {{ simulation_count() }}
- **File Size**: {{ per_sim_size() }} per simulation (HDF5 format)
- **DOI**: [10.18419/DARUS-4801](https://doi.org/10.18419/DARUS-4801)

## Operations

| Operation | Description |
|-----------|-------------|
| **OP10** | Deep drawing (forming) with springback after tool removal |
| **OP20** | Cutting operation with springback after cutting |

## Components

| Component | Description |
|-----------|-------------|
| **blank** | The sheet metal workpiece being formed |
| **punch** | Lower forming tool |
| **die** | Upper forming tool |
| **binder** | Blank holder that controls material flow |

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_tools.png" width="700">

## Parameter Specifications

### Geometry

| `geometry` value | Description | `curvature_radius` range |
|------------------|-------------|--------------------------|
| `rectangular` | Rectangular corners | 30 / 40 mm |
| `concave`     | Concave (inward) corners | 50 / 150 mm |
| `convex`      | Convex (outward) corners | 100 / 150 mm |

All geometries share `bottom_radius` (5 / 7.5 / 10 mm) and `wall_angle` (10 / 20 / 30°).

**OP10 (After Forming):**

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_geometries_op10.png" width="700">

**OP20 (After Cutting):**

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_geometries_op20.png" width="700">

### Parameter columns

{{ simulation_stats() }}

{{ process_parameters_table() }}

## Timesteps

The timestep structure varies between components:

| Component | Timesteps | Description |
|-----------|-----------|-------------|
| **OP10 blank** | 4 | Initial → 50% forming → max forming → springback |
| **OP10 tools** | 3 | Initial → 50% forming → max forming |
| **OP20 blank** | 2 | After cutting → springback |

!!! warning "Timestep Indexing"
    Using index `-1` provides the last timestep but represents different physical states:

    - OP10 blank: springback (index 3)
    - OP10 tools: max forming (index 2)
    - OP20 blank: springback after cutting (index 1)

## Metadata file

Each simulation is indexed in `process_parameters.csv`:

```
index,geometry,curvature_radius,bottom_radius,wall_angle,material_scaling_factor,sheet_metal_thickness,friction_coefficient,blankholder_force,split,rddac
16039,rectangular,30,5,10,0.9,0.95,0.05,100000.0,train,False
16040,rectangular,30,5,10,0.9,0.95,0.06,100000.0,train,False
...
```

`index` matches the HDF5 filename (`{index}.h5`). For the full column reference, see the [Parameter columns](#parameter-columns) section above.

## Excluded simulations

Eleven simulations from the original parameter grid are not part of the v3 release. Six failed during the v3 repackaging because of corrupted gzip chunks in their OP20 datasets, and five did not complete the original FE run on the HLRS cluster.

| Sim ID | Geometry | Reason |
|--------|----------|--------|
| 16095, 17061, 19044 | rectangular (min corner) | Corrupted gzip chunk in OP20 dataset |
| 259953, 259968 | concave (min corner) | Corrupted gzip chunk in OP20 dataset |
| 398073 | convex (max corner) | Corrupted gzip chunk in OP20 dataset |
| 5 IDs in the R_min block | rectangular (min corner) | FE solver did not produce output |

Excluded IDs are absent from `process_parameters.csv`, so simply iterating the CSV gives only the simulations that successfully shipped.

## Common Analysis Pitfalls

1. **OP10 vs OP20**: OP20 is a cutting operation, not just tool removal
2. **Timestep indexing**: Index `-1` means different things for different components
3. **Stress/strain layers**: Middle layer (index 1) is most representative; surface layers capture bending effects

## File Format

All data is stored in HDF5 format. See [HDF5 Structure](hdf5-structure.md) for detailed field explanations.
