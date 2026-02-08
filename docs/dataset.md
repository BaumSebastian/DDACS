# Dataset Overview

The DDACS (Deep Drawing and Cutting Simulations) dataset contains **32,071 finite element simulations** of sheet metal forming processes for a dual-phase steel part. Each simulation captures two distinct operations: deep drawing (OP10) followed by cutting (OP20), both including springback analysis.

- **Total Size**: ~1 TB
- **Simulations**: 32,071
- **File Size**: ~35 MB per simulation (HDF5 format)
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
| **punch** | Upper forming tool |
| **die** | Lower forming tool |
| **binder** | Blank holder that controls material flow |

## Parameter Specifications

### Geometry Types

| Type | Description | Radius Range |
|------|-------------|--------------|
| **GEO_R** | Rectangular corners | 30-40 mm (2 mm steps) |
| **GEO_V** | Concave (inward) corners | 50-150 mm (20 mm steps) |
| **GEO_X** | Convex (outward) corners | 100-150 mm (10 mm steps) |

All geometries share: bottom radius (5-10 mm), wall angle (10-30°)

### Material Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| **MAT** | Material scaling factor | 0.9 - 1.1 (0.025 steps) |
| **SHTK** | Sheet thickness | 0.95 - 1.00 mm (0.01 mm steps) |

### Process Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| **FC** | Friction coefficient | 0.05 - 0.15 (0.01 steps) |
| **BF** | Blank holder force | 100,000 - 500,000 N (50,000 N steps) |

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

## Metadata File

The `metadata.csv` file contains all simulation parameters:

```
ID,GEO_R,GEO_V,GEO_X,RAD,MAT,FC,SHTK,BF
16039,1,0,0,30,0.9,0.05,0.95,100000.0
16040,1,0,0,30,0.9,0.06,0.95,100000.0
...
```

- **ID**: Simulation ID (maps to `{ID}.h5` file)
- **GEO_R/V/X**: One-hot encoded geometry type
- **RAD**: Corner radius [mm]
- **MAT**: Material scaling factor
- **FC**: Friction coefficient
- **SHTK**: Sheet thickness [mm]
- **BF**: Blank holder force [N]

## Common Analysis Pitfalls

1. **OP10 vs OP20**: OP20 is a cutting operation, not just tool removal
2. **Timestep indexing**: Index `-1` means different things for different components
3. **Stress/strain layers**: Middle layer (index 1) is most representative; surface layers capture bending effects

## File Format

All data is stored in HDF5 format. See [HDF5 Structure](hdf5-structure.md) for detailed field explanations.
