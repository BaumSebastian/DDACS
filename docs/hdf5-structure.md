# HDF5 File Structure

Each simulation is stored as a self-contained HDF5 file (~35 MB). This page documents the complete structure.

## File Organization

```
simulation_{id}.h5
├── Attributes (root level)
│   ├── Geometry_Parameters: [radius, bottom_radius, wall_angle]
│   └── Material_Parameters: [MAT, FC, SHTK, BF]
│
├── OP10/ (deep drawing operation)
│   ├── blank/   → Sheet metal workpiece
│   ├── die/     → Lower tool
│   ├── punch/   → Upper tool
│   ├── binder/  → Blank holder
│   └── general/ → Simulation metrics
│
└── OP20/ (cutting operation)
    └── blank/   → Part after cutting
```

## Inspecting Structure

Use the `display_structure` function to inspect any HDF5 file:

```python
from ddacs import display_structure
display_structure("path/to/simulation.h5")
```

## Field Reference

### Node Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `node_ids` | (n,) | Unique integer identifiers for each node |
| `node_coordinates` | (n, 3) | Initial XYZ positions [mm] |
| `node_displacement` | (t, n, 3) | Position at each timestep [mm] |
| `node_velocity` | (t, n, 3) | Velocity vectors [mm/s] |
| `node_acceleration` | (t, n, 3) | Acceleration vectors [mm/s²] |

### Element Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `element_shell_ids` | (m,) | Unique element identifiers |
| `element_shell_node_ids` | (m, 4) | Global node IDs defining each element |
| `element_shell_node_indexes` | (m, 4) | Node indices for each quad element |
| `element_shell_part_indexes` | (m,) | Part membership labels |
| `element_shell_thickness` | (t, m) | Current thickness [mm] |

### Stress and Strain Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `element_shell_stress` | (t, m, 3, 6) | Stress tensor at 3 integration points |
| `element_shell_strain` | (t, m, 2, 6) | Strain tensor at 2 integration points |
| `element_shell_stress_all_ipt` | (m, 7, 6) | Stress at all 7 through-thickness points |
| `element_shell_stress_all_ipt_thickness` | (1, 7) | Normalized positions of integration points |
| `element_shell_effective_plastic_strain` | (t, m, 3) | Effective plastic strain |
| `element_shell_effective_plastic_strain_all_ipt` | (m, 7) | Plastic strain at all integration points |

**Stress/strain tensor components**: `[σxx, σyy, σzz, τxy, τyz, τzx]`

**Integration point layers**:

- Index 0: Bottom surface
- Index 1: Middle (most representative for overall behavior)
- Index 2: Top surface

### Force Fields

| Field | Shape | Description |
|-------|-------|-------------|
| `element_shell_bending_moment` | (t, m, 3) | Bending moments [N·mm] |
| `element_shell_normal_force` | (t, m, 3) | Normal forces [N] |
| `element_shell_shear_force` | (t, m, 2) | Shear forces [N] |
| `element_shell_internal_energy` | (t, m) | Internal energy [mJ] |

### Global Fields (in `general/` group)

| Field | Shape | Description |
|-------|-------|-------------|
| `global_timesteps` | (t,) | Simulation time values |
| `global_internal_energy` | (t,) | Total internal energy [mJ] |
| `global_kinetic_energy` | (t,) | Total kinetic energy [mJ] |
| `global_total_energy` | (t,) | Sum of all energy [mJ] |
| `global_velocity` | (t, 3) | Global velocity [mm/s] |
| `part_internal_energy` | (t, 4) | Internal energy per part |
| `part_kinetic_energy` | (t, 4) | Kinetic energy per part |
| `part_hourglass_energy` | (t, 4) | Hourglass energy (mesh quality) |
| `part_mass` | (t, 4) | Mass per part [kg] |

## Shape Notation

- `t` = number of timesteps (3 for tools, 4 for OP10/blank, 2 for OP20/blank)
- `n` = number of nodes
- `m` = number of elements

## Component Availability

Not all fields are available for all components:

| Field Category | blank | die | punch | binder |
|----------------|-------|-----|-------|--------|
| Node data | ✓ | ✓ | ✓ | ✓ |
| Element connectivity | ✓ | ✓ | ✓ | ✓ |
| Stress/strain | ✓ | - | - | - |
| Thickness | ✓ | - | - | - |
| Forces | ✓ | - | - | - |

## Example: Reading Data

```python
import h5py
import numpy as np

with h5py.File("simulation.h5", "r") as f:
    # Get thickness after springback (last timestep)
    thickness = np.array(f["OP10/blank/element_shell_thickness"])[-1]

    # Get Von Mises stress at middle integration point
    stress = np.array(f["OP10/blank/element_shell_stress"])[-1, :, 1, :]

    # Get node positions at final timestep
    positions = np.array(f["OP10/blank/node_displacement"])[-1]
```
