---
created: 2025-03-26
updated: 2025-08-13
---

# Deep Drawing Simulations Dataset

## What is Deep Drawing?

Deep drawing is a manufacturing process that transforms flat metal sheets into 3D parts by pressing them through a die. Think of shaping aluminum foil with a muffin tin, but with precise control over forces and material flow.

![[Pasted image 20250326120029.png]]
*Simulation showing the four main components: binder (top), punch (moving down), blank (metal sheet), and die (bottom)*

**Process Overview:**
1. **Blank** (metal sheet) is placed between tools
2. **Punch** presses the metal into the **die** cavity
3. **Binder** holds the edges to control material flow
4. The result is a formed 3D part

## Dataset Overview

This dataset contains 32,071 finite element simulations of deep drawing processes with varying:
- **Materials**: Different sheet thicknesses and properties
- **Geometries**: Rectangular, concave, and convex part shapes  
- **Process conditions**: Friction coefficients and blank holder forces

**Applications:**
- Train AI models for manufacturing optimization
- Predict forming failures and defects
- Optimize process parameters

## Data Access

**This repository**: DDACS Python package for data loading and analysis utilities

**Dataset download**: ~1TB total available via [DaRUS repository](https://github.com/BaumSebastian/DaRUS-Dataset-Interaction)

**File format**: HDF5 with complete simulation data in single files

**What you get**: Stress, strain, displacement, and force data for 32,071 simulations covering the complete forming and springback process

## Quick Start

```python
from ddacs import iter_ddacs
from ddacs.utils import display_structure

# Count available simulations  
from ddacs import count_available_simulations
count = count_available_simulations("data/")
print(f"Available simulations: {count}")
# Output: Available simulations: 32071

# Iterate through data
for sim_id, metadata, h5_path in iter_ddacs("data/"):
    display_structure(h5_path)
    break  # Just show first one
```

**Example output:**
```
HDF5 Structure: 113525.h5
============================================================
Root attributes: Geometry_Parameters=[40.  5. 10.], Material_Parameters=[1.05e+00 9.00e-02 9.50e-01 2.00e+05]
----------------------------------------
OP10/ (Group)
    binder/ (Group)
        element_shell_ids (Dataset: shape=(97,), dtype=int64)
        node_coordinates (Dataset: shape=(112, 3), dtype=float64)
        node_displacement (Dataset: shape=(3, 112, 3), dtype=float64)
        ...
    blank/ (Group)
        element_shell_stress (Dataset: shape=(4, 11025, 3, 6), dtype=float64)
        element_shell_strain (Dataset: shape=(4, 11025, 2, 6), dtype=float64)
        node_displacement (Dataset: shape=(4, 11236, 3), dtype=float64)
        ...
    die/ (Group)
        ...
    punch/ (Group)
        ...
    general/ (Group)
        global_timesteps (Dataset: shape=(3,), dtype=float64)
        ...
OP20/ (Group)
    blank/ (Group)  # Only blank geometry - no tools
        element_shell_stress (Dataset: shape=(2, 3994, 3, 6), dtype=float64)
        node_displacement (Dataset: shape=(2, 4204, 3), dtype=float64)
        ...
```

## Dataset Structure

Each simulation is stored as an HDF5 file containing:

### Basic Structure
- **Root attributes**: Geometry and material parameters
- **OP10/**: Forming operation (3 timesteps for tools, 4 for blank)
  - `blank/`: The deforming metal sheet (most important) - 4 timesteps
  - `die/`, `punch/`, `binder/`: Tool geometries and motion - 3 timesteps
  - `general/`: Global simulation data (energies, timesteps)
- **OP20/**: Springback analysis after tool removal (2 timesteps, blank only)

**Timestep explanation:**
- **OP10 Tools (3 steps)**: Initial → Mid-forming → End of forming
- **OP10 Blank (4 steps)**: Initial → Mid-forming → End of forming → After springback
- **OP20 (2 steps)**: Tool removal analysis → Final springback state

### Key Data Types
- **Coordinates**: Node positions defining geometry
- **Displacements**: How each point moves during forming
- **Stress/Strain**: Material response and deformation
- **Forces**: Applied loads and reactions

---

## Parameters

### Material Parameters
- **SHTK**: Sheet thickness [0.95-1.0 mm]
- **MAT**: Material scaling factor [0.9-1.1]

### Process Parameters  
- **FC**: Friction coefficient [0.05-0.15]
- **BF**: Blank holder force [100,000-500,000 N]

### Geometry Parameters
- **GEO**: Shape type (R=Rectangular, V=Concave, X=Convex)
- **RAD**: Characteristic radius [30-150 mm]

---

## Advanced Details

<details>
<summary><strong>Complete Data Field Reference</strong></summary>

### Understanding the Data Fields

#### Geometry and Connectivity
| Field | Dimensions | What it contains |
|-------|------------|------------------|
| `node_coordinates` | (n, 3) | X, Y, Z positions of every point on the component surface |
| `node_ids` | (n,) | Unique reference numbers for each point |
| `element_shell_ids` | (m,) | Unique reference numbers for each surface patch |
| `element_shell_node_indexes` | (m, 4) | Which 4 points connect to form each rectangular surface patch |

#### Motion and Deformation  
| Field | Dimensions | What it contains |
|-------|------------|------------------|
| `node_displacement` | (t, n, 3) | How far each point moved from its starting position |
| `node_velocity` | (t, n, 3) | How fast each point is moving in X, Y, Z directions |
| `node_acceleration` | (t, n, 3) | How quickly the velocity is changing at each point |

#### Material Response
| Field | Dimensions | What it contains |
|-------|------------|------------------|
| `element_shell_stress` | (t, m, layers, 6) | Internal forces within the material (σxx, σyy, σzz, τxy, τyz, τzx) |
| `element_shell_strain` | (t, m, layers, 6) | How much the material is stretched or compressed (εxx, εyy, εzz, γxy, γyz, γzx) |
| `element_shell_thickness` | (t, m) | Current thickness of each patch (shows thinning during forming) |
| `element_shell_effective_plastic_strain` | (t, m, 3) | Permanent deformation that won't recover (εp_eff components) |

#### Forces and Energy
| Field | Dimensions | What it contains |
|-------|------------|------------------|
| `element_shell_bending_moment` | (t, m, 3) | Twisting forces causing bending (Mx, My, Mz) |
| `element_shell_normal_force` | (t, m, 3) | Forces pushing perpendicular to the surface (Fx, Fy, Fz) |
| `element_shell_shear_force` | (t, m, 2) | Forces sliding parallel to the surface (Fxy, Fyz) |
| `element_shell_internal_energy` | (t, m) | Energy absorbed by deformation (U) |

*Where: m=surface patches, n=points, t=timesteps, layers=through thickness*

### OP10 - Forming Operation

#### Blank Component
The most important component containing comprehensive deformation data:
- Complete stress/strain history through thickness
- Plastic deformation and thinning
- Force distributions and energy absorption
- 4 timesteps including final springback state

#### Tool Components (Die, Punch, Binder)
- **Die**: Stationary lower tool defining final geometry
- **Punch**: Moving upper tool with prescribed displacement
- **Binder**: Force-controlled blank holder preventing wrinkling

Tool data includes geometry, prescribed motion, and contact forces.

#### General Data
Global simulation information:
- Energy conservation (kinetic, internal, total)
- Part-wise mass and velocity tracking
- Timestep information
- Material type assignments

### OP20 - Springback Analysis
Post-forming analysis after tool removal (blank component only):
- Residual stress and strain states
- Final part geometry after elastic recovery
- 2 timesteps showing springback progression
- No tool data (tools are removed for springback analysis)

</details>

<details>
<summary><strong>Simulation Technical Details</strong></summary>

### Finite Element Setup
- **Material**: DP600 dual-phase steel
- **Elements**: Fully integrated shells (ELFORM=16, NIP=7)
- **Contact**: One-way forming contact algorithm
- **Material Model**: Kinematic hardening with transverse anisotropy

### Geometry Specifications
- **Part dimensions**: 210×210 mm base, 30 mm depth
- **Radius variations**:
  - Rectangular: 30-40 mm corner radius
  - Concave: 50-150 mm concave radius  
  - Convex: 100-150 mm convex radius
- **Wall angle**: 10-20°
- **Bottom radius**: 5-10 mm

</details>

