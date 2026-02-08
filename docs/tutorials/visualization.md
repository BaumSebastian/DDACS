# Visualization Tutorial

This tutorial demonstrates how to visualize DDACS simulation results using
the built-in visualization utilities.

## Setup

Install DDACS:

```bash
pip install ddacs
```

Matplotlib is included as a core dependency.

## Basic Workflow

The visualization module is designed to work with extracted data.
This lets you inspect your data before plotting:

```python
from ddacs.utils import extract_mesh, extract_element_thickness
from ddacs.visualization import plot_mesh

# Step 1: Extract data
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1)
thickness = extract_element_thickness("simulation.h5", timestep=-1)

# Step 2: Inspect
print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
print(f"Thickness: {thickness.min():.3f} - {thickness.max():.3f} mm")

# Step 3: Visualize
ax, cbar = plot_mesh(vertices, faces, values=thickness)
```

## Mesh Visualization

### Solid Color Mesh

```python
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh
import matplotlib.pyplot as plt

# Extract mesh at final forming timestep
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=2)

# Plot with solid color
ax = plot_mesh(vertices, faces, color="blue", title="Blank Mesh")
plt.tight_layout()
plt.show()
```

### Thickness Distribution

```python
from ddacs.utils import extract_mesh, extract_element_thickness
from ddacs.visualization import plot_mesh

# Extract mesh and thickness
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1)
thickness = extract_element_thickness("simulation.h5", timestep=-1)

# Plot with thickness coloring
ax, cbar = plot_mesh(
    vertices, faces,
    values=thickness,
    cmap="viridis",
    vmin=0.8,
    vmax=1.15,
    colorbar_label="Thickness [mm]",
    title="Thickness Distribution"
)
plt.tight_layout()
plt.show()
```

### Stress Visualization

```python
from ddacs.utils import extract_mesh, extract_element_stress, compute_von_mises
from ddacs.visualization import plot_mesh

# Extract mesh and stress
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1)
stress = extract_element_stress("simulation.h5", timestep=-1)
von_mises = compute_von_mises(stress)

# Plot von Mises stress
ax, cbar = plot_mesh(
    vertices, faces,
    values=von_mises,
    cmap="plasma",
    colorbar_label="von Mises Stress [MPa]",
    title="Stress Distribution"
)
plt.show()
```

## Point Cloud Visualization

### Simple Point Cloud

```python
from ddacs.utils import extract_point_cloud
from ddacs.visualization import plot_point_cloud

# Extract coordinates
coords = extract_point_cloud("simulation.h5", "blank", timestep=2)

# Plot
ax = plot_point_cloud(coords, color="red", point_size=0.5)
plt.show()
```

### Multiple Components

```python
from ddacs.utils import extract_point_cloud
from ddacs.visualization import COMPONENT_COLORS
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection="3d")

for component in ["blank", "die", "punch", "binder"]:
    coords = extract_point_cloud("simulation.h5", component, timestep=2)
    ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=COMPONENT_COLORS[component],
        label=component,
        s=0.5,
        alpha=0.7
    )

ax.legend()
plt.show()
```

## Springback Visualization

Springback is the elastic recovery after tool removal.

### 3D Springback Magnitude

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_point_cloud

# Extract springback data
coords, displacement = extract_point_springback("simulation.h5", operation=20)

# Calculate magnitude
magnitude = np.linalg.norm(displacement, axis=1)

print(f"Springback range: {magnitude.min():.3f} - {magnitude.max():.3f} mm")

# Visualize
ax, cbar = plot_point_cloud(
    coords,
    values=magnitude,
    cmap="plasma",
    vmin=0.0,
    vmax=1.4,
    colorbar_label="Springback [mm]",
    title="Springback Magnitude"
)
plt.show()
```

### Springback Vector Field

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_vectors

# Extract springback vectors
coords, displacement = extract_point_springback("simulation.h5", operation=20)
magnitude = np.linalg.norm(displacement, axis=1)

# Plot with arrows and colored by magnitude
ax, cbar = plot_vectors(
    coords, displacement,
    values=magnitude,
    step=25,
    scale=10.0,
    arrow_color="black",
    colorbar_label="Springback [mm]",
    title="Springback Vectors"
)
plt.show()
```

### 2D Top View

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_2d_projection

# Extract data
coords, displacement = extract_point_springback("simulation.h5", operation=20)
magnitude = np.linalg.norm(displacement, axis=1)

# 2D projection
ax, cbar = plot_2d_projection(
    coords,
    values=magnitude,
    projection="xy",
    colorbar_label="Springback [mm]",
    title="Springback (Top View)"
)
plt.show()
```

## Comparing Timesteps

Visualize the forming process across timesteps:

```python
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={"projection": "3d"})

timestep_labels = ["Initial", "Forming", "Max Depth", "Springback"]

for i, (ax, label) in enumerate(zip(axes, timestep_labels)):
    vertices, faces = extract_mesh("simulation.h5", "blank", timestep=i)
    plot_mesh(vertices, faces, ax=ax, title=label, color="steelblue")

plt.tight_layout()
plt.show()
```

## Customizing Plots

### Using Existing Axes

```python
import matplotlib.pyplot as plt
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh

# Create your own figure and axes
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Pass axes to plot functions
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=2)
plot_mesh(vertices, faces, ax=ax1, title="OP10 Final", color="blue")

vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1, operation=20)
plot_mesh(vertices, faces, ax=ax2, title="OP20 Final", color="red")

plt.tight_layout()
plt.show()
```

### Custom Axis Limits

```python
from ddacs.visualization import plot_mesh

ax = plot_mesh(
    vertices, faces,
    axis_limits=[0, 150],  # Custom range
    title="Extended View"
)
```

### Publication-Ready Figures

```python
from ddacs.visualization import plot_mesh

# Use appropriate size for papers
ax, cbar = plot_mesh(
    vertices, faces,
    values=thickness,
    figsize=(5, 3.5),
    colorbar_label="Thickness [mm]"
)

plt.savefig("figure.pdf", dpi=300, bbox_inches="tight")
```

## Next Steps

- Explore the [API Reference](../api/visualization.md) for all options
- See `notebooks/dataset_demo.ipynb` for more advanced examples
- Check the [Dataset Overview](../dataset.md) for physics background
