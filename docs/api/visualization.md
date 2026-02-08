# Visualization

Visualization utilities for DDACS simulation data.

## Overview

The visualization module provides flexible plotting functions that work with extracted data.
This design allows you to see and understand your data before visualizing it.

### Basic Workflow

```python
from ddacs.utils import extract_mesh, extract_element_thickness
from ddacs.visualization import plot_mesh

# Step 1: Extract data (you see what you're working with)
vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1)
thickness = extract_element_thickness("simulation.h5", timestep=-1)

print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
print(f"Thickness range: {thickness.min():.3f} - {thickness.max():.3f} mm")

# Step 2: Visualize
ax, cbar = plot_mesh(
    vertices, faces,
    values=thickness,
    cmap="viridis",
    vmin=0.8, vmax=1.15,
    colorbar_label="Thickness [mm]"
)
```

## Functions

::: ddacs.visualization.plot_mesh

::: ddacs.visualization.plot_point_cloud

::: ddacs.visualization.plot_vectors

::: ddacs.visualization.plot_2d_projection

## Constants

### COMPONENT_COLORS

Default colors for simulation components:

```python
COMPONENT_COLORS = {
    "blank": "red",
    "die": "blue",
    "punch": "green",
    "binder": "orange",
}
```
