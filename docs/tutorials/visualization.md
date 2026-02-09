# Visualization Tutorial

This tutorial demonstrates how to visualize DDACS simulation results using
the built-in visualization utilities.

All examples use simulation `16336.h5` from the dataset for reproducibility.

---

### Point Cloud - Multiple Components

Visualize all components of the deep drawing setup together.

```python
import matplotlib.pyplot as plt
from ddacs.utils import extract_point_cloud
from ddacs.visualization import plot_point_cloud, COMPONENT_COLORS, COMPONENT_NAMES

h5_path = "./data/h5/16336.h5"

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111, projection="3d")

for component in COMPONENT_COLORS.keys():
    coords = extract_point_cloud(h5_path, component, timestep=2)
    alpha = 0.3 if component == "blank" else 0.7
    plot_point_cloud(
        coords,
        ax=ax,
        c=COMPONENT_COLORS[component],
        s=0.5,
        alpha=alpha,
        axis_limits=[0, 110],
    )
    ax.scatter([], [], [], c=COMPONENT_COLORS[component], label=COMPONENT_NAMES[component])
    print(f"{component}: {coords.shape[0]} nodes")

ax.set_title("Deep Drawing Setup - Simulation 16336")
ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize="small")
plt.savefig("point_cloud.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

Output:
```
blank: 11041 nodes
die: 1089 nodes
punch: 529 nodes
binder: 168 nodes
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_point_cloud.png" width="700">

---

### Mesh Visualization

Plot the blank mesh with shading for better 3D perception.

```python
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

vertices, triangles = extract_mesh(h5_path, "blank", timestep=2)
print(f"Mesh: {vertices.shape[0]} vertices, {triangles.shape[0]} faces")

ax = plot_mesh(
    vertices, triangles,
    facecolors="red",
    edgecolors="red",
    linewidth=0.3,
    title="Blank Mesh - Simulation 16336",
    figsize=(12, 5),
    axis_limits=[0, 110],
)
plt.savefig("mesh.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

Output:
```
Mesh: 11041 vertices, 43296 faces
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_mesh.png" width="700">

---

### Thickness Distribution

Visualize material thinning during the forming process.

```python
from ddacs.utils import extract_mesh, extract_element_thickness
from ddacs.visualization import plot_mesh
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

vertices, triangles = extract_mesh(h5_path, "blank", timestep=-1)
thickness = extract_element_thickness(h5_path, timestep=-1)
print(f"Thickness range: {thickness.min():.4f} - {thickness.max():.4f} mm")

ax, cbar = plot_mesh(
    vertices, triangles,
    values=thickness,
    cmap="viridis",
    vmin=0.8,
    vmax=1.15,
    colorbar_label="Thickness [mm]",
    title="Thickness Distribution - Simulation 16336",
    figsize=(12, 5),
    axis_limits=[0, 110],
    edgecolors="face",
    linewidth=0.3,
)
plt.savefig("thickness.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

Output:
```
Thickness range: 0.8734 - 1.0892 mm
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_thickness.png" width="700">

---

### Springback Magnitude

Visualize elastic recovery after tool removal.

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_point_cloud
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

coords, displacement = extract_point_springback(h5_path, operation=20)
magnitude = np.linalg.norm(displacement, axis=1)
print(f"Springback range: {magnitude.min():.4f} - {magnitude.max():.4f} mm")

ax, cbar = plot_point_cloud(
    coords,
    values=magnitude,
    cmap="plasma",
    vmin=0.0,
    vmax=1.4,
    colorbar_label="Springback [mm]",
    title="Springback Magnitude - Simulation 16336",
    figsize=(12, 5),
    axis_limits=[0, 110],
    s=1,
    alpha=0.8,
)
plt.savefig("springback.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

Output:
```
Springback range: 0.0000 - 1.2847 mm
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_springback.png" width="700">

---

### Springback Vector Field

Show displacement direction with arrows.

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_vectors
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

coords, displacement = extract_point_springback(h5_path, operation=20)
magnitude = np.linalg.norm(displacement, axis=1)

ax, cbar = plot_vectors(
    coords,
    displacement,
    values=magnitude,
    step=25,
    scale=10.0,
    cmap="plasma",
    vmin=0.0,
    vmax=1.4,
    colorbar_label="Springback [mm]",
    title="Springback Vectors - Simulation 16336",
    figsize=(12, 5),
    axis_limits=[0, 110],
    point_kwargs={"s": 1, "alpha": 0.6},
    arrow_kwargs={"color": "black", "alpha": 0.8},
)
plt.savefig("vectors.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_vectors.png" width="700">

---

### 2D Top View Projection

Project springback data onto a 2D plane.

```python
import numpy as np
from ddacs.utils import extract_point_springback
from ddacs.visualization import plot_2d_projection
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

coords, displacement = extract_point_springback(h5_path, operation=20)
magnitude = np.linalg.norm(displacement, axis=1)

ax, cbar = plot_2d_projection(
    coords,
    values=magnitude,
    projection="xy",
    cmap="plasma",
    vmin=0.0,
    vmax=1.4,
    colorbar_label="Springback [mm]",
    title="Springback Top View - Simulation 16336",
    figsize=(12, 5),
    s=2,
    alpha=0.8,
)
plt.savefig("2d_projection.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
```

<img src="https://raw.githubusercontent.com/BaumSebastian/DDACS/main/docs/images/example_2d_projection.png" width="700">

---

## Additional Features

### Using Existing Axes

```python
import matplotlib.pyplot as plt
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh

h5_path = "./data/h5/16336.h5"

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

vertices, faces = extract_mesh(h5_path, "blank", timestep=2)
plot_mesh(vertices, faces, ax=ax1, title="OP10 Final", facecolors="blue")

vertices, faces = extract_mesh(h5_path, "blank", timestep=-1, operation=20)
plot_mesh(vertices, faces, ax=ax2, title="OP20 Final", facecolors="red")

plt.tight_layout()
plt.show()
```

### Custom Axis Limits

```python
from ddacs.utils import extract_mesh
from ddacs.visualization import plot_mesh

h5_path = "./data/h5/16336.h5"

vertices, faces = extract_mesh(h5_path, "blank", timestep=2)
ax = plot_mesh(
    vertices, faces,
    axis_limits=[0, 150],
    title="Extended View"
)
```

### Publication-Ready Figures

```python
from ddacs.utils import extract_mesh, extract_element_thickness
from ddacs.visualization import plot_mesh
import matplotlib.pyplot as plt

h5_path = "./data/h5/16336.h5"

vertices, faces = extract_mesh(h5_path, "blank", timestep=-1)
thickness = extract_element_thickness(h5_path, timestep=-1)

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
- See [`notebooks/dataset_demo.ipynb`](https://github.com/BaumSebastian/DDACS/tree/main/notebooks/dataset_demo.ipynb) for more advanced examples
- Check the [Dataset Overview](../dataset.md) for physics background
