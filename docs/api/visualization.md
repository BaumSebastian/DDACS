# Visualization

Matplotlib plotting helpers that operate on numpy arrays. Pair them with `ddacs.open_h5` to read the input arrays from a single simulation. See the [Visualization tutorial](../tutorials/visualization.md) for end to end examples.

```python
import ddacs

with ddacs.open_h5(258864) as f:
    vertices = f["OP10/blank/node_displacement"][-1]
    faces = f["OP10/blank/element_shell_node_indexes"][:]
    thickness = f["OP10/blank/element_shell_thickness"][-1]

ax, cbar = ddacs.plot_mesh(vertices, faces, values=thickness, cmap="viridis")
```

## Functions

::: ddacs.visualization.plot_mesh

::: ddacs.visualization.plot_point_cloud

::: ddacs.visualization.plot_vectors

::: ddacs.visualization.plot_2d_projection

## Constants

```python
ddacs.visualization.COMPONENT_COLORS = {
    "blank": "red",
    "die": "blue",
    "punch": "green",
    "binder": "orange",
}
```
