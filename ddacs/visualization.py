"""Visualization utilities for DDACS simulation data.

Plotting helpers (`plot_mesh`, `plot_point_cloud`, `plot_vectors`,
`plot_2d_projection`) operate on numpy arrays the caller already pulled out
of an HDF5 file. Pair them with `ddacs.open_h5` for end-to-end inspection:

    >>> import ddacs, numpy as np
    >>> with ddacs.open_h5(258864, data_dir="./data") as f:
    ...     vertices = f["OP10/blank/node_displacement"][-1]      # (n_nodes, 3)
    ...     thickness = f["OP10/blank/element_shell_thickness"][-1]  # (n_elems,)
    ...     faces = f["OP10/blank/element_shell_node_indexes"][:]
    >>> ax, cbar = ddacs.plot_mesh(vertices, faces, values=thickness, cmap="viridis")
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Classic false-color FEM colormap (German: Falschfarbenbild). Red at the
# lower bound, green at the centre, blue at the upper bound. Pair with a
# centred `vmin`/`vmax` so the green band falls on the reference value
# (e.g. nominal sheet thickness).
FALSE_COLOR_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "false_color",
    ["#a50026", "#f46d43", "#fee08b", "#1a9850", "#74add1", "#4575b4", "#313695"],
)

# Component display settings
COMPONENT_COLORS = {
    "blank": "red",
    "die": "blue",
    "punch": "green",
    "binder": "orange",
}

COMPONENT_NAMES = {
    "blank": "Blank (Workpiece)",
    "die": "Die (Upper Tool)",
    "punch": "Punch (Lower Tool)",
    "binder": "Binder (Clamp)",
}

# Default view settings
DEFAULT_VIEW_ELEVATION = 30
DEFAULT_VIEW_AZIMUTH = 45
DEFAULT_AXIS_LIMITS = [0, 110]
DEFAULT_DPI = 150


def plot_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    axis_limits: list[float] | None = None,
    mirror: bool = False,
    shade: bool = True,
    **kwargs,
):
    """
    Plot 3D mesh with optional per-face coloring.

    Args:
        vertices: Vertex coordinates array of shape (n_vertices, 3).
        faces: Face indices array of shape (n_faces, 3).
        values: Optional per-face values for coloring (e.g., thickness, strain).
            If None, uses solid color.
        ax: Existing matplotlib 3D axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling. If None, uses min(values).
        vmax: Maximum value for color scaling. If None, uses max(values).
        colorbar_label: Label for the colorbar (e.g., "Thickness [mm]").
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].
        mirror: Reflect the quarter-symmetric mesh across the `x = 0` and
            `y = 0` planes to reconstruct the full part. Assumes the input
            data lives in the positive quadrant (i.e. raw OP10/OP20 node
            coordinates in mm). On normalised or centred data the four
            reflections overlay each other on the origin, so pass
            `mirror=False` in that case.
        shade: When True (default), matplotlib lights each face by its surface
            normal. The shading subtly biases the rendered face colour away
            from the pure cmap value, which can misalign the visual mesh with
            the colorbar. Pass `shade=False` for exact cmap-vs-colorbar match.
        **kwargs: Additional arguments passed to Poly3DCollection.
            Common options: color, edgecolor, linewidth, alpha.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Examples:
        Solid color mesh:
            >>> vertices, faces = extract_mesh("sim.h5", "blank", timestep=2)
            >>> ax = plot_mesh(vertices, faces, color="blue")

        Thickness distribution:
            >>> vertices, faces = extract_mesh("sim.h5", "blank", timestep=-1)
            >>> thickness = extract_element_thickness("sim.h5", timestep=-1)
            >>> ax, cbar = plot_mesh(
            ...     vertices, faces,
            ...     values=thickness,
            ...     cmap="viridis",
            ...     vmin=0.8, vmax=1.15,
            ...     colorbar_label="Thickness [mm]"
            ... )

        With custom edge styling:
            >>> ax = plot_mesh(vertices, faces, color="red",
            ...                edgecolor="black", linewidth=0.5)
    """
    own_axes = ax is None
    if own_axes:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS

    # Some HDF5 sources store element_shell_node_indexes as float; numpy fancy
    # indexing requires integer dtype, so coerce here so callers do not have to.
    faces = np.asarray(faces)
    if not np.issubdtype(faces.dtype, np.integer):
        faces = faces.astype(np.int64)

    if mirror:
        # Reflect across y, then across x, reversing face winding on each
        # reflection so per-face normals (and the shading) stay consistent.
        def _refl(v, sign):
            out = v.copy()
            out[:, 0] *= sign[0]
            out[:, 1] *= sign[1]
            return out

        v_y = _refl(vertices, (1, -1))
        f_y = (faces + len(vertices))[:, ::-1]
        vertices = np.vstack([vertices, v_y])
        faces = np.vstack([faces, f_y])
        if values is not None:
            values = np.concatenate([values, values])

        v_x = _refl(vertices, (-1, 1))
        f_x = (faces + len(vertices))[:, ::-1]
        vertices = np.vstack([vertices, v_x])
        faces = np.vstack([faces, f_x])
        if values is not None:
            values = np.concatenate([values, values])

    # Build face geometry
    face_vertices = vertices[faces]

    # Set defaults for kwargs
    kwargs["shade"] = shade

    # Determine face colors
    if values is not None:
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        face_colors = plt.cm.get_cmap(cmap)(norm(values))

        kwargs.setdefault("facecolors", face_colors)
        kwargs.setdefault("edgecolors", face_colors)

        collection = Poly3DCollection(face_vertices, **kwargs)
        ax.add_collection3d(collection)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.1)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label, labelpad=10)

    else:
        # Set default color if not provided
        kwargs.setdefault("facecolors", "red")
        if "edgecolors" not in kwargs:
            kwargs["edgecolors"] = kwargs["facecolors"]

        collection = Poly3DCollection(face_vertices, **kwargs)
        ax.add_collection3d(collection)

    if title:
        ax.set_title(title)
    # ISO 80000-2 / DIN 1313: variables italic, units upright.
    ax.set_xlabel(r"$x$ in mm", labelpad=10)
    ax.set_ylabel(r"$y$ in mm", labelpad=10)
    ax.set_zlabel(r"$z$ in mm", labelpad=3)

    # Limits and box aspect from the data we just drew → true isotropic scaling
    # per axis, so a 2 × 2 grid of components reads correctly even though each
    # subplot has its own extents.
    xmin, xmax = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    ymin, ymax = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    zmin, zmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    # Clamp dz so a flat geometry (binder, dz=0) still gets a non-singular zlim
    # and a meaningful box-aspect z component.
    dz_safe = max(dz, max(dx, dy) * 0.01 if max(dx, dy) > 0 else 1e-3)
    pad = 0.05
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_zlim(zmin - pad * dz_safe, zmax + pad * dz_safe)
    ax.set_box_aspect((max(dx, 1e-3), max(dy, 1e-3), dz_safe))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)
    if own_axes:
        ax.figure.text(0.16, 0.5, " ", va="center", ha="left")

    if values is not None:
        return ax, cbar
    return ax


def plot_point_cloud(
    coords: np.ndarray,
    values: np.ndarray | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    axis_limits: list[float] | None = None,
    mirror: bool = False,
    **kwargs,
):
    """
    Plot 3D point cloud with optional per-point coloring.

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        values: Optional per-point values for coloring.
            If None, uses solid color.
        ax: Existing matplotlib 3D axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar.
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].
        mirror: Reflect the quarter-symmetric data across the `x = 0` and
            `y = 0` planes to reconstruct the full part. Assumes the input
            data lives in the positive quadrant (raw OP10/OP20 coordinates
            in mm). On normalised or centred data the four reflections
            overlay each other on the origin, so pass `mirror=False`.
        **kwargs: Additional arguments passed to ax.scatter.
            Common options: c (color), s (size), alpha, marker.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Examples:
        Simple point cloud:
            >>> coords = extract_point_cloud("sim.h5", "blank", timestep=2)
            >>> ax = plot_point_cloud(coords, c="blue", s=2)

        Colored by springback:
            >>> coords, displacement = extract_point_springback("sim.h5")
            >>> magnitude = np.linalg.norm(displacement, axis=1)
            >>> ax, cbar = plot_point_cloud(
            ...     coords,
            ...     values=magnitude,
            ...     colorbar_label="Springback [mm]"
            ... )
    """
    coords = np.asarray(coords, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64) if values is not None else None

    if mirror:
        # Reflect across y, then across x. Values (per-point scalars) are
        # concatenated alongside so the colour map keeps lining up.
        v_y = coords.copy()
        v_y[:, 1] *= -1
        coords = np.vstack([coords, v_y])
        if values_arr is not None:
            values_arr = np.concatenate([values_arr, values_arr])

        v_x = coords.copy()
        v_x[:, 0] *= -1
        coords = np.vstack([coords, v_x])
        if values_arr is not None:
            values_arr = np.concatenate([values_arr, values_arr])

    own_axes = ax is None
    if own_axes:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    kwargs.setdefault("s", 1.0)
    kwargs.setdefault("alpha", 0.8)
    cbar = None

    if values_arr is not None:
        vmin = vmin if vmin is not None else float(np.min(values_arr))
        vmax = vmax if vmax is not None else float(np.max(values_arr))
        norm = Normalize(vmin=vmin, vmax=vmax)
        point_colors = plt.colormaps.get_cmap(cmap)(norm(values_arr))
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=point_colors, **kwargs)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label, labelpad=10)
    else:
        kwargs.setdefault("c", "red")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **kwargs)

    if title:
        ax.set_title(title)
    # ISO 80000-2 / DIN 1313: variables italic, units upright.
    ax.set_xlabel(r"$x$ in mm", labelpad=10)
    ax.set_ylabel(r"$y$ in mm", labelpad=10)
    ax.set_zlabel(r"$z$ in mm", labelpad=3)

    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    zmin, zmax = float(coords[:, 2].min()), float(coords[:, 2].max())
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    pad = 0.05
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_zlim(zmin - pad * dz, zmax + pad * dz)
    ax.set_box_aspect(
        (max(dx, 1e-3), max(dy, 1e-3), max(dz, max(dx, dy) * 0.01 if max(dx, dy) > 0 else 1e-3))
    )
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)
    if own_axes:
        ax.figure.text(0.16, 0.5, " ", va="center", ha="left")

    return (ax, cbar) if values_arr is not None else ax


def plot_vectors(
    coords: np.ndarray,
    vectors: np.ndarray,
    values: np.ndarray | None = None,
    ax=None,
    figsize: tuple[float, float] | None = None,
    step: int = 25,
    scale: float = 10.0,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    axis_limits: list[float] | None = None,
    mirror: bool = False,
    arrow_kwargs: dict | None = None,
    outline_kwargs: dict | None = None,
):
    """
    Plot a 3D vector field (arrows only) for displacement visualization.

    `plot_vectors` is a pure arrow renderer — it does not draw an underlying
    point cloud or mesh. To layer arrows on a surface, draw `plot_mesh` (or
    `plot_point_cloud`) first on an axis, then pass the same `ax` here.

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        vectors: Vector components array of shape (n_points, 3).
        values: Optional per-point values for colouring the arrows by magnitude.
            If None, every arrow uses `arrow_kwargs["color"]` (default "red").
        ax: Existing matplotlib 3D axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        step: Subsampling step (every Nth point) for arrow density.
        scale: Scaling factor for arrow length.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar (if values is provided).
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].
        mirror: Reflect the quarter-symmetric data across the `x = 0` and
            `y = 0` planes to reconstruct the full part. Assumes the input
            data lives in the positive quadrant (raw OP10/OP20 coordinates
            in mm). On normalised or centred data the four reflections
            overlay each other on the origin, so pass `mirror=False`.
        arrow_kwargs: Additional arguments passed to `ax.quiver` for the
            coloured arrows on top. Common options: color (only used when
            values is None), alpha, arrow_length_ratio.
        outline_kwargs: Additional arguments passed to `ax.quiver` for the
            black outline drawn behind each arrow. Defaults: color="black",
            linewidth=1.5, alpha matches arrow_kwargs. Pass
            `outline_kwargs={"linewidth": 0}` to suppress the outline.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Examples:
        Simple vectors:
            >>> ax = plot_vectors(coords, displacement, step=50, scale=15)

        Arrows coloured by springback magnitude:
            >>> magnitude = np.linalg.norm(displacement, axis=1)
            >>> ax, cbar = plot_vectors(
            ...     coords, displacement,
            ...     values=magnitude,
            ...     colorbar_label="Springback in mm",
            ... )

        Arrows over a mesh surface (compose on a shared axis):
            >>> ax = plot_mesh(nodes, faces, ax=my_ax, facecolors="lightgrey")
            >>> ax, cbar = plot_vectors(
            ...     coords, displacement,
            ...     ax=my_ax,
            ...     values=magnitude,
            ...     colorbar_label="Springback in mm",
            ... )
    """
    coords = np.asarray(coords, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64) if values is not None else None

    if mirror:
        # Reflect across y, then across x. Both coords AND vector components
        # are reflected on each axis so arrows still point outward correctly.
        for sign in ((1, -1), (-1, 1)):
            cc = coords.copy()
            vv = vectors.copy()
            cc[:, 0] *= sign[0]
            cc[:, 1] *= sign[1]
            vv[:, 0] *= sign[0]
            vv[:, 1] *= sign[1]
            coords = np.vstack([coords, cc])
            vectors = np.vstack([vectors, vv])
            if values_arr is not None:
                values_arr = np.concatenate([values_arr, values_arr])

    own_axes = ax is None
    if own_axes:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    arrow_kwargs = dict(arrow_kwargs or {})
    arrow_kwargs.setdefault("alpha", 0.8)
    arrow_kwargs.setdefault("arrow_length_ratio", 0.4)
    explicit_color = arrow_kwargs.pop("color", None)

    outline_kwargs = dict(outline_kwargs or {})
    outline_kwargs.setdefault("color", "black")
    outline_kwargs.setdefault("linewidth", 1.5)
    outline_kwargs.setdefault("alpha", arrow_kwargs["alpha"])
    outline_kwargs.setdefault("arrow_length_ratio", arrow_kwargs["arrow_length_ratio"])
    cbar = None

    # Subsample for arrow density (post-mirror so the full mirrored set is sampled)
    coords_sub = coords[::step]
    vectors_sub = vectors[::step] * scale

    # Draw a thicker outline first so the coloured arrows on top look like
    # solid-fill-with-border. Suppress by passing `outline_kwargs={"linewidth": 0}`.
    ax.quiver(
        coords_sub[:, 0],
        coords_sub[:, 1],
        coords_sub[:, 2],
        vectors_sub[:, 0],
        vectors_sub[:, 1],
        vectors_sub[:, 2],
        **outline_kwargs,
    )

    if values_arr is not None and explicit_color is None:
        # Per-arrow colours from the cmap.
        values_sub = values_arr[::step]
        vmin = vmin if vmin is not None else float(np.min(values_arr))
        vmax = vmax if vmax is not None else float(np.max(values_arr))
        norm = Normalize(vmin=vmin, vmax=vmax)
        arrow_colors = plt.colormaps.get_cmap(cmap)(norm(values_sub))
        ax.quiver(
            coords_sub[:, 0],
            coords_sub[:, 1],
            coords_sub[:, 2],
            vectors_sub[:, 0],
            vectors_sub[:, 1],
            vectors_sub[:, 2],
            colors=arrow_colors,
            **arrow_kwargs,
        )
    else:
        # Solid colour — either explicitly requested or no values to map.
        ax.quiver(
            coords_sub[:, 0],
            coords_sub[:, 1],
            coords_sub[:, 2],
            vectors_sub[:, 0],
            vectors_sub[:, 1],
            vectors_sub[:, 2],
            color=explicit_color or "red",
            **arrow_kwargs,
        )

    if values_arr is not None:
        # Colorbar is always drawn when `values` is provided, even when the
        # arrows themselves render in a solid colour, so the value range is
        # still legible.
        vmin = vmin if vmin is not None else float(np.min(values_arr))
        vmax = vmax if vmax is not None else float(np.max(values_arr))
        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label, labelpad=10)

    if own_axes:
        if title:
            ax.set_title(title)
        # ISO 80000-2 / DIN 1313: variables italic, units upright.
        ax.set_xlabel(r"$x$ in mm", labelpad=10)
        ax.set_ylabel(r"$y$ in mm", labelpad=10)
        ax.set_zlabel(r"$z$ in mm", labelpad=3)

        xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
        ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
        zmin, zmax = float(coords[:, 2].min()), float(coords[:, 2].max())
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        pad = 0.05
        ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
        ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
        ax.set_zlim(zmin - pad * dz, zmax + pad * dz)
        ax.set_box_aspect(
            (max(dx, 1e-3), max(dy, 1e-3), max(dz, max(dx, dy) * 0.01 if max(dx, dy) > 0 else 1e-3))
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)
        ax.figure.text(0.16, 0.5, " ", va="center", ha="left")

    return (ax, cbar) if values_arr is not None else ax


def plot_2d_projection(
    coords: np.ndarray,
    values: np.ndarray | None = None,
    projection: str = "xy",
    ax=None,
    figsize: tuple[float, float] | None = None,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    **kwargs,
):
    """
    Plot 2D projection of point cloud (top view, side view, etc.).

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        values: Optional per-point values for coloring.
        projection: Which 2D plane to project onto ("xy", "xz", "yz").
        ax: Existing matplotlib axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar.
        title: Plot title.
        **kwargs: Additional arguments passed to ax.scatter.
            Common options: c (color), s (size), alpha, marker.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Example:
        >>> coords, displacement = extract_point_springback("sim.h5")
        >>> magnitude = np.linalg.norm(displacement, axis=1)
        >>> ax, cbar = plot_2d_projection(
        ...     coords, values=magnitude,
        ...     projection="xy",
        ...     colorbar_label="Springback [mm]",
        ...     s=2, alpha=0.8
        ... )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)

    # Select projection axes
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    label_map = {"xy": ("X [mm]", "Y [mm]"), "xz": ("X [mm]", "Z [mm]"), "yz": ("Y [mm]", "Z [mm]")}

    idx1, idx2 = axis_map.get(projection, (0, 1))
    xlabel, ylabel = label_map.get(projection, ("X [mm]", "Y [mm]"))

    # Set defaults
    kwargs.setdefault("s", 2.0)
    kwargs.setdefault("alpha", 0.8)

    if values is not None:
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        point_colors = plt.cm.get_cmap(cmap)(norm(values))

        ax.scatter(coords[:, idx1], coords[:, idx2], c=point_colors, **kwargs)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        kwargs.setdefault("c", "red")
        ax.scatter(coords[:, idx1], coords[:, idx2], **kwargs)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    if values is not None:
        return ax, cbar
    return ax
