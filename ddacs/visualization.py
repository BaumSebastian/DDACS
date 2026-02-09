"""
Visualization utilities for DDACS simulation data.

This module provides functions for visualizing deep drawing simulation results
including meshes, point clouds, and springback analysis.

Example workflow:
    >>> from ddacs.utils import extract_mesh, extract_element_thickness
    >>> from ddacs.visualization import plot_mesh
    >>>
    >>> # Extract data (user sees what they're working with)
    >>> vertices, faces = extract_mesh("simulation.h5", "blank", timestep=-1)
    >>> thickness = extract_element_thickness("simulation.h5", timestep=-1)
    >>>
    >>> # Visualize with thickness coloring
    >>> ax, cbar = plot_mesh(vertices, faces, values=thickness, cmap="viridis")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        **kwargs: Additional arguments passed to Poly3DCollection.
            Common options: color, edgecolor, linewidth, alpha, shade.

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
    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS

    # Build face geometry
    face_vertices = vertices[faces]

    # Set defaults for kwargs
    kwargs.setdefault("shade", True)

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
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        # Set default color if not provided
        kwargs.setdefault("facecolors", "red")
        if "edgecolors" not in kwargs:
            kwargs["edgecolors"] = kwargs["facecolors"]

        collection = Poly3DCollection(face_vertices, **kwargs)
        ax.add_collection3d(collection)

    # Configure axes
    if title:
        ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)

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
    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS

    # Set defaults
    kwargs.setdefault("s", 1.0)
    kwargs.setdefault("alpha", 0.8)

    if values is not None:
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        point_colors = plt.cm.get_cmap(cmap)(norm(values))

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=point_colors,
            **kwargs,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        kwargs.setdefault("c", "red")
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            **kwargs,
        )

    if title:
        ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)

    if values is not None:
        return ax, cbar
    return ax


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
    show_points: bool = True,
    point_kwargs: dict | None = None,
    arrow_kwargs: dict | None = None,
):
    """
    Plot 3D vector field (quiver plot) for displacement visualization.

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        vectors: Vector components array of shape (n_points, 3).
        values: Optional per-point values for coloring the point cloud.
            If None, uses solid point color.
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
        show_points: Whether to show point cloud beneath vectors.
        point_kwargs: Additional arguments passed to ax.scatter for points.
            Common options: c (color), s (size), alpha.
        arrow_kwargs: Additional arguments passed to ax.quiver for arrows.
            Common options: color, alpha, arrow_length_ratio.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Examples:
        Simple vectors:
            >>> coords, displacement = extract_point_springback("sim.h5")
            >>> ax = plot_vectors(coords, displacement, step=50, scale=15)

        With springback magnitude coloring:
            >>> coords, displacement = extract_point_springback("sim.h5")
            >>> magnitude = np.linalg.norm(displacement, axis=1)
            >>> ax, cbar = plot_vectors(
            ...     coords, displacement,
            ...     values=magnitude,
            ...     colorbar_label="Springback [mm]",
            ...     arrow_kwargs={"color": "darkred", "alpha": 0.8}
            ... )
    """
    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS
    point_kwargs = point_kwargs or {}
    arrow_kwargs = arrow_kwargs or {}
    cbar = None

    # Set defaults for points
    point_kwargs.setdefault("s", 1.0)
    point_kwargs.setdefault("alpha", 0.5)

    # Set defaults for arrows
    arrow_kwargs.setdefault("color", "red")
    arrow_kwargs.setdefault("alpha", 0.8)
    arrow_kwargs.setdefault("arrow_length_ratio", 0.4)

    if show_points:
        if values is not None:
            vmin = vmin if vmin is not None else values.min()
            vmax = vmax if vmax is not None else values.max()

            norm = Normalize(vmin=vmin, vmax=vmax)
            point_colors = plt.cm.get_cmap(cmap)(norm(values))

            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=point_colors,
                **point_kwargs,
            )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            if colorbar_label:
                cbar.set_label(colorbar_label)
        else:
            point_kwargs.setdefault("c", "blue")
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                **point_kwargs,
            )

    # Subsample for arrow density
    coords_sub = coords[::step]
    vectors_sub = vectors[::step] * scale

    # Draw black outline first (slightly thicker)
    ax.quiver(
        coords_sub[:, 0],
        coords_sub[:, 1],
        coords_sub[:, 2],
        vectors_sub[:, 0],
        vectors_sub[:, 1],
        vectors_sub[:, 2],
        color="black",
        alpha=arrow_kwargs.get("alpha", 0.8),
        arrow_length_ratio=arrow_kwargs.get("arrow_length_ratio", 0.4),
        linewidth=1.5,
    )

    # Draw colored arrows on top
    ax.quiver(
        coords_sub[:, 0],
        coords_sub[:, 1],
        coords_sub[:, 2],
        vectors_sub[:, 0],
        vectors_sub[:, 1],
        vectors_sub[:, 2],
        **arrow_kwargs,
    )

    if title:
        ax.set_title(title)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_xlim(axis_limits)
    ax.set_ylim(axis_limits)
    ax.set_zlim(axis_limits)
    ax.view_init(DEFAULT_VIEW_ELEVATION, DEFAULT_VIEW_AZIMUTH)

    if values is not None:
        return ax, cbar
    return ax


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
