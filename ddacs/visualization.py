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
    "die": "Die (Lower Tool)",
    "punch": "Punch (Upper Tool)",
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
    color: str = "red",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    axis_limits: list[float] | None = None,
    show_edges: bool = True,
    edge_color: str = "black",
    edge_linewidth: float = 0.1,
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
        color: Solid color when values is None.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling. If None, uses min(values).
        vmax: Maximum value for color scaling. If None, uses max(values).
        colorbar_label: Label for the colorbar (e.g., "Thickness [mm]").
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].
        show_edges: Whether to show mesh edges.
        edge_color: Color of mesh edges.
        edge_linewidth: Width of mesh edges.

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
    """
    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS

    # Build face geometry
    face_vertices = vertices[faces]

    # Determine face colors
    if values is not None:
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        face_colors = plt.cm.get_cmap(cmap)(norm(values))

        collection = Poly3DCollection(
            face_vertices,
            facecolors=face_colors,
            edgecolors=edge_color if show_edges else face_colors,
            linewidth=edge_linewidth if show_edges else 0,
            alpha=1,
        )
        ax.add_collection3d(collection)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        collection = Poly3DCollection(
            face_vertices,
            facecolor=color,
            edgecolor=edge_color if show_edges else color,
            linewidth=edge_linewidth if show_edges else 0,
        )
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
    color: str = "red",
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    axis_limits: list[float] | None = None,
    point_size: float = 1.0,
    alpha: float = 0.8,
):
    """
    Plot 3D point cloud with optional per-point coloring.

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        values: Optional per-point values for coloring.
            If None, uses solid color.
        ax: Existing matplotlib 3D axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        color: Solid color when values is None.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar.
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].
        point_size: Size of scatter points.
        alpha: Transparency of points.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Examples:
        Simple point cloud:
            >>> coords = extract_point_cloud("sim.h5", "blank", timestep=2)
            >>> ax = plot_point_cloud(coords, color="blue")

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
            s=point_size,
            alpha=alpha,
        )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=color,
            s=point_size,
            alpha=alpha,
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
    arrow_color: str = "red",
    arrow_alpha: float = 0.8,
    show_points: bool = True,
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    point_color: str = "blue",
    point_size: float = 1.0,
    point_alpha: float = 0.5,
    title: str | None = None,
    axis_limits: list[float] | None = None,
):
    """
    Plot 3D vector field (quiver plot) for displacement visualization.

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        vectors: Vector components array of shape (n_points, 3).
        values: Optional per-point values for coloring the point cloud.
            If None, uses solid point_color.
        ax: Existing matplotlib 3D axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        step: Subsampling step (every Nth point) for arrow density.
        scale: Scaling factor for arrow length.
        arrow_color: Arrow color.
        arrow_alpha: Arrow transparency.
        show_points: Whether to show point cloud beneath vectors.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar (if values is provided).
        point_color: Color of background points (when values is None).
        point_size: Size of background points.
        point_alpha: Transparency of background points.
        title: Plot title.
        axis_limits: Axis limits as [min, max]. Defaults to [0, 110].

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
            ...     colorbar_label="Springback [mm]"
            ... )
    """
    if ax is None:
        fig = plt.figure(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)
        ax = fig.add_subplot(111, projection="3d")

    axis_limits = axis_limits or DEFAULT_AXIS_LIMITS
    cbar = None

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
                s=point_size,
                alpha=point_alpha,
            )

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
            cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
            if colorbar_label:
                cbar.set_label(colorbar_label)
        else:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=point_color,
                s=point_size,
                alpha=point_alpha,
            )

    # Subsample for arrow density
    coords_sub = coords[::step]
    vectors_sub = vectors[::step] * scale

    ax.quiver(
        coords_sub[:, 0],
        coords_sub[:, 1],
        coords_sub[:, 2],
        vectors_sub[:, 0],
        vectors_sub[:, 1],
        vectors_sub[:, 2],
        color=arrow_color,
        alpha=arrow_alpha,
        arrow_length_ratio=0.1,
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
    color: str = "red",
    cmap: str = "plasma",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str | None = None,
    title: str | None = None,
    point_size: float = 2.0,
    alpha: float = 0.8,
):
    """
    Plot 2D projection of point cloud (top view, side view, etc.).

    Args:
        coords: Point coordinates array of shape (n_points, 3).
        values: Optional per-point values for coloring.
        projection: Which 2D plane to project onto ("xy", "xz", "yz").
        ax: Existing matplotlib axis. If None, creates new figure.
        figsize: Figure size as (width, height) in inches.
        color: Solid color when values is None.
        cmap: Colormap name when values is provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        colorbar_label: Label for the colorbar.
        title: Plot title.
        point_size: Size of scatter points.
        alpha: Transparency of points.

    Returns:
        If values is None: matplotlib axis object.
        If values is provided: Tuple of (axis, colorbar).

    Example:
        >>> coords, displacement = extract_point_springback("sim.h5")
        >>> magnitude = np.linalg.norm(displacement, axis=1)
        >>> ax, cbar = plot_2d_projection(
        ...     coords, values=magnitude,
        ...     projection="xy",
        ...     colorbar_label="Springback [mm]"
        ... )
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize or (10, 6), dpi=DEFAULT_DPI)

    # Select projection axes
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    label_map = {"xy": ("X [mm]", "Y [mm]"), "xz": ("X [mm]", "Z [mm]"), "yz": ("Y [mm]", "Z [mm]")}

    idx1, idx2 = axis_map.get(projection, (0, 1))
    xlabel, ylabel = label_map.get(projection, ("X [mm]", "Y [mm]"))

    if values is not None:
        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()

        norm = Normalize(vmin=vmin, vmax=vmax)
        point_colors = plt.cm.get_cmap(cmap)(norm(values))

        ax.scatter(coords[:, idx1], coords[:, idx2], c=point_colors, s=point_size, alpha=alpha)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        if colorbar_label:
            cbar.set_label(colorbar_label)

    else:
        ax.scatter(coords[:, idx1], coords[:, idx2], c=color, s=point_size, alpha=alpha)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")

    if values is not None:
        return ax, cbar
    return ax
