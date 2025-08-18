"""
DDACS utilities for extracting simulation data.

The DDACS dataset contains deep drawing simulations with:
- Material parameters: FC (friction), MAT (material), STHK (sheet thickness), BF (blank holder force)
- Geometry types: R (rectangular), V (concave), X (convex) with radius parameter
- Components: blank (workpiece), die, binder, punch (forming tools)
- Operations: OP10 (forming), OP20 (springback analysis)
"""

import numpy as np
import h5py
from pathlib import Path
from typing import Union, Tuple


def extract_point_cloud(
    h5_path: Union[str, Path], component: str, timestep: int = 0
) -> np.ndarray:
    """
    Extract point cloud coordinates from H5 simulation file.

    Args:
        h5_path: Path to the H5 simulation file.
        component: Component name ('binder', 'blank', 'die', 'punch').
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state. Tools may have different timestep
                 structures than the workpiece.

    Returns:
        np.ndarray: Node coordinates array with shape (n_nodes, 3).
                   Returns the actual node positions at the specified timestep.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component is not found in the file.

    Examples:
        >>> coords = extract_point_cloud('simulation_001.h5', 'blank', timestep=0)
        >>> print(f"Initial blank shape: {coords.shape}")

        >>> final_coords = extract_point_cloud('simulation_001.h5', 'blank', timestep=-1)
        >>> print(f"Final deformed shape: {final_coords.shape}")

    Note:
        The function reads from node_displacement which contains actual
        node coordinates at each timestep. Timestep structure differs by component:
        - blank: 4 timesteps (0: initial, 1-2: forming, 3: springback after tool removal)
        - tools (die, punch, binder): 3 timesteps (0: initial, 1-2: forming positions)
        Tools are removed at the final blank timestep, so timestep=3 or timestep=-1
        only exists for the blank component.
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP10/{component}"]
        coords = np.array(comp_group["node_displacement"], copy=True)[timestep]

    return coords


def extract_mesh(
    h5_path: Union[str, Path], component: str, timestep: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh data for matplotlib visualization.

    Args:
        h5_path: Path to the H5 simulation file.
        component: Component name ('binder', 'blank', 'die', 'punch').
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices: Node coordinates array with shape (n_nodes, 3)
            - triangles: Triangle connectivity array with shape (n_faces, 3)
                        Each row contains indices of triangle vertices

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component is not found in the file.

    Examples:
        >>> vertices, triangles = extract_mesh('simulation_001.h5', 'blank')
        >>> print(f"Mesh has {len(vertices)} vertices and {len(triangles)} triangles")

        >>> # Extract final deformed mesh
        >>> vertices, triangles = extract_mesh('simulation_001.h5', 'blank', timestep=-1)

    Note:
        Quad elements are automatically converted to triangles by splitting
        each quad into two triangles.
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP10/{component}"]

        # Get vertices
        vertices = extract_point_cloud(h5_path, component, timestep)

        # Get element connectivity
        element_node_ids = np.array(comp_group["element_shell_node_ids"])
        node_ids = np.array(comp_group["node_ids"])

        # Map node IDs to indices
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Convert quads to triangles
        triangles = []
        for elem_nodes in element_node_ids:
            indices = [id_to_index[nid] for nid in elem_nodes if nid in id_to_index]
            if len(indices) == 4:
                triangles.extend(
                    [
                        [indices[0], indices[1], indices[2]],
                        [indices[0], indices[2], indices[3]],
                    ]
                )

    return vertices, np.array(triangles)


def extract_element_thickness(
    h5_path: Union[str, Path], component: str, timestep: int = 0
) -> np.ndarray:
    """
    Extract element thickness data from H5 simulation file.

    Args:
        h5_path: Path to the H5 simulation file.
        component: Component name ('binder', 'blank', 'die', 'punch').
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state.

    Returns:
        np.ndarray: Element thickness array with shape (n_elements,).
                   Each value represents the thickness of one element.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component or thickness data is not found.

    Examples:
        >>> thickness = extract_element_thickness('simulation_001.h5', 'blank', timestep=0)
        >>> print(f"Initial thickness range: {thickness.min():.3f} - {thickness.max():.3f}")

        >>> # Compare initial vs final thickness
        >>> t0 = extract_element_thickness('simulation_001.h5', 'blank', timestep=0)
        >>> t_final = extract_element_thickness('simulation_001.h5', 'blank', timestep=-1)
        >>> thinning = (t0 - t_final) / t0 * 100
        >>> print(f"Maximum thinning: {thinning.max():.1f}%")

    Note:
        Thickness data shows how the material thickness changes during forming.
        This is particularly useful for analyzing thinning in deep drawing processes.
        Can be used with matplotlib's color mapping on mesh visualizations.
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP10/{component}"]
        thickness_data = comp_group["element_shell_thickness"]

        # Handle negative timestep indexing
        if timestep == -1:
            timestep = thickness_data.shape[0] - 1

        thickness = np.array(thickness_data[timestep], copy=True)

    return thickness


def display_structure(h5_path: Union[str, Path], max_depth: int = None) -> None:
    """
    Display the complete hierarchical structure of an HDF5 file in tree format.

    Args:
        h5_path: Path to the H5 file to analyze.
        max_depth: Maximum depth to display (default: None for unlimited depth).
                  Useful for limiting output when files have deep nesting.

    Returns:
        None: Prints the structure directly to stdout.

    Raises:
        FileNotFoundError: If the H5 file does not exist.

    Examples:
        >>> display_structure('simulation_001.h5')
        HDF5 Structure: simulation_001.h5
        ============================================================
        OP10/ (Group)
          blank/ (Group)
            node_coordinates (Dataset: shape=(1024, 3), dtype=float64)
            ...

        >>> # Limit depth for large files
        >>> display_structure('simulation_001.h5', max_depth=2)

    Note:
        Displays groups, datasets, and their attributes in a hierarchical
        tree format. Shows dataset shapes, data types, and any HDF5 attributes.
    """

    def _print_structure(name, obj, depth=0, prefix=""):
        if max_depth is not None and depth > max_depth:
            return

        indent = "  " * depth

        # Print attributes if they exist
        if hasattr(obj, "attrs") and len(obj.attrs) > 0:
            attrs_str = ", ".join([f"{k}={v}" for k, v in obj.attrs.items()])
            attrs_info = f" [attrs: {attrs_str}]"
        else:
            attrs_info = ""

        if isinstance(obj, h5py.Group):
            print(f"{prefix}{indent}{name}/ (Group){attrs_info}")
            items = list(obj.items())
            for i, (key, item) in enumerate(items):
                is_last = i == len(items) - 1
                child_prefix = prefix + ("  " if depth == 0 else "  ")
                _print_structure(key, item, depth + 1, child_prefix)
        elif isinstance(obj, h5py.Dataset):
            shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
            dtype_str = f"dtype={obj.dtype}"
            print(
                f"{prefix}{indent}{name} (Dataset: {shape_str}, {dtype_str}){attrs_info}"
            )

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    print(f"\nHDF5 Structure: {h5_path.name}")
    print("=" * 60)

    with h5py.File(h5_path, "r") as f:
        # Print root level attributes if they exist
        if len(f.attrs) > 0:
            attrs_str = ", ".join([f"{k}={v}" for k, v in f.attrs.items()])
            print(f"Root attributes: {attrs_str}")
            print("-" * 40)

        if len(f.keys()) == 0:
            print("Empty file")
            return

        for key in f.keys():
            _print_structure(key, f[key])
