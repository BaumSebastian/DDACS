"""DDACS utilities for extracting simulation data from HDF5 files.

The DDACS dataset contains deep drawing simulations with:
- Geometries: rectangular, concave, convex
- Components: blank (workpiece), die, binder, punch (forming tools)
- Operations: OP10 (forming + springback), OP20 (cutting + cutting-induced springback)

Per-simulation parameters are stored as named scalar attributes on each HDF5
root and indexed in ``process_parameters.csv``. See ``ddacs.metadata`` and the
Croissant ``metadata.json`` for the full field-map and column descriptions.
"""

from pathlib import Path

import h5py
import numpy as np


def extract_point_cloud(
    h5_path: str | Path, component: str, timestep: int = 0, operation: int = 10
) -> np.ndarray:
    """
    Extract point cloud coordinates from H5 simulation file.

    Args:
        h5_path: Path to the H5 simulation file.
        component: Component name ('binder', 'blank', 'die', 'punch').
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state. Tools may have different timestep
                 structures than the workpiece.
        operation: Operation index (default: 10). Use 10 for deep drawing
                 process and 20 for cutting process.

    Returns:
        np.ndarray: Node coordinates array with shape (n_nodes, 3).
                   Returns the actual node positions at the specified timestep.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component or operation is not found in the file.

    Examples:
        >>> coords = extract_point_cloud('simulation_001.h5', 'blank', timestep=0)
        >>> print(f"Initial blank shape: {coords.shape}")

        >>> final_coords = extract_point_cloud('simulation_001.h5', 'blank', timestep=-1)
        >>> print(f"Final deformed shape: {final_coords.shape}")

        >>> cutting_coords = extract_point_cloud('simulation_001.h5', 'blank', timestep=0, operation=20)
        >>> print(f"Cutting shape: {cutting_coords.shape}")

    Note:
        The function reads from node_displacement which contains actual
        node coordinates at each timestep. Timestep structure differs by component:
        - blank: 4 timesteps (0: initial, 1-2: forming, 3: springback after tool removal)
        - tools (die, punch, binder): 3 timesteps (0: initial, 1-2: forming positions)
        Tools are removed at the final blank timestep, so timestep=3 or timestep=-1
        only exists for the blank component. The cutting operation (operation=20)
        only contains the blank component.
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP{operation}/{component}"]
        coords = np.array(comp_group["node_displacement"], copy=True)[timestep]

    return coords


def extract_mesh(
    h5_path: str | Path, component: str, timestep: int = 0, operation: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh data for matplotlib visualization.

    Args:
        h5_path: Path to the H5 simulation file.
        component: Component name ('binder', 'blank', 'die', 'punch').
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state.
        operation: Operation index (default: 10). Use 10 for deep drawing
                 process and 20 for cutting process.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - vertices: Node coordinates array with shape (n_nodes, 3)
            - triangles: Triangle connectivity array with shape (n_faces, 3)
                        Each row contains indices of triangle vertices

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component or operation is not found in the file.

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
        comp_group = f[f"OP{operation}/{component}"]

        vertices = extract_point_cloud(h5_path, component, timestep, operation=operation)

        element_node_ids = np.array(comp_group["element_shell_node_indexes"])
        element_node_ids -= element_node_ids.min()

        triangles = np.concatenate([element_node_ids[:, [0, 1, 2]], element_node_ids[:, [0, 2, 3]]])

    return vertices, np.array(triangles, dtype=int)


def extract_element_thickness(
    h5_path: str | Path, timestep: int = 0, operation: int = 10
) -> np.ndarray:
    """
    Extract element thickness data from H5 simulation file.

    Args:
        h5_path: Path to the H5 simulation file.
        timestep: Timestep index (default: 0). Use 0 for initial state,
                 -1 for final state.
        operation: Operation index (default: 10). Use 10 for deep drawing
                 process and 20 for cutting process.

    Returns:
        np.ndarray: Element thickness array with shape (n_elements,).
                   Each value represents the thickness of one element.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified component, operation or thickness data is not found.

    Examples:
        >>> thickness = extract_element_thickness('simulation_001.h5',  timestep=0)
        >>> print(f"Initial thickness range: {thickness.min():.3f} - {thickness.max():.3f}")

        >>> # Compare initial vs final thickness
        >>> t0 = extract_element_thickness('simulation_001.h5',  timestep=0)
        >>> t_final = extract_element_thickness('simulation_001.h5', timestep=-1)
        >>> thinning = (t0 - t_final) / t0 * 100
        >>> print(f"Maximum thinning: {thinning.max():.1f}%")

    Note:
        Thickness data shows how the material thickness changes during forming.
        This is particularly useful for analyzing thinning in deep drawing processes.
        Can be used with matplotlib's color mapping on mesh visualizations.
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP{operation}/blank"]
        thickness = np.array(comp_group["element_shell_thickness"], copy=True)[timestep]
    # Scale up according to fit the triangles conversion
    thickness = np.tile(thickness, 2)
    return thickness


def compute_von_mises(stress: np.ndarray) -> np.ndarray:
    """
    Compute Von Mises equivalent stress from stress tensor in Voigt notation.

    The Von Mises stress is a scalar value used to predict yielding of materials
    under complex loading conditions based on the distortion energy theory.

    Args:
        stress: Stress tensor array in Voigt notation with shape (..., 6).
               Components are ordered as [σxx, σyy, σzz, σxy, σyz, σxz].

    Returns:
        np.ndarray: Von Mises stress values with shape (...).

    Examples:
        >>> # Single stress state
        >>> stress = np.array([100, 50, 0, 25, 0, 0])  # MPa
        >>> vm = compute_von_mises(stress)
        >>> print(f"Von Mises stress: {vm:.1f} MPa")

        >>> # Batch of stress states
        >>> stresses = np.random.randn(100, 6) * 100
        >>> vm_batch = compute_von_mises(stresses)
        >>> print(f"Max Von Mises: {vm_batch.max():.1f} MPa")

    Note:
        Formula: σ_vm = √(0.5 * [(σxx-σyy)² + (σyy-σzz)² + (σzz-σxx)² + 6(σxy² + σyz² + σxz²)])
    """
    sxx, syy, szz = stress[..., 0], stress[..., 1], stress[..., 2]
    sxy, syz, sxz = stress[..., 3], stress[..., 4], stress[..., 5]

    von_mises = np.sqrt(
        0.5
        * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2 + 6 * (sxy**2 + syz**2 + sxz**2))
    )
    return von_mises


def extract_element_stress(
    h5_path: str | Path, timestep: int = 0, operation: int = 10, integration_point: int = 0
) -> np.ndarray:
    """
    Extract Von Mises stress for shell elements.

    Args:
        h5_path: Path to the H5 simulation file.
        timestep: Timestep index (default: 0). Use -1 for final state.
        operation: Operation index (default: 10). Use 10 for forming, 20 for cutting.
        integration_point: Through-thickness integration point (default: 0 for top).
                          Shell elements typically have multiple integration points
                          through the thickness (0=top, 1=middle, 2=bottom).

    Returns:
        np.ndarray: Von Mises stress array with shape (n_triangles,).
                   Values are duplicated to match triangle mesh conversion.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If stress data is not found in the file.

    Examples:
        >>> stress = extract_element_stress('simulation_001.h5', timestep=-1)
        >>> print(f"Max stress: {stress.max():.1f} MPa")

        >>> # Compare top vs bottom surface stress
        >>> stress_top = extract_element_stress('sim.h5', integration_point=0)
        >>> stress_bot = extract_element_stress('sim.h5', integration_point=2)
        >>> print(f"Stress difference: {(stress_top - stress_bot).mean():.1f} MPa")

    Note:
        Stress is computed from the full stress tensor using Von Mises criterion.
        The result is scaled to match the triangle mesh from extract_mesh().
    """
    with h5py.File(h5_path, "r") as f:
        stress_tensor = np.array(f[f"OP{operation}/blank/element_shell_stress"])[
            timestep, :, integration_point, :
        ]
        von_mises = compute_von_mises(stress_tensor)
    return np.tile(von_mises, 2)


def extract_element_strain(
    h5_path: str | Path, timestep: int = 0, operation: int = 10, integration_point: int = 0
) -> np.ndarray:
    """
    Extract effective plastic strain for shell elements.

    Args:
        h5_path: Path to the H5 simulation file.
        timestep: Timestep index (default: 0). Use -1 for final state.
        operation: Operation index (default: 10). Use 10 for forming, 20 for cutting.
        integration_point: Through-thickness integration point (default: 0 for top).

    Returns:
        np.ndarray: Effective plastic strain array with shape (n_triangles,).
                   Values are duplicated to match triangle mesh conversion.

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If strain data is not found in the file.

    Examples:
        >>> strain = extract_element_strain('simulation_001.h5', timestep=-1)
        >>> print(f"Max plastic strain: {strain.max():.3f}")

        >>> # Find elements with high plastic deformation
        >>> high_strain_mask = strain > 0.1
        >>> print(f"Elements with >10% strain: {high_strain_mask.sum()}")

    Note:
        Effective plastic strain is a scalar measure of accumulated plastic
        deformation. Values > 0 indicate permanent deformation has occurred.
    """
    with h5py.File(h5_path, "r") as f:
        strain = np.array(f[f"OP{operation}/blank/element_shell_effective_plastic_strain"])[
            timestep, :, integration_point
        ]
    return np.tile(strain, 2)


def non_degenerate_mask(triangles: np.ndarray) -> np.ndarray:
    """
    Return boolean mask identifying non-degenerate triangles.

    Args:
        triangles: Array of triangle vertex indices where each row represents one triangle with three vertex indices.

    Returns:
        np.ndarray: Boolean mask where True indicates a valid (non-degenerate) triangle
        and False indicates a degenerate triangle with duplicate vertices.

    Examples:
        >>> triangles = np.array([[0, 1, 2],      # Valid triangle
        ...                       [3, 3, 4],      # Degenerate (vertices 3,3)
        ...                       [5, 6, 7],      # Valid triangle
        ...                       [8, 9, 8]])     # Degenerate (vertices 8,8)
        >>> mask = non_degenerate_mask(triangles)
        >>> mask
        array([ True, False,  True, False])
        >>> valid_triangles = triangles[mask]
        >>> valid_triangles
        array([[0, 1, 2],
            [5, 6, 7]])
    Note:
        The function only checks for vertex uniqueness, not geometric validity (e.g., collinear vertices that form zero-area triangles).
    """
    return (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )


def extract_point_springback(
    h5_path: str | Path, operation: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract springback data for the blank component.

    Springback is calculated as the difference between the last timestep
    and the second-to-last timestep, representing the material's elastic
    recovery after tool removal.

    Args:
        h5_path: Path to the H5 simulation file.
        operation: Operation index (default: 10). Use 10 for deep drawing
                 process and 20 for cutting process.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - final_coords: Node coordinates after springback with shape (n_nodes, 3)
            - displacement_vectors: Springback displacement vectors with shape (n_nodes, 3)
                                  Each row contains [x, y, z] displacement due to springback

    Raises:
        FileNotFoundError: If the H5 file does not exist.
        KeyError: If the specified operation or blank component is not found.

    Examples:
        >>> final_coords, displacement = extract_point_springback('simulation_001.h5')
        >>> magnitude = np.linalg.norm(displacement, axis=1)
        >>>
        >>> # Visualize final shape colored by springback magnitude
        >>> plt.scatter(final_coords[:, 0], final_coords[:, 1], c=magnitude)
        >>> plt.colorbar(label='Springback magnitude [mm]')
        >>>
        >>> # Show springback vectors
        >>> plt.quiver(final_coords[:, 0], final_coords[:, 1],
        ...           displacement[:, 0], displacement[:, 1])

    Note:
        - For OP10 (forming): Compares timestep -1 (final springback) vs timestep -2 (max forming)
        - For OP20 (cutting): Compares timestep -1 (final) vs timestep 0 (initial cutting state)
        - Only works for the blank component as tools are removed before springback occurs
    """
    with h5py.File(h5_path, "r") as f:
        comp_group = f[f"OP{operation}/blank"]
        coords_data = np.array(comp_group["node_displacement"])

        coords_final = coords_data[-1]
        coords_before = coords_data[-2]

        # Calculate springback displacement vectors
        displacement_vectors = coords_final - coords_before

    return coords_final, displacement_vectors


def display_structure(h5_path: str | Path, max_depth: int = None) -> None:
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
            for key, item in items:
                child_prefix = prefix + ("  " if depth == 0 else "  ")
                _print_structure(key, item, depth + 1, child_prefix)
        elif isinstance(obj, h5py.Dataset):
            shape_str = f"shape={obj.shape}" if obj.shape else "scalar"
            dtype_str = f"dtype={obj.dtype}"
            print(f"{prefix}{indent}{name} (Dataset: {shape_str}, {dtype_str}){attrs_info}")

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
