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


def extract_point_cloud(h5_path: Union[str, Path], 
                       component: str,
                       timestep: int = 0) -> np.ndarray:
    """
    Extract point cloud coordinates from H5 simulation file.
    
    Args:
        h5_path: Path to H5 file
        component: Component name ('binder', 'blank', 'die', 'punch')
        timestep: Timestep index (0=initial, -1=final)
                 Note: Tools (die, binder, punch) may have different timestep 
                 structures than the blank (workpiece)
    
    Returns:
        Coordinates array (n_nodes, 3)
        
    Note:
        Different components may have different displacement data structures:
        - blank: Usually (timesteps, nodes, 3) - full deformation history
        - tools: May have (nodes, 3) - single/prescribed motion
    """
    with h5py.File(h5_path, 'r') as f:
        comp_group = f[f'OP10/{component}']
        coords = np.array(comp_group['node_coordinates'])
        
        if 'node_displacement' in comp_group:
            disp = np.array(comp_group['node_displacement'])
            if len(disp.shape) == 3:  # (timesteps, nodes, 3)
                if timestep == -1:
                    timestep = disp.shape[0] - 1
                disp = disp[timestep]
            coords += disp
            
    return coords


def extract_mesh(h5_path: Union[str, Path],
                component: str, 
                timestep: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract mesh data for matplotlib visualization.
    
    Args:
        h5_path: Path to H5 file
        component: Component name ('binder', 'blank', 'die', 'punch')  
        timestep: Timestep index (0=initial, -1=final)
                 Note: Tools may have different timestep structures than blank
    
    Returns:
        (vertices, triangles) - vertices (n_nodes, 3), triangles (n_faces, 3)
    """
    with h5py.File(h5_path, 'r') as f:
        comp_group = f[f'OP10/{component}']
        
        # Get vertices
        vertices = extract_point_cloud(h5_path, component, timestep)
        
        # Get element connectivity
        element_node_ids = np.array(comp_group['element_shell_node_ids'])
        node_ids = np.array(comp_group['node_ids'])
        
        # Map node IDs to indices
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Convert quads to triangles
        triangles = []
        for elem_nodes in element_node_ids:
            indices = [id_to_index[nid] for nid in elem_nodes if nid in id_to_index]
            if len(indices) == 4:
                triangles.extend([[indices[0], indices[1], indices[2]], 
                                 [indices[0], indices[2], indices[3]]])
        
    return vertices, np.array(triangles)


def display_structure(h5_path: Union[str, Path], max_depth: int = None) -> None:
    """
    Display the complete hierarchical structure of an HDF5 file in tree format.
    
    Args:
        h5_path: Path to H5 file
        max_depth: Maximum depth to display (None for unlimited)
    """
    def _print_structure(name, obj, depth=0, prefix=""):
        if max_depth is not None and depth > max_depth:
            return
            
        indent = "  " * depth
        
        # Print attributes if they exist
        if hasattr(obj, 'attrs') and len(obj.attrs) > 0:
            attrs_str = ", ".join([f"{k}={v}" for k, v in obj.attrs.items()])
            attrs_info = f" [attrs: {attrs_str}]"
        else:
            attrs_info = ""
        
        if isinstance(obj, h5py.Group):
            print(f"{prefix}{indent}{name}/ (Group){attrs_info}")
            items = list(obj.items())
            for i, (key, item) in enumerate(items):
                is_last = (i == len(items) - 1)
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
    
    with h5py.File(h5_path, 'r') as f:
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