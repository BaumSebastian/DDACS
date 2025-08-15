import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import h5py

def load_geometry_data(h5_path: Union[str, Path], 
                      timestep: int = 0,
                      components: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load coordinate and displacement data for geometry components.
    
    Args:
        h5_path: Path to H5 simulation file
        timestep: Timestep to load (0=initial, -1=final with springback)
        components: List of components to load. If None, loads all available.
    
    Returns:
        Dictionary with geometry data for each component:
        {
            'component_name': {
                'coordinates': ndarray,  # Deformed coordinates
                'elements': ndarray,     # Element connectivity 
                'n_nodes': int,         # Number of nodes
                'n_elements': int       # Number of elements
            }
        }
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    if components is None:
        components = ['binder', 'blank', 'die', 'punch']
    
    geometries = {}
    
    with h5py.File(h5_path, 'r') as f:
        for component in components:
            component_path = f'OP10/{component}'
            
            if component_path not in f:
                warnings.warn(f"Component '{component}' not found in {h5_path.name}")
                continue
                
            comp_group = f[component_path]
            
            # Get base coordinates
            if 'node_coordinates' not in comp_group:
                warnings.warn(f"No coordinates found for component '{component}'")
                continue
                
            coords = np.array(comp_group['node_coordinates'])
            
            # Get displacements if available
            if 'node_displacement' in comp_group:
                displacements = np.array(comp_group['node_displacement'])
                
                # Handle different timestep dimensions
                if len(displacements.shape) == 3:  # (timesteps, nodes, 3)
                    max_timestep = displacements.shape[0] - 1
                    if timestep == -1:
                        timestep = max_timestep
                    elif timestep > max_timestep:
                        warnings.warn(f"Component '{component}' only has {max_timestep + 1} timesteps, "
                                    f"requested timestep {timestep}. Using last available timestep {max_timestep}.")
                        timestep = max_timestep
                    disp = displacements[timestep]
                else:  # (nodes, 3)
                    if timestep > 0:
                        warnings.warn(f"Component '{component}' has no timestep data, using static geometry.")
                    disp = displacements
                    
                # Add displacement to get deformed coordinates
                deformed_coords = coords + disp
            else:
                deformed_coords = coords
            
            # Get element connectivity - need to map node IDs to indices
            elements = None
            if 'element_shell_node_ids' in comp_group and 'node_ids' in comp_group:
                # Use actual node IDs from the element connectivity
                element_node_ids = np.array(comp_group['element_shell_node_ids'])
                node_ids = np.array(comp_group['node_ids'])
                
                # Create mapping from node ID to index
                id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
                
                # Convert element node IDs to indices
                elements = []
                for elem_nodes in element_node_ids:
                    elem_indices = []
                    for node_id in elem_nodes:
                        if node_id in id_to_index:
                            elem_indices.append(id_to_index[node_id])
                    if len(elem_indices) == 4:  # Only keep complete quads
                        elements.append(elem_indices)
                
                elements = np.array(elements) if elements else None
                
            elif 'element_shell_node_indexes' in comp_group:
                # Fallback to node indexes if available
                elements = np.array(comp_group['element_shell_node_indexes'])
            
            geometries[component] = {
                'coordinates': deformed_coords,
                'elements': elements,
                'n_nodes': len(deformed_coords),
                'n_elements': len(elements) if elements is not None else 0
            }
    
    return geometries


def create_surface_plot(ax, coords: np.ndarray, elements: Optional[np.ndarray] = None,
                       color: str = 'blue', alpha: float = 0.7, 
                       max_elements: int = 2000, point_size: int = 10) -> None:
    """
    Create a 3D surface plot from coordinates and element connectivity.
    
    Args:
        ax: Matplotlib 3D axis
        coords: Node coordinates array (n_nodes, 3)
        elements: Element connectivity array (n_elements, 4) or None
        color: Surface color
        alpha: Transparency level
        max_elements: Maximum elements to plot (for performance)
        point_size: Point size for scatter plot fallback
    """
    if elements is not None and len(elements) > 0:
        # Create surface using element connectivity
        faces = []
        n_plot = min(len(elements), max_elements)
        
        for i, elem in enumerate(elements[:n_plot]):
            # Each element has 4 nodes (quad), split into 2 triangles
            if len(elem) >= 4 and all(idx < len(coords) for idx in elem):
                try:
                    # Triangle 1: nodes 0, 1, 2
                    faces.append([coords[elem[0]], coords[elem[1]], coords[elem[2]]])
                    # Triangle 2: nodes 0, 2, 3  
                    faces.append([coords[elem[0]], coords[elem[2]], coords[elem[3]]])
                except IndexError:
                    continue
        
        if faces:
            # Create 3D collection of polygons for solid surface
            poly_collection = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                                             edgecolor='darkgray', linewidth=0.05)
            ax.add_collection3d(poly_collection)
        else:
            # Fallback to scatter if faces creation failed
            _create_scatter_plot(ax, coords, color, alpha, point_size)
    else:
        # Fallback: scatter plot if no element connectivity
        _create_scatter_plot(ax, coords, color, alpha, point_size)


def _create_scatter_plot(ax, coords: np.ndarray, color: str, alpha: float, point_size: int):
    """Create scatter plot fallback."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    ax.scatter(x, y, z, c=color, alpha=alpha, s=point_size)


def plot_geometries(sim_id: Union[int, str, Path], 
                   timestep: int = 0,
                   components: Optional[List[str]] = None,
                   figsize: Tuple[int, int] = (12, 9),
                   colors: Optional[Dict[str, str]] = None,
                   save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot all geometry components for a simulation.
    
    Args:
        sim_id: Simulation ID, filename, or full path to H5 file
        timestep: Timestep to visualize (0=initial, -1=final)
        components: List of components to plot. Default: ['binder', 'blank', 'die', 'punch']
        figsize: Figure size tuple
        colors: Color mapping for components
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure object
    """
    
    # Handle different input types for sim_id
    if isinstance(sim_id, (int, str)) and not str(sim_id).endswith('.h5'):
        # Assume it's a simulation ID, construct path
        h5_path = Path(f"data/h5/{sim_id}.h5")
    else:
        h5_path = Path(sim_id)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"Simulation file not found: {h5_path}")
    
    # Set default components and colors
    if components is None:
        components = ['binder', 'blank', 'die', 'punch']
    
    if colors is None:
        colors = {
            'die': 'blue',      # Lower forming tool
            'blank': 'red',     # Sheet metal being formed
            'binder': 'yellow', # Holds the blank
            'punch': 'green'    # Upper forming tool
        }
    
    # Component descriptions
    descriptions = {
        'binder': 'Holds sheet metal in place',
        'blank': 'Sheet metal being formed', 
        'die': 'Lower forming tool',
        'punch': 'Upper forming tool'
    }
    
    # Load geometry data
    geometries = load_geometry_data(h5_path, timestep, components)
    
    if not geometries:
        raise ValueError(f"No geometry data found in {h5_path.name}")
    
    # Create single figure with all components
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all components together as solid surfaces
    for component_name, geom_data in geometries.items():
        coords = geom_data['coordinates']  
        elements = geom_data['elements']
        
        create_surface_plot(ax, coords, elements,
                          color=colors.get(component_name, 'gray'),
                          alpha=0.8, max_elements=5000)  # Increase element limit for better surfaces
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12) 
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(f'DDACS Simulation: {h5_path.stem}\nTimestep {timestep} - All Components', fontsize=14)
    
    # Create legend
    handles = []
    for component_name in geometries.keys():
        from matplotlib.patches import Patch
        handles.append(Patch(color=colors.get(component_name, 'gray'), 
                            label=component_name.title()))
    ax.legend(handles=handles, loc='upper left', fontsize=11)
    
    # Set viewing angle for best view
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect([1, 1, 0.6])
    
    # Remove grid for cleaner look
    ax.grid(False)
    
    # Set background color
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


def plot_component(sim_id: Union[int, str, Path], 
                  component: str,
                  timestep: int = 0,
                  figsize: Tuple[int, int] = (10, 8),
                  color: str = 'blue',
                  save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot a single geometry component.
    
    Args:
        sim_id: Simulation ID, filename, or full path to H5 file
        component: Component name ('binder', 'blank', 'die', 'punch')
        timestep: Timestep to visualize
        figsize: Figure size tuple
        color: Surface color
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure object
    """
    
    # Handle input path
    if isinstance(sim_id, (int, str)) and not str(sim_id).endswith('.h5'):
        h5_path = Path(f"data/h5/{sim_id}.h5")
    else:
        h5_path = Path(sim_id)
    
    # Load single component
    geometries = load_geometry_data(h5_path, timestep, [component])
    
    if component not in geometries:
        raise ValueError(f"Component '{component}' not found in {h5_path.name}")
    
    geom_data = geometries[component]
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot component
    coords = geom_data['coordinates']
    elements = geom_data['elements']
    
    create_surface_plot(ax, coords, elements, color=color, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{component.title()} - {h5_path.stem} (Timestep {timestep})\n'
                f'{geom_data["n_nodes"]} nodes, {geom_data["n_elements"]} elements')
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 0.5])
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig


# Convenience functions
def plot_blank(sim_id: Union[int, str, Path], timestep: int = 0, **kwargs) -> plt.Figure:
    """Plot only the blank (sheet metal) component."""
    return plot_component(sim_id, 'blank', timestep, color='orange', **kwargs)


def plot_setup(sim_id: Union[int, str, Path], timestep: int = 0, **kwargs) -> plt.Figure:
    """Plot complete forming setup (all components)."""
    return plot_geometries(sim_id, timestep, **kwargs)


def plot_springback(sim_id: Union[int, str, Path], **kwargs) -> plt.Figure:
    """Plot blank after springback (timestep 3, tools removed)."""
    return plot_component(sim_id, 'blank', timestep=3, color='orange', **kwargs)