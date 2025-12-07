"""
Cytokinesis (cell division) operations for vertex model.

Implements cell division process where a cell contracts along a specified
division axis (simulating an actomyosin ring) via two "contracting vertices"
inserted at the division line, then splits topologically when sufficiently
constricted.
"""

import numpy as np
from typing import Optional, Dict, Tuple, Union
from dataclasses import dataclass

from .core import Tissue, Cell
from .geometry import GeometryCalculator


@dataclass
class CytokinesisParams:
    """Parameters for cytokinesis process.
    
    Attributes:
        constriction_threshold: Distance between contracting vertices below which
            the cell is considered sufficiently constricted to divide (default: 0.1).
        initial_separation_fraction: Initial separation of contracting vertices as
            fraction of division axis length (default: 0.95, meaning they start
            at 95% of the full width).
        contractile_force_magnitude: Magnitude of contractile force applied to
            contracting vertices to simulate actomyosin ring (default: 10.0).
    """
    constriction_threshold: float = 0.1
    initial_separation_fraction: float = 0.95
    contractile_force_magnitude: float = 10.0


def compute_division_axis(
    cell: Cell,
    tissue: Tissue,
    axis_angle: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the division axis for a cell.
    
    Args:
        cell: Cell to divide.
        tissue: Tissue containing the cell.
        axis_angle: Optional angle in radians for division axis. If None,
            uses the long axis of the cell (principal component).
            
    Returns:
        Tuple of (centroid, axis_direction, perpendicular_direction):
            - centroid: (x, y) center of the cell
            - axis_direction: Unit vector along division axis
            - perpendicular_direction: Unit vector perpendicular to axis
    """
    geom = GeometryCalculator()
    
    # Get cell vertices
    if cell.vertex_indices.shape[0] > 0 and tissue.vertices.shape[0] > 0:
        vertices = tissue.vertices[cell.vertex_indices]
    else:
        vertices = cell.vertices
    
    if vertices.shape[0] < 3:
        raise ValueError(f"Cell {cell.id} has fewer than 3 vertices, cannot divide")
    
    # Compute centroid
    centroid = np.array(geom.calculate_centroid(vertices))
    
    # If axis angle is specified, use it
    if axis_angle is not None:
        axis_direction = np.array([np.cos(axis_angle), np.sin(axis_angle)])
        perpendicular_direction = np.array([-np.sin(axis_angle), np.cos(axis_angle)])
        return centroid, axis_direction, perpendicular_direction
    
    # Otherwise, compute principal axis via PCA
    # Center the vertices
    centered = vertices - centroid
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Get eigenvectors (principal components)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Long axis is the first eigenvector
    axis_direction = eigenvectors[:, 0]
    perpendicular_direction = eigenvectors[:, 1]
    
    return centroid, axis_direction, perpendicular_direction


def _find_ray_edge_intersection(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Find intersection between a ray and an edge.
    
    Args:
        ray_origin: Starting point of ray (2D)
        ray_dir: Direction of ray (2D)
        edge_start: Start point of edge (2D)
        edge_end: End point of edge (2D)
        
    Returns:
        Tuple of (t, s) if intersection found, where:
        - t: Parameter along ray (intersection at ray_origin + t * ray_dir)
        - s: Parameter along edge (intersection at edge_start + s * (edge_end - edge_start))
        Returns None if no valid intersection exists.
    """
    edge_dir = edge_end - edge_start
    
    # Set up linear system: [ray_dir, -edge_dir] @ [t, s]^T = edge_start - ray_origin
    A = np.column_stack([ray_dir, -edge_dir])
    b = edge_start - ray_origin
    
    try:
        # Solve for t and s
        ts = np.linalg.solve(A, b)
        t, s = ts[0], ts[1]
        
        # Check if intersection is valid (on the edge and ray goes forward)
        if 0 <= s <= 1 and t > 0:
            return (t, s)
        return None
    except np.linalg.LinAlgError:
        # Lines are parallel
        return None


def insert_contracting_vertices(
    cell: Cell,
    tissue: Tissue,
    axis_angle: Optional[float] = None,
    params: Optional[CytokinesisParams] = None
) -> Tuple[int, int]:
    """Insert two contracting vertices at the division line.
    
    This function inserts two new vertices on opposite sides of the cell's
    division axis. These vertices will be the attachment points for the
    contractile ring.
    
    IMPORTANT: This function also updates neighboring cells that share the
    edges where the new vertices are inserted, maintaining tissue connectivity.

    Args:
        cell: Cell to prepare for division.
        tissue: Tissue containing the cell.
        axis_angle: Optional angle in radians for division axis.
        params: Cytokinesis parameters (uses defaults if None).
        
    Returns:
        Tuple of (vertex_idx_1, vertex_idx_2): Global indices of the two
            contracting vertices in tissue.vertices.
            
    Raises:
        ValueError: If cell has insufficient vertices or tissue not initialized.
    """
    if params is None:
        params = CytokinesisParams()
    
    # Ensure tissue has global vertices
    if tissue.vertices.shape[0] == 0:
        tissue.build_global_vertices()
    
    # Compute division axis
    centroid, axis_direction, perpendicular = compute_division_axis(
        cell, tissue, axis_angle
    )
    
    # Get cell vertices
    if cell.vertex_indices.shape[0] > 0:
        vertices = tissue.vertices[cell.vertex_indices]
    else:
        vertices = cell.vertices
    
    # Find intersections of division line with cell boundary
    # The division line is perpendicular to the axis_direction
    division_line_direction = perpendicular
    
    # Find the two points where the division line intersects the cell
    # We'll shoot rays in both directions from the centroid
    intersections = []
    
    for sign in [1, -1]:
        # Ray from centroid in direction sign * division_line_direction
        ray_origin = centroid
        ray_dir = sign * division_line_direction
        
        # Find intersection with cell boundary
        best_t = None
        best_edge_idx = None
        
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            
            # Find ray-edge intersection
            result = _find_ray_edge_intersection(ray_origin, ray_dir, v1, v2)
            
            if result is not None:
                t, s = result
                if best_t is None or t < best_t:
                    best_t = t
                    best_edge_idx = i
        
        if best_t is not None and best_edge_idx is not None:
            intersection_point = ray_origin + best_t * ray_dir
            intersections.append((intersection_point, best_edge_idx))
    
    if len(intersections) != 2:
        # Provide debug information
        debug_info = (
            f"Cell {cell.id}: Found {len(intersections)} intersections "
            f"(expected 2).\n"
            f"  Division axis: {axis_direction}\n"
            f"  Division line direction: {division_line_direction}\n"
            f"  Centroid: {centroid}\n"
            f"  Cell has {len(vertices)} vertices"
        )
        raise ValueError(debug_info)
    
    # Sort intersections by edge index to maintain consistent ordering
    intersections.sort(key=lambda x: x[1])
    
    # Calculate initial positions with some separation
    point1, edge_idx1 = intersections[0]
    point2, edge_idx2 = intersections[1]
    
    # Move points toward centroid by (1 - initial_separation_fraction)
    sep_frac = params.initial_separation_fraction
    point1_initial = centroid + sep_frac * (point1 - centroid)
    point2_initial = centroid + sep_frac * (point2 - centroid)
    
    # Get the edge endpoints (global indices) for finding neighbors
    edge1_v1_global = cell.vertex_indices[edge_idx1]
    edge1_v2_global = cell.vertex_indices[(edge_idx1 + 1) % len(cell.vertex_indices)]
    edge2_v1_global = cell.vertex_indices[edge_idx2]
    edge2_v2_global = cell.vertex_indices[(edge_idx2 + 1) % len(cell.vertex_indices)]

    # Add new vertices to global pool
    new_vertex_idx1 = tissue.vertices.shape[0]
    tissue.vertices = np.vstack([tissue.vertices, point1_initial])
    
    new_vertex_idx2 = tissue.vertices.shape[0]
    tissue.vertices = np.vstack([tissue.vertices, point2_initial])
    
    # Insert these vertices into the cell's vertex_indices at appropriate positions
    # We need to insert vertex1 after edge_idx1 and vertex2 after edge_idx2
    old_indices = cell.vertex_indices.copy()
    
    # Insert in reverse order of edge indices to avoid index shifting issues
    if edge_idx1 < edge_idx2:
        # Insert vertex2 first (higher index)
        new_indices = np.concatenate([
            old_indices[:edge_idx2 + 1],
            [new_vertex_idx2],
            old_indices[edge_idx2 + 1:]
        ])
        # Then insert vertex1 (lower index)
        new_indices = np.concatenate([
            new_indices[:edge_idx1 + 1],
            [new_vertex_idx1],
            new_indices[edge_idx1 + 1:]
        ])
    else:
        # Insert vertex1 first (higher index)
        new_indices = np.concatenate([
            old_indices[:edge_idx1 + 1],
            [new_vertex_idx1],
            old_indices[edge_idx1 + 1:]
        ])
        # Then insert vertex2 (lower index, but index has shifted by 1)
        new_indices = np.concatenate([
            new_indices[:edge_idx2 + 1],
            [new_vertex_idx2],
            new_indices[edge_idx2 + 1:]
        ])
    
    cell.vertex_indices = new_indices
    
    # Reconstruct cell vertices
    cell.vertices = tissue.vertices[cell.vertex_indices]
    
    # Update neighboring cells that share the edges where we inserted vertices
    # For each edge (v1, v2), find neighbor cells that have both v1 and v2 in their vertex_indices
    # and insert the new vertex between them
    for other_cell in tissue.cells:
        if other_cell.id == cell.id:
            continue

        other_indices = list(other_cell.vertex_indices)
        modified = False

        # Check if this cell shares edge 1 (edge1_v1_global -> edge1_v2_global)
        # In the neighbor, the edge will be in reverse order: edge1_v2_global -> edge1_v1_global
        if edge1_v1_global in other_indices and edge1_v2_global in other_indices:
            idx_v1 = other_indices.index(edge1_v1_global)
            idx_v2 = other_indices.index(edge1_v2_global)
            n = len(other_indices)

            # Check if they are adjacent (the edge exists in this cell)
            # In neighbor cell, edge is reversed: v2 -> v1 (or cyclically adjacent)
            if (idx_v2 + 1) % n == idx_v1:
                # Edge exists as v2 -> v1, insert new vertex after v2
                insert_pos = idx_v2 + 1
                other_indices.insert(insert_pos, new_vertex_idx1)
                modified = True
            elif (idx_v1 + 1) % n == idx_v2:
                # Edge exists as v1 -> v2 (same direction as dividing cell - shouldn't happen for shared edge)
                # But handle it anyway - insert after v1
                insert_pos = idx_v1 + 1
                other_indices.insert(insert_pos, new_vertex_idx1)
                modified = True

        # Check if this cell shares edge 2 (edge2_v1_global -> edge2_v2_global)
        if edge2_v1_global in other_indices and edge2_v2_global in other_indices:
            idx_v1 = other_indices.index(edge2_v1_global)
            idx_v2 = other_indices.index(edge2_v2_global)
            n = len(other_indices)

            # Check if they are adjacent
            if (idx_v2 + 1) % n == idx_v1:
                # Edge exists as v2 -> v1, insert new vertex after v2
                insert_pos = idx_v2 + 1
                other_indices.insert(insert_pos, new_vertex_idx2)
                modified = True
            elif (idx_v1 + 1) % n == idx_v2:
                # Edge exists as v1 -> v2
                insert_pos = idx_v1 + 1
                other_indices.insert(insert_pos, new_vertex_idx2)
                modified = True

        if modified:
            other_cell.vertex_indices = np.array(other_indices, dtype=int)
            other_cell.vertices = tissue.vertices[other_cell.vertex_indices]

    # Store metadata about contracting vertices in cell
    if not hasattr(cell, 'cytokinesis_data'):
        cell.cytokinesis_data = {}
    
    cell.cytokinesis_data['contracting_vertices'] = (new_vertex_idx1, new_vertex_idx2)
    cell.cytokinesis_data['division_axis'] = axis_direction
    cell.cytokinesis_data['centroid'] = centroid
    
    return new_vertex_idx1, new_vertex_idx2


def compute_contractile_forces(
    cell: Cell,
    tissue: Tissue,
    params: Optional[CytokinesisParams] = None
) -> np.ndarray:
    """Compute contractile forces for a dividing cell.
    
    This function computes active forces that pull the two contracting vertices
    together, simulating the actomyosin contractile ring.
    
    Args:
        cell: Cell undergoing division (must have contracting vertices).
        tissue: Tissue containing the cell.
        params: Cytokinesis parameters.
        
    Returns:
        Array of shape (N, 2) where N is the number of vertices in the cell.
        Forces are zero for all vertices except the contracting vertices.
        
    Raises:
        ValueError: If cell doesn't have contracting vertices metadata.
    """
    if params is None:
        params = CytokinesisParams()
    
    if not hasattr(cell, 'cytokinesis_data') or 'contracting_vertices' not in cell.cytokinesis_data:
        raise ValueError(f"Cell {cell.id} does not have contracting vertices")
    
    # Get contracting vertex indices
    v1_global, v2_global = cell.cytokinesis_data['contracting_vertices']
    
    # Find positions of these vertices in the cell's vertex_indices
    v1_local = np.where(cell.vertex_indices == v1_global)[0]
    v2_local = np.where(cell.vertex_indices == v2_global)[0]
    
    if len(v1_local) == 0 or len(v2_local) == 0:
        raise ValueError(
            f"Contracting vertices not found in cell {cell.id} vertex_indices"
        )
    
    v1_local = v1_local[0]
    v2_local = v2_local[0]
    
    # Get vertex positions
    pos1 = tissue.vertices[v1_global]
    pos2 = tissue.vertices[v2_global]
    
    # Compute force direction (from each vertex toward the other)
    direction = pos2 - pos1
    distance = np.linalg.norm(direction)
    
    if distance > 1e-10:
        direction = direction / distance
    else:
        # Vertices are very close, no force needed
        direction = np.zeros(2)
    
    # Create force array (zeros for all vertices)
    forces = np.zeros_like(cell.vertices)
    
    # Apply contractile force
    magnitude = params.contractile_force_magnitude
    forces[v1_local] = magnitude * direction  # Pull v1 toward v2
    forces[v2_local] = -magnitude * direction  # Pull v2 toward v1
    
    return forces


def check_constriction(
    cell: Cell,
    tissue: Tissue,
    params: Optional[CytokinesisParams] = None
) -> bool:
    """Check if a dividing cell is sufficiently constricted.
    
    Args:
        cell: Cell undergoing division.
        tissue: Tissue containing the cell.
        params: Cytokinesis parameters.
        
    Returns:
        True if the distance between contracting vertices is below the
        constriction threshold.
    """
    if params is None:
        params = CytokinesisParams()
    
    if not hasattr(cell, 'cytokinesis_data') or 'contracting_vertices' not in cell.cytokinesis_data:
        return False
    
    v1_global, v2_global = cell.cytokinesis_data['contracting_vertices']
    
    pos1 = tissue.vertices[v1_global]
    pos2 = tissue.vertices[v2_global]
    
    distance = np.linalg.norm(pos2 - pos1)
    
    return distance <= params.constriction_threshold


def split_cell(
    cell: Cell,
    tissue: Tissue,
    daughter1_id: Optional[Union[int, str]] = None,
    daughter2_id: Optional[Union[int, str]] = None
) -> Tuple[Cell, Cell]:
    """Split a constricted cell into two daughter cells.
    
    This performs the topological split, creating two new cells from the
    dividing cell. The division plane is defined by the two contracting
    vertices.
    
    Args:
        cell: Cell to split (must have contracting vertices).
        tissue: Tissue containing the cell.
        daughter1_id: ID for first daughter cell (auto-generated if None).
        daughter2_id: ID for second daughter cell (auto-generated if None).
        
    Returns:
        Tuple of (daughter1, daughter2): The two new daughter cells.
        
    Raises:
        ValueError: If cell doesn't have contracting vertices.
        
    Notes:
        - The original cell is removed from the tissue
        - The two daughter cells are added to the tissue
        - Each daughter cell inherits vertices from one side of the division
        - The contracting vertices are duplicated (one for each daughter)
    """
    if not hasattr(cell, 'cytokinesis_data') or 'contracting_vertices' not in cell.cytokinesis_data:
        raise ValueError(f"Cell {cell.id} does not have contracting vertices")
    
    v1_global, v2_global = cell.cytokinesis_data['contracting_vertices']
    
    # Find positions in vertex_indices
    v1_local = np.where(cell.vertex_indices == v1_global)[0][0]
    v2_local = np.where(cell.vertex_indices == v2_global)[0][0]
    
    # Ensure v1_local < v2_local for consistent ordering
    if v1_local > v2_local:
        v1_local, v2_local = v2_local, v1_local
    
    # Split the vertex indices into two groups
    # Daughter 1: vertices from v1 to v2 (inclusive)
    # Daughter 2: vertices from v2 to v1 (wrapping around, inclusive)
    
    indices = cell.vertex_indices
    
    # Daughter 1 gets vertices from v1_local to v2_local (inclusive)
    daughter1_indices = indices[v1_local:v2_local + 1]
    
    # Daughter 2 gets vertices from v2_local to v1_local (wrapping, inclusive)
    # This includes vertices from v2_local to the end, then from start to v1_local
    daughter2_indices = np.concatenate([
        indices[v2_local:],
        indices[:v1_local + 1]
    ])
    
    # Create daughter cells
    if daughter1_id is None:
        daughter1_id = f"{cell.id}_d1"
    if daughter2_id is None:
        daughter2_id = f"{cell.id}_d2"
    
    daughter1 = Cell(
        cell_id=daughter1_id,
        vertex_indices=daughter1_indices
    )
    daughter1.vertices = tissue.vertices[daughter1_indices]
    
    daughter2 = Cell(
        cell_id=daughter2_id,
        vertex_indices=daughter2_indices
    )
    daughter2.vertices = tissue.vertices[daughter2_indices]
    
    # Validate that both daughters have at least 3 vertices
    if len(daughter1.vertex_indices) < 3:
        raise ValueError(
            f"Daughter 1 would have only {len(daughter1.vertex_indices)} vertices "
            f"(minimum 3 required)"
        )
    if len(daughter2.vertex_indices) < 3:
        raise ValueError(
            f"Daughter 2 would have only {len(daughter2.vertex_indices)} vertices "
            f"(minimum 3 required)"
        )
    
    # Remove original cell from tissue
    tissue.cells = [c for c in tissue.cells if c.id != cell.id]
    
    # Add daughter cells
    tissue.add_cell(daughter1)
    tissue.add_cell(daughter2)
    
    return daughter1, daughter2


def update_global_vertices_from_cells(tissue: Tissue) -> None:
    """Update global vertex pool from cell vertices without rebuilding indices.
    
    This preserves the global vertex indices while updating their positions
    based on the current cell.vertices arrays. Useful during simulation to
    maintain consistent vertex tracking.
    
    Args:
        tissue: Tissue with global vertex pool and cells with vertex_indices.
        
    Notes:
        - Requires tissue.vertices and cell.vertex_indices to be already populated
        - Updates tissue.vertices in place based on cell.vertices
        - Does NOT change vertex_indices (preserves connectivity)
    
    Raises:
        ValueError: If vertex indices are out of bounds or array sizes don't match.
    """
    if tissue.vertices.shape[0] == 0:
        return
    
    # For each cell, update the global vertices based on cell's local vertices
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] > 0 and cell.vertices.shape[0] > 0:
            # Validate that indices and vertices match
            if cell.vertex_indices.shape[0] != cell.vertices.shape[0]:
                raise ValueError(
                    f"Cell {cell.id}: vertex_indices size ({cell.vertex_indices.shape[0]}) "
                    f"doesn't match vertices size ({cell.vertices.shape[0]})"
                )
            
            # Validate that all indices are within bounds
            if np.any(cell.vertex_indices >= tissue.vertices.shape[0]):
                raise ValueError(
                    f"Cell {cell.id}: vertex_indices contains out-of-bounds indices "
                    f"(max index: {np.max(cell.vertex_indices)}, "
                    f"tissue has {tissue.vertices.shape[0]} vertices)"
                )
            
            # Update global vertices from this cell's local vertices
            tissue.vertices[cell.vertex_indices] = cell.vertices


def perform_cytokinesis(
    cell: Cell,
    tissue: Tissue,
    axis_angle: Optional[float] = None,
    params: Optional[CytokinesisParams] = None,
    daughter1_id: Optional[Union[int, str]] = None,
    daughter2_id: Optional[Union[int, str]] = None
) -> Dict:
    """Perform complete cytokinesis on a cell.
    
    This is a high-level function that:
    1. Inserts contracting vertices
    2. Returns information for simulation with contractile forces
    3. Can be called again to check constriction and split when ready
    
    Args:
        cell: Cell to divide.
        tissue: Tissue containing the cell.
        axis_angle: Optional division axis angle.
        params: Cytokinesis parameters.
        daughter1_id: ID for first daughter (if splitting).
        daughter2_id: ID for second daughter (if splitting).
        
    Returns:
        Dictionary with:
            - 'stage': Current stage ('initiated', 'constricting', or 'completed')
            - 'contracting_vertices': Tuple of vertex indices (if initiated/constricting)
            - 'daughter_cells': Tuple of daughter cells (if completed)
            - 'constriction_distance': Current distance between vertices (if constricting)
    """
    if params is None:
        params = CytokinesisParams()
    
    # Check if cell already has contracting vertices
    if hasattr(cell, 'cytokinesis_data') and 'contracting_vertices' in cell.cytokinesis_data:
        # Check if ready to split
        if check_constriction(cell, tissue, params):
            # Perform split
            daughter1, daughter2 = split_cell(
                cell, tissue, daughter1_id, daughter2_id
            )
            return {
                'stage': 'completed',
                'daughter_cells': (daughter1, daughter2)
            }
        else:
            # Still constricting
            v1, v2 = cell.cytokinesis_data['contracting_vertices']
            pos1 = tissue.vertices[v1]
            pos2 = tissue.vertices[v2]
            distance = np.linalg.norm(pos2 - pos1)
            
            return {
                'stage': 'constricting',
                'contracting_vertices': (v1, v2),
                'constriction_distance': distance
            }
    else:
        # Initiate cytokinesis
        v1, v2 = insert_contracting_vertices(cell, tissue, axis_angle, params)
        return {
            'stage': 'initiated',
            'contracting_vertices': (v1, v2)
        }
