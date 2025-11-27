"""ACAM tissue importer for vertex model.

Converts ACAM epithelium (center-based cell model with adhesion) into a vertex
model tissue by approximating cells as regular polygons and using high tolerance
vertex merging to create mechanical coupling.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Tuple, NamedTuple
from dataclasses import dataclass
from collections import Counter
import json
import pickle
from scipy.spatial import cKDTree, ConvexHull
from shapely.geometry import Polygon
from .core import Tissue, Cell

__all__ = ["load_acam_tissue", "load_acam_from_json", "convert_acam_with_topology"]


class _ACamStubUnpickler(pickle.Unpickler):
    """Custom unpickler that creates stub objects for missing ACAM classes."""

    def find_class(self, module, name):
        """Override to provide stub classes for missing ACAM modules."""
        # Map known ACAM classes to stubs
        if 'eptm' in module.lower() or 'acam' in module.lower() or 'cell' in module.lower() or 'adhesion' in module.lower():
            # Create dynamic stub class
            return type(name, (), {})

        # Try normal import for everything else
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # Create generic stub
            return type(name, (), {})


def load_acam_from_json(
    filepath: Union[str, Path],
    approx_radius: float = None,  # Deprecated, kept for compatibility
    vertex_count: int = None,     # Deprecated, kept for compatibility
    adhesion_distance: float = 1.0,
    skip_boundary: bool = True,
) -> Tissue:
    """Load ACAM tissue from extracted JSON cell data.

    This loads pre-extracted cell boundaries from JSON format (output of extract_acam_data.py).
    The new format includes actual vertex arrays extracted from ACAM finite element meshes
    via convex hull, so approx_radius and vertex_count are no longer needed.

    Args:
        filepath: Path to JSON file with cell data (must include 'vertices' key).
        approx_radius: DEPRECATED - No longer used (vertices come from extraction).
        vertex_count: DEPRECATED - No longer used (vertices come from extraction).
        adhesion_distance: Tolerance for merging vertices into global pool.
        skip_boundary: If True, skip cells with identifier == 'bnd' (default True).

    Returns:
        Tissue with global vertex pool established.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, 'r') as f:
        cells_data = json.load(f)

    # Filter cells
    if skip_boundary:
        cells_data = [c for c in cells_data if c.get('identifier') != 'bnd']

    # Check if new format (with 'vertices' key) or old format (with 'x', 'y' keys)
    if cells_data and 'vertices' in cells_data[0]:
        # New format: use actual vertices from extraction
        return _build_tissue_from_vertices(cells_data, adhesion_distance)
    elif cells_data and 'x' in cells_data[0] and 'y' in cells_data[0]:
        # Old format: fall back to polygon generation (deprecated)
        if approx_radius is None:
            approx_radius = 0.4
        if vertex_count is None:
            vertex_count = 6
        return _build_tissue_from_cell_data(cells_data, approx_radius, vertex_count, adhesion_distance)
    else:
        raise ValueError(
            "Invalid JSON format: expected either 'vertices' (new) or 'x'/'y' (old) keys in cell data"
        )


def load_acam_tissue(
    filepath: Union[str, Path],
    approx_radius: float = 0.4,
    vertex_count: int = 6,
    adhesion_distance: float = 1.0,
    skip_boundary: bool = True,
) -> Tissue:
    """Load an ACAM epithelium and convert to vertex model tissue.

    Converts a center-based ACAM tissue (cells represented by (x, y) centers with
    adhesion forces) into a vertex model tissue by:
    1. Loading the ACAM epithelium from a dill-pickled file
    2. Extracting cell centers and identifiers
    3. Approximating each cell as a regular polygon around its center
    4. Merging vertices within adhesion distance to create shared vertex pool

    Args:
        filepath: Path to ACAM epithelium file (dill pickle).
        approx_radius: Radius of regular polygon approximation for each cell.
                      Should be small enough to avoid initial overlaps (~0.4).
        vertex_count: Number of vertices per regular polygon (default 6 = hexagon).
        adhesion_distance: Tolerance for merging vertices into global pool.
                          Typically the adhesion equilibrium distance (~1.0).
        skip_boundary: If True, skip cells with identifier == 'bnd' (default True).

    Returns:
        Tissue with global vertex pool established and cells mechanically coupled.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If ACAM epithelium structure is invalid.
        ImportError: If ACAM file requires unavailable modules.

    Example:
        >>> tissue = load_acam_tissue('80_cells', approx_radius=0.4, adhesion_distance=1.0)
        >>> print(f"Loaded {len(tissue.cells)} cells with {tissue.vertices.shape[0]} vertices")

    Notes:
        - Boundary cells (identifier='bnd') are filtered out by default.
        - The high tolerance in build_global_vertices creates vertex sharing between
          adjacent cells, essential for mechanical coupling in vertex model simulations.
        - Original ACAM cell identifiers are preserved as cell.id (converted to int if needed).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ACAM tissue file not found: {filepath}")

    # Try JSON first (simpler, more robust)
    if filepath.suffix == '.json':
        return load_acam_from_json(filepath, approx_radius, vertex_count, adhesion_distance, skip_boundary)

    # Attempt to load ACAM epithelium with custom unpickler
    try:
        with open(filepath, "rb") as f:
            unpickler = _ACamStubUnpickler(f)
            epithelium = unpickler.load()
    except Exception as e:
        raise ImportError(
            f"Failed to load ACAM tissue from {filepath}. "
            f"Error: {e}. "
            "Consider using extract_acam_data.py to create a JSON version first, "
            "then load with load_acam_from_json()."
        ) from e

    # Validate structure
    if not hasattr(epithelium, "cells"):
        raise ValueError(
            f"Invalid ACAM epithelium structure: missing 'cells' attribute. "
            f"Type: {type(epithelium)}, Attributes: {[a for a in dir(epithelium) if not a.startswith('_')]}"
        )

    cells_data = []

    # Extract cell data
    for i, acam_cell in enumerate(epithelium.cells):
        # Check for boundary cell marker
        identifier = getattr(acam_cell, "identifier", None)
        if skip_boundary and identifier == "bnd":
            continue

        # Extract position
        x = getattr(acam_cell, "x", None)
        y = getattr(acam_cell, "y", None)

        if x is None or y is None:
            continue  # Skip cells without position

        # Handle numpy arrays
        if hasattr(x, '__iter__') and not isinstance(x, str):
            x = float(x[0]) if len(x) > 0 else 0.0
        if hasattr(y, '__iter__') and not isinstance(y, str):
            y = float(y[0]) if len(y) > 0 else 0.0

        x = float(x)
        y = float(y)

        # Extract or generate cell ID
        cell_id = getattr(acam_cell, "id", i + 1)
        if hasattr(cell_id, '__iter__'):
            cell_id = int(cell_id[0]) if len(cell_id) > 0 else i + 1
        else:
            cell_id = int(cell_id)

        cells_data.append({"id": cell_id, "x": x, "y": y, "identifier": identifier})

    if not cells_data:
        raise ValueError(
            f"No valid cells found in ACAM tissue. "
            f"Total cells in file: {len(epithelium.cells)}"
        )

    # Build tissue from extracted data
    return _build_tissue_from_cell_data(cells_data, approx_radius, vertex_count, adhesion_distance)


def _build_tissue_from_vertices(
    cells_data: List[Dict[str, Any]],
    adhesion_distance: float,
) -> Tissue:
    """Build vertex tissue from extracted cell boundary vertices.

    Args:
        cells_data: List of dicts with 'vertices', 'id', 'identifier' keys.
        adhesion_distance: Tolerance for vertex merging.

    Returns:
        Tissue with global vertex pool.
    """
    tissue = Tissue()

    for cell_data in cells_data:
        # Extract vertices (already in proper format from convex hull extraction)
        vertices = np.array(cell_data["vertices"], dtype=float)
        cell_id = cell_data.get("id", len(tissue.cells) + 1)

        # Create cell with vertices
        cell = Cell(cell_id=cell_id, vertices=vertices)

        # Store ACAM identifier as a custom attribute for later use
        identifier = cell_data.get("identifier", None)
        if identifier:
            cell.acam_identifier = str(identifier)  # Store original ACAM label

        tissue.add_cell(cell)

    # Merge vertices within adhesion distance to create global vertex pool
    tissue.build_global_vertices(tol=adhesion_distance)

    return tissue


def _build_tissue_from_cell_data(
    cells_data: List[Dict[str, Any]],
    approx_radius: float,
    vertex_count: int,
    adhesion_distance: float,
) -> Tissue:
    """Build vertex tissue from extracted cell data.

    Args:
        cells_data: List of dicts with 'id', 'x', 'y' keys.
        approx_radius: Polygon radius.
        vertex_count: Number of vertices per polygon.
        adhesion_distance: Tolerance for vertex merging.

    Returns:
        Tissue with global vertex pool.
    """
    tissue = Tissue()

    for cell_data in cells_data:
        # Generate regular polygon centered at (x, y)
        vertices = _generate_regular_polygon(
            center_x=cell_data["x"],
            center_y=cell_data["y"],
            radius=approx_radius,
            n_sides=vertex_count,
        )
        tissue.add_cell(Cell(cell_id=cell_data["id"], vertices=vertices))

    # Merge vertices within adhesion distance to create global vertex pool
    tissue.build_global_vertices(tol=adhesion_distance)

    return tissue


def _generate_regular_polygon(
    center_x: float, center_y: float, radius: float, n_sides: int
) -> np.ndarray:
    """Generate vertices of a regular polygon.

    Args:
        center_x: X coordinate of center.
        center_y: Y coordinate of center.
        radius: Distance from center to each vertex.
        n_sides: Number of sides (vertices).

    Returns:
        Array of shape (n_sides, 2) with vertex coordinates.
    """
    if n_sides < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides}")

    # Start at angle 0 (vertex at (center_x + radius, center_y))
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = np.zeros((n_sides, 2), dtype=float)
    vertices[:, 0] = center_x + radius * np.cos(angles)
    vertices[:, 1] = center_y + radius * np.sin(angles)
    return vertices


# ============================================================================
# Topology-Aware ACAM Conversion (Advanced)
# ============================================================================


class ConversionResult(NamedTuple):
    """Results from topology-aware ACAM tissue conversion."""
    tissue: Tissue
    summary: Dict[str, Any]
    validation: Optional[Dict[str, Any]] = None


@dataclass
class _CellPolygon:
    """Internal data structure for cell polygon during conversion."""
    name: str
    cell_id: int
    pref_area: Optional[float]
    vertices: np.ndarray  # Simplified polygon coordinates (N×2, CCW)
    hull_vertex_count: int
    fe_point_count: int
    is_boundary: bool


def _load_acam_neighbors(filepath: Union[str, Path]) -> Dict[str, List[str]]:
    """Load ACAM neighbor topology from JSON file."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"ACAM neighbor topology file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_cells_with_topology(
    acam_file: Union[str, Path],
    neighbor_topology: Dict[str, List[str]],
    target_identifiers: Optional[List[str]] = None
) -> Tuple[Dict[str, Dict], Dict[str, List[str]]]:
    """Load all cells from ACAM pickle file with neighbor topology.

    Returns:
        cells_data: {identifier: {'x', 'y', 'id', 'pref_area', 'n_neighbors', 'is_boundary'}}
        neighbor_map: {identifier: [list of neighbor identifiers (excluding 'bnd')]}
    """
    acam_file = Path(acam_file)
    if not acam_file.exists():
        raise FileNotFoundError(f"ACAM tissue file not found: {acam_file}")

    with open(acam_file, 'rb') as f:
        epithelium = _ACamStubUnpickler(f).load()

    if not hasattr(epithelium, "cells"):
        raise ValueError(
            f"Invalid ACAM epithelium structure: missing 'cells' attribute. "
            f"Type: {type(epithelium)}"
        )

    # If no target identifiers specified, use all from neighbor topology
    if target_identifiers is None:
        target_identifiers = list(neighbor_topology.keys())

    cells_data = {}
    neighbor_map = {}

    # Build cell reference mapping
    available_cells = {}
    for cell in epithelium.cells:
        identifier = str(getattr(cell, 'identifier', None))
        if identifier in target_identifiers:
            available_cells[identifier] = cell

    # Process each cell
    for identifier, cell in available_cells.items():
        x = getattr(cell, 'x', None)
        y = getattr(cell, 'y', None)
        cell_id = getattr(cell, 'id', None)
        pref_area = getattr(cell, 'pref_area', None)

        if x is None or y is None:
            continue

        # Convert to arrays
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        # Get neighbors from topology
        acam_neighbors = neighbor_topology.get(identifier, [])

        # Check if boundary cell
        is_boundary = 'bnd' in acam_neighbors

        # Filter out 'bnd' from neighbors
        neighbors = [n for n in acam_neighbors if n != 'bnd']
        n_neighbors = len(neighbors)

        cells_data[identifier] = {
            'x': x,
            'y': y,
            'id': int(cell_id) if cell_id else len(cells_data),
            'pref_area': float(pref_area) if pref_area else None,
            'n_points': len(x),
            'n_neighbors': n_neighbors,
            'is_boundary': is_boundary
        }
        neighbor_map[identifier] = neighbors

    return cells_data, neighbor_map


def _compute_convex_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return convex hull vertices ordered counter-clockwise."""
    points = np.column_stack([x, y])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    polygon = Polygon(hull_points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    coords = np.array(polygon.exterior.coords[:-1], dtype=float)
    return _ensure_ccw(coords)


def _ensure_ccw(coords: np.ndarray) -> np.ndarray:
    """Ensure polygon vertices are ordered counter-clockwise."""
    if coords.shape[0] < 3:
        return coords
    area = 0.5 * np.sum(
        coords[:, 0] * np.roll(coords[:, 1], -1) - coords[:, 1] * np.roll(coords[:, 0], -1)
    )
    if area < 0:
        return coords[::-1]
    return coords


def _simplify_polygon(
    coords: np.ndarray,
    target_vertices: int,
    tol_min: float = 1e-3,
    tol_max: float = 50.0
) -> np.ndarray:
    """Simplify polygon to target vertex count using binary search."""
    if coords.shape[0] <= target_vertices:
        return coords

    polygon = Polygon(coords)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    best = coords
    best_diff = float('inf')
    lo, hi = tol_min, tol_max

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        simplified = polygon.simplify(mid, preserve_topology=True)
        simplified_coords = np.array(simplified.exterior.coords[:-1], dtype=float)

        if simplified_coords.shape[0] < 3:
            hi = mid
            continue

        diff = abs(simplified_coords.shape[0] - target_vertices)
        if diff < best_diff:
            best = simplified_coords
            best_diff = diff

        if simplified_coords.shape[0] == target_vertices:
            best = simplified_coords
            break

        if simplified_coords.shape[0] <= target_vertices:
            hi = mid
        else:
            lo = mid

    return _ensure_ccw(best)


def _build_cell_polygons(
    cells_data: Dict[str, Dict],
    neighbor_map: Dict[str, List[str]],
    max_vertices: int
) -> List[_CellPolygon]:
    """Generate simplified polygons for all ACAM cells using neighbor topology.

    For boundary cells (cells with 'bnd' in their ACAM neighbors), we add 1 to the
    neighbor count before simplification to compensate for removing 'bnd' from the
    neighbor list. This ensures boundary cells have enough vertices to stay connected
    to the tissue interior.
    """
    polygons: List[_CellPolygon] = []

    for name, data in cells_data.items():
        hull_coords = _compute_convex_hull(data['x'], data['y'])

        # Use ACAM neighbor count as target
        n_neighbors = data.get('n_neighbors', 6)

        # For boundary cells, add 1 to compensate for removed 'bnd' neighbor
        if data.get('is_boundary', False):
            n_neighbors += 1

        target_vertices = min(n_neighbors, max_vertices) if n_neighbors > 0 else max_vertices
        target_vertices = max(target_vertices, 3)  # minimum triangle

        simplified = _simplify_polygon(hull_coords, target_vertices=target_vertices)

        polygons.append(
            _CellPolygon(
                name=name,
                cell_id=data['id'],
                pref_area=data.get('pref_area'),
                vertices=simplified,
                hull_vertex_count=int(hull_coords.shape[0]),
                fe_point_count=data['n_points'],
                is_boundary=data['is_boundary']
            )
        )

    return polygons


def _fuse_junction_vertices(
    cell_polygons: List[_CellPolygon],
    merge_radius: float
) -> Tuple[Tissue, Dict[str, List[int]]]:
    """Cluster inter-cell vertices and build a Tissue with shared vertex pool."""
    vertex_records: List[Tuple[str, int, np.ndarray]] = []
    record_lookup: Dict[Tuple[str, int], int] = {}

    for poly in cell_polygons:
        for local_idx, coord in enumerate(poly.vertices):
            vertex_records.append((poly.name, local_idx, coord))
            record_lookup[(poly.name, local_idx)] = len(vertex_records) - 1

    coords = np.array([rec[2] for rec in vertex_records], dtype=float)
    tree = cKDTree(coords)
    parent = list(range(len(coords)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i in range(len(coords)):
        neighbors = tree.query_ball_point(coords[i], merge_radius)
        for j in neighbors:
            if i == j:
                continue
            if vertex_records[i][0] == vertex_records[j][0]:
                continue  # only fuse across different cells
            union(i, j)

    cluster_map: Dict[int, List[int]] = {}
    for idx in range(len(coords)):
        root = find(idx)
        cluster_map.setdefault(root, []).append(idx)

    cluster_centers: Dict[int, np.ndarray] = {
        root: np.mean(coords[members], axis=0) for root, members in cluster_map.items()
    }

    index_lookup: Dict[int, int] = {}
    global_vertices: List[np.ndarray] = []
    for new_idx, (root, center) in enumerate(cluster_centers.items()):
        global_vertices.append(center)
        index_lookup[root] = new_idx

    tissue = Tissue()
    tissue.vertices = np.array(global_vertices, dtype=float)

    per_cell_vertex_indices: Dict[str, List[int]] = {}
    for poly in cell_polygons:
        indices: List[int] = []
        for local_idx, coord in enumerate(poly.vertices):
            rec_index = record_lookup[(poly.name, local_idx)]
            root = find(rec_index)
            indices.append(index_lookup[root])

        cell = Cell(cell_id=poly.cell_id, vertex_indices=np.array(indices, dtype=int))
        cell.vertices = tissue.vertices[cell.vertex_indices]
        cell.acam_identifier = poly.name
        tissue.add_cell(cell)
        per_cell_vertex_indices[poly.name] = indices

    tissue.reconstruct_cell_vertices()
    return tissue, per_cell_vertex_indices


def _validate_connectivity(
    tissue: Tissue,
    neighbor_map: Dict[str, List[str]],
    vertex_indices: Dict[str, List[int]]
) -> Tuple[List, List, List]:
    """Check ACAM neighbor connectivity and categorize sharing type.

    Returns:
        edge_connected: Pairs sharing ≥2 vertices (edge contact)
        corner_connected: Pairs sharing exactly 1 vertex (corner contact)
        disconnected: Pairs sharing 0 vertices (truly disconnected)
    """
    edge_connected = []
    corner_connected = []
    disconnected = []

    for cell_id, acam_neighbors in neighbor_map.items():
        cell_verts = set(vertex_indices[cell_id])

        for neighbor_id in acam_neighbors:
            if neighbor_id not in vertex_indices:
                continue  # Skip if neighbor not in tissue

            neighbor_verts = set(vertex_indices[neighbor_id])
            shared = cell_verts & neighbor_verts

            if len(shared) >= 2:
                edge_connected.append((cell_id, neighbor_id, len(shared)))
            elif len(shared) == 1:
                corner_connected.append((cell_id, neighbor_id, 1))
            else:
                disconnected.append((cell_id, neighbor_id))

    return edge_connected, corner_connected, disconnected


def _create_summary(
    polygons: List[_CellPolygon],
    vertex_indices: Dict[str, List[int]],
    cells_data: Dict[str, Dict],
    neighbor_map: Dict[str, List[str]],
    merge_radius: float,
    max_vertices: int
) -> Dict[str, Any]:
    """Create comprehensive summary including boundary classification."""
    boundary_count = sum(1 for p in polygons if p.is_boundary)
    interior_count = len(polygons) - boundary_count

    summary = {
        'merge_radius': merge_radius,
        'max_vertices': max_vertices,
        'total_cells': len(polygons),
        'boundary_cells': boundary_count,
        'interior_cells': interior_count,
        'cells': [],
    }

    for poly in polygons:
        n_neighbors = cells_data[poly.name].get('n_neighbors', 0)
        neighbors = neighbor_map.get(poly.name, [])

        summary['cells'].append({
            'name': poly.name,
            'acam_id': poly.cell_id,
            'is_boundary': poly.is_boundary,
            'fe_point_count': poly.fe_point_count,
            'hull_vertex_count': poly.hull_vertex_count,
            'simplified_vertex_count': int(poly.vertices.shape[0]),
            'acam_neighbors': n_neighbors,
            'acam_neighbor_ids': neighbors,
            'target_vertices': min(n_neighbors, max_vertices) if n_neighbors > 0 else max_vertices,
            'vertex_indices': [int(v) for v in vertex_indices[poly.name]],
        })

    return summary


def _create_validation_dict(
    edge_connected: List,
    corner_connected: List,
    disconnected: List
) -> Dict[str, Any]:
    """Create validation results dictionary."""
    total_neighbor_pairs = len(edge_connected) + len(corner_connected) + len(disconnected)
    total_connected = len(edge_connected) + len(corner_connected)

    return {
        'total_neighbor_pairs': total_neighbor_pairs,
        'edge_connected': {
            'count': len(edge_connected),
            'percentage': len(edge_connected) / total_neighbor_pairs * 100 if total_neighbor_pairs > 0 else 0,
            'pairs': edge_connected
        },
        'corner_connected': {
            'count': len(corner_connected),
            'percentage': len(corner_connected) / total_neighbor_pairs * 100 if total_neighbor_pairs > 0 else 0,
            'pairs': corner_connected
        },
        'disconnected': {
            'count': len(disconnected),
            'percentage': len(disconnected) / total_neighbor_pairs * 100 if total_neighbor_pairs > 0 else 0,
            'pairs': disconnected
        },
        'total_connectivity': {
            'count': total_connected,
            'total': total_neighbor_pairs,
            'percentage': total_connected / total_neighbor_pairs * 100 if total_neighbor_pairs > 0 else 0
        }
    }


def convert_acam_with_topology(
    acam_file: Union[str, Path] = 'acam_tissues/80_cells',
    neighbor_json: Union[str, Path] = 'acam_tissues/acam_79_neighbors.json',
    merge_radius: float = 14.0,
    max_vertices: int = 10,
    validate_connectivity: bool = False,
    save_summary: Optional[Union[str, Path]] = None,
    save_validation: Optional[Union[str, Path]] = None,
    verbose: bool = False
) -> ConversionResult:
    """Convert ACAM tissue to vertex model using neighbor topology information.

    This is an advanced converter that uses ACAM neighbor topology to create
    topology-aware vertex models where each cell's vertex count matches its
    neighbor count. This approach ensures better mechanical coupling between cells.

    Args:
        acam_file: Path to ACAM pickle file containing cell geometry data.
        neighbor_json: Path to JSON file with neighbor topology information.
        merge_radius: Radius for fusing junction vertices across cells.
        max_vertices: Maximum vertices per cell (safety cap).
        validate_connectivity: If True, validate ACAM neighbor connectivity.
        save_summary: Optional path to save summary JSON file.
        save_validation: Optional path to save validation report text file.
        verbose: If True, print progress information.

    Returns:
        ConversionResult containing:
            - tissue: Converted Tissue object with global vertex pool
            - summary: Dictionary with conversion statistics and cell information
            - validation: Optional dictionary with connectivity validation results

    Example:
        >>> result = convert_acam_with_topology(
        ...     acam_file='acam_tissues/80_cells',
        ...     neighbor_json='acam_tissues/acam_79_neighbors.json',
        ...     merge_radius=14.0,
        ...     validate_connectivity=True
        ... )
        >>> print(f"Converted {result.summary['total_cells']} cells")
        >>> tissue = result.tissue

    Notes:
        - Boundary cells (with 'bnd' in neighbor list) get extra vertices
        - Vertex fusion uses spatial clustering with configurable merge radius
        - Validation checks if ACAM neighbors share vertices (edge/corner/disconnected)
    """
    if verbose:
        print("="*70)
        print("ACAM → Vertex Model Converter (Topology-Aware)")
        print("="*70)
        print(f"\nLoading neighbor topology from: {neighbor_json}")

    # Load neighbor topology
    neighbor_topology = _load_acam_neighbors(neighbor_json)

    if verbose:
        print(f"Found {len(neighbor_topology)} cells in topology")
        print(f"\nLoading ACAM file: {acam_file}")

    # Load cells with topology
    all_cell_ids = list(neighbor_topology.keys())
    cells_data, neighbor_map = _load_cells_with_topology(
        acam_file, neighbor_topology, all_cell_ids
    )

    if verbose:
        print(f"✓ Loaded {len(cells_data)} cells with geometry data")

        # Classify boundary vs interior
        boundary_cells = [cid for cid, data in cells_data.items() if data['is_boundary']]
        interior_cells = [cid for cid, data in cells_data.items() if not data['is_boundary']]

        print(f"\nCell Classification:")
        print(f"  Boundary cells: {len(boundary_cells)}")
        print(f"  Interior cells: {len(interior_cells)}")

        # Show neighbor count distribution
        neighbor_counts = [data['n_neighbors'] for data in cells_data.values()]
        count_dist = Counter(neighbor_counts)
        print(f"\nNeighbor count distribution:")
        for n_neighbors in sorted(count_dist.keys()):
            print(f"  {n_neighbors} neighbors: {count_dist[n_neighbors]} cells")

    # Simplify all cells
    if verbose:
        print(f"\nSimplifying {len(cells_data)} cells (topology-aware)...")

    polygons = _build_cell_polygons(cells_data, neighbor_map, max_vertices=max_vertices)

    if verbose:
        vertex_counts = [len(poly.vertices) for poly in polygons]
        vertex_dist = Counter(vertex_counts)
        print(f"\nSimplified vertex count distribution:")
        for n_verts in sorted(vertex_dist.keys()):
            print(f"  {n_verts} vertices: {vertex_dist[n_verts]} cells")

    # Fuse junctions
    if verbose:
        print(f"\nFusing junction vertices (merge_radius={merge_radius})...")

    tissue, vertex_indices = _fuse_junction_vertices(polygons, merge_radius=merge_radius)

    if verbose:
        total_local_verts = sum(len(poly.vertices) for poly in polygons)
        sharing_ratio = (total_local_verts - tissue.vertices.shape[0]) / total_local_verts * 100

        print(f"\nGlobal vertex statistics:")
        print(f"  Total local vertices: {total_local_verts}")
        print(f"  Global vertices: {tissue.vertices.shape[0]}")
        print(f"  Sharing ratio: {sharing_ratio:.1f}%")

    # Create summary
    summary = _create_summary(
        polygons, vertex_indices, cells_data, neighbor_map,
        merge_radius, max_vertices
    )

    # Save summary if requested
    if save_summary:
        save_summary_path = Path(save_summary)
        with open(save_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"\n✓ Saved summary to {save_summary_path}")

    # Validate connectivity if requested
    validation_dict = None
    if validate_connectivity:
        if verbose:
            print(f"\nValidating ACAM neighbor connectivity...")

        edge_connected, corner_connected, disconnected = _validate_connectivity(
            tissue, neighbor_map, vertex_indices
        )

        validation_dict = _create_validation_dict(
            edge_connected, corner_connected, disconnected
        )

        if verbose:
            total_pairs = validation_dict['total_neighbor_pairs']
            print(f"\nConnectivity breakdown:")
            print(f"  Edge-connected (≥2 vertices): {validation_dict['edge_connected']['count']}/{total_pairs} ({validation_dict['edge_connected']['percentage']:.1f}%)")
            print(f"  Corner-connected (1 vertex): {validation_dict['corner_connected']['count']}/{total_pairs} ({validation_dict['corner_connected']['percentage']:.1f}%)")
            print(f"  Disconnected (0 vertices): {validation_dict['disconnected']['count']}/{total_pairs} ({validation_dict['disconnected']['percentage']:.1f}%)")
            print(f"  Total connectivity: {validation_dict['total_connectivity']['count']}/{total_pairs} ({validation_dict['total_connectivity']['percentage']:.1f}%)")

            if disconnected:
                print(f"\n⚠ WARNING: {len(disconnected)} truly disconnected pairs")
                print(f"  Recommendation: Check ACAM data or increase merge_radius")
            else:
                print(f"\n✓ SUCCESS: All ACAM neighbor pairs connected!")
                if corner_connected:
                    print(f"  Note: {len(corner_connected)} are corner-connected (single vertex)")

        # Save validation report if requested
        if save_validation:
            save_validation_path = Path(save_validation)
            with open(save_validation_path, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("ACAM NEIGHBOR CONNECTIVITY VALIDATION\n")
                f.write("="*70 + "\n\n")

                f.write(f"Total ACAM neighbor pairs: {validation_dict['total_neighbor_pairs']}\n")
                f.write(f"Edge-connected pairs (≥2 shared vertices): {validation_dict['edge_connected']['count']} ({validation_dict['edge_connected']['percentage']:.1f}%)\n")
                f.write(f"Corner-connected pairs (1 shared vertex): {validation_dict['corner_connected']['count']} ({validation_dict['corner_connected']['percentage']:.1f}%)\n")
                f.write(f"Disconnected pairs (0 shared vertices): {validation_dict['disconnected']['count']} ({validation_dict['disconnected']['percentage']:.1f}%)\n")
                f.write(f"Total connectivity: {validation_dict['total_connectivity']['count']}/{validation_dict['total_connectivity']['total']} ({validation_dict['total_connectivity']['percentage']:.1f}%)\n\n")

                if disconnected:
                    f.write("WARNING: The following ACAM neighbor pairs share NO vertices:\n")
                    f.write("-"*70 + "\n")
                    for cell1, cell2 in disconnected:
                        f.write(f"  {cell1} ↔ {cell2}\n")
                    f.write("\nRecommendation: Increase merge_radius or check ACAM data\n")
                else:
                    f.write("✓ SUCCESS: All ACAM neighbor pairs are connected!\n")
                    f.write("  (Either edge-connected or corner-connected)\n\n")

                if corner_connected:
                    f.write(f"\nNote: {validation_dict['corner_connected']['count']} pairs are corner-connected (single vertex).\n")
                    f.write("This is geometrically valid but provides no mechanical coupling.\n")
                    f.write("If edge connectivity is required, consider:\n")
                    f.write("  - Increasing number of vertices per cell\n")
                    f.write("  - Junction-aware vertex placement algorithm\n")

            if verbose:
                print(f"✓ Saved validation report to {save_validation_path}")

    if verbose:
        print("\n" + "="*70)
        print("Conversion complete!")
        print("="*70)

    return ConversionResult(
        tissue=tissue,
        summary=summary,
        validation=validation_dict
    )
