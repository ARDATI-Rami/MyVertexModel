"""
Core data structures for vertex model.
"""

import numpy as np
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
# Import for type hints; avoid heavy usage until implementation proceeds
try:
    from .geometry import GeometryCalculator  # type: ignore
except Exception:  # pragma: no cover - safe fallback for partial environments
    GeometryCalculator = object  # type: ignore


class Cell:
    """Represents a single cell in the vertex model.

    Current representation stores per-cell vertex coordinates in `self.vertices` as a local
    (N, 2) float array.

    FUTURE REPRESENTATION:
        The preferred approach will reference a *global* vertex pool owned by `Tissue`.
        For that, each cell will store integer indices into `Tissue.vertices` instead of
        (or in addition to) a copy of the coordinates. To prepare for this, an optional
        attribute `vertex_indices` is introduced here as a 1D int array. It is not yet
        used by other code, but serves as the migration path.
    """

    def __init__(self, cell_id: Union[int, str], vertices: Optional[np.ndarray] = None, vertex_indices: Optional[np.ndarray] = None):
        """
        Initialize a cell.
        
        Args:
            cell_id: Unique identifier for the cell (int or str).
            vertices: Array of vertex coordinates (N x 2) in local form (optional if using indices).
            vertex_indices: Optional 1D array of integer indices into a global `Tissue.vertices` array.

        Raises:
            ValueError: If vertices array doesn't have shape (N, 2) or vertex_indices is not 1D.
        """
        self.id = cell_id

        # Local coordinate storage (legacy / current usage)
        if vertices is None:
            self.vertices = np.empty((0, 2), dtype=float)
        else:
            vertices = np.asarray(vertices, dtype=float)
            if vertices.ndim == 0 or (vertices.ndim == 1 and len(vertices) == 0):
                # Handle empty array edge cases
                self.vertices = np.empty((0, 2), dtype=float)
            elif vertices.ndim == 1:
                raise ValueError(f"Vertices must have shape (N, 2), got shape {vertices.shape}")
            elif vertices.ndim == 2:
                if vertices.shape[1] != 2:
                    raise ValueError(f"Vertices must have shape (N, 2), got shape {vertices.shape}")
                self.vertices = vertices
            else:
                raise ValueError(f"Vertices must be a 2D array with shape (N, 2), got {vertices.ndim}D array")

        # Optional global index representation (future usage)
        if vertex_indices is None:
            self.vertex_indices = np.empty((0,), dtype=int)
        else:
            vi = np.asarray(vertex_indices, dtype=int)
            if vi.ndim != 1:
                raise ValueError(f"vertex_indices must be a 1D array of ints, got shape {vi.shape}")
            self.vertex_indices = vi

    def __repr__(self):
        return f"Cell(id={self.id}, n_vertices={len(self.vertices)})"


class Tissue:
    """Represents a collection of cells forming a tissue.

    Attributes:
        cells: List of `Cell` objects.
        vertices: Global vertex coordinate pool (M, 2) float array. Initially empty.
                  In a future refactor, cells will primarily reference this array via
                  their `vertex_indices` instead of storing local copies in `cell.vertices`.
    """

    def __init__(self):
        """Initialize an empty tissue."""
        self.cells: List[Cell] = []
        self.vertices: np.ndarray = np.empty((0, 2), dtype=float)

    def add_cell(self, cell: Cell):
        """Add a cell to the tissue."""
        self.cells.append(cell)

    def cell_neighbor_counts(self) -> Dict[int, int]:
        """Return the number of neighbors for each cell.

        Two cells are considered neighbors if they share at least one polygon edge.
        The method prefers the global vertex representation via ``cell.vertex_indices``
        when available; otherwise it falls back to comparing local ``cell.vertices``
        coordinates with a tolerance.

        Returns:
            Dict[int, int]: Mapping from cell.id to number of distinct neighboring cells.
        """
        # Prefer global vertex indices when available for robustness
        use_global = any(cell.vertex_indices.shape[0] > 0 for cell in self.cells)

        # Build per-cell edge sets represented as unordered vertex index pairs
        cell_edges: Dict[int, set] = {}

        if use_global and self.vertices.shape[0] > 0:
            for cell in self.cells:
                idx = cell.vertex_indices
                if idx.shape[0] < 2:
                    cell_edges[cell.id] = set()
                    continue
                # Close the polygon
                edges = set()
                for i in range(len(idx)):
                    a = int(idx[i])
                    b = int(idx[(i + 1) % len(idx)])
                    if a == b:
                        continue
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
                cell_edges[cell.id] = edges
        else:
            # Fallback: build a temporary global pool based on coordinates with a tight tolerance
            # This avoids double-implementing the globalisation logic
            # Reuse build_global_vertices non-destructively by working on a shallow copy
            tmp = Tissue()
            for c in self.cells:
                tmp.add_cell(Cell(cell_id=c.id, vertices=c.vertices.copy()))
            tmp.build_global_vertices(tol=1e-10)
            for cell in tmp.cells:
                idx = cell.vertex_indices
                if idx.shape[0] < 2:
                    cell_edges[cell.id] = set()
                    continue
                edges = set()
                for i in range(len(idx)):
                    a = int(idx[i])
                    b = int(idx[(i + 1) % len(idx)])
                    if a == b:
                        continue
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
                cell_edges[cell.id] = edges

        # Build neighbor sets by shared edges
        neighbors: Dict[int, set] = {cell.id: set() for cell in self.cells}
        cell_ids = [cell.id for cell in self.cells]
        id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}

        # Compare edges of every pair of cells; typical cell count is modest so O(N^2) is acceptable
        for i, ci in enumerate(cell_ids):
            edges_i = cell_edges.get(ci, set())
            if not edges_i:
                continue
            for j in range(i + 1, len(cell_ids)):
                cj = cell_ids[j]
                edges_j = cell_edges.get(cj, set())
                if not edges_j:
                    continue
                if edges_i.intersection(edges_j):
                    neighbors[ci].add(cj)
                    neighbors[cj].add(ci)

        return {cid: len(nbrs) for cid, nbrs in neighbors.items()}

    def __repr__(self):
        return f"Tissue(n_cells={len(self.cells)})"

    def validate(self) -> None:
        """
        Validate the structural integrity of the tissue.

        Checks that:
        - Each cell has properly shaped vertices (N x 2)
        - Each cell polygon is valid (non-self-intersecting)
        - Each cell has non-negative area

        Raises:
            ValueError: If any validation check fails
        """
        from .geometry import GeometryCalculator, is_valid_polygon

        for cell in self.cells:
            # Check vertices shape
            if cell.vertices.ndim != 2:
                raise ValueError(f"Cell {cell.id}: vertices must be a 2D array, got {cell.vertices.ndim}D")

            if cell.vertices.shape[0] > 0 and cell.vertices.shape[1] != 2:
                raise ValueError(f"Cell {cell.id}: vertices must have shape (N, 2), got {cell.vertices.shape}")

            # Skip empty cells
            if len(cell.vertices) == 0:
                continue

            # Check polygon validity
            if not is_valid_polygon(cell.vertices):
                raise ValueError(f"Cell {cell.id}: polygon is invalid (self-intersecting, degenerate, or has zero area)")

            # Check non-negative area
            area = GeometryCalculator.calculate_area(cell.vertices)
            if area < 0:
                raise ValueError(f"Cell {cell.id}: polygon has negative area {area}")

    def build_global_vertices(self, tol: float = 1e-8) -> None:
        """
        Build a global vertex pool from per-cell local vertices.

        Scans all cell.vertices arrays, merges geometrically identical vertices
        (within tolerance tol), and constructs:
        - self.vertices: Global (M, 2) array of unique vertex coordinates
        - cell.vertex_indices: For each cell, a 1D array of indices into self.vertices

        This migration prepares the tissue for a shared vertex representation
        where vertices belong to the tissue, not individual cells.

        Args:
            tol: Tolerance for considering two vertices identical (default: 1e-8)

        Notes:
            - cell.vertices arrays are NOT modified by this method
            - After calling this, cells reference vertices via both:
              * cell.vertices (local copy, unchanged)
              * cell.vertex_indices (indices into self.vertices)
            - Use reconstruct_cell_vertices() to rebuild local copies from global pool
        """
        # Collect all vertices from all cells
        all_vertices = []
        cell_vertex_counts = []

        for cell in self.cells:
            if cell.vertices.shape[0] > 0:
                all_vertices.append(cell.vertices)
                cell_vertex_counts.append(cell.vertices.shape[0])
            else:
                cell_vertex_counts.append(0)

        if not all_vertices:
            # No vertices in any cell
            self.vertices = np.empty((0, 2), dtype=float)
            for cell in self.cells:
                cell.vertex_indices = np.empty((0,), dtype=int)
            return

        # Stack all vertices into a single array
        all_verts_array = np.vstack(all_vertices)

        # Find unique vertices within tolerance
        unique_vertices = []
        vertex_map = {}  # Map from (approx_x, approx_y) -> global index
        global_indices = []  # List of global indices for each vertex in all_verts_array

        for i, vert in enumerate(all_verts_array):
            # Round to tolerance grid to find nearby vertices
            key = (round(vert[0] / tol) * tol, round(vert[1] / tol) * tol)

            # Check if we've seen this vertex before (within tolerance)
            found = False
            for existing_key, idx in vertex_map.items():
                if abs(existing_key[0] - key[0]) < tol and abs(existing_key[1] - key[1]) < tol:
                    # Found a match
                    global_indices.append(idx)
                    found = True
                    break

            if not found:
                # New unique vertex
                idx = len(unique_vertices)
                unique_vertices.append(vert.copy())
                vertex_map[key] = idx
                global_indices.append(idx)

        # Store global vertex array
        self.vertices = np.array(unique_vertices, dtype=float)

        # Assign vertex_indices to each cell and deduplicate consecutive duplicates
        offset = 0
        for cell, count in zip(self.cells, cell_vertex_counts):
            if count > 0:
                indices = np.array(global_indices[offset:offset+count], dtype=int)

                cell.vertex_indices = indices
                offset += count
            else:
                cell.vertex_indices = np.empty((0,), dtype=int)

    def reconstruct_cell_vertices(self) -> None:
        """
        Reconstruct cell.vertices from global vertex pool.

        For each cell that has non-empty vertex_indices, recomputes cell.vertices
        by looking up coordinates from self.vertices[cell.vertex_indices].

        This is the inverse of build_global_vertices(), allowing round-trip conversion
        between local per-cell representation and global shared representation.

        Notes:
            - Cells with empty vertex_indices are left unchanged
            - Requires self.vertices to be populated (e.g., via build_global_vertices)
            - After calling this, cell.vertices will match the global pool coordinates
        """
        for cell in self.cells:
            if cell.vertex_indices.shape[0] > 0:
                # Reconstruct from global pool
                cell.vertices = self.vertices[cell.vertex_indices].copy()
            elif cell.vertices.shape[0] > 0 and self.vertices.shape[0] > 0:
                # Legacy cells may only store local vertices; build indices on demand
                indices = []
                for vert in cell.vertices:
                    diffs = np.linalg.norm(self.vertices - vert, axis=1)
                    idx = int(np.argmin(diffs))
                    if diffs[idx] > 1e-10:
                        raise ValueError(
                            "Cannot reconstruct vertex_indices for cell without matching global vertices"
                        )
                    # Avoid duplicate consecutive indices while preserving order
                    if indices and indices[-1] == idx:
                        continue
                    indices.append(idx)
                cell.vertex_indices = np.array(indices, dtype=int)


# ---------------- Energy API (interfaces only / stubs) ---------------- #

@dataclass
class EnergyParameters:
    """Holds mechanical energy parameters for the vertex model.

    Parameters reflect common vertex model formulations (e.g. Farhadifar et al. 2007;
    Nagai & Honda 2009), typically combining area elasticity, perimeter contractility,
    and edge tension terms.

    Attributes:
        k_area: Coefficient for area elasticity term.
        k_perimeter: Coefficient for perimeter contractility term.
        gamma: Effective edge tension (may later be refined per-edge).
        target_area: Preferred area A0 for cells. Can be:
            - float: Single global target area for all cells
            - Dict[int, float]: Per-cell target areas keyed by cell.id
    """
    k_area: float = 1.0
    k_perimeter: float = 0.1
    gamma: float = 0.05
    target_area: Union[float, Dict[int, float]] = 1.0


def cell_energy(cell: Cell, params: EnergyParameters, geometry: "GeometryCalculator",
                global_vertices: Optional[np.ndarray] = None) -> float:
    """Compute the mechanical energy of a single cell.

    Implemented basic vertex-model style energy composed of:
        E_area  = 0.5 * k_area * (A - A0)^2
        E_perim = 0.5 * k_perimeter * P^2
        E_line  = gamma * P
    Total:
        E_cell = 0.5 k_area (A - A0)^2 + 0.5 k_perimeter P^2 + gamma P

    Where:
        A   = cell area (polygon area)
        A0  = target (preferred) area (params.target_area)
        P   = cell perimeter (polygon perimeter)
        gamma = uniform line / junction tension parameter

    Notes:
        - This is a simplified placeholder for more detailed formulations where
          gamma may vary per edge and perimeter term may include (P - P0)^2.
        - Supports both local (cell.vertices) and global (Tissue.vertices) representations.
        - Prefers global representation when available (vertex_indices non-empty and global_vertices provided).
        - For cells with fewer than 3 vertices, area defaults to 0.0.

    Args:
        cell: Cell instance.
        params: EnergyParameters object with model coefficients.
        geometry: GeometryCalculator (static methods used for area/perimeter).
        global_vertices: Optional global vertex array (e.g., Tissue.vertices). If provided
                        and cell.vertex_indices is non-empty, uses global representation.
                        Otherwise falls back to cell.vertices.

    Returns:
        float: Energy value (non-negative under current formulation).
    """
    # Choose vertex representation: prefer global if available
    if global_vertices is not None and cell.vertex_indices.shape[0] > 0:
        # Use global vertex pool via indices
        verts = global_vertices[cell.vertex_indices]
    else:
        # Fall back to local per-cell vertices
        verts = cell.vertices

    # Compute geometric quantities using provided geometry calculator
    area = geometry.calculate_area(verts)
    perimeter = geometry.calculate_perimeter(verts)

    # Get target area (per-cell or global)
    if isinstance(params.target_area, dict):
        target = params.target_area.get(cell.id, 1.0)  # Default to 1.0 if not found
    else:
        target = params.target_area

    # Energy components
    e_area = 0.5 * params.k_area * (area - target) ** 2
    e_perim = 0.5 * params.k_perimeter * perimeter ** 2
    e_line = params.gamma * perimeter

    return e_area + e_perim + e_line


def tissue_energy(tissue: Tissue, params: EnergyParameters, geometry: "GeometryCalculator") -> float:
    """Compute the total mechanical energy of a tissue by summing cell energies.

    If the tissue has a populated global vertex pool (tissue.vertices), passes it to
    cell_energy to enable global vertex representation. Otherwise uses local per-cell vertices.

    Args:
        tissue: Tissue instance containing cells.
        params: Energy parameters used for each cell energy evaluation.
        geometry: GeometryCalculator instance.

    Returns:
        float: Sum of individual cell energies.
    """
    # Use global vertices if available
    global_verts = tissue.vertices if tissue.vertices.shape[0] > 0 else None
    return float(sum(cell_energy(cell, params, geometry, global_vertices=global_verts)
                     for cell in tissue.cells))


def cell_energy_gradient_analytic(
    cell: Cell,
    params: EnergyParameters,
    geometry: "GeometryCalculator"
) -> np.ndarray:
    """
    Compute the analytic gradient of cell energy with respect to vertex positions.

    This function will eventually compute the exact gradient of the energy functional:
        E_cell = 0.5 * k_area * (A - A0)^2 + 0.5 * k_perimeter * P^2 + gamma * P

    with respect to each vertex coordinate (x_i, y_i), using analytical derivatives
    rather than finite differences.

    The analytic gradient will provide:
        - Exact derivatives (no numerical approximation error)
        - Better computational efficiency (no need for multiple energy evaluations)
        - More stable numerical behavior

    Mathematical formulation (to be implemented):
        ∂E/∂x_i = ∂E/∂A * ∂A/∂x_i + ∂E/∂P * ∂P/∂x_i

    where:
        ∂E/∂A = k_area * (A - A0)
        ∂E/∂P = k_perimeter * P + gamma
        ∂A/∂x_i, ∂P/∂x_i = geometric derivatives (to be derived)

    Current Implementation:
        NOT YET IMPLEMENTED - raises NotImplementedError.
        This is a placeholder for future analytical gradient computation.
        Use finite_difference_cell_gradient from simulation module for now.

    Args:
        cell: Cell instance with vertices to compute gradient for.
        params: EnergyParameters containing k_area, k_perimeter, gamma, target_area.
        geometry: GeometryCalculator (may be used for geometric computations).

    Returns:
        np.ndarray: Gradient array of shape (N, 2) where N is number of vertices.
                   Each row (i, :) contains [∂E/∂x_i, ∂E/∂y_i].

    Raises:
        NotImplementedError: Always raised in current implementation.

    Notes:
        - When implemented, should give identical results to finite-difference gradient
          (within numerical precision).
        - Will require careful derivation of area and perimeter derivatives.
        - See references: Farhadifar et al. 2007, Nagai & Honda 2009 for vertex model
          mechanics and force calculations.

    Future Work:
        1. Derive analytical expressions for ∂A/∂x_i, ∂A/∂y_i using shoelace formula
        2. Derive analytical expressions for ∂P/∂x_i, ∂P/∂y_i from edge lengths
        3. Implement and test against finite-difference gradient
        4. Optimize for computational efficiency
        5. Consider extending to support global vertex representation
    """
    raise NotImplementedError(
        "Analytic gradient computation not yet implemented. "
        "Use finite_difference_cell_gradient from myvertexmodel.simulation instead. "
        "This placeholder is reserved for future analytical derivative implementation."
    )


# ---------------- Vertex Merging and Edge Meshing ---------------- #

def _remove_consecutive_duplicates(indices: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate indices, including wrap-around.

    Args:
        indices: 1D array of integer indices.

    Returns:
        1D array with consecutive duplicates removed.
    """
    if len(indices) <= 1:
        return indices.copy()

    # Remove interior consecutive duplicates
    mask = np.ones(len(indices), dtype=bool)
    mask[1:] = indices[1:] != indices[:-1]
    result = indices[mask]

    # Remove wrap-around duplicate (first == last)
    if len(result) > 1 and result[0] == result[-1]:
        result = result[:-1]

    return result


def merge_nearby_vertices(
    tissue: "Tissue",
    distance_tol: float = 1e-8,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10,
    energy_params: Optional["EnergyParameters"] = None,
) -> Dict:
    """Merge vertices that are within distance_tol of each other.

    This operation detects clusters of near-coincident vertices in the global
    vertex pool and replaces them with a single representative vertex (the
    centroid of the cluster), updating all cell references accordingly.

    Args:
        tissue: Tissue with populated global vertex pool (tissue.vertices)
                and per-cell vertex_indices.
        distance_tol: Maximum distance between vertices to consider for merging.
        energy_tol: Maximum allowed change in total tissue energy.
        geometry_tol: Maximum allowed change in per-cell area and perimeter.
        energy_params: Optional EnergyParameters for energy validation. If None,
                      uses default parameters.

    Returns:
        dict: Diagnostics including:
            - 'vertices_before': Number of vertices before merging
            - 'vertices_after': Number of vertices after merging
            - 'clusters_merged': Number of clusters that were merged
            - 'energy_change': Absolute change in energy
            - 'max_area_change': Maximum change in any cell's area
            - 'max_perimeter_change': Maximum change in any cell's perimeter

    Raises:
        ValueError: If energy or geometry change exceeds tolerance, or if
                   merging would create a degenerate cell (< 3 vertices).

    Notes:
        - Requires tissue.vertices to be populated (call build_global_vertices first).
        - Modifies tissue in-place: updates tissue.vertices, cell.vertex_indices,
          and cell.vertices.
    """
    from .geometry import GeometryCalculator

    if tissue.vertices.shape[0] == 0:
        # Nothing to merge
        return {
            "vertices_before": 0,
            "vertices_after": 0,
            "clusters_merged": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    geom = GeometryCalculator()
    params = energy_params if energy_params is not None else EnergyParameters()

    # Step 1: Pre-compute metrics
    energy_before = tissue_energy(tissue, params, geom)
    cell_metrics_before = {}
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] >= 3:
            verts = tissue.vertices[cell.vertex_indices]
            cell_metrics_before[cell.id] = (
                geom.calculate_area(verts),
                geom.calculate_perimeter(verts),
            )

    vertices_before = tissue.vertices.shape[0]

    # Step 2: Find nearby vertex pairs and build clusters using union-find
    n_verts = len(tissue.vertices)
    parent = list(range(n_verts))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Pairwise distance check (O(n^2) - could be optimized with KD-tree for large n)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            dist = np.linalg.norm(tissue.vertices[i] - tissue.vertices[j])
            if dist <= distance_tol:
                union(i, j)

    # Build clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(n_verts):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Step 3: Build new vertex pool and index mapping
    new_vertices = []
    old_to_new: Dict[int, int] = {}
    clusters_merged = 0

    for root, members in clusters.items():
        new_idx = len(new_vertices)
        if len(members) == 1:
            # Single vertex, keep as-is
            new_vertices.append(tissue.vertices[members[0]].copy())
        else:
            # Compute centroid as representative
            cluster_verts = tissue.vertices[members]
            centroid = np.mean(cluster_verts, axis=0)
            new_vertices.append(centroid)
            clusters_merged += 1

        for old_idx in members:
            old_to_new[old_idx] = new_idx

    new_vertices_array = np.array(new_vertices, dtype=float)

    # Step 4: Update cell vertex indices
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] == 0:
            continue

        # Map old indices to new indices
        new_indices = np.array([old_to_new[idx] for idx in cell.vertex_indices], dtype=int)

        # Remove consecutive duplicates
        new_indices = _remove_consecutive_duplicates(new_indices)

        # Check for degenerate cell
        if len(new_indices) < 3:
            raise ValueError(
                f"Merging would reduce cell {cell.id} to {len(new_indices)} vertices "
                f"(minimum 3 required). Consider using a smaller distance_tol."
            )

        # Ensure CCW orientation
        temp_verts = new_vertices_array[new_indices]
        signed_area = geom.signed_area(temp_verts)
        if signed_area < 0:
            new_indices = new_indices[::-1]

        cell.vertex_indices = new_indices

    # Step 5: Update tissue and validate
    tissue.vertices = new_vertices_array
    tissue.reconstruct_cell_vertices()

    # Validate energy invariance
    energy_after = tissue_energy(tissue, params, geom)
    energy_change = abs(energy_after - energy_before)
    if energy_change > energy_tol:
        raise ValueError(
            f"Energy change {energy_change:.2e} exceeds tolerance {energy_tol:.2e}. "
            f"Merging may have altered geometry too much."
        )

    # Validate per-cell geometry invariance
    max_area_change = 0.0
    max_perimeter_change = 0.0
    for cell in tissue.cells:
        if cell.id in cell_metrics_before:
            area_before, perim_before = cell_metrics_before[cell.id]
            verts = tissue.vertices[cell.vertex_indices]
            area_after = geom.calculate_area(verts)
            perim_after = geom.calculate_perimeter(verts)
            area_change = abs(area_after - area_before)
            perim_change = abs(perim_after - perim_before)
            max_area_change = max(max_area_change, area_change)
            max_perimeter_change = max(max_perimeter_change, perim_change)
            if area_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} area change {area_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )
            if perim_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} perimeter change {perim_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )

    return {
        "vertices_before": vertices_before,
        "vertices_after": len(new_vertices_array),
        "clusters_merged": clusters_merged,
        "energy_change": energy_change,
        "max_area_change": max_area_change,
        "max_perimeter_change": max_perimeter_change,
    }


def mesh_edges(
    tissue: "Tissue",
    mode: str = "none",
    length_scale: float = 1.0,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10,
    energy_params: Optional["EnergyParameters"] = None,
) -> Dict:
    """Subdivide edges according to specified meshing mode.

    This operation refines the mesh by inserting intermediate vertices along
    edges. The subdivision density is controlled by the mode parameter.

    Args:
        tissue: Tissue with populated global vertex pool.
        mode: Meshing mode, one of:
            - "none": No subdivision (identity operation)
            - "low": Add single midpoint to each edge
            - "medium": Target ~1 vertex per 1.0 * length_scale units
            - "high": Target ~1 vertex per 0.5 * length_scale units
        length_scale: Base unit for edge subdivision (default 1.0).
        energy_tol: Maximum allowed change in total tissue energy.
        geometry_tol: Maximum allowed change in per-cell area and perimeter.
        energy_params: Optional EnergyParameters for energy validation.

    Returns:
        dict: Diagnostics including:
            - 'mode': The meshing mode used
            - 'vertices_before': Number of vertices before meshing
            - 'vertices_after': Number of vertices after meshing
            - 'edges_subdivided': Number of edges that were subdivided
            - 'energy_change': Absolute change in energy
            - 'max_area_change': Maximum change in any cell's area
            - 'max_perimeter_change': Maximum change in any cell's perimeter

    Raises:
        ValueError: If mode is invalid or if energy/geometry tolerance exceeded.

    Notes:
        - Requires tissue.vertices to be populated (call build_global_vertices first).
        - Shared edges between cells receive identical intermediate vertices.
        - Very short edges (< 0.1 * length_scale) are not subdivided in medium/high modes.
    """
    from .geometry import GeometryCalculator

    valid_modes = {"none", "low", "medium", "high"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    if tissue.vertices.shape[0] == 0:
        return {
            "mode": mode,
            "vertices_before": 0,
            "vertices_after": 0,
            "edges_subdivided": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    geom = GeometryCalculator()
    params = energy_params if energy_params is not None else EnergyParameters()

    # Step 1: Pre-compute metrics
    energy_before = tissue_energy(tissue, params, geom)
    cell_metrics_before = {}
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] >= 3:
            verts = tissue.vertices[cell.vertex_indices]
            cell_metrics_before[cell.id] = (
                geom.calculate_area(verts),
                geom.calculate_perimeter(verts),
            )

    vertices_before = tissue.vertices.shape[0]

    if mode == "none":
        # No-op
        return {
            "mode": mode,
            "vertices_before": vertices_before,
            "vertices_after": vertices_before,
            "edges_subdivided": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    # Step 2: Collect all unique edges
    all_edges: Dict[tuple, None] = {}  # Use dict for ordered unique set
    for cell in tissue.cells:
        indices = cell.vertex_indices
        n = len(indices)
        for i in range(n):
            v_start = indices[i]
            v_end = indices[(i + 1) % n]
            edge_key = (min(v_start, v_end), max(v_start, v_end))
            all_edges[edge_key] = None

    # Step 3: Build edge subdivision map
    edge_subdivision: Dict[tuple, List[int]] = {}
    new_vertices_list = list(tissue.vertices)
    next_vertex_index = len(tissue.vertices)
    edges_subdivided = 0
    min_edge_for_subdivision = 0.1 * length_scale

    for edge_key in all_edges:
        min_idx, max_idx = edge_key
        v_start = tissue.vertices[min_idx]
        v_end = tissue.vertices[max_idx]
        edge_length = np.linalg.norm(v_end - v_start)

        # Determine number of segments
        if mode == "low":
            n_segments = 2
        elif mode == "medium":
            if edge_length < min_edge_for_subdivision:
                n_segments = 1
            else:
                n_segments = max(1, round(edge_length / length_scale))
        elif mode == "high":
            if edge_length < min_edge_for_subdivision:
                n_segments = 1
            else:
                n_segments = max(1, round(edge_length / (0.5 * length_scale)))
        else:  # mode == "none" already handled above
            n_segments = 1

        if n_segments <= 1:
            edge_subdivision[edge_key] = [min_idx, max_idx]
        else:
            # Create intermediate vertices
            intermediate_indices = []
            for k in range(1, n_segments):
                t = k / n_segments
                v_new = (1 - t) * v_start + t * v_end
                new_vertices_list.append(v_new)
                intermediate_indices.append(next_vertex_index)
                next_vertex_index += 1
            edge_subdivision[edge_key] = [min_idx] + intermediate_indices + [max_idx]
            edges_subdivided += 1

    # Step 4: Reconstruct cell polygons
    new_vertices_array = np.array(new_vertices_list, dtype=float)

    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] == 0:
            continue

        old_indices = cell.vertex_indices
        n = len(old_indices)
        new_poly: List[int] = []

        for i in range(n):
            v_start = old_indices[i]
            v_end = old_indices[(i + 1) % n]
            edge_key = (min(v_start, v_end), max(v_start, v_end))
            edge_seq = edge_subdivision[edge_key]

            # Determine orientation: are we going min->max or max->min?
            if v_start == edge_key[0]:
                # Forward: min -> max, exclude last to avoid duplication
                segment = edge_seq[:-1]
            else:
                # Reverse: max -> min, exclude last after reversing
                segment = edge_seq[::-1][:-1]

            new_poly.extend(segment)

        # Remove consecutive duplicates
        new_indices = _remove_consecutive_duplicates(np.array(new_poly, dtype=int))

        # Ensure CCW orientation
        temp_verts = new_vertices_array[new_indices]
        signed_area = geom.signed_area(temp_verts)
        if signed_area < 0:
            new_indices = new_indices[::-1]

        cell.vertex_indices = new_indices

    # Step 5: Update tissue
    tissue.vertices = new_vertices_array
    tissue.reconstruct_cell_vertices()

    # Validate energy invariance
    energy_after = tissue_energy(tissue, params, geom)
    energy_change = abs(energy_after - energy_before)
    if energy_change > energy_tol:
        raise ValueError(
            f"Energy change {energy_change:.2e} exceeds tolerance {energy_tol:.2e}."
        )

    # Validate per-cell geometry invariance
    max_area_change = 0.0
    max_perimeter_change = 0.0
    for cell in tissue.cells:
        if cell.id in cell_metrics_before:
            area_before, perim_before = cell_metrics_before[cell.id]
            verts = tissue.vertices[cell.vertex_indices]
            area_after = geom.calculate_area(verts)
            perim_after = geom.calculate_perimeter(verts)
            area_change = abs(area_after - area_before)
            perim_change = abs(perim_after - perim_before)
            max_area_change = max(max_area_change, area_change)
            max_perimeter_change = max(max_perimeter_change, perim_change)
            if area_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} area change {area_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )
            if perim_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} perimeter change {perim_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )

    return {
        "mode": mode,
        "vertices_before": vertices_before,
        "vertices_after": len(new_vertices_array),
        "edges_subdivided": edges_subdivided,
        "energy_change": energy_change,
        "max_area_change": max_area_change,
        "max_perimeter_change": max_perimeter_change,
    }
