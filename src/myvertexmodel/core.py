"""
Core data structures for vertex model.
"""

import numpy as np
from typing import List, Optional, Union, Dict, Set, Sequence

# Import for type hints; avoid heavy usage until implementation proceeds
try:
    from .geometry import GeometryCalculator  # type: ignore
except Exception:  # pragma: no cover - safe fallback for partial environments
    GeometryCalculator = object  # type: ignore

CellID = Union[int, str]


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

        # Apoptosis flag (metadata only; logic handled by apoptosis module / simulation)
        self.is_apoptotic: bool = False

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

    def cell_neighbors(self) -> Dict[CellID, List[CellID]]:
        """
        Return the neighbor IDs for each cell.

        Two cells are considered neighbors if they share at least one polygon edge.
        The method prefers the global vertex representation via ``cell.vertex_indices``
        when available; otherwise it falls back to comparing local ``cell.vertices``
        coordinates with a tolerance.

        :return: Mapping from ``cell.id`` to a list of neighbor cell IDs.
        :rtype: Dict[CellID, List[CellID]]
        """
        # Prefer global vertex indices when available for robustness.
        use_global = any(
            hasattr(cell, "vertex_indices")
            and cell.vertex_indices is not None
            and len(cell.vertex_indices) > 0
            for cell in self.cells
        )

        # Build per-cell edge sets represented as unordered vertex-index pairs.
        cell_edges: Dict[CellID, Set[tuple[int, int]]] = {}

        if use_global and getattr(self, "vertices", None) is not None and len(self.vertices) > 0:
            for cell in self.cells:
                idx = cell.vertex_indices
                if idx is None or len(idx) < 2:
                    cell_edges[cell.id] = set()
                    continue

                edges: Set[tuple[int, int]] = set()
                n = len(idx)
                for i in range(n):
                    a = int(idx[i])
                    b = int(idx[(i + 1) % n])  # close polygon
                    if a == b:
                        continue
                    edges.add((a, b) if a < b else (b, a))

                cell_edges[cell.id] = edges
        else:
            # Fallback: build a temporary global pool based on coordinates with a tight tolerance.
            # This avoids re-implementing the globalisation logic.
            tmp = Tissue()
            for c in self.cells:
                tmp.add_cell(Cell(cell_id=c.id, vertices=c.vertices.copy()))
            tmp.build_global_vertices(tol=1e-10)

            for cell in tmp.cells:
                idx = cell.vertex_indices
                if idx is None or len(idx) < 2:
                    cell_edges[cell.id] = set()
                    continue

                edges: Set[tuple[int, int]] = set()
                n = len(idx)
                for i in range(n):
                    a = int(idx[i])
                    b = int(idx[(i + 1) % n])  # close polygon
                    if a == b:
                        continue
                    edges.add((a, b) if a < b else (b, a))

                cell_edges[cell.id] = edges

        # Build neighbor sets by shared edges.
        neighbors: Dict[CellID, Set[CellID]] = {cell.id: set() for cell in self.cells}
        cell_ids: List[CellID] = [cell.id for cell in self.cells]

        # Compare edges of every pair of cells; O(N^2) is acceptable at typical tissue sizes.
        for i, ci in enumerate(cell_ids):
            edges_i = cell_edges.get(ci, set())
            if not edges_i:
                continue
            for j in range(i + 1, len(cell_ids)):
                cj = cell_ids[j]
                edges_j = cell_edges.get(cj, set())
                if not edges_j:
                    continue
                if edges_i & edges_j:
                    neighbors[ci].add(cj)
                    neighbors[cj].add(ci)

        # Keep return type stable: List[CellID] (no int coercion).
        return {cid: sorted(nbrs, key=str) for cid, nbrs in neighbors.items()}

    def cell_neighbor_counts(self) -> Dict[CellID, int]:
        """
        Return the number of neighbors for each cell.

        Two cells are considered neighbors if they share at least one polygon edge.

        :return: Mapping from ``cell.id`` to the number of distinct neighboring cells.
        :rtype: Dict[CellID, int]
        """
        return {cid: len(nbrs) for cid, nbrs in self.cell_neighbors().items()}

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

    def ensure_global_vertices(self, tol: float = 1e-10, rebuild: bool = False) -> None:
        """Ensure the tissue has a coherent global vertex pool.

        This is a safe, idempotent wrapper around:
        - build_global_vertices()
        - reconstruct_cell_vertices()

        It is intended to be called by builder utilities (e.g., honeycomb builders)
        so they return a mechanically coupled tissue by default.

        Args:
            tol: Tolerance for considering vertices identical (passed to build_global_vertices).
            rebuild: If True, forces rebuilding the global pool even if it appears present.

        Notes:
            A global pool is considered "present" when:
            - tissue.vertices is non-empty AND
            - every cell either has empty geometry, or has vertex_indices of the same
              length as its local vertices.
        """
        if not rebuild:
            if self.vertices.shape[0] > 0:
                ok = True
                for cell in self.cells:
                    if cell.vertices.shape[0] == 0:
                        continue
                    if cell.vertex_indices.shape[0] != cell.vertices.shape[0]:
                        ok = False
                        break
                if ok:
                    return

        self.build_global_vertices(tol=tol)
        self.reconstruct_cell_vertices()

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

    def remove_cells(self, cell_ids: Union[List[Union[int, str]], Dict[Union[int, str], None]]):
        """Remove cells with the given IDs from the tissue.

        Args:
            cell_ids: Iterable of cell IDs to remove.
        """
        ids_set = set(cell_ids)
        self.cells = [cell for cell in self.cells if cell.id not in ids_set]

    def compute_vertex_degrees(self, tol: float = 1e-10) -> Dict[int, int]:
        """Compute the degree (number of incident cells) for each global vertex.

        For each global vertex index in tissue.vertices, counts how many cells
        reference that vertex in their vertex_indices array.

        Args:
            tol: Tolerance for considering vertices identical (unused in current implementation,
                 but kept for API consistency with other methods).

        Returns:
            Dictionary mapping global vertex index to its degree (number of cells sharing it).

        Notes:
            - Requires tissue to have global vertex representation (vertex_indices populated).
            - A vertex not referenced by any cell will not appear in the returned dict.
            - Typical degrees: 2 (edge), 3 (tri-vertex), 4+ (multi-vertex junction).
        """
        vertex_degrees: Dict[int, int] = {}

        for cell in self.cells:
            if cell.vertex_indices.shape[0] == 0:
                continue
            for idx in cell.vertex_indices:
                idx = int(idx)
                vertex_degrees[idx] = vertex_degrees.get(idx, 0) + 1

        return vertex_degrees

    def find_shared_vertices_between_cells(
        self,
        cell_id1: CellID,
        cell_id2: CellID,
        tol: float = 1e-10
    ) -> List[int]:
        """Find which global vertex indices are shared between two specific cells.

        Args:
            cell_id1: ID of first cell.
            cell_id2: ID of second cell.
            tol: Tolerance for considering vertices identical (used for fallback coordinate comparison).

        Returns:
            List of global vertex indices that appear in both cells' vertex_indices arrays.
            Returns empty list if cells don't share vertices or if either cell is not found.

        Notes:
            - Prefers global vertex_indices when available for exact matching.
            - Falls back to coordinate comparison if global representation not available.
        """
        # Find the two cells
        cell1 = next((c for c in self.cells if c.id == cell_id1), None)
        cell2 = next((c for c in self.cells if c.id == cell_id2), None)

        if cell1 is None or cell2 is None:
            return []

        # If using global vertex indices, compute intersection
        if (cell1.vertex_indices.shape[0] > 0 and cell2.vertex_indices.shape[0] > 0 and
            self.vertices.shape[0] > 0):
            shared = set(cell1.vertex_indices) & set(cell2.vertex_indices)
            return sorted(shared)

        # Fallback: compare coordinates with tolerance
        shared_indices = []
        if cell1.vertices.shape[0] > 0 and cell2.vertices.shape[0] > 0:
            for i, v1 in enumerate(cell1.vertices):
                for j, v2 in enumerate(cell2.vertices):
                    if np.linalg.norm(v1 - v2) < tol:
                        # This is a shared vertex; try to find its global index
                        if self.vertices.shape[0] > 0:
                            diffs = np.linalg.norm(self.vertices - v1, axis=1)
                            idx = int(np.argmin(diffs))
                            if diffs[idx] < tol and idx not in shared_indices:
                                shared_indices.append(idx)
                        break

        return sorted(shared_indices)

    def get_cells_sharing_vertices_with(
        self,
        cell_id: CellID,
        tol: float = 1e-10
    ) -> Dict[CellID, List[int]]:
        """Get all neighbors of a cell and their specific shared vertex indices.

        For a given cell, finds all neighboring cells (those sharing at least one vertex)
        and returns which specific global vertex indices they share.

        Args:
            cell_id: ID of the cell to find neighbors for.
            tol: Tolerance for considering vertices identical.

        Returns:
            Dictionary mapping neighbor cell IDs to lists of shared global vertex indices.
            Only includes neighbors that share at least one vertex.

        Notes:
            - This is more detailed than cell_neighbors() which only returns neighbor IDs.
            - Useful for topological operations that need to know exact shared vertices.
        """
        neighbors_shared: Dict[CellID, List[int]] = {}

        target_cell = next((c for c in self.cells if c.id == cell_id), None)
        if target_cell is None:
            return neighbors_shared

        for other_cell in self.cells:
            if other_cell.id == cell_id:
                continue

            shared = self.find_shared_vertices_between_cells(cell_id, other_cell.id, tol=tol)
            if shared:
                neighbors_shared[other_cell.id] = shared

        return neighbors_shared

    def move_shared_vertices_toward_point(
        self,
        vertex_indices: Sequence[int],
        target_point: np.ndarray,
        move_fraction: float = 1.0
    ) -> None:
        """Move specified global vertices toward a target point.

        Updates the global vertex pool (tissue.vertices) by moving the specified
        vertices toward a target point (typically the centroid of a removed cell).

        Args:
            vertex_indices: List of global vertex indices to move.
            target_point: Target coordinates (2D array or list) to move vertices toward.
            move_fraction: How far to move (0.0 = no move, 1.0 = all the way to target).
                          Values > 1.0 will overshoot. Default: 1.0.

        Notes:
            - Directly modifies tissue.vertices in place.
            - After calling this, you should call reconstruct_cell_vertices() to update
              cell-local vertex arrays.
            - move_fraction allows gradual merging if needed for stability.

        Raises:
            ValueError: If move_fraction is negative.
        """
        if move_fraction < 0:
            raise ValueError(f"move_fraction must be >= 0, got {move_fraction}")

        if self.vertices.shape[0] == 0:
            return

        target_point = np.asarray(target_point, dtype=float).flatten()[:2]

        for idx in vertex_indices:
            idx = int(idx)
            if idx < 0 or idx >= self.vertices.shape[0]:
                continue

            # Compute displacement vector
            displacement = target_point - self.vertices[idx]

            # Move vertex
            self.vertices[idx] += move_fraction * displacement

