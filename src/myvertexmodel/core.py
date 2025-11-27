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

    def __init__(self, cell_id: int, vertices: Optional[np.ndarray] = None, vertex_indices: Optional[np.ndarray] = None):
        """
        Initialize a cell.
        
        Args:
            cell_id: Unique identifier for the cell.
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
