"""
Core data structures for vertex model.
"""

import numpy as np
from typing import List, Optional
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
        target_area: Preferred area A0 for cells (can later be per-cell).
    """
    k_area: float = 1.0
    k_perimeter: float = 0.1
    gamma: float = 0.05
    target_area: float = 1.0


def cell_energy(cell: Cell, params: EnergyParameters, geometry: "GeometryCalculator") -> float:
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
        - Uses the cell's current `vertices` local coordinates; in future will derive
          geometry from global `vertex_indices` referencing `Tissue.vertices`.
        - For cells with fewer than 3 vertices, area defaults to 0.0.

    Args:
        cell: Cell instance.
        params: EnergyParameters object with model coefficients.
        geometry: GeometryCalculator (static methods used for area/perimeter).

    Returns:
        float: Energy value (non-negative under current formulation).
    """
    verts = cell.vertices
    # Compute geometric quantities using provided geometry calculator
    area = geometry.calculate_area(verts)
    perimeter = geometry.calculate_perimeter(verts)

    # Energy components
    e_area = 0.5 * params.k_area * (area - params.target_area) ** 2
    e_perim = 0.5 * params.k_perimeter * perimeter ** 2
    e_line = params.gamma * perimeter

    return e_area + e_perim + e_line


def tissue_energy(tissue: Tissue, params: EnergyParameters, geometry: "GeometryCalculator") -> float:
    """Compute (stub) the total mechanical energy of a tissue by summing cell energies.

    Args:
        tissue: Tissue instance containing cells.
        params: Energy parameters used for each cell energy evaluation.
        geometry: GeometryCalculator instance.

    Returns:
        float: Sum of individual cell energies (currently always 0.0).
    """
    return float(sum(cell_energy(cell, params, geometry) for cell in tissue.cells))
