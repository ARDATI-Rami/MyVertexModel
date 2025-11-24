"""
Core data structures for vertex model.
"""

import numpy as np
from typing import List, Optional


class Cell:
    """Represents a single cell in the vertex model."""
    
    def __init__(self, cell_id: int, vertices: Optional[np.ndarray] = None):
        """
        Initialize a cell.
        
        Args:
            cell_id: Unique identifier for the cell
            vertices: Array of vertex coordinates (N x 2)
        """
        self.id = cell_id
        self.vertices = vertices if vertices is not None else np.array([])
        
    def __repr__(self):
        return f"Cell(id={self.id}, n_vertices={len(self.vertices)})"


class Tissue:
    """Represents a collection of cells forming a tissue."""
    
    def __init__(self):
        """Initialize an empty tissue."""
        self.cells: List[Cell] = []
        
    def add_cell(self, cell: Cell):
        """Add a cell to the tissue."""
        self.cells.append(cell)
        
    def __repr__(self):
        return f"Tissue(n_cells={len(self.cells)})"

