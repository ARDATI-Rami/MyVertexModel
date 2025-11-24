"""Plotting utilities for the vertex model.

Provides a convenience function to visualize a Tissue.

Example
-------
>>> import numpy as np
>>> from myvertexmodel import Tissue, Cell, plot_tissue
>>> import matplotlib.pyplot as plt
>>> tissue = Tissue()
>>> tissue.add_cell(Cell(cell_id=1, vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]])))
>>> ax = plot_tissue(tissue)  # doctest: +SKIP (visual)
>>> plt.show()  # doctest: +SKIP
"""

from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon as MplPolygon
from .core import Tissue, Cell


def plot_tissue(
    tissue: Tissue,
    ax: Optional[Axes] = None,
    show_vertices: bool = True,
    fill: bool = True,
    vertex_color: str = "k",
    edge_color: str = "#1f77b4",
    face_color: str = "#1f77b4",
    alpha: float = 0.2,
    linewidth: float = 1.0,
    colors: Optional[Sequence[str]] = None,
) -> Axes:
    """Plot all cells in a tissue as polygons.

    Each cell's local vertex coordinates are used directly. In a future global
    vertex representation, reconstruction from shared indices will be added.

    Args:
        tissue: Tissue instance to plot.
        ax: Optional matplotlib Axes; creates a new one if None.
        show_vertices: If True, draw vertex markers.
        fill: If True, lightly fill polygons.
        vertex_color: Color for vertex markers.
        edge_color: Default edge color if per-cell colors not provided.
        face_color: Base face color (overridden if colors provided).
        alpha: Face alpha transparency.
        linewidth: Polygon edge width.
        colors: Optional list of face colors per cell.

    Returns:
        Axes: The matplotlib Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if colors is not None and len(colors) != len(tissue.cells):
        raise ValueError("Length of colors must match number of cells if provided.")

    for idx, cell in enumerate(tissue.cells):
        verts = cell.vertices
        if verts.shape[0] < 3:
            # Skip degenerate polygons
            continue
        fc = colors[idx] if colors is not None else (face_color if fill else "none")
        patch = MplPolygon(verts, closed=True, facecolor=fc, edgecolor=edge_color, alpha=alpha if fill else 1.0, linewidth=linewidth)
        ax.add_patch(patch)
        if show_vertices:
            ax.scatter(verts[:, 0], verts[:, 1], s=15, c=vertex_color, zorder=3)

    # Auto-scale view
    all_vertices = np.vstack([c.vertices for c in tissue.cells if c.vertices.shape[0] > 0]) if tissue.cells else np.empty((0, 2))
    if all_vertices.size > 0:
        xmin, ymin = all_vertices.min(axis=0)
        xmax, ymax = all_vertices.max(axis=0)
        dx = max(1e-3, (xmax - xmin) * 0.05)
        dy = max(1e-3, (ymax - ymin) * 0.05)
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Tissue plot ({} cells)".format(len(tissue.cells)))
    return ax

__all__ = ["plot_tissue"]

