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
    show_vertices: bool = False,
    fill: bool = True,
    vertex_color: str = "k",
    edge_color: str = "#000000",
    face_color: str = "none",
    alpha: float = 1,
    linewidth: float = 2.0,
    colors: Optional[Sequence[str]] = None,
    show_vertex_ids: bool = False,
    show_cell_ids: bool = False,
    vertex_id_fontsize: int = 8,
    cell_id_fontsize: int = 10,
    show_neighbor_counts: bool = False,
    neighbor_fontsize: Optional[int] = None,
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
        show_vertex_ids: If True, label vertices with their global vertex indices.
        show_cell_ids: If True, label cells with their cell IDs.
        vertex_id_fontsize: Font size for vertex ID labels.
        cell_id_fontsize: Font size for cell ID labels.
        show_neighbor_counts: If True, label cells with their number of neighbors.
        neighbor_fontsize: Optional font size for neighbor count labels. If None,
            defaults to ``cell_id_fontsize``.

    Returns:
        Axes: The matplotlib Axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if colors is not None and len(colors) != len(tissue.cells):
        raise ValueError("Length of colors must match number of cells if provided.")

    neighbor_counts = None
    if show_neighbor_counts:
        # Compute neighbor counts once for the whole tissue
        neighbor_counts = tissue.cell_neighbor_counts()
        if neighbor_fontsize is None:
            neighbor_fontsize = cell_id_fontsize

    for idx, cell in enumerate(tissue.cells):
        verts = cell.vertices
        if verts.shape[0] < 3:
            # Skip degenerate polygons
            continue

        fc = colors[idx] if colors is not None else (face_color if fill else "none")
        patch = MplPolygon(
            verts,
            closed=True,
            facecolor=fc,
            edgecolor=edge_color,
            alpha=alpha if fill else 1.0,
            linewidth=linewidth,
        )
        ax.add_patch(patch)

        if show_vertices:
            # draw vertices as hollow markers (no filled dot)
            ax.scatter(
                verts[:, 0],
                verts[:, 1],
                s=30,
                facecolors="none",
                edgecolors=vertex_color,
                linewidths=0.8,
                zorder=3,
            )

        # Show cell ID at centroid
        if show_cell_ids:
            centroid = np.mean(verts, axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                str(cell.id),
                fontsize=cell_id_fontsize,
                ha="center",
                va="center",
                zorder=4,
            )

        # Show neighbor count at centroid (possibly alongside ID)
        if show_neighbor_counts and neighbor_counts is not None:
            centroid = np.mean(verts, axis=0)
            count = neighbor_counts.get(cell.id, 0)
            # If also showing cell IDs, place neighbor count slightly below
            if show_cell_ids:
                y_min, y_max = ax.get_ylim()
                dy = -0.05 * (y_max - y_min)
            else:
                dy = 0.0
            ax.text(
                centroid[0],
                centroid[1] + dy,
                str(count),
                fontsize=neighbor_fontsize if neighbor_fontsize is not None else cell_id_fontsize,
                ha="center",
                va="center",
                color="black",
                zorder=5,
            )

    # Show vertex IDs from global vertex pool
    if show_vertex_ids and hasattr(tissue, "vertices") and tissue.vertices.shape[0] > 0:
        # Plot each global vertex with its ID
        for vi, vertex in enumerate(tissue.vertices):
            ax.text(
                vertex[0],
                vertex[1],
                str(vi),
                fontsize=vertex_id_fontsize,
                ha="right",
                va="bottom",
                color="red",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="red"),
                zorder=5,
            )

    # Auto-scale view
    all_vertices = (
        np.vstack([c.vertices for c in tissue.cells if c.vertices.shape[0] > 0])
        if tissue.cells
        else np.empty((0, 2))
    )
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
