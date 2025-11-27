"""Tissue builder utilities.

Provides pure functions that construct common tissue layouts without performing
any simulation steps. Builders return a populated `Tissue` whose per–cell local
vertices are set; callers may optionally invoke `tissue.build_global_vertices()`
to construct a shared global vertex pool for mechanical coupling.

Design principles:
- Deterministic: repeated calls with same parameters yield identical geometry.
- Separation of concerns: no energy parameters or simulation logic here.
- Lightweight: avoid duplicating logic already present in examples.

Available builders:
- build_grid_tissue(nx, ny, cell_size): axis‑aligned square grid of size nx × ny.
- build_honeycomb_2_3_4_3_2(hex_size): small honeycomb cluster (14 cells) in pattern 2–3–4–3–2.
- build_honeycomb_3_4_5_4_3(hex_size): larger honeycomb cluster (19 cells) in pattern 3–4–5–4–3.

Future extensions could include random Voronoi tissues, lattices with periodic
boundary conditions, or loading from external geometry specs.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple
from .core import Tissue, Cell

__all__ = [
    "build_grid_tissue",
    "build_honeycomb_2_3_4_3_2",
    "build_honeycomb_3_4_5_4_3",
]

def build_grid_tissue(nx: int, ny: int, cell_size: float = 1.0) -> Tissue:
    """Construct an axis-aligned square grid tissue.

    Each cell is an independent polygon with 4 vertices (no shared global pool yet).
    Call `tissue.build_global_vertices()` after construction to merge duplicate vertices
    into a global pool.

    Args:
        nx: Number of cells along x-direction.
        ny: Number of cells along y-direction.
        cell_size: Side length of each square cell.

    Returns:
        Tissue containing nx * ny square cells with consecutive cell IDs starting at 1.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")
    tissue = Tissue()
    cid = 1
    for j in range(ny):  # y index first for visual grouping
        for i in range(nx):
            x0, y0 = i * cell_size, j * cell_size
            verts = np.array([
                [x0, y0],
                [x0 + cell_size, y0],
                [x0 + cell_size, y0 + cell_size],
                [x0, y0 + cell_size],
            ], dtype=float)
            tissue.add_cell(Cell(cell_id=cid, vertices=verts))
            cid += 1
    return tissue


def build_honeycomb_2_3_4_3_2(hex_size: float = 1.0) -> Tissue:
    """Construct a 14-cell honeycomb tissue with 5 staggered rows (2–3–4–3–2 pattern).

    Hexagons are pointy-top oriented: a vertex points toward +y direction.

    Pattern row layout (top to bottom):
        Row 0: 2 cells (IDs 1–2)
        Row 1: 3 cells (IDs 3–5)
        Row 2: 4 cells (IDs 6–9)   (central band)
        Row 3: 3 cells (IDs 10–12)
        Row 4: 2 cells (IDs 13–14)

    Args:
        hex_size: Distance from center to each vertex (circumradius) of hexagon.

    Returns:
        Tissue populated with 14 regular hexagonal cells.
    """
    if hex_size <= 0:
        raise ValueError("hex_size must be positive")

    def hexagon_vertices(cx: float, cy: float, size: float) -> np.ndarray:
        # Angles for pointy-top orientation: start at 30°, then every 60°
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        x = cx + size * np.cos(angles)
        y = cy + size * np.sin(angles)
        return np.column_stack([x, y]).astype(float)

    dx = hex_size * np.sqrt(3.0)   # horizontal spacing between adjacent hex centers
    dy = 1.5 * hex_size            # vertical spacing between rows

    centers = {
        # Row 0 (top, y = 2*dy)
        1: (-dx/2,  2*dy),
        2: ( dx/2,  2*dy),
        # Row 1 (y = dy)
        3: (-dx,    dy),
        4: (0.0,    dy),
        5: ( dx,    dy),
        # Row 2 (center, y = 0)
        6: (-1.5*dx, 0.0),
        7: (-dx/2,   0.0),
        8: ( dx/2,   0.0),
        9: (1.5*dx,  0.0),
        # Row 3 (y = -dy)
        10: (-dx,   -dy),
        11: (0.0,   -dy),
        12: ( dx,   -dy),
        # Row 4 (bottom, y = -2*dy)
        13: (-dx/2, -2*dy),
        14: ( dx/2, -2*dy),
    }

    tissue = Tissue()
    for cid in sorted(centers.keys()):
        cx, cy = centers[cid]
        verts = hexagon_vertices(cx, cy, hex_size)
        tissue.add_cell(Cell(cell_id=cid, vertices=verts))
    return tissue


def build_honeycomb_3_4_5_4_3(hex_size: float = 1.0) -> Tissue:
    """Construct a 19-cell honeycomb tissue with 5 staggered rows (3–4–5–4–3 pattern).

    This represents two complete rings around a central hexagon:
    - 1 center hexagon
    - 6 hexagons in first ring
    - 12 hexagons in second (outer) ring
    Total: 19 hexagons

    Hexagons are pointy-top oriented: a vertex points toward +y direction.

    Pattern row layout (top to bottom):
        Row 0: 3 cells (IDs 1–3)
        Row 1: 4 cells (IDs 4–7)
        Row 2: 5 cells (IDs 8–12)   (central band, includes center cell 10)
        Row 3: 4 cells (IDs 13–16)
        Row 4: 3 cells (IDs 17–19)

    Args:
        hex_size: Distance from center to each vertex (circumradius) of hexagon.

    Returns:
        Tissue populated with 19 regular hexagonal cells.
    """
    if hex_size <= 0:
        raise ValueError("hex_size must be positive")

    def hexagon_vertices(cx: float, cy: float, size: float) -> np.ndarray:
        # Angles for pointy-top orientation: start at 30°, then every 60°
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
        x = cx + size * np.cos(angles)
        y = cy + size * np.sin(angles)
        return np.column_stack([x, y]).astype(float)

    dx = hex_size * np.sqrt(3.0)   # horizontal spacing between adjacent hex centers
    dy = 1.5 * hex_size            # vertical spacing between rows

    centers = {
        # Row 0 (top, y = 2*dy): 3 cells
        1: (-dx,     2*dy),
        2: (0.0,     2*dy),
        3: (dx,      2*dy),

        # Row 1 (y = dy): 4 cells
        4: (-1.5*dx, dy),
        5: (-dx/2,   dy),
        6: (dx/2,    dy),
        7: (1.5*dx,  dy),

        # Row 2 (center, y = 0): 5 cells
        8:  (-2*dx,  0.0),
        9:  (-dx,    0.0),
        10: (0.0,    0.0),   # CENTER hexagon
        11: (dx,     0.0),
        12: (2*dx,   0.0),

        # Row 3 (y = -dy): 4 cells
        13: (-1.5*dx, -dy),
        14: (-dx/2,   -dy),
        15: (dx/2,    -dy),
        16: (1.5*dx,  -dy),

        # Row 4 (bottom, y = -2*dy): 3 cells
        17: (-dx,    -2*dy),
        18: (0.0,    -2*dy),
        19: (dx,     -2*dy),
    }

    tissue = Tissue()
    for cid in sorted(centers.keys()):
        cx, cy = centers[cid]
        verts = hexagon_vertices(cx, cy, hex_size)
        tissue.add_cell(Cell(cell_id=cid, vertices=verts))
    return tissue

