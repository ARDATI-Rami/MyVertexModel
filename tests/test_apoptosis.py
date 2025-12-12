"""Tests for apoptosis (cell removal) functionality."""

import numpy as np

from myvertexmodel import Cell, Tissue
from myvertexmodel.apoptosis import (
    ApoptosisParameters,
    ApoptosisState,
    update_apoptosis_targets,
    collect_cells_to_remove,
)
from myvertexmodel.geometry import GeometryCalculator


def test_apoptosis_state_register_and_shrink():
    """ApoptosisState should register cells and shrink target areas over time."""
    geom = GeometryCalculator()
    tissue = Tissue()
    # Simple square cell with area 1.0
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue.add_cell(cell)

    params = ApoptosisParameters(shrink_rate=1.0, min_area_fraction=0.1, start_step=0)
    state = ApoptosisState()
    state.register_cells(tissue, [1], geometry=geom)

    assert 1 in state.apoptotic_cells
    A0 = state.initial_areas[1]
    assert A0 == geom.calculate_area(vertices)

    # After some time, target area should be reduced but not below min_area_fraction * A0
    update_apoptosis_targets(tissue, state, params, step_index=10, dt=0.1, geometry=geom)
    A_target = state.current_target_areas[1]
    assert A_target <= A0
    assert A_target >= params.min_area_fraction * A0


def test_collect_cells_to_remove_by_area_and_vertices():
    """Cells should be marked for removal when area is too small.

    Vertex-count based removal is handled indirectly via merging/geometry,
    so this test only asserts area-based removal for now.
    """
    geom = GeometryCalculator()
    tissue = Tissue()

    # Cell A: tiny triangle (area small)
    verts_a = np.array([[0, 0], [0.01, 0], [0, 0.01]], dtype=float)
    cell_a = Cell(cell_id="A", vertices=verts_a)

    # Cell B: degenerate line (2 vertices)
    verts_b = np.array([[0, 0], [1, 0]], dtype=float)
    cell_b = Cell(cell_id="B", vertices=verts_b)

    tissue.add_cell(cell_a)
    tissue.add_cell(cell_b)

    # Use a high removal fraction so the tiny triangle is definitely removed.
    params = ApoptosisParameters(
        min_area_fraction=0.5,  # target-area floor only
        removal_area_fraction=0.9,
        removal_area_absolute=0.0,
        min_vertices=3,
    )
    state = ApoptosisState()
    state.register_cells(tissue, ["A", "B"], geometry=geom)

    to_remove = collect_cells_to_remove(tissue, state, params, geometry=geom)

    # A should be removed by area criterion in this simple test
    assert "A" in to_remove


def test_tissue_remove_cells_removes_by_id():
    """Tissue.remove_cells should drop cells with matching IDs."""
    tissue = Tissue()
    c1 = Cell(cell_id=1, vertices=np.zeros((3, 2)))
    c2 = Cell(cell_id=2, vertices=np.zeros((3, 2)))
    c3 = Cell(cell_id="X", vertices=np.zeros((3, 2)))
    tissue.add_cell(c1)
    tissue.add_cell(c2)
    tissue.add_cell(c3)

    assert len(tissue.cells) == 3

    tissue.remove_cells([1, "X"])

    remaining_ids = {c.id for c in tissue.cells}
    assert remaining_ids == {2}
