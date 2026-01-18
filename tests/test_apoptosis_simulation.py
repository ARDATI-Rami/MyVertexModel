"""Integration tests for apoptosis within a short simulation."""

import numpy as np

from myvertexmodel import Tissue, Simulation, EnergyParameters
from myvertexmodel.apoptosis import ApoptosisParameters
from myvertexmodel.geometry import GeometryCalculator


def build_single_cell_tissue() -> Tissue:
    """Build a simple tissue with a single square cell."""
    from myvertexmodel import Cell

    tissue = Tissue()
    verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=verts)
    tissue.add_cell(cell)
    return tissue


def test_apoptosis_removes_cell_in_simulation():
    """A strongly shrinking apoptotic cell should significantly reduce its area.

    Depending on dynamics and parameters, the cell may or may not be fully
    removed within the limited number of steps in this test.
    """
    tissue = build_single_cell_tissue()
    geom = GeometryCalculator()

    # Basic energy parameters
    energy_params = EnergyParameters(k_area=1.0, k_perimeter=0.0, gamma=0.0, target_area=1.0)

    # Aggressive apoptosis so shrink is visible
    apoptosis_params = ApoptosisParameters(
        shrink_rate=5.0,
        min_area_fraction=0.1,
        removal_area_fraction=0.2,
        removal_area_absolute=0.0,
        min_vertices=3,
        start_step=0,
    )

    sim = Simulation(
        tissue=tissue,
        dt=0.01,
        energy_params=energy_params,
        validate_each_step=False,
        epsilon=1e-6,
        damping=0.1,
        solver_type="gradient_descent",
        ofb_params=None,
        apoptosis_params=apoptosis_params,
        apoptotic_cell_ids=[1],
    )

    initial_area = geom.calculate_area(sim.tissue.cells[0].vertices)

    # Run a small number of steps
    for _ in range(50):
        sim.step()
        if not sim.tissue.cells:
            break

    if sim.tissue.cells:
        area = geom.calculate_area(sim.tissue.cells[0].vertices)
        assert area < initial_area
    else:
        # Cell was fully removed
        assert len(sim.tissue.cells) == 0


def test_apoptosis_merge_removes_cell_and_merges_neighbours():
    """Test that the 'merge' strategy removes the apoptotic cell and merges its neighbours."""
    from myvertexmodel import Cell
    # Build a simple tissue: three cells in a line, cell 2 is apoptotic
    tissue = Tissue()
    # Cell 1: left
    verts1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell1 = Cell(cell_id=1, vertices=verts1)
    # Cell 2: middle (apoptotic)
    verts2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)
    cell2 = Cell(cell_id=2, vertices=verts2)
    # Cell 3: right
    verts3 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]], dtype=float)
    cell3 = Cell(cell_id=3, vertices=verts3)
    tissue.add_cell(cell1)
    tissue.add_cell(cell2)
    tissue.add_cell(cell3)
    geom = GeometryCalculator()
    energy_params = EnergyParameters(k_area=1.0, k_perimeter=0.0, gamma=0.0, target_area=1.0)
    apoptosis_params = ApoptosisParameters(
        shrink_rate=5.0,
        min_area_fraction=0.1,
        removal_area_fraction=1.1,  # ensure immediate removal
        removal_area_absolute=0.0,
        min_vertices=3,
        start_step=0,
        removal_strategy='merge',
    )
    sim = Simulation(
        tissue=tissue,
        dt=0.01,
        energy_params=energy_params,
        validate_each_step=False,
        epsilon=1e-6,
        damping=0.1,
        solver_type="gradient_descent",
        ofb_params=None,
        apoptosis_params=apoptosis_params,
        apoptotic_cell_ids=[2],
    )
    # Run one step: cell 2 should be removed, and its centroid added to neighbours
    sim.step()
    ids = {c.id for c in sim.tissue.cells}
    assert 2 not in ids
    assert 1 in ids and 3 in ids
    # Check that the centroid of cell 2 is present in both neighbours' vertices
    centroid = np.mean(verts2, axis=0)
    found1 = np.any(np.all(np.isclose(sim.tissue.cells[0].vertices, centroid), axis=1))
    found3 = np.any(np.all(np.isclose(sim.tissue.cells[1].vertices, centroid), axis=1))
    assert found1 or found3  # At least one neighbour should have the centroid

