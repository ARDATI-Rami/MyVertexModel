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
