"""Integration-style tests for apoptosis-driven mechanical shrink.

These tests ensure that when an apoptotic cell's target area is reduced over time,
its *geometric* area decreases under global-vertex gradient descent.

This mirrors the update loop used in examples/simulate_cell_growth.py and
examples/simulate_cell_apoptosis.py.
"""

from __future__ import annotations

import numpy as np

from myvertexmodel import EnergyParameters, GeometryCalculator
from myvertexmodel.apoptosis import ApoptosisParameters, ApoptosisState, build_apoptosis_target_area_mapping, update_apoptosis_targets
from myvertexmodel.builders import build_honeycomb_2_3_4_3_2
from myvertexmodel.energy import tissue_energy


def _compute_global_gradient(tissue, energy_params, geometry: GeometryCalculator, epsilon: float = 1e-6) -> np.ndarray:
    """Finite-difference gradient w.r.t. global vertices."""
    if tissue.vertices.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    grad = np.zeros_like(tissue.vertices)

    for i in range(tissue.vertices.shape[0]):
        # X
        x0 = tissue.vertices[i, 0]
        tissue.vertices[i, 0] = x0 + epsilon
        tissue.reconstruct_cell_vertices()
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = x0 - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = x0
        grad[i, 0] = (e_plus - e_minus) / (2 * epsilon)

        # Y
        y0 = tissue.vertices[i, 1]
        tissue.vertices[i, 1] = y0 + epsilon
        tissue.reconstruct_cell_vertices()
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = y0 - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = y0
        grad[i, 1] = (e_plus - e_minus) / (2 * epsilon)

    tissue.reconstruct_cell_vertices()
    return grad


def test_apoptotic_cell_geometric_area_decreases_under_global_descent():
    geom = GeometryCalculator()
    tissue = build_honeycomb_2_3_4_3_2()

    apoptotic_id = "7"
    cell = next(c for c in tissue.cells if str(c.id) == apoptotic_id)

    # Start near equilibrium using per-cell geometric areas
    target_areas = {c.id: geom.calculate_area(c.vertices) for c in tissue.cells}
    energy_params = EnergyParameters(k_area=1.0, k_perimeter=0.1, gamma=0.05, target_area=target_areas)

    apoptosis_params = ApoptosisParameters(
        shrink_rate=1.0,
        min_area_fraction=0.05,
        removal_area_fraction=0.0,
        removal_area_absolute=0.0,
        start_step=0,
    )
    state = ApoptosisState()
    state.register_cells(tissue, [apoptotic_id], geometry=geom)

    A0 = geom.calculate_area(cell.vertices)

    dt = 0.01
    damping = 1.0

    # Run a few steps
    for step in range(1, 21):
        update_apoptosis_targets(tissue, state, apoptosis_params, step_index=step, dt=dt, geometry=geom)
        for cid, A_target in build_apoptosis_target_area_mapping(state).items():
            target_areas[cid] = A_target
        energy_params.target_area = target_areas

        grad = _compute_global_gradient(tissue, energy_params, geom, epsilon=1e-6)
        tissue.vertices = tissue.vertices - dt * damping * grad
        tissue.reconstruct_cell_vertices()

    A1 = geom.calculate_area(cell.vertices)

    # The cell should have shrunk geometrically
    assert A1 < A0

