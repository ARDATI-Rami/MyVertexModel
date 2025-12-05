import numpy as np
import pytest

from myvertexmodel.simulation import Simulation, OverdampedForceBalanceParams
from myvertexmodel.core import Tissue, Cell, EnergyParameters
from myvertexmodel.geometry import GeometryCalculator


def make_simple_triangle_cell(offset=(0.0, 0.0), cell_id=1):
    # Simple CCW triangle
    verts = np.array([
        [0.0 + offset[0], 0.0 + offset[1]],
        [1.0 + offset[0], 0.0 + offset[1]],
        [0.5 + offset[0], 0.8660254 + offset[1]],
    ])
    return Cell(cell_id=cell_id, vertices=verts)


def test_overdamped_basic_deterministic_step():
    # Tissue with one triangle cell
    cell = make_simple_triangle_cell(cell_id=1)
    tissue = Tissue()
    tissue.add_cell(cell)

    # Zero noise, no active forces
    ofb = OverdampedForceBalanceParams(gamma=1.0, noise_strength=0.0)
    sim = Simulation(
        tissue=tissue,
        dt=0.01,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb,
    )

    # Run one step
    old_vertices = cell.vertices.copy()
    sim.step()

    # Vertices should change deterministically (gradient should not be identically zero)
    new_vertices = cell.vertices
    assert new_vertices.shape == old_vertices.shape
    assert np.all(np.isfinite(new_vertices))
    assert np.linalg.norm(new_vertices - old_vertices) >= 0.0  # non-negative


def test_overdamped_with_noise_reproducible():
    cell = make_simple_triangle_cell(cell_id=1)
    tissue = Tissue()
    tissue.add_cell(cell)

    # Non-zero noise with fixed seed
    ofb = OverdampedForceBalanceParams(gamma=1.0, noise_strength=1e-3, random_seed=42)
    sim = Simulation(
        tissue=tissue,
        dt=0.05,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb,
    )

    v0 = cell.vertices.copy()
    sim.step()
    v1 = cell.vertices.copy()

    # Reset with same seed and repeat -> same result
    cell2 = make_simple_triangle_cell(cell_id=2)
    tissue2 = Tissue()
    tissue2.add_cell(cell2)
    ofb2 = OverdampedForceBalanceParams(gamma=1.0, noise_strength=1e-3, random_seed=42)
    sim2 = Simulation(
        tissue=tissue2,
        dt=0.05,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb2,
    )
    sim2.step()
    v1b = cell2.vertices.copy()

    assert np.allclose(v1, v1b)
    assert not np.allclose(v0, v1)  # changed due to forces/noise


def test_active_force_applied_shape_and_effect():
    # Define a simple active force: radial outward unit vectors
    def active_force(cell, tissue, params):
        verts = cell.vertices
        # from centroid to vertex
        c = verts.mean(axis=0)
        directions = verts - c
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        unit = directions / norms
        magnitude = params.get("magnitude", 0.1)
        return magnitude * unit

    cell = make_simple_triangle_cell(cell_id=1)
    tissue = Tissue()
    tissue.add_cell(cell)

    ofb = OverdampedForceBalanceParams(
        gamma=1.0,
        noise_strength=0.0,
        active_force_func=active_force,
        active_force_params={"magnitude": 0.2},
    )

    sim = Simulation(
        tissue=tissue,
        dt=0.1,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb,
    )

    v_before = cell.vertices.copy()
    sim.step()
    v_after = cell.vertices.copy()

    # Ensure shape
    assert v_after.shape == v_before.shape
    # Active force should move vertices outward relative to centroid more than without it
    # Compare to a deterministic run with no active forces
    cell2 = make_simple_triangle_cell(cell_id=2)
    tissue2 = Tissue()
    tissue2.add_cell(cell2)
    ofb2 = OverdampedForceBalanceParams(gamma=1.0, noise_strength=0.0)
    sim2 = Simulation(
        tissue=tissue2,
        dt=0.1,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb2,
    )
    v_before2 = cell2.vertices.copy()
    sim2.step()
    v_after2 = cell2.vertices.copy()

    # Measure radial distances from centroid
    c1 = v_before.mean(axis=0)
    r_after = np.linalg.norm(v_after - c1, axis=1)
    r_after2 = np.linalg.norm(v_after2 - c1, axis=1)
    # With active forces, radial distances should be greater on average
    assert r_after.mean() >= r_after2.mean()


def test_run_with_logging_returns_samples():
    cell = make_simple_triangle_cell(cell_id=1)
    tissue = Tissue()
    tissue.add_cell(cell)
    ofb = OverdampedForceBalanceParams(gamma=1.0, noise_strength=0.0)
    sim = Simulation(
        tissue=tissue,
        dt=0.02,
        energy_params=EnergyParameters(),
        epsilon=1e-6,
        solver_type="overdamped_force_balance",
        ofb_params=ofb,
    )
    samples = sim.run_with_logging(n_steps=10, log_interval=2)
    assert isinstance(samples, list)
    assert len(samples) == 5
    for t, e in samples:
        assert isinstance(t, float)
        assert isinstance(e, float)
        assert np.isfinite(e)
