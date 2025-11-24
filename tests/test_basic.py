"""
Basic tests for MyVertexModel.
"""

import pytest
import numpy as np
from myvertexmodel import Cell, Tissue, GeometryCalculator, Simulation, EnergyParameters, cell_energy, tissue_energy
from myvertexmodel.io import save_state, load_state
from myvertexmodel.__main__ import main as cli_main


def test_cell_creation():
    """Test creating a cell."""
    cell = Cell(cell_id=1)
    assert cell.id == 1
    assert len(cell.vertices) == 0
    assert cell.vertices.shape == (0, 2)
    assert cell.vertices.dtype == float


def test_cell_with_vertices():
    """Test creating a cell with vertices."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=vertices)
    assert cell.id == 1
    assert len(cell.vertices) == 4
    assert cell.vertices.shape == (4, 2)


def test_cell_vertices_validation():
    """Test that Cell validates vertices array shape."""
    # Should raise ValueError for 1D array
    with pytest.raises(ValueError, match="must have shape"):
        Cell(cell_id=1, vertices=np.array([1, 2, 3]))

    # Should raise ValueError for wrong second dimension
    with pytest.raises(ValueError, match="must have shape"):
        Cell(cell_id=1, vertices=np.array([[1, 2, 3], [4, 5, 6]]))

    # Should raise ValueError for 3D array
    with pytest.raises(ValueError, match="must be a 2D array"):
        Cell(cell_id=1, vertices=np.array([[[1, 2]]]))

    # Empty list should work
    cell = Cell(cell_id=1, vertices=[])
    assert cell.vertices.shape == (0, 2)


def test_tissue_creation():
    """Test creating a tissue."""
    tissue = Tissue()
    assert len(tissue.cells) == 0


def test_tissue_add_cell():
    """Test adding a cell to tissue."""
    tissue = Tissue()
    cell = Cell(cell_id=1)
    tissue.add_cell(cell)
    assert len(tissue.cells) == 1


def test_geometry_area():
    """Test area calculation."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    area = GeometryCalculator.calculate_area(vertices)
    assert area == pytest.approx(1.0)


def test_geometry_perimeter():
    """Test perimeter calculation."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    perimeter = GeometryCalculator.calculate_perimeter(vertices)
    assert perimeter == pytest.approx(4.0)


def test_geometry_centroid():
    """Test centroid calculation."""
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    centroid = GeometryCalculator.calculate_centroid(vertices)
    assert centroid[0] == pytest.approx(1.0)
    assert centroid[1] == pytest.approx(1.0)


def test_simulation_creation():
    """Test creating a simulation."""
    sim = Simulation()
    assert sim.time == 0.0
    assert sim.dt == 0.01


def test_simulation_step():
    """Test simulation step."""
    sim = Simulation()
    sim.step()
    assert sim.time == pytest.approx(0.01)


def test_simulation_run():
    """Test running simulation."""
    sim = Simulation(dt=0.1)
    sim.run(n_steps=10)
    assert sim.time == pytest.approx(1.0)


# ---------------- New Energy API stub tests ---------------- #

def test_cell_energy_stub():
    """Test cell_energy with implemented basic energy.
    Ensures returned energy is finite and non-negative (not asserting exact value to avoid fragility)."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=vertices)
    params = EnergyParameters()  # default parameters
    geometry = GeometryCalculator()
    e = cell_energy(cell, params, geometry)
    assert np.isfinite(e)
    assert e >= 0.0


def test_tissue_energy_stub():
    """Test tissue_energy with implemented basic energy.
    Ensures sum of cell energies is finite and non-negative."""
    tissue = Tissue()
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=vertices)
    tissue.add_cell(cell)
    params = EnergyParameters()
    geometry = GeometryCalculator()
    e_total = tissue_energy(tissue, params, geometry)
    assert np.isfinite(e_total)
    assert e_total >= 0.0


def test_simulation_total_energy():
    """Test Simulation.total_energy returns finite non-negative value for a simple tissue."""
    tissue = Tissue()
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    tissue.add_cell(Cell(cell_id=1, vertices=square))
    sim = Simulation(tissue=tissue)
    e = sim.total_energy()
    assert np.isfinite(e)
    assert e >= 0.0


def test_simulation_energy_descent():
    """Run several simulation steps and assert energy does not increase beyond tolerance.
    Uses two adjacent square cells to create a trivial multi-cell tissue.
    """
    tissue = Tissue()
    # Two squares offset in x
    square1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    square2 = np.array([[1, 0], [2, 0], [2, 1], [1, 1]])
    tissue.add_cell(Cell(cell_id=1, vertices=square1))
    tissue.add_cell(Cell(cell_id=2, vertices=square2))
    sim = Simulation(tissue=tissue, dt=0.001)  # small dt for stability
    initial_energy = sim.total_energy()
    for _ in range(10):
        sim.step()
    final_energy = sim.total_energy()
    # Allow tiny numerical fluctuation tolerance
    assert final_energy <= initial_energy + 1e-6


def test_cli_main_smoke():
    """Smoke test for CLI entry point (no plotting)."""
    rc = cli_main(["--steps", "2", "--grid-size", "1", "--no-energy-print"])  # should return 0
    assert rc == 0


def test_cli_main_output_file(tmp_path):
    """Ensure CLI saves plot when --output is provided."""
    out_file = tmp_path / "test_plot.png"
    rc = cli_main(["--steps", "2", "--grid-size", "1", "--plot", "--output", str(out_file), "--no-energy-print"])
    assert rc == 0
    assert out_file.exists()


if __name__ == "__main__":
    pytest.main([__file__])
