"""
Basic tests for MyVertexModel.
"""

import pytest
import numpy as np
from myvertexmodel import Cell, Tissue, GeometryCalculator, Simulation
from myvertexmodel.io import save_state, load_state


def test_cell_creation():
    """Test creating a cell."""
    cell = Cell(cell_id=1)
    assert cell.id == 1
    assert len(cell.vertices) == 0


def test_cell_with_vertices():
    """Test creating a cell with vertices."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=vertices)
    assert cell.id == 1
    assert len(cell.vertices) == 4


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


if __name__ == "__main__":
    pytest.main([__file__])

