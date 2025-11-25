"""
Basic tests for MyVertexModel.
"""

import pytest
import numpy as np
from myvertexmodel import Cell, Tissue, GeometryCalculator, Simulation, EnergyParameters, cell_energy, tissue_energy
from myvertexmodel.io import save_state, load_state
from myvertexmodel.__main__ import main as cli_main
from myvertexmodel.geometry import is_valid_polygon, polygon_orientation, ensure_ccw


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


def test_cli_main_with_parameters(tmp_path):
    """CLI test using non-default energy and simulation parameters to ensure wiring works."""
    out_file = tmp_path / "param_plot.png"
    rc = cli_main([
        "--steps", "3",
        "--grid-size", "2",
        "--dt", "0.02",
        "--k-area", "2.5",
        "--k-perimeter", "0.25",
        "--gamma", "0.10",
        "--target-area", "0.8",
        "--epsilon", "1e-5",
        "--damping", "0.5",
        "--plot",
        "--output", str(out_file),
        "--no-energy-print",
    ])
    assert rc == 0
    assert out_file.exists()


def test_valid_square():
    """Test validation of a simple square."""
    # CCW square
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert is_valid_polygon(square)
    assert polygon_orientation(square) > 0  # CCW is positive

    # Ensure CCW should return same order
    ccw_square = ensure_ccw(square)
    assert np.allclose(ccw_square, square)


def test_degenerate_polygon():
    """Test degenerate cases with collinear or repeated vertices."""
    # Collinear points
    collinear = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    assert not is_valid_polygon(collinear)

    # Repeated vertices
    repeated = np.array([[0, 0], [1, 0], [1, 0]], dtype=float)
    assert not is_valid_polygon(repeated)

    # Single point
    single = np.array([[0, 0]], dtype=float)
    assert not is_valid_polygon(single)

    # Two points
    two_points = np.array([[0, 0], [1, 1]], dtype=float)
    assert not is_valid_polygon(two_points)


def test_orientation_detection():
    """Test orientation detection for CW vs CCW polygons."""
    # CCW square
    ccw_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert polygon_orientation(ccw_square) > 0

    # CW square (reversed)
    cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    assert polygon_orientation(cw_square) < 0

    # CCW triangle
    ccw_triangle = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
    assert polygon_orientation(ccw_triangle) > 0

    # CW triangle
    cw_triangle = np.array([[0, 0], [0.5, 1], [1, 0]], dtype=float)
    assert polygon_orientation(cw_triangle) < 0


def test_ensure_ccw_correction():
    """Test that ensure_ccw correctly reorders CW polygons."""
    # Start with CW square
    cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    assert polygon_orientation(cw_square) < 0

    # Convert to CCW
    ccw_square = ensure_ccw(cw_square)
    assert polygon_orientation(ccw_square) > 0

    # Should be reversed
    assert np.allclose(ccw_square, cw_square[::-1])

    # Original should be unchanged
    assert polygon_orientation(cw_square) < 0

    # Already CCW polygon should remain unchanged
    ccw_input = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    ccw_output = ensure_ccw(ccw_input)
    assert np.allclose(ccw_output, ccw_input)

    # Should return a copy, not the same array
    assert ccw_output is not ccw_input


def test_tissue_validate_invalid_cell():
    """Test that tissue.validate() raises for invalid cells."""
    tissue = Tissue()

    # Cell with only 2 vertices (invalid polygon)
    invalid_cell = Cell(cell_id=1, vertices=np.array([[0, 0], [1, 1]]))
    tissue.add_cell(invalid_cell)

    with pytest.raises(ValueError, match="Cell 1.*invalid"):
        tissue.validate()


def test_tissue_validate_collinear_cell():
    """Test that tissue.validate() raises for collinear vertices."""
    tissue = Tissue()

    # Cell with collinear vertices (zero area)
    collinear_cell = Cell(cell_id=2, vertices=np.array([[0, 0], [1, 0], [2, 0]]))
    tissue.add_cell(collinear_cell)

    with pytest.raises(ValueError, match="Cell 2.*invalid"):
        tissue.validate()


def test_tissue_validate_valid_tissue():
    """Test that tissue.validate() passes for valid tissue."""
    tissue = Tissue()

    # Valid square cell
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=square)
    tissue.add_cell(cell)

    # Should not raise
    tissue.validate()


def test_tissue_validate_empty_cell():
    """Test that tissue.validate() handles empty cells."""
    tissue = Tissue()

    # Empty cell should be valid
    empty_cell = Cell(cell_id=1)
    tissue.add_cell(empty_cell)

    # Should not raise
    tissue.validate()


def test_simulation_validate_each_step():
    """Test that Simulation with validate_each_step=True runs validation."""
    tissue = Tissue()

    # Valid square cell
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=square)
    tissue.add_cell(cell)

    # Create simulation with validation enabled
    sim = Simulation(tissue=tissue, dt=0.01, validate_each_step=True)

    # Should run without raising
    sim.step()
    assert sim.time == pytest.approx(0.01)


def test_simulation_validate_each_step_catches_invalid():
    """Test that validation during step catches invalid tissue states."""
    tissue = Tissue()

    # Start with valid cell
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    cell = Cell(cell_id=1, vertices=square)
    tissue.add_cell(cell)

    sim = Simulation(tissue=tissue, dt=0.01, validate_each_step=True)

    # Manually corrupt the cell to make it invalid
    sim.tissue.cells[0].vertices = np.array([[0, 0], [1, 1]])  # only 2 vertices

    # Step should raise due to validation
    with pytest.raises(ValueError, match="Cell 1.*invalid"):
        sim.step()


def test_build_global_vertices_simple():
    """Test build_global_vertices with a simple 2x1 grid."""
    tissue = Tissue()

    # Create a 2x1 grid of square cells (2 cells side by side)
    # Cell 1: [0,0] -> [1,0] -> [1,1] -> [0,1]
    # Cell 2: [1,0] -> [2,0] -> [2,1] -> [1,1]
    # They share vertices [1,0] and [1,1]

    cell1_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell2_verts = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)

    cell1 = Cell(cell_id=1, vertices=cell1_verts)
    cell2 = Cell(cell_id=2, vertices=cell2_verts)

    tissue.add_cell(cell1)
    tissue.add_cell(cell2)

    # Before build_global_vertices
    total_local_vertices = cell1.vertices.shape[0] + cell2.vertices.shape[0]
    assert total_local_vertices == 8  # 4 + 4

    # Build global vertex pool
    tissue.build_global_vertices()

    # Check that vertices are shared (should have 6 unique vertices, not 8)
    # Unique vertices: [0,0], [1,0], [2,0], [2,1], [1,1], [0,1]
    assert tissue.vertices.shape[0] == 6
    assert tissue.vertices.shape[1] == 2

    # Check that each cell has vertex_indices
    assert cell1.vertex_indices.shape[0] == 4
    assert cell2.vertex_indices.shape[0] == 4

    # Check that original cell.vertices are unchanged
    assert np.allclose(cell1.vertices, cell1_verts)
    assert np.allclose(cell2.vertices, cell2_verts)


def test_reconstruct_cell_vertices():
    """Test that reconstruct_cell_vertices reproduces original coordinates."""
    tissue = Tissue()

    # Create a 2x1 grid
    cell1_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell2_verts = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)

    cell1 = Cell(cell_id=1, vertices=cell1_verts.copy())
    cell2 = Cell(cell_id=2, vertices=cell2_verts.copy())

    tissue.add_cell(cell1)
    tissue.add_cell(cell2)

    # Save original vertices
    orig_cell1 = cell1.vertices.copy()
    orig_cell2 = cell2.vertices.copy()

    # Build global pool
    tissue.build_global_vertices()

    # Reconstruct cell vertices from global pool
    tissue.reconstruct_cell_vertices()

    # Should reproduce original coordinates (within tolerance)
    assert np.allclose(cell1.vertices, orig_cell1, atol=1e-8)
    assert np.allclose(cell2.vertices, orig_cell2, atol=1e-8)


def test_build_global_vertices_with_tolerance():
    """Test that build_global_vertices merges nearly-identical vertices."""
    tissue = Tissue()

    # Create two cells with slightly different but close vertices
    # (simulating numerical noise)
    cell1_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell2_verts = np.array([[1.0 + 1e-10, 0], [2, 0], [2, 1], [1.0 + 1e-10, 1]], dtype=float)

    cell1 = Cell(cell_id=1, vertices=cell1_verts)
    cell2 = Cell(cell_id=2, vertices=cell2_verts)

    tissue.add_cell(cell1)
    tissue.add_cell(cell2)

    # Build with default tolerance (1e-8)
    tissue.build_global_vertices(tol=1e-8)

    # Should merge the nearly-identical vertices [1,0] and [1.0+1e-10, 0]
    # and [1,1] and [1.0+1e-10, 1]
    assert tissue.vertices.shape[0] == 6  # 6 unique vertices

    # Reconstruct should give approximately correct coordinates
    tissue.reconstruct_cell_vertices()
    assert np.allclose(cell1.vertices, cell1_verts, atol=1e-7)
    assert np.allclose(cell2.vertices, cell2_verts, atol=1e-7)


def test_build_global_vertices_empty_cells():
    """Test build_global_vertices with empty cells."""
    tissue = Tissue()

    # Mix of empty and non-empty cells
    empty_cell = Cell(cell_id=1)
    square_cell = Cell(cell_id=2, vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]))

    tissue.add_cell(empty_cell)
    tissue.add_cell(square_cell)

    tissue.build_global_vertices()

    # Empty cell should have empty vertex_indices
    assert empty_cell.vertex_indices.shape[0] == 0

    # Square cell should have 4 indices
    assert square_cell.vertex_indices.shape[0] == 4

    # Global pool should have 4 vertices
    assert tissue.vertices.shape[0] == 4

    # Reconstruct should work
    tissue.reconstruct_cell_vertices()
    assert square_cell.vertices.shape[0] == 4


def test_build_global_vertices_larger_grid():
    """Test with a larger 2x2 grid to verify sharing across multiple cells."""
    tissue = Tissue()

    # Build a 2x2 grid (like the CLI)
    cell_size = 1.0
    cid = 1
    for i in range(2):
        for j in range(2):
            x0, y0 = i * cell_size, j * cell_size
            verts = np.array([
                [x0, y0],
                [x0 + cell_size, y0],
                [x0 + cell_size, y0 + cell_size],
                [x0, y0 + cell_size],
            ], dtype=float)
            tissue.add_cell(Cell(cell_id=cid, vertices=verts))
            cid += 1

    # 4 cells × 4 vertices each = 16 total vertex references
    total_local = sum(cell.vertices.shape[0] for cell in tissue.cells)
    assert total_local == 16

    # Build global pool
    tissue.build_global_vertices()

    # A 2x2 grid should have 9 unique vertices (3×3 grid points)
    # [0,0] [1,0] [2,0]
    # [0,1] [1,1] [2,1]
    # [0,2] [1,2] [2,2]
    assert tissue.vertices.shape[0] == 9

    # Verify reconstruction
    original_vertices = [cell.vertices.copy() for cell in tissue.cells]
    tissue.reconstruct_cell_vertices()

    for i, cell in enumerate(tissue.cells):
        assert np.allclose(cell.vertices, original_vertices[i], atol=1e-8)


def test_energy_with_local_vertices():
    """Test that cell_energy works with local cell.vertices (backward compatibility)."""
    tissue = Tissue()
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square)
    tissue.add_cell(cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Test cell_energy directly (should use cell.vertices)
    e_cell = cell_energy(cell, params, geometry)
    assert np.isfinite(e_cell)
    assert e_cell >= 0.0

    # Test tissue_energy (should also use cell.vertices since no global pool)
    e_tissue = tissue_energy(tissue, params, geometry)
    assert np.isfinite(e_tissue)
    assert e_tissue >= 0.0
    assert e_tissue == pytest.approx(e_cell)


def test_energy_with_global_vertices():
    """Test that energy calculation works with global vertex pool."""
    tissue = Tissue()

    # Create a 2x1 grid
    cell1_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell2_verts = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)

    cell1 = Cell(cell_id=1, vertices=cell1_verts.copy())
    cell2 = Cell(cell_id=2, vertices=cell2_verts.copy())

    tissue.add_cell(cell1)
    tissue.add_cell(cell2)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Calculate energy using local vertices (before build_global_vertices)
    e_local = tissue_energy(tissue, params, geometry)

    # Build global vertex pool
    tissue.build_global_vertices()

    # Calculate energy using global vertices
    e_global = tissue_energy(tissue, params, geometry)

    # Energies should agree within tolerance
    assert np.isclose(e_local, e_global, rtol=1e-10, atol=1e-10)


def test_energy_local_vs_global_agreement():
    """Test that local and global vertex representations give identical energies."""
    tissue = Tissue()

    # Create a more complex tissue (2x2 grid)
    cell_size = 1.0
    cid = 1
    for i in range(2):
        for j in range(2):
            x0, y0 = i * cell_size, j * cell_size
            verts = np.array([
                [x0, y0],
                [x0 + cell_size, y0],
                [x0 + cell_size, y0 + cell_size],
                [x0, y0 + cell_size],
            ], dtype=float)
            tissue.add_cell(Cell(cell_id=cid, vertices=verts))
            cid += 1

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Energy before building global pool
    energy_before = tissue_energy(tissue, params, geometry)

    # Build global pool
    tissue.build_global_vertices()

    # Energy after building global pool (using global vertices)
    energy_after = tissue_energy(tissue, params, geometry)

    # Should be identical (within numerical precision)
    assert np.isclose(energy_before, energy_after, rtol=1e-12, atol=1e-12)

    # Reconstruct local vertices from global pool
    tissue.reconstruct_cell_vertices()

    # Energy should still be the same
    energy_reconstructed = tissue_energy(tissue, params, geometry)
    assert np.isclose(energy_before, energy_reconstructed, rtol=1e-12, atol=1e-12)


def test_cell_energy_explicit_global_vertices():
    """Test cell_energy with explicit global_vertices parameter."""
    # Create a cell with local vertices
    local_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=local_verts)

    # Create a fake global vertex pool
    global_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell.vertex_indices = np.array([0, 1, 2, 3], dtype=int)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Energy using local vertices (global_vertices=None)
    e_local = cell_energy(cell, params, geometry, global_vertices=None)

    # Energy using global vertices
    e_global = cell_energy(cell, params, geometry, global_vertices=global_verts)

    # Should be identical
    assert e_local == pytest.approx(e_global)


def test_cell_energy_fallback_to_local():
    """Test that cell_energy falls back to local vertices when vertex_indices is empty."""
    local_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=local_verts)

    # vertex_indices is empty (default)
    assert cell.vertex_indices.shape[0] == 0

    # Create a global vertex pool (won't be used since vertex_indices is empty)
    global_verts = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Energy with global_vertices provided but vertex_indices empty
    e_with_global = cell_energy(cell, params, geometry, global_vertices=global_verts)

    # Energy without global_vertices
    e_without_global = cell_energy(cell, params, geometry, global_vertices=None)

    # Should be identical (both use local vertices)
    assert e_with_global == pytest.approx(e_without_global)

    # Verify it actually used local vertices (not the bogus global ones)
    # The local square at (0,0)-(1,1) has area 1.0
    area = geometry.calculate_area(local_verts)
    assert area == pytest.approx(1.0)


def test_finite_difference_cell_gradient_basic():
    """Test that finite_difference_cell_gradient computes a valid gradient."""
    from myvertexmodel import finite_difference_cell_gradient

    tissue = Tissue()
    # Create a square cell (not at equilibrium - area != target_area)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())
    tissue.add_cell(cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Compute gradient
    grad = finite_difference_cell_gradient(cell, tissue, params, geometry)

    # Should return array of same shape as vertices
    assert grad.shape == cell.vertices.shape
    assert grad.shape == (4, 2)

    # Gradient should be finite
    assert np.all(np.isfinite(grad))


def test_finite_difference_cell_gradient_energy_descent():
    """Test that following the negative gradient reduces energy."""
    from myvertexmodel import finite_difference_cell_gradient, tissue_energy

    tissue = Tissue()
    # Create a cell that's not at equilibrium
    # Square with area 1.0, but target_area = 1.0, so area term is zero
    # But perimeter and line tension still contribute
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())
    tissue.add_cell(cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Initial energy
    e_initial = tissue_energy(tissue, params, geometry)

    # Compute gradient
    grad = finite_difference_cell_gradient(cell, tissue, params, geometry)

    # Take a small step in the negative gradient direction
    step_size = 0.001
    new_vertices = cell.vertices - step_size * grad

    # Update cell with new vertices
    cell.vertices = new_vertices

    # New energy should be lower (or at least not higher, within tolerance)
    e_after = tissue_energy(tissue, params, geometry)

    # Energy should decrease or stay the same (gradient descent property)
    # Allow small tolerance for numerical errors
    assert e_after <= e_initial + 1e-10


def test_finite_difference_cell_gradient_empty_cell():
    """Test that gradient function handles empty cells correctly."""
    from myvertexmodel import finite_difference_cell_gradient

    tissue = Tissue()
    empty_cell = Cell(cell_id=1)  # No vertices
    tissue.add_cell(empty_cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Should return empty gradient
    grad = finite_difference_cell_gradient(empty_cell, tissue, params, geometry)
    assert grad.shape == (0, 2)


def test_finite_difference_cell_gradient_custom_epsilon():
    """Test that gradient can be computed with custom epsilon."""
    from myvertexmodel import finite_difference_cell_gradient

    tissue = Tissue()
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())
    tissue.add_cell(cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Compute with default epsilon
    grad_default = finite_difference_cell_gradient(cell, tissue, params, geometry)

    # Compute with custom epsilon
    grad_custom = finite_difference_cell_gradient(cell, tissue, params, geometry, epsilon=1e-5)

    # Should be close but potentially slightly different
    assert grad_default.shape == grad_custom.shape
    assert np.allclose(grad_default, grad_custom, rtol=0.1)  # Allow ~10% difference


def test_simulation_uses_gradient_helper():
    """Test that Simulation.step() correctly uses the gradient helper."""
    tissue = Tissue()
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())
    tissue.add_cell(cell)

    sim = Simulation(tissue=tissue, dt=0.01)

    # Get initial energy
    e_initial = sim.total_energy()

    # Take one step
    sim.step()

    # Energy should decrease (or stay same within tolerance)
    e_after = sim.total_energy()
    assert e_after <= e_initial + 1e-10

    # Time should advance
    assert sim.time == pytest.approx(0.01)


def test_finite_difference_gradient_multiple_cells():
    """Test gradient computation with multiple cells in tissue."""
    from myvertexmodel import finite_difference_cell_gradient, tissue_energy

    tissue = Tissue()

    # Create a 2x1 grid
    cell1 = Cell(cell_id=1, vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    cell2 = Cell(cell_id=2, vertices=np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float))
    tissue.add_cell(cell1)
    tissue.add_cell(cell2)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Compute gradient for first cell
    grad1 = finite_difference_cell_gradient(cell1, tissue, params, geometry)

    # Should have correct shape
    assert grad1.shape == (4, 2)
    assert np.all(np.isfinite(grad1))

    # Take a small descent step
    step_size = 0.001
    e_before = tissue_energy(tissue, params, geometry)

    # Update only cell1
    cell1.vertices = cell1.vertices - step_size * grad1

    e_after = tissue_energy(tissue, params, geometry)

    # Energy should not increase
    assert e_after <= e_before + 1e-10


def test_analytic_gradient_not_implemented():
    """Test that cell_energy_gradient_analytic raises NotImplementedError."""
    from myvertexmodel import cell_energy_gradient_analytic

    tissue = Tissue()
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square)
    tissue.add_cell(cell)

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        cell_energy_gradient_analytic(cell, params, geometry)

    # Check error message is informative
    error_msg = str(exc_info.value)
    assert "not yet implemented" in error_msg.lower()
    assert "finite_difference_cell_gradient" in error_msg


def test_analytic_gradient_placeholder_signature():
    """Test that cell_energy_gradient_analytic has the correct signature."""
    from myvertexmodel import cell_energy_gradient_analytic
    import inspect

    # Get function signature
    sig = inspect.signature(cell_energy_gradient_analytic)

    # Should have 3 parameters: cell, params, geometry
    params = list(sig.parameters.keys())
    assert len(params) == 3
    assert 'cell' in params
    assert 'params' in params
    assert 'geometry' in params

    # Should have return type annotation
    assert sig.return_annotation != inspect.Signature.empty


def test_analytic_gradient_has_documentation():
    """Test that cell_energy_gradient_analytic has comprehensive documentation."""
    from myvertexmodel import cell_energy_gradient_analytic

    # Should have a docstring
    assert cell_energy_gradient_analytic.__doc__ is not None
    docstring = cell_energy_gradient_analytic.__doc__

    # Docstring should mention key concepts
    assert "analytic" in docstring.lower() or "analytical" in docstring.lower()
    assert "gradient" in docstring.lower()
    assert "energy" in docstring.lower()

    # Should mention it's not implemented
    assert "not" in docstring.lower() and "implemented" in docstring.lower()

    # Should mention the finite difference alternative
    assert "finite" in docstring.lower() or "finite_difference" in docstring


def test_analytic_gradient_future_interface():
    """Test that when implemented, analytic gradient will have correct interface."""
    from myvertexmodel import cell_energy_gradient_analytic

    # When implemented, it should accept a cell and return array
    # For now, we just verify the signature is correct for future implementation

    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square)
    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Try to call it (will raise NotImplementedError)
    try:
        result = cell_energy_gradient_analytic(cell, params, geometry)
        # If it doesn't raise, it was implemented - check return type
        assert isinstance(result, np.ndarray)
        assert result.shape == cell.vertices.shape
    except NotImplementedError:
        # Expected for now - this is the placeholder
        pass


def test_geometry_is_valid_polygon_square_and_degenerate():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert GeometryCalculator.is_valid_polygon(square)

    # Degenerate: repeated vertices
    degenerate = np.array([[0, 0], [1, 0], [1, 0], [0, 1]], dtype=float)
    assert not GeometryCalculator.is_valid_polygon(degenerate)


def test_geometry_signed_area_orientation_square():
    # CCW square
    ccw_square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    assert GeometryCalculator.signed_area(ccw_square) > 0

    # CW square (reverse order)
    cw_square = ccw_square[::-1]
    assert GeometryCalculator.signed_area(cw_square) < 0


def test_geometry_ensure_ccw_makes_ccw():
    # Start with CW ordering
    cw_square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    assert GeometryCalculator.signed_area(cw_square) < 0

    ccw_square = GeometryCalculator.ensure_ccw(cw_square)
    assert GeometryCalculator.signed_area(ccw_square) > 0
    # Should be reversed copy
    assert np.allclose(ccw_square, cw_square[::-1])
    # Input unchanged
    assert GeometryCalculator.signed_area(cw_square) < 0


def test_energy_translation_invariance():
    """Energy should be invariant under translations (rigid motion)."""
    from myvertexmodel import Cell, EnergyParameters, cell_energy, GeometryCalculator
    import numpy as np

    # Unit square at origin
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Energy before translation
    e_before = cell_energy(cell, params, geometry)

    # Translate by arbitrary vector
    t = np.array([5.0, -3.0], dtype=float)
    translated = square + t

    # Update cell vertices and recompute energy
    cell.vertices = translated
    e_after = cell_energy(cell, params, geometry)

    # Energies should be identical within numerical tolerance
    assert np.isclose(e_before, e_after, rtol=1e-12, atol=1e-12)


def test_energy_rotation_invariance_90deg():
    """Energy should be invariant under 90-degree rotation around the origin."""
    from myvertexmodel import Cell, EnergyParameters, cell_energy, GeometryCalculator
    import numpy as np

    # Unit square at origin
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=square.copy())

    params = EnergyParameters()
    geometry = GeometryCalculator()

    # Energy before rotation
    e_before = cell_energy(cell, params, geometry)

    # 90-degree rotation matrix (counterclockwise)
    R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    rotated = square @ R.T

    # Update cell vertices and recompute energy
    cell.vertices = rotated
    e_after = cell_energy(cell, params, geometry)

    # Energies should be identical within numerical tolerance
    assert np.isclose(e_before, e_after, rtol=1e-12, atol=1e-12)


def test_global_vertex_pool_with_build_grid_tissue_sharing():
    """Using build_grid_tissue(2x1), global vertex pool should share boundary vertices."""
    from myvertexmodel.__main__ import build_grid_tissue

    # Build a 2x1 grid (grid_size=2 along x, 1 along y using custom helper)
    # The helper builds grid_size^2 cells; for 2x1 we simulate by building 2x2 and slicing,
    # but here we build grid_size=2 and then select first row to emulate 2x1.
    tissue_full = build_grid_tissue(grid_size=2, cell_size=1.0)

    # Reduce to 2x1 by taking the first two cells (at y=0 row)
    tissue = Tissue()
    tissue.add_cell(tissue_full.cells[0])
    tissue.add_cell(tissue_full.cells[1])

    # Sum of local vertices before pooling
    total_local = sum(cell.vertices.shape[0] for cell in tissue.cells)
    assert total_local == 8  # 2 cells × 4 vertices

    # Build global vertex pool
    tissue.build_global_vertices()

    # Global vertices should be fewer than total local (shared boundary)
    assert tissue.vertices.shape[0] < total_local


def test_global_vertex_pool_reconstruct_matches_original_build_grid_tissue():
    """Reconstructing cell vertices from global pool should match originals (within tolerance)."""
    from myvertexmodel.__main__ import build_grid_tissue

    # Build 2x1 tissue similarly by selecting first row cells
    tissue_full = build_grid_tissue(grid_size=2, cell_size=1.0)
    tissue = Tissue()
    tissue.add_cell(tissue_full.cells[0])
    tissue.add_cell(tissue_full.cells[1])

    # Keep copies of original local vertices
    originals = [cell.vertices.copy() for cell in tissue.cells]

    # Build global pool and reconstruct
    tissue.build_global_vertices()
    tissue.reconstruct_cell_vertices()

    # Each cell's vertices should match originals within tolerance
    for cell, orig in zip(tissue.cells, originals):
        assert np.allclose(cell.vertices, orig, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
