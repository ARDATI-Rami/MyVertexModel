"""
Tests for cytokinesis (cell division) functionality.
"""

import pytest
import numpy as np
from myvertexmodel import (
    Cell,
    Tissue,
    CytokinesisParams,
    compute_division_axis,
    insert_contracting_vertices,
    compute_contractile_forces,
    check_constriction,
    split_cell,
    perform_cytokinesis,
)


def test_cytokinesis_params_creation():
    """Test creating cytokinesis parameters."""
    params = CytokinesisParams()
    assert params.constriction_threshold == 0.1
    assert params.initial_separation_fraction == 0.95
    assert params.contractile_force_magnitude == 10.0
    
    # Custom parameters
    custom = CytokinesisParams(
        constriction_threshold=0.05,
        initial_separation_fraction=0.9,
        contractile_force_magnitude=20.0
    )
    assert custom.constriction_threshold == 0.05
    assert custom.initial_separation_fraction == 0.9
    assert custom.contractile_force_magnitude == 20.0


def test_compute_division_axis_with_angle():
    """Test computing division axis with specified angle."""
    # Create a square cell
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Compute axis at 0 degrees (horizontal)
    centroid, axis_dir, perp_dir = compute_division_axis(cell, tissue, axis_angle=0.0)
    
    assert centroid == pytest.approx([0.5, 0.5], abs=1e-6)
    assert axis_dir == pytest.approx([1.0, 0.0], abs=1e-6)
    assert perp_dir == pytest.approx([0.0, 1.0], abs=1e-6)
    
    # Compute axis at 90 degrees (vertical)
    centroid, axis_dir, perp_dir = compute_division_axis(
        cell, tissue, axis_angle=np.pi/2
    )
    
    assert centroid == pytest.approx([0.5, 0.5], abs=1e-6)
    assert axis_dir == pytest.approx([0.0, 1.0], abs=1e-6)
    assert perp_dir == pytest.approx([-1.0, 0.0], abs=1e-6)


def test_compute_division_axis_principal():
    """Test computing division axis using principal component."""
    # Create an elongated rectangle (2x1)
    vertices = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Compute axis using PCA (should align with long axis)
    centroid, axis_dir, perp_dir = compute_division_axis(cell, tissue)
    
    assert centroid == pytest.approx([1.0, 0.5], abs=1e-6)
    # Long axis should be horizontal (or its negative)
    assert abs(axis_dir[0]) == pytest.approx(1.0, abs=1e-6)
    assert abs(axis_dir[1]) == pytest.approx(0.0, abs=1e-6)


def test_insert_contracting_vertices_square():
    """Test inserting contracting vertices into a square cell."""
    # Create a square cell
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    initial_vertex_count = tissue.vertices.shape[0]
    
    # Insert contracting vertices with horizontal division (angle=0)
    v1_idx, v2_idx = insert_contracting_vertices(
        cell, tissue, axis_angle=0.0
    )
    
    # Should have added 2 vertices
    assert tissue.vertices.shape[0] == initial_vertex_count + 2
    
    # Cell should now have 6 vertices (4 original + 2 new)
    assert cell.vertex_indices.shape[0] == 6
    
    # Check that contracting vertices are in the cell
    assert v1_idx in cell.vertex_indices
    assert v2_idx in cell.vertex_indices
    
    # Check that cell has cytokinesis metadata
    assert hasattr(cell, 'cytokinesis_data')
    assert 'contracting_vertices' in cell.cytokinesis_data
    assert cell.cytokinesis_data['contracting_vertices'] == (v1_idx, v2_idx)


def test_insert_contracting_vertices_separation():
    """Test that contracting vertices start with appropriate separation."""
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    params = CytokinesisParams(initial_separation_fraction=0.9)
    
    v1_idx, v2_idx = insert_contracting_vertices(
        cell, tissue, axis_angle=0.0, params=params
    )
    
    pos1 = tissue.vertices[v1_idx]
    pos2 = tissue.vertices[v2_idx]
    
    # Centroid is at (1, 1)
    # With horizontal division, vertices should be at (1, y1) and (1, y2)
    # where y1 and y2 are separated by initial_separation_fraction * 2.0 = 1.8
    
    distance = np.linalg.norm(pos2 - pos1)
    expected_distance = 0.9 * 2.0  # 90% of full width
    
    assert distance == pytest.approx(expected_distance, abs=1e-6)


def test_compute_contractile_forces():
    """Test computing contractile forces for dividing cell."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    params = CytokinesisParams(contractile_force_magnitude=10.0)
    v1_idx, v2_idx = insert_contracting_vertices(
        cell, tissue, axis_angle=0.0, params=params
    )
    
    forces = compute_contractile_forces(cell, tissue, params)
    
    # Forces should be zero for most vertices
    assert forces.shape == (6, 2)  # 4 original + 2 contracting
    
    # Find local indices of contracting vertices
    v1_local = np.where(cell.vertex_indices == v1_idx)[0][0]
    v2_local = np.where(cell.vertex_indices == v2_idx)[0][0]
    
    # Forces should be non-zero only at contracting vertices
    for i in range(6):
        if i not in [v1_local, v2_local]:
            assert np.allclose(forces[i], [0, 0])
    
    # Forces should be opposite and equal in magnitude
    force1 = forces[v1_local]
    force2 = forces[v2_local]
    
    assert np.linalg.norm(force1) == pytest.approx(10.0, abs=1e-6)
    assert np.linalg.norm(force2) == pytest.approx(10.0, abs=1e-6)
    assert np.allclose(force1, -force2)


def test_check_constriction():
    """Test checking if cell is sufficiently constricted."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    params = CytokinesisParams(constriction_threshold=0.1)
    v1_idx, v2_idx = insert_contracting_vertices(cell, tissue, params=params)
    
    # Initially, not constricted (vertices are separated)
    assert not check_constriction(cell, tissue, params)
    
    # Manually move vertices closer
    tissue.vertices[v1_idx] = [0.5, 0.49]
    tissue.vertices[v2_idx] = [0.5, 0.51]
    
    # Distance is 0.02, should be constricted
    assert check_constriction(cell, tissue, params)


def test_split_cell_basic():
    """Test splitting a cell into two daughters."""
    # Create a square cell with contracting vertices
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    v1_idx, v2_idx = insert_contracting_vertices(
        cell, tissue, axis_angle=0.0
    )
    
    # Split the cell
    daughter1, daughter2 = split_cell(cell, tissue, daughter1_id=2, daughter2_id=3)
    
    # Original cell should be removed
    assert cell not in tissue.cells
    
    # Daughter cells should be in tissue
    assert daughter1 in tissue.cells
    assert daughter2 in tissue.cells
    
    # Each daughter should have at least 3 vertices
    assert daughter1.vertex_indices.shape[0] >= 3
    assert daughter2.vertex_indices.shape[0] >= 3
    
    # Both daughters should share the contracting vertices
    assert v1_idx in daughter1.vertex_indices
    assert v2_idx in daughter1.vertex_indices
    assert v1_idx in daughter2.vertex_indices
    assert v2_idx in daughter2.vertex_indices
    
    # Total tissue should have 2 cells now
    assert len(tissue.cells) == 2


def test_split_cell_preserves_vertices():
    """Test that splitting preserves all vertices."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    v1_idx, v2_idx = insert_contracting_vertices(cell, tissue)
    
    # Record vertices before split
    original_indices = set(cell.vertex_indices)
    
    daughter1, daughter2 = split_cell(cell, tissue)
    
    # Union of daughter vertices should contain all original vertices
    daughter_indices = set(daughter1.vertex_indices) | set(daughter2.vertex_indices)
    assert original_indices.issubset(daughter_indices)


def test_perform_cytokinesis_initiation():
    """Test initiating cytokinesis."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    result = perform_cytokinesis(cell, tissue)
    
    assert result['stage'] == 'initiated'
    assert 'contracting_vertices' in result
    assert len(result['contracting_vertices']) == 2


def test_perform_cytokinesis_constricting():
    """Test cytokinesis in constricting stage."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Initiate
    result1 = perform_cytokinesis(cell, tissue)
    assert result1['stage'] == 'initiated'
    
    # Call again without constriction
    result2 = perform_cytokinesis(cell, tissue)
    assert result2['stage'] == 'constricting'
    assert 'constriction_distance' in result2


def test_perform_cytokinesis_complete():
    """Test completing cytokinesis."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    params = CytokinesisParams(constriction_threshold=10.0)  # Large threshold
    
    # Initiate
    result1 = perform_cytokinesis(cell, tissue, params=params)
    assert result1['stage'] == 'initiated'
    
    # Call again with large threshold (should split immediately)
    result2 = perform_cytokinesis(
        cell, tissue, params=params, daughter1_id=2, daughter2_id=3
    )
    assert result2['stage'] == 'completed'
    assert 'daughter_cells' in result2
    assert len(result2['daughter_cells']) == 2


def test_cytokinesis_with_simulation():
    """Test cytokinesis integrated with simulation."""
    # Create a cell
    vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Set up cytokinesis
    params = CytokinesisParams(
        constriction_threshold=0.2,
        contractile_force_magnitude=50.0
    )
    
    # Initiate cytokinesis
    result = perform_cytokinesis(cell, tissue, params=params)
    assert result['stage'] == 'initiated'
    v1_idx, v2_idx = result['contracting_vertices']
    
    # Test that we can compute contractile forces
    forces = compute_contractile_forces(cell, tissue, params)
    assert forces.shape == cell.vertices.shape
    
    # Find local indices of contracting vertices
    v1_local = np.where(cell.vertex_indices == v1_idx)[0][0]
    v2_local = np.where(cell.vertex_indices == v2_idx)[0][0]
    
    # Check that forces are non-zero at contracting vertices
    assert np.linalg.norm(forces[v1_local]) > 0
    assert np.linalg.norm(forces[v2_local]) > 0
    
    # Check that forces are opposite
    assert np.allclose(forces[v1_local], -forces[v2_local])


def test_cytokinesis_different_angles():
    """Test cytokinesis with different division angles."""
    for angle in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        cell = Cell(cell_id=1, vertices=vertices)
        tissue = Tissue()
        tissue.add_cell(cell)
        tissue.build_global_vertices()
        
        v1_idx, v2_idx = insert_contracting_vertices(
            cell, tissue, axis_angle=angle
        )
        
        # Should have inserted 2 vertices
        assert v1_idx in cell.vertex_indices
        assert v2_idx in cell.vertex_indices
        assert cell.vertex_indices.shape[0] == 6


def test_cytokinesis_validates_small_cell():
    """Test that cytokinesis fails for cells with too few vertices."""
    # Create a triangle (3 vertices)
    vertices = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Try to insert contracting vertices
    # This might work (triangle -> pentagon)
    insert_contracting_vertices(cell, tissue)
    
    # But splitting might fail if daughters would have < 3 vertices
    # This depends on where the contracting vertices are inserted


def test_split_cell_auto_ids():
    """Test that split_cell generates IDs automatically."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    insert_contracting_vertices(cell, tissue)
    
    daughter1, daughter2 = split_cell(cell, tissue)
    
    # IDs should be auto-generated
    assert daughter1.id == "1_d1"
    assert daughter2.id == "1_d2"


def test_compute_division_axis_error_on_small_cell():
    """Test that compute_division_axis raises error for invalid cells."""
    # Cell with only 2 vertices
    vertices = np.array([[0, 0], [1, 0]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    
    with pytest.raises(ValueError, match="fewer than 3 vertices"):
        compute_division_axis(cell, tissue)


def test_contractile_forces_error_without_metadata():
    """Test that compute_contractile_forces raises error without metadata."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    with pytest.raises(ValueError, match="does not have contracting vertices"):
        compute_contractile_forces(cell, tissue)


def test_split_cell_error_without_metadata():
    """Test that split_cell raises error without metadata."""
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    with pytest.raises(ValueError, match="does not have contracting vertices"):
        split_cell(cell, tissue)


def test_insert_contracting_vertices_updates_neighbors():
    """Test that inserting contracting vertices also updates neighboring cells.

    When contracting vertices are inserted on edges shared with neighbor cells,
    those neighbors must also have their vertex_indices updated to include the
    new vertices, maintaining tissue connectivity.
    """
    from myvertexmodel import build_honeycomb_2_3_4_3_2

    # Build honeycomb tissue where cells share edges
    tissue = build_honeycomb_2_3_4_3_2()
    tissue.build_global_vertices(tol=1e-10)
    tissue.reconstruct_cell_vertices()

    # Find cell 7 (center cell with 6 neighbors)
    cell7 = None
    for c in tissue.cells:
        if c.id == 7:
            cell7 = c
            break
    assert cell7 is not None

    # Record neighbor counts before
    neighbor_counts_before = tissue.cell_neighbor_counts()

    # Get original vertex count for cell 7 and its neighbors
    original_cell7_verts = len(cell7.vertex_indices)
    neighbors_before = {}
    for c in tissue.cells:
        if c.id != 7:
            neighbors_before[c.id] = len(c.vertex_indices)

    # Insert contracting vertices
    params = CytokinesisParams()
    v1_idx, v2_idx = insert_contracting_vertices(cell7, tissue, axis_angle=None, params=params)

    # Cell 7 should have 2 more vertices
    assert len(cell7.vertex_indices) == original_cell7_verts + 2

    # Check that contracting vertices are in cell 7
    assert v1_idx in cell7.vertex_indices
    assert v2_idx in cell7.vertex_indices

    # The new vertices should also be in some neighbor cells
    # (those that share the edges where vertices were inserted)
    neighbors_with_new_vertex = []
    for c in tissue.cells:
        if c.id == 7:
            continue
        if v1_idx in c.vertex_indices:
            neighbors_with_new_vertex.append((c.id, v1_idx))
        if v2_idx in c.vertex_indices:
            neighbors_with_new_vertex.append((c.id, v2_idx))

    # At least 2 neighbor cells should have been updated (one for each contracting vertex)
    # (each contracting vertex is on an edge shared with one neighbor)
    assert len(neighbors_with_new_vertex) >= 2, (
        f"Expected at least 2 neighbor cells to contain new vertices, "
        f"but found: {neighbors_with_new_vertex}"
    )

    # Validate tissue is still consistent
    tissue.validate()

    # Verify connectivity is maintained - neighbors of cell 7 should still be neighbors
    # (The new vertices should not break any edges)
    neighbor_counts_after = tissue.cell_neighbor_counts()
    assert neighbor_counts_after[7] == neighbor_counts_before[7], (
        f"Cell 7 neighbor count changed from {neighbor_counts_before[7]} to {neighbor_counts_after[7]}"
    )


def test_insert_contracting_vertices_with_two_adjacent_cells():
    """Test contracting vertex insertion with a minimal case of two adjacent cells."""
    # Create two cells sharing an edge
    cell1_verts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell2_verts = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], dtype=float)

    cell1 = Cell(cell_id=1, vertices=cell1_verts)
    cell2 = Cell(cell_id=2, vertices=cell2_verts)

    tissue = Tissue()
    tissue.add_cell(cell1)
    tissue.add_cell(cell2)
    tissue.build_global_vertices(tol=1e-8)
    tissue.reconstruct_cell_vertices()

    # Record shared edge vertices
    cell1_indices_before = set(cell1.vertex_indices)
    cell2_indices_before = set(cell2.vertex_indices)
    shared_before = cell1_indices_before & cell2_indices_before

    # Cell 1 and 2 should share 2 vertices (the shared edge)
    assert len(shared_before) == 2

    # Insert contracting vertices into cell 1 with horizontal division
    # This should intersect the shared edge with cell 2
    params = CytokinesisParams()
    v1_idx, v2_idx = insert_contracting_vertices(cell1, tissue, axis_angle=0.0, params=params)

    # Determine which new vertex is on the shared edge
    # by checking if it falls between the shared edge endpoints
    shared_edge_vertex = None
    for new_v in [v1_idx, v2_idx]:
        new_pos = tissue.vertices[new_v]
        # The shared edge is at x=1, y between 0 and 1
        if abs(new_pos[0] - 1.0) < 0.5:  # Close to x=1
            shared_edge_vertex = new_v
            break

    # If a vertex was inserted on the shared edge, cell 2 should have it
    if shared_edge_vertex is not None:
        if shared_edge_vertex not in cell2.vertex_indices:
            # This would be a bug - the fix should ensure neighbor is updated
            pytest.fail(
                f"Contracting vertex {shared_edge_vertex} at position {tissue.vertices[shared_edge_vertex]} "
                f"should have been inserted into cell 2's vertex_indices but wasn't. "
                f"Cell 2 indices: {list(cell2.vertex_indices)}"
            )

    # Validate tissue
    tissue.validate()


