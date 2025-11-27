"""
Cell-by-cell tissue validation test.

This test loads ALL tissues from pickled_tissues/ and validates that:
1. Every cell has vertices in counter-clockwise (CCW) order
2. The CCW ordering is preserved when vertices are passed through the global vertex pool
3. Cell vertices match their global vertex pool references
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple

from myvertexmodel import load_tissue, Tissue, Cell
from myvertexmodel.geometry import polygon_orientation, is_valid_polygon


def get_all_tissue_files() -> List[Path]:
    """Get all .dill tissue files from pickled_tissues directory."""
    tissue_dir = Path("pickled_tissues")
    if not tissue_dir.exists():
        return []
    return sorted(tissue_dir.glob("*.dill"))


def validate_cell_ccw_ordering(cell: Cell, cell_idx: int, tissue_name: str) -> Tuple[bool, str]:
    """
    Validate that a single cell has CCW vertex ordering.
    
    Args:
        cell: The cell to validate
        cell_idx: Index of cell in tissue (for error reporting)
        tissue_name: Name of tissue (for error reporting)
    
    Returns:
        (is_valid, error_message) tuple
    """
    if len(cell.vertices) < 3:
        return True, ""  # Skip cells with < 3 vertices
    
    orientation = polygon_orientation(cell.vertices)
    
    if orientation < 0:
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Vertices are in CLOCKWISE order (orientation={orientation:.6f})"
        )
        return False, error_msg
    elif orientation == 0:
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Vertices are DEGENERATE (orientation=0, collinear or zero area)"
        )
        return False, error_msg
    
    return True, ""


def validate_global_vertex_consistency(cell: Cell, tissue: Tissue, 
                                       cell_idx: int, tissue_name: str) -> Tuple[bool, str]:
    """
    Validate that cell vertices match their global vertex pool references.
    
    Args:
        cell: The cell to validate
        tissue: The parent tissue
        cell_idx: Index of cell in tissue (for error reporting)
        tissue_name: Name of tissue (for error reporting)
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
        return True, ""  # No global pool to check against
    
    if not hasattr(cell, 'vertex_indices') or len(cell.vertex_indices) == 0:
        return True, ""  # No indices to check
    
    # Check that vertex_indices are valid
    if np.any(cell.vertex_indices < 0):
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Has negative vertex indices: {cell.vertex_indices[cell.vertex_indices < 0]}"
        )
        return False, error_msg
    
    if np.any(cell.vertex_indices >= len(tissue.vertices)):
        invalid = cell.vertex_indices[cell.vertex_indices >= len(tissue.vertices)]
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Has out-of-bounds vertex indices: {invalid} (max={len(tissue.vertices)-1})"
        )
        return False, error_msg
    
    # Check that cell.vertices matches global pool
    if len(cell.vertices) != len(cell.vertex_indices):
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"vertex_indices length ({len(cell.vertex_indices)}) != "
            f"vertices length ({len(cell.vertices)})"
        )
        return False, error_msg
    
    # Reconstruct from global pool and compare
    reconstructed = tissue.vertices[cell.vertex_indices]
    max_diff = np.max(np.abs(reconstructed - cell.vertices))
    
    if max_diff > 1e-10:
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Cell vertices don't match global pool references (max diff={max_diff:.2e})"
        )
        return False, error_msg
    
    # Check that CCW ordering is preserved through global pool
    if len(cell.vertices) >= 3:
        local_orientation = polygon_orientation(cell.vertices)
        global_orientation = polygon_orientation(reconstructed)
        
        if np.sign(local_orientation) != np.sign(global_orientation):
            error_msg = (
                f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
                f"Orientation changed through global pool! "
                f"Local={local_orientation:.6f}, Global={global_orientation:.6f}"
            )
            return False, error_msg
    
    return True, ""


def validate_no_duplicate_consecutive_vertices(cell: Cell, cell_idx: int, 
                                               tissue_name: str) -> Tuple[bool, str]:
    """
    Validate that a cell has no duplicate consecutive vertices in vertex_indices.
    
    Args:
        cell: The cell to validate
        cell_idx: Index of cell in tissue (for error reporting)
        tissue_name: Name of tissue (for error reporting)
    
    Returns:
        (is_valid, error_message) tuple
    """
    if not hasattr(cell, 'vertex_indices') or len(cell.vertex_indices) < 2:
        return True, ""
    
    indices = cell.vertex_indices
    duplicates = []
    
    for i in range(len(indices) - 1):
        if indices[i] == indices[i+1]:
            duplicates.append((i, indices[i]))
    
    if duplicates:
        dup_str = ", ".join([f"pos {i}: vi={vi}" for i, vi in duplicates])
        error_msg = (
            f"[{tissue_name}] Cell {cell.id} (index {cell_idx}): "
            f"Has duplicate consecutive vertices in vertex_indices: {dup_str}"
        )
        return False, error_msg
    
    return True, ""


def test_tissue_cell_by_cell():
    """
    Load ALL tissues and validate every cell's vertex ordering and global consistency.
    
    This test ensures:
    1. All cells have CCW vertex ordering
    2. Vertex ordering is preserved through global vertex pool
    3. Cell vertices match their global vertex references
    4. No duplicate consecutive vertices exist
    """
    tissue_files = get_all_tissue_files()
    
    if len(tissue_files) == 0:
        pytest.skip("No tissue files found in pickled_tissues/")
    
    print(f"\n{'='*70}")
    print(f"CELL-BY-CELL VALIDATION: {len(tissue_files)} tissue(s)")
    print(f"{'='*70}")
    
    all_errors = []
    tissue_stats = {}
    
    for tissue_file in tissue_files:
        tissue_name = tissue_file.stem
        print(f"\n[{tissue_name}] Loading from {tissue_file}...")
        
        try:
            tissue = load_tissue(str(tissue_file))
        except Exception as e:
            error_msg = f"[{tissue_name}] Failed to load: {e}"
            all_errors.append(error_msg)
            print(f"  ❌ {error_msg}")
            continue
        
        print(f"[{tissue_name}] Loaded {len(tissue.cells)} cells")
        
        # Build global vertex pool if missing
        if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
            print(f"[{tissue_name}] Building global vertex pool...")
            tissue.build_global_vertices(tol=1e-10)
            tissue.reconstruct_cell_vertices()
            print(f"[{tissue_name}] Global vertices: {tissue.vertices.shape[0]}")
        else:
            print(f"[{tissue_name}] Global vertices: {tissue.vertices.shape[0]}")
        
        # Validate each cell
        cell_errors = []
        ccw_violations = []
        global_mismatches = []
        duplicate_vertices = []
        
        for cell_idx, cell in enumerate(tissue.cells):
            # Check 1: CCW ordering
            is_valid, error = validate_cell_ccw_ordering(cell, cell_idx, tissue_name)
            if not is_valid:
                cell_errors.append(error)
                ccw_violations.append(cell.id)
            
            # Check 2: Global vertex consistency
            is_valid, error = validate_global_vertex_consistency(cell, tissue, cell_idx, tissue_name)
            if not is_valid:
                cell_errors.append(error)
                global_mismatches.append(cell.id)
            
            # Check 3: No duplicate consecutive vertices
            is_valid, error = validate_no_duplicate_consecutive_vertices(cell, cell_idx, tissue_name)
            if not is_valid:
                cell_errors.append(error)
                duplicate_vertices.append(cell.id)
        
        # Store statistics
        tissue_stats[tissue_name] = {
            'num_cells': len(tissue.cells),
            'num_errors': len(cell_errors),
            'ccw_violations': len(ccw_violations),
            'global_mismatches': len(global_mismatches),
            'duplicate_vertices': len(duplicate_vertices),
        }
        
        # Report results for this tissue
        if len(cell_errors) == 0:
            print(f"[{tissue_name}] ✓ All {len(tissue.cells)} cells validated successfully!")
        else:
            print(f"[{tissue_name}] ❌ Found {len(cell_errors)} error(s):")
            for error in cell_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(cell_errors) > 10:
                print(f"  ... and {len(cell_errors) - 10} more errors")
            
            all_errors.extend(cell_errors)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {len(tissue_files)} tissue(s) validated")
    print(f"{'='*70}")
    
    for tissue_name, stats in tissue_stats.items():
        status = "✓" if stats['num_errors'] == 0 else "❌"
        print(f"{status} [{tissue_name}]: {stats['num_cells']} cells, {stats['num_errors']} errors")
        if stats['ccw_violations'] > 0:
            print(f"    - CCW violations: {stats['ccw_violations']}")
        if stats['global_mismatches'] > 0:
            print(f"    - Global mismatches: {stats['global_mismatches']}")
        if stats['duplicate_vertices'] > 0:
            print(f"    - Duplicate vertices: {stats['duplicate_vertices']}")
    
    total_cells = sum(s['num_cells'] for s in tissue_stats.values())
    total_errors = len(all_errors)
    
    print(f"\nTotal: {total_cells} cells validated, {total_errors} errors found")
    print(f"{'='*70}\n")
    
    # Assert no errors
    if len(all_errors) > 0:
        error_summary = "\n".join(all_errors[:20])  # Show first 20 errors
        if len(all_errors) > 20:
            error_summary += f"\n... and {len(all_errors) - 20} more errors"
        
        pytest.fail(
            f"Cell-by-cell validation failed with {len(all_errors)} error(s):\n{error_summary}"
        )


def test_tissue_cell_by_cell_specific():
    """
    Parametrized version - test each tissue file individually.
    This allows pytest to report results per tissue.
    """
    tissue_files = get_all_tissue_files()
    
    if len(tissue_files) == 0:
        pytest.skip("No tissue files found in pickled_tissues/")
    
    # This is a placeholder - the main test above does all the work
    # But we keep this for potential future parametrization
    pass


if __name__ == "__main__":
    # Allow running directly
    test_tissue_cell_by_cell()

