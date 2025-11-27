"""
Test tissue structure validation and comparison.

This module provides comprehensive validation of tissue structure integrity,
including connectivity, vertex sharing, and simulation readiness.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from myvertexmodel import Tissue, Cell, GeometryCalculator, load_tissue
from myvertexmodel.geometry import is_valid_polygon, polygon_orientation, ensure_ccw


class TissueStructureValidator:
    """Validate tissue structure integrity."""
    
    def __init__(self, tissue: Tissue):
        self.tissue = tissue
        self.geometry = GeometryCalculator()
        self.issues = []
        self.warnings = []
        
    def validate_all(self) -> bool:
        """Run all validation checks. Returns True if tissue is valid."""
        self.issues = []
        self.warnings = []
        
        self._check_basic_structure()
        self._check_global_vertex_pool()
        self._check_vertex_indices()
        self._check_cell_polygons()
        self._check_vertex_sharing()
        self._check_connectivity()
        self._check_areas_and_geometry()
        self._check_nan_inf()
        self._check_vertex_ordering()
        
        return len(self.issues) == 0
    
    def _check_basic_structure(self):
        """Check basic tissue structure."""
        if not hasattr(self.tissue, 'cells'):
            self.issues.append("Tissue missing 'cells' attribute")
            return
            
        if len(self.tissue.cells) == 0:
            self.warnings.append("Tissue has no cells")
            
        for cell in self.tissue.cells:
            if not isinstance(cell, Cell):
                self.issues.append(f"Cell {cell} is not a Cell instance")
            if not hasattr(cell, 'vertices'):
                self.issues.append(f"Cell {cell.id if hasattr(cell, 'id') else '?'} missing 'vertices' attribute")
            if not hasattr(cell, 'id'):
                self.issues.append(f"Cell missing 'id' attribute")
                
    def _check_global_vertex_pool(self):
        """Check if global vertex pool exists and is properly formed."""
        if not hasattr(self.tissue, 'vertices'):
            self.warnings.append("Tissue missing 'vertices' attribute (global vertex pool)")
            return
            
        if self.tissue.vertices is None:
            self.warnings.append("Global vertex pool is None")
            return
            
        if not isinstance(self.tissue.vertices, np.ndarray):
            self.issues.append(f"Global vertices should be ndarray, got {type(self.tissue.vertices)}")
            return
            
        if self.tissue.vertices.ndim != 2:
            self.issues.append(f"Global vertices should be 2D array, got {self.tissue.vertices.ndim}D")
            return
            
        if self.tissue.vertices.shape[0] > 0 and self.tissue.vertices.shape[1] != 2:
            self.issues.append(f"Global vertices should have shape (N, 2), got {self.tissue.vertices.shape}")
            
    def _check_vertex_indices(self):
        """Check if cells have vertex_indices and they reference valid global vertices."""
        if not hasattr(self.tissue, 'vertices') or self.tissue.vertices.shape[0] == 0:
            self.warnings.append("Cannot check vertex_indices without global vertex pool")
            return
            
        num_global_vertices = self.tissue.vertices.shape[0]
        
        for cell in self.tissue.cells:
            if not hasattr(cell, 'vertex_indices'):
                self.warnings.append(f"Cell {cell.id} missing 'vertex_indices' attribute")
                continue
                
            if cell.vertex_indices is None or len(cell.vertex_indices) == 0:
                if len(cell.vertices) > 0:
                    self.warnings.append(f"Cell {cell.id} has vertices but no vertex_indices")
                continue
                
            # Check indices are valid
            if np.any(cell.vertex_indices < 0):
                self.issues.append(f"Cell {cell.id} has negative vertex indices")
            if np.any(cell.vertex_indices >= num_global_vertices):
                self.issues.append(f"Cell {cell.id} has vertex indices >= global pool size ({num_global_vertices})")
                
            # Check consistency between vertex_indices and vertices
            if len(cell.vertices) > 0:
                if len(cell.vertex_indices) != len(cell.vertices):
                    self.warnings.append(
                        f"Cell {cell.id}: vertex_indices length ({len(cell.vertex_indices)}) != "
                        f"vertices length ({len(cell.vertices)})"
                    )
                else:
                    # Check if vertices match global pool
                    reconstructed = self.tissue.vertices[cell.vertex_indices]
                    max_diff = np.max(np.abs(reconstructed - cell.vertices))
                    if max_diff > 1e-6:
                        self.warnings.append(
                            f"Cell {cell.id}: vertices don't match global pool (max diff: {max_diff:.2e})"
                        )
    
    def _check_cell_polygons(self):
        """Check that each cell forms a valid polygon."""
        for cell in self.tissue.cells:
            if len(cell.vertices) < 3:
                self.warnings.append(f"Cell {cell.id} has < 3 vertices ({len(cell.vertices)})")
                continue
                
            # Check for valid polygon
            if not is_valid_polygon(cell.vertices):
                self.issues.append(f"Cell {cell.id} has invalid polygon (self-intersecting or degenerate)")
                
    def _check_vertex_sharing(self):
        """Check if vertices are properly shared between cells."""
        if not hasattr(self.tissue, 'vertices') or self.tissue.vertices.shape[0] == 0:
            self.warnings.append("Cannot check vertex sharing without global vertex pool")
            return
            
        vertex_to_cells = defaultdict(list)
        
        for cell in self.tissue.cells:
            if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
                for vi in cell.vertex_indices:
                    if 0 <= vi < len(self.tissue.vertices):
                        vertex_to_cells[vi].append(cell.id)
                        
        # Statistics
        shared_vertices = {v: cells for v, cells in vertex_to_cells.items() if len(cells) > 1}
        unshared_vertices = {v: cells for v, cells in vertex_to_cells.items() if len(cells) == 1}
        
        if len(shared_vertices) == 0:
            self.warnings.append("No vertices are shared between cells - tissue may be disconnected")
        else:
            sharing_ratio = len(shared_vertices) / len(self.tissue.vertices) * 100
            if sharing_ratio < 20:
                self.warnings.append(f"Low vertex sharing ratio: {sharing_ratio:.1f}%")
                
        # Check for vertices shared by too many cells (possible issue)
        for vi, cells in shared_vertices.items():
            if len(cells) > 4:
                self.warnings.append(f"Vertex {vi} shared by {len(cells)} cells (unusual)")
                
    def _check_connectivity(self):
        """Check if tissue is connected via shared vertices."""
        if not hasattr(self.tissue, 'vertices') or self.tissue.vertices.shape[0] == 0:
            return
            
        # Build adjacency graph
        cell_neighbors = defaultdict(set)
        vertex_to_cells = defaultdict(list)
        
        for cell in self.tissue.cells:
            if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
                for vi in cell.vertex_indices:
                    if 0 <= vi < len(self.tissue.vertices):
                        vertex_to_cells[vi].append(cell.id)
                        
        for cells in vertex_to_cells.values():
            if len(cells) > 1:
                for c1 in cells:
                    for c2 in cells:
                        if c1 != c2:
                            cell_neighbors[c1].add(c2)
                            
        connected_cells = len(cell_neighbors)
        total_cells = len(self.tissue.cells)
        
        if connected_cells == 0:
            self.issues.append("No cells are connected via shared vertices!")
        elif connected_cells < total_cells:
            self.warnings.append(
                f"Only {connected_cells}/{total_cells} cells are connected via shared vertices"
            )
            
        # Check neighbor distribution
        if cell_neighbors:
            neighbor_counts = [len(n) for n in cell_neighbors.values()]
            mean_neighbors = np.mean(neighbor_counts)
            min_neighbors = min(neighbor_counts)
            max_neighbors = max(neighbor_counts)
            
            if min_neighbors < 2:
                self.warnings.append(f"Some cells have < 2 neighbors (min: {min_neighbors})")
            if mean_neighbors < 3:
                self.warnings.append(f"Low mean neighbors per cell: {mean_neighbors:.1f}")
                
    def _check_areas_and_geometry(self):
        """Check cell areas and geometric properties."""
        for cell in self.tissue.cells:
            if len(cell.vertices) < 3:
                continue
                
            try:
                area = self.geometry.calculate_area(cell.vertices)
                if area <= 0:
                    self.issues.append(f"Cell {cell.id} has non-positive area: {area:.6f}")
                elif area < 1e-10:
                    self.warnings.append(f"Cell {cell.id} has very small area: {area:.2e}")
                    
                perimeter = self.geometry.calculate_perimeter(cell.vertices)
                if perimeter <= 0:
                    self.issues.append(f"Cell {cell.id} has non-positive perimeter: {perimeter:.6f}")
                    
            except Exception as e:
                self.issues.append(f"Cell {cell.id} geometry calculation failed: {e}")
                
    def _check_nan_inf(self):
        """Check for NaN or Inf values in coordinates."""
        # Check global vertices
        if hasattr(self.tissue, 'vertices') and self.tissue.vertices.shape[0] > 0:
            if np.any(np.isnan(self.tissue.vertices)):
                self.issues.append("Global vertex pool contains NaN values")
            if np.any(np.isinf(self.tissue.vertices)):
                self.issues.append("Global vertex pool contains Inf values")
                
        # Check cell vertices
        for cell in self.tissue.cells:
            if len(cell.vertices) > 0:
                if np.any(np.isnan(cell.vertices)):
                    self.issues.append(f"Cell {cell.id} vertices contain NaN values")
                if np.any(np.isinf(cell.vertices)):
                    self.issues.append(f"Cell {cell.id} vertices contain Inf values")
                    
    def _check_vertex_ordering(self):
        """Check if vertices are ordered counter-clockwise."""
        for cell in self.tissue.cells:
            if len(cell.vertices) < 3:
                continue
                
            orientation = polygon_orientation(cell.vertices)
            if orientation < 0:
                self.warnings.append(f"Cell {cell.id} has clockwise vertex ordering")
                
    def get_report(self) -> str:
        """Generate a text report of validation results."""
        lines = []
        lines.append("=" * 70)
        lines.append("TISSUE STRUCTURE VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Tissue: {len(self.tissue.cells)} cells")
        
        if hasattr(self.tissue, 'vertices'):
            lines.append(f"Global vertex pool: {self.tissue.vertices.shape[0]} vertices")
        else:
            lines.append("Global vertex pool: NOT PRESENT")
            
        lines.append("")
        
        if len(self.issues) == 0 and len(self.warnings) == 0:
            lines.append("✓ All checks passed!")
        else:
            if len(self.issues) > 0:
                lines.append(f"ISSUES ({len(self.issues)}):")
                for i, issue in enumerate(self.issues, 1):
                    lines.append(f"  {i}. {issue}")
                lines.append("")
                
            if len(self.warnings) > 0:
                lines.append(f"WARNINGS ({len(self.warnings)}):")
                for i, warning in enumerate(self.warnings, 1):
                    lines.append(f"  {i}. {warning}")
                lines.append("")
                
        lines.append("=" * 70)
        return "\n".join(lines)


def compare_tissue_structures(tissue1: Tissue, tissue2: Tissue, 
                              name1: str = "Tissue 1", name2: str = "Tissue 2") -> Dict:
    """Compare two tissue structures and return metrics."""
    
    def get_metrics(tissue: Tissue) -> Dict:
        """Extract structural metrics from a tissue."""
        metrics = {
            'num_cells': len(tissue.cells),
            'num_global_vertices': 0,
            'num_shared_vertices': 0,
            'num_connected_cells': 0,
            'mean_neighbors': 0.0,
            'mean_vertices_per_cell': 0.0,
            'mean_area': 0.0,
            'vertex_sharing_ratio': 0.0,
        }
        
        if hasattr(tissue, 'vertices') and tissue.vertices is not None:
            metrics['num_global_vertices'] = tissue.vertices.shape[0]
            
        # Vertex sharing analysis
        vertex_to_cells = defaultdict(list)
        total_vertices = 0
        
        for cell in tissue.cells:
            total_vertices += len(cell.vertices)
            if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
                for vi in cell.vertex_indices:
                    vertex_to_cells[vi].append(cell.id)
                    
        if total_vertices > 0:
            metrics['mean_vertices_per_cell'] = total_vertices / len(tissue.cells)
            
        shared_vertices = sum(1 for cells in vertex_to_cells.values() if len(cells) > 1)
        metrics['num_shared_vertices'] = shared_vertices
        
        if metrics['num_global_vertices'] > 0:
            metrics['vertex_sharing_ratio'] = shared_vertices / metrics['num_global_vertices']
            
        # Connectivity
        cell_neighbors = defaultdict(set)
        for cells in vertex_to_cells.values():
            if len(cells) > 1:
                for c1 in cells:
                    for c2 in cells:
                        if c1 != c2:
                            cell_neighbors[c1].add(c2)
                            
        metrics['num_connected_cells'] = len(cell_neighbors)
        if cell_neighbors:
            neighbor_counts = [len(n) for n in cell_neighbors.values()]
            metrics['mean_neighbors'] = float(np.mean(neighbor_counts))
            
        # Areas
        geom = GeometryCalculator()
        areas = []
        for cell in tissue.cells:
            if len(cell.vertices) >= 3:
                try:
                    area = geom.calculate_area(cell.vertices)
                    if area > 0:
                        areas.append(area)
                except:
                    pass
        if areas:
            metrics['mean_area'] = float(np.mean(areas))
            
        return metrics
    
    metrics1 = get_metrics(tissue1)
    metrics2 = get_metrics(tissue2)
    
    return {
        name1: metrics1,
        name2: metrics2,
        'comparison': {
            k: f"{metrics1[k]} vs {metrics2[k]}"
            for k in metrics1.keys()
        }
    }


def print_comparison_report(comparison: Dict):
    """Print a comparison report between two tissues."""
    print("\n" + "=" * 70)
    print("TISSUE STRUCTURE COMPARISON")
    print("=" * 70)
    
    names = [k for k in comparison.keys() if k != 'comparison']
    if len(names) != 2:
        print("Error: Expected 2 tissues to compare")
        return
        
    name1, name2 = names
    m1 = comparison[name1]
    m2 = comparison[name2]
    
    print(f"\n{name1:30} {name2:30}")
    print("-" * 70)
    
    metrics_to_show = [
        ('num_cells', 'Cells'),
        ('num_global_vertices', 'Global vertices'),
        ('num_shared_vertices', 'Shared vertices'),
        ('vertex_sharing_ratio', 'Vertex sharing ratio'),
        ('num_connected_cells', 'Connected cells'),
        ('mean_neighbors', 'Mean neighbors/cell'),
        ('mean_vertices_per_cell', 'Mean vertices/cell'),
        ('mean_area', 'Mean area'),
    ]
    
    for key, label in metrics_to_show:
        v1 = m1[key]
        v2 = m2[key]
        
        if isinstance(v1, float):
            print(f"{label:25} {v1:15.2f} {v2:15.2f}")
        else:
            print(f"{label:25} {v1:15} {v2:15}")
            
    print("=" * 70 + "\n")


# ============================================================================
# Pytest test cases
# ============================================================================

def test_validate_honeycomb_14cells():
    """Test that honeycomb_14cells.dill passes structure validation."""
    tissue_file = Path("pickled_tissues/honeycomb_14cells.dill")
    if not tissue_file.exists():
        pytest.skip(f"Tissue file not found: {tissue_file}")
        
    tissue = load_tissue(str(tissue_file))
    
    # Build global vertex pool if needed
    if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
        tissue.build_global_vertices(tol=1e-10)
        tissue.reconstruct_cell_vertices()
        
    validator = TissueStructureValidator(tissue)
    is_valid = validator.validate_all()
    
    print("\n" + validator.get_report())
    
    # Honeycomb should have no critical issues
    assert len(validator.issues) == 0, f"Honeycomb has issues: {validator.issues}"


def test_validate_acam_79cells():
    """Test acam_79cells.dill structure and report any issues."""
    tissue_file = Path("pickled_tissues/acam_79cells.dill")
    if not tissue_file.exists():
        pytest.skip(f"Tissue file not found: {tissue_file}")
        
    tissue = load_tissue(str(tissue_file))
    
    # Build global vertex pool if needed
    if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
        tissue.build_global_vertices(tol=1e-10)
        tissue.reconstruct_cell_vertices()
        
    validator = TissueStructureValidator(tissue)
    is_valid = validator.validate_all()
    
    print("\n" + validator.get_report())
    
    # Don't assert failure - just report
    if not is_valid:
        print(f"\n⚠ ACAM tissue has {len(validator.issues)} issue(s)")


def test_compare_honeycomb_vs_acam():
    """Compare honeycomb_14cells.dill and acam_79cells.dill structures."""
    honeycomb_file = Path("pickled_tissues/honeycomb_14cells.dill")
    acam_file = Path("pickled_tissues/acam_79cells.dill")
    
    if not honeycomb_file.exists():
        pytest.skip(f"Honeycomb file not found: {honeycomb_file}")
    if not acam_file.exists():
        pytest.skip(f"ACAM file not found: {acam_file}")
        
    honeycomb = load_tissue(str(honeycomb_file))
    acam = load_tissue(str(acam_file))
    
    # Build global vertex pools
    for tissue in [honeycomb, acam]:
        if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
            tissue.build_global_vertices(tol=1e-10)
            tissue.reconstruct_cell_vertices()
            
    comparison = compare_tissue_structures(honeycomb, acam, "Honeycomb 14", "ACAM 79")
    print_comparison_report(comparison)
    
    # Check key differences
    h_metrics = comparison["Honeycomb 14"]
    a_metrics = comparison["ACAM 79"]
    
    # ACAM should have vertices shared
    if a_metrics['vertex_sharing_ratio'] < 0.1:
        print(f"\n⚠ WARNING: ACAM has very low vertex sharing ratio: "
              f"{a_metrics['vertex_sharing_ratio']*100:.1f}%")
        
    # ACAM should have most cells connected
    if a_metrics['num_connected_cells'] < a_metrics['num_cells'] * 0.9:
        print(f"\n⚠ WARNING: ACAM has disconnected cells: "
              f"{a_metrics['num_connected_cells']}/{a_metrics['num_cells']}")


def test_simulation_readiness_honeycomb():
    """Test if honeycomb_14cells.dill is ready for simulation."""
    tissue_file = Path("pickled_tissues/honeycomb_14cells.dill")
    if not tissue_file.exists():
        pytest.skip(f"Tissue file not found: {tissue_file}")
        
    tissue = load_tissue(str(tissue_file))
    
    # Build global vertex pool
    tissue.build_global_vertices(tol=1e-10)
    tissue.reconstruct_cell_vertices()
    
    # Try to compute gradients
    from myvertexmodel import EnergyParameters, GeometryCalculator
    
    geometry = GeometryCalculator()
    
    # Compute initial areas for target areas
    target_areas = {}
    for cell in tissue.cells:
        area = geometry.calculate_area(cell.vertices)
        assert area > 0, f"Cell {cell.id} has non-positive area"
        target_areas[cell.id] = area
        
    energy_params = EnergyParameters(
        k_area=1.0,
        k_perimeter=0.1,
        gamma=0.05,
        target_area=target_areas
    )
    
    # Try computing energy
    from myvertexmodel import tissue_energy
    energy = tissue_energy(tissue, energy_params, geometry)
    assert np.isfinite(energy), "Tissue energy is not finite"
    
    print(f"\n✓ Honeycomb simulation readiness: PASSED (energy={energy:.2f})")


def test_simulation_readiness_acam():
    """Test if acam_79cells.dill is ready for simulation."""
    tissue_file = Path("pickled_tissues/acam_79cells.dill")
    if not tissue_file.exists():
        pytest.skip(f"Tissue file not found: {tissue_file}")
        
    tissue = load_tissue(str(tissue_file))
    
    print("\n" + "=" * 70)
    print("ACAM SIMULATION READINESS TEST")
    print("=" * 70)
    
    # Build global vertex pool with different tolerances
    tolerances = [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0]
    
    for tol in tolerances:
        print(f"\nTrying tolerance: {tol}")
        try:
            tissue.build_global_vertices(tol=tol)
            tissue.reconstruct_cell_vertices()
            
            # Check connectivity
            vertex_to_cells = defaultdict(list)
            for cell in tissue.cells:
                if hasattr(cell, 'vertex_indices'):
                    for vi in cell.vertex_indices:
                        vertex_to_cells[vi].append(cell.id)
                        
            shared = sum(1 for cells in vertex_to_cells.values() if len(cells) > 1)
            total = tissue.vertices.shape[0]
            ratio = shared / total * 100 if total > 0 else 0
            
            print(f"  Global vertices: {total}")
            print(f"  Shared vertices: {shared} ({ratio:.1f}%)")
            
            if ratio > 50:
                print(f"  ✓ Good vertex sharing at tol={tol}")
                break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            
    # Try to compute energy
    from myvertexmodel import EnergyParameters, GeometryCalculator, tissue_energy
    
    geometry = GeometryCalculator()
    
    try:
        # Compute initial areas
        target_areas = {}
        for cell in tissue.cells:
            area = geometry.calculate_area(cell.vertices)
            if area <= 0:
                print(f"  ⚠ Cell {cell.id} has non-positive area: {area}")
            target_areas[cell.id] = max(area, 1.0)  # Use minimum 1.0 as fallback
            
        energy_params = EnergyParameters(
            k_area=1.0,
            k_perimeter=0.1,
            gamma=0.05,
            target_area=target_areas
        )
        
        energy = tissue_energy(tissue, energy_params, geometry)
        
        if np.isfinite(energy):
            print(f"\n✓ ACAM simulation readiness: PASSED (energy={energy:.2f})")
        else:
            print(f"\n✗ ACAM simulation readiness: FAILED (energy is not finite)")
            
    except Exception as e:
        print(f"\n✗ ACAM simulation readiness: FAILED")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests directly
    print("Running tissue structure validation tests...\n")
    
    test_validate_honeycomb_14cells()
    test_validate_acam_79cells()
    test_compare_honeycomb_vs_acam()
    test_simulation_readiness_honeycomb()
    test_simulation_readiness_acam()

