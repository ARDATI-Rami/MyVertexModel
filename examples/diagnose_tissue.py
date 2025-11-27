"""diagnose_tissue.py

Diagnostic tool for analyzing tissue structure and comparing against reference tissues.

Usage:
    # Diagnose a single tissue
    python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill
    
    # Compare against a reference tissue
    python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill --reference pickled_tissues/honeycomb_14cells.dill
    
    # Visualize the tissue
    python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill --plot
    
    # Save diagnostic report to file
    python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill --output acam_diagnostic.txt
    
    # Try to auto-repair common issues
    python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill --repair --save-repaired acam_79cells_fixed.dill
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
from collections import defaultdict
import numpy as np

from myvertexmodel import Tissue, load_tissue, save_tissue, GeometryCalculator
from myvertexmodel.geometry import is_valid_polygon, polygon_orientation, ensure_ccw


def analyze_tissue_structure(tissue: Tissue, name: str = "Tissue") -> dict:
    """Perform comprehensive structure analysis of a tissue."""
    
    print(f"\n{'=' * 70}")
    print(f"ANALYZING: {name}")
    print(f"{'=' * 70}\n")
    
    results = {
        'name': name,
        'num_cells': len(tissue.cells),
        'issues': [],
        'warnings': [],
        'metrics': {}
    }
    
    # Basic structure
    print(f"Basic Structure:")
    print(f"  Cells: {len(tissue.cells)}")
    
    if hasattr(tissue, 'vertices') and tissue.vertices is not None:
        print(f"  Global vertices: {tissue.vertices.shape[0]}")
        results['metrics']['global_vertices'] = tissue.vertices.shape[0]
    else:
        print(f"  Global vertices: NOT PRESENT")
        results['warnings'].append("No global vertex pool")
        results['metrics']['global_vertices'] = 0
        
    # Check for basic attributes
    for i, cell in enumerate(tissue.cells):
        if not hasattr(cell, 'id'):
            results['issues'].append(f"Cell {i} missing 'id' attribute")
        if not hasattr(cell, 'vertices'):
            results['issues'].append(f"Cell {i} missing 'vertices' attribute")
        if not hasattr(cell, 'vertex_indices'):
            results['warnings'].append(f"Cell {cell.id if hasattr(cell, 'id') else i} missing 'vertex_indices'")
            
    # Analyze vertex sharing
    print(f"\nVertex Connectivity:")
    
    vertex_to_cells = defaultdict(list)
    total_vertex_refs = 0
    
    for cell in tissue.cells:
        total_vertex_refs += len(cell.vertices)
        if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
            for vi in cell.vertex_indices:
                vertex_to_cells[vi].append(cell.id)
                
    shared_vertices = {v: cells for v, cells in vertex_to_cells.items() if len(cells) > 1}
    unshared_vertices = {v: cells for v, cells in vertex_to_cells.items() if len(cells) == 1}
    
    print(f"  Total vertex references: {total_vertex_refs}")
    print(f"  Unique vertices in use: {len(vertex_to_cells)}")
    print(f"  Shared vertices (2+ cells): {len(shared_vertices)}")
    print(f"  Unshared vertices: {len(unshared_vertices)}")
    
    if results['metrics']['global_vertices'] > 0:
        sharing_ratio = len(shared_vertices) / results['metrics']['global_vertices'] * 100
        print(f"  Vertex sharing ratio: {sharing_ratio:.1f}%")
        results['metrics']['sharing_ratio'] = sharing_ratio
        
        if sharing_ratio < 20:
            results['warnings'].append(f"Low vertex sharing ratio: {sharing_ratio:.1f}%")
    
    # Vertex sharing distribution
    sharing_dist = defaultdict(int)
    for cells in shared_vertices.values():
        sharing_dist[len(cells)] += 1
        
    if sharing_dist:
        print(f"  Sharing distribution:")
        for num_cells in sorted(sharing_dist.keys(), reverse=True):
            count = sharing_dist[num_cells]
            print(f"    {count} vertices shared by {num_cells} cells")
            
    # Cell connectivity
    print(f"\nCell Connectivity:")
    
    cell_neighbors = defaultdict(set)
    for cells in shared_vertices.values():
        for c1 in cells:
            for c2 in cells:
                if c1 != c2:
                    cell_neighbors[c1].add(c2)
                    
    connected_cells = len(cell_neighbors)
    print(f"  Connected cells: {connected_cells}/{len(tissue.cells)}")
    results['metrics']['connected_cells'] = connected_cells
    
    if connected_cells == 0:
        results['issues'].append("No cells are connected via shared vertices!")
    elif connected_cells < len(tissue.cells):
        disconnected = len(tissue.cells) - connected_cells
        results['warnings'].append(f"{disconnected} cells are disconnected")
        
    if cell_neighbors:
        neighbor_counts = [len(n) for n in cell_neighbors.values()]
        mean_neighbors = np.mean(neighbor_counts)
        min_neighbors = min(neighbor_counts)
        max_neighbors = max(neighbor_counts)
        
        print(f"  Mean neighbors per cell: {mean_neighbors:.1f}")
        print(f"  Neighbor range: {min_neighbors} - {max_neighbors}")
        results['metrics']['mean_neighbors'] = mean_neighbors
        
        if min_neighbors < 2:
            results['warnings'].append(f"Some cells have < 2 neighbors (min: {min_neighbors})")
        if mean_neighbors < 3:
            results['warnings'].append(f"Low mean neighbors: {mean_neighbors:.1f}")
            
    # Check cell geometry
    print(f"\nCell Geometry:")
    
    geom = GeometryCalculator()
    areas = []
    perimeters = []
    vertex_counts = []
    invalid_cells = []
    ccw_violations = []
    
    for cell in tissue.cells:
        vertex_counts.append(len(cell.vertices))
        
        if len(cell.vertices) < 3:
            results['warnings'].append(f"Cell {cell.id} has < 3 vertices")
            continue
            
        # Check polygon validity
        if not is_valid_polygon(cell.vertices):
            invalid_cells.append(cell.id)
            results['issues'].append(f"Cell {cell.id} has invalid polygon")
            continue
            
        # Check orientation
        if polygon_orientation(cell.vertices) < 0:
            ccw_violations.append(cell.id)
            
        # Calculate metrics
        try:
            area = geom.calculate_area(cell.vertices)
            perimeter = geom.calculate_perimeter(cell.vertices)
            
            if area <= 0:
                results['issues'].append(f"Cell {cell.id} has non-positive area: {area:.6f}")
            else:
                areas.append(area)
                perimeters.append(perimeter)
                
        except Exception as e:
            results['issues'].append(f"Cell {cell.id} geometry calculation failed: {e}")
            
    if vertex_counts:
        print(f"  Mean vertices per cell: {np.mean(vertex_counts):.1f}")
        print(f"  Vertex count range: {min(vertex_counts)} - {max(vertex_counts)}")
        
    if areas:
        print(f"  Mean area: {np.mean(areas):.2f}")
        print(f"  Area range: {min(areas):.2f} - {max(areas):.2f}")
        results['metrics']['mean_area'] = np.mean(areas)
        
    if invalid_cells:
        print(f"  ⚠ Invalid polygons: {len(invalid_cells)} cells")
        
    if ccw_violations:
        print(f"  ⚠ CCW violations: {len(ccw_violations)} cells")
        results['warnings'].append(f"{len(ccw_violations)} cells have clockwise ordering")
        
    # Check for NaN/Inf
    print(f"\nData Quality:")
    
    nan_inf_issues = []
    if hasattr(tissue, 'vertices') and tissue.vertices.shape[0] > 0:
        if np.any(np.isnan(tissue.vertices)):
            nan_inf_issues.append("Global vertices contain NaN")
            results['issues'].append("Global vertices contain NaN")
        if np.any(np.isinf(tissue.vertices)):
            nan_inf_issues.append("Global vertices contain Inf")
            results['issues'].append("Global vertices contain Inf")
            
    for cell in tissue.cells:
        if len(cell.vertices) > 0:
            if np.any(np.isnan(cell.vertices)):
                nan_inf_issues.append(f"Cell {cell.id} vertices contain NaN")
                results['issues'].append(f"Cell {cell.id} vertices contain NaN")
            if np.any(np.isinf(cell.vertices)):
                nan_inf_issues.append(f"Cell {cell.id} vertices contain Inf")
                results['issues'].append(f"Cell {cell.id} vertices contain Inf")
                
    if nan_inf_issues:
        print(f"  ⚠ Found {len(nan_inf_issues)} NaN/Inf issues")
        for issue in nan_inf_issues[:5]:  # Show first 5
            print(f"    - {issue}")
        if len(nan_inf_issues) > 5:
            print(f"    ... and {len(nan_inf_issues) - 5} more")
    else:
        print(f"  ✓ No NaN/Inf values detected")
        
    # Summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    
    if len(results['issues']) == 0 and len(results['warnings']) == 0:
        print(f"✓ ALL CHECKS PASSED - Tissue appears healthy!")
    else:
        if results['issues']:
            print(f"\n❌ CRITICAL ISSUES ({len(results['issues'])}):")
            for issue in results['issues']:
                print(f"  - {issue}")
                
        if results['warnings']:
            print(f"\n⚠ WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings']:
                print(f"  - {warning}")
                
    print(f"{'=' * 70}\n")
    
    return results


def compare_tissues(tissue1: Tissue, tissue2: Tissue, name1: str, name2: str):
    """Compare two tissues side-by-side."""
    
    print(f"\n{'=' * 70}")
    print(f"TISSUE COMPARISON: {name1} vs {name2}")
    print(f"{'=' * 70}\n")
    
    def get_metrics(tissue):
        metrics = {}
        metrics['cells'] = len(tissue.cells)
        metrics['global_vertices'] = tissue.vertices.shape[0] if hasattr(tissue, 'vertices') else 0
        
        vertex_to_cells = defaultdict(list)
        for cell in tissue.cells:
            if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
                for vi in cell.vertex_indices:
                    vertex_to_cells[vi].append(cell.id)
                    
        shared = sum(1 for cells in vertex_to_cells.values() if len(cells) > 1)
        metrics['shared_vertices'] = shared
        metrics['sharing_ratio'] = (shared / metrics['global_vertices'] * 100) if metrics['global_vertices'] > 0 else 0
        
        cell_neighbors = defaultdict(set)
        for cells in vertex_to_cells.values():
            if len(cells) > 1:
                for c1 in cells:
                    for c2 in cells:
                        if c1 != c2:
                            cell_neighbors[c1].add(c2)
                            
        metrics['connected_cells'] = len(cell_neighbors)
        metrics['mean_neighbors'] = np.mean([len(n) for n in cell_neighbors.values()]) if cell_neighbors else 0
        
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
        metrics['mean_area'] = np.mean(areas) if areas else 0
        
        return metrics
        
    m1 = get_metrics(tissue1)
    m2 = get_metrics(tissue2)
    
    print(f"{'Metric':<30} {name1:<20} {name2:<20}")
    print("-" * 70)
    print(f"{'Cells':<30} {m1['cells']:<20} {m2['cells']:<20}")
    print(f"{'Global vertices':<30} {m1['global_vertices']:<20} {m2['global_vertices']:<20}")
    print(f"{'Shared vertices':<30} {m1['shared_vertices']:<20} {m2['shared_vertices']:<20}")
    print(f"{'Vertex sharing ratio':<30} {m1['sharing_ratio']:<20.1f} {m2['sharing_ratio']:<20.1f}")
    print(f"{'Connected cells':<30} {m1['connected_cells']:<20} {m2['connected_cells']:<20}")
    print(f"{'Mean neighbors/cell':<30} {m1['mean_neighbors']:<20.2f} {m2['mean_neighbors']:<20.2f}")
    print(f"{'Mean area':<30} {m1['mean_area']:<20.2f} {m2['mean_area']:<20.2f}")
    
    print(f"\n{'=' * 70}\n")


def attempt_repair(tissue: Tissue) -> Tissue:
    """Attempt to repair common tissue issues."""
    
    print(f"\n{'=' * 70}")
    print(f"ATTEMPTING TISSUE REPAIR")
    print(f"{'=' * 70}\n")
    
    # 1. Rebuild global vertex pool with adaptive tolerance
    print("Step 1: Rebuilding global vertex pool...")
    
    tolerances = [1e-10, 1e-8, 1e-6, 1e-4, 0.01, 0.1, 1.0, 5.0]
    best_tol = 1e-10
    best_sharing_ratio = 0
    
    for tol in tolerances:
        tissue.build_global_vertices(tol=tol)
        tissue.reconstruct_cell_vertices()
        
        vertex_to_cells = defaultdict(list)
        for cell in tissue.cells:
            if hasattr(cell, 'vertex_indices'):
                for vi in cell.vertex_indices:
                    vertex_to_cells[vi].append(cell.id)
                    
        shared = sum(1 for cells in vertex_to_cells.values() if len(cells) > 1)
        total = tissue.vertices.shape[0]
        ratio = (shared / total * 100) if total > 0 else 0
        
        print(f"  Tolerance {tol:8.2e}: {total} vertices, {shared} shared ({ratio:.1f}%)")
        
        if ratio > best_sharing_ratio:
            best_sharing_ratio = ratio
            best_tol = tol
            
    print(f"  → Using tolerance {best_tol} (sharing ratio: {best_sharing_ratio:.1f}%)")
    tissue.build_global_vertices(tol=best_tol)
    tissue.reconstruct_cell_vertices()
    
    # 2. Remove duplicate consecutive vertices
    print("\nStep 2: Removing duplicate consecutive vertices...")

    dup_count = 0
    for cell in tissue.cells:
        if not hasattr(cell, 'vertex_indices') or len(cell.vertex_indices) == 0:
            continue

        # Check for duplicate consecutive indices
        indices = cell.vertex_indices
        has_duplicates = False

        for i in range(len(indices) - 1):
            if indices[i] == indices[i+1]:
                has_duplicates = True
                break

        if has_duplicates:
            # Remove consecutive duplicates
            unique_indices = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] != indices[i-1]:
                    unique_indices.append(indices[i])

            # Update vertex_indices
            cell.vertex_indices = np.array(unique_indices, dtype=int)

            # Reconstruct vertices from global pool
            cell.vertices = tissue.vertices[cell.vertex_indices].copy()

            dup_count += 1

    print(f"  → Fixed {dup_count} cells with duplicate vertices")

    # 3. Ensure counter-clockwise ordering
    print("\nStep 3: Ensuring CCW vertex ordering...")

    fixed_count = 0
    for cell in tissue.cells:
        if len(cell.vertices) >= 3:
            if polygon_orientation(cell.vertices) < 0:
                cell.vertices = ensure_ccw(cell.vertices)
                fixed_count += 1
                
    print(f"  → Fixed {fixed_count} cells with clockwise ordering")
    
    # 4. Rebuild global pool again after fixes
    if dup_count > 0 or fixed_count > 0:
        print("\nStep 4: Rebuilding global pool after fixes...")
        tissue.build_global_vertices(tol=best_tol)
        tissue.reconstruct_cell_vertices()
        print(f"  → Done")
        
    print(f"\n{'=' * 70}")
    print("REPAIR COMPLETE")
    print(f"{'=' * 70}\n")
    
    return tissue


def plot_tissue_diagnostic(tissue: Tissue, title: str = "Tissue"):
    """Plot tissue with diagnostic overlays."""
    try:
        import matplotlib.pyplot as plt
        from myvertexmodel import plot_tissue
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Basic tissue plot
        ax = axes[0]
        plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=0.3)
        ax.set_title(f"{title} - Structure", fontsize=14, fontweight='bold')
        
        # Right: Connectivity visualization
        ax = axes[1]
        plot_tissue(tissue, ax=ax, show_vertices=False, fill=False, alpha=0.5)
        
        # Highlight shared vertices
        vertex_to_cells = defaultdict(list)
        for cell in tissue.cells:
            if hasattr(cell, 'vertex_indices'):
                for vi in cell.vertex_indices:
                    vertex_to_cells[vi].append(cell.id)
                    
        # Plot shared vertices in red, unshared in blue
        for vi, cells in vertex_to_cells.items():
            if vi < len(tissue.vertices):
                v = tissue.vertices[vi]
                color = 'red' if len(cells) > 1 else 'blue'
                size = 50 if len(cells) > 1 else 20
                ax.scatter(v[0], v[1], c=color, s=size, zorder=10, alpha=0.7)
                
        ax.set_title(f"{title} - Vertex Sharing (Red=Shared, Blue=Unshared)", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{title.replace(' ', '_')}_diagnostic.png", dpi=150, bbox_inches='tight')
        print(f"\nSaved diagnostic plot to {title.replace(' ', '_')}_diagnostic.png")
        plt.show()
        
    except Exception as e:
        print(f"Plot failed: {e}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose tissue structure issues")
    parser.add_argument("tissue_file", type=str, help="Path to tissue file to diagnose")
    parser.add_argument("--reference", type=str, help="Reference tissue file for comparison")
    parser.add_argument("--plot", action="store_true", help="Show diagnostic plots")
    parser.add_argument("--output", type=str, help="Save report to file")
    parser.add_argument("--repair", action="store_true", help="Attempt to repair common issues")
    parser.add_argument("--save-repaired", type=str, help="Save repaired tissue to file")
    
    args = parser.parse_args(argv)
    
    # Load tissue
    tissue_path = Path(args.tissue_file)
    if not tissue_path.exists():
        tissue_path = tissue_path.with_suffix(".dill")
        if not tissue_path.exists():
            print(f"Error: Tissue file not found: {args.tissue_file}")
            return 1
            
    print(f"Loading tissue from: {tissue_path}")
    tissue = load_tissue(str(tissue_path))
    
    # Build global vertex pool if needed
    if not hasattr(tissue, 'vertices') or tissue.vertices.shape[0] == 0:
        print("Building global vertex pool...")
        tissue.build_global_vertices(tol=1e-10)
        tissue.reconstruct_cell_vertices()
        
    # Analyze structure
    results = analyze_tissue_structure(tissue, tissue_path.stem)
    
    # Compare with reference if provided
    if args.reference:
        ref_path = Path(args.reference)
        if not ref_path.exists():
            ref_path = ref_path.with_suffix(".dill")
            
        if ref_path.exists():
            print(f"\nLoading reference tissue from: {ref_path}")
            ref_tissue = load_tissue(str(ref_path))
            
            if not hasattr(ref_tissue, 'vertices') or ref_tissue.vertices.shape[0] == 0:
                ref_tissue.build_global_vertices(tol=1e-10)
                ref_tissue.reconstruct_cell_vertices()
                
            compare_tissues(tissue, ref_tissue, tissue_path.stem, ref_path.stem)
        else:
            print(f"Warning: Reference file not found: {args.reference}")
            
    # Repair if requested
    if args.repair:
        tissue = attempt_repair(tissue)
        
        # Re-analyze after repair
        print("\nRe-analyzing after repair...")
        results = analyze_tissue_structure(tissue, tissue_path.stem + " (repaired)")
        
        if args.save_repaired:
            save_path = Path(args.save_repaired)
            save_tissue(tissue, str(save_path))
            print(f"\nSaved repaired tissue to: {save_path}")
            
    # Plot if requested
    if args.plot:
        plot_tissue_diagnostic(tissue, tissue_path.stem)
        
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        # This would require capturing print output - simplified for now
        print(f"\nNote: Full report output to file not yet implemented")
        print(f"      Use shell redirection: python {sys.argv[0]} ... > {args.output}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

