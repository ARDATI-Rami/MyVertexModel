"""simulate_cell_growth.py

Simulate growth of a single cell in a vertex model tissue until it reaches double its initial area.

The simulation:
1. Builds or loads a tissue (honeycomb built on-the-fly by default, or load from file)
2. Selects one cell as the "growing cell" by cell ID
3. Gradually increases the target area of that cell from A_initial to 2*A_initial
4. Allows the tissue to mechanically equilibrate as the growing cell expands
5. Tracks the growing cell's actual area vs target area throughout

Usage (default - builds honeycomb 14 cells):
    python examples/simulate_cell_growth.py --growing-cell-id 7 --plot

Usage (build honeycomb 19 cells):
    python examples/simulate_cell_growth.py --build-honeycomb 19 --growing-cell-id 10 --plot

Usage (load from file):
    python examples/simulate_cell_growth.py --tissue-file pickled_tissues/YOUR_TISSUE --growing-cell-id 7 --plot

Options:
    --tissue-file           Path to tissue file (optional, builds honeycomb if not specified)
    --build-honeycomb       Build honeycomb: '14' for 2-3-4-3-2 (default), '19' for 3-4-5-4-3
    --growing-cell-id       Cell ID to grow (default: 7)
    --growth-steps          Number of steps over which to ramp target area (default: 100)
    --total-steps           Total simulation steps (default: 200)
    --dt                    Time step size (default: 0.01)
    --plot                  Show initial and final tissue plots
    --log-interval          Print progress every N steps (default: 10)
    --save-csv              Save growth tracking data to CSV file
    --k-area                Area elasticity coefficient (default: 1.0)
    --k-perimeter           Perimeter coefficient (default: 0.1)
    --gamma                 Line tension (default: 0.05)
"""
from __future__ import annotations
import argparse
import sys
from typing import Optional, List
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import random
import numpy as np
from myvertexmodel import Tissue, Cell, Simulation, EnergyParameters, GeometryCalculator, plot_tissue
from myvertexmodel import load_tissue

def compute_global_gradient(tissue: Tissue, energy_params: EnergyParameters,
                           geometry: GeometryCalculator, epsilon: float = 1e-6) -> np.ndarray:
    """Compute energy gradient with respect to global vertex positions.

    This ensures that shared vertices move consistently across all cells.

    Args:
        tissue: Tissue with global vertex pool (tissue.vertices populated)
        energy_params: Energy parameters
        geometry: Geometry calculator
        epsilon: Finite difference step size

    Returns:
        Gradient array of shape (M, 2) for M global vertices
    """
    from myvertexmodel import tissue_energy

    if tissue.vertices.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    gradient = np.zeros_like(tissue.vertices)

    # Compute gradient for each global vertex
    for i in range(tissue.vertices.shape[0]):
        # X gradient
        original_x = tissue.vertices[i, 0]
        tissue.vertices[i, 0] = original_x + epsilon
        tissue.reconstruct_cell_vertices()  # Update cell.vertices from global pool
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = original_x - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = original_x  # Restore
        gradient[i, 0] = (e_plus - e_minus) / (2 * epsilon)

        # Y gradient
        original_y = tissue.vertices[i, 1]
        tissue.vertices[i, 1] = original_y + epsilon
        tissue.reconstruct_cell_vertices()
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = original_y - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = original_y  # Restore
        gradient[i, 1] = (e_plus - e_minus) / (2 * epsilon)

    # Final reconstruction to ensure consistency
    tissue.reconstruct_cell_vertices()

    return gradient


def analyze_shared_vertices(tissue: Tissue, tolerance: float = 1e-8) -> dict:
    """Analyze which vertices are shared between cells using global vertex pool.

    Args:
        tissue: Tissue to analyze (must have global vertex pool)
        tolerance: Not used (kept for compatibility)

    Returns:
        Dictionary mapping vertex index to list of cell IDs sharing that vertex
    """
    from collections import defaultdict

    vertex_to_cells = defaultdict(list)

    # Use vertex_indices to track sharing via global vertex pool
    for cell in tissue.cells:
        if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
            for vi in cell.vertex_indices:
                vertex_to_cells[vi].append(cell.id)
        else:
            # Fallback: analyze local vertices (less accurate)
            for vertex in cell.vertices:
                vertex_key = (round(vertex[0] / tolerance) * tolerance,
                             round(vertex[1] / tolerance) * tolerance)
                vertex_to_cells[vertex_key].append(cell.id)

    # Keep only vertices that are shared (appear in multiple cells)
    shared_vertices = {v: cells for v, cells in vertex_to_cells.items() if len(cells) > 1}

    return shared_vertices


def print_shared_vertices_report(tissue: Tissue, tolerance: float = 1e-8):
    """Print a detailed report of shared vertices between cells using global vertex pool.

    Args:
        tissue: Tissue to analyze (must have global vertex pool)
        tolerance: Not used (kept for compatibility)
    """
    print("\n" + "="*70)
    print("SHARED VERTICES ANALYSIS REPORT (Global Vertex Pool)")
    print("="*70)

    shared_vertices = analyze_shared_vertices(tissue, tolerance)

    # Summary statistics using global vertex pool
    if hasattr(tissue, 'vertices') and tissue.vertices is not None:
        total_global_vertices = len(tissue.vertices)
        total_local_refs = sum(len(c.vertices) for c in tissue.cells)
    else:
        # Fallback if no global pool
        total_global_vertices = len(set(shared_vertices.keys()))
        total_local_refs = sum(len(c.vertices) for c in tissue.cells)

    num_shared = len(shared_vertices)

    print(f"\nSummary:")
    print(f"  Total vertex references: {total_local_refs}")
    print(f"  Global vertex pool size: {total_global_vertices}")
    print(f"  Shared vertices (used by 2+ cells): {num_shared}")
    if total_global_vertices > 0:
        print(f"  Vertex sharing ratio: {num_shared / total_global_vertices * 100:.1f}%")

    # Group shared vertices by number of cells sharing them
    sharing_groups = defaultdict(list)
    for vertex, cells in shared_vertices.items():
        sharing_groups[len(cells)].append((vertex, cells))

    if sharing_groups:
        print(f"\nSharing distribution:")
        for num_cells in sorted(sharing_groups.keys(), reverse=True):
            vertices_list = sharing_groups[num_cells]
            print(f"  Vertices shared by {num_cells} cells: {len(vertices_list)}")

    # Cell neighbor analysis
    cell_neighbors = defaultdict(set)
    for vertex, cells in shared_vertices.items():
        for cell1 in cells:
            for cell2 in cells:
                if cell1 != cell2:
                    cell_neighbors[cell1].add(cell2)

    if cell_neighbors:
        neighbor_counts = [len(n) for n in cell_neighbors.values()]
        print(f"\nCell connectivity:")
        print(f"  Connected cells: {len(cell_neighbors)}/{len(tissue.cells)}")
        print(f"  Mean neighbors per cell: {np.mean(neighbor_counts):.1f}")
        print(f"  Neighbor range: {min(neighbor_counts)}-{max(neighbor_counts)}")
    else:
        print(f"\n⚠ WARNING: No cells are connected via shared vertices!")

    print("="*70 + "\n")


def _connectivity_stats(tissue: Tissue) -> tuple[int, int, int, float]:
    """Compute connectivity stats: (global_vertices, shared_vertices, connected_cells, mean_neighbors)."""
    from collections import defaultdict
    global_vertices = tissue.vertices.shape[0]
    vertex_to_cells = defaultdict(list)
    for cell in tissue.cells:
        if hasattr(cell, 'vertex_indices') and cell.vertex_indices is not None:
            for vi in cell.vertex_indices:
                vertex_to_cells[vi].append(cell.id)
    shared_vertices = sum(1 for cells in vertex_to_cells.values() if len(cells) > 1)
    cell_neighbors = defaultdict(set)
    for cells in vertex_to_cells.values():
        if len(cells) > 1:
            for c1 in cells:
                for c2 in cells:
                    if c1 != c2:
                        cell_neighbors[c1].add(c2)
    connected_cells = len(cell_neighbors)
    mean_neighbors = float(np.mean([len(n) for n in cell_neighbors.values()])) if cell_neighbors else 0.0
    return global_vertices, shared_vertices, connected_cells, mean_neighbors


def stitch_global_vertices(tissue: Tissue, tol_max: float = 5.0, steps: int = 10, max_passes: int = 3) -> None:
    """Iteratively rebuild the global vertex pool sweeping tolerance from 1..tol_max until convergence.

    For up to max_passes, sweep tol values linspace(1.0, tol_max, steps). At each tol:
    - tissue.build_global_vertices(tol)
    - tissue.reconstruct_cell_vertices()
    Stop early if an entire pass yields no changes in global vertex count, shared-vertex count,
    or number of connected cells.
    """
    tol_min = 1.0
    schedule = np.linspace(tol_min, tol_max, num=max(2, steps))

    # Establish baseline
    tissue.build_global_vertices(tol=tol_min)
    tissue.reconstruct_cell_vertices()
    prev_stats = _connectivity_stats(tissue)
    print(f"[Stitch] Start @ tol={tol_min:.2f}: GV={prev_stats[0]}, shared={prev_stats[1]}, "
          f"connected={prev_stats[2]}, meanN={prev_stats[3]:.2f}")

    for p in range(1, max_passes + 1):
        changed = False
        print(f"[Stitch] Pass {p}/{max_passes}")
        for tol in schedule:
            tissue.build_global_vertices(tol=float(tol))
            tissue.reconstruct_cell_vertices()
            stats = _connectivity_stats(tissue)
            if stats != prev_stats:
                changed = True
                print(f"  tol={tol:.2f}: GV {prev_stats[0]} -> {stats[0]}, shared {prev_stats[1]} -> {stats[1]}, "
                      f"connected {prev_stats[2]} -> {stats[2]}, meanN {prev_stats[3]:.2f} -> {stats[3]:.2f}")
                prev_stats = stats
        if not changed:
            print("[Stitch] Converged (no changes this pass).")
            break
    gv, sh, cc, mn = prev_stats
    ratio = (sh / gv * 100.0) if gv else 0.0
    print(f"[Stitch] Final: GV={gv}, shared={sh} ({ratio:.1f}%), connected={cc}/{len(tissue.cells)}, meanN={mn:.2f}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate single-cell growth in vertex model tissue.")
    parser.add_argument("--tissue-file", type=str, default=None,
                       help="Path to tissue file (optional if using --build-honeycomb).")
    parser.add_argument("--build-honeycomb", type=str, choices=['14', '19'], default='14',
                       help="Build honeycomb tissue on-the-fly: '14' for 2-3-4-3-2 (default), '19' for 3-4-5-4-3.")
    parser.add_argument("--growing-cell-id", type=int, default=7,
                       help="Cell ID to grow.")
    parser.add_argument("--growth-steps", type=int, default=100, help="Steps to ramp up target area.")
    parser.add_argument("--total-steps", type=int, default=200, help="Total simulation steps.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument("--plot", action="store_true", help="Show initial and final tissue plots.")
    parser.add_argument("--log-interval", type=int, default=10, help="Log progress every N steps.")
    parser.add_argument("--save-csv", type=str, default=None, help="Save tracking data to CSV file.")
    # Energy parameters
    parser.add_argument("--k-area", type=float, default=1.0, help="Area elasticity coefficient.")
    parser.add_argument("--k-perimeter", type=float, default=0.1, help="Perimeter coefficient.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Line tension parameter.")
    # Simulation parameters
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Finite difference epsilon.")
    parser.add_argument("--damping", type=float, default=1.0, help="Damping factor for gradient descent.")
    return parser.parse_args(argv)


def create_simulation_folder(tissue_identifier: str, growing_cell_id: int) -> Path:
    """Create a unique simulation folder for this run.

    Args:
        tissue_identifier: Identifier for the tissue (e.g., 'honeycomb14', 'acam_79cells')
        growing_cell_id: ID of the growing cell

    Returns:
        Path to the created simulation folder
    """
    # Create timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add random number for extra uniqueness (in case multiple runs in same second)
    random_num = random.randint(1000, 9999)

    # Create folder name: Sim_<tissue>_cell<id>_<timestamp>_<random>
    folder_name = f"Sim_{tissue_identifier}_cell{growing_cell_id}_{timestamp}_{random_num}"

    # Create in current directory
    sim_folder = Path(folder_name)
    sim_folder.mkdir(parents=True, exist_ok=True)

    return sim_folder



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Determine tissue identifier for folder naming
    tissue_identifier = ""

    # Load or build tissue
    if args.tissue_file:
        # Load from file
        print(f"Loading tissue from: {args.tissue_file}")
        candidate = Path(args.tissue_file)
        if candidate.suffix != ".dill":
            candidate = candidate.with_suffix(".dill")
        if not candidate.exists():
            print(f"Error: Tissue file not found: {candidate}")
            return 1
        tissue = load_tissue(str(candidate))
        # Extract tissue identifier from filename (without extension)
        tissue_identifier = candidate.stem
    else:
        # Build honeycomb tissue on-the-fly
        from myvertexmodel import build_honeycomb_2_3_4_3_2, build_honeycomb_3_4_5_4_3

        if args.build_honeycomb == '14':
            print("Building honeycomb tissue (14 cells, 2-3-4-3-2 pattern)...")
            tissue = build_honeycomb_2_3_4_3_2()
            tissue_identifier = "honeycomb14"
        else:  # '19'
            print("Building honeycomb tissue (19 cells, 3-4-5-4-3 pattern)...")
            tissue = build_honeycomb_3_4_5_4_3()
            tissue_identifier = "honeycomb19"

    # Create unique simulation folder
    sim_folder = create_simulation_folder(tissue_identifier, args.growing_cell_id)
    print(f"\nSimulation folder created: {sim_folder}")
    print(f"  All outputs will be saved to this folder")

    # Build global vertex pool
    tissue.build_global_vertices(tol=1e-10)

    # CRITICAL: Sync cell.vertices with global vertex pool
    # After loading, cell.vertices may be out of sync with tissue.vertices[cell.vertex_indices]
    tissue.reconstruct_cell_vertices()

    print(f"Loaded {len(tissue.cells)} cells from tissue")
    print(f"  Global vertices: {tissue.vertices.shape[0]} unique positions")

    geometry = GeometryCalculator()
    
    # Find growing cell by ID
    growing_cell = None
    for cell in tissue.cells:
        if cell.id == args.growing_cell_id:
            growing_cell = cell
            break

    if growing_cell is None:
        print(f"\nError: Cell with ID '{args.growing_cell_id}' not found!")
        print(f"Available cell IDs: {sorted([c.id for c in tissue.cells])}")
        return 1

    growing_cell_id = growing_cell.id
    print(f"\nFound growing cell: ID {growing_cell_id}")

    # Compute initial areas for all cells
    print("\nComputing initial cell areas...")
    initial_areas = {}
    for cell in tissue.cells:
        initial_areas[cell.id] = geometry.calculate_area(cell.vertices)
    
    # Set up per-cell target areas (initially equal to actual areas for equilibrium)
    target_areas = initial_areas.copy()
    
    # Define growth target for selected cell
    A_initial = initial_areas[growing_cell_id]
    A_final = 2.0 * A_initial
    growth_rate = (A_final - A_initial) / args.growth_steps
    
    print(f"\nGrowth configuration:")
    print(f"  Tissue: {len(tissue.cells)} cells")
    print(f"  Growing cell ID: {growing_cell_id}")
    print(f"  Initial area: {A_initial:.2f}")
    print(f"  Target area: {A_final:.2f} (2× initial)")
    print(f"  Growth steps: {args.growth_steps}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Time step dt: {args.dt}")

    # Create energy parameters with per-cell target areas
    energy_params = EnergyParameters(
        k_area=args.k_area,
        k_perimeter=args.k_perimeter,
        gamma=args.gamma,
        target_area=target_areas  # dict instead of float
    )
    
    # Create simulation
    sim = Simulation(
        tissue=tissue,
        dt=args.dt,
        energy_params=energy_params,
        epsilon=args.epsilon,
        damping=args.damping
    )
    
    # Show initial tissue
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=0.3)
            ax.set_title(f"Initial Tissue (Growing cell ID: {growing_cell_id})", fontsize=12, fontweight='bold')
            initial_plot_path = sim_folder / "growth_initial.png"
            plt.savefig(str(initial_plot_path), dpi=150)
            print(f"\nSaved initial tissue plot to {initial_plot_path}")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"Initial plot failed: {e}")
    
    # Prepare CSV logging if requested
    csv_data = []
    csv_path = None
    if args.save_csv:
        # Use provided name but save in simulation folder
        csv_path = sim_folder / Path(args.save_csv).name
        csv_data.append(["step", "time", "growing_cell_area", "target_area", "total_energy", "progress_percent"])
    else:
        # Auto-generate CSV file in simulation folder
        csv_path = sim_folder / "growth_tracking.csv"
        csv_data.append(["step", "time", "growing_cell_area", "target_area", "total_energy", "progress_percent"])
    
    # Simulation loop with growth using GLOBAL VERTEX POOL
    print("\nStarting simulation with globally-coupled vertices...\n")
    print(f"{'Step':>6} {'Time':>8} {'Area':>10} {'Target':>10} {'Progress':>8} {'Energy':>12}")
    print("-" * 65)

    sim_time = 0.0

    for step in range(args.total_steps):
        # Update target area for growing cell (linear ramp during growth phase)
        if step < args.growth_steps:
            target_areas[growing_cell_id] = A_initial + step * growth_rate
        else:
            # After growth phase, maintain final target
            target_areas[growing_cell_id] = A_final
        
        # Compute gradient with respect to GLOBAL vertices
        # This ensures shared vertices move consistently across all adjacent cells
        gradient = compute_global_gradient(tissue, energy_params, geometry, epsilon=args.epsilon)

        # Update GLOBAL vertices using gradient descent
        tissue.vertices = tissue.vertices - args.dt * args.damping * gradient

        # Reconstruct per-cell vertices from global pool
        tissue.reconstruct_cell_vertices()

        # Advance time
        sim_time += args.dt

        # Get current state
        current_area = geometry.calculate_area(growing_cell.vertices)
        current_target = target_areas[growing_cell_id]
        current_energy = sim.total_energy()
        progress = (current_area - A_initial) / (A_final - A_initial) * 100
        
        # Log progress
        if step % args.log_interval == 0 or step == args.total_steps - 1:
            print(f"{step:6d} {sim_time:8.3f} {current_area:10.2f} {current_target:10.2f} {progress:7.1f}% {current_energy:12.2f}")

        # Save CSV data (always save now)
        csv_data.append([step, sim_time, current_area, current_target, current_energy, progress])

        # Check stopping criterion (reached target area)
        if current_area >= 0.99 * A_final:
            print(f"\n✓ Growth complete at step {step}! Cell reached {current_area/A_initial:.2f}× initial area.")
            break
    
    # Final statistics
    final_area = geometry.calculate_area(growing_cell.vertices)
    print(f"\nFinal statistics:")
    print(f"  Growing cell ID {growing_cell_id} area: {final_area:.2f} ({final_area/A_initial:.2f}× initial)")
    print(f"  Target reached: {final_area >= 0.99 * A_final}")
    print(f"  Final energy: {sim.total_energy():.2f}")
    print(f"  Total simulation time: {sim_time:.3f}")

    # Show final tissue
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=0.3)
            ax.set_title(f"Final Tissue (Cell {growing_cell_id} area: {final_area:.2f})", fontsize=12, fontweight='bold')
            final_plot_path = sim_folder / "growth_final.png"
            plt.savefig(str(final_plot_path), dpi=150, bbox_inches='tight')
            print(f"\nSaved final tissue plot to {final_plot_path}")
            plt.close()
        except Exception as e:
            print(f"Final plot failed: {e}")
    
    # Save CSV (always save now)
    try:
        with open(csv_path, "w") as f:
            for row in csv_data:
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"\nSaved tracking data to {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

    # Print summary of simulation folder contents
    print(f"\nSimulation complete! All outputs saved to: {sim_folder}/")
    print(f"  Contents:")
    for file in sorted(sim_folder.iterdir()):
        print(f"    - {file.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
