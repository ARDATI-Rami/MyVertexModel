"""simulate_cell_growth.py

Simulate growth of one or more cells in a vertex model tissue until they reach double their initial area.

The simulation:
1. Builds or loads a tissue (honeycomb built on-the-fly by default, or load from file)
2. Selects cells as "growing cells" by cell ID(s)
3. Gradually increases target area of each growing cell from A_initial to 2*A_initial
4. Allows the tissue to mechanically equilibrate as cells expand
5. Tracks actual area vs target area for each growing cell

Usage Examples:
--------------
Single cell (honeycomb):
    python examples/simulate_cell_growth.py --growing-cell-ids 7 --plot

Multiple cells (honeycomb):
    python examples/simulate_cell_growth.py --growing-cell-ids 3,7,10 --plot

Load tissue from file with multiple cells:
    python examples/simulate_cell_growth.py --tissue-file pickled_tissues/acam_79cells.dill \\
        --growing-cell-ids I,AW,AB,AA,V,BF,AV,BR,AL --total-steps 100 --dt 0.00001 --plot --enable-merge

Basic Options:
    --tissue-file FILE      Path to tissue file (.dill). If not specified, builds honeycomb.
    --build-honeycomb {14,19}
                            Build honeycomb on-the-fly: '14' (2-3-4-3-2) or '19' (3-4-5-4-3).
    --growing-cell-ids IDS  Comma-separated cell IDs to grow (e.g., 'I,AW,AB' or '7,10').
    --growing-cell-id ID    (Deprecated) Use --growing-cell-ids instead.
    --growth-steps N        Steps over which to ramp target area (default: 100).
    --total-steps N         Total simulation steps (default: 200).
    --dt FLOAT              Time step size (default: 0.01).
    --plot                  Show and save initial/final tissue plots.
    --log-interval N        Print progress every N steps (default: 10).
    --save-csv FILE         Custom CSV filename (default: growth_tracking.csv).

Energy Parameters:
    --k-area FLOAT          Area elasticity coefficient (default: 1.0).
    --k-perimeter FLOAT     Perimeter coefficient (default: 0.1).
    --gamma FLOAT           Line tension parameter (default: 0.05).

Solver Options:
    --solver {gradient_descent,overdamped_force_balance}
                            Solver type (default: gradient_descent).
    --epsilon FLOAT         Finite difference epsilon (default: 1e-6).
    --damping FLOAT         Damping factor for gradient descent (default: 1.0).

OFB (Overdamped Force Balance) Parameters:
    --ofb-gamma FLOAT       Friction coefficient γ (default: 1.0).
    --ofb-noise FLOAT       Noise strength (default: 0.0).
    --ofb-seed INT          Random seed for reproducibility.

Vertex Merging Options:
    --enable-merge          Enable periodic merging of nearby vertices.
    --merge-distance-tol FLOAT
                            Distance tolerance for merging (default: 0.1).
    --merge-energy-tol FLOAT
                            Max energy change allowed (default: 0.1).
    --merge-geometry-tol FLOAT
                            Max area/perimeter change allowed (default: 1.0).
    --merge-interval N      Merge every N steps (default: 10).

Meshing Options:
    --mesh-mode {none,low,medium,high}
                            Edge meshing mode at start (default: none).
    --mesh-length-scale FLOAT
                            Base length scale for meshing (default: 1.0).
    --enable-mesh-dynamic   Enable periodic meshing during simulation.
    --mesh-interval N       Apply meshing every N steps (default: 10).

Other Options:
    --relabel-alpha-mode {direct,order}
                            Relabel cell IDs to alphabetic: 'direct' (1->A) or 'order' (sorted).

Output:
    Creates a simulation folder (Sim_<tissue>_<cells>_<timestamp>_<random>/) containing:
    - growth_initial.png    Initial tissue visualization (if --plot)
    - growth_final.png      Final tissue visualization (if --plot)
    - growth_tracking.csv   Per-step tracking data for all growing cells
"""
from __future__ import annotations
import argparse
import sys
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import random
import numpy as np
from myvertexmodel import Tissue, Simulation, EnergyParameters, GeometryCalculator, plot_tissue
from myvertexmodel import load_tissue
from myvertexmodel.simulation import OverdampedForceBalanceParams
# Import merge and mesh operations from package (re-exported from mesh_ops.py)
from myvertexmodel import merge_nearby_vertices, mesh_edges
from myvertexmodel import relabel_cells_alpha

# Local helper to save and optionally show Matplotlib figures
def _save_plot(fig, path: Path, show: bool = True) -> None:
    """Save a Matplotlib figure to disk and optionally display it.

    Args:
        fig: Matplotlib figure object.
        path: Destination path for the image file.
        show: If True, display the plot with plt.show().
    """
    import matplotlib.pyplot as plt

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        if show:
            plt.show()
    finally:
        plt.close(fig)


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


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate cell growth in vertex model tissue.")
    parser.add_argument("--tissue-file", type=str, default=None,
                       help="Path to tissue file (optional if using --build-honeycomb).")
    parser.add_argument("--build-honeycomb", type=str, choices=['14', '19'], default='14',
                       help="Build honeycomb tissue on-the-fly: '14' for 2-3-4-3-2 (default), '19' for 3-4-5-4-3.")
    parser.add_argument("--growing-cell-ids", type=str, default="7",
                       help="Comma-separated cell IDs to grow (e.g., 'I,AW,AB' or '7').")
    # Keep backward compatibility with old argument name
    parser.add_argument("--growing-cell-id", type=str, default=None,
                       help="(Deprecated) Use --growing-cell-ids instead. Single cell ID to grow.")
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
    # Solver selection
    parser.add_argument("--solver", type=str, choices=["gradient_descent", "overdamped_force_balance"], default="gradient_descent",
                       help="Choose solver: 'gradient_descent' (default) or 'overdamped_force_balance' (OFB).")
    # OFB parameters
    parser.add_argument("--ofb-gamma", type=float, default=1.0, help="OFB friction coefficient γ (>0).")
    parser.add_argument("--ofb-noise", type=float, default=0.0, help="OFB noise strength (>=0).")
    parser.add_argument("--ofb-seed", type=int, default=None, help="OFB random seed for reproducibility.")
    # Vertex merging options
    parser.add_argument("--enable-merge", action="store_true", help="Enable periodic merging of nearby vertices during simulation.")
    parser.add_argument("--merge-distance-tol", type=float, default=1e-1, help="Distance tolerance for merging nearby vertices.")
    parser.add_argument("--merge-energy-tol", type=float, default=1e-1, help="Max allowed energy change due to merge.")
    parser.add_argument("--merge-geometry-tol", type=float, default=1e0, help="Max allowed per-cell area/perimeter change due to merge.")
    parser.add_argument("--merge-interval", type=int, default=10, help="Merge every N steps (if enabled).")
    # Meshing options
    parser.add_argument("--mesh-mode", type=str, choices=["none", "low", "medium", "high"], default="none", help="Edge meshing mode to apply at start: none|low|medium|high.")
    parser.add_argument("--mesh-length-scale", type=float, default=1.0, help="Base length scale used by meshing (medium/high).")
    # Dynamic meshing options
    parser.add_argument("--enable-mesh-dynamic", action="store_true", help="Enable periodic dynamic meshing during simulation.")
    parser.add_argument("--mesh-interval", type=int, default=10, help="Apply meshing every N steps when enabled.")
    # Relabel cells to alphabetic IDs (single-mode argument)
    parser.add_argument("--relabel-alpha-mode", type=str, choices=["direct", "order"], default=None,
                       help="Relabel cell IDs to alphabetic labels: 'direct' converts each numeric ID (1->A), 'order' assigns labels by sorted order (1..N->A..).")
    return parser.parse_args(argv)


def create_simulation_folder(tissue_identifier: str, growing_cell_ids: List[str]) -> Path:
    """Create a unique simulation folder for this run.

    Args:
        tissue_identifier: Identifier for the tissue (e.g., 'honeycomb14', 'acam_79cells')
        growing_cell_ids: List of growing cell IDs

    Returns:
        Path to the created simulation folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_num = random.randint(1000, 9999)

    # Create cell identifier (use count if many cells, else list them)
    if len(growing_cell_ids) <= 3:
        cell_str = "_".join(str(c) for c in growing_cell_ids)
    else:
        cell_str = f"{len(growing_cell_ids)}cells"

    folder_name = f"Sim_{tissue_identifier}_{cell_str}_{timestamp}_{random_num}"
    sim_folder = Path(folder_name)
    sim_folder.mkdir(parents=True, exist_ok=True)

    return sim_folder



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    
    # Handle backward compatibility: --growing-cell-id -> --growing-cell-ids
    if args.growing_cell_id is not None:
        cell_ids_str = args.growing_cell_id
    else:
        cell_ids_str = args.growing_cell_ids

    # Parse comma-separated cell IDs
    growing_cell_id_list = [cid.strip() for cid in cell_ids_str.split(",") if cid.strip()]

    # Determine tissue identifier for folder naming
    tissue_identifier = ""

    # Load or build tissue
    if args.tissue_file:
        print(f"Loading tissue from: {args.tissue_file}")
        candidate = Path(args.tissue_file)
        if candidate.suffix != ".dill":
            candidate = candidate.with_suffix(".dill")
        if not candidate.exists():
            print(f"Error: Tissue file not found: {candidate}")
            return 1
        tissue = load_tissue(str(candidate))
        tissue_identifier = candidate.stem
    else:
        from myvertexmodel import build_honeycomb_2_3_4_3_2, build_honeycomb_3_4_5_4_3
        if args.build_honeycomb == '14':
            print("Building honeycomb tissue (14 cells, 2-3-4-3-2 pattern)...")
            tissue = build_honeycomb_2_3_4_3_2()
            tissue_identifier = "honeycomb14"
        else:
            print("Building honeycomb tissue (19 cells, 3-4-5-4-3 pattern)...")
            tissue = build_honeycomb_3_4_5_4_3()
            tissue_identifier = "honeycomb19"

    # Create unique simulation folder
    sim_folder = create_simulation_folder(tissue_identifier, growing_cell_id_list)
    print(f"\nSimulation folder created: {sim_folder}")

    # Build global vertex pool
    tissue.build_global_vertices(tol=1e-10)
    tissue.reconstruct_cell_vertices()

    # Optional: relabel cell IDs to alphabetic
    if args.relabel_alpha_mode is not None:
        print(f"\nRelabeling cell IDs to alphabetic labels (mode={args.relabel_alpha_mode})...")
        try:
            relabel_cells_alpha(tissue, by_order=(args.relabel_alpha_mode == "order"))
            print(f"[relabel] Applied alphabetic labels (mode={args.relabel_alpha_mode})")
        except Exception as e:
            print(f"[relabel] skipped: {e}")

    print(f"Loaded {len(tissue.cells)} cells from tissue")
    print(f"  Global vertices: {tissue.vertices.shape[0]} unique positions")

    geometry = GeometryCalculator()

    # Find growing cells by ID - helper function
    def find_cell_by_id(provided_id: str):
        for cell in tissue.cells:
            if str(cell.id) == provided_id:
                return cell
        # Fallback: try numeric match
        try:
            pid_int = int(provided_id)
            for cell in tissue.cells:
                try:
                    if int(cell.id) == pid_int:
                        return cell
                except Exception:
                    continue
        except ValueError:
            pass
        return None

    # Find all growing cells
    growing_cells = {}
    not_found = []
    for cid in growing_cell_id_list:
        cell = find_cell_by_id(cid)
        if cell:
            growing_cells[cell.id] = cell
        else:
            not_found.append(cid)

    if not_found:
        print(f"\nError: Cell(s) not found: {not_found}")
        print(f"Available cell IDs: {sorted([str(c.id) for c in tissue.cells])}")
        return 1

    if not growing_cells:
        print("\nError: No growing cells specified!")
        return 1

    print(f"\nFound {len(growing_cells)} growing cell(s): {list(growing_cells.keys())}")

    # Compute initial areas for all cells
    initial_areas = {cell.id: geometry.calculate_area(cell.vertices) for cell in tissue.cells}
    target_areas = initial_areas.copy()

    # Define growth targets for each growing cell
    growth_info = {}
    for cell_id, cell in growing_cells.items():
        A_initial = initial_areas[cell_id]
        A_final = 2.0 * A_initial
        growth_rate = (A_final - A_initial) / args.growth_steps
        growth_info[cell_id] = {"initial": A_initial, "final": A_final, "rate": growth_rate, "cell": cell}

    print(f"\nGrowth configuration:")
    print(f"  Tissue: {len(tissue.cells)} cells")
    print(f"  Growing cells: {len(growing_cells)}")
    for cid, info in growth_info.items():
        print(f"    - {cid}: {info['initial']:.2f} → {info['final']:.2f}")
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

    # Optional meshing at the beginning (after energy_params are defined)
    if args.mesh_mode and args.mesh_mode != "none":
        try:
            mstats = mesh_edges(
                tissue,
                mode=args.mesh_mode,
                length_scale=args.mesh_length_scale,
                energy_tol=1e-10,
                geometry_tol=1e-10,
                energy_params=energy_params,
            )
            print(
                f"[mesh] mode={mstats['mode']} edges_subdivided={mstats['edges_subdivided']} "
                f"verts {mstats['vertices_before']} -> {mstats['vertices_after']} "
                f"dE={mstats['energy_change']:.2e} dA_max={mstats['max_area_change']:.2e} dP_max={mstats['max_perimeter_change']:.2e}"
            )
        except ValueError as e:
            print(f"[mesh] skipped: {e}")

    # Create simulation
    if args.solver == "overdamped_force_balance":
        ofb_params = OverdampedForceBalanceParams(
            gamma=args.ofb_gamma,
            noise_strength=args.ofb_noise,
            random_seed=args.ofb_seed,
        )
        sim = Simulation(
            tissue=tissue,
            dt=args.dt,
            energy_params=energy_params,
            epsilon=args.epsilon,
            damping=args.damping,  # not used in OFB solver
            solver_type="overdamped_force_balance",
            ofb_params=ofb_params,
        )
    else:
        sim = Simulation(
            tissue=tissue,
            dt=args.dt,
            energy_params=energy_params,
            epsilon=args.epsilon,
            damping=args.damping,
            solver_type="gradient_descent",
        )

    # Show initial tissue
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=1, show_cell_ids=True, show_neighbor_counts=False)
            cell_ids_display = ", ".join(str(c) for c in growing_cells.keys())
            ax.set_title(f"Initial Tissue (Growing cells: {cell_ids_display})", fontsize=12, fontweight='bold')
            initial_plot_path = sim_folder / "growth_initial.png"
            _save_plot(fig, initial_plot_path, show=True)
            print(f"\nSaved initial tissue plot to {initial_plot_path}")
        except Exception as e:
            print(f"Initial plot failed: {e}")
    
    # Prepare CSV logging
    csv_path = sim_folder / (Path(args.save_csv).name if args.save_csv else "growth_tracking.csv")
    # Build CSV header: step, time, energy, then area/target/progress per cell
    csv_header = ["step", "time", "total_energy"]
    for cid in growing_cells.keys():
        csv_header.extend([f"area_{cid}", f"target_{cid}", f"progress_{cid}"])
    csv_data = [csv_header]

    # Simulation loop
    print("\nStarting simulation...\n")
    # Build dynamic header based on number of cells
    if len(growing_cells) == 1:
        print(f"{'Step':>6} {'Time':>8} {'Area':>10} {'Target':>10} {'Progress':>8} {'Energy':>12}")
        print("-" * 65)
    else:
        print(f"{'Step':>6} {'Time':>8} {'AvgProgress':>10} {'Energy':>12}")
        print("-" * 45)

    sim_time = 0.0

    for step in range(args.total_steps):
        # Update target areas for all growing cells (linear ramp during growth phase)
        for cid, info in growth_info.items():
            if step < args.growth_steps:
                target_areas[cid] = info["initial"] + step * info["rate"]
            else:
                target_areas[cid] = info["final"]

        if args.solver == "gradient_descent":
            # Compute gradient with respect to GLOBAL vertices
            gradient = compute_global_gradient(tissue, energy_params, geometry, epsilon=args.epsilon)
            # Update GLOBAL vertices using gradient descent
            tissue.vertices = tissue.vertices - args.dt * args.damping * gradient
            # Reconstruct per-cell vertices from global pool
            tissue.reconstruct_cell_vertices()
        else:
            # Overdamped force-balance on GLOBAL vertex pool
            # Forces from energy: F = -∇E (global gradient)
            grad = compute_global_gradient(tissue, energy_params, geometry, epsilon=args.epsilon)
            forces = -grad
            # Optional noise term (Gaussian, zero mean)
            if args.ofb_noise and args.ofb_noise > 0.0:
                noise = np.random.default_rng(args.ofb_seed).normal(loc=0.0, scale=args.ofb_noise, size=forces.shape)
            else:
                noise = 0.0
            # Integrate: x <- x + (dt/gamma) * (F + noise)
            tissue.vertices = tissue.vertices + (args.dt / max(args.ofb_gamma, 1e-12)) * (forces + noise)
            # Reconstruct per-cell vertices
            tissue.reconstruct_cell_vertices()

        # Periodic vertex merging (optional)
        if args.enable_merge and args.merge_interval > 0 and (step % args.merge_interval == 0):
            try:
                stats = merge_nearby_vertices(
                    tissue,
                    distance_tol=args.merge_distance_tol,
                    energy_tol=args.merge_energy_tol,
                    geometry_tol=args.merge_geometry_tol,
                    energy_params=energy_params,
                )
                if stats.get("clusters_merged", 0) > 0:
                    print(
                        f"[merge] step={step} clusters={stats['clusters_merged']} verts {stats['vertices_before']} -> {stats['vertices_after']} "
                        f"dE={stats['energy_change']:.2e} dA_max={stats['max_area_change']:.2e} dP_max={stats['max_perimeter_change']:.2e}"
                    )
            except ValueError as e:
                print(f"[merge] step={step} skipped: {e}")

        # Periodic dynamic meshing (optional)
        if args.enable_mesh_dynamic and args.mesh_interval > 0 and (step % args.mesh_interval == 0) and args.mesh_mode != "none":
            try:
                mstats = mesh_edges(
                    tissue,
                    mode=args.mesh_mode,
                    length_scale=args.mesh_length_scale,
                    energy_tol=1e-10,
                    geometry_tol=1e-10,
                    energy_params=energy_params,
                )
                if mstats.get("edges_subdivided", 0) > 0:
                    print(
                        f"[mesh] step={step} mode={mstats['mode']} edges_subdivided={mstats['edges_subdivided']} "
                        f"verts {mstats['vertices_before']} -> {mstats['vertices_after']} "
                        f"dE={mstats['energy_change']:.2e} dA_max={mstats['max_area_change']:.2e} dP_max={mstats['max_perimeter_change']:.2e}"
                    )
            except ValueError as e:
                print(f"[mesh] step={step} skipped: {e}")

        # Advance time
        sim_time += args.dt

        # Get current state for all growing cells
        current_energy = sim.total_energy()
        cell_progress = {}
        csv_row = [step, sim_time, current_energy]

        for cid, info in growth_info.items():
            cell = info["cell"]
            current_area = geometry.calculate_area(cell.vertices)
            current_target = target_areas[cid]
            progress = (current_area - info["initial"]) / (info["final"] - info["initial"]) * 100
            cell_progress[cid] = {"area": current_area, "target": current_target, "progress": progress}
            csv_row.extend([current_area, current_target, progress])

        csv_data.append(csv_row)

        # Log progress
        if step % args.log_interval == 0 or step == args.total_steps - 1:
            if len(growing_cells) == 1:
                cid = list(growing_cells.keys())[0]
                p = cell_progress[cid]
                print(f"{step:6d} {sim_time:8.3f} {p['area']:10.2f} {p['target']:10.2f} {p['progress']:7.1f}% {current_energy:12.2f}")
            else:
                avg_progress = sum(p["progress"] for p in cell_progress.values()) / len(cell_progress)
                print(f"{step:6d} {sim_time:8.3f} {avg_progress:9.1f}% {current_energy:12.2f}")

        # Check stopping criterion (all cells reached target area)
        all_complete = all(
            cell_progress[cid]["area"] >= 0.99 * growth_info[cid]["final"]
            for cid in growing_cells.keys()
        )
        if all_complete:
            print(f"\n✓ Growth complete at step {step}! All cells reached target area.")
            break
    
    # Final statistics
    print(f"\nFinal statistics:")
    print(f"  Growing cells: {len(growing_cells)}")
    all_reached = True
    for cid, info in growth_info.items():
        final_area = geometry.calculate_area(info["cell"].vertices)
        ratio = final_area / info["initial"]
        reached = final_area >= 0.99 * info["final"]
        all_reached = all_reached and reached
        print(f"    - {cid}: {final_area:.2f} ({ratio:.2f}× initial) {'✓' if reached else '✗'}")
    print(f"  All targets reached: {all_reached}")
    print(f"  Final energy: {sim.total_energy():.2f}")
    print(f"  Total simulation time: {sim_time:.3f}")

    # Show final tissue
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=1, show_cell_ids=True, show_neighbor_counts=False)
            cell_ids_display = ", ".join(str(c) for c in growing_cells.keys())
            ax.set_title(f"Final Tissue (Growing cells: {cell_ids_display})", fontsize=12, fontweight='bold')
            final_plot_path = sim_folder / "growth_final.png"
            _save_plot(fig, final_plot_path, show=True)
            print(f"\nSaved final tissue plot to {final_plot_path}")
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

    # Final matplotlib cleanup to ensure no figures remain open
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
