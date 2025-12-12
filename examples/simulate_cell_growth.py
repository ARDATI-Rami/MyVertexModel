"""simulate_cell_growth.py

Simulate growth of one or more cells in a vertex model tissue until they reach double their initial area.
Optionally, cells can undergo cytokinesis (cell division) when reaching a specified area threshold.

The simulation:
1. Builds or loads a tissue (honeycomb built on-the-fly by default, or load from file)
2. Selects cells as "growing cells" by cell ID(s)
3. Gradually increases target area of each growing cell from A_initial to 2*A_initial
4. Allows the tissue to mechanically equilibrate as cells expand
5. Tracks actual area vs target area for each growing cell
6. (Optional) When cells reach 2x initial area, initiates cytokinesis:
   - Inserts contracting vertices perpendicular to the cell's long axis
   - Applies contractile forces simulating the actomyosin ring
   - Splits the cell into two daughter cells when sufficiently constricted

Usage Examples:
--------------
Single cell (honeycomb):
    python examples/simulate_cell_growth.py --growing-cell-ids 7 --plot

Multiple cells (honeycomb):
    python examples/simulate_cell_growth.py --growing-cell-ids 3,7,10 --plot

Growth with cytokinesis (cell division when reaching 2x area):
    python examples/simulate_cell_growth.py --growing-cell-ids 7 --enable-cytokinesis \\
        --cyto-force-magnitude 50.0 --total-steps 500 --dt 0.005 --plot

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

Cytokinesis (Cell Division) Options:
    --enable-cytokinesis    Enable cell division when cells reach threshold area.
    --cyto-division-area-ratio FLOAT
                            Area ratio to trigger division initiation (default: 2.0 = 2x initial area).
    --cyto-constriction-percentage FLOAT
                            Percentage of bounding box diagonal for split threshold (default: 10.0%).
                            Cell splits when contracting vertices are closer than this % of bbox diagonal.
    --cyto-constriction-threshold FLOAT
                            Absolute distance threshold for splitting (overrides percentage if set).
    --cyto-initial-separation FLOAT
                            Initial separation of contracting vertices (default: 0.95).
    --cyto-force-magnitude FLOAT
                            Contractile force magnitude (default: 10.0, use ~150000 for ACAM tissue).

Apoptosis Options:
    --apoptotic-cell-ids IDS
                            Comma-separated cell IDs to undergo apoptosis (e.g., '5' or 'I,AW').
    --apoptosis-shrink-rate FLOAT
                            Fractional shrink rate per unit time (default: 0.5).
    --apoptosis-min-area-fraction FLOAT
                            Minimum area fraction used as a floor for the apoptotic target area (default: 0.05).
    --apoptosis-removal-area-fraction FLOAT
                            Fraction of initial area below which apoptotic cells are removed (default: 0.1; set 0 to disable).
    --apoptosis-removal-area-absolute FLOAT
                            Absolute area threshold below which apoptotic cells are removed (default: 0.0; set 0 to disable).
    --apoptosis-start-step INT
                            Step index at which apoptosis starts (default: 0).
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
# Import cytokinesis functions for cell division
from myvertexmodel import (
    CytokinesisParams,
    compute_division_axis,
    insert_contracting_vertices,
    compute_contractile_forces,
    check_constriction,
    split_cell,
    update_global_vertices_from_cells,
)
from myvertexmodel.apoptosis import (
    ApoptosisParameters,
    ApoptosisState,
    update_apoptosis_targets,
    collect_cells_to_remove,
    build_apoptosis_target_area_mapping,
)

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
    parser.add_argument("--plot", action="store_true", help="Show and save initial and final tissue plots.")
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
    # Cytokinesis (cell division) options
    parser.add_argument("--enable-cytokinesis", action="store_true",
                       help="Enable cytokinesis: cells divide when reaching specified area ratio.")
    parser.add_argument("--cyto-constriction-threshold", type=float, default=None,
                       help="Absolute distance threshold for splitting (default: None, use percentage instead).")
    parser.add_argument("--cyto-constriction-percentage", type=float, default=10.0,
                       help="Percentage of bounding box diagonal for split threshold (default: 10.0%%). "
                            "When contracting vertices are closer than this %% of bbox diagonal, cell splits.")
    parser.add_argument("--cyto-initial-separation", type=float, default=0.95,
                       help="Initial separation of contracting vertices as fraction (default: 0.95).")
    parser.add_argument("--cyto-force-magnitude", type=float, default=10.0,
                       help="Contractile force magnitude (default: 10.0).")
    parser.add_argument("--cyto-division-area-ratio", type=float, default=2.0,
                       help="Area ratio at which to trigger division initiation (default: 2.0, meaning 2x initial area).")
    # Apoptosis options
    parser.add_argument("--apoptotic-cell-ids", type=str, default=None,
                       help="Comma-separated cell IDs to undergo apoptosis (e.g., '5' or 'I,AW').")
    parser.add_argument("--apoptosis-shrink-rate", type=float, default=0.5,
                       help="Fractional shrink rate per unit time for apoptotic cells.")
    parser.add_argument("--apoptosis-min-area-fraction", type=float, default=0.05,
                       help="Floor for apoptotic target area as a fraction of initial area (A_target >= frac*A0).")
    parser.add_argument("--apoptosis-removal-area-fraction", type=float, default=0.1,
                       help="Remove apoptotic cells when area < frac*A0 (set 0 to disable).")
    parser.add_argument("--apoptosis-removal-area-absolute", type=float, default=0.0,
                       help="Remove apoptotic cells when area < absolute threshold (set 0 to disable).")
    parser.add_argument("--apoptosis-start-step", type=int, default=0,
                       help="Step index at which apoptosis starts.")
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

    # Allow empty or whitespace-only growing-cell-ids to mean "no growth"
    if cell_ids_str is None:
        cell_ids_str = ""

    # Parse comma-separated cell IDs (may be empty)
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
    sim_folder = create_simulation_folder(tissue_identifier, growing_cell_id_list or ["no_growth"])
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

    # Helper to find cell by ID
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

    # Find all growing cells (may be empty)
    growing_cells = {}
    not_found = []
    for cid in growing_cell_id_list:
        cell = find_cell_by_id(cid)
        if cell:
            growing_cells[cell.id] = cell
        else:
            not_found.append(cid)

    if not_found:
        print(f"\nWarning: Growing cell(s) not found and will be ignored: {not_found}")
        print(f"Available cell IDs: {sorted([str(c.id) for c in tissue.cells])}")

    # No longer treat "no growing cells" as fatal; allow apoptosis-only runs
    if not growing_cells:
        print("\nNo growing cells specified: running without growth (apoptosis and/or cytokinesis only).")

    # Compute initial areas for all cells
    initial_areas = {cell.id: geometry.calculate_area(cell.vertices) for cell in tissue.cells}
    target_areas = initial_areas.copy()

    # Define growth targets only for growing cells
    growth_info = {}
    for cell_id, cell in growing_cells.items():
        A_initial = initial_areas[cell_id]
        A_final = 2.0 * A_initial
        growth_rate = (A_final - A_initial) / args.growth_steps
        growth_info[cell_id] = {"initial": A_initial, "final": A_final, "rate": growth_rate, "cell": cell}

    # Cytokinesis tracking state
    cytokinesis_params = None
    dividing_cells = {}  # cell_id -> {"cell": cell, "initial_area": area, "division_threshold": area * ratio}
    division_counter = 0  # For generating unique daughter cell IDs
    cytokinesis_started = False  # Track if any cell has started cytokinesis (for per-step plotting)

    if args.enable_cytokinesis:
        cytokinesis_params = CytokinesisParams(
            constriction_threshold=args.cyto_constriction_threshold if args.cyto_constriction_threshold is not None else 0.1,
            constriction_percentage=args.cyto_constriction_percentage if args.cyto_constriction_threshold is None else None,
            initial_separation_fraction=args.cyto_initial_separation,
            contractile_force_magnitude=args.cyto_force_magnitude,
        )
        # Initialize division tracking for each growing cell
        for cell_id, cell in growing_cells.items():
            A_initial = initial_areas[cell_id]
            division_threshold = A_initial * args.cyto_division_area_ratio
            dividing_cells[cell_id] = {
                "cell": cell,
                "initial_area": A_initial,
                "division_threshold": division_threshold,
                "division_initiated": False,
                "contracting_vertices": None,
            }

    # Parse apoptotic cell IDs (if any)
    apoptotic_cell_ids: Optional[List[str]] = None
    if args.apoptotic_cell_ids:
        apoptotic_cell_ids = [cid.strip() for cid in args.apoptotic_cell_ids.split(",") if cid.strip()]

    # DEBUG: show available cell IDs and parsed apoptotic IDs
    print("Available cell IDs:", [str(c.id) for c in tissue.cells])
    print("Requested apoptotic IDs:", apoptotic_cell_ids)

    print(f"\nGrowth configuration:")
    print(f"  Tissue: {len(tissue.cells)} cells")
    print(f"  Growing cells: {len(growing_cells)}")
    if args.enable_cytokinesis:
        print(f"  Cytokinesis: ENABLED (divide at {args.cyto_division_area_ratio}x initial area)")
    if apoptotic_cell_ids:
        print(f"  Apoptosis: ENABLED (shrink rate={args.apoptosis_shrink_rate}, min area fraction={args.apoptosis_min_area_fraction})")
    for cid, info in growth_info.items():
        print(f"    - {cid}: {info['initial']:.2f} → {info['final']:.2f}")
    print(f"  Growth steps: {args.growth_steps}")
    print(f"  Total steps: {args.total_steps}")
    print(f"  Time step dt: {args.dt}")

    if apoptotic_cell_ids:
        print("  Apoptotic cells:", ", ".join(str(c) for c in apoptotic_cell_ids))

    # Create energy parameters with per-cell target areas
    energy_params = EnergyParameters(
        k_area=args.k_area,
        k_perimeter=args.k_perimeter,
        gamma=args.gamma,
        target_area=target_areas,
    )

    # Set up solver-specific parameters
    if args.solver == "overdamped_force_balance":
        ofb_params = OverdampedForceBalanceParams(
            gamma=args.ofb_gamma,
            noise_strength=args.ofb_noise,
            random_seed=args.ofb_seed,
        )
    else:
        ofb_params = None

    # Apoptosis parameters (if any apoptotic cells specified)
    if apoptotic_cell_ids:
        apoptosis_params = ApoptosisParameters(
            shrink_rate=args.apoptosis_shrink_rate,
            min_area_fraction=args.apoptosis_min_area_fraction,
            removal_area_fraction=args.apoptosis_removal_area_fraction,
            removal_area_absolute=args.apoptosis_removal_area_absolute,
            start_step=args.apoptosis_start_step,
        )
        apoptosis_state = ApoptosisState()
        apoptosis_state.register_cells(tissue, apoptotic_cell_ids, geometry=geometry)
    else:
        apoptosis_params = None
        apoptosis_state = None

    # Create simulation (used only for energy evaluation and convenience)
    sim = Simulation(
        tissue=tissue,
        dt=args.dt,
        energy_params=energy_params,
        validate_each_step=False,
        epsilon=args.epsilon,
        damping=args.damping,
        solver_type=args.solver,
        ofb_params=ofb_params,
        # Do NOT pass apoptosis here; we handle it explicitly in this script
        apoptosis_params=None,
        apoptotic_cell_ids=None,
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
    if len(growing_cells) == 1:
        print(f"{'Step':>6} {'Time':>8} {'Area':>10} {'Target':>10} {'Progress':>8} {'Energy':>12}")
        print("-" * 65)
    elif len(growing_cells) > 1:
        print(f"{'Step':>6} {'Time':>8} {'AvgProgress':>10} {'Energy':>12}")
        print("-" * 45)
    else:
        # No growth: log only step, time, energy, and apoptotic areas if any
        print(f"{'Step':>6} {'Time':>8} {'Energy':>12}  ApoptoticAreas")
        print("-" * 80)

    sim_time = 0.0

    for step in range(args.total_steps):
        # Update target areas only if we have growing cells
        if growth_info:
            for cid, info in growth_info.items():
                if step < args.growth_steps:
                    target_areas[cid] = info["initial"] + step * info["rate"]
                else:
                    target_areas[cid] = info["final"]

        # Apoptosis: update per-cell target areas and override target_areas
        apoptotic_areas = {}
        if apoptosis_state is not None and apoptosis_params is not None:
            update_apoptosis_targets(
                tissue,
                apoptosis_state,
                apoptosis_params,
                step_index=step,
                dt=args.dt,
                geometry=geometry,
            )
            for cid, A_target in build_apoptosis_target_area_mapping(apoptosis_state).items():
                target_areas[cid] = A_target
                # Track current geometric area for logging
                cell = find_cell_by_id(str(cid))
                if cell is not None:
                    apoptotic_areas[cid] = geometry.calculate_area(cell.vertices)

        # Ensure energy_params sees the updated per-cell targets
        energy_params.target_area = target_areas

        if args.solver == "gradient_descent":
            # Compute gradient with respect to GLOBAL vertices
            gradient = compute_global_gradient(tissue, energy_params, geometry, epsilon=args.epsilon)

            # Add contractile forces for dividing cells (if cytokinesis enabled)
            if args.enable_cytokinesis:
                for cid, div_info in list(dividing_cells.items()):
                    if div_info["division_initiated"] and div_info["contracting_vertices"] is not None:
                        cell = div_info["cell"]
                        try:
                            cyto_forces = compute_contractile_forces(cell, tissue, cytokinesis_params)
                            # Apply contractile forces to global vertices
                            for local_idx, global_idx in enumerate(cell.vertex_indices):
                                gradient[global_idx] -= cyto_forces[local_idx]  # Subtract because we do -gradient
                        except (ValueError, IndexError) as e:
                            pass  # Cell may have been split already

            # Update GLOBAL vertices using gradient descent
            tissue.vertices = tissue.vertices - args.dt * args.damping * gradient
            # Reconstruct per-cell vertices from global pool
            tissue.reconstruct_cell_vertices()
        else:
            # Overdamped force-balance on GLOBAL vertex pool
            # Forces from energy: F = -∇E (global gradient)
            grad = compute_global_gradient(tissue, energy_params, geometry, epsilon=args.epsilon)
            forces = -grad

            # Add contractile forces for dividing cells (if cytokinesis enabled)
            if args.enable_cytokinesis:
                for cid, div_info in list(dividing_cells.items()):
                    if div_info["division_initiated"] and div_info["contracting_vertices"] is not None:
                        cell = div_info["cell"]
                        try:
                            cyto_forces = compute_contractile_forces(cell, tissue, cytokinesis_params)
                            # Apply contractile forces to global vertices
                            for local_idx, global_idx in enumerate(cell.vertex_indices):
                                forces[global_idx] += cyto_forces[local_idx]
                        except (ValueError, IndexError) as e:
                            pass  # Cell may have been split already

            # Optional noise term (Gaussian, zero mean)
            if args.ofb_noise and args.ofb_noise > 0.0:
                noise = np.random.default_rng(args.ofb_seed).normal(loc=0.0, scale=args.ofb_noise, size=forces.shape)
            else:
                noise = 0.0
            # Integrate: x <- x + (dt/gamma) * (F + noise)
            tissue.vertices = tissue.vertices + (args.dt / max(args.ofb_gamma, 1e-12)) * (forces + noise)
            # Reconstruct per-cell vertices
            tissue.reconstruct_cell_vertices()

        # Cytokinesis: Check if any growing cell has reached division threshold
        if args.enable_cytokinesis:
            for cid, div_info in list(dividing_cells.items()):
                cell = div_info["cell"]
                current_area = geometry.calculate_area(cell.vertices)

                # Check if cell should start dividing (not yet initiated and reached threshold)
                if not div_info["division_initiated"] and current_area >= div_info["division_threshold"]:
                    try:
                        # Initiate cytokinesis - perpendicular to long axis (axis_angle=None uses PCA)
                        centroid, axis_dir, perp_dir = compute_division_axis(cell, tissue, axis_angle=None)
                        v1_idx, v2_idx = insert_contracting_vertices(cell, tissue, axis_angle=None, params=cytokinesis_params)
                        div_info["division_initiated"] = True
                        div_info["contracting_vertices"] = (v1_idx, v2_idx)
                        cytokinesis_started = True  # Mark that cytokinesis has begun
                        print(f"[cytokinesis] step={step} Cell {cid}: Division initiated (area={current_area:.2f} >= threshold={div_info['division_threshold']:.2f})")
                        print(f"              Contracting vertices: global indices {v1_idx}, {v2_idx}")
                    except ValueError as e:
                        print(f"[cytokinesis] step={step} Cell {cid}: Failed to initiate - {e}")

                # Check if dividing cell should split
                elif div_info["division_initiated"]:
                    try:
                        if check_constriction(cell, tissue, cytokinesis_params):
                            # Split the cell (let split_cell auto-generate appropriate IDs)
                            division_counter += 1
                            daughter1, daughter2 = split_cell(cell, tissue)

                            # Get the actual IDs from the daughter cells
                            d1_id = daughter1.id
                            d2_id = daughter2.id

                            print(f"[cytokinesis] step={step} Cell {cid}: SPLIT into {d1_id} and {d2_id}")
                            d1_area = geometry.calculate_area(daughter1.vertices)
                            d2_area = geometry.calculate_area(daughter2.vertices)
                            print(f"              Daughter areas: {d1_id}={d1_area:.2f}, {d2_id}={d2_area:.2f}")

                            # Remove from dividing_cells tracking
                            del dividing_cells[cid]

                            # Update growing_cells reference if needed
                            if cid in growing_cells:
                                del growing_cells[cid]

                            # Update target_areas with new daughter cells
                            target_areas[d1_id] = d1_area
                            target_areas[d2_id] = d2_area
                            initial_areas[d1_id] = d1_area
                            initial_areas[d2_id] = d2_area

                    except (ValueError, IndexError) as e:
                        print(f"[cytokinesis] step={step} Cell {cid}: Split check failed - {e}")

        # Apoptosis: Update state of apoptotic cells (if any)
        # REMOVE old inline apoptosis block here (now handled by Simulation via apoptosis module)
        # Previously:
        # if apoptotic_cell_ids and step >= args.apoptosis_start_step:
        #     for cid in apoptotic_cell_ids:
        #         ... manual new_target_area and initial_areas[cid] access ...

        # Per-step plotting during cytokinesis (save each frame without showing)
        if args.enable_cytokinesis and cytokinesis_started and args.plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=1, show_cell_ids=True, show_neighbor_counts=False)

                # Add info about contracting vertices
                for cid, div_info in dividing_cells.items():
                    if div_info["division_initiated"] and div_info["contracting_vertices"] is not None:
                        v1_idx, v2_idx = div_info["contracting_vertices"]
                        if v1_idx < tissue.vertices.shape[0] and v2_idx < tissue.vertices.shape[0]:
                            pos1 = tissue.vertices[v1_idx]
                            pos2 = tissue.vertices[v2_idx]
                            distance = np.linalg.norm(pos2 - pos1)
                            # Draw contracting vertices as large red dots
                            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r-', linewidth=2, zorder=10)
                            ax.scatter([pos1[0], pos2[0]], [pos1[1], pos2[1]], c='red', s=100, zorder=11, marker='o')
                            ax.set_title(f"Step {step}: Cell {cid} dividing (distance={distance:.3f})", fontsize=12, fontweight='bold')

                step_plot_path = sim_folder / f"cytokinesis_step_{step:04d}.png"
                fig.savefig(step_plot_path, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[plot] step={step} failed: {e}")

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

        # Apoptosis removal: actually delete cells that are small enough
        if apoptosis_state is not None and apoptosis_params is not None:
            to_remove = collect_cells_to_remove(
                tissue,
                apoptosis_state,
                apoptosis_params,
                geometry=geometry,
            )
            if to_remove:
                print(f"[apoptosis] step={step} removing cells: {to_remove}")
                tissue.remove_cells(to_remove)
                # Rebuild global vertices and per-cell vertices after topology change
                tissue.build_global_vertices(tol=1e-10)
                tissue.reconstruct_cell_vertices()
                # Drop their target areas
                for cid in to_remove:
                    target_areas.pop(cid, None)
                # Update energy_params mapping
                energy_params.target_area = target_areas

        # Advance time
        sim_time += args.dt

        current_energy = sim.total_energy()
        cell_progress = {}
        csv_row = [step, sim_time, current_energy]

        # Get current state for all growing cells
        if growth_info:
            for cid, info in list(growth_info.items()):
                cell = info["cell"]
                # Check if cell still exists in tissue (may have been split)
                cell_exists = any(c.id == cell.id for c in tissue.cells)
                if cell_exists:
                    current_area = geometry.calculate_area(cell.vertices)
                    current_target = target_areas.get(cid, info["final"])
                    progress = (current_area - info["initial"]) / (info["final"] - info["initial"]) * 100
                    cell_progress[cid] = {"area": current_area, "target": current_target, "progress": progress}
                    csv_row.extend([current_area, current_target, progress])
                else:
                    # Cell was split - mark as complete (100%)
                    cell_progress[cid] = {"area": info["final"], "target": info["final"], "progress": 100.0}
                    csv_row.extend([info["final"], info["final"], 100.0])

        csv_data.append(csv_row)

        # Log progress
        if step % args.log_interval == 0 or step == args.total_steps - 1:
            if len(growing_cells) == 1 and len(cell_progress) == 1:
                cid = list(cell_progress.keys())[0]
                p = cell_progress[cid]
                print(f"{step:6d} {sim_time:8.3f} {p['area']:10.2f} {p['target']:10.2f} {p['progress']:7.1f}% {current_energy:12.2f}")
            elif len(growing_cells) > 1 and cell_progress:
                avg_progress = sum(p["progress"] for p in cell_progress.values()) / max(len(cell_progress), 1)
                print(f"{step:6d} {sim_time:8.3f} {avg_progress:9.1f}% {current_energy:12.2f}")
            else:
                # No growth: just log step, time, energy and apoptotic areas
                if apoptotic_areas:
                    ap_str = ", ".join(
                        f"{cid} A={apoptotic_areas[cid]:.4f} T={target_areas.get(cid, float('nan')):.4f}"
                        for cid in sorted(apoptotic_areas.keys(), key=str)
                    )
                else:
                    ap_str = "(no apoptotic cells found this step)"
                print(f"{step:6d} {sim_time:8.3f} {current_energy:12.2f}  {ap_str}")

        # Check stopping criterion (all cells reached target area or were split)
        all_complete = True
        if growth_info:
            for cid in list(growth_info.keys()):
                if cid in cell_progress:
                    if cell_progress[cid]["progress"] < 99.0:
                        all_complete = False
                        break

        # Also stop if all dividing cells have completed (been split)
        if args.enable_cytokinesis and len(dividing_cells) == 0 and division_counter > 0:
            print(f"\n✓ All cells have divided at step {step}!")
            break

        if growth_info and all_complete and not args.enable_cytokinesis:
            print(f"\n✓ Growth complete at step {step}! All cells reached target area.")
            break

    # Final statistics
    print(f"\nFinal statistics:")
    print(f"  Initial growing cells: {len(growth_info)}")
    print(f"  Final tissue cells: {len(tissue.cells)}")

    if args.enable_cytokinesis:
        print(f"  Division events: {division_counter}")

    if growth_info:
        all_reached = True
        for cid, info in growth_info.items():
            cell = info["cell"]
            # Check if cell still exists (may have been split)
            cell_exists = any(c.id == cell.id for c in tissue.cells)
            if cell_exists:
                final_area = geometry.calculate_area(cell.vertices)
                ratio = final_area / info["initial"]
                reached = final_area >= 0.99 * info["final"]
                all_reached = all_reached and reached
                print(f"    - {cid}: {final_area:.2f} ({ratio:.2f}× initial) {'✓' if reached else '✗'}")
            else:
                print(f"    - {cid}: DIVIDED into daughter cells")

    print(f"  Final energy: {sim.total_energy():.2f}")
    print(f"  Total simulation time: {sim_time:.3f}")

    # Show final tissue
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_tissue(tissue, ax=ax, show_vertices=True, fill=True, alpha=1, show_cell_ids=True, show_neighbor_counts=False)

            # Build title showing original growing cells and any daughters
            if args.enable_cytokinesis and division_counter > 0:
                ax.set_title(f"Final Tissue ({len(tissue.cells)} cells, {division_counter} divisions)", fontsize=12, fontweight='bold')
            else:
                cell_ids_display = ", ".join(str(c) for c in growth_info.keys())
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
