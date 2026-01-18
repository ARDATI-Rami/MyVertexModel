"""simulate_cell_apoptosis.py

Simulate apoptosis of a specific cell (e.g., cell 7) in a 14-cell honeycomb tissue.

This script uses Simulation with use_global_gradient=True to properly handle mechanical
shrinking in tissues with shared vertices. The apoptotic cell will shrink as:
    - Apoptosis target area decays exponentially
    - Global gradient descent relaxes all vertices toward the new energy minimum
    - Removal triggers when geometric area crosses thresholds

Usage:
    python examples/simulate_cell_apoptosis.py --plot

"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from myvertexmodel import EnergyParameters, GeometryCalculator, Simulation, plot_tissue
from myvertexmodel.apoptosis import ApoptosisParameters
from myvertexmodel import build_honeycomb_2_3_4_3_2



def parse_args():
    parser = argparse.ArgumentParser(description="Simulate apoptosis of a cell in honeycomb tissue.")
    parser.add_argument("--plot", action="store_true", help="Save initial/final plots (PNG).")
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="apoptosis_frames",
        help="Directory to write PNG frames into when --plot is enabled.",
    )
    parser.add_argument("--plot-every", type=int, default=50, help="If >0, also save intermediate plots every N steps.")
    parser.add_argument("--total-steps", type=int, default=300, help="Total simulation steps.")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step size.")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Finite-difference step for gradient estimation.")
    parser.add_argument("--damping", type=float, default=1.0, help="Gradient descent damping factor.")
    parser.add_argument("--log-every", type=int, default=1, help="Print area/target area every N steps.")
    parser.add_argument(
        "--apoptotic-cell-ids",
        type=str,
        nargs="+",
        default=["7"],
        help="Cell IDs to undergo apoptosis (space-separated, e.g., --apoptotic-cell-ids 7 3 11).",
    )
    parser.add_argument(
        "--removal-strategy",
        type=str,
        default="shrink",
        choices=["shrink", "merge"],
        help="Apoptosis removal strategy (shrink or merge).",
    )
    parser.add_argument(
        "--shrink-rate",
        type=float,
        default=0.5,
        help="Target area shrink rate (fraction per unit time, default: 0.5).",
    )
    parser.add_argument(
        "--removal-area-fraction",
        type=float,
        default=0.1,
        help="Fraction of initial area below which cell is removed (default: 0.1).",
    )
    parser.add_argument(
        "--removal-area-absolute",
        type=float,
        default=0.35,
        help="Absolute area threshold below which cell is removed (default: 0.35).",
    )
    return parser.parse_args()


def _find_cell_by_id(tissue, cell_id: str):
    for c in tissue.cells:
        if str(c.id) == str(cell_id):
            return c
    return None



def main():
    args = parse_args()

    tissue = build_honeycomb_2_3_4_3_2()
    geometry = GeometryCalculator()

    apoptotic_cell_ids = [str(cid) for cid in args.apoptotic_cell_ids]

    # IMPORTANT: Start with per-cell target areas based on actual geometric areas
    # so the system is initially near equilibrium.
    target_areas = {cell.id: geometry.calculate_area(cell.vertices) for cell in tissue.cells}

    energy_params = EnergyParameters(
        k_area=1.0,
        k_perimeter=0.1,
        gamma=0.05,
        target_area=target_areas,
    )

    # Apoptosis configuration
    apoptosis_params = ApoptosisParameters(
        shrink_rate=args.shrink_rate,
        min_area_fraction=0.05,
        removal_area_fraction=args.removal_area_fraction,
        removal_area_absolute=args.removal_area_absolute,
        min_vertices=3,
        start_step=0,
        removal_strategy=args.removal_strategy,
    )

    # Create Simulation with apoptosis enabled and global gradient descent
    # use_global_gradient=True is essential for proper mechanical shrinking in shared-vertex tissues
    sim = Simulation(
        tissue=tissue,
        dt=args.dt,
        energy_params=energy_params,
        validate_each_step=False,
        epsilon=args.epsilon,
        damping=args.damping,
        solver_type="gradient_descent",
        ofb_params=None,
        apoptosis_params=apoptosis_params,
        apoptotic_cell_ids=apoptotic_cell_ids,
        use_global_gradient=True,  # Essential for apoptosis to work correctly!
    )

    def save_frame(step: int, label: str):
        from pathlib import Path

        out_dir = Path(args.plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        ax = plot_tissue(sim.tissue, show_cell_ids=True, show_vertices=True)
        ax.set_title(label)
        out_path = out_dir / f"cell_apoptosis_{step:05d}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

    if args.plot:
        save_frame(0, f"Step 0 (apoptosis of cells {', '.join(apoptotic_cell_ids)})")

    # Run simulation using Simulation.step()
    for step in range(1, args.total_steps + 1):
        sim.step()

        if args.log_every and (step % args.log_every == 0 or step == 1):
            # Build unified status line for all apoptotic cells
            status_parts = [f"step={step:5d} time={sim.time:8.3f}"]
            for cell_id in apoptotic_cell_ids:
                cell = _find_cell_by_id(sim.tissue, cell_id)
                if cell is not None:
                    A_geom = geometry.calculate_area(cell.vertices)
                    A_target = sim.energy_params.target_area.get(cell.id) if isinstance(sim.energy_params.target_area, dict) else sim.energy_params.target_area
                    n_vertices = cell.vertices.shape[0]
                    status_parts.append(f"cell={cell_id}[A_geom={A_geom:7.4f} A_target={A_target:7.4f} n={n_vertices}]")
                else:
                    status_parts.append(f"cell={cell_id}[REMOVED]")
            print("  ".join(status_parts))

        if args.plot and args.plot_every and (step % args.plot_every == 0):
            save_frame(step, f"Step {step}")

    if args.plot:
        save_frame(args.total_steps, f"Final (step {args.total_steps})")


if __name__ == "__main__":
    main()

