"""Command-line demo for the MyVertexModel package.

Run with:
    python -m myvertexmodel [--steps N] [--dt DT] [--grid-size G] [--plot]

Example:
    python -m myvertexmodel --steps 10 --grid-size 2 --plot
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
from typing import List, Optional
from pathlib import Path
from . import Tissue, Cell, Simulation, plot_tissue, EnergyParameters


def build_grid_tissue(grid_size: int = 2, cell_size: float = 1.0) -> Tissue:
    """Construct a simple square grid of non-overlapping cells.

    Each cell is currently independent (no shared vertex arrays yet).
    Future versions will collapse duplicate vertices into a global pool.

    Args:
        grid_size: Number of cells along one axis (produces grid_size^2 cells).
        cell_size: Linear size of each square cell.

    Returns:
        Tissue with populated cells.
    """
    tissue = Tissue()
    cid = 1
    for i in range(grid_size):
        for j in range(grid_size):
            x0, y0 = i * cell_size, j * cell_size
            verts = np.array([
                [x0, y0],
                [x0 + cell_size, y0],
                [x0 + cell_size, y0 + cell_size],
                [x0, y0 + cell_size],
            ], dtype=float)
            tissue.add_cell(Cell(cell_id=cid, vertices=verts))
            cid += 1
    return tissue


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny vertex model demo simulation.")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps to run.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument("--grid-size", type=int, default=2, help="Grid size (creates grid-size^2 cells).")
    parser.add_argument("--plot", action="store_true", help="Display final tissue plot.")
    parser.add_argument("--output", type=str, default=None, help="If set, save plot image to this path (implies --plot).")
    parser.add_argument("--no-energy-print", action="store_true", help="Suppress initial/final energy output printing.")
    # Logging / instrumentation
    parser.add_argument("--log-energy", action="store_true", help="Log energy vs time samples during the run.")
    parser.add_argument("--log-interval", type=int, default=1, help="Interval (in steps) between energy log samples.")
    # Energy parameterization
    parser.add_argument("--k-area", type=float, default=1.0, help="Area elasticity coefficient k_area.")
    parser.add_argument("--k-perimeter", type=float, default=0.1, help="Perimeter contractility coefficient k_perimeter.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Line tension parameter gamma.")
    parser.add_argument("--target-area", type=float, default=1.0, help="Preferred target cell area A0.")
    # Simulation gradient configuration
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Finite-difference step size for gradient estimation.")
    parser.add_argument("--damping", type=float, default=1.0, help="Damping / learning-rate factor applied to gradient descent.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    """Main entry point for CLI demo.

    Builds a tissue, runs a short simulation, and prints initial/final energy.
    Optionally plots the final tissue state if --plot is provided.
    When --log-energy is specified, uses run_with_logging to record energy samples.
    """
    args = parse_args(argv)

    tissue = build_grid_tissue(grid_size=args.grid_size)
    energy_params = EnergyParameters(
        k_area=args.k_area,
        k_perimeter=args.k_perimeter,
        gamma=args.gamma,
        target_area=args.target_area,
    )
    sim = Simulation(
        tissue=tissue,
        dt=args.dt,
        energy_params=energy_params,
        epsilon=args.epsilon,
        damping=args.damping,
    )

    initial_energy = sim.total_energy()
    samples = None
    if args.log_energy:
        samples = sim.run_with_logging(n_steps=args.steps, log_interval=args.log_interval)
    else:
        for _ in range(args.steps):
            sim.step()
    final_energy = sim.total_energy()

    if not args.no_energy_print:
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy:   {final_energy:.6f}")
        print(
            f"Parameters: k_area={args.k_area} k_perimeter={args.k_perimeter} gamma={args.gamma} "
            f"target_area={args.target_area} epsilon={args.epsilon} damping={args.damping} dt={args.dt} steps={args.steps}"
        )

    if args.log_energy and samples is not None:
        # Print table header
        print("\nEnergy log (time, energy):")
        print("time,energy")
        for t, e in samples:
            print(f"{t:.6f},{e:.6f}")
        # If output path provided, also write CSV next to plot image
        if args.output:
            out_base = Path(args.output)
            csv_path = out_base.with_suffix("")  # strip existing suffix
            # Build energy log filename: original stem + '_energy.csv'
            energy_csv = out_base.parent / f"{out_base.stem}_energy.csv"
            try:
                with open(energy_csv, "w", encoding="utf-8") as f:
                    f.write("time,energy\n")
                    for t, e in samples:
                        f.write(f"{t:.6f},{e:.6f}\n")
                print(f"Saved energy log CSV to {energy_csv}")
            except Exception as e:  # pragma: no cover
                print(f"Failed to write energy CSV: {e}")

    if args.plot or args.output:
        try:
            import matplotlib
            import matplotlib.pyplot as plt  # noqa: F401
        except Exception as e:  # pragma: no cover
            print(f"Plotting unavailable: {e}")
        else:
            plot_tissue(sim.tissue)
            import matplotlib.pyplot as plt
            plt.title(f"Final tissue (steps={args.steps})")
            backend = matplotlib.get_backend().lower()
            # Determine output filename
            out_path = args.output or ("tissue_plot.png" if backend.startswith("agg") else None)
            if out_path is not None:
                plt.savefig(out_path, dpi=150)
                print(f"Saved plot to {out_path} (backend={backend})")
            if args.plot and not backend.startswith("agg"):
                plt.show()
            elif args.plot and backend.startswith("agg") and args.output is None:
                print("Backend is non-interactive; image saved instead of shown. Use --output to control filename.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
