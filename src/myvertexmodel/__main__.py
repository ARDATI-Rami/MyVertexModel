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
    parser.add_argument("--no-energy-print", action="store_true", help="Suppress energy output printing.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    """Main entry point for CLI demo.

    Builds a tissue, runs a short simulation, and prints initial/final energy.
    Optionally plots the final tissue state if --plot is provided.
    """
    args = parse_args(argv)

    tissue = build_grid_tissue(grid_size=args.grid_size)
    sim = Simulation(tissue=tissue, dt=args.dt, energy_params=EnergyParameters())

    initial_energy = sim.total_energy()
    for _ in range(args.steps):
        sim.step()
    final_energy = sim.total_energy()

    if not args.no_energy_print:
        print(f"Initial energy: {initial_energy:.6f}")
        print(f"Final energy:   {final_energy:.6f}")

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
            # Show only if backend supports interaction and user requested --plot explicitly
            if args.plot and not backend.startswith("agg"):
                plt.show()
            elif args.plot and backend.startswith("agg") and args.output is None:
                print("Backend is non-interactive; image saved instead of shown. Use --output to control filename.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
