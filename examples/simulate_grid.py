"""simulate_grid.py

Run a vertex-model simulation on an Nx × Ny rectangular grid of square cells.

Usage (from project root):
    python examples/simulate_grid.py --nx 3 --ny 2 --cell-size 1.0 \
        --steps 20 --dt 0.01 --log-interval 5 --plot \
        --k-area 1.0 --k-perimeter 0.1 --gamma 0.05 --target-area 1.0

Key options:
    --nx, --ny          Grid dimensions (# cells in x and y)
    --cell-size         Linear size of each cell (default 1.0)
    --steps, --dt       Number of steps and time step size
    --log-interval      Print energy every this many steps (includes initial step 0)
    --plot              Plot initial and final tissue states
    --k-area            Area elasticity coefficient
    --k-perimeter       Perimeter contractility coefficient
    --gamma             Line tension parameter
    --target-area       Preferred cell area
    --epsilon           Finite difference step size for gradient computation
    --damping           Damping/learning rate for gradient descent updates

Notes:
    - Avoids duplicating the square-grid helper in myvertexmodel.__main__; reuses build_grid_tissue for square grids.
    - For rectangular grids (nx != ny) constructs explicitly.
    - Prints energy at t=0 and then every log_interval steps after updates.
"""
from __future__ import annotations
import argparse
import sys
from typing import Optional, List
import numpy as np
from myvertexmodel import Tissue, Cell, Simulation, EnergyParameters, plot_tissue
from myvertexmodel.__main__ import build_grid_tissue


def build_rect_grid_tissue(nx: int, ny: int, cell_size: float = 1.0) -> Tissue:
    """Build an Nx × Ny rectangular grid of square cells.

    Uses existing build_grid_tissue when nx == ny to avoid code duplication.

    Args:
        nx: Number of cells along x-axis (must be >= 1)
        ny: Number of cells along y-axis (must be >= 1)
        cell_size: Linear size of each square cell

    Returns:
        Tissue populated with nx * ny cells.
    """
    if nx <= 0 or ny <= 0:
        raise ValueError("nx and ny must be positive integers")
    if nx == ny:
        return build_grid_tissue(grid_size=nx, cell_size=cell_size)

    tissue = Tissue()
    cid = 1
    for i in range(nx):
        for j in range(ny):
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
    parser = argparse.ArgumentParser(description="Run a rectangular grid tissue simulation.")
    parser.add_argument("--nx", type=int, default=2, help="Number of cells along x-axis.")
    parser.add_argument("--ny", type=int, default=2, help="Number of cells along y-axis.")
    parser.add_argument("--cell-size", type=float, default=1.0, help="Linear size of each cell.")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps.")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step size.")
    parser.add_argument("--log-interval", type=int, default=1, help="Print energy every N steps (>=1).")
    parser.add_argument("--plot", action="store_true", help="Plot initial and final tissue state.")
    # Energy parameters
    parser.add_argument("--k-area", type=float, default=1.0, help="Area elasticity coefficient k_area.")
    parser.add_argument("--k-perimeter", type=float, default=0.1, help="Perimeter contractility coefficient k_perimeter.")
    parser.add_argument("--gamma", type=float, default=0.05, help="Line tension parameter gamma.")
    parser.add_argument("--target-area", type=float, default=1.0, help="Preferred cell area A0.")
    # Simulation gradient settings
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Finite-difference step size for gradient.")
    parser.add_argument("--damping", type=float, default=1.0, help="Damping (learning-rate multiplier) for updates.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    tissue = build_rect_grid_tissue(args.nx, args.ny, cell_size=args.cell_size)
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

    if args.plot:
        try:
            plot_tissue(sim.tissue)
        except Exception as e:  # pragma: no cover
            print(f"Initial plot failed: {e}")

    initial_energy = sim.total_energy()
    print(f"step 0 time {sim.time:.6f} energy {initial_energy:.6f}")

    if args.log_interval < 1:
        raise ValueError("--log-interval must be >= 1")

    for step in range(1, args.steps + 1):
        sim.step()
        if step % args.log_interval == 0:
            e = sim.total_energy()
            print(f"step {step} time {sim.time:.6f} energy {e:.6f}")

    if args.plot:
        try:
            plot_tissue(sim.tissue)
        except Exception as e:  # pragma: no cover
            print(f"Final plot failed: {e}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

