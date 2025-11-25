"""
Simulation engine for vertex model dynamics.
"""

import numpy as np
from typing import Optional
from .core import Tissue, Cell, EnergyParameters, tissue_energy
from .geometry import GeometryCalculator


def finite_difference_cell_gradient(
    cell: Cell,
    tissue: Tissue,
    energy_params: EnergyParameters,
    geometry: GeometryCalculator,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute the energy gradient for a single cell using finite differences.

    Uses central finite difference approximation:
        dE/dx_i ≈ [E(x_i + ε) - E(x_i - ε)] / (2ε)
        dE/dy_i ≈ [E(y_i + ε) - E(y_i - ε)] / (2ε)

    Args:
        cell: Cell for which to compute gradient.
        tissue: Tissue containing the cell (needed for total energy evaluation).
        energy_params: Energy parameters for energy computation.
        geometry: GeometryCalculator instance.
        epsilon: Finite difference step size (default: 1e-6).

    Returns:
        np.ndarray: Gradient array of shape (N, 2) where N is the number of vertices.
                   Returns empty array (0, 2) if cell has no vertices.

    Notes:
        - Modifies cell.vertices temporarily during computation but restores original values.
        - Computes total tissue energy at each perturbed state (includes all cells).
        - Not optimized; performs 4*N tissue energy evaluations where N = number of vertices.
    """
    verts = cell.vertices
    if verts.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    grad = np.zeros_like(verts)

    # Iterate over each vertex
    for i in range(verts.shape[0]):
        # X gradient
        original_x = verts[i, 0]
        verts[i, 0] = original_x + epsilon
        e_plus = tissue_energy(tissue, energy_params, geometry)
        verts[i, 0] = original_x - epsilon
        e_minus = tissue_energy(tissue, energy_params, geometry)
        verts[i, 0] = original_x  # restore
        grad[i, 0] = (e_plus - e_minus) / (2 * epsilon)

        # Y gradient
        original_y = verts[i, 1]
        verts[i, 1] = original_y + epsilon
        e_plus = tissue_energy(tissue, energy_params, geometry)
        verts[i, 1] = original_y - epsilon
        e_minus = tissue_energy(tissue, energy_params, geometry)
        verts[i, 1] = original_y  # restore
        grad[i, 1] = (e_plus - e_minus) / (2 * epsilon)

    return grad


class Simulation:
    """Main simulation class for vertex model dynamics.

    Attributes:
        tissue: Tissue being simulated.
        dt: Time step.
        time: Current simulation time.
        geometry: Geometry calculator instance.
        energy_params: Parameters used for energy evaluation.
        epsilon: Finite-difference step size for gradient estimation.
        damping: Scalar multiplier applied to gradient descent updates ("learning rate" factor).
    """

    def __init__(
        self,
        tissue: Optional[Tissue] = None,
        dt: float = 0.01,
        energy_params: Optional[EnergyParameters] = None,
        validate_each_step: bool = False,
        epsilon: float = 1e-6,
        damping: float = 1.0,
    ):
        """Initialize a simulation.

        Args:
            tissue: Tissue to simulate (creates empty if None)
            dt: Time step for simulation
            energy_params: Optional EnergyParameters instance (default constructed if None)
            validate_each_step: If True, validate tissue structure after each step
            epsilon: Finite-difference step size used for gradient estimation.
            damping: Multiplier on the gradient when updating vertex positions (allows tuning descent magnitude).
        """
        self.tissue = tissue if tissue is not None else Tissue()
        self.dt = dt
        self.time = 0.0
        self.geometry = GeometryCalculator()
        self.energy_params = energy_params if energy_params is not None else EnergyParameters()
        self.validate_each_step = validate_each_step
        self.epsilon = epsilon
        self.damping = damping

    def step(self):
        """Perform a single simulation step.

        Implements a naive explicit gradient descent on cell vertex positions.
        For each vertex coordinate (x, y) of every cell, estimates the gradient
        of the total energy via central finite differences using the
        finite_difference_cell_gradient helper function.

        Then updates vertices by:
            V_new = V_old - dt * damping * grad

        Notes:
            - Uses per-cell local vertex arrays (future: migrate to global Tissue.vertices).
            - Not optimized; O(N_vertices * cells) energy evaluations.
            - ε (epsilon) and damping are user-configurable via Simulation constructor.
            - Does NOT handle topological changes (T1 transitions, division, etc.).
        """
        # Validate before computing energy if requested
        if self.validate_each_step:
            self.tissue.validate()

        # Compute gradient and update each cell
        for cell in self.tissue.cells:
            if cell.vertices.shape[0] == 0:
                continue  # nothing to do

            # Compute gradient using finite differences
            grad = finite_difference_cell_gradient(
                cell, self.tissue, self.energy_params, self.geometry, epsilon=self.epsilon
            )

            # Gradient descent update
            cell.vertices = cell.vertices - self.dt * self.damping * grad

        # Advance time after position updates
        self.time += self.dt

    def run(self, n_steps: int = 100):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            n_steps: Number of simulation steps to run
        """
        for _ in range(n_steps):
            self.step()

    def total_energy(self) -> float:
        """Compute total tissue energy using current energy parameters.

        Returns:
            float: Total energy value (sum over cells).
        """
        return tissue_energy(self.tissue, self.energy_params, self.geometry)

    def __repr__(self):
        return f"Simulation(time={self.time:.2f}, dt={self.dt})"
