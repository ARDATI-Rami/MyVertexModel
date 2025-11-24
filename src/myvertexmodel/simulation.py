"""
Simulation engine for vertex model dynamics.
"""

import numpy as np
from typing import Optional
from .core import Tissue, EnergyParameters, tissue_energy
from .geometry import GeometryCalculator


class Simulation:
    """Main simulation class for vertex model dynamics.

    Attributes:
        tissue: Tissue being simulated.
        dt: Time step.
        time: Current simulation time.
        geometry: Geometry calculator instance.
        energy_params: Parameters used for energy evaluation.
    """

    def __init__(self, tissue: Optional[Tissue] = None, dt: float = 0.01, energy_params: Optional[EnergyParameters] = None):
        """
        Initialize a simulation.
        
        Args:
            tissue: Tissue to simulate (creates empty if None)
            dt: Time step for simulation
            energy_params: Optional EnergyParameters instance (default constructed if None)
        """
        self.tissue = tissue if tissue is not None else Tissue()
        self.dt = dt
        self.time = 0.0
        self.geometry = GeometryCalculator()
        self.energy_params = energy_params if energy_params is not None else EnergyParameters()

    def step(self):
        """Perform a single simulation step.

        Implements a naive explicit gradient descent on cell vertex positions.
        For each vertex coordinate (x, y) of every cell, estimates the gradient
        of the total energy via central finite differences:

            dE/dx ≈ [E(x+ε) - E(x-ε)] / (2ε)
            dE/dy ≈ [E(y+ε) - E(y-ε)] / (2ε)

        Then updates vertices by:
            V_new = V_old - dt * grad

        Notes:
            - Uses per-cell local vertex arrays (future: migrate to global Tissue.vertices).
            - Not optimized; O(N_vertices * cells) energy evaluations.
            - ε chosen small; may need tuning for stability/accuracy.
            - Does NOT handle topological changes (T1 transitions, division, etc.).
        """
        epsilon = 1e-6
        # Assemble gradients for each cell
        for cell in self.tissue.cells:
            verts = cell.vertices
            if verts.shape[0] == 0:
                continue  # nothing to do
            grad = np.zeros_like(verts)
            # Iterate vertices
            for i in range(verts.shape[0]):
                # X gradient
                original_x = verts[i, 0]
                verts[i, 0] = original_x + epsilon
                e_plus = tissue_energy(self.tissue, self.energy_params, self.geometry)
                verts[i, 0] = original_x - epsilon
                e_minus = tissue_energy(self.tissue, self.energy_params, self.geometry)
                verts[i, 0] = original_x  # restore
                grad[i, 0] = (e_plus - e_minus) / (2 * epsilon)

                # Y gradient
                original_y = verts[i, 1]
                verts[i, 1] = original_y + epsilon
                e_plus = tissue_energy(self.tissue, self.energy_params, self.geometry)
                verts[i, 1] = original_y - epsilon
                e_minus = tissue_energy(self.tissue, self.energy_params, self.geometry)
                verts[i, 1] = original_y  # restore
                grad[i, 1] = (e_plus - e_minus) / (2 * epsilon)
            # Gradient descent update
            cell.vertices = verts - self.dt * grad
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
