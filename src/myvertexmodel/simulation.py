"""
Simulation engine for vertex model dynamics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, Callable, Dict, Any, Sequence, Union
from .core import Tissue, Cell
from .energy import EnergyParameters, tissue_energy
from .geometry import GeometryCalculator
from .apoptosis import ApoptosisParameters, ApoptosisState, update_apoptosis_targets, collect_cells_to_remove, build_apoptosis_target_area_mapping


@dataclass
class OverdampedForceBalanceParams:
    """Parameters for the overdamped force-balance dynamics solver.

    The overdamped force-balance dynamics are governed by:
        γ dx_k/dt = F_k = -∇_{x_k} E + F_k^{active} + η_k

    where:
        γ: friction coefficient (viscous drag)
        -∇_{x_k} E: mechanical force derived from energy gradient
        F_k^{active}: active forces (e.g., actomyosin contractility)
        η_k: stochastic noise term (optional)

    Attributes:
        gamma: Friction coefficient (viscous drag). Higher values slow dynamics. Must be > 0.
        noise_strength: Intensity of thermal/stochastic fluctuations. Set to 0 for deterministic dynamics.
        active_force_func: Optional callable for computing active forces. Signature:
            (cell: Cell, tissue: Tissue, params: Dict[str, Any]) -> np.ndarray
            Should return an array of shape (N, 2) where N is the number of vertices.
            If None, no active forces are applied.
        active_force_params: Dictionary of parameters passed to the active force function.
        random_seed: Optional seed for the random number generator (for reproducibility).
    """
    gamma: float = 1.0
    noise_strength: float = 0.0
    active_force_func: Optional[Callable[[Cell, Tissue, Dict[str, Any]], np.ndarray]] = None
    active_force_params: Dict[str, Any] = field(default_factory=dict)
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.noise_strength < 0:
            raise ValueError(f"noise_strength must be >= 0, got {self.noise_strength}")


def compute_active_forces(
    cell: Cell,
    tissue: Tissue,
    active_force_func: Optional[Callable[[Cell, Tissue, Dict[str, Any]], np.ndarray]],
    active_force_params: Dict[str, Any]
) -> np.ndarray:
    """Compute active forces for a cell's vertices.

    Active forces represent non-conservative forces such as actomyosin contractility,
    polarity-driven forces, or other biological active processes.

    Args:
        cell: Cell for which to compute active forces.
        tissue: Tissue containing the cell (may be used for neighbor interactions).
        active_force_func: Callable that computes active forces. If None, returns zeros.
        active_force_params: Parameters passed to the active force function.

    Returns:
        np.ndarray: Active force array of shape (N, 2) where N is the number of vertices.
    """
    if active_force_func is None:
        return np.zeros_like(cell.vertices)
    return active_force_func(cell, tissue, active_force_params)


def overdamped_force_balance_step(
    cell: Cell,
    tissue: Tissue,
    energy_params: EnergyParameters,
    geometry: GeometryCalculator,
    ofb_params: OverdampedForceBalanceParams,
    dt: float,
    epsilon: float = 1e-6,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Perform an overdamped force-balance update for a single cell.

    Implements the Euler-Maruyama integration scheme for overdamped Langevin dynamics:
        x_new = x_old + (dt / γ) * F

    where:
        F = -∇E + F_active + η

    and η is Gaussian noise with variance 2 * noise_strength * dt.

    Args:
        cell: Cell to update.
        tissue: Tissue containing the cell.
        energy_params: Energy parameters for gradient computation.
        geometry: GeometryCalculator instance.
        ofb_params: Overdamped force-balance parameters (γ, noise, active forces).
        dt: Time step.
        epsilon: Finite difference step size for gradient estimation.
        rng: Optional NumPy random generator for reproducible noise.

    Returns:
        np.ndarray: New vertex positions of shape (N, 2).
    """
    verts = cell.vertices
    if verts.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    # 1. Compute energy gradient (mechanical force = -gradE)
    grad_e = finite_difference_cell_gradient(
        cell, tissue, energy_params, geometry, epsilon=epsilon
    )
    mechanical_force = -grad_e

    # 2. Compute active forces
    active_force = compute_active_forces(
        cell, tissue, ofb_params.active_force_func, ofb_params.active_force_params
    )

    # 3. Compute noise term (if enabled)
    if ofb_params.noise_strength > 0 and dt > 0:
        if rng is None:
            noise = np.sqrt(2 * ofb_params.noise_strength * dt) * np.random.standard_normal(verts.shape)
        else:
            noise = np.sqrt(2 * ofb_params.noise_strength * dt) * rng.standard_normal(verts.shape)
    else:
        noise = np.zeros_like(verts)

    # 4. Total force
    total_force = mechanical_force + active_force + noise

    # 5. Overdamped update: dx = (dt / γ) * F
    new_vertices = verts + (dt / ofb_params.gamma) * total_force

    return new_vertices


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


def finite_difference_global_gradient(
    tissue: Tissue,
    energy_params: EnergyParameters,
    geometry: GeometryCalculator,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute the energy gradient for the GLOBAL vertex pool using finite differences.

    This function operates on tissue.vertices (the shared global vertex array) and
    is necessary for proper mechanical relaxation in tissues with shared vertices,
    such as honeycombs. Per-cell gradients don't properly communicate forces between
    neighbors because they don't respect the shared vertex constraint.

    Uses central finite difference approximation:
        dE/dx_i ≈ [E(x_i + ε) - E(x_i - ε)] / (2ε)
        dE/dy_i ≈ [E(y_i + ε) - E(y_i - ε)] / (2ε)

    Args:
        tissue: Tissue containing the global vertex pool.
        energy_params: Energy parameters for energy computation.
        geometry: GeometryCalculator instance.
        epsilon: Finite difference step size (default: 1e-6).

    Returns:
        np.ndarray: Gradient array of shape (N, 2) where N is the number of global vertices.
                   Returns empty array (0, 2) if tissue has no vertices.

    Notes:
        - Modifies tissue.vertices temporarily during computation but restores original values.
        - Calls tissue.reconstruct_cell_vertices() after each perturbation to update cell views.
        - Computes total tissue energy at each perturbed state.
        - Not optimized; performs 4*N tissue energy evaluations where N = number of global vertices.
    """
    if tissue.vertices.shape[0] == 0:
        return np.empty((0, 2), dtype=float)

    grad = np.zeros_like(tissue.vertices)

    # Iterate over each global vertex
    for i in range(tissue.vertices.shape[0]):
        # X gradient
        original_x = tissue.vertices[i, 0]
        tissue.vertices[i, 0] = original_x + epsilon
        tissue.reconstruct_cell_vertices()
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = original_x - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 0] = original_x  # restore
        grad[i, 0] = (e_plus - e_minus) / (2 * epsilon)

        # Y gradient
        original_y = tissue.vertices[i, 1]
        tissue.vertices[i, 1] = original_y + epsilon
        tissue.reconstruct_cell_vertices()
        e_plus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = original_y - epsilon
        tissue.reconstruct_cell_vertices()
        e_minus = tissue_energy(tissue, energy_params, geometry)

        tissue.vertices[i, 1] = original_y  # restore
        grad[i, 1] = (e_plus - e_minus) / (2 * epsilon)

    # Final reconstruction to ensure consistency
    tissue.reconstruct_cell_vertices()

    return grad


class Simulation:
    """Main simulation class for vertex model dynamics.

    Supports two solver types:
        - "gradient_descent": Pure energy descent using finite-difference gradients (default).
        - "overdamped_force_balance": Overdamped Langevin dynamics with optional active forces and noise.

    Attributes:
        tissue: Tissue being simulated.
        dt: Time step.
        time: Current simulation time.
        geometry: Geometry calculator instance.
        energy_params: Parameters used for energy evaluation.
        epsilon: Finite-difference step size for gradient estimation.
        damping: Scalar multiplier applied to gradient descent updates ("learning rate" factor).
        solver_type: Type of solver ("gradient_descent" or "overdamped_force_balance").
        ofb_params: Parameters for overdamped force-balance solver (if applicable).
    """

    def __init__(
        self,
        tissue: Optional[Tissue] = None,
        dt: float = 0.01,
        energy_params: Optional[EnergyParameters] = None,
        validate_each_step: bool = False,
        epsilon: float = 1e-6,
        damping: float = 1.0,
        solver_type: Literal["gradient_descent", "overdamped_force_balance"] = "gradient_descent",
        ofb_params: Optional[OverdampedForceBalanceParams] = None,
        apoptosis_params: Optional[ApoptosisParameters] = None,
        apoptotic_cell_ids: Optional[Sequence[Union[int, str]]] = None,
        use_global_gradient: bool = False,
    ):
        """Initialize a simulation.

        Args:
            tissue: Tissue to simulate (creates empty if None)
            dt: Time step for simulation
            energy_params: Optional EnergyParameters instance (default constructed if None)
            validate_each_step: If True, validate tissue structure after each step
            epsilon: Finite-difference step size used for gradient estimation.
            damping: Multiplier on the gradient when updating vertex positions (for gradient_descent solver).
            solver_type: Type of solver to use. Options:
                - "gradient_descent": Pure energy descent (default). Uses `damping` parameter.
                - "overdamped_force_balance": Overdamped Langevin dynamics. Uses `ofb_params`.
            ofb_params: Parameters for the overdamped force-balance solver. Required if
                solver_type is "overdamped_force_balance". If None and solver_type is
                "overdamped_force_balance", default OverdampedForceBalanceParams will be used.
            apoptosis_params: Optional parameters for apoptosis dynamics.
            apoptotic_cell_ids: Optional sequence of initial apoptotic cell IDs (integers or strings).
            use_global_gradient: If True, use global vertex gradient descent instead of per-cell
                gradients. This is necessary for proper mechanical relaxation in tissues with
                shared vertices (e.g., honeycombs). Only applies when solver_type is "gradient_descent".
        """
        self.tissue = tissue if tissue is not None else Tissue()
        self.dt = dt
        self.time = 0.0
        self.geometry = GeometryCalculator()
        self.energy_params = energy_params if energy_params is not None else EnergyParameters()
        self.validate_each_step = validate_each_step
        self.epsilon = epsilon
        self.damping = damping
        self.solver_type = solver_type
        self.use_global_gradient = use_global_gradient

        # Set up overdamped force-balance parameters
        if solver_type == "overdamped_force_balance":
            self.ofb_params = ofb_params if ofb_params is not None else OverdampedForceBalanceParams()
            # Initialize random generator if seed is provided
            if self.ofb_params.random_seed is not None:
                self._rng = np.random.default_rng(self.ofb_params.random_seed)
            else:
                self._rng = None
        else:
            self.ofb_params = ofb_params  # Store even if not used, for introspection
            self._rng = None

        # Apoptosis configuration/state
        self.apoptosis_params = apoptosis_params
        self.apoptosis_state: Optional[ApoptosisState] = None
        if apoptosis_params is not None and apoptotic_cell_ids is not None:
            self.apoptosis_state = ApoptosisState()
            self.apoptosis_state.register_cells(self.tissue, apoptotic_cell_ids, geometry=self.geometry)
            # Mark cells for convenience
            for cell in self.tissue.cells:
                if cell.id in self.apoptosis_state.apoptotic_cells:
                    cell.is_apoptotic = True
        # Store the selected apoptosis removal strategy for use in step()
        self.apoptosis_removal_strategy = (
            apoptosis_params.removal_strategy if apoptosis_params is not None else 'shrink'
        )

    def step(self):
        """Perform a single simulation step.

        Uses the solver specified by `solver_type`:
            - "gradient_descent": Pure energy descent.
            - "overdamped_force_balance": Overdamped Langevin dynamics.

        Notes:
            - Uses per-cell local vertex arrays (future: migrate to global Tissue.vertices).
            - Does NOT handle topological changes (T1 transitions, division, etc.).
        """
        # Validate before updates if requested to catch pre-existing invalid states
        if self.validate_each_step:
            self.tissue.validate()

        # Update apoptosis target areas before computing forces
        if self.apoptosis_params is not None and self.apoptosis_state is not None:
            update_apoptosis_targets(
                self.tissue,
                self.apoptosis_state,
                self.apoptosis_params,
                step_index=int(self.time / self.dt) if self.dt > 0 else 0,
                dt=self.dt,
                geometry=self.geometry,
            )
            # Build per-cell target area mapping and inject into energy parameters
            mapping = build_apoptosis_target_area_mapping(self.apoptosis_state)
            # Combine with existing target_area if it is a dict
            if isinstance(self.energy_params.target_area, dict):
                combined = dict(self.energy_params.target_area)
                combined.update(mapping)
                self.energy_params.target_area = combined
            else:
                self.energy_params.target_area = mapping

        if self.solver_type == "gradient_descent":
            if self.use_global_gradient:
                self._step_global_gradient_descent()
            else:
                self._step_gradient_descent()
        elif self.solver_type == "overdamped_force_balance":
            self._step_overdamped_force_balance()
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}")

        # Advance time after position updates
        self.time += self.dt

        # After position updates, check for apoptotic cells to remove
        if self.apoptosis_params is not None and self.apoptosis_state is not None:
            # Use the configured removal strategy
            if self.apoptosis_removal_strategy == 'shrink':
                # Placeholder for shrink strategy (to be implemented)
                pass
            elif self.apoptosis_removal_strategy == "merge":
                # Merge-neighbours strategy:
                # Use new topology methods to find shared vertices and move them to centroid,
                # then remove the apoptotic cell.
                to_remove = collect_cells_to_remove(
                    self.tissue,
                    self.apoptosis_state,
                    self.apoptosis_params,
                    geometry=self.geometry,
                )
                if to_remove:
                    tol = 1e-10
                    print(f"[DEBUG merge] Cells to remove: {to_remove}")

                    for cid in to_remove:
                        cell = next((c for c in self.tissue.cells if c.id == cid), None)
                        if cell is None:
                            print(f"[DEBUG merge] Cell {cid} not found in tissue, skipping")
                            continue

                        # Compute centroid of apoptotic cell
                        centroid = np.mean(cell.vertices, axis=0)
                        print(f"[DEBUG merge] Apoptotic cell {cid} vertices:\n{cell.vertices}")
                        print(f"[DEBUG merge] Apoptotic cell {cid} centroid: {centroid}")

                        # Find all neighbors and their shared vertices using new method
                        neighbors_shared_verts = self.tissue.get_cells_sharing_vertices_with(
                            cell_id=cid,
                            tol=tol
                        )
                        print(f"[DEBUG merge] Neighbors and shared vertices: {neighbors_shared_verts}")

                        # Collect all unique shared vertex indices
                        all_shared_vertex_indices = set()
                        for neighbor_id, shared_indices in neighbors_shared_verts.items():
                            all_shared_vertex_indices.update(shared_indices)
                            print(f"[DEBUG merge] Cell {neighbor_id} shares vertices {shared_indices} with cell {cid}")

                        print(f"[DEBUG merge] All shared vertex indices: {sorted(all_shared_vertex_indices)}")

                        # Compute vertex degrees before removal (for debugging)
                        vertex_degrees = self.tissue.compute_vertex_degrees(tol=tol)
                        for idx in sorted(all_shared_vertex_indices):
                            degree = vertex_degrees.get(idx, 0)
                            coord = self.tissue.vertices[idx] if idx < len(self.tissue.vertices) else None
                            print(f"[DEBUG merge] Vertex {idx} at {coord} has degree {degree}")

                        # Move shared vertices toward centroid
                        self.tissue.move_shared_vertices_toward_point(
                            vertex_indices=list(all_shared_vertex_indices),
                            target_point=centroid,
                            move_fraction=1.0  # Move all the way to centroid
                        )
                        print(f"[DEBUG merge] Moved {len(all_shared_vertex_indices)} shared vertices to centroid")

                        # Reconstruct cell-local vertices to reflect global changes
                        self.tissue.reconstruct_cell_vertices()

                        # Remove the apoptotic cell
                        print(f"[DEBUG merge] Removing apoptotic cell {cid}")
                        self.tissue.remove_cells([cid])
                        self.apoptosis_state.completed_cells.add(cid)

                        # After topology change, rebuild global vertices and per-cell vertices
                        print("[DEBUG merge] Rebuilding global vertices after removal")
                        self.tissue.build_global_vertices(tol=1e-10)
                        self.tissue.reconstruct_cell_vertices()

                        print(f"[DEBUG merge] Global vertices after rebuild: {self.tissue.vertices.shape[0]} vertices")
                        for c in self.tissue.cells:
                            print(f"[DEBUG merge] Cell {c.id} has {c.vertices.shape[0]} vertices")

            elif self.apoptosis_removal_strategy == 'purse_string':
                # Placeholder for purse-string contractile ring strategy (to be implemented)
                pass
            elif self.apoptosis_removal_strategy == 't2_collapse':
                # Placeholder for T2-like collapse strategy (to be implemented)
                pass
            elif self.apoptosis_removal_strategy == 'weighted_merge':
                # Placeholder for weighted neighbour assimilation strategy (to be implemented)
                pass
            elif self.apoptosis_removal_strategy == 're_tessellate':
                # Placeholder for local re-tessellation strategy (to be implemented)
                pass
            else:
                raise ValueError(f"Unknown apoptosis_removal_strategy: {self.apoptosis_removal_strategy}")

        # Validate after position updates if requested
        if self.validate_each_step:
            self.tissue.validate()

    def _step_gradient_descent(self):
        """Perform gradient descent update for all cells.

        Implements a naive explicit gradient descent on cell vertex positions.
        For each vertex coordinate (x, y) of every cell, estimates the gradient
        of the total energy via central finite differences.

        Updates vertices by:
            V_new = V_old - dt * damping * grad
        """
        for cell in self.tissue.cells:
            if cell.vertices.shape[0] == 0:
                continue  # nothing to do

            # Compute gradient using finite differences
            grad = finite_difference_cell_gradient(
                cell, self.tissue, self.energy_params, self.geometry, epsilon=self.epsilon
            )

            # Gradient descent update
            cell.vertices = cell.vertices - self.dt * self.damping * grad

    def _step_global_gradient_descent(self):
        """Perform global gradient descent update on the shared vertex pool.

        This method operates on tissue.vertices (the global vertex array) instead of
        per-cell vertices. It is necessary for proper mechanical relaxation in tissues
        with shared vertices (e.g., honeycombs), where per-cell gradients don't properly
        communicate forces between neighbors.

        Updates vertices by:
            V_global_new = V_global_old - dt * damping * grad_global
        """
        if self.tissue.vertices.shape[0] == 0:
            return  # nothing to do

        # Compute global gradient using finite differences
        grad = finite_difference_global_gradient(
            self.tissue, self.energy_params, self.geometry, epsilon=self.epsilon
        )

        # Global gradient descent update
        self.tissue.vertices = self.tissue.vertices - self.dt * self.damping * grad

        # Reconstruct cell vertices from updated global vertices
        self.tissue.reconstruct_cell_vertices()

    def _step_overdamped_force_balance(self):
        """Perform overdamped force-balance update for all cells.

        Implements overdamped Langevin dynamics:
            γ dx/dt = -∇E + F_active + η

        Using Euler-Maruyama integration:
            x_new = x_old + (dt / γ) * F
        """
        for cell in self.tissue.cells:
            if cell.vertices.shape[0] == 0:
                continue  # nothing to do

            # Use the overdamped force-balance step function
            new_vertices = overdamped_force_balance_step(
                cell=cell,
                tissue=self.tissue,
                energy_params=self.energy_params,
                geometry=self.geometry,
                ofb_params=self.ofb_params,
                dt=self.dt,
                epsilon=self.epsilon,
                rng=self._rng,
            )
            cell.vertices = new_vertices

    def run(self, n_steps: int = 100):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            n_steps: Number of simulation steps to run
        """
        for _ in range(n_steps):
            self.step()

    def run_with_logging(self, n_steps: int, log_interval: int = 1) -> list[tuple[float, float]]:
        """Run the simulation while logging energy vs time samples.

        Records (time, total_energy) every ``log_interval`` steps AFTER performing each step.
        Does not record the initial (time=0) state to keep semantics simple and match interval logic.

        Args:
            n_steps: Total number of steps to perform.
            log_interval: Interval (in steps) at which to record samples. Must be >= 1.

        Returns:
            list of (time, energy) tuples sampled after steps where (step_index + 1) % log_interval == 0.
            Length will be n_steps // log_interval if n_steps is divisible; otherwise floor division result.

        Raises:
            ValueError: If log_interval < 1.
        """
        if log_interval < 1:
            raise ValueError("log_interval must be >= 1")
        samples: list[tuple[float, float]] = []
        for step_idx in range(n_steps):
            self.step()
            if (step_idx + 1) % log_interval == 0:
                samples.append((self.time, self.total_energy()))
        return samples

    def total_energy(self) -> float:
        """Compute total tissue energy using current energy parameters.

        Returns:
            float: Total energy value (sum over cells).
        """
        return tissue_energy(self.tissue, self.energy_params, self.geometry)

    def __repr__(self):
        return f"Simulation(time={self.time:.2f}, dt={self.dt})"
