"""Apoptosis support for MyVertexModel.

Provides configuration and per-cell state for apoptosis, plus helper functions
for updating target areas and deciding when apoptotic cells should be removed.

Initial version: simple time-based shrink of target area, relying on existing
vertex merging and validation to collapse and remove cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import numpy as np

from .core import Tissue, Cell
from .geometry import GeometryCalculator


CellId = Union[int, str]


@dataclass
class ApoptosisParameters:
    """Configuration for apoptosis dynamics.

    Attributes:
        shrink_rate: Fraction of initial area lost per unit time (per step * dt).
        min_area_fraction: Minimum fraction of initial area used as a *floor* for
            the apoptotic target area (i.e., the target will not shrink below
            min_area_fraction * A0).
        removal_area_fraction: Fraction of initial area below which an apoptotic
            cell is removed. Set to 0.0 to disable.
        removal_area_absolute: Absolute area threshold below which an apoptotic
            cell is removed. Set to 0.0 to disable.
        min_vertices: Minimum number of vertices; if a cell falls below this,
            it is considered for removal.
        start_step: Optional global step index at which apoptosis starts.
    """

    shrink_rate: float = 0.5
    min_area_fraction: float = 0.05
    removal_area_fraction: float = 0.1
    removal_area_absolute: float = 0.0
    min_vertices: int = 3
    start_step: int = 0


@dataclass
class ApoptosisState:
    """Tracks per-cell apoptosis state.

    Attributes:
        apoptotic_cells: Set of cell IDs undergoing apoptosis.
        initial_areas: Mapping from cell ID to initial area at apoptosis start.
        current_target_areas: Mapping from cell ID to current target area.
        completed_cells: Set of cell IDs that have finished apoptosis and
            should be (or have been) removed.
    """

    apoptotic_cells: Set[CellId] = field(default_factory=set)
    initial_areas: Dict[CellId, float] = field(default_factory=dict)
    current_target_areas: Dict[CellId, float] = field(default_factory=dict)
    completed_cells: Set[CellId] = field(default_factory=set)

    def register_cells(self, tissue: Tissue, cell_ids: Iterable[CellId], geometry: Optional[GeometryCalculator] = None) -> None:
        """Register cells for apoptosis and record their initial areas.

        Accepts IDs as either the exact type used in cell.id (e.g. int/str)
        or their string representation (e.g. '4' for cell.id == 4).
        """
        if geometry is None:
            geometry = GeometryCalculator()

        # Primary lookup by exact ID object
        id_to_cell: Dict[CellId, Cell] = {cell.id: cell for cell in tissue.cells}
        # Secondary lookup by string form of ID to handle CLI inputs
        str_to_cell: Dict[str, Cell] = {str(cell.id): cell for cell in tissue.cells}

        for raw_cid in cell_ids:
            # Try exact match first
            cell = id_to_cell.get(raw_cid)
            if cell is None:
                # Fallback: match by string representation
                cell = str_to_cell.get(str(raw_cid))
            if cell is None:
                continue

            cid: CellId = cell.id  # normalize to actual ID object
            self.apoptotic_cells.add(cid)
            area = geometry.calculate_area(cell.vertices)
            self.initial_areas[cid] = area
            self.current_target_areas[cid] = area

    def is_apoptotic(self, cell_id: CellId) -> bool:
        return cell_id in self.apoptotic_cells


def update_apoptosis_targets(
    tissue: Tissue,
    state: ApoptosisState,
    params: ApoptosisParameters,
    step_index: int,
    dt: float,
    geometry: Optional[GeometryCalculator] = None,
) -> None:
    """Update target areas for apoptotic cells.

    Simple exponential decay model:
        A_target(t) = max(min_area_fraction * A0, A0 * exp(-k * t))
    where k is derived from shrink_rate.

    Args:
        tissue: Tissue being simulated.
        state: ApoptosisState tracking apoptotic cells.
        params: ApoptosisParameters controlling shrink dynamics.
        step_index: Current simulation step index (0-based).
        dt: Time step size.
        geometry: Optional GeometryCalculator (unused for now but kept for symmetry).
    """
    if step_index < params.start_step:
        return

    # Convert shrink_rate (fraction per unit time) to exponential rate k
    # such that after time T, area ~ A0 * exp(-shrink_rate * T)
    k = params.shrink_rate
    t = (step_index - params.start_step) * dt

    for cid in list(state.apoptotic_cells):
        A0 = state.initial_areas.get(cid)
        if A0 is None:
            continue
        min_A = params.min_area_fraction * A0
        new_target = max(min_A, A0 * np.exp(-k * t))
        state.current_target_areas[cid] = new_target


def collect_cells_to_remove(
    tissue: Tissue,
    state: ApoptosisState,
    params: ApoptosisParameters,
    geometry: Optional[GeometryCalculator] = None,
) -> List[CellId]:
    """Determine which apoptotic cells should be removed.

    Criteria (any triggers removal):
        - Geometric area < removal_area_fraction * initial area (if enabled), OR
        - Geometric area < removal_area_absolute (if enabled), OR
        - Number of vertices <= min_vertices.

    Notes:
        The target-area floor (min_area_fraction) is intentionally *not* used as
        a removal threshold. This decouples mechanical shrink targets from the
        logical deletion condition.

    Args:
        tissue: Tissue containing cells.
        state: ApoptosisState.
        params: ApoptosisParameters.
        geometry: Optional GeometryCalculator; if None, a new one is created.

    Returns:
        List of cell IDs that should be removed.
    """
    if geometry is None:
        geometry = GeometryCalculator()

    id_to_cell: Dict[CellId, Cell] = {cell.id: cell for cell in tissue.cells}
    to_remove: List[CellId] = []

    for cid in state.apoptotic_cells:
        if cid in state.completed_cells:
            continue
        cell = id_to_cell.get(cid)
        if cell is None:
            # Already removed
            state.completed_cells.add(cid)
            continue

        A0 = state.initial_areas.get(cid)
        if A0 is None or A0 <= 0:
            continue

        area = geometry.calculate_area(cell.vertices)
        n_vertices = cell.vertices.shape[0]

        remove_by_fraction = (
            params.removal_area_fraction > 0.0
            and area < params.removal_area_fraction * A0
        )
        remove_by_absolute = (
            params.removal_area_absolute > 0.0
            and area < params.removal_area_absolute
        )
        remove_by_vertices = n_vertices <= params.min_vertices

        if remove_by_fraction or remove_by_absolute or remove_by_vertices:
            to_remove.append(cid)

    return to_remove


def build_apoptosis_target_area_mapping(state: ApoptosisState) -> Dict[CellId, float]:
    """Build a mapping from cell ID to target area for apoptotic cells.

    Non-apoptotic cells are not included and should use the global target area.
    """
    return dict(state.current_target_areas)
