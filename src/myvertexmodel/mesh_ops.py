"""
Mesh operations for vertex model.

Provides functions for vertex merging and edge meshing/subdivision.
"""

import numpy as np
from typing import Optional, Dict, List

from .core import Tissue
from .energy import EnergyParameters, tissue_energy
from .geometry import GeometryCalculator


def _remove_consecutive_duplicates(indices: np.ndarray) -> np.ndarray:
    """Remove consecutive duplicate indices, including wrap-around.

    Args:
        indices: 1D array of integer indices.

    Returns:
        1D array with consecutive duplicates removed.
    """
    if len(indices) <= 1:
        return indices.copy()

    # Remove interior consecutive duplicates
    mask = np.ones(len(indices), dtype=bool)
    mask[1:] = indices[1:] != indices[:-1]
    result = indices[mask]

    # Remove wrap-around duplicate (first == last)
    if len(result) > 1 and result[0] == result[-1]:
        result = result[:-1]

    return result


def merge_nearby_vertices(
    tissue: Tissue,
    distance_tol: float = 1e-8,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10,
    energy_params: Optional[EnergyParameters] = None,
) -> Dict:
    """Merge vertices that are within distance_tol of each other.

    This operation detects clusters of near-coincident vertices in the global
    vertex pool and replaces them with a single representative vertex (the
    centroid of the cluster), updating all cell references accordingly.

    Args:
        tissue: Tissue with populated global vertex pool (tissue.vertices)
                and per-cell vertex_indices.
        distance_tol: Maximum distance between vertices to consider for merging.
        energy_tol: Maximum allowed change in total tissue energy.
        geometry_tol: Maximum allowed change in per-cell area and perimeter.
        energy_params: Optional EnergyParameters for energy validation. If None,
                      uses default parameters.

    Returns:
        dict: Diagnostics including:
            - 'vertices_before': Number of vertices before merging
            - 'vertices_after': Number of vertices after merging
            - 'clusters_merged': Number of clusters that were merged
            - 'energy_change': Absolute change in energy
            - 'max_area_change': Maximum change in any cell's area
            - 'max_perimeter_change': Maximum change in any cell's perimeter

    Raises:
        ValueError: If energy or geometry change exceeds tolerance, or if
                   merging would create a degenerate cell (< 3 vertices).

    Notes:
        - Requires tissue.vertices to be populated (call build_global_vertices first).
        - Modifies tissue in-place: updates tissue.vertices, cell.vertex_indices,
          and cell.vertices.
    """
    if tissue.vertices.shape[0] == 0:
        # Nothing to merge
        return {
            "vertices_before": 0,
            "vertices_after": 0,
            "clusters_merged": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    geom = GeometryCalculator()
    params = energy_params if energy_params is not None else EnergyParameters()

    # Step 1: Pre-compute metrics
    energy_before = tissue_energy(tissue, params, geom)
    cell_metrics_before = {}
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] >= 3:
            verts = tissue.vertices[cell.vertex_indices]
            cell_metrics_before[cell.id] = (
                geom.calculate_area(verts),
                geom.calculate_perimeter(verts),
            )

    vertices_before = tissue.vertices.shape[0]

    # Step 2: Find nearby vertex pairs and build clusters using union-find
    n_verts = len(tissue.vertices)
    parent = list(range(n_verts))

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Pairwise distance check (O(n^2) - could be optimized with KD-tree for large n)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            dist = np.linalg.norm(tissue.vertices[i] - tissue.vertices[j])
            if dist <= distance_tol:
                union(i, j)

    # Build clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(n_verts):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)

    # Step 3: Build new vertex pool and index mapping
    new_vertices = []
    old_to_new: Dict[int, int] = {}
    clusters_merged = 0

    for root, members in clusters.items():
        new_idx = len(new_vertices)
        if len(members) == 1:
            # Single vertex, keep as-is
            new_vertices.append(tissue.vertices[members[0]].copy())
        else:
            # Compute centroid as representative
            cluster_verts = tissue.vertices[members]
            centroid = np.mean(cluster_verts, axis=0)
            new_vertices.append(centroid)
            clusters_merged += 1

        for old_idx in members:
            old_to_new[old_idx] = new_idx

    new_vertices_array = np.array(new_vertices, dtype=float)

    # Step 4: Update cell vertex indices
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] == 0:
            continue

        # Map old indices to new indices
        new_indices = np.array([old_to_new[idx] for idx in cell.vertex_indices], dtype=int)

        # Remove consecutive duplicates
        new_indices = _remove_consecutive_duplicates(new_indices)

        # Check for degenerate cell
        if len(new_indices) < 3:
            raise ValueError(
                f"Merging would reduce cell {cell.id} to {len(new_indices)} vertices "
                f"(minimum 3 required). Consider using a smaller distance_tol."
            )

        # Ensure CCW orientation
        temp_verts = new_vertices_array[new_indices]
        signed_area = geom.signed_area(temp_verts)
        if signed_area < 0:
            new_indices = new_indices[::-1]

        cell.vertex_indices = new_indices

    # Step 5: Update tissue and validate
    tissue.vertices = new_vertices_array
    tissue.reconstruct_cell_vertices()

    # Validate energy invariance
    energy_after = tissue_energy(tissue, params, geom)
    energy_change = abs(energy_after - energy_before)
    if energy_change > energy_tol:
        raise ValueError(
            f"Energy change {energy_change:.2e} exceeds tolerance {energy_tol:.2e}. "
            f"Merging may have altered geometry too much."
        )

    # Validate per-cell geometry invariance
    max_area_change = 0.0
    max_perimeter_change = 0.0
    for cell in tissue.cells:
        if cell.id in cell_metrics_before:
            area_before, perim_before = cell_metrics_before[cell.id]
            verts = tissue.vertices[cell.vertex_indices]
            area_after = geom.calculate_area(verts)
            perim_after = geom.calculate_perimeter(verts)
            area_change = abs(area_after - area_before)
            perim_change = abs(perim_after - perim_before)
            max_area_change = max(max_area_change, area_change)
            max_perimeter_change = max(max_perimeter_change, perim_change)
            if area_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} area change {area_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )
            if perim_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} perimeter change {perim_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )

    return {
        "vertices_before": vertices_before,
        "vertices_after": len(new_vertices_array),
        "clusters_merged": clusters_merged,
        "energy_change": energy_change,
        "max_area_change": max_area_change,
        "max_perimeter_change": max_perimeter_change,
    }


def mesh_edges(
    tissue: Tissue,
    mode: str = "none",
    length_scale: float = 1.0,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10,
    energy_params: Optional[EnergyParameters] = None,
) -> Dict:
    """Subdivide edges according to specified meshing mode.

    This operation refines the mesh by inserting intermediate vertices along
    edges. The subdivision density is controlled by the mode parameter.

    Args:
        tissue: Tissue with populated global vertex pool.
        mode: Meshing mode, one of:
            - "none": No subdivision (identity operation)
            - "low": Add single midpoint to each edge
            - "medium": Target ~1 vertex per 1.0 * length_scale units
            - "high": Target ~1 vertex per 0.5 * length_scale units
        length_scale: Base unit for edge subdivision (default 1.0).
        energy_tol: Maximum allowed change in total tissue energy.
        geometry_tol: Maximum allowed change in per-cell area and perimeter.
        energy_params: Optional EnergyParameters for energy validation.

    Returns:
        dict: Diagnostics including:
            - 'mode': The meshing mode used
            - 'vertices_before': Number of vertices before meshing
            - 'vertices_after': Number of vertices after meshing
            - 'edges_subdivided': Number of edges that were subdivided
            - 'energy_change': Absolute change in energy
            - 'max_area_change': Maximum change in any cell's area
            - 'max_perimeter_change': Maximum change in any cell's perimeter

    Raises:
        ValueError: If mode is invalid or if energy/geometry tolerance exceeded.

    Notes:
        - Requires tissue.vertices to be populated (call build_global_vertices first).
        - Shared edges between cells receive identical intermediate vertices.
        - Very short edges (< 0.1 * length_scale) are not subdivided in medium/high modes.
    """
    valid_modes = {"none", "low", "medium", "high"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

    if tissue.vertices.shape[0] == 0:
        return {
            "mode": mode,
            "vertices_before": 0,
            "vertices_after": 0,
            "edges_subdivided": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    geom = GeometryCalculator()
    params = energy_params if energy_params is not None else EnergyParameters()

    # Step 1: Pre-compute metrics
    energy_before = tissue_energy(tissue, params, geom)
    cell_metrics_before = {}
    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] >= 3:
            verts = tissue.vertices[cell.vertex_indices]
            cell_metrics_before[cell.id] = (
                geom.calculate_area(verts),
                geom.calculate_perimeter(verts),
            )

    vertices_before = tissue.vertices.shape[0]

    if mode == "none":
        # No-op
        return {
            "mode": mode,
            "vertices_before": vertices_before,
            "vertices_after": vertices_before,
            "edges_subdivided": 0,
            "energy_change": 0.0,
            "max_area_change": 0.0,
            "max_perimeter_change": 0.0,
        }

    # Step 2: Collect all unique edges
    all_edges: Dict[tuple, None] = {}  # Use dict for ordered unique set
    for cell in tissue.cells:
        indices = cell.vertex_indices
        n = len(indices)
        for i in range(n):
            v_start = indices[i]
            v_end = indices[(i + 1) % n]
            edge_key = (min(v_start, v_end), max(v_start, v_end))
            all_edges[edge_key] = None

    # Step 3: Build edge subdivision map
    edge_subdivision: Dict[tuple, List[int]] = {}
    new_vertices_list = list(tissue.vertices)
    next_vertex_index = len(tissue.vertices)
    edges_subdivided = 0
    min_edge_for_subdivision = 0.1 * length_scale

    for edge_key in all_edges:
        min_idx, max_idx = edge_key
        v_start = tissue.vertices[min_idx]
        v_end = tissue.vertices[max_idx]
        edge_length = np.linalg.norm(v_end - v_start)

        # Determine number of segments
        if mode == "low":
            n_segments = 2
        elif mode == "medium":
            if edge_length < min_edge_for_subdivision:
                n_segments = 1
            else:
                n_segments = max(1, round(edge_length / length_scale))
        elif mode == "high":
            if edge_length < min_edge_for_subdivision:
                n_segments = 1
            else:
                n_segments = max(1, round(edge_length / (0.5 * length_scale)))
        else:  # mode == "none" already handled above
            n_segments = 1

        if n_segments <= 1:
            edge_subdivision[edge_key] = [min_idx, max_idx]
        else:
            # Create intermediate vertices
            intermediate_indices = []
            for k in range(1, n_segments):
                t = k / n_segments
                v_new = (1 - t) * v_start + t * v_end
                new_vertices_list.append(v_new)
                intermediate_indices.append(next_vertex_index)
                next_vertex_index += 1
            edge_subdivision[edge_key] = [min_idx] + intermediate_indices + [max_idx]
            edges_subdivided += 1

    # Step 4: Reconstruct cell polygons
    new_vertices_array = np.array(new_vertices_list, dtype=float)

    for cell in tissue.cells:
        if cell.vertex_indices.shape[0] == 0:
            continue

        old_indices = cell.vertex_indices
        n = len(old_indices)
        new_poly: List[int] = []

        for i in range(n):
            v_start = old_indices[i]
            v_end = old_indices[(i + 1) % n]
            edge_key = (min(v_start, v_end), max(v_start, v_end))
            edge_seq = edge_subdivision[edge_key]

            # Determine orientation: are we going min->max or max->min?
            if v_start == edge_key[0]:
                # Forward: min -> max, exclude last to avoid duplication
                segment = edge_seq[:-1]
            else:
                # Reverse: max -> min, exclude last after reversing
                segment = edge_seq[::-1][:-1]

            new_poly.extend(segment)

        # Remove consecutive duplicates
        new_indices = _remove_consecutive_duplicates(np.array(new_poly, dtype=int))

        # Ensure CCW orientation
        temp_verts = new_vertices_array[new_indices]
        signed_area = geom.signed_area(temp_verts)
        if signed_area < 0:
            new_indices = new_indices[::-1]

        cell.vertex_indices = new_indices

    # Step 5: Update tissue
    tissue.vertices = new_vertices_array
    tissue.reconstruct_cell_vertices()

    # Validate energy invariance
    energy_after = tissue_energy(tissue, params, geom)
    energy_change = abs(energy_after - energy_before)
    if energy_change > energy_tol:
        raise ValueError(
            f"Energy change {energy_change:.2e} exceeds tolerance {energy_tol:.2e}."
        )

    # Validate per-cell geometry invariance
    max_area_change = 0.0
    max_perimeter_change = 0.0
    for cell in tissue.cells:
        if cell.id in cell_metrics_before:
            area_before, perim_before = cell_metrics_before[cell.id]
            verts = tissue.vertices[cell.vertex_indices]
            area_after = geom.calculate_area(verts)
            perim_after = geom.calculate_perimeter(verts)
            area_change = abs(area_after - area_before)
            perim_change = abs(perim_after - perim_before)
            max_area_change = max(max_area_change, area_change)
            max_perimeter_change = max(max_perimeter_change, perim_change)
            if area_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} area change {area_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )
            if perim_change > geometry_tol:
                raise ValueError(
                    f"Cell {cell.id} perimeter change {perim_change:.2e} exceeds geometry tolerance {geometry_tol:.2e}."
                )

    return {
        "mode": mode,
        "vertices_before": vertices_before,
        "vertices_after": len(new_vertices_array),
        "edges_subdivided": edges_subdivided,
        "energy_change": energy_change,
        "max_area_change": max_area_change,
        "max_perimeter_change": max_perimeter_change,
    }

