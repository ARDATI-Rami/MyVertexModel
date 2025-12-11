# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025  Rami Ardati
"""
Build a 2D vertex-model representation of the tissue from ACAM adhesions.

Pipeline:
  1. Compute midpoint of each adhesion
  2. Group midpoints by cell pair
  3. Sort midpoints along principal axis (PCA) to form polylines
  4. Cluster polyline endpoints into junction vertices
  5. Build edge list with vertex indices and polyline geometry
  6. Plot the result
"""
import os
import dill
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle
import argparse

from myvertexmodel.core import Tissue, Cell
from myvertexmodel.io import save_tissue

class StubClass:
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        return obj
    def __init__(self, *args, **kwargs):
        pass
    def __setstate__(self, state):
        # Accept any state and store it generically
        self.__dict__.update(state if isinstance(state, dict) else {})

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError, TypeError):
            # Return a generic stub class that can be instantiated with any arguments
            return type(name, (StubClass,), {})

def parse_args():
    parser = argparse.ArgumentParser(description="Build vertex model from ACAM adhesions.")
    parser.add_argument("--acam-file", required=True, help="Path to the ACAM tissue file (relative to project root)")
    parser.add_argument("--output", required=True, help="Output path for the vertex model tissue (.dill file)")
    return parser.parse_args()

# =============================================================================
# Load tissue (using StubUnpickler for ACAM data independence)
# =============================================================================
args = parse_args()

project_path = Path("/home/ardati/PycharmProjects/MyVertexModel")
acam_file_path = project_path / args.acam_file
output_path = project_path / args.output

print(f"Loading tissue from: {acam_file_path}")

with open(acam_file_path, 'rb') as file_in:
    eptm = StubUnpickler(file_in).load()

TISSUE_NAME = acam_file_path.stem  # Extract name from file path

print(f"Tissue '{TISSUE_NAME}' loaded successfully!")
print(f"Number of cells: {len(eptm.cells)}")

# # Update adhesion
# print("\nUpdating adhesion points between all cortices...")
# eptm.update_adhesion_points_between_all_cortices()
print(f"Number of slow adhesions: {len(eptm.slow_adhesions)}")

# =============================================================================
# Step 1 & 2: Compute midpoints and group by cell pair
# =============================================================================
def get_cell_pair_key(cell1_id, cell2_id):
    """Return ordered tuple (min_id, max_id) for consistent grouping."""
    return (min(cell1_id, cell2_id), max(cell1_id, cell2_id))

def extract_adhesion_coords(ad):
    """Extract adhesion coordinates robustly from stub or real objects."""
    # Try x/y or coords
    if hasattr(ad, 'x') and hasattr(ad, 'y'):
        return [[ad.x[0], ad.y[0]], [ad.x[1], ad.y[1]]]
    if hasattr(ad, 'coords'):
        return ad.coords
    # Try cell_1_index/cell_2_index
    if hasattr(ad, 'cell_1') and hasattr(ad, 'cell_2'):
        cell_1, cell_2 = ad.cell_1, ad.cell_2
        idx1 = getattr(ad, 'cell_1_index', None)
        idx2 = getattr(ad, 'cell_2_index', None)
        if idx1 is not None and idx2 is not None:
            if all(hasattr(c, 'x') and hasattr(c, 'y') for c in [cell_1, cell_2]):
                try:
                    return [[cell_1.x[idx1], cell_1.y[idx1]], [cell_2.x[idx2], cell_2.y[idx2]]]
                except Exception:
                    return None
        # Fallback to cell_1_s/cell_2_s
        s1 = getattr(ad, 'cell_1_s', None)
        s2 = getattr(ad, 'cell_2_s', None)
        if isinstance(s1, int) and isinstance(s2, int):
            if all(hasattr(c, 'x') and hasattr(c, 'y') for c in [cell_1, cell_2]):
                try:
                    return [[cell_1.x[s1], cell_1.y[s1]], [cell_2.x[s2], cell_2.y[s2]]]
                except Exception:
                    return None
    return None

def log_warning(msg):
    print(f"Warning: {msg}")

# Dictionary: cell_pair -> list of midpoints
midpoints_by_pair = defaultdict(list)
for ad_idx, ad in enumerate(eptm.slow_adhesions):
    xy = extract_adhesion_coords(ad)
    if xy is None:
        log_warning(f"Adhesion {ad_idx} missing valid coordinates. Skipping.")
        continue
    midpoint = np.mean(xy, axis=0)
    cell1_id = getattr(getattr(ad, 'cell_1', None), 'identifier', None)
    cell2_id = getattr(getattr(ad, 'cell_2', None), 'identifier', None)
    pair_key = get_cell_pair_key(cell1_id, cell2_id)
    midpoints_by_pair[pair_key].append(midpoint)

print(f"\nNumber of cell pairs with adhesions: {len(midpoints_by_pair)}")

# =============================================================================
# Step 3: Sort midpoints along principal axis (PCA) to form polylines
# =============================================================================
def sort_points_by_pca(points):
    """
    Sort points along their principal axis using PCA.
    Returns sorted points as (N, 2) array.
    """
    points = np.array(points)
    if len(points) <= 1:
        return points

    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # Compute covariance matrix and principal direction
    cov = np.cov(centered.T)
    if cov.ndim == 0:  # Single point edge case
        return points

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Principal axis is the eigenvector with largest eigenvalue
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Project points onto principal axis and sort
    projections = centered @ principal_axis
    sorted_indices = np.argsort(projections)

    return points[sorted_indices]

# Dictionary: cell_pair -> sorted polyline (N, 2) array
polylines_by_pair = {}

for pair_key, midpoints in midpoints_by_pair.items():
    sorted_polyline = sort_points_by_pca(midpoints)
    polylines_by_pair[pair_key] = sorted_polyline

print(f"Created {len(polylines_by_pair)} polylines")

# =============================================================================
# Step 4: Collect start and end points as candidate junction vertices
# =============================================================================
# Each endpoint: (x, y, pair_key, is_start)
endpoints = []

for pair_key, polyline in polylines_by_pair.items():
    if len(polyline) >= 1:
        # Start point
        endpoints.append({
            'xy': polyline[0],
            'pair_key': pair_key,
            'is_start': True
        })
        # End point
        endpoints.append({
            'xy': polyline[-1],
            'pair_key': pair_key,
            'is_start': False
        })

print(f"Total candidate endpoints: {len(endpoints)}")

# =============================================================================
# Step 5: Cluster nearby endpoints into junction vertices
# =============================================================================
def cluster_endpoints(endpoints, eps=0.05):
    """
    Cluster endpoints within distance eps into single vertices using
    connected-components (transitive closure) approach.

    Returns:
        vertices: list of (x, y) vertex positions
        endpoint_to_vertex: dict mapping (pair_key, is_start) -> vertex_index
    """
    n = len(endpoints)
    if n == 0:
        return [], {}

    # Extract coordinates
    coords = np.array([ep['xy'] for ep in endpoints])

    # Build adjacency: find all pairs within eps using distance matrix
    # For efficiency with larger datasets, we compute pairwise distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Union-Find data structure for connected components
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Union all pairs within eps
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] < eps:
                union(i, j)

    # Group by connected component
    component_to_indices = defaultdict(list)
    for i in range(n):
        component_to_indices[find(i)].append(i)

    # Compute vertex positions (centroid of each cluster)
    vertices = []
    endpoint_to_vertex = {}

    for cluster_idx, indices in enumerate(component_to_indices.values()):
        cluster_coords = coords[indices]
        centroid = np.mean(cluster_coords, axis=0)
        vertices.append(centroid)

        # Map each endpoint in cluster to this vertex
        for ep_idx in indices:
            ep = endpoints[ep_idx]
            key = (ep['pair_key'], ep['is_start'])
            endpoint_to_vertex[key] = cluster_idx

    return vertices, endpoint_to_vertex

# Estimate eps based on distances between nearby endpoints
# First compute all pairwise distances between endpoints to understand the distribution
endpoint_coords = np.array([ep['xy'] for ep in endpoints])
n_endpoints = len(endpoint_coords)

# Compute pairwise distances
pairwise_dists = []
for i in range(n_endpoints):
    for j in range(i + 1, n_endpoints):
        d = np.linalg.norm(endpoint_coords[i] - endpoint_coords[j])
        pairwise_dists.append(d)

pairwise_dists = np.array(pairwise_dists)
print(f"\nEndpoint distance statistics:")
print(f"  Min: {pairwise_dists.min():.4f}")
print(f"  25th percentile: {np.percentile(pairwise_dists, 25):.4f}")
print(f"  Median: {np.median(pairwise_dists):.4f}")
print(f"  5th percentile: {np.percentile(pairwise_dists, 5):.4f}")
print(f"  1st percentile: {np.percentile(pairwise_dists, 1):.4f}")

# Use a small percentile of pairwise distances for eps
# Tricellular junctions should have 3 endpoints very close together
# Look at the smallest distances to find the natural clustering scale
sorted_dists = np.sort(pairwise_dists)
# Find the gap in the distribution - look at the first 5% of distances
n_small = max(10, int(0.05 * len(sorted_dists)))
small_dists = sorted_dists[:n_small]

# Use a threshold that captures the tightly clustered endpoints
# The eps should be slightly larger than typical junction clustering distance
eps = np.percentile(pairwise_dists, 0.25) * 1.5  # Use 0.25th percentile

# Ensure eps is reasonable - not too small, not too large
min_eps = pairwise_dists.min() * 2
max_eps = np.percentile(pairwise_dists, 5)
eps = np.clip(eps, min_eps, max_eps)

print(f"Clustering with eps = {eps:.4f}")

vertices, endpoint_to_vertex = cluster_endpoints(endpoints, eps=eps)
print(f"Number of junction vertices: {len(vertices)}")

# =============================================================================
# Step 6: Build edge list
# =============================================================================
# Each edge: {
#   'v_start': vertex index,
#   'v_end': vertex index,
#   'cells': (cell1_id, cell2_id),
#   'polyline': (N, 2) array of ordered points
# }

edges = []

for pair_key, polyline in polylines_by_pair.items():
    v_start = endpoint_to_vertex.get((pair_key, True))
    v_end = endpoint_to_vertex.get((pair_key, False))

    if v_start is not None and v_end is not None:
        edge = {
            'v_start': v_start,
            'v_end': v_end,
            'cells': pair_key,
            'polyline': polyline
        }
        edges.append(edge)

print(f"Number of edges: {len(edges)}")

# =============================================================================
# Step 6b: Refine edges by inserting vertices at regular intervals
# =============================================================================
def compute_arc_length(polyline):
    """Compute cumulative arc length along polyline."""
    diffs = np.diff(polyline, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.zeros(len(polyline))
    cumulative[1:] = np.cumsum(segment_lengths)
    return cumulative


def find_regular_split_indices(polyline, max_segment_length=None, num_segments=None,
                                min_points_from_endpoints=5):
    """
    Find indices to split polyline at regular arc-length intervals.

    Parameters:
        polyline: (N, 2) array of points
        max_segment_length: maximum length between vertices (if provided)
        num_segments: number of segments to divide into (if provided)
        min_points_from_endpoints: minimum index distance from start/end

    Returns:
        List of indices where splits should occur
    """
    N = len(polyline)
    if N < 3:
        return []

    # Compute arc length
    arc_length = compute_arc_length(polyline)
    total_length = arc_length[-1]

    if total_length < 1e-10:
        return []

    # Determine number of segments
    if num_segments is not None:
        n_segs = max(1, num_segments)
    elif max_segment_length is not None:
        n_segs = max(1, int(np.ceil(total_length / max_segment_length)))
    else:
        # Default: aim for segments of reasonable length
        n_segs = max(1, int(np.ceil(total_length / 20.0)))

    if n_segs <= 1:
        return []

    # Target arc lengths for split points
    target_lengths = np.linspace(0, total_length, n_segs + 1)[1:-1]  # exclude endpoints

    # Find indices closest to target arc lengths
    split_indices = []
    for target in target_lengths:
        # Find index where arc_length is closest to target
        idx = np.argmin(np.abs(arc_length - target))

        # Ensure we're not too close to endpoints
        if idx >= min_points_from_endpoints and idx <= N - 1 - min_points_from_endpoints:
            # Avoid duplicates
            if idx not in split_indices:
                split_indices.append(idx)

    split_indices.sort()
    return split_indices


def refine_edges_by_arc_length(vertices, edges, max_segment_length=None,
                                 num_segments_per_edge=3,
                                 min_points_from_endpoints=5):
    """
    Refine edges by splitting at regular arc-length intervals.

    Parameters:
        vertices: list of (x, y) vertex positions (will be extended in place)
        edges: list of edge dicts
        max_segment_length: maximum length between vertices
        num_segments_per_edge: target number of segments per edge
        min_points_from_endpoints: minimum distance from junctions

    Returns:
        refined_edges: new list of edge dicts
    """
    refined_edges = []

    for edge in edges:
        polyline = edge['polyline']
        cells = edge['cells']
        v_start = edge['v_start']
        v_end = edge['v_end']

        # Find split indices
        split_indices = find_regular_split_indices(
            polyline,
            max_segment_length=max_segment_length,
            num_segments=num_segments_per_edge,
            min_points_from_endpoints=min_points_from_endpoints
        )

        if not split_indices:
            # No splits, keep original edge
            refined_edges.append(edge)
            continue

        # Create new vertices at split points
        new_vertex_indices = []
        for split_idx in split_indices:
            new_vertex_pos = polyline[split_idx]
            new_vertex_idx = len(vertices)
            vertices.append(new_vertex_pos)
            new_vertex_indices.append(new_vertex_idx)

        # Build sequence of vertex indices along the polyline
        vertex_sequence = [v_start] + new_vertex_indices + [v_end]
        index_sequence = [0] + split_indices + [len(polyline) - 1]

        # Create sub-edges for each consecutive pair
        for i in range(len(vertex_sequence) - 1):
            va = vertex_sequence[i]
            vb = vertex_sequence[i + 1]
            idx_a = index_sequence[i]
            idx_b = index_sequence[i + 1]

            # Extract segment of polyline (inclusive on both ends)
            segment = polyline[idx_a:idx_b + 1]

            sub_edge = {
                'v_start': va,
                'v_end': vb,
                'cells': cells,
                'polyline': segment
            }
            refined_edges.append(sub_edge)

    return refined_edges


print("\nRefining edges by inserting vertices at regular intervals...")

# Store original counts for comparison
n_vertices_before = len(vertices)
n_edges_before = len(edges)

# Refine edges: split each edge into ~3 segments with vertices in the middle
edges = refine_edges_by_arc_length(
    vertices, edges,
    num_segments_per_edge=3,  # Each edge becomes 3 sub-edges
    min_points_from_endpoints=10  # Keep vertices away from junctions
)

n_vertices_after = len(vertices)
n_edges_after = len(edges)

print(f"  Vertices: {n_vertices_before} -> {n_vertices_after} (+{n_vertices_after - n_vertices_before} interior vertices)")
print(f"  Edges: {n_edges_before} -> {n_edges_after} (+{n_edges_after - n_edges_before} sub-edges)")

# Print summary
print("\n" + "="*60)
print("VERTEX MODEL SUMMARY")
print("="*60)
print(f"Vertices (junctions): {len(vertices)}")
print(f"Edges (cell-cell interfaces): {len(edges)}")
print(f"Cells: {len(eptm.cells)}")

# Verify: count vertex valencies
valency = defaultdict(int)
for edge in edges:
    valency[edge['v_start']] += 1
    valency[edge['v_end']] += 1

valency_counts = defaultdict(int)
for v, val in valency.items():
    valency_counts[val] += 1

print("\nVertex valencies:")
for val, count in sorted(valency_counts.items()):
    print(f"  Valency {val}: {count} vertices")

# =============================================================================
# Step 7: Plot the vertex graph on top of original cortices
# =============================================================================
def plot_vertex_model(vertices, edges, cells, filename, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 10))
    for cell in cells:
        x_closed = np.append(cell.x, cell.x[0])
        y_closed = np.append(cell.y, cell.y[0])
        ax.plot(x_closed, y_closed, 'b-', alpha=0.3, linewidth=0.5)
    for edge in edges:
        polyline = edge['polyline']
        ax.plot(polyline[:, 0], polyline[:, 1], 'g-', linewidth=1.5, alpha=0.7)
    valency = defaultdict(int)
    for edge in edges:
        valency[edge['v_start']] += 1
        valency[edge['v_end']] += 1
    junction_vertices = [vertices[i] for i, v in valency.items() if v >= 3]
    interior_vertices = [vertices[i] for i, v in valency.items() if v < 3]
    if interior_vertices:
        int_arr = np.array(interior_vertices)
        ax.scatter(int_arr[:, 0], int_arr[:, 1], c='orange', s=40, zorder=4, edgecolors='black', linewidth=0.5)
    if junction_vertices:
        junc_arr = np.array(junction_vertices)
        ax.scatter(junc_arr[:, 0], junc_arr[:, 1], c='red', s=100, zorder=5, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nPlot saved to: {filename}")
    plt.show()

output_filename = f'vertex_model_from_adhesions_{TISSUE_NAME}.png'
plot_vertex_model(vertices, edges, eptm.cells, project_path / output_filename,
                 f'Vertex Model from ACAM Adhesions (Refined)\n({len(vertices)} vertices, {len(edges)} edges, {len(eptm.cells)} cells)')

print("polyline by pair keys:")
for key in polylines_by_pair.keys():
    print(f"  {key}")
# =============================================================================
# Create true vertex model tissue
# =============================================================================
def get_cell_vertices(edges_for_cell):
    """Extract ordered vertex indices for a cell's boundary from its edges."""
    if not edges_for_cell:
        return []
    adj = defaultdict(list)
    for e in edges_for_cell:
        a, b = e['v_start'], e['v_end']
        adj[a].append(b)
        adj[b].append(a)
    if not adj:
        return []
    start = min(adj.keys())
    path = []
    visited_edges = set()
    current = start
    prev = -1
    while len(path) < len(adj) + 1:
        path.append(current)
        if prev != -1:
            edge = tuple(sorted([prev, current]))
            visited_edges.add(edge)
        neighbors = [n for n in adj[current] if n != prev]
        if not neighbors:
            break
        next_v = None
        for n in neighbors:
            edge = tuple(sorted([current, n]))
            if edge not in visited_edges:
                next_v = n
                break
        if next_v is None:
            next_v = neighbors[0]
        prev = current
        current = next_v
        if current == start and len(path) > 2:
            break
    if path and path[0] == path[-1]:
        path.pop()
    return path

# Create the tissue
vertices_array = np.array(vertices)
tissue = Tissue()
tissue.vertices = vertices_array
cell_ids = [getattr(c, 'identifier', i) for i, c in enumerate(eptm.cells)]
for cell_id in cell_ids:
    if cell_id == 'bnd':
        continue
    edges_for_cell = [e for e in edges if cell_id in e['cells']]
    vertex_indices = get_cell_vertices(edges_for_cell)
    if vertex_indices:
        cell = Cell(cell_id, vertex_indices=np.array(vertex_indices))
        tissue.add_cell(cell)

# Save the tissue
save_tissue(tissue, str(output_path))
print(f"Tissue saved to {output_path}.dill")

# =============================================================================
# Export data structures for further use
# =============================================================================
vertex_model = {
    'vertices': vertices,  # List of (x, y) positions
    'edges': edges,        # List of edge dicts
    'polylines_by_pair': polylines_by_pair,  # Dict of cell_pair -> polyline
    'valency': dict(valency),  # Dict of vertex_index -> valency
}

print("\nVertex model data available in 'vertex_model' dictionary")
