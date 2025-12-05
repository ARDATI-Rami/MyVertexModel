# Plan: Safe Vertex Merging and Edge Meshing Modes

## Problem Statement

The MyVertexModel project requires two related capabilities:

1. **Safe Vertex Merging**: The ability to detect and merge vertices that are extremely close to each other in the global vertex pool, without altering the mechanical energy of the tissue beyond numerical tolerance.

2. **Edge Meshing with Discrete Modes**: The ability to refine ("mesh") edges by inserting intermediate vertices at controlled densities, enabling higher-resolution simulations while preserving geometry and energy.

Both features must maintain the structural validity of the tissue (no degenerate cells, no cracks, consistent polygon orientation) and preserve per-cell geometry (area, perimeter) and global energy within tight numerical tolerances.

---

## Energy Functional

The vertex model uses the following energy functional:

```
E = Σ_cells [0.5 * k_area * (A - A0)^2 + 0.5 * k_perimeter * P^2 + gamma * P]
```

Where:
- `A` = cell area (polygon area)
- `A0` = target (preferred) area
- `P` = cell perimeter (polygon perimeter)
- `k_area`, `k_perimeter`, `gamma` = energy parameters

---

## Feature 1: Safe Vertex Merging

### Contract

```python
def merge_nearby_vertices(
    tissue: Tissue,
    distance_tol: float = 1e-8,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10
) -> dict:
    """
    Merge vertices that are within distance_tol of each other.
    
    Args:
        tissue: Tissue with populated global vertex pool (tissue.vertices)
                and per-cell vertex_indices.
        distance_tol: Maximum distance between vertices to consider for merging.
        energy_tol: Maximum allowed change in total tissue energy.
        geometry_tol: Maximum allowed change in per-cell area and perimeter.
    
    Returns:
        dict: Diagnostics including:
            - 'vertices_before': Number of vertices before merging
            - 'vertices_after': Number of vertices after merging
            - 'clusters_merged': Number of clusters merged
            - 'energy_change': Absolute change in energy
            - 'max_area_change': Maximum change in any cell's area
            - 'max_perimeter_change': Maximum change in any cell's perimeter
    
    Raises:
        ValueError: If energy or geometry change exceeds tolerance.
    
    Postconditions:
        - tissue.vertices contains merged global vertex pool
        - Each cell.vertex_indices references valid indices in tissue.vertices
        - No consecutive duplicate indices in any cell.vertex_indices
        - Each cell has at least 3 distinct vertices
        - All cell polygons maintain CCW orientation
        - Shared edges between cells reference identical vertex indices
    """
```

### Invariants

1. **Energy Invariance**: `|E_after - E_before| <= energy_tol`
2. **Area Invariance**: For each cell, `|A_after - A_before| <= geometry_tol`
3. **Perimeter Invariance**: For each cell, `|P_after - P_before| <= geometry_tol`
4. **Topological Validity**:
   - No cell has fewer than 3 distinct vertices
   - No consecutive duplicate vertex indices in any cell
   - All polygons are oriented CCW (positive signed area)
5. **Structural Integrity**:
   - No "cracks" along shared edges (cells sharing an edge reference the same vertex indices)
   - All vertex indices are in valid range [0, len(tissue.vertices))

### Algorithm

**Step 1: Build Vertex Proximity Graph**
```
1. Compute pre-merge energy: E_before = tissue_energy(tissue)
2. Compute pre-merge per-cell metrics: {cell_id: (area, perimeter)}
3. Build KD-tree or grid-based spatial index from tissue.vertices
4. Find all vertex pairs (i, j) where distance(v_i, v_j) <= distance_tol
5. Build undirected graph G where vertices are indices and edges connect nearby pairs
6. Find connected components C_1, C_2, ..., C_k (these are merge clusters)
```

**Step 2: Compute Representative Vertices**
```
For each cluster C_k with |C_k| > 1:
    v_rep = (1/|C_k|) * Σ_{i in C_k} v_i  (centroid)
    
    Alternative fallback if spread is large:
        v_rep = v_i where i = min(C_k)  (lowest index)
```

**Step 3: Build New Vertex Pool and Index Map**
```
1. Create old_to_new index mapping
2. For each cluster C_k:
   - If |C_k| == 1: map single index to new sequential index
   - If |C_k| > 1: all old indices map to same new index (the representative)
3. Build new_vertices array with only representative vertices
```

**Step 4: Update Cell Vertex Indices**
```
For each cell in tissue.cells:
    1. Map each old index to new index using old_to_new
    2. Remove consecutive duplicates (including wrap-around: first == last)
    3. Verify at least 3 distinct vertices remain
    4. Ensure CCW orientation (reverse if signed_area < 0)
    5. Update cell.vertex_indices
```

**Step 5: Validate Invariants**
```
1. Compute E_after = tissue_energy(tissue)
2. Verify |E_after - E_before| <= energy_tol
3. For each cell:
   - Compute new area and perimeter
   - Verify |new_area - old_area| <= geometry_tol
   - Verify |new_perimeter - old_perimeter| <= geometry_tol
4. Update tissue.vertices with new_vertices
5. Reconstruct cell.vertices from tissue.vertices[cell.vertex_indices]
```

### Test Strategy

1. **Artificial Near-Duplicates Test**:
   - Create tissue with intentionally duplicated vertices (offset by < distance_tol)
   - Verify merging reduces vertex count
   - Verify energy unchanged within tolerance

2. **No-Merge Test**:
   - Create tissue with all vertices separated by > distance_tol
   - Verify no merging occurs

3. **Shared Edge Test**:
   - Create 2-cell tissue where cells share an edge
   - Add near-duplicate vertices along shared edge
   - Verify after merge, both cells reference same vertex indices for shared edge

4. **Three-Way Junction Test**:
   - Create 3-cell tissue meeting at a point
   - Add near-duplicate vertices at junction
   - Verify all three cells reference same vertex index at junction

5. **Degenerate Prevention Test**:
   - Create scenario where naive merging would reduce a cell to < 3 vertices
   - Verify merge fails or adjusts appropriately

---

## Feature 2: Edge Meshing with Discrete Modes

### Contract

```python
def mesh_edges(
    tissue: Tissue,
    mode: str = "none",
    length_scale: float = 1.0,
    energy_tol: float = 1e-10,
    geometry_tol: float = 1e-10
) -> dict:
    """
    Subdivide edges according to specified meshing mode.
    
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
    
    Postconditions:
        - tissue.vertices contains updated global vertex pool
        - Each cell.vertex_indices references valid indices
        - Shared edges have identical intermediate vertices
        - Cell polygons maintain CCW orientation and ≥3 vertices
        - Area and perimeter are preserved within tolerance
    """
```

### Meshing Modes

| Mode     | Subdivision Logic                                    |
|----------|------------------------------------------------------|
| `none`   | `n_segments = 1`, no new vertices                    |
| `low`    | `n_segments = 2`, add single midpoint at `t = 0.5`   |
| `medium` | `n_segments = max(1, round(L / (1.0 * length_scale)))` |
| `high`   | `n_segments = max(1, round(L / (0.5 * length_scale)))` |

For each edge with `n_segments > 1`, intermediate vertices are placed at:
```
t_k = k / n_segments  for k = 1, 2, ..., n_segments - 1
v_intermediate = (1 - t_k) * v_start + t_k * v_end
```

### Edge Key Concept

To ensure shared edges receive identical subdivision:
```python
edge_key = (min(i, j), max(i, j))  # Canonical ordering
```

All cells sharing an edge `(v_i, v_j)` use the same `edge_key` to look up the subdivision.

### Algorithm

**Step 1: Pre-compute Metrics**
```
1. E_before = tissue_energy(tissue)
2. Per-cell metrics: {cell_id: (area_before, perimeter_before)}
3. Collect all edges from all cells with their edge_keys
```

**Step 2: Build Edge Subdivision Map**
```
edge_subdivision = {}  # edge_key -> list of (new_vertex_index, ...)

next_vertex_index = len(tissue.vertices)
new_vertices_list = list(tissue.vertices)

For each unique edge_key (min_idx, max_idx):
    v_start = tissue.vertices[min_idx]
    v_end = tissue.vertices[max_idx]
    L = distance(v_start, v_end)
    
    If L < min_edge_length:
        edge_subdivision[edge_key] = [min_idx, max_idx]  # No subdivision
        continue
    
    Compute n_segments based on mode:
        - none: n_segments = 1
        - low: n_segments = 2
        - medium: n_segments = max(1, round(L / length_scale))
        - high: n_segments = max(1, round(L / (0.5 * length_scale)))
    
    If n_segments == 1:
        edge_subdivision[edge_key] = [min_idx, max_idx]
    Else:
        intermediate_indices = []
        For k in 1..n_segments-1:
            t = k / n_segments
            v_new = (1 - t) * v_start + t * v_end
            new_vertices_list.append(v_new)
            intermediate_indices.append(next_vertex_index)
            next_vertex_index += 1
        
        edge_subdivision[edge_key] = [min_idx] + intermediate_indices + [max_idx]
```

**Step 3: Reconstruct Cell Polygons**
```
For each cell in tissue.cells:
    new_poly = []
    old_indices = cell.vertex_indices
    
    For i in range(len(old_indices)):
        v_start = old_indices[i]
        v_end = old_indices[(i + 1) % len(old_indices)]
        
        edge_key = (min(v_start, v_end), max(v_start, v_end))
        edge_seq = edge_subdivision[edge_key]
        
        # Determine if we need to reverse the sequence
        if v_start == edge_key[0]:
            # Forward: min -> max
            segment = edge_seq[:-1]  # Exclude last to avoid duplication
        else:
            # Reverse: max -> min
            segment = edge_seq[::-1][:-1]
        
        new_poly.extend(segment)
    
    # Remove consecutive duplicates and ensure CCW
    new_poly = remove_consecutive_duplicates(new_poly)
    cell.vertex_indices = np.array(new_poly, dtype=int)
    
    # Ensure CCW orientation
    if signed_area(tissue.vertices[cell.vertex_indices]) < 0:
        cell.vertex_indices = cell.vertex_indices[::-1]
```

**Step 4: Update Tissue and Validate**
```
1. tissue.vertices = np.array(new_vertices_list)
2. tissue.reconstruct_cell_vertices()
3. E_after = tissue_energy(tissue)
4. Verify |E_after - E_before| <= energy_tol
5. For each cell:
   - Verify area change within tolerance
   - Verify perimeter change within tolerance
```

### Test Strategy

1. **Mode "none" Test**:
   - Apply mesh_edges with mode="none"
   - Verify vertex count unchanged
   - Verify energy and geometry identical

2. **Mode "low" Test**:
   - Create single triangle cell
   - Apply mode="low"
   - Verify each edge has one midpoint
   - Verify polygon now has 6 vertices
   - Verify energy unchanged

3. **Mode "medium" Test**:
   - Create cell with edges of varying lengths
   - Apply mode="medium" with length_scale=1.0
   - Verify long edges have multiple subdivisions
   - Verify short edges have fewer subdivisions

4. **Mode "high" Test**:
   - Similar to medium but with denser subdivision
   - Verify subdivision count matches expected formula

5. **Shared Edge Consistency Test**:
   - Create 2-cell tissue with shared edge
   - Apply meshing
   - Verify both cells reference same intermediate vertex indices for shared edge

6. **Three-Way Junction Test**:
   - Create tissue where 3 cells meet at a point
   - Apply meshing
   - Verify edges maintain shared vertex at junction

7. **Tiny Edge Test**:
   - Create cell with one very short edge (< 0.1 * length_scale)
   - Verify tiny edge is not subdivided

8. **Energy/Geometry Invariance Test**:
   - Apply meshing to honeycomb tissue
   - Verify total energy unchanged within tolerance
   - Verify per-cell area and perimeter unchanged

---

## Integration Considerations

### Workflow

Typical usage pattern:

```python
# Build tissue
tissue = build_honeycomb_3_4_5_4_3(hex_size=1.0)
tissue.build_global_vertices(tol=1e-8)

# Optional: merge any near-coincident vertices from construction
result = merge_nearby_vertices(tissue, distance_tol=1e-8)

# Optional: increase mesh resolution for simulation
result = mesh_edges(tissue, mode="medium", length_scale=0.5)

# Run simulation
params = EnergyParameters(...)
sim = Simulation(tissue=tissue, energy_params=params)
sim.run(n_steps=1000)
```

### Builder Integration

Builders like `build_honeycomb_*` could optionally accept a `mesh_mode` parameter:

```python
def build_honeycomb_2_3_4_3_2(hex_size: float = 1.0, mesh_mode: str = "none") -> Tissue:
    tissue = _build_basic_honeycomb(hex_size)
    tissue.build_global_vertices()
    if mesh_mode != "none":
        mesh_edges(tissue, mode=mesh_mode)
    return tissue
```

### Performance Considerations

- Vertex merging: O(V log V) with KD-tree, O(V²) with naive pairwise
- Edge meshing: O(E) where E = number of edges
- Both operations should complete quickly for typical tissue sizes (< 1000 cells)

---

## Future Extensions

1. **Adaptive Meshing**: Vary subdivision density based on local curvature or energy gradient
2. **Periodic Boundary Conditions**: Handle edge wrapping for periodic tissues
3. **Mesh Quality Metrics**: Report statistics on edge length distribution, cell aspect ratios
4. **Undo/Rollback**: Allow reverting to pre-mesh state if simulation becomes unstable
