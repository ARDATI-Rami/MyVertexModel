## Plan: Safe Vertex Merging and Edge Meshing Modes

High-level: introduce two tightly specified geometry operations on the global vertex pool—safe merging of near-coincident vertices and shared-edge meshing—together with contracts, invariants, and test strategies that guarantee energy invariance (within tolerance) and structural validity of tissues.

---

## Feature 1 – Safe Vertex Merging

### 1. Problem Statement

Vertices in the global vertex pool can become extremely close due to numerical drift or prior operations. If left unmerged, they increase vertex count and may cause pathological geometry. We want a robust, deterministic algorithm that merges clusters of near-coincident vertices across the entire tissue, updating all cells consistently while:

- Preserving each cell’s polygon geometry (area and perimeter) and thus mechanical energy within tight numerical tolerance.
- Preserving mesh connectivity and orientation constraints (no cracks, no duplicate consecutive vertices, CCW polygons, etc.).
- Respecting the existing global vertex pool and structural validation conventions already present in `myvertexmodel`.

This is a “cleanup” / regularization pass over an existing tissue, not part of the time integrator itself.

---

### 2. Contracts and Invariants

#### 2.1 Function-Level Contract (conceptual)

Introduce a high-level operation, e.g.:

- Conceptual entry point: `merge_nearby_vertices(tissue, distance_tol, energy_tol, geometry_tol)`.
- Inputs:
  - `tissue`:
    - Global vertex array: `vertices: np.ndarray[float64]` of shape `(N_vertices, 2)` or similar.
    - Cell list: each cell with an ordered list of vertex indices (polygons), consistent with existing model.
    - Model parameters needed to recompute energy: `k_area`, `k_perimeter`, `gamma`, and target areas `A0` per cell.
  - `distance_tol`:
    - Scalar > 0 defining “near-coincident” (e.g., relative to typical edge length; chosen by caller).
  - `energy_tol`:
    - Scalar > 0 specifying acceptable relative/absolute change in total energy.
  - `geometry_tol`:
    - Scalar > 0 specifying acceptable relative/absolute change in per-cell area and perimeter.
- Outputs:
  - New tissue object, or in-place updated `tissue`, with:
    - Reduced number of vertices.
    - Updated cell vertex index lists.
  - Diagnostics (optional but recommended):
    - Number of vertices before/after.
    - Number of merged clusters.
    - Max deviation in area, perimeter, and energy.

#### 2.2 Invariants and Guarantees

- Topological / structural invariants:
  - No cell has fewer than 3 distinct vertices.
  - No duplicate consecutive vertex indices in any cell polygon.
  - Polygons remain consistently oriented (e.g., CCW as required by the existing validation).
  - Shared edges remain shared:
    - If two cells shared an edge before merging, they share exactly the same (possibly updated) vertex indices afterward.
  - Global vertex indices are dense: `0..N'-1` with no gaps after cleanup.
- Geometric invariants (within tolerance):
  - For each cell `c`:
    - `|A_c_after - A_c_before| <= geometry_tol_abs + geometry_tol_rel * |A_c_before|`.
    - `|P_c_after - P_c_before| <= geometry_tol_abs + geometry_tol_rel * |P_c_before|`.
  - No polygon self-intersections are introduced by merging.
  - No new “degenerate” edges with length ≈ 0 beyond what was already effectively degenerate (tiny edges may be removed/merged as part of cleanup).
- Energy invariants (within tolerance):
  - Define total energy:
    - `E = Σ_cells [0.5 * k_area * (A - A0)^2 + 0.5 * k_perimeter * P^2 + gamma * P]`.
  - Require:
    - `|E_after - E_before| <= energy_tol_abs + energy_tol_rel * |E_before|`.
  - Defaults: energy tolerances may be tied to machine eps times problem scales; tests will set concrete numerical values.
- Non-goals / allowed changes:
  - Vertex indices may be completely renumbered (stable mapping is not required as long as topology is preserved).
  - Cells that were geometrically degenerate (e.g., collinear vertices, tiny edges) may be slightly “regularized” as long as invariants above are met.

---

### 3. Algorithm for Merging Near-Coincident Vertices

The algorithm operates on the entire global vertex pool and all cell polygons. It proceeds in three phases: cluster formation, representative position selection, and global index/geometry update, followed by validation.

#### 3.1 Phase 1: Identify Clusters of Near-Coincident Vertices

Goal: partition the set of vertex indices into disjoint clusters where all members are mutually close (within `distance_tol`).

Steps:

1. **Collect candidate pairs with spatial indexing:**
   - Use a spatial data structure appropriate to `vertices` (e.g., KD-tree or uniform grid bucketing).
   - For each vertex `i`, query neighbors `j` where `||v_i - v_j|| <= distance_tol`.
   - Create an undirected edge `(i, j)` for each such pair.

2. **Construct connected components:**
   - Treat the graph `G = (V_vertices, E_pairs)` as an undirected graph.
   - Find connected components via union-find or BFS/DFS.
   - Each component is a cluster `C_k` (set of old vertex indices).

3. **Filter trivial clusters:**
   - Clusters with size 1 are left unchanged.
   - For clusters whose maximum pairwise distance exceeds a safety multiple of `distance_tol` (e.g., `2 * distance_tol`), either:
     - (Default) still treat them as a cluster (since graph connectivity defined them), or
     - (Conservative) split them via more refined clustering (hierarchical) if needed.
   - For initial implementation, assuming `distance_tol` small, keeping components as is is acceptable.

#### 3.2 Phase 2: Select Representative Vertex Positions

We need a rule to define the replacement position for each cluster that maintains geometry and energy as much as possible.

Design options:

- **Option A (Centroid representative):**
  - Compute centroid of the cluster:
    - `v_rep = (1 / |C_k|) * Σ_{i in C_k} v_i`.
  - Use this for all merged vertices in cluster.
  - Pros: symmetric, consistent.
  - Cons: might move vertices that originally define large-scale geometry.

- **Option B (Non-moving primary vertex):**
  - Choose a “primary” vertex within the cluster based on:
    - The vertex with minimal degree change impact (e.g., appears in most cells, or has highest incident edge length).
    - Or simply the vertex with smallest index.
  - Set `v_rep = v_primary` (no movement).
  - Pros: no movement for main vertex; local geometry changes less for cells using this vertex only.
  - Cons: other vertices may be farther.

Recommended plan:

- Use a hybrid approach with a configurable policy:
  - Default policy: **centroid**, because it minimizes average displacement and smooths small numerical noise.
  - Allow a “snap-to-existing” option if desired (not required for first implementation but plan for it).

Algorithm details:

1. For each cluster `C_k`:
   - Compute centroid `v_centroid`.
   - Compute maximum displacement `max_disp = max_i ||v_i - v_centroid||`.
   - If `max_disp <= small_threshold` (e.g., `0.1 * distance_tol`), accept `v_rep = v_centroid`.
   - Otherwise, optionally switch to non-moving representative for robustness:
     - Pick `v_primary` as vertex of cluster with minimal sum of squared distances to all others, or minimal index.
     - Set `v_rep = v_primary`.
   - Record mapping `rep_pos[k] = v_rep`.

2. Record index mapping:
   - Choose a single new representative index for cluster `C_k`, e.g. `rep_idx_k`:
     - For planning, assume we assign contiguous new indices after all clusters are known.
   - Maintain mapping `old_idx -> new_idx`:
     - For each `i in C_k`, `new_index_of[i] = rep_idx_k`.

#### 3.3 Phase 3: Update Cells and Global Vertex Pool

1. **Build new vertex array:**
   - Let `K` be number of clusters.
   - Create `vertices_new` of length `K`.
   - For each cluster `C_k`, assign:
     - `vertices_new[rep_idx_k] = rep_pos[k]`.

2. **Update cell vertex index lists:**
   - For each cell `c` with polygon `[i_0, i_1, ..., i_{m-1}]`:
     - Map each index: `j_t = new_index_of[i_t]`.
     - Remove duplicate consecutive vertices:
       - If `j_t == j_{t-1}`, skip `j_t`.
     - Also check wrap-around:
       - If first and last indices are equal after mapping, drop one.
     - If polygon length drops below 3:
       - Mark cell as invalid / raise error (in practice this should be rare with small `distance_tol`; tests should detect).
     - Ensure or enforce CCW orientation:
       - Compute polygon signed area; if negative, reverse vertex order.
   - Store updated polygons as the cell’s vertex index list.

3. **Clean up orphaned vertices and reindex (if needed):**
   - In this plan, old indices are fully replaced by new dense `[0..K-1]` indices.
   - There should be no unused vertices in `vertices_new`.
   - Ensure internal consistency with existing data structures (e.g., edge lists or neighbors, if present) are updated via the same `old_idx -> new_idx` mapping.

4. **Optional: local sanity checks:**
   - For each cell:
     - Compute area and perimeter using `vertices_new`.
     - Check for self-intersections (if utility exists).
   - Ensure no cell fails structural validation logic already in `myvertexmodel`.

#### 3.4 Phase 4: Energy and Geometry Validation

After merging:

1. Recompute total energy `E_after` using existing energy functional.
2. Compare:
   - `|E_after - E_before|` against absolute/relative `energy_tol`.
   - For each cell:
     - Compare area/perimeter changes with `geometry_tol`.
3. If violation detected:
   - For now, fail/raise and report diagnostics (clusters that moved the most, largest offending cell).
   - Optionally, a more advanced implementation could:
     - Retry with a smaller `distance_tol`, or
     - Use primary-vertex instead of centroid for problematic clusters.
   - The initial implementation plan assumes we treat this as an error.

---

### 4. Test Strategy for Safe Vertex Merging

We want tests that assert both structural correctness and energy/geometry invariance across various tissues and parameter settings.

#### 4.1 Test Tissue Construction

Use existing builders and fixtures already present in `tests/` and `examples/`:

- Base tissues:
  - Honeycomb tissue with moderate number of cells (e.g., from `build_honeycomb_tissues.py` or `builders`).
  - ACAM-converted tissue sample (e.g., based on `acam_79cells`).
  - Simple 3–4 cell toy tissue for manual reasoning.
- For each base tissue:
  - Ensure we can compute:
    - Global vertex array.
    - Per-cell vertex index lists.
    - Energy parameters (`k_area`, `k_perimeter`, `gamma`) and `A0` per cell.

#### 4.2 Generating Artificial Near-Duplicates

Construct scenarios where vertices are intentionally perturbed to create near-coincident clusters:

1. **Cluster around a single vertex:**
   - Pick a vertex `v` in the global pool.
   - Create copies:
     - Add new entries `v_i = v + δ_i`, where `||δ_i|| << distance_tol`.
   - For some cells that use `v`, replace references with the new near-duplicate indices (e.g., alternate cells use different copies).
   - Result: multiple indices at nearly same location used by different cells.

2. **Pairs on an edge:**
   - For an existing edge `(i, j)`:
     - Create a near-duplicate `i'` and `j'` with very small displacements.
     - Some cells use `(i, j)`, others use `(i', j')`.
   - This tests shared-edge consistency after merging.

3. **Stress cases:**
   - Small clusters near each other (e.g., three vertices forming a tiny triangle, all within `distance_tol`).
   - Clusters in high-degree junctions (vertices with many incident cells).

Ensure these modifications still pass basic structural validation before running the merge algorithm.

#### 4.3 Measuring Energy and Geometry Invariance

For each test scenario:

1. Compute pre-merge metrics:
   - For each cell:
     - Area `A_before`.
     - Perimeter `P_before`.
   - Total energy `E_before`.
2. Run `merge_nearby_vertices` with a selected `distance_tol`:
   - Use distance tolerance comfortably larger than perturbations but small relative to typical edge lengths.
3. Compute post-merge metrics:
   - Recompute `A_after`, `P_after` per cell and `E_after`.
4. Assertions:
   - For every cell:
     - Assert `|A_after - A_before| <= geometry_tol_abs + geometry_tol_rel * |A_before|`.
     - Assert `|P_after - P_before| <= geometry_tol_abs + geometry_tol_rel * |P_before|`.
   - For total energy:
     - Assert `|E_after - E_before| <= energy_tol_abs + energy_tol_rel * |E_before|`.
   - Additionally:
     - Assert no cell has fewer than 3 vertices.
     - Assert polygons remain CCW (using existing orientation/validation).
     - Assert no duplicate consecutive vertices.

Choose specific tolerances for tests (example numbers; final choice should reflect model scales):

- `geometry_tol_rel ≈ 1e-10`, `geometry_tol_abs ≈ 1e-12`.
- `energy_tol_rel ≈ 1e-10`, `energy_tol_abs ≈ 1e-12`.

#### 4.4 Cluster and Topology Specific Tests

Focus tests on vertex mapping correctness:

- **Shared vertex reference merge:**
  - Create a scenario where two cells share a vertex via different near-duplicate indices.
  - After merging:
    - assert they share the same index at that junction.
- **No cracks along shared edges:**
  - For every edge `(i, j)` traversed in two different cells:
    - After merge, confirm that:
      - Both cells use the same `(i', j')` pair up to orientation.
- **Index remapping stability:**
  - Confirm that all indices in cell polygons are within `[0, N_new-1]` and no vertex is unused.
- **Regression tests:**
  - Add regression tests based on real tissues known to be problematic (e.g., ACAM examples that previously produced near-duplicates).

---

## Feature 2 – Edge Meshing with Discrete Modes

### 1. Problem Statement

We want to refine cell boundaries by inserting extra vertices along edges according to discrete “meshing modes”:

- Modes:
  - `none`: no additional vertices.
  - `low`: one midpoint vertex per edge.
  - `medium`: target ~1 vertex per 1.0 unit of edge length.
  - `high`: target ~1 vertex per 0.5 units of edge length.

Because the model uses a global vertex pool with shared vertices along cell interfaces, meshing must:

- Operate on the global vertex list.
- Respect shared edges (no cracks): new intermediate vertices on a shared edge must be shared by all adjacent cells.
- Preserve polygon geometry (shape/area/perimeter) and thus energy (within tolerance).
- Maintain structural invariants (no self-intersections, consistent orientation, no degenerate polygons).

This is a geometric refinement operation that can be applied, for example, at tissue initialization or prior to simulation.

---

### 2. Contracts and Invariants for Meshing

#### 2.1 Function-Level Contract (conceptual)

Define a function such as:

- Conceptual entry point: `mesh_edges(tissue, mode, length_scale=1.0, energy_tol, geometry_tol)`.
- Inputs:
  - `tissue`:
    - Same structure as for merging: global vertex array and cell polygons.
  - `mode`: enumeration `{none, low, medium, high}`.
  - `length_scale` (optional): base length scale used for “medium” and “high” modes; default 1.0 units.
  - `energy_tol`, `geometry_tol`: tolerances as before.
- Outputs:
  - New / updated `tissue` with:
    - Possibly more vertices.
    - Updated cell vertex index lists.
  - Diagnostics:
    - Number of vertices before/after.
    - Average refinement factor.
    - Max area/perimeter/energy deviation.

#### 2.2 Invariants and Guarantees

- Geometry invariants (within tolerance):
  - For each cell:
    - `A_after ≈ A_before` (using same criterion as in merging).
    - `P_after ≈ P_before` (tiny increases allowed due to numerical noise only).
  - Edges are subdivided linearly; no new global shape distortion.
- Energy invariants:
  - Same total energy invariance contract as in merging.
- Topological / structural invariants:
  - Shared edges remain shared:
    - If two cells share an original edge `(i, j)`, their post-meshing vertex lists along that edge are identical sets of vertices in reverse order.
  - No cracks:
    - Every intermediate vertex on an interface is used consistently by all adjacent cells.
  - No degenerate polygons:
    - All cells have at least 3 distinct vertices.
    - No zero-length edges introduced except where original edge was already effectively zero-length (degenerate edge handling covers this).
  - Orientation consistency:
    - Cell polygons stay CCW (or consistent with current convention).
- Non-goals:
  - No requirement to preserve previous vertex indexing order beyond what’s needed to maintain polygon adjacency.

---

### 3. Detailed Meshing Algorithm

Key design: meshing is performed in the space of the global vertex pool and cell polygons, with a central edge-key mapping to ensure shared intermediate vertices.

#### 3.1 Edge Key Mapping and Edge Traversal

We treat each unique undirected edge as:

- Edge key: `(min(i, j), max(i, j))` where `i` and `j` are vertex indices in the global vertex array.

Steps:

1. **Enumerate edges per cell:**
   - For each cell polygon `[v_0, v_1, ..., v_{m-1}]`, form edges:
     - `(v_k, v_{(k+1) mod m})` for `k=0..m-1`.
   - For each edge, compute:
     - `edge_key = (min(v_k, v_{k+1}), max(v_k, v_{k+1}))`.
   - Record:
     - For each cell, a list of edges with orientation:
       - `edge_list_c = [ (v_k, v_{k+1}, edge_key) ]`.
2. **Shared edge gathering:**
   - Build a dictionary: `edge_key -> list of (cell_id, local_edge_index, v_start, v_end)`.
   - This allows us to ensure that all cells sharing an edge get the same subdivision.

#### 3.2 Mode-Dependent Subdivision Rule

Define a function to determine subdivisions for an edge of length `L`:

- Compute edge length:
  - `L = ||vertices[j] - vertices[i]||`.

- For each mode:

  - `none`:
    - `n_segments = 1` (no new points).
    - List of parametric positions (normalized from 0 to 1): `[0.0, 1.0]` only (endpoints only).

  - `low`:
    - Always add a single midpoint for non-degenerate edges:
      - `n_segments = 2`.
      - Positions: `[0.0, 0.5, 1.0]`.
    - For degenerate edges (`L < tiny_thresh`):
      - Treat as `none` (no subdivision) to avoid creating clustered points.

  - `medium` (~1 vertex per 1.0 unit of length):
    - Only meaningfully applied if `L` is not tiny.
    - Choose:
      - `target_segment_length = 1.0 * length_scale`.
      - `n_segments = max(1, round(L / target_segment_length))`.
    - Ensure not to over-refine:
      - Optional cap on `n_segments` (e.g., <= 20 or based on global vertex count).
    - Parametric positions:
      - `t_k = k / n_segments` for `k=0..n_segments`.

  - `high` (~1 vertex per 0.5 units):
    - `target_segment_length = 0.5 * length_scale`.
    - Same formula:
      - `n_segments = max(1, round(L / target_segment_length))`.
      - `t_k = k / n_segments`.

- Degenerate/tiny edges:
  - If `L < tiny_thresh` (e.g., `tiny_thresh ≈ 1e-8` or relative to average edge length):
    - Treat as `none`:
      - No intermediate vertices; keep endpoints only.
    - Alternatively, one might collapse tiny edges via merging, but that should be part of the merging feature, not meshing.

#### 3.3 Computing Positions of Intermediate Vertices

Given endpoints `p0 = vertices[i]`, `p1 = vertices[j]` and parametric positions `t_k`:

- For each intermediate index `1 <= k <= n_segments-1`:
  - `p_k = (1 - t_k) * p0 + t_k * p1` (linear interpolation).
- Important:
  - We always include endpoints `t_0=0` and `t_{n_segments}=1`, but they correspond to existing vertices and should not create new global entries.

#### 3.4 Global Vertex Sharing via Edge Key Map

We must ensure that each edge gets a single, shared list of intermediate vertex indices:

1. For each unique `edge_key`:

   - If `mode == none`:
     - Store:
       - `subdivision[edge_key] = [ (i, j) ]` (no intermediate vertices).

   - Otherwise:
     - Compute edge length `L`.
     - Compute `n_segments` and parametric list `t_k`.
     - Derive coordinates for each new intermediate vertex.
     - For each new `p_k`:
       - Assign a new global vertex index from the vertex pool (append to `vertices_new`).
     - Create an ordered list of vertex indices along this edge in a canonical orientation:
       - Choose orientation `i -> j` for storage:
         - `edge_sequence = [i, new_idx_1, ..., new_idx_{n_segments-1}, j]`.
     - Record:
       - `edge_subdivision[edge_key] = edge_sequence`.

2. **Orientation consistency per cell:**
   - When reconstructing a cell polygon, if the local edge is `(v_start, v_end)`:
     - If `(v_start, v_end)` aligned with `edge_seq[0], edge_seq[-1]`:
       - Use `edge_sequence` as stored.
     - If it is reversed `(v_end, v_start)`:
       - Use reversed list (excluding double-count of shared vertices):
         - When stitching polygons, care is taken not to duplicate endpoints (see next section).

This guarantees identical subdivision and shared vertices along each common edge across all cells.

#### 3.5 Reconstructing Cell Polygons

For each cell with original vertex list `[v_0, v_1, ..., v_{m-1}]`:

1. Initialize an empty list `new_poly`.
2. For each edge in order `k=0..m-1`:
   - Let `v_start = v_k`, `v_end = v_{(k+1) mod m}`.
   - Derive `edge_key = (min(v_start, v_end), max(v_start, v_end))`.
   - Retrieve `edge_seq = edge_subdivision[edge_key]`.
   - Based on orientation:
     - If `(v_start, v_end)` aligned with `edge_seq[0], edge_seq[-1]`:
       - Use `seq = edge_seq`.
     - If reversed:
       - Use `seq = reversed(edge_seq)`.
   - Append to `new_poly` the sequence `seq`, but avoid duplicating vertices at joins:
     - For `k > 0`:
       - Drop the first element of `seq` (it was the last vertex of previous edge).
     - For `k == 0`:
       - Keep full `seq`.
3. After the last edge:
   - The polygon will close automatically since final vertex of last edge equals the initial vertex of first edge.
   - Check and remove any dual closure if present (no need for last == first duplication).
4. Remove any duplicate consecutive vertices created by `none` / tiny-edge handling.
5. If polygon ends up with fewer than 3 distinct vertices, treat as error; such cases should not appear in normal meshing because we’re only subdividing edges.

6. Ensure CCW orientation:
   - Compute polygon signed area; reverse if needed.

#### 3.6 Handling Tiny/Degenerate Edges

To avoid introducing unstable small segments:

- Define `tiny_thresh` globally or per tissue:
  - e.g., `tiny_thresh = eps_rel * typical_edge_length + eps_abs`.
- For any edge with `L < tiny_thresh`:
  - Set mode effectively to `none` regardless of global `mode`.
  - Do not create intermediate vertices.
- Optionally, combine with safe vertex merging:
  - Run merging before meshing to reduce degeneracies.
  - Plan allows, but doesn’t enforce, such a sequence.

---

### 4. Test Strategy for Edge Meshing

We want to validate both correctness of geometry/energy and the correctness of shared-edge handling for all modes.

#### 4.1 Geometry and Energy Invariance Tests

For each tissue (e.g., honeycomb, ACAM, simple grid):

1. Compute baseline metrics:
   - For each cell:
     - `A_before`, `P_before`.
   - Total energy `E_before`.

2. For each `mode in {none, low, medium, high}`:

   - Clone tissue.
   - Apply `mesh_edges` with chosen mode.
   - Compute `A_after`, `P_after` per cell and `E_after`.

3. Assertions:

   - For `mode == none`:
     - Tissues should remain bitwise-identical at structure level (if implemented purely as no-op), or at least:
       - No change in number of vertices.
       - No change in polygons (same sequences of indices).
       - Energy and geometry should be exactly equal (or within machine precision).

   - For `mode in {low, medium, high}`:
     - Number of vertices should increase.
     - Per-cell invariants:
       - `|A_after - A_before| <= geometry_tol_abs + geometry_tol_rel * |A_before|`.
       - `|P_after - P_before| <= geometry_tol_abs + geometry_tol_rel * |P_before|`.
     - Global energy invariance:
       - `|E_after - E_before| <= energy_tol_abs + energy_tol_rel * |E_before|`.

   - Structural invariants:
     - Every cell has ≥ 3 vertices.
     - Polygons oriented CCW (consistent with existing expectations).

#### 4.2 Shared Edge and Orientation Tests

Design specific tests to assert correctness of shared-edge meshing:

1. **Two-cell shared edge test:**
   - Create a simple tissue with two cells sharing a single long edge.
   - Apply meshing in each mode.
   - For the shared edge:
     - Extract edge vertex sequences from each cell along that edge.
     - Assert:
       - Same vertex index multiset on the edge.
       - One sequence is exactly the reverse of the other (orientation difference).
     - Confirm number of intermediate vertices matches expected `n_segments - 1` from mode rules.

2. **Three-way junction test:**
   - A vertex where three cells meet and share three edges.
   - After meshing:
     - Confirm that on each edge, all cells share the same intermediate vertices.
     - Confirm that the cyclic ordering of vertices around the junction is consistent.

3. **Opposite-orientation polygon test:**
   - Construct two cells where one uses the shared edge in the opposite direction in its vertex list.
   - Meshing must yield reversed sequences but consistent indices:
     - `edge_seq_cellA == reversed(edge_seq_cellB)`.

#### 4.3 Degenerate Edge Tests

Specifically cover the tiny-edge behavior:

1. Create a tissue with:
   - One very short edge (length << `tiny_thresh`) and another long edge.
   - Apply `medium` or `high` meshing.
2. Assertions:
   - Short edge:
     - No new vertices are introduced on the short edge.
     - Endpoints remain unchanged.
   - Long edge:
     - Subdivided as expected based on length.
   - Geometry and energy invariance still hold.

#### 4.4 Performance Expectations

While exact performance targets are project-dependent, tests can set basic expectations:

1. Create a moderately large tissue (e.g., several hundred cells) with typical edge lengths.
2. For each mode (`low`, `medium`, `high`):
   - Time `mesh_edges` execution (informal; not strict benchmark).
   - Assert:
     - Runtime is approximately O(N_edges) (i.e., linear in number of edges).
     - Number of vertices increase is bounded:
       - Check that average `n_segments` isn’t exploding due to mis-specified target lengths.
3. Add a guard test to ensure:
   - No pathological explosion in vertices:
     - e.g., `n_vertices_after <= 50 * n_vertices_before` for given test tissue and modes.

---

## Overall Integration Considerations

### 1. Ordering of Operations

Plan for a clear sequence when both features are used:

1. Build tissue using existing builders.
2. Optionally run **safe vertex merging**:
   - Clean up near-coincident vertices to avoid redundant subdivisions or degenerate geometry.
3. Apply **edge meshing** according to chosen mode.
4. Run existing tissue validation logic and new tests.

### 2. Function Responsibilities and Placement

- Likely locations (conceptual, based on current layout):
  - Geometry and low-level operations: `geometry.py`-style helpers for:
    - Area/perimeter calculations.
    - Edge length, polygon orientation.
  - High-level tissue operations:
    - `simulation.py` or `builders.py`-style modules for:
      - `merge_nearby_vertices`.
      - `mesh_edges`.
  - Tests:
    - New test modules under `tests/`, e.g.:
      - `test_vertex_merging.py`.
      - `test_edge_meshing.py`.

Both features should reuse common utilities for:

- Computing areas/perimeters.
- Energy evaluation.
- Structural validation.

---

### 3. Further Considerations

1. Error-handling policy for invariance failures:
   - Should implementation raise exceptions, warn and continue, or attempt fallback strategies (e.g., reducing `distance_tol` or changing representative selection)?
2. Configuration and defaults:
   - Where should default tolerances and meshing modes live (e.g., global config, function arguments, or simulation parameters)?
3. Interplay with future features:
   - Consider leaving hooks for adaptive meshing based on local curvature or stress, and for energy-aware vertex merging strategies beyond simple distance thresholds.

