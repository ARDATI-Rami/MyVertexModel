# Vertex Model Design Document

## 1. Project Goal
Implement a 2D vertex model for epithelial tissue mechanics. The model represents cells as polygons sharing vertices in a global pool. It will support computing mechanical energies and (later) forces for simulation of tissue dynamics (growth, rearrangements, shape changes).

## 2. Core Data Structures
### 2.1 Global Vertex Array
- `Tissue.vertices`: shape (M, 2) float array of unique vertex coordinates.
- Vertices are shared among neighboring cells to enforce geometric/topological consistency.

### 2.2 Cells
- Each `Cell` represents a polygon via an ordered sequence of indices into `Tissue.vertices` (future: `Cell.vertex_indices`).
- Transitional phase: cells currently also store a local copy of coordinates (`Cell.vertices`) for backward compatibility.
- Orientation convention: counterclockwise ordering preferred (positive area); validation utilities will be added.

### 2.3 Indices / Connectivity
- `Cell.vertex_indices`: 1D int array; indices reference rows of `Tissue.vertices`.
- Edge list derivable by successive pairs of indices (last wraps to first).
- Future additional structures:
  - Half-edge or winged-edge structure for efficient neighbor queries.
  - Per-edge or per-junction metadata (tension coefficients, adhesion parameters).

### 2.4 Tissue
- Holds: global vertex pool, list of cells.
- Will later maintain derived caches (areas, perimeters, adjacency) for performance.

## 3. Energy Functional (Planned)
Total tissue energy: sum over cells + optional edge-level terms.

For a cell c:
\[
E_c = k_{area} (A_c - A_{0})^2 + k_{perimeter} P_c^2 + \sum_{e \in c} \gamma_e L_e
\]
Where:
- \(A_c\): polygon area of cell c.
- \(A_0\): target (preferred) area (global or per-cell future).
- \(P_c\): perimeter length.
- \(L_e\): length of edge e.
- \(\gamma_e\): line (junction) tension; currently uniform (\(\gamma\)), may become heterogeneous.

### 3.1 Components
- Area elasticity: penalizes deviation from preferred area.
- Perimeter contractility: models acto-myosin cortical tension.
- Line/junction tension: captures adhesion and differential contractility along cell-cell interfaces.

### 3.2 Extensions (Deferred)
- Bending / curvature terms (not typical for pure vertex but possible for specialized tissues).
- Area constraints via Lagrange multipliers for incompressibility.
- Tension anisotropy (direction-dependent \(\gamma\)).

## 4. Simulation Roadmap
1. Data migration: replace per-cell local `vertices` usage with global `vertex_indices`.
2. Energy evaluation: implement geometry queries (area, perimeter, edge lengths) from global indices.
3. Force derivation: compute gradients of energy w.r.t. vertex positions (analytical where feasible; fallback to automatic differentiation optional).
4. Time integration: explicit Euler prototype, then switch to semi-implicit or adaptive integrators if stiffness observed.
5. Topological events (future): T1 transitions (edge swaps), cell division (vertex insertion), apoptosis/extrusion (polygon removal).

## 5. Current Status (Interface Layer)
- `EnergyParameters` defined with placeholders: `k_area`, `k_perimeter`, `gamma`, `target_area`.
- `cell_energy` and `tissue_energy` stubs return 0.0; tests assert placeholder behavior.
- Global vertex pool present (`Tissue.vertices`) but not yet populated or referenced by cells.

## 6. Reference Implementation Considerations
- Numerical stability: energy gradients may require damping or adaptive dt.
- Consistency: shared vertices must update all incident cells deterministically.
- Performance: caching polygon area/perimeter; invalidated only when vertex subset moves.

## 7. Testing Strategy
- Unit tests for geometry (area, perimeter, centroid).
- Upcoming: tests for energy after functional implemented (non-zero expected values).
- Later: randomized meshes to validate invariants (no self-intersections, positive areas).

## 8. Key References
- Farhadifar et al. 2007. Mechanics of epithelial tissue: DOI:10.1016/j.cub.2007.11.049
- Nagai & Honda 2009. Vertex dynamics modeling methodology: DOI:10.1016/j.physd.2009.01.001
- Supplementary sources (future): analytical derivations for polygon energy gradients; computational geometry texts for robust area & intersection handling.

## 9. Planned Enhancements (Future Work)
- Per-cell parameter variability (heterogeneous target areas / tensions).
- Interface with image-derived segmentation to initialize vertex meshes.
- Persistence of topological events (event log for reproducibility).
- Optional integration of autograd (JAX / PyTorch) for force calculation prototyping.

## 10. Summary
The design centers on a shared global vertex representation enabling physically consistent multi-cell interactions, a modular energy functional capturing key biophysical terms, and an extensible pathway toward dynamic remodeling (topology changes). Current code establishes interfaces; next steps focus on migrating geometry and implementing real energy + forces.

