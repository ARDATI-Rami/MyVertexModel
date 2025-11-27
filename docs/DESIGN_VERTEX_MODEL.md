# Vertex Model Design Document

## 1. Project Goal
Implement a 2D vertex model for epithelial tissue mechanics. The model represents cells as polygons sharing vertices in a global pool. It will support computing mechanical energies and (later) forces for simulation of tissue dynamics (growth, rearrangements, shape changes).

## 2. Core Data Structures
### 2.1 Global Vertex Array
- `Tissue.vertices`: shape (M, 2) float array of unique vertex coordinates.
- Vertices are shared among neighboring cells to enforce geometric/topological consistency.

### 2.2 Cells
- Each `Cell` represents a polygon via an ordered sequence of indices into `Tissue.vertices` via `Cell.vertex_indices` (1D int array).
- **Dual representation**: Cells maintain both `vertex_indices` (references to global pool) and `vertices` (local coordinates) for flexibility and backward compatibility.
- The global representation is preferred for simulations; use `Tissue.build_global_vertices()` and `reconstruct_cell_vertices()` to synchronize.
- **Orientation convention**: Counterclockwise (CCW) ordering required for positive area. Validation utilities implemented in `geometry.py` (`is_valid_polygon`, `polygon_orientation`, `ensure_ccw`).

### 2.3 Indices / Connectivity
- `Cell.vertex_indices`: 1D int array; indices reference rows of `Tissue.vertices`.
- Edge list derivable by successive pairs of indices (last wraps to first).
- Future additional structures:
  - Half-edge or winged-edge structure for efficient neighbor queries.
  - Per-edge or per-junction metadata (tension coefficients, adhesion parameters).

### 2.4 Tissue
- Holds: global vertex pool (`vertices` array), list of cells.
- **Key methods**:
  - `build_global_vertices(tol)`: Merges cell vertices into global pool with tolerance-based deduplication
  - `reconstruct_cell_vertices()`: Rebuilds `cell.vertices` from global pool using `vertex_indices`
  - `validate()`: Checks structural integrity (polygon validity, areas, etc.)
- **Known issue**: ACAM tissue conversion can create duplicate consecutive vertices in `vertex_indices`, requiring validation and repair (see validation tools in `tests/test_tissue_cell_by_cell.py` and `examples/diagnose_tissue.py`).

## 3. Energy Functional (Implemented)
Total tissue energy: sum over cells + edge-level line tension.

**Implemented energy formula** for a cell c:
\[
E_c = \frac{1}{2} k_{area} (A_c - A_{0})^2 + \frac{1}{2} k_{perimeter} P_c^2 + \gamma P_c
\]
Where:
- \(A_c\): polygon area of cell c.
- \(A_0\): target (preferred) area (per-cell via dict or global float).
- \(P_c\): perimeter length.
- \(\gamma\): uniform line (junction) tension coefficient.

**Implementation details**:
- `cell_energy()`: Computes energy for a single cell, supports both local and global vertex representations
- `tissue_energy()`: Sums all cell energies, preferring global vertex pool when available
- Energy gradient computed via finite differences (`finite_difference_cell_gradient()`)
- Analytical gradient also available (`cell_energy_gradient_analytic()`)

### 3.1 Components
- Area elasticity: penalizes deviation from preferred area.
- Perimeter contractility: models acto-myosin cortical tension.
- Line/junction tension: captures adhesion and differential contractility along cell-cell interfaces.

### 3.2 Extensions (Deferred)
- Bending / curvature terms (not typical for pure vertex but possible for specialized tissues).
- Area constraints via Lagrange multipliers for incompressibility.
- Tension anisotropy (direction-dependent \(\gamma\)).

## 4. Simulation Status
**Implemented:**
1. ✅ Data structures: `Cell.vertex_indices` and `Tissue.vertices` with global vertex pool
2. ✅ Energy evaluation: Full implementation with area, perimeter, and line tension terms
3. ✅ Force derivation: Finite difference gradient computation + analytical gradient option
4. ✅ Time integration: Explicit gradient descent with configurable damping and timestep
5. ✅ Tissue builders: Grid and honeycomb tissue generators with automatic global vertex pool
6. ✅ ACAM tissue import: Topology-aware converter from ACAM center-based models
7. ✅ Validation tools: Cell-by-cell validation, polygon validity checks, duplicate vertex detection
8. ✅ Visualization: Enhanced plotting with cell/vertex ID labels

**Future work:**
- Topological events: T1 transitions (edge swaps), cell division (vertex insertion), apoptosis/extrusion
- Adaptive timestep based on gradient magnitude
- Per-edge heterogeneous tension (currently uniform γ)
- Cached geometry computations for performance

## 5. Current Implementation Status
**Fully Implemented:**
- `EnergyParameters`: All fields functional (`k_area`, `k_perimeter`, `gamma`, `target_area`)
- `cell_energy()`: Complete implementation with area elasticity, perimeter contractility, and line tension
- `tissue_energy()`: Sums cell energies, supports global vertex representation
- `Simulation` class: Gradient descent with configurable dt, damping, and validation
- `GeometryCalculator`: Area, perimeter, centroid calculations with polygon validation
- Global vertex pool: `build_global_vertices()` and `reconstruct_cell_vertices()` fully operational
- Tissue builders: `build_grid_tissue()`, `build_honeycomb_2_3_4_3_2()`, `build_honeycomb_3_4_5_4_3()`
- ACAM importer: `convert_acam_with_topology()` with neighbor topology awareness
- Validation: `validate()` method, polygon validity checks, CCW ordering enforcement

**Key Implementation Notes:**
- Energy formula: E = 0.5*k_area*(A-A0)² + 0.5*k_perimeter*P² + γ*P
- Gradient computation: Central finite differences (analytical gradient also available)
- Timestep requirements: ACAM tissues (~79 cells) need dt≈0.0001; Honeycomb (~14 cells) can use dt≈0.01
- Known validation requirement: ACAM tissues should be validated for duplicate consecutive vertices post-conversion

## 6. Reference Implementation Considerations
- Numerical stability: energy gradients may require damping or adaptive dt.
- Consistency: shared vertices must update all incident cells deterministically.
- Performance: caching polygon area/perimeter; invalidated only when vertex subset moves.

## 7. Testing Strategy
**Implemented:**
- ✅ Unit tests for geometry (area, perimeter, centroid) - `tests/test_basic.py`
- ✅ Energy tests with non-zero validation - energy functional fully tested
- ✅ Cell-by-cell structural validation - `tests/test_tissue_cell_by_cell.py` validates all tissues
- ✅ Polygon validity tests - checks for self-intersection, degenerate cases, CCW ordering
- ✅ Duplicate vertex detection - identifies and reports duplicate consecutive vertices
- ✅ Simulation stability tests - validates energy evolution during dynamics
- ✅ ACAM conversion validation - topology connectivity and vertex sharing validation

**Test Coverage:**
- Geometry calculations (area, perimeter, valid polygons)
- Energy evaluation (area elasticity, perimeter, line tension)
- Gradient computation (finite differences vs analytical)
- Tissue building (grid, honeycomb patterns)
- Global vertex pool operations (build, reconstruct, merge)
- Simulation dynamics (timestep stability, energy conservation)
- Validation tools (structure integrity, duplicate detection)

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

