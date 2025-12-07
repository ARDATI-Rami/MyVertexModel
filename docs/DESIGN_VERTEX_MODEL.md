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

### 2.5 Module Organization
The codebase is organized into focused modules for maintainability:

| Module | Contents | Lines |
|--------|----------|-------|
| `core.py` | `Cell`, `Tissue` classes with global vertex pool | ~330 |
| `energy.py` | `EnergyParameters`, `cell_energy()`, `tissue_energy()` | ~180 |
| `mesh_ops.py` | `merge_nearby_vertices()`, `mesh_edges()` | ~450 |
| `geometry.py` | `GeometryCalculator`, polygon utilities | ~150 |
| `simulation.py` | `Simulation` class, gradient computation | ~390 |
| `builders.py` | Tissue builders (honeycomb, grid) | ~240 |
| `acam_importer.py` | ACAM tissue conversion | ~910 |
| `io.py` | Save/load utilities | ~100 |
| `plotting.py` | Visualization utilities | ~185 |

All public symbols are re-exported through `__init__.py` for convenient package-level imports.

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
- `cell_energy()`: Computes energy for a single cell, supports both local and global vertex representations (in `energy.py`)
- `tissue_energy()`: Sums all cell energies, preferring global vertex pool when available (in `energy.py`)
- Energy gradient computed via finite differences (`finite_difference_cell_gradient()` in `simulation.py`)
- Analytical gradient also available (`cell_energy_gradient_analytic()` in `energy.py`)

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
9. ✅ Cytokinesis (cell division): Contracting vertices, active forces, topological splitting - in `cytokinesis.py`

**Future work:**
- Additional topological events: T1 transitions (edge swaps), apoptosis/extrusion
- Adaptive timestep based on gradient magnitude
- Per-edge heterogeneous tension (currently uniform γ)
- Cached geometry computations for performance

## 5. Current Implementation Status
**Fully Implemented:**
- `EnergyParameters`: All fields functional (`k_area`, `k_perimeter`, `gamma`, `target_area`) - in `energy.py`
- `cell_energy()`: Complete implementation with area elasticity, perimeter contractility, and line tension - in `energy.py`
- `tissue_energy()`: Sums cell energies, supports global vertex representation - in `energy.py`
- `merge_nearby_vertices()`: Vertex clustering with union-find algorithm - in `mesh_ops.py`
- `mesh_edges()`: Edge subdivision with configurable density - in `mesh_ops.py`
- `Simulation` class: Gradient descent with configurable dt, damping, and validation - in `simulation.py`
- `GeometryCalculator`: Area, perimeter, centroid calculations with polygon validation - in `geometry.py`
- Global vertex pool: `build_global_vertices()` and `reconstruct_cell_vertices()` fully operational - in `core.py`
- Tissue builders: `build_grid_tissue()`, `build_honeycomb_2_3_4_3_2()`, `build_honeycomb_3_4_5_4_3()` - in `builders.py`
- **Cytokinesis**: Complete cell division with contracting vertices, active forces, and topological splitting - in `cytokinesis.py`
- ACAM importer: `convert_acam_with_topology()` with neighbor topology awareness - in `acam_importer.py`
- Validation: `validate()` method, polygon validity checks, CCW ordering enforcement - in `core.py` and `geometry.py`

**Key Implementation Notes:**
- Energy formula: E = 0.5*k_area*(A-A0)² + 0.5*k_perimeter*P² + γ*P
- Gradient computation: Central finite differences (analytical gradient also available)
- Timestep requirements: ACAM tissues (~79 cells) need dt≈0.0001; Honeycomb (~14 cells) can use dt≈0.01
- Known validation requirement: ACAM tissues should be validated for duplicate consecutive vertices post-conversion

## 5.1 Cell Growth Simulation

The `examples/simulate_cell_growth.py` script provides a complete simulation framework for studying cell growth in vertex model tissues.

**Features:**
- **Single or multi-cell growth**: Grow one cell or multiple cells simultaneously using comma-separated IDs
- **Gradual target area increase**: Cells grow from initial area to 2× initial area over configurable steps
- **Global vertex coupling**: Shared vertices move consistently across all cells
- **Two solver options**: Gradient descent or overdamped force-balance (OFB)
- **Vertex merging**: Optional periodic merging of nearby vertices during simulation
- **Edge meshing**: Optional edge subdivision for finer mesh resolution

**Usage Examples:**
```bash
# Single cell growth (honeycomb)
python examples/simulate_cell_growth.py --growing-cell-ids 7 --plot

# Multiple cells (honeycomb)
python examples/simulate_cell_growth.py --growing-cell-ids 3,7,10 --plot

# Multiple cells (ACAM tissue)
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/acam_79cells.dill \
  --growing-cell-ids I,AW,AB,AA,V,BF,AV,BR,AL \
  --total-steps 100 --dt 0.00001 --plot --enable-merge
```

**Output:**
- Creates `Sim_<tissue>_<cells>_<timestamp>/` folder
- `growth_tracking.csv`: Per-step data (area, target, progress) for each growing cell
- `growth_initial.png` and `growth_final.png`: Tissue visualizations

**Key Command-Line Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--growing-cell-ids` | Comma-separated cell IDs to grow | `7` |
| `--tissue-file` | Path to tissue file (.dill) | builds honeycomb |
| `--total-steps` | Total simulation steps | `200` |
| `--growth-steps` | Steps to ramp target area | `100` |
| `--dt` | Time step size | `0.01` |
| `--solver` | `gradient_descent` or `overdamped_force_balance` | `gradient_descent` |
| `--enable-merge` | Enable vertex merging | disabled |
| `--plot` | Show/save tissue plots | disabled |

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

