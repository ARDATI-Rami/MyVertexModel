# Plan: Cytokinesis Cell Division in Vertex Model

Implement a cytokinesis process where a cell contracts along a specified division axis (simulating an actomyosin ring) via two "contracting vertices" inserted at the division line, then splits topologically when sufficiently constricted.

## Steps

### 1. Add division direction computation API in `geometry.py`

Create `compute_cell_major_axis(vertices)` using PCA (or inertia tensor) to return eigenvalues/eigenvectors, and `compute_division_direction(vertices, mode)` that returns a unit vector for `"parallel_cell_axis"`, `"perp_cell_axis"`, `"parallel_stress_axis"`, or `"perp_stress_axis"` modes. Add `compute_principal_stress_axis(cell, tissue)` as a placeholder that accepts a stress tensor or defers to a future implementation.

### 2. Add line-polygon intersection utility in `geometry.py`

Implement `line_polygon_intersection(centroid, direction, vertices)` → returns two intersection points (as `(t, edge_index, point)` tuples) where the line through `centroid` along `±direction` crosses the polygon boundary. This forms the basis for inserting contracting vertices.

### 3. Implement vertex insertion for division in `mesh_ops.py`

Create `prepare_division(tissue, cell_id, division_direction)` which:
- (a) calls `line_polygon_intersection` to find boundary crossings
- (b) inserts two new vertices into the global pool at those points
- (c) updates `cell.vertex_indices` to include the new vertices in correct polygon order
- (d) updates neighbor cells if edge is shared
- (e) returns a `DivisionContext` dataclass containing `{cell_id, contracting_vertex_indices, D0, ...}`

### 4. Implement contractile force function in new module `cytokinesis.py`

Create `ActinRingForce` callable class (compatible with `active_force_func` signature) that applies equal-and-opposite forces on two contracting vertices, aligned with `division_direction`. Support configurable force models:
- (a) constant tension
- (b) spring with target length decreasing linearly over time
- (c) Hookean spring with configurable stiffness

Store `DivisionContext` in `active_force_params`.

### 5. Add constriction monitor and topological split in `cytokinesis.py`

Implement `check_constriction_threshold(tissue, division_ctx, constriction_percentage)` → bool.

Implement `execute_topological_split(tissue, division_ctx)` which:
- (a) partitions cell vertices into two sides of the division line
- (b) creates two new `Cell` objects (A1, A2) with correct `vertex_indices`
- (c) sets the segment between contracting vertices as the shared edge between A1 and A2
- (d) removes original cell from `tissue.cells`
- (e) updates all neighbor adjacencies for cells that shared edges with the original

### 6. Integrate with simulation loop in `simulation.py`

Add optional `division_events: List[DivisionContext]` to `Simulation`. After each `step()`, call `check_constriction_threshold` for active divisions; when triggered, call `execute_topological_split` and remove from active list. Provide `Simulation.schedule_division(cell_id, division_direction, ...)` API to initiate the process.

## Further Considerations

### 1. Stress tensor computation

Compute the cell stress tensor from vertex forces using the virial stress formulation. For a 2D vertex model, the Cauchy stress tensor for a cell is:

```
σ_αβ = (1 / A) Σ_i (r_i)_α (F_i)_β
```

where:
- `A` is the cell area
- `r_i` is the position vector of vertex `i` relative to the cell centroid
- `F_i` is the total force on vertex `i` (mechanical + active)
- α, β ∈ {x, y}

The principal stress axis is the eigenvector corresponding to the largest eigenvalue of σ. Implementation:

```python
def compute_cell_stress_tensor(
    cell: Cell,
    tissue: Tissue,
    forces: np.ndarray,  # (N, 2) force array for cell vertices
) -> np.ndarray:
    """Compute 2x2 Cauchy stress tensor for a cell.
    
    Args:
        cell: Cell to compute stress for.
        tissue: Tissue containing the cell.
        forces: (N, 2) array of forces on each vertex.
        
    Returns:
        (2, 2) stress tensor.
    """
```

Add `compute_principal_stress_axis(stress_tensor)` → unit vector along max principal stress. This enables `"parallel_stress_axis"` and `"perp_stress_axis"` division modes.

### 2. Shared edge handling at split boundaries

After topological split, each daughter cell inherits neighbor relationships from the original cell based on which edges belong to which daughter. The handling is:

1. **During split**: For each edge of the original cell that was shared with a neighbor:
   - Determine which daughter (A1 or A2) now owns that edge
   - The neighbor's `vertex_indices` remain unchanged (they still reference the same global vertices)
   - The daughter inherits the adjacency; the other daughter does not share that edge with that neighbor

2. **The new shared edge**: The segment between the two contracting vertices becomes a shared edge between A1 and A2:
   - Both daughters include these two vertex indices in their `vertex_indices`
   - A1 references them in one order (e.g., `[v1, v2]`), A2 in reverse order (`[v2, v1]`)

3. **No explicit neighbor list update needed**: Since adjacency is computed dynamically via shared edges in `cell_neighbor_counts()`, daughters automatically become neighbors of the correct cells by virtue of sharing the same global vertex indices.

4. **Post-split validation**: Call `tissue.validate()` and verify `cell_neighbor_counts()` returns expected values (original neighbors now neighbor appropriate daughter(s), A1 and A2 are neighbors of each other).

### 3. Test scope

Unit tests should cover:
- (a) `line_polygon_intersection` for convex/concave polygons
- (b) vertex insertion preserves polygon validity + area
- (c) `execute_topological_split` yields two valid cells with correct combined area
- (d) no dangling vertices or invalid `vertex_indices` post-split

Integration test: full cytokinesis cycle on a honeycomb cell.

## Proposed API Signatures

### `geometry.py`

```python
def compute_cell_major_axis(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute major/minor axes of cell polygon via PCA.
    
    Args:
        vertices: (N, 2) array of polygon vertices.
        
    Returns:
        eigenvalues: (2,) array, sorted descending.
        eigenvectors: (2, 2) array, columns are unit eigenvectors (major, minor).
    """

def compute_cell_stress_tensor(
    cell: Cell,
    tissue: Tissue,
    forces: np.ndarray,
) -> np.ndarray:
    """Compute 2x2 Cauchy stress tensor for a cell using virial stress formulation.
    
    σ_αβ = (1 / A) Σ_i (r_i)_α (F_i)_β
    
    where r_i is vertex position relative to centroid, F_i is force on vertex i.
    
    Args:
        cell: Cell to compute stress for.
        tissue: Tissue containing the cell.
        forces: (N, 2) array of forces on each vertex.
        
    Returns:
        (2, 2) stress tensor.
    """

def compute_principal_stress_axis(stress_tensor: np.ndarray) -> np.ndarray:
    """Extract principal stress axis from stress tensor.
    
    Args:
        stress_tensor: (2, 2) stress tensor.
        
    Returns:
        (2,) unit vector along maximum principal stress direction.
    """

def compute_division_direction(
    vertices: np.ndarray,
    mode: Literal["parallel_cell_axis", "perp_cell_axis", "parallel_stress_axis", "perp_stress_axis"],
    stress_tensor: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute division direction unit vector.
    
    Args:
        vertices: (N, 2) array of polygon vertices.
        mode: Division direction mode.
        stress_tensor: Optional (2, 2) stress tensor for stress-based modes.
        
    Returns:
        (2,) unit vector indicating division direction.
        
    Raises:
        NotImplementedError: For stress-based modes if stress_tensor not provided.
    """

def line_polygon_intersection(
    centroid: np.ndarray,
    direction: np.ndarray,
    vertices: np.ndarray,
) -> List[Tuple[float, int, np.ndarray]]:
    """Find intersections of line through centroid with polygon boundary.
    
    Args:
        centroid: (2,) point on the line.
        direction: (2,) unit direction vector.
        vertices: (N, 2) polygon vertices in order.
        
    Returns:
        List of (t, edge_index, point) tuples where:
            t: parameter along line (centroid + t * direction)
            edge_index: index of edge (vertices[i] to vertices[(i+1) % N])
            point: (2,) intersection coordinates
        Typically returns exactly 2 intersections for convex polygons.
    """
```

### `mesh_ops.py`

```python
@dataclass
class DivisionContext:
    """Context for an ongoing cell division event."""
    cell_id: Union[int, str]
    division_direction: np.ndarray  # (2,) unit vector
    contracting_vertex_indices: Tuple[int, int]  # global vertex pool indices
    D0: float  # initial distance between contracting vertices
    start_time: float  # simulation time when division initiated
    constriction_percentage: float  # threshold for triggering split (0-100)


def prepare_division(
    tissue: Tissue,
    cell_id: Union[int, str],
    division_direction: np.ndarray,
    constriction_percentage: float = 20.0,
    start_time: float = 0.0,
) -> DivisionContext:
    """Prepare a cell for division by inserting contracting vertices.
    
    Args:
        tissue: Tissue with global vertex pool.
        cell_id: ID of cell to divide.
        division_direction: (2,) unit vector for division axis.
        constriction_percentage: Threshold (0-100) for triggering split.
        start_time: Current simulation time.
        
    Returns:
        DivisionContext with all information needed for force application and split.
        
    Raises:
        ValueError: If cell not found or division line doesn't intersect properly.
    """
```

### `cytokinesis.py` (new module)

```python
@dataclass
class ActinRingForceParams:
    """Parameters for actin ring contractile force."""
    force_model: Literal["constant_tension", "decreasing_spring", "hookean"] = "constant_tension"
    tension: float = 1.0  # for constant_tension
    stiffness: float = 1.0  # for hookean / decreasing_spring
    target_length_rate: float = 0.1  # for decreasing_spring: rate of target length decrease per time


class ActinRingForce:
    """Callable active force for cytokinesis contraction.
    
    Compatible with OverdampedForceBalanceParams.active_force_func signature.
    """
    
    def __init__(self, division_contexts: Dict[Union[int, str], DivisionContext]):
        """
        Args:
            division_contexts: Map from cell_id to active DivisionContext.
        """
    
    def __call__(
        self,
        cell: Cell,
        tissue: Tissue,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Compute actin ring forces for a cell's vertices.
        
        Args:
            cell: Cell being updated.
            tissue: Tissue containing cell.
            params: Must contain 'force_params' (ActinRingForceParams) and 'current_time'.
            
        Returns:
            (N, 2) force array; non-zero only at contracting vertices.
        """


def check_constriction_threshold(
    tissue: Tissue,
    division_ctx: DivisionContext,
) -> bool:
    """Check if contracting vertices have reached constriction threshold.
    
    Returns True when distance <= constriction_percentage / 100 * D0.
    """


def execute_topological_split(
    tissue: Tissue,
    division_ctx: DivisionContext,
) -> Tuple[Cell, Cell]:
    """Perform topological division of cell into two daughter cells.
    
    Args:
        tissue: Tissue containing the dividing cell.
        division_ctx: Context with division information.
        
    Returns:
        (daughter_A1, daughter_A2): The two new cells.
        
    Side effects:
        - Removes original cell from tissue.cells
        - Adds two daughter cells to tissue.cells
        - Updates tissue.vertices if needed
        - Updates neighbor cell vertex_indices for shared edges
    """
```

### `simulation.py` additions

```python
class Simulation:
    # ...existing attributes...
    division_events: List[DivisionContext]  # active division events
    
    def schedule_division(
        self,
        cell_id: Union[int, str],
        division_direction: Optional[np.ndarray] = None,
        direction_mode: Literal["parallel_cell_axis", "perp_cell_axis", ...] = "perp_cell_axis",
        constriction_percentage: float = 20.0,
        force_params: Optional[ActinRingForceParams] = None,
    ) -> DivisionContext:
        """Schedule a cell for division.
        
        If division_direction is None, computes it from direction_mode.
        Inserts contracting vertices and registers the division event.
        """
    
    def _check_and_execute_divisions(self):
        """Check all active divisions and execute splits when threshold reached.
        
        Called automatically after each step().
        """
```

## Data Flow for One Cytokinesis Event

1. **Initiation**: User calls `sim.schedule_division(cell_id="A", direction_mode="perp_cell_axis")`
   - Computes division direction via `compute_division_direction()`
   - Calls `prepare_division()` which inserts two new vertices at boundary intersections
   - Returns `DivisionContext` and adds to `sim.division_events`
   - Registers `ActinRingForce` with the simulation's active force system

2. **Each simulation step**:
   - `sim.step()` performs force-balance update
   - `ActinRingForce.__call__()` applies contractile forces to the two contracting vertices
   - Forces pull vertices toward each other along `division_direction`
   - After position update, `sim._check_and_execute_divisions()` is called

3. **Constriction check**:
   - For each active `DivisionContext`, compute current distance between contracting vertices
   - Compare to `constriction_percentage / 100 * D0`
   - If threshold reached, proceed to split

4. **Topological split**:
   - `execute_topological_split()` partitions vertices into two sides
   - Creates daughter cells A1, A2 with correct vertex ordering
   - Shared edge (between contracting vertices) becomes boundary between A1 and A2
   - Original cell removed from tissue
   - Neighbor cells updated for any shared edges they had with original
   - `DivisionContext` removed from `sim.division_events`

5. **Post-split validation**:
   - Call `tissue.validate()` to ensure structural integrity
   - Verify combined area of A1 + A2 ≈ original cell area
   - Verify no dangling vertices in global pool

## Invariants to Preserve

Before and after division, the following must hold:

1. **Polygon validity**: All cells have valid, non-self-intersecting polygons with ≥3 vertices
2. **Global pool consistency**: All `cell.vertex_indices` reference valid indices in `tissue.vertices`
3. **Shared edge consistency**: If two cells share an edge, both reference the same two vertex indices (in opposite order)
4. **No orphaned vertices**: Every vertex in `tissue.vertices` is referenced by at least one cell (or we accept orphans with a cleanup pass)
5. **Area conservation**: Sum of daughter areas equals original cell area (within numerical tolerance)
6. **CCW ordering**: All cell vertex orderings remain counter-clockwise
7. **Neighbor relationships**: Cells that shared edges with original now share edges with appropriate daughter(s)

## Minimal Test Plan

### Unit Tests

#### `test_geometry_division.py`

```python
def test_compute_cell_major_axis_square():
    """Square cell has equal eigenvalues, arbitrary axes."""

def test_compute_cell_major_axis_elongated():
    """Elongated cell has major axis along long dimension."""

def test_compute_division_direction_modes():
    """All mode strings return valid unit vectors."""

def test_line_polygon_intersection_convex():
    """Line through hexagon center yields exactly 2 intersections."""

def test_line_polygon_intersection_at_vertex():
    """Line passing through existing vertex handled correctly."""

def test_line_polygon_intersection_concave():
    """Concave polygon may yield >2 intersections; verify handling."""
```

#### `test_mesh_ops_division.py`

```python
def test_prepare_division_inserts_vertices():
    """Verify two new vertices added to global pool."""

def test_prepare_division_updates_cell_indices():
    """Cell vertex_indices includes new vertices in correct order."""

def test_prepare_division_updates_neighbor():
    """If division line crosses shared edge, neighbor is also updated."""

def test_prepare_division_preserves_area():
    """Cell area unchanged after vertex insertion."""

def test_prepare_division_D0_computed():
    """DivisionContext.D0 equals initial distance between contracting vertices."""
```

#### `test_cytokinesis.py`

```python
def test_actin_ring_force_zero_for_non_dividing():
    """Non-dividing cells receive zero active force."""

def test_actin_ring_force_applied_to_contracting_vertices():
    """Force only non-zero at contracting vertex indices."""

def test_actin_ring_force_directions_opposite():
    """Forces on two contracting vertices are equal and opposite."""

def test_check_constriction_threshold_false_initially():
    """At t=0, threshold not reached."""

def test_check_constriction_threshold_true_when_close():
    """Returns True when distance < constriction_percentage * D0."""

def test_execute_topological_split_creates_two_cells():
    """Original cell replaced by two daughter cells."""

def test_execute_topological_split_area_conserved():
    """Sum of daughter areas equals original."""

def test_execute_topological_split_shared_edge():
    """Daughters share the contracting edge."""

def test_execute_topological_split_neighbors_updated():
    """Neighbors of original now neighbor appropriate daughter."""

def test_execute_topological_split_polygon_validity():
    """Both daughters are valid polygons."""
```

### Integration Tests

```python
def test_full_cytokinesis_cycle_honeycomb():
    """Complete division of center cell in honeycomb tissue."""
    tissue = build_honeycomb_2_3_4_3_2()
    tissue.build_global_vertices()
    
    sim = Simulation(tissue, solver_type="overdamped_force_balance", ...)
    ctx = sim.schedule_division(cell_id=7, direction_mode="perp_cell_axis")
    
    # Run until division completes
    for _ in range(1000):
        sim.step()
        if len(sim.division_events) == 0:
            break
    
    # Verify division occurred
    assert len(tissue.cells) == 15  # was 14
    tissue.validate()
    
def test_multiple_simultaneous_divisions():
    """Two cells dividing at the same time."""
```

