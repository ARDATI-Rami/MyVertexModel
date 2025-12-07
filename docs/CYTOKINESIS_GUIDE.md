# Cytokinesis (Cell Division) Guide

## Overview

The MyVertexModel package includes a complete implementation of cytokinesis (cell division) where a cell contracts along a specified division axis (simulating an actomyosin ring) via two "contracting vertices" inserted at the division line, then splits topologically when sufficiently constricted.

## Key Concepts

### Division Process

Cytokinesis in the vertex model proceeds in three stages:

1. **Initiation**: Two "contracting vertices" are inserted at the division line
2. **Constriction**: Active contractile forces pull the vertices together, simulating the actomyosin ring
3. **Splitting**: When sufficiently constricted, the cell topologically splits into two daughter cells

### Division Axis

The division axis can be:
- **Automatic**: Computed using Principal Component Analysis (PCA) to find the long axis
- **Manual**: Specified by providing an angle in radians

### Contracting Vertices

These special vertices represent the attachment points of the actomyosin contractile ring. They:
- Start separated by a fraction of the cell width (default: 95%)
- Experience active contractile forces pulling them together
- Serve as the division plane when splitting occurs

## API Reference

### CytokinesisParams

Configuration for the division process:

```python
from myvertexmodel import CytokinesisParams

params = CytokinesisParams(
    constriction_threshold=0.1,        # Distance threshold for splitting
    initial_separation_fraction=0.95,  # Initial vertex separation (0-1)
    contractile_force_magnitude=10.0   # Force magnitude
)
```

### High-Level API: perform_cytokinesis()

The simplest way to perform cell division:

```python
from myvertexmodel import perform_cytokinesis, CytokinesisParams

# Stage 1: Initiate division
result = perform_cytokinesis(cell, tissue, axis_angle=0.0)
# Returns: {'stage': 'initiated', 'contracting_vertices': (idx1, idx2)}

# Stage 2: Check constriction (during simulation)
result = perform_cytokinesis(cell, tissue)
# Returns: {'stage': 'constricting', 'constriction_distance': 0.5}

# Stage 3: Split when ready
result = perform_cytokinesis(cell, tissue, daughter1_id=2, daughter2_id=3)
# Returns: {'stage': 'completed', 'daughter_cells': (daughter1, daughter2)}
```

### Low-Level API

For more control, use individual functions:

```python
from myvertexmodel import (
    compute_division_axis,
    insert_contracting_vertices,
    compute_contractile_forces,
    check_constriction,
    split_cell,
    update_global_vertices_from_cells,
)

# 1. Compute division axis
centroid, axis_dir, perp_dir = compute_division_axis(cell, tissue)

# 2. Insert contracting vertices
v1_idx, v2_idx = insert_contracting_vertices(
    cell, tissue, 
    axis_angle=None,  # Use automatic axis
    params=params
)

# 3. During simulation, compute contractile forces
forces = compute_contractile_forces(cell, tissue, params)

# After each simulation step, update global vertices
# (preserves vertex indices, unlike build_global_vertices)
update_global_vertices_from_cells(tissue)

# 4. Check if ready to split
if check_constriction(cell, tissue, params):
    # 5. Split the cell
    daughter1, daughter2 = split_cell(cell, tissue)
```

## Integration with Simulation

Cytokinesis uses the **active forces** mechanism in the overdamped force-balance solver:

```python
from myvertexmodel import (
    Simulation,
    EnergyParameters,
    OverdampedForceBalanceParams,
    perform_cytokinesis,
    compute_contractile_forces,
)

# Define active force function
def active_force_func(cell, tissue, params):
    """Compute contractile forces for dividing cells."""
    cyto_params = params.get('cytokinesis_params')
    if hasattr(cell, 'cytokinesis_data'):
        return compute_contractile_forces(cell, tissue, cyto_params)
    return np.zeros_like(cell.vertices)

# Set up simulation
energy_params = EnergyParameters(k_area=1.0, k_perimeter=0.5, gamma=0.1)
ofb_params = OverdampedForceBalanceParams(
    gamma=1.0,
    active_force_func=active_force_func,
    active_force_params={'cytokinesis_params': cyto_params}
)

sim = Simulation(
    tissue=tissue,
    energy_params=energy_params,
    dt=0.005,
    solver_type='overdamped_force_balance',
    ofb_params=ofb_params
)

# Run simulation with contractile forces
sim.run(n_steps=1)

# IMPORTANT: Update global vertices to preserve indices
from myvertexmodel import update_global_vertices_from_cells
update_global_vertices_from_cells(tissue)
```

**Note**: When using cytokinesis during simulation, you must call `update_global_vertices_from_cells(tissue)` after each simulation step to update the global vertex positions while preserving the vertex indices. Do NOT use `tissue.build_global_vertices()` as it will rebuild indices and break the tracking of contracting vertices.

## Complete Example

See `examples/demonstrate_cytokinesis.py` for a full working example.

```python
import numpy as np
from myvertexmodel import (
    Cell, Tissue, perform_cytokinesis, 
    CytokinesisParams, check_constriction
)

# Create a cell
vertices = np.array([[0, 0], [2, 0], [2, 1.5], [0, 1.5]], dtype=float)
cell = Cell(cell_id=1, vertices=vertices)
tissue = Tissue()
tissue.add_cell(cell)
tissue.build_global_vertices()

# Set up cytokinesis
params = CytokinesisParams(
    constriction_threshold=0.5,  # Adjust based on cell size and forces
    contractile_force_magnitude=80.0
)

# Initiate division (horizontal axis)
result = perform_cytokinesis(cell, tissue, axis_angle=np.pi/2, params=params)
print(f"Stage: {result['stage']}")

# ... run simulation with contractile forces ...

# Check and split when ready
if check_constriction(cell, tissue, params):
    result = perform_cytokinesis(cell, tissue, params=params)
    if result['stage'] == 'completed':
        daughter1, daughter2 = result['daughter_cells']
        print(f"Division complete! {len(tissue.cells)} cells now.")
```

## Algorithm Details

### 1. Division Axis Computation

If no angle is specified, the division axis is computed using PCA:

1. Center vertices at cell centroid
2. Compute covariance matrix
3. Extract eigenvectors (principal components)
4. Long axis = first eigenvector (largest eigenvalue)
5. Division line is perpendicular to long axis

### 2. Contracting Vertex Insertion

1. Compute division line (perpendicular to axis)
2. Find two intersection points with cell boundary
3. Place vertices at `initial_separation_fraction` of full width
4. Insert into cell's vertex_indices at appropriate positions
5. Store metadata in `cell.cytokinesis_data`

### 3. Contractile Forces

For a cell with contracting vertices at positions `p1` and `p2`:

```
direction = (p2 - p1) / ||p2 - p1||
force1 = magnitude * direction      (pull toward p2)
force2 = -magnitude * direction     (pull toward p1)
```

All other vertices experience zero active force.

### 4. Cell Splitting

When `||p2 - p1|| < constriction_threshold`:

1. Find positions of contracting vertices in vertex_indices
2. Split indices into two groups (one on each side of division line)
3. Create two daughter cells with appropriate vertex indices
4. Both daughters share the contracting vertices
5. Remove parent cell, add daughters to tissue

## Visualization

The process can be visualized using `plot_tissue()`:

```python
from myvertexmodel import plot_tissue
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Before division
plot_tissue(tissue, ax=axes[0], show_cell_ids=True)
axes[0].set_title('Before Division')

# After inserting contracting vertices
perform_cytokinesis(cell, tissue)
plot_tissue(tissue, ax=axes[1], show_cell_ids=True, show_vertex_ids=True)
axes[1].set_title('Contracting')

# After split
# ... (after simulation and splitting)
plot_tissue(tissue, ax=axes[2], show_cell_ids=True)
axes[2].set_title('After Division')

plt.savefig('division_process.png')
```

## Tips and Best Practices

### Parameter Selection

- **constriction_threshold**: Typically 5-15% of cell size. Too small may cause numerical issues.
- **initial_separation_fraction**: 0.9-0.95 works well. Too low causes immediate splitting.
- **contractile_force_magnitude**: Balance with energy parameters. Start with 10-100.

### Timestep Considerations

- Cytokinesis may require smaller timesteps than regular simulation
- Use `dt=0.001-0.01` for stable constriction
- Monitor for oscillations; reduce `dt` if unstable

### Energy Parameters

For realistic division:
- Set `target_area` to original cell area for daughters
- Moderate `k_area` (1.0-2.0) for flexibility
- Moderate `k_perimeter` (0.1-0.5) to allow deformation
- Low `gamma` (0.1-0.2) for surface tension

### Multi-Cell Tissues

When dividing cells in a tissue with neighbors:
- Ensure global vertex pool is built: `tissue.build_global_vertices()`
- Sync after simulation: `tissue.build_global_vertices()` and `tissue.reconstruct_cell_vertices()`
- Vertices may be shared with neighbors; they will move accordingly

## Limitations and Future Work

### Current Limitations

1. **Division axis**: Currently 2D only; no out-of-plane divisions
2. **Force model**: Simple linear spring model for actomyosin ring
3. **Daughter cell properties**: No automatic inheritance of cell-specific parameters
4. **Synchronization**: Manual sync needed between local and global vertices during simulation

### Planned Enhancements

1. **Automated simulation loop**: Built-in function to run division to completion
2. **Property inheritance**: Automatic transfer of cell properties to daughters
3. **Adaptive parameters**: Automatic parameter adjustment based on cell size
4. **3D support**: Extension to 3D vertex models
5. **Oriented division**: Division based on mechanical cues or polarity

## References

- Farhadifar et al. (2007). *The influence of cell mechanics on epithelial morphogenesis*. Current Biology.
- Nagai & Honda (2009). *A dynamic cell model for the formation of epithelial tissues*. Physica D.
- Staddon et al. (2018). *Pulsatile contractions and pattern formation in excitable actomyosin cortex*. PLoS Computational Biology.

## Troubleshooting

### Division not completing

**Symptom**: Contracting vertices oscillate, never reach threshold

**Solutions**:
- Reduce timestep (`dt`)
- Increase contractile force magnitude
- Reduce energy parameter `k_perimeter`
- Increase constriction threshold

### Cell deformation issues

**Symptom**: Cell becomes highly deformed or invalid during constriction

**Solutions**:
- Reduce contractile force magnitude
- Increase `k_area` to resist area changes
- Use smaller timesteps
- Check that initial cell is valid (CCW orientation, no self-intersections)

### Splitting creates degenerate cells

**Symptom**: One daughter has < 3 vertices

**Solutions**:
- Ensure cell has enough vertices before division (≥6 recommended)
- Check division axis is perpendicular to long axis
- Verify initial cell geometry is reasonable (not too elongated or irregular)

### Force synchronization issues

**Symptom**: Forces don't affect global vertex positions

**Solutions**:
- Use overdamped force-balance solver (not gradient descent)
- Ensure active_force_func is properly configured
- Call `tissue.build_global_vertices()` periodically
- Use `tissue.reconstruct_cell_vertices()` after updating global vertices
