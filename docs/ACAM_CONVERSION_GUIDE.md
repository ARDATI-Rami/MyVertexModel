# Topology-Aware ACAM Tissue Conversion

This document describes how to use the new `convert_acam_with_topology()` function to convert ACAM tissues to vertex model format with neighbor topology awareness.

## ⚠️ Important: Post-Conversion Validation Required

**CRITICAL:** ACAM tissue conversion can produce tissues with duplicate consecutive vertices that cause simulation crashes. **Always validate and repair converted tissues before use.**

**Quick validation workflow:**
```bash
# 1. Convert
python scripts/convert_acam_tissue.py --acam-file acam_tissues/80_cells \
  --neighbor-json acam_tissues/acam_79_neighbors.json --output-prefix my_tissue

# 2. Validate (REQUIRED)
pytest tests/test_tissue_cell_by_cell.py -v

# 3. Repair if needed
python examples/diagnose_tissue.py pickled_tissues/my_tissue.dill \
  --repair --save-repaired pickled_tissues/my_tissue_repaired.dill
```

See [Known Issues and Validation](#known-issues-and-validation) section for details.

## Overview

The `convert_acam_with_topology()` function is an advanced ACAM converter that uses neighbor topology information to create more accurate vertex models. Unlike the simple converters that use fixed vertex counts, this function:

- Uses each cell's actual neighbor count to determine vertex count
- Handles boundary cells specially (cells touching the tissue boundary)
- Validates connectivity between neighboring cells
- Provides detailed conversion statistics and validation reports

**Note:** Converted tissues should always be validated for duplicate consecutive vertices and repaired if necessary before running simulations.

## Basic Usage

### Programmatic Usage

```python
from myvertexmodel import convert_acam_with_topology

# Convert ACAM tissue with topology information
result = convert_acam_with_topology(
    acam_file='acam_tissues/80_cells',
    neighbor_json='acam_tissues/acam_79_neighbors.json',
    merge_radius=14.0,
    max_vertices=10,
    validate_connectivity=True,
    verbose=True
)

# Access the converted tissue
tissue = result.tissue

# Access conversion statistics
summary = result.summary
print(f"Converted {summary['total_cells']} cells")
print(f"Global vertices: {tissue.vertices.shape[0]}")

# Access validation results (if validate_connectivity=True)
if result.validation:
    connectivity = result.validation['total_connectivity']['percentage']
    print(f"Connectivity: {connectivity:.1f}%")
```

### Command-Line Usage

```bash
# Basic conversion
python scripts/convert_acam_tissue.py \
    --acam-file acam_tissues/80_cells \
    --neighbor-json acam_tissues/acam_79_neighbors.json \
    --merge-radius 14.0 \
    --output-prefix acam_79cells

# With connectivity validation
python scripts/convert_acam_tissue.py \
    --acam-file acam_tissues/80_cells \
    --neighbor-json acam_tissues/acam_79_neighbors.json \
    --merge-radius 14.0 \
    --output-prefix acam_79cells \
    --validate-connectivity

# Quiet mode (no progress output)
python scripts/convert_acam_tissue.py \
    --acam-file acam_tissues/80_cells \
    --neighbor-json acam_tissues/acam_79_neighbors.json \
    --output-prefix acam_79cells \
    --quiet
```

## Parameters

### Required Inputs

- **acam_file**: Path to ACAM pickle file containing cell geometry data
  - Default: `'acam_tissues/80_cells'`
  
- **neighbor_json**: Path to JSON file with neighbor topology information
  - Default: `'acam_tissues/acam_79_neighbors.json'`
  - Format: `{"cell_id": ["neighbor1", "neighbor2", ...], ...}`

### Conversion Parameters

- **merge_radius**: Radius for fusing junction vertices across cells (float)
  - Default: `14.0`
  - Higher values create more vertex sharing
  - Lower values preserve more independent vertices
  
- **max_vertices**: Maximum vertices per cell (safety cap) (int)
  - Default: `10`
  - Prevents cells from having too many vertices

### Optional Features

- **validate_connectivity**: Enable connectivity validation (bool)
  - Default: `False`
  - Checks if ACAM neighbors share vertices
  - Categorizes connections as edge/corner/disconnected
  
- **save_summary**: Path to save summary JSON file (str or Path)
  - Default: `None` (no file saved)
  - Contains cell statistics and vertex information
  
- **save_validation**: Path to save validation report text file (str or Path)
  - Default: `None` (no file saved)
  - Detailed connectivity analysis
  
- **verbose**: Enable progress output (bool)
  - Default: `False`
  - Prints detailed conversion progress

## Return Value

The function returns a `ConversionResult` named tuple with:

- **tissue**: The converted `Tissue` object with global vertex pool
- **summary**: Dictionary with conversion statistics
- **validation**: Dictionary with connectivity validation results (if enabled)

### Summary Dictionary Structure

```python
{
    'merge_radius': 14.0,
    'max_vertices': 10,
    'total_cells': 79,
    'boundary_cells': 28,
    'interior_cells': 51,
    'cells': [
        {
            'name': 'A',
            'acam_id': 1,
            'is_boundary': False,
            'fe_point_count': 150,
            'hull_vertex_count': 45,
            'simplified_vertex_count': 6,
            'acam_neighbors': 6,
            'acam_neighbor_ids': ['B', 'C', 'D', 'E', 'F', 'G'],
            'target_vertices': 6,
            'vertex_indices': [0, 1, 2, 3, 4, 5]
        },
        # ... more cells
    ]
}
```

### Validation Dictionary Structure

```python
{
    'total_neighbor_pairs': 237,
    'edge_connected': {
        'count': 230,
        'percentage': 97.0,
        'pairs': [('A', 'B', 2), ('A', 'C', 3), ...]
    },
    'corner_connected': {
        'count': 5,
        'percentage': 2.1,
        'pairs': [('X', 'Y', 1), ...]
    },
    'disconnected': {
        'count': 2,
        'percentage': 0.8,
        'pairs': [('M', 'N'), ('P', 'Q')]
    },
    'total_connectivity': {
        'count': 235,
        'total': 237,
        'percentage': 99.2
    }
}
```

## Output Files

When using the CLI script, the following files are created:

1. **pickled_tissues/{prefix}.dill**
   - The converted tissue ready for simulations
   - Can be loaded with `load_tissue()`

2. **{prefix}_summary.json**
   - Detailed conversion statistics
   - Cell-by-cell information

3. **{prefix}_validation.txt** (if `--validate-connectivity` is used)
   - Human-readable connectivity report
   - Lists disconnected pairs if any

## Examples

See `examples/convert_acam_example.py` for a complete working example.

## Migration from fix_acam_interactive_79cells.py

The old script has been archived as `fix_acam_interactive_79cells.py.old`. To migrate:

**Old:**
```bash
python fix_acam_interactive_79cells.py \
    --acam-file acam_tissues/80_cells \
    --merge-radius 14 \
    --output-prefix acam_79cells \
    --validate-connectivity
```

**New:**
```bash
python scripts/convert_acam_tissue.py \
    --acam-file acam_tissues/80_cells \
    --neighbor-json acam_tissues/acam_79_neighbors.json \
    --merge-radius 14.0 \
    --output-prefix acam_79cells \
    --validate-connectivity
```

The key difference is that the neighbor topology is now explicitly specified via `--neighbor-json` parameter rather than being hardcoded in the script.

## Advanced Usage

### Custom Neighbor Topology

You can create custom neighbor topology JSON files for different tissues:

```json
{
  "cell1": ["cell2", "cell3", "bnd"],
  "cell2": ["cell1", "cell3", "cell4"],
  "cell3": ["cell1", "cell2", "cell4", "bnd"]
}
```

Where:
- Keys are cell identifiers (matching ACAM cell identifiers)
- Values are lists of neighboring cell identifiers
- `"bnd"` indicates boundary (edge of tissue)

### Saving Output Programmatically

```python
from pathlib import Path
from myvertexmodel import convert_acam_with_topology, save_tissue
import json

result = convert_acam_with_topology(
    acam_file='acam_tissues/80_cells',
    neighbor_json='acam_tissues/acam_79_neighbors.json',
    merge_radius=14.0
)

# Save tissue
output_dir = Path('pickled_tissues')
output_dir.mkdir(exist_ok=True)
save_tissue(result.tissue, str(output_dir / 'my_tissue.dill'))

# Save summary
with open('my_summary.json', 'w') as f:
    json.dump(result.summary, f, indent=2)
```

## Known Issues and Validation

### ⚠️ CRITICAL: Duplicate Consecutive Vertices

**Problem:** The vertex fusion algorithm can occasionally create duplicate consecutive vertices in `vertex_indices`, resulting in invalid polygons that cause simulation crashes.

**Symptoms:**
- Simulation energy explodes exponentially
- Errors about self-intersecting polygons
- Invalid polygon validation failures

**Example of the issue:**
```python
# Cell 24 might have:
cell.vertex_indices = [51, 67, 66, 66, 52]  # ❌ vertex 66 appears twice consecutively
```

This creates a zero-length edge, making the polygon self-intersecting.

**Solution:** Always validate and repair converted tissues before simulation:

```bash
# 1. Validate the converted tissue
pytest tests/test_tissue_cell_by_cell.py::test_tissue_cell_by_cell -v

# 2. If validation fails, repair the tissue
python examples/diagnose_tissue.py pickled_tissues/YOUR_TISSUE.dill \
  --repair --save-repaired pickled_tissues/YOUR_TISSUE_repaired.dill

# 3. Re-validate the repaired tissue
pytest tests/test_tissue_cell_by_cell.py::test_tissue_cell_by_cell -v
```

**Programmatic validation:**
```python
from myvertexmodel import load_tissue

tissue = load_tissue('pickled_tissues/acam_79cells.dill')

# Check for duplicate consecutive vertices
for cell in tissue.cells:
    indices = cell.vertex_indices
    for i in range(len(indices) - 1):
        if indices[i] == indices[i+1]:
            print(f"⚠ Cell {cell.id} has duplicate at position {i}: {indices}")
```

**See Also:**
- `ACAM_DIAGNOSTIC_SUMMARY.md` - Detailed analysis of duplicate vertex issue
- `tests/test_tissue_cell_by_cell.py` - Comprehensive validation test
- `examples/diagnose_tissue.py` - Diagnostic and repair tool

### Post-Conversion Validation Checklist

After converting an ACAM tissue, **always** perform these checks:

1. **Structure validation:**
   ```bash
   pytest tests/test_tissue_cell_by_cell.py -v -s
   ```

2. **Visual inspection:**
   ```bash
   python examples/plot_acam_79cells_labeled.py
   # Check for overlapping vertices or irregular cells
   ```

3. **Simulation parameters test:**
   ```bash
   python examples/diagnose_acam_simulation.py
   # Identifies appropriate dt and damping values
   ```

### Recommended Workflow

```bash
# 1. Convert ACAM tissue
python scripts/convert_acam_tissue.py \
  --acam-file acam_tissues/80_cells \
  --neighbor-json acam_tissues/acam_79_neighbors.json \
  --output-prefix acam_79cells \
  --validate-connectivity

# 2. Validate structure
pytest tests/test_tissue_cell_by_cell.py -v

# 3. Repair if needed
python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill \
  --repair --save-repaired pickled_tissues/acam_79cells_repaired.dill

# 4. Test simulation parameters
python examples/diagnose_acam_simulation.py

# 5. Run simulation with appropriate parameters
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/acam_79cells_repaired.dill \
  --dt 0.0001 \
  --total-steps 100 \
  --plot
```

## Troubleshooting

### Duplicate Consecutive Vertices

**Symptom:** `Cell X has duplicate consecutive vertices in vertex_indices`

**Fix:**
```bash
python examples/diagnose_tissue.py pickled_tissues/YOUR_TISSUE.dill \
  --repair --save-repaired pickled_tissues/YOUR_TISSUE_fixed.dill
```

The repair tool automatically removes duplicate consecutive vertices while preserving polygon topology.

### Low Connectivity

If validation shows many disconnected pairs:
- Increase `merge_radius` to create more vertex sharing
- Check that neighbor topology JSON is correct
- Verify ACAM file contains the expected cells

### Too Many/Few Vertices

If cells have unexpected vertex counts:
- Adjust `max_vertices` parameter
- Check neighbor topology (each cell's vertex count ≈ neighbor count)
- For boundary cells, an extra vertex is added automatically

### Simulation Energy Explosion

**Symptom:** Energy increases exponentially during simulation

**Causes:**
1. **Duplicate consecutive vertices** (see above) - Fix with repair tool
2. **Timestep too large** - ACAM tissues need dt=0.0001 (not 0.01)
3. **Invalid polygons** - Run validation and repair

**Fix:**
```bash
# Check for structural issues
python examples/diagnose_tissue.py pickled_tissues/YOUR_TISSUE.dill

# Test simulation parameters
python examples/diagnose_acam_simulation.py

# Use smaller timestep for ACAM tissues
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/YOUR_TISSUE.dill \
  --dt 0.0001 \
  --total-steps 100
```

### Import Errors

If you get import errors:
```python
# Make sure the package is installed or in your path
import sys
sys.path.insert(0, 'path/to/MyVertexModel/src')
from myvertexmodel import convert_acam_with_topology
```

## Simulation Parameters for ACAM Tissues

ACAM tissues require different simulation parameters than smaller tissues like honeycomb due to their size and geometry.

### Key Differences from Honeycomb

| Property | Honeycomb | ACAM | Impact |
|----------|-----------|------|--------|
| Cells | 14 | 79 | 5.6× more cells |
| Mean area | ~2.6 | ~3,738 | 1,438× larger |
| Gradient magnitude | ~10 | ~279 | 28× higher forces |

### Recommended Parameters

**For ACAM tissues (79 cells):**
```python
dt = 0.0001              # Timestep (100× smaller than honeycomb)
damping = 1.0            # Damping factor
k_area = 1.0             # Area elasticity
k_perimeter = 0.1        # Perimeter elasticity
gamma = 0.05             # Line tension
epsilon = 1e-6           # Finite difference epsilon
```

**For honeycomb tissues (14 cells):**
```python
dt = 0.01                # Can use larger timestep
damping = 1.0            # Same damping
# ... same energy parameters
```

### Why Smaller Timestep is Required

With gradient magnitude ~279 and dt=0.01:
- Vertex displacement per step: 279 × 0.01 = **2.79 units** (too large!)
- Results in energy explosion

With gradient magnitude ~279 and dt=0.0001:
- Vertex displacement per step: 279 × 0.0001 = **0.028 units** (stable!)
- Energy increases gradually during growth

### Example: Stable ACAM Simulation

```bash
# Test simulation parameters first
python examples/diagnose_acam_simulation.py

# Run with recommended parameters
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/acam_79cells_repaired.dill \
  --growing-cell-id 1 \
  --total-steps 100 \
  --dt 0.0001 \
  --damping 1.0 \
  --log-interval 10 \
  --plot \
  --save-csv acam_growth.csv
```

**Expected results:**
- Energy: ~215,725 → ~223,359 (stable, only +3.5%)
- Cell grows smoothly to target area
- No numerical instabilities

### Adaptive Timestep (Future Enhancement)

For robust simulations, consider implementing adaptive timestep:
```python
max_grad = np.max(np.abs(gradient))
max_displacement = max_grad * dt * damping

if max_displacement > threshold:
    dt_adaptive = threshold / (max_grad * damping)
```

## See Also

- `load_acam_tissue()` - Simple ACAM loader (regular polygons)
- `load_acam_from_json()` - Load from pre-extracted JSON data
- `examples/convert_acam_example.py` - Complete working example
- `examples/diagnose_acam_simulation.py` - Parameter testing tool
- `ACAM_SIMULATION_SUMMARY.md` - Detailed simulation guide

