# Topology-Aware ACAM Tissue Conversion

**Note:** Only the midpoint-adhesion pipeline in `scripts/tissue_acam_to_vertex.py` is supported for ACAM→vertex conversion. All previous neighbor-based and tangent-based methods are deprecated and have been removed from this guide.

## ⚠️ Important: Post-Conversion Validation Required

**CRITICAL:** ACAM tissue conversion can produce tissues with duplicate consecutive vertices that cause simulation crashes. **Always validate and repair converted tissues before use.**

**Quick validation workflow:**
```bash
# 1. Convert using the supported pipeline
python scripts/tissue_acam_to_vertex.py --acam-file acam_tissues/20_cells_adhesion --output pickled_tissues/vertex_20cells.dill

# 2. Validate (REQUIRED)
pytest tests/test_tissue_cell_by_cell.py -v

# 3. Repair if needed
python examples/diagnose_tissue.py pickled_tissues/vertex_20cells.dill --repair --save-repaired pickled_tissues/vertex_20cells_repaired.dill
```

See [Known Issues and Validation](#known-issues-and-validation) section for details.

## Overview

The supported ACAM→vertex conversion pipeline uses midpoint grouping, PCA-based sorting, endpoint clustering, and edge building. See `scripts/tissue_acam_to_vertex.py` for details and usage.

## Basic Usage

### Programmatic Usage

```python
# Example: Load and convert ACAM tissue using midpoint-adhesion pipeline
import dill
from pathlib import Path
from scripts.tissue_acam_to_vertex import convert_acam_to_vertex

# Load ACAM tissue
with open('acam_tissues/20_cells_adhesion', 'rb') as f:
    acam_tissue = dill.load(f)

# Convert using midpoint-adhesion pipeline
vertex_tissue = convert_acam_to_vertex(acam_tissue)
```

### Command-Line Usage

```bash
# Convert ACAM tissue to vertex model
python scripts/tissue_acam_to_vertex.py --acam-file acam_tissues/20_cells_adhesion --output pickled_tissues/vertex_20cells.dill
```

## Parameters

### Required Inputs
- **acam_file**: Path to ACAM pickle file containing cell geometry data
  - Example: `'acam_tissues/20_cells_adhesion'`

### Output
- **output**: Path to save the converted vertex model tissue
  - Example: `'pickled_tissues/vertex_20cells.dill'`

## Loading ACAM Data Without Source Code

To load ACAM pickled tissue files without requiring the original ACAM source code, use a StubUnpickler. This replaces missing classes with generic objects, allowing you to access basic data (attributes, arrays) even if the original class definitions are unavailable.

**Example:**
```python
import pickle

class StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return type(name, (), {})

with open('acam_tissues/20_cells_adhesion', 'rb') as file_in:
    acam_tissue = StubUnpickler(file_in).load()

# Now you can access acam_tissue.cells, acam_tissue.slow_adhesions, etc.
```

This approach is now used in `scripts/tissue_acam_to_vertex.py` and makes ACAM conversion independent of the original ACAM codebase.

## Known Issues and Validation

### ⚠️ CRITICAL: Duplicate Consecutive Vertices

**Problem:** The conversion algorithm can occasionally create duplicate consecutive vertices in `vertex_indices`, resulting in invalid polygons that cause simulation crashes.

**Solution:** Always validate and repair converted tissues before simulation:

```bash
# 1. Validate the converted tissue
pytest tests/test_tissue_cell_by_cell.py::test_tissue_cell_by_cell -v

# 2. If validation fails, repair the tissue
python examples/diagnose_tissue.py pickled_tissues/vertex_20cells.dill --repair --save-repaired pickled_tissues/vertex_20cells_repaired.dill

# 3. Re-validate the repaired tissue
pytest tests/test_tissue_cell_by_cell.py::test_tissue_cell_by_cell -v
```

## Recommended Workflow

```bash
# 1. Convert ACAM tissue
python scripts/tissue_acam_to_vertex.py --acam-file acam_tissues/20_cells_adhesion --output pickled_tissues/vertex_20cells.dill

# 2. Validate structure
pytest tests/test_tissue_cell_by_cell.py -v

# 3. Repair if needed
python examples/diagnose_tissue.py pickled_tissues/vertex_20cells.dill --repair --save-repaired pickled_tissues/vertex_20cells_repaired.dill

# 4. Run simulation with appropriate parameters
python examples/simulate_cell_growth.py --tissue-file pickled_tissues/vertex_20cells_repaired.dill --growing-cell-ids 1 --dt 0.00001 --total-steps 100 --plot --enable-merge
```

## Troubleshooting

### Duplicate Consecutive Vertices

**Symptom:** `Cell X has duplicate consecutive vertices in vertex_indices`

**Fix:**
```bash
python examples/diagnose_tissue.py pickled_tissues/vertex_20cells.dill --repair --save-repaired pickled_tissues/vertex_20cells_fixed.dill
```

## Simulation Parameters for ACAM Tissues

ACAM tissues require different simulation parameters than smaller tissues like honeycomb due to their size and geometry.

### Recommended Parameters

**For ACAM tissues (e.g., 20 cells):**
```python
dt = 0.0001              # Timestep (smaller than honeycomb)
damping = 1.0            # Damping factor
k_area = 1.0             # Area elasticity
k_perimeter = 0.1        # Perimeter elasticity
gamma = 0.05             # Line tension
epsilon = 1e-6           # Finite difference epsilon
```

## See Also
- `examples/diagnose_tissue.py` - Diagnostic and repair tool
- `tests/test_tissue_cell_by_cell.py` - Comprehensive validation test
- `scripts/tissue_acam_to_vertex.py` - Main ACAM→vertex conversion pipeline
