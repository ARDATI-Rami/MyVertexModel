# MyVertexModel

A 2D epithelial vertex model implementation for simulating tissue mechanics. It represents a confluent tissue as polygons (cells) whose vertices are shared in a global pool, enabling physically consistent multi-cell interactions. The implementation includes:

- **Complete energy functional** with area elasticity, perimeter contractility, and line (junction) tension terms
- **Gradient descent simulation** with global vertex coupling for mechanical equilibration
- **Global vertex pool** with automatic merging and reconstruction capabilities
- **ACAM tissue import** with topology-aware conversion from center-based models
- **Cell growth simulation** with configurable parameters and organized output folders
- **Comprehensive validation tools** for tissue structure integrity
- **Enhanced visualization** with cell and vertex ID labeling
- **Tissue builders** for honeycomb and grid patterns
- Extensive test suite and documentation

## Features

### Core Implementation
- **Cell & Tissue data structures** with dual representation (local vertices + global vertex pool)
- **Global vertex pool** with `build_global_vertices()` and `reconstruct_cell_vertices()` methods
- **Energy functional**: E = ½k_area(A-A₀)² + ½k_perimeter·P² + γ·P
- **Gradient computation**: Central finite differences + analytical gradient implementation
- **Simulation engine**: Configurable timestep, damping, and epsilon parameters
- **Validation suite**: Polygon validity, CCW ordering, duplicate vertex detection

### Tissue Building & Import
- **Honeycomb builders**: 14-cell (2-3-4-3-2) and 19-cell (3-4-5-4-3) patterns
- **Grid builder**: Rectangular tissue lattices
- **ACAM importer**: Topology-aware conversion with neighbor connectivity (`convert_acam_with_topology()`)
- **Automatic global vertex pool** construction with tolerance-based merging

### Simulation & Analysis
- **Cell growth simulation** (`examples/simulate_cell_growth.py`):
  - Single-cell growth with gradual target area increase
  - Global vertex coupling for mechanical consistency
  - Organized output in unique `Sim_*` folders per run
  - CSV tracking and PNG visualization
- **Parameter diagnostic tools** for stability testing
- **Tissue comparison** and validation utilities

### Visualization
- **Enhanced plotting** with `plot_tissue()`:
  - Optional cell ID labels (yellow)
  - Optional vertex ID labels (red, showing global indices)
  - Customizable colors, transparency, and styling
- **Diagnostic visualizations** for connectivity and structure analysis

### Validation & Diagnostics
- **Cell-by-cell validation** (`tests/test_tissue_cell_by_cell.py`):
  - Tests all tissues for structural integrity
  - Detects duplicate consecutive vertices
  - Validates CCW ordering, polygon validity, vertex sharing
- **Tissue diagnostic tool** (`examples/diagnose_tissue.py`):
  - Structure analysis with detailed metrics
  - Tissue comparison capability
  - **Automated repair** for duplicate vertices and CCW violations
  - Visualization of connectivity issues

## Quick Start

### Cell Growth Simulation

Simulate a single cell growing in a honeycomb tissue:

```bash
# Honeycomb tissue (14 cells), cell 7 grows, creates output folder
python examples/simulate_cell_growth.py \
  --build-honeycomb 14 \
  --growing-cell-id 7 \
  --total-steps 100 \
  --plot
```

Simulate with ACAM tissue (requires smaller timestep):

```bash
# ACAM tissue (~79 cells), use dt=0.0001 for stability
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/acam_79cells_repaired.dill \
  --growing-cell-id 1 \
  --total-steps 100 \
  --dt 0.0001 \
  --plot
```

**Output**: Creates a unique `Sim_<tissue>_cell<id>_<timestamp>_<random>/` folder containing:
- `growth_tracking.csv` - Area, energy, and progress data
- `growth_initial.png` - Initial tissue state
- `growth_final.png` - Final tissue state

### ACAM Tissue Conversion

Convert ACAM center-based model to vertex model:

```bash
# Convert with topology awareness and validation
python scripts/convert_acam_tissue.py \
  --acam-file acam_tissues/80_cells \
  --neighbor-json acam_tissues/acam_79_neighbors.json \
  --merge-radius 14.0 \
  --output-prefix acam_79cells \
  --validate-connectivity
```

**Important**: Always validate and repair ACAM tissues after conversion:

```bash
# Validate structure
pytest tests/test_tissue_cell_by_cell.py -v

# Repair if needed (removes duplicate consecutive vertices)
python examples/diagnose_tissue.py pickled_tissues/acam_79cells.dill \
  --repair --save-repaired pickled_tissues/acam_79cells_repaired.dill
```

### Tissue Validation

Validate any tissue structure:

```bash
# Run comprehensive validation
pytest tests/test_tissue_cell_by_cell.py::test_tissue_cell_by_cell -v -s

# Diagnose specific tissue
python examples/diagnose_tissue.py pickled_tissues/YOUR_TISSUE.dill

# Compare with reference tissue
python examples/diagnose_tissue.py pickled_tissues/YOUR_TISSUE.dill \
  --reference pickled_tissues/honeycomb_14cells.dill
```

### Visualization

Plot tissue with cell and vertex IDs:

```python
from myvertexmodel import load_tissue, plot_tissue
import matplotlib.pyplot as plt

tissue = load_tissue('pickled_tissues/honeycomb_14cells.dill')

# Plot with labels
plot_tissue(tissue, show_cell_ids=True, show_vertex_ids=True)
plt.savefig('tissue_labeled.png', dpi=150)
plt.show()
```

## Installation

Using conda with the provided environment specification:

```bash
# Create environment
conda env create -f environment.yml
# Activate
conda activate myvertexmodel
# Editable install (pyproject.toml present)
pip install -e .
```

If you update dependencies, regenerate the environment or adjust `environment.yml` accordingly.

## Running Tests

The project has comprehensive test coverage:

```bash
# Run all tests
python -m pytest -v

# Run specific test modules
python -m pytest tests/test_basic.py -v              # Core functionality
python -m pytest tests/test_tissue_structure.py -v   # Structure validation
python -m pytest tests/test_tissue_cell_by_cell.py -v # Cell-by-cell validation

# Quiet mode
python -m pytest -q

# With output (shows print statements)
python -m pytest -v -s
```

### Key Test Modules

- **`tests/test_basic.py`**: Core geometry, energy, and simulation tests
- **`tests/test_tissue_structure.py`**: Tissue structure validation and comparison
- **`tests/test_tissue_cell_by_cell.py`**: Comprehensive cell-by-cell validation for all tissues
  - Validates all pickled tissues
  - Checks for duplicate consecutive vertices
  - Validates CCW ordering and polygon validity
  - Reports structural issues and connectivity

## CLI Usage

### Basic Demo Simulation

Run a small simulation (2×2 grid, 10 steps) and show a plot:

```bash
python -m myvertexmodel --steps 10 --grid-size 2 --plot
```

Customize parameters:

```bash
python -m myvertexmodel \
  --steps 20 \
  --grid-size 3 \
  --dt 0.02 \
  --k-area 2.0 \
  --k-perimeter 0.25 \
  --gamma 0.1 \
  --target-area 0.8 \
  --epsilon 1e-5 \
  --damping 0.5 \
  --plot --output tissue_custom.png
```

Suppress energy printing:

```bash
python -m myvertexmodel --no-energy-print
```

### Cell Growth Simulation (Recommended)

For realistic simulations, use the dedicated growth script:

```bash
# Honeycomb tissue
python examples/simulate_cell_growth.py \
  --build-honeycomb 14 \
  --growing-cell-id 7 \
  --total-steps 100 \
  --dt 0.01 \
  --plot

# ACAM tissue (use smaller timestep)
python examples/simulate_cell_growth.py \
  --tissue-file pickled_tissues/acam_79cells_repaired.dill \
  --growing-cell-id 1 \
  --total-steps 100 \
  --dt 0.0001 \
  --plot
```

See [Quick Start](#quick-start) for more examples.

## Plotting Example Script

A minimal example is provided in `examples/plot_example.py`:

```bash
python examples/plot_example.py
```

This script constructs a simple tissue and renders it with matplotlib.

## Design & Documentation

### Design Document

See `docs/design_vertex_model.md` for technical details:

- **Implemented features**: Global vertex pool, energy functional, simulation engine
- **Data structures**: Dual representation (local + global vertices), Cell.vertex_indices
- **Energy formula**: E = ½k_area(A-A₀)² + ½k_perimeter·P² + γ·P
- **Gradient computation**: Finite differences + analytical implementation
- **Future extensions**: Topological transitions (T1, division), adaptive timestep

### Additional Documentation

- **`docs/ACAM_CONVERSION_GUIDE.md`**: Complete guide for ACAM tissue conversion
  - Topology-aware conversion workflow
  - Known issues and validation requirements
  - Simulation parameters for ACAM tissues
  - Troubleshooting duplicate consecutive vertices

- **`ACAM_DIAGNOSTIC_SUMMARY.md`**: Analysis of ACAM tissue issues and solutions
- **`ACAM_SIMULATION_SUMMARY.md`**: Simulation parameters and stability guide
- **`TISSUE_VALIDATION_RESULTS.md`**: Validation tool documentation
- **`VISUALIZATION_COMPLETE.md`**: Enhanced plotting features guide

### Key References

- Farhadifar et al. 2007. *The influence of cell mechanics on epithelial morphogenesis*. DOI:10.1016/j.cub.2007.11.049
- Nagai & Honda 2009. *A dynamic cell model for the formation of epithelial tissues*. DOI:10.1016/j.physd.2009.01.001

## Project Structure

```
environment.yml                    # Conda environment specification
pyproject.toml                     # Package metadata / editable install
README.md                          # This file
.gitignore                         # Git ignore patterns (includes Sim_*/)

src/myvertexmodel/                 # Core package modules
├── __init__.py                    # Package exports
├── core.py                        # Cell, Tissue, Energy, global vertex pool
├── geometry.py                    # Polygon geometry calculations
├── simulation.py                  # Simulation engine
├── plotting.py                    # Visualization (enhanced with ID labels)
├── builders.py                    # Tissue builders (honeycomb, grid)
├── acam_importer.py               # ACAM tissue conversion
└── io.py                          # Save/load utilities

examples/                          # Example scripts
├── simulate_cell_growth.py        # Main cell growth simulation (creates Sim_*/ folders)
├── diagnose_tissue.py             # Tissue validation and repair tool
├── diagnose_acam_simulation.py    # ACAM parameter diagnostic
├── plot_acam_79cells_labeled.py   # Visualization with labels
├── plot_problem_cells_detail.py   # Detailed cell inspection
├── convert_acam_example.py        # ACAM conversion example
└── ...                            # Additional examples

tests/                             # Pytest suite
├── test_basic.py                  # Core functionality tests
├── test_tissue_structure.py       # Structure validation tests
└── test_tissue_cell_by_cell.py    # Comprehensive cell-by-cell validation

docs/                              # Documentation
├── design_vertex_model.md         # Technical design document
├── ACAM_CONVERSION_GUIDE.md       # ACAM conversion guide
└── ...                            # Additional documentation

scripts/                           # Utility scripts
└── convert_acam_tissue.py         # ACAM conversion CLI tool

pickled_tissues/                   # Pre-built tissue files (.dill)
acam_tissues/                      # ACAM source data
Sim_*/                             # Simulation output folders (gitignored)
```

## Future Development

### Planned Features

1. **Topological transitions**: 
   - T1 transitions (edge swaps for neighbor rearrangement)
   - Cell division (vertex insertion and polygon splitting)
   - Cell extrusion/apoptosis (polygon removal)

2. **Adaptive simulation**:
   - Adaptive timestep based on gradient magnitude
   - Energy-based step acceptance/rejection
   - Automatic parameter tuning for different tissue sizes

3. **Enhanced mechanics**:
   - Per-edge heterogeneous tension (currently uniform γ)
   - Bending/curvature energy terms
   - Area constraints via Lagrange multipliers

4. **Performance optimization**:
   - Cached geometry computations (areas, perimeters)
   - Sparse gradient calculations
   - Vectorized operations for large tissues

5. **Analysis tools**:
   - Cell tracking across timesteps
   - Stress/strain field visualization
   - Automated parameter sweep utilities

### Known Issues

- **ACAM tissues** may contain duplicate consecutive vertices after conversion (use validation and repair tools)
- **Timestep requirements** vary by tissue size (ACAM: dt≈0.0001, Honeycomb: dt≈0.01)
- **Energy explosion** can occur with inappropriate parameters (use diagnostic tools)

See documentation in `docs/` for detailed guides and troubleshooting.

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`pytest -v`)
- New features include tests and documentation
- Code follows existing style conventions
- ACAM tissues are validated after conversion

## License

(Provide license details here if applicable.)

---

**Status**: Fully functional vertex model implementation with comprehensive validation and diagnostic tools.

For questions, issues, or feature requests, please refer to the documentation in `docs/` or file an issue.
