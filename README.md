# MyVertexModel

A prototype 2D epithelial vertex model implementation. It represents a confluent tissue as polygons (cells) whose vertices define geometry and mechanics. The current prototype includes:

- Energy functional with area elasticity, perimeter contractility, and line (junction) tension terms.
- Finite-difference (central) gradient descent for vertex position updates.
- Global vertex pool scaffolding (migration in progress from per-cell vertices).
- Plotting utilities to visualize tissue geometry.
- Command-line demo (`python -m myvertexmodel`) for quick simulation and plotting.
- Extensive test suite and design/documentation artifacts.

## Features (Current State)

- Cell & Tissue data structures with validation (non-self-intersecting polygons, area checks).
- Energy API: `EnergyParameters`, `cell_energy`, `tissue_energy` (area, perimeter, tension).
- Numerical gradient: `finite_difference_cell_gradient` plus placeholder analytic gradient API.
- Simulation loop with configurable `dt`, finite-difference `epsilon`, and `damping` factor.
- CLI exposing energy and simulation parameters (`--k-area`, `--k-perimeter`, `--gamma`, `--target-area`, `--epsilon`, `--damping`, `--dt`, `--steps`).
- Matplotlib plotting (`plot_tissue`) and example script.

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

All functionality is covered by a pytest suite:

```bash
python -m pytest -q            # Quiet run
python -m pytest -v            # Verbose
python -m pytest tests/        # Explicit path
```

## CLI Usage

Run a small simulation (2×2 grid, 10 steps) and show a plot:

```bash
python -m myvertexmodel --steps 10 --grid-size 2 --plot
```

Customize parameters (example):

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

## Plotting Example Script

A minimal example is provided in `examples/plot_example.py`:

```bash
python examples/plot_example.py
```

This script constructs a simple tissue and renders it with matplotlib.

## Design Document

See `design_vertex_model.md` for a concise technical outline of:

- Data model evolution (per-cell vertices → global vertex indices)
- Planned analytic gradient derivation
- Energy functional references (Farhadifar 2007, Nagai & Honda 2009)
- Future extensions (topological transitions, force calculations)

## Project Structure (Summary)

```
environment.yml          # Conda environment
pyproject.toml           # Package metadata / editable install
README.md                # This file
design_vertex_model.md   # Design outline
src/myvertexmodel/       # Core package modules
examples/                # Example scripts
tests/                   # Pytest suite
```

## Next Steps (Roadmap Snapshot)

1. Implement analytic gradient (`cell_energy_gradient_analytic`).
2. Migrate simulation to fully use global vertex pool.
3. Introduce topological events (T1 transitions, division).
4. Add force & energy logging and parameter sweep utilities.

## License

(Provide license details here if applicable.)

---

Feedback, issues, and contributions are welcome during prototype evolution.
