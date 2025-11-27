"""
Input/output utilities for saving and loading simulation states.
"""

from pathlib import Path
from typing import Any

import dill

from .core import Tissue

# Schema version for serialized tissue assets (allows migration in future)
TISSUE_SCHEMA_VERSION = 1
TISSUE_FILE_EXT = ".dill"


def _normalize_tissue_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.suffix.lower() != TISSUE_FILE_EXT:
        path = path.with_suffix(TISSUE_FILE_EXT)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_state(obj: Any, filepath: str):
    """
    Save an object to a file using dill.
    
    Args:
        obj: Object to save (e.g., Tissue, Simulation)
        filepath: Path to save the object
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        dill.dump(obj, f)
    print(f"Saved state to {filepath}")


def load_state(filepath: str) -> Any:
    """
    Load an object from a file using dill.
    
    Args:
        filepath: Path to the saved object
        
    Returns:
        The loaded object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = dill.load(f)
    print(f"Loaded state from {filepath}")
    return obj


# ---------------- New tissue serialization helpers ---------------- #


def save_tissue(tissue: Tissue, filepath: str) -> None:
    """Serialize a Tissue to disk using a single dill file."""
    path = _normalize_tissue_path(filepath)

    # Ensure vertex_indices are populated so we can reconstruct geometry on load
    if tissue.vertices.shape[0] == 0:
        tissue.build_global_vertices(tol=1e-10)
    tissue.reconstruct_cell_vertices()

    payload = {
        "schema_version": TISSUE_SCHEMA_VERSION,
        "tissue": tissue,
    }
    with open(path, "wb") as f:
        dill.dump(payload, f)
    print(f"Saved tissue to {path}")


def load_tissue(basepath: str) -> Tissue:
    """Load a Tissue previously serialized by :func:`save_tissue`."""
    path = _normalize_tissue_path(basepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Tissue file not found: {path} (only {TISSUE_FILE_EXT} assets are supported)"
        )

    with open(path, "rb") as f:
        payload = dill.load(f)

    schema_version = payload.get("schema_version")
    if schema_version != TISSUE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported tissue schema version {schema_version} (expected {TISSUE_SCHEMA_VERSION})"
        )

    tissue = payload.get("tissue")
    if not isinstance(tissue, Tissue):
        raise TypeError("Loaded payload does not contain a Tissue instance")
    return tissue
