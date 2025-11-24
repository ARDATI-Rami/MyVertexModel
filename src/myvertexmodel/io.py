"""
Input/output utilities for saving and loading simulation states.
"""

import dill
from pathlib import Path
from typing import Any


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

