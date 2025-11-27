#!/usr/bin/env python
"""Build and store canonical honeycomb tissues for downstream simulations.

This script builds two deterministic honeycomb layouts and saves them under
pickled_tissues/ so runtime scripts can simply load the pre-built tissues.

- honeycomb_14cells: 2-3-4-3-2 pattern (14 total cells)
- honeycomb_19cells: 3-4-5-4-3 pattern (19 total cells)
"""
from __future__ import annotations

from pathlib import Path

from myvertexmodel import (
    build_honeycomb_2_3_4_3_2,
    build_honeycomb_3_4_5_4_3,
    save_tissue,
)

OUTPUT_DIR = Path("pickled_tissues")


def build_and_save(label: str, builder_fn) -> None:
    tissue = builder_fn()
    tissue.build_global_vertices(tol=1e-10)
    save_path = OUTPUT_DIR / label
    save_tissue(tissue, str(save_path))
    print(f"✓ Saved {label} ({len(tissue.cells)} cells, {tissue.vertices.shape[0]} vertices)")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building canonical honeycomb tissues...")
    build_and_save("honeycomb_14cells", build_honeycomb_2_3_4_3_2)
    build_and_save("honeycomb_19cells", build_honeycomb_3_4_5_4_3)
    print("All honeycomb tissues built and saved.")


if __name__ == "__main__":
    main()

