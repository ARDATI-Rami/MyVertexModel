"""Prebuild and serialize canonical tissue geometries.

Generates a set of reference tissues using builder utilities and stores them
under the `pickled_tissues/` directory using dill serialization.

Run:
    python examples/prebuild_tissues.py --all

Or specify individual tissues:
    python examples/prebuild_tissues.py --grid 4 4 --cell-size 1.0
    python examples/prebuild_tissues.py --honeycomb --hex-size 1.0

Outputs:
    pickled_tissues/index.json  (manifest listing available tissues and metadata)
    pickled_tissues/<name>.dill (geometry data serialized with dill)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any
from myvertexmodel import (
    build_grid_tissue,
    build_honeycomb_2_3_4_3_2,
    save_tissue,
)

OUTPUT_DIR = Path("pickled_tissues")
MANIFEST_PATH = OUTPUT_DIR / "index.json"


def parse_args():
    p = argparse.ArgumentParser(description="Prebuild tissue geometries and serialize them.")
    p.add_argument("--grid", nargs=2, type=int, metavar=("NX", "NY"), help="Build an NX × NY grid tissue")
    p.add_argument("--cell-size", type=float, default=1.0, help="Cell size for grid builder")
    p.add_argument("--honeycomb", action="store_true", help="Build 14-cell honeycomb (2–3–4–3–2 pattern)")
    p.add_argument("--hex-size", type=float, default=1.0, help="Hex size (circumradius) for honeycomb builder")
    p.add_argument("--all", action="store_true", help="Build all canonical tissues")
    return p.parse_args()


def build_and_save(manifest: Dict[str, Any], name: str, tissue) -> None:
    base = OUTPUT_DIR / name
    save_tissue(tissue, str(base))
    manifest[name] = {
        "n_cells": len(tissue.cells),
        "path": f"{name}.dill",
    }


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}

    if args.all or args.grid:
        if args.grid:
            nx, ny = args.grid
        else:
            nx, ny = 2, 2  # default small grid
        grid = build_grid_tissue(nx=nx, ny=ny, cell_size=args.cell_size)
        build_and_save(manifest, f"grid_{nx}x{ny}", grid)

    if args.all or args.honeycomb:
        hc = build_honeycomb_2_3_4_3_2(hex_size=args.hex_size)
        build_and_save(manifest, "honeycomb_2_3_4_3_2", hc)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":  # pragma: no cover
    main()
