#!/usr/bin/env python
"""
CLI wrapper for topology-aware ACAM tissue conversion.

This script provides a command-line interface to the convert_acam_with_topology()
function from myvertexmodel.acam_importer, maintaining backward compatibility with
the original fix_acam_interactive_79cells.py interface.

Usage:
    python scripts/convert_acam_tissue.py \\
        --acam-file acam_tissues/80_cells \\
        --neighbor-json acam_tissues/acam_79_neighbors.json \\
        --merge-radius 14.0 \\
        --max-vertices 10 \\
        --output-prefix acam_79cells \\
        --validate-connectivity

Outputs:
    - pickled_tissues/<prefix>.dill        (Full tissue ready for simulations)
    - <prefix>_summary.json                (Cell topology and statistics)
    - <prefix>_validation.txt              (Connectivity report, if --validate-connectivity)
"""
import argparse
import sys
from pathlib import Path

# Add src to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from myvertexmodel import convert_acam_with_topology, save_tissue


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert ACAM tissue to vertex model using neighbor topology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python scripts/convert_acam_tissue.py \\
        --acam-file acam_tissues/80_cells \\
        --neighbor-json acam_tissues/acam_79_neighbors.json \\
        --merge-radius 14.0 \\
        --output-prefix acam_79cells \\
        --validate-connectivity
        """
    )
    
    parser.add_argument(
        '--acam-file',
        type=str,
        default='acam_tissues/80_cells',
        help='Path to ACAM pickle file (default: acam_tissues/80_cells)'
    )
    
    parser.add_argument(
        '--neighbor-json',
        type=str,
        default='acam_tissues/acam_79_neighbors.json',
        help='Path to neighbor topology JSON file (default: acam_tissues/acam_79_neighbors.json)'
    )
    
    parser.add_argument(
        '--merge-radius',
        type=float,
        default=14.0,
        help='Radius for junction vertex fusion (default: 14.0)'
    )
    
    parser.add_argument(
        '--max-vertices',
        type=int,
        default=10,
        help='Maximum vertices per cell (safety cap) (default: 10)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='acam_79cells',
        help='Prefix for output files (default: acam_79cells)'
    )
    
    parser.add_argument(
        '--validate-connectivity',
        action='store_true',
        help='Validate all ACAM neighbor pairs are connected'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Setup output paths
    output_dir = Path('pickled_tissues')
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue_path = output_dir / f"{args.output_prefix}.dill"
    summary_path = Path(f"{args.output_prefix}_summary.json")
    validation_path = Path(f"{args.output_prefix}_validation.txt") if args.validate_connectivity else None
    
    # Convert tissue
    result = convert_acam_with_topology(
        acam_file=args.acam_file,
        neighbor_json=args.neighbor_json,
        merge_radius=args.merge_radius,
        max_vertices=args.max_vertices,
        validate_connectivity=args.validate_connectivity,
        save_summary=summary_path,
        save_validation=validation_path,
        verbose=not args.quiet
    )
    
    # Save tissue
    save_tissue(result.tissue, str(tissue_path))
    
    if not args.quiet:
        print(f"\n✓ Saved tissue to {tissue_path}")
        print(f"✓ Saved summary to {summary_path}")
        if validation_path:
            print(f"✓ Saved validation report to {validation_path}")
        
        print(f"\nSummary:")
        print(f"  Total cells: {result.summary['total_cells']}")
        print(f"  Boundary cells: {result.summary['boundary_cells']}")
        print(f"  Interior cells: {result.summary['interior_cells']}")
        print(f"  Global vertices: {result.tissue.vertices.shape[0]}")
        
        if result.validation is not None:
            print(f"\nValidation:")
            print(f"  Total connectivity: {result.validation['total_connectivity']['percentage']:.1f}%")
            if result.validation['disconnected']['count'] > 0:
                print(f"  ⚠ Warning: {result.validation['disconnected']['count']} disconnected pairs")


if __name__ == "__main__":
    main()

