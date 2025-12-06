"""
Example demonstrating cell division (cytokinesis) in the vertex model.

This script:
1. Creates a single cell
2. Initiates cytokinesis (inserts contracting vertices)
3. Simulates with contractile forces
4. Splits the cell when sufficiently constricted
5. Visualizes the process
"""

import numpy as np
import matplotlib.pyplot as plt
from myvertexmodel import (
    Cell,
    Tissue,
    EnergyParameters,
    Simulation,
    OverdampedForceBalanceParams,
    plot_tissue,
    CytokinesisParams,
    perform_cytokinesis,
    compute_contractile_forces,
    check_constriction,
)


def cytokinesis_active_force_func(cell, tissue, params):
    """Compute active forces for dividing cells."""
    cytokinesis_params = params.get('cytokinesis_params')
    if cytokinesis_params is None:
        return np.zeros_like(cell.vertices)
    
    if hasattr(cell, 'cytokinesis_data') and 'contracting_vertices' in cell.cytokinesis_data:
        return compute_contractile_forces(cell, tissue, cytokinesis_params)
    return np.zeros_like(cell.vertices)


def main():
    """Run cytokinesis example."""
    print("=== Cell Division (Cytokinesis) Example ===\n")
    
    # Create a rectangular cell (2x1.5 units)
    print("1. Creating initial cell...")
    vertices = np.array([
        [0, 0],
        [2, 0],
        [2, 1.5],
        [0, 1.5]
    ], dtype=float)
    
    cell = Cell(cell_id=1, vertices=vertices)
    tissue = Tissue()
    tissue.add_cell(cell)
    tissue.build_global_vertices()
    
    # Plot initial state
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    plot_tissue(tissue, ax=ax1, show_cell_ids=True, show_vertex_ids=True)
    ax1.set_title('Initial Cell')
    ax1.set_aspect('equal')
    
    # Initiate cytokinesis
    print("2. Initiating cytokinesis (inserting contracting vertices)...")
    cyto_params = CytokinesisParams(
        constriction_threshold=0.15,
        initial_separation_fraction=0.9,
        contractile_force_magnitude=80.0
    )
    
    # Use horizontal division axis (perpendicular to long axis)
    result = perform_cytokinesis(cell, tissue, axis_angle=np.pi/2, params=cyto_params)
    print(f"   Stage: {result['stage']}")
    print(f"   Contracting vertices: {result['contracting_vertices']}")
    
    # Plot after inserting contracting vertices
    plot_tissue(tissue, ax=ax2, show_cell_ids=True, show_vertex_ids=True)
    v1_idx, v2_idx = result['contracting_vertices']
    v1_pos = tissue.vertices[v1_idx]
    v2_pos = tissue.vertices[v2_idx]
    ax2.plot([v1_pos[0], v2_pos[0]], [v1_pos[1], v2_pos[1]], 'r--', linewidth=2, 
             label='Division line')
    ax2.legend()
    ax2.set_title('After Inserting Contracting Vertices')
    ax2.set_aspect('equal')
    
    # Simulate with contractile forces
    print("3. Running simulation with contractile forces...")
    energy_params = EnergyParameters(
        k_area=1.0,
        k_perimeter=0.5,
        gamma=0.1,
        target_area=3.0  # Original area
    )
    
    ofb_params = OverdampedForceBalanceParams(
        gamma=1.0,
        active_force_func=cytokinesis_active_force_func,
        active_force_params={'cytokinesis_params': cyto_params}
    )
    
    sim = Simulation(
        tissue=tissue,
        energy_params=energy_params,
        dt=0.005,
        solver_type='overdamped_force_balance',
        ofb_params=ofb_params
    )
    
    # Run simulation until constricted
    max_steps = 500
    step = 0
    while step < max_steps:
        sim.run(n_steps=10)
        step += 10
        
        # Update global vertices from local
        tissue.build_global_vertices()
        tissue.reconstruct_cell_vertices()
        
        # Check constriction
        if check_constriction(cell, tissue, cyto_params):
            print(f"   Cell constricted after {step} steps!")
            break
        
        if step % 50 == 0:
            distance = np.linalg.norm(tissue.vertices[v1_idx] - tissue.vertices[v2_idx])
            print(f"   Step {step}: Distance = {distance:.4f}")
    
    # Plot constricted state
    plot_tissue(tissue, ax=ax3, show_cell_ids=True, show_vertex_ids=True)
    v1_pos = tissue.vertices[v1_idx]
    v2_pos = tissue.vertices[v2_idx]
    ax3.plot([v1_pos[0], v2_pos[0]], [v1_pos[1], v2_pos[1]], 'r--', linewidth=2)
    ax3.set_title(f'After Constriction ({step} steps)')
    ax3.set_aspect('equal')
    
    # Split the cell
    print("4. Splitting cell into two daughters...")
    result = perform_cytokinesis(
        cell, tissue, params=cyto_params,
        daughter1_id=2, daughter2_id=3
    )
    
    if result['stage'] == 'completed':
        daughter1, daughter2 = result['daughter_cells']
        print(f"   Division complete!")
        print(f"   Daughter 1 (ID={daughter1.id}): {len(daughter1.vertices)} vertices")
        print(f"   Daughter 2 (ID={daughter2.id}): {len(daughter2.vertices)} vertices")
        print(f"   Total cells in tissue: {len(tissue.cells)}")
    
    # Plot final state
    plot_tissue(tissue, ax=ax4, show_cell_ids=True, show_vertex_ids=True)
    ax4.set_title('After Division (Two Daughter Cells)')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('cytokinesis_example.png', dpi=150)
    print("\n5. Saved visualization to 'cytokinesis_example.png'")
    plt.show()
    
    print("\n=== Cytokinesis Complete ===")


if __name__ == "__main__":
    main()
