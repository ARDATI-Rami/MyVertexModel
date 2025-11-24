"""Small example script to plot a tiny tissue.
Run:
    python -m examples.plot_example
"""
import numpy as np
import matplotlib.pyplot as plt
from myvertexmodel import Tissue, Cell, plot_tissue

def main():
    tissue = Tissue()
    tissue.add_cell(Cell(cell_id=1, vertices=np.array([[0, 0], [1, 0], [1, 1], [0, 1]])))
    tissue.add_cell(Cell(cell_id=2, vertices=np.array([[1, 0], [2, 0], [2, 1], [1, 1]])))
    ax = plot_tissue(tissue)
    plt.show()

if __name__ == "__main__":
    main()

