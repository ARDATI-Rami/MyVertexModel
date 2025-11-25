"""
Geometric calculations for vertex model.
"""

import numpy as np
from shapely.geometry import Polygon
from typing import Tuple


class GeometryCalculator:
    """Handles geometric calculations for cells and tissues."""
    
    @staticmethod
    def calculate_area(vertices: np.ndarray) -> float:
        """
        Calculate the area of a polygon defined by vertices.
        
        Args:
            vertices: Array of vertex coordinates (N x 2)
            
        Returns:
            Area of the polygon
        """
        if len(vertices) < 3:
            return 0.0
        poly = Polygon(vertices)
        return poly.area
    
    @staticmethod
    def calculate_perimeter(vertices: np.ndarray) -> float:
        """
        Calculate the perimeter of a polygon defined by vertices.
        
        Args:
            vertices: Array of vertex coordinates (N x 2)
            
        Returns:
            Perimeter of the polygon
        """
        if len(vertices) < 2:
            return 0.0
        poly = Polygon(vertices)
        return poly.length
    
    @staticmethod
    def calculate_centroid(vertices: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the centroid of a polygon defined by vertices.
        
        Args:
            vertices: Array of vertex coordinates (N x 2)
            
        Returns:
            Tuple of (x, y) coordinates of centroid
        """
        if len(vertices) == 0:
            return (0.0, 0.0)
        poly = Polygon(vertices)
        centroid = poly.centroid
        return (centroid.x, centroid.y)


def is_valid_polygon(vertices: np.ndarray) -> bool:
    """
    Check if vertices form a valid polygon using shapely.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        True if polygon is valid (non-self-intersecting, non-zero area)
    """
    if len(vertices) < 3:
        return False

    try:
        poly = Polygon(vertices)
        return poly.is_valid and poly.area > 0
    except Exception:
        return False


def polygon_orientation(vertices: np.ndarray) -> float:
    """
    Calculate signed area of polygon using shoelace formula.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        Signed area (positive for CCW, negative for CW)
    """
    if len(vertices) < 3:
        return 0.0

    x = vertices[:, 0]
    y = vertices[:, 1]

    # Shoelace formula: 0.5 * sum(x[i]*y[i+1] - x[i+1]*y[i])
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    return signed_area


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """
    Return vertices ordered counter-clockwise.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        New array with vertices in CCW order
    """
    if polygon_orientation(vertices) < 0:
        return vertices[::-1].copy()
    return vertices.copy()
