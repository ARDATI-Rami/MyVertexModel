"""
Geometric calculations for vertex model.
"""

import numpy as np
from shapely.geometry import Polygon
from typing import Tuple


class GeometryCalculator:
    """Handles geometric calculations for cells and tissues."""
    
    @staticmethod
    def is_valid_polygon(vertices: np.ndarray) -> bool:
        """Return True if vertices form a valid, non-self-intersecting polygon with non-zero area.

        - Fewer than 3 vertices -> False
        - Rejects polygons with repeated vertices
        - Uses shapely Polygon to check validity and area > 0
        """
        if vertices is None:
            return False
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape[0] < 3:
            return False
        # Reject repeated vertices (degenerate inputs)
        uniq = np.unique(vertices, axis=0)
        if uniq.shape[0] < vertices.shape[0]:
            return False
        try:
            poly = Polygon(vertices)
            return bool(poly.is_valid) and (poly.area > 0)
        except Exception:
            return False

    @staticmethod
    def signed_area(vertices: np.ndarray) -> float:
        """Compute signed area via shoelace formula (positive for CCW, negative for CW).

        Does not use shapely; works with Nx2 arrays. Returns 0.0 if <3 vertices.
        """
        vertices = np.asarray(vertices, dtype=float)
        if vertices.shape[0] < 3:
            return 0.0
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))

    @staticmethod
    def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
        """Return a copy of vertices ordered CCW; reverse order if signed_area is negative.

        Input is not modified in-place.
        """
        vertices = np.asarray(vertices, dtype=float)
        if GeometryCalculator.signed_area(vertices) < 0:
            return vertices[::-1].copy()
        return vertices.copy()

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


# Keep module-level helper functions for backward compatibility

def is_valid_polygon(vertices: np.ndarray) -> bool:
    """
    Check if vertices form a valid polygon using shapely.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        True if polygon is valid (non-self-intersecting, non-zero area)
    """
    return GeometryCalculator.is_valid_polygon(vertices)


def polygon_orientation(vertices: np.ndarray) -> float:
    """
    Calculate signed area of polygon using shoelace formula.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        Signed area (positive for CCW, negative for CW)
    """
    return GeometryCalculator.signed_area(vertices)


def ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """
    Return vertices ordered counter-clockwise.

    Args:
        vertices: Nx2 array of vertex coordinates

    Returns:
        New array with vertices in CCW order
    """
    return GeometryCalculator.ensure_ccw(vertices)
