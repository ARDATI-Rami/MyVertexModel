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

