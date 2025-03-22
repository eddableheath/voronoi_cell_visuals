"""
Utility code for picking points for the voronoi cells to be drawn
"""

import numpy as np


def pick_points_number(depth: float, max_points: float) -> int:
    """
    Pick the number of points to drop into a square, minimum of 1

    Args:
        depth: val in [0,1] representing avg depth of square.
        max_points: maximum number of points to drop into that square.

    Returns:
        Number of points to sprinkle into region
    """
    return max(np.around((max_points - 1) ** depth - 1).astype(int), 1)


def compute_max_dots(bit_rate: float, pixels_square) -> float:
    """
    Compute the max dots value for a given bit reduction goal.

    Args:
        bit_rate: bit reduction rate.
        pixels_square: number of pixels in the square

    Returns:
        Upper bound of maximum number of points
    """
    return bit_rate * pixels_square


def scatter_points(n_points, xy_min, xy_max) -> np.ndarray:
    """
        Uniform randomly sample points within a square

    Args:
        n_points: number of points
        xy_min: bottom left coordinates for square
        xy_max: top right coordinates for square

    Return:
        2d array of points
    """
    return np.random.uniform(xy_min, xy_max, size=(n_points, 2))
