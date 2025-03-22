"""
Main script for image generation
"""

from matplotlib.figure import Figure
from PIL import Image

from vor_cell_vis.utils.load_model import load_model_and_fe


def gen_vor_image(
    img: Image, filter_size: int, bit_rate: float, model: str = "Intel/dpt-hybrid-midas"
) -> Figure:
    """Generate a voronoi cell image

    Args:
        img: Input images
        filter_size: size of the filter for coverage
        bit_rate: how much to reduce the image quality by

    Returns:
        matplotlib figure of coloured voronoi cells
    """
