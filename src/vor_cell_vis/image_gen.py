"""
Main script for image generation
"""

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image
from scipy.spatial import Voronoi
from sklearn import preprocessing

from vor_cell_vis.utils.load_model import load_model_and_fe
from vor_cell_vis.utils.point_picking import (
    compute_max_dots,
    pick_points_number,
    scatter_points,
)


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
    # load depth estimation model and feature extractor
    depth_model, feature_extractor = load_model_and_fe(model)

    # extract depth plots
    inputs = feature_extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # filter image, avg pooling
    avg_depth = torch.nn.functional.avg_pool2d(prediction, filter_size)

    # normalise
    avgs = avg_depth[0][0]
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_avg = min_max_scaler.fit_transform(avgs)

    # sample points for voronoi cells
    max_dots = compute_max_dots(bit_rate, filter_size)
    points = []
    max_y_squares = norm_avg.shape[0]
    for i in range(norm_avg.shape[0]):
        for j in range(norm_avg.shape[1]):
            xy_min = [filter_size * j, filter_size * (max_y_squares - i - 1)]
            xy_max = [filter_size * (j + 1), filter_size * (max_y_squares - i)]
            points.append(
                scatter_points(
                    pick_points_number(float(norm_avg[i, j]), max_dots),
                    xy_min=xy_min,
                    xy_max=xy_max,
                )
            )
    all_points = np.concatenate(points)

    # create voronoi cells and plot
    vor = Voronoi(all_points)

    # scale voronoi points for sampling colours:
    scaled_points = copy(all_points)
    scaled_points[:, 0] = scaled_points[:, 0] * (
        img.size[0] / (norm_avg.shape[1] * filter_size)
    )
    scaled_points[:, 1] = scaled_points[:, 1] * (
        img.size[1] / (norm_avg.shape[0] * filter_size)
    )

    # load image into memory to extract rgb values:
    pix = img.load()

    # matplotlib stuff
    fig, ax = plt.subplots()
    ax.set_axis_off()
    plt.axis("scaled")
    ax.set_ylim(0, norm_avg.shape[0] * filter_size)
    ax.set_xlim(0, norm_avg.shape[1] * filter_size)

    # sample rgb values and colour polygons:
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if -1 not in region:
            polygon = [vor.vertices[i] for i in region]
            colour = pix[scaled_points[r][0], -scaled_points[r][1]]
            ax.fill(
                *zip(*polygon),
                color=(colour[0] / 255, colour[1] / 255, colour[2] / 255),
            )

    return fig
