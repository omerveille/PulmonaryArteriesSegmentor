import numpy as np

# Color palette generated from this beautiful website : https://medialab.github.io/iwanthue/
colors = [
    [217, 38, 106],
    [0, 152, 91],
    [173, 102, 220],
    [142, 169, 18],
    [84, 0, 102],
    [0, 112, 26],
    [238, 169, 255],
    [91, 97, 0],
    [0, 111, 202],
    [255, 183, 86],
    [0, 61, 130],
    [203, 99, 14],
    [121, 0, 90],
    [249, 182, 130],
    [166, 0, 41],
    [255, 138, 174],
    [143, 36, 0],
    [237, 65, 101],
    [110, 52, 16],
    [255, 120, 89],
]

vessel_colors: list[list[float, float, float]] = [
    (np.array(color, dtype=float) / 255.0).tolist() for color in colors
]
contour_color: list[float, float, float] = [1.0, 215.0 / 255.0, 0.0]
direction_points_color: list[float, float, float] = (
    np.array([130, 255, 70], dtype=float) / 255.0
).tolist()

centerline_color: list[float, float, float] = (
    np.array([139, 152, 255], dtype=float) / 255.0
).tolist()
contour_points_color: list[float, float, float] = (
    np.array([255, 147, 161], dtype=float) / 255.0
).tolist()
