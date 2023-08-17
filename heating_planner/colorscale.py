import numpy as np
from PIL import Image

LEFT = 1228
TOP = 337
RIGHT = LEFT + 1
BOTTOM = 1493


def get_colorscale(im: Image.Image) -> np.ndarray:
    colorscale = np.asarray(im.crop((LEFT, TOP, RIGHT, BOTTOM)))[:, 0, :]
    return colorscale


def get_ordered_hashed_colorscale_items(colorscale_hashed):
    ordered_hashed_colors = []
    for c in colorscale_hashed:
        # discard 0 that is black
        if c in ordered_hashed_colors or c == 0:
            continue
        else:
            ordered_hashed_colors.append(c)
    return np.flip(ordered_hashed_colors)


def get_out_of_range_values(scale_range, alpha=1.2):
    m = min(scale_range)
    M = max(scale_range)
    segment_len = M - m
    half = m + segment_len / 2
    m_lower = half - (segment_len / 2) * alpha
    M_upper = half + (segment_len / 2) * alpha
    return (m_lower, M_upper)


def hash1D_color(colorized_map: np.ndarray) -> np.ndarray:
    output = colorized_map * np.power(256, range(colorized_map.shape[-1]))
    return output.sum(axis=-1)
