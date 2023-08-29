import numpy as np
from PIL import Image
from PIL.Image import Resampling

LEFT = 1230
TOP = 340
BOTTOM = 1490


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


def get_hashedcol2values_dict(ordered_hashed_colors, scale_range):
    m, M = get_out_of_range_values(scale_range)
    hashedcol2value = {
        ordered_hashed_colors[0]: m,  # upper extremum out of bound
        ordered_hashed_colors[-1]: M,  # lower extremum out of bound
    }
    for hashed_col, vmin, vmax in zip(
        ordered_hashed_colors[1:-1], scale_range[:-1], scale_range[1:]
    ):
        hashedcol2value[hashed_col] = (vmax + vmin) / 2
    return hashedcol2value


def get_hashedcol2values_fn(ordered_hashed_colors, scale_range):
    hashing_dict = get_hashedcol2values_dict(ordered_hashed_colors, scale_range)

    def f(value):
        return hashing_dict.get(value, np.nan)

    return f


def convert(fpath, values, resize_factor=None, **kwargs):
    im = Image.open(fpath)
    if resize_factor and resize_factor > 1:
        im = im.resize(
            (im.width // resize_factor, im.height // resize_factor),
            resample=Resampling.NEAREST,  # NEAREST to keep colormap
        )
    img = np.asarray(im, dtype=np.float16)
    # get colorscale
    v_min = TOP // (resize_factor if resize_factor else 1) + 1
    v_max = BOTTOM // (resize_factor if resize_factor else 1)
    h_min = LEFT // (resize_factor if resize_factor else 1)
    h_max = h_min + 1
    colorscale = img[v_min:v_max, h_min:h_max][:, 0, :]
    # hash it
    colorscale_hashed = hash1D_color(colorscale)
    ordered_hashed_colors = get_ordered_hashed_colorscale_items(colorscale_hashed)
    # hash the image
    hashed_img = hash1D_color(img)
    # convert to values
    value_map = np.vectorize(get_hashedcol2values_fn(ordered_hashed_colors, values))(
        hashed_img
    )
    return value_map
