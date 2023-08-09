import numpy as np
from PIL import Image

from heating_planner.utils import softdict_assignation

COLORSCALE_LEFT_PIX = 280
COLORSCALE_TOP_PIX = 1040
COLORSCALE_RIGHT_PIX = 300
COLORSCALE_BOTTOM_PIX = 2079


def extract_colorscale_from_image(
    img: Image.Image,
    left: int = COLORSCALE_LEFT_PIX,
    top: int = COLORSCALE_TOP_PIX,
    right: int = COLORSCALE_RIGHT_PIX,
    bottom: int = COLORSCALE_BOTTOM_PIX,
) -> np.ndarray:
    return np.asarray(img.crop((left, top, right, bottom)))[:, 0, :3]


def get_value_scale_from_min_max(
    mini: float, maxi: float, len_of_scale: int
) -> np.ndarray:
    return np.linspace(start=maxi, stop=mini, num=len_of_scale)


def hash1D_color(colorized_map: np.ndarray) -> np.ndarray:
    output = np.zeros_like(colorized_map[..., 0])
    for channel_id in range(colorized_map.shape[-1]):
        output += colorized_map[..., channel_id] * (255**channel_id)
    return output


def get_colorsha2value(color_scale: np.ndarray, value_scale: np.ndarray) -> dict:
    hashed_color_scale = hash1D_color(color_scale)
    return {
        hashed_color: value
        for hashed_color, value in zip(hashed_color_scale, value_scale)
    }


def approx_retrieve_values_from_hashed_colormap(
    hashedcolor_value: int,
    colorhash2value: dict,
    colorhashes: np.ndarray,
    values: np.ndarray,
    factor: float = 10,
) -> float:
    """approx_retrieve_values_from_hashed_colormap retrieve exact value from a dict if it exists or an approximation otherwise

    Parameters
    ----------
    hashedcolor_value : int
        the hashed value to retrieve
    colorhash2value : dict
        dictionnary of assignation colorhash => values for existing colorhashes
    colorhashes : np.ndarray
        the array of every known
    values : np.ndarray
        the array of values associated to known colorhashes
    factor : float, optional
        numerical stability factor, by default 10

    Returns
    -------
    float
        the exact value if the hashedcolor is known or an approximation of it
    """
    if hashedcolor_value in colorhash2value:
        return colorhash2value[hashedcolor_value]
    else:
        return softdict_assignation(
            hashedcolor_value, colorhashes, values, factor=factor
        )
