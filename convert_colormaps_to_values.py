import os
from os import path as osp

import numpy as np
import pandas as pd
import typer
from loguru import logger as log
from PIL import Image
from functools import lru_cache
from heating_planner import constantes as cst
from heating_planner.colorscale import extract_colorscale_from_image
from matplotlib import pyplot as plt
import traceback

CROP_LEFT = 1350
CROP_TOP = 1050
CROP_RIGHT = CROP_LEFT + 600
CROP_BOTTOM = CROP_TOP + 600


def crop_to_france(im: Image.Image) -> Image.Image:
    return im.crop((CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM))


def hash_color(arrays_on_3channels):
    return np.sum(arrays_on_3channels * np.power(255, [0, 1, 2]), -1)


def approx_hash_color2value(
    hashed_color,
    hashed_color2value: dict,
    hashed_colorbar: np.ndarray,
    values_range: np.ndarray,
):
    if hashed_color in hashed_color2value:
        return hashed_color2value[hashed_colorbar]
    else:
        return _approx_hash_color2value(hashed_color, hashed_colorbar, values_range)


def _approx_hash_color2value(
    hashed_color, hashed_colorbar: np.ndarray, values_range: np.ndarray
):
    weights = 1 / (np.abs(hashed_colorbar - hashed_color) ** 2)
    weights /= weights.sum()
    return np.sum(weights * values_range)


def get_association_fn(
    hashed_color2value: dict, hashed_colorbar: np.ndarray, values_range: np.ndarray
):
    @lru_cache(maxsize=1024)
    def fn_out_of_dict(hashed_color) -> float:
        return _approx_hash_color2value(hashed_color, hashed_colorbar, values_range)

    def fn(hashed_color):
        if hashed_color in hashed_color2value:
            return hashed_color2value[hashed_color]
        else:
            return fn_out_of_dict(hashed_color)

    return fn


def get_value_map(
    screenshot_fname, scale_mini, scale_maxi, crop=True, **kwargs
) -> np.ndarray:
    img_fname = osp.join("data", cst.SCREENSHOT_DIRECTORY, screenshot_fname)
    im = Image.open(img_fname)
    # get colorbar
    colorbar = extract_colorscale_from_image(im)

    # maybe crop and convert to numpy
    if crop:
        im = crop_to_france(im)
    img = np.asarray(im, dtype=np.float32)[:, :, :3]

    # hash colors everywhere and prepare value scale
    img_hashed = hash_color((img))
    hashed_colorbar = hash_color(colorbar)
    values_range = np.linspace(scale_maxi, scale_mini, len(hashed_colorbar))
    hashed_color2value = {
        hash_color: value for hash_color, value in zip(hashed_colorbar, values_range)
    }

    # prepare and execute the color => value association
    get_value = get_association_fn(hashed_color2value, hashed_colorbar, values_range)
    get_value_vec = np.vectorize(get_value)

    return get_value_vec(img_hashed)


def get_valuemap_fname(s: pd.Series) -> str:
    return (
        s[cst.COL_SCREENSHOTS_FNAME].replace("screen", "values").replace("png", "npy")
    )


def craft_title(variable, season, **kwargs):
    return f"Map for var={variable} during {season}"


def convert(
    metadata_file_path: str = typer.Option(..., "-i"),
    output_dir: str = typer.Option("data", "-o"),
):
    log.info("Starting conversion tool to cast color maps to maps of values")
    log.info(f"Reading from {metadata_file_path}")
    df = pd.read_json(metadata_file_path, lines=True)
    # appending new column for the numpy output
    df[cst.COL_VALUE_MAP_ARRAY_FNAME] = df[cst.COL_SCREENSHOTS_FNAME].apply(
        lambda s: s.replace("screen_", "array_").replace(".png", ".pny")
    )
    # appending new column for the img map output
    df[cst.COL_VALUE_MAP_IMG_FNAME] = df[cst.COL_SCREENSHOTS_FNAME].apply(
        lambda s: s.replace("screen_", "map_")
    )

    # craft output subdirs
    array_dir = osp.join(output_dir, cst.VALUE_MAP_ARRAY_DIRECTORY)
    os.makedirs(array_dir, exist_ok=True)
    img_dir = osp.join(output_dir, cst.VALUE_MAP_IMG_DIRECTORY)
    os.makedirs(img_dir, exist_ok=True)

    # loop over screenshots to convert them into value-maps
    log.info("Converting colormaps to value-maps ...")
    for i, data in df.iterrows():
        try:
            log.info(f"{i=} : converting {data[cst.COL_SCREENSHOTS_FNAME]}")
            value_map = get_value_map(**data.to_dict())
            array_fname = osp.join(
                array_dir,
                data[cst.COL_VALUE_MAP_ARRAY_FNAME],
            )
            img_fname = osp.join(
                img_dir,
                data[cst.COL_VALUE_MAP_IMG_FNAME],
            )
            with open(array_fname, "wb") as f:
                np.save(f, value_map)
                log.info(f"Serialized as array at {array_fname}")
            with open(img_fname, "wb") as f:
                plt.imshow(value_map)
                plt.colorbar()
                plt.title(craft_title(**data.to_dict()))
                plt.savefig(f)
                plt.close()
                log.info(f"Serialized as array at {img_fname}")
        except Exception as e:
            log.warning(f"Exception {e}")
            traceback.print_exception(e)

    log.info("Updating metadata")
    df.to_json(metadata_file_path, orient="records", lines=True)
    log.info("Done :)")


if __name__ == "__main__":
    typer.run(convert)
