import os
import traceback
from os import path as osp

import numpy as np
import pandas as pd
import typer
from loguru import logger as log
from matplotlib import pyplot as plt
from PIL import Image

from heating_planner.colorscale import (
    hash1D_color,
    get_colorscale,
    get_ordered_hashed_colorscale_items,
    get_out_of_range_values,
)


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


def convert(fpath, value_range, **kwargs):
    im = Image.open(fpath)
    img = np.asarray(im)
    # get colorscale
    colorscale = get_colorscale(im)
    colorscale_hashed = hash1D_color(colorscale)
    ordered_hashed_colors = get_ordered_hashed_colorscale_items(colorscale_hashed)
    # hash the image
    hashed_img = hash1D_color(img)
    # convert to values
    value_map = np.vectorize(
        get_hashedcol2values_fn(ordered_hashed_colors, value_range)
    )(hashed_img)
    return value_map


def craft_title(var, term, season, **kwargs):
    return f"Map {term} term for var={var} during {season}"


def go(
    metadata_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata.json"
    ),
    output_dir: str = typer.Option("/Users/f.weber/Downloads/out"),
):
    df = pd.read_json(metadata_path)

    array_output_subdir = osp.join(output_dir, "arrays")
    os.makedirs(array_output_subdir, exist_ok=True)
    maps_output_subdir = osp.join(output_dir, "maps")
    os.makedirs(maps_output_subdir, exist_ok=True)

    col_array_output = "fpath_array"
    col_map_output = "fpath_map"
    output_summary = []

    for fpath, metadata in df.iterrows():
        log.info(f"Dealing with {fpath=}")
        try:
            # get value map
            value_map = convert(fpath, **metadata.to_dict())
            value_map = value_map[330:, :1200]
            # set IO
            fname, _ = osp.splitext(osp.basename(fpath))
            array_output_path = osp.join(array_output_subdir, fname + ".npy")
            maps_output_path = osp.join(maps_output_subdir, fname + ".png")

            with open(array_output_path, "wb") as f:
                np.save(f, value_map)
                log.info(f"\terialized as array at {array_output_path}")
            with open(maps_output_path, "wb") as f:
                plt.imshow(value_map)
                plt.colorbar()
                plt.title(craft_title(**metadata.to_dict()))
                plt.savefig(f)
                plt.close()
                log.info(f"\tSerialized as array at {maps_output_path}")

            output_summary.append(
                {
                    "file": fpath,
                    col_array_output: array_output_path,
                    col_map_output: maps_output_path,
                }
            )
        except Exception as e:
            output_summary.append(
                {
                    col_array_output: None,
                    col_map_output: None,
                }
            )
            log.warning(f"\tException {e}")
            traceback.print_exception(e)
    _df = pd.DataFrame(data=output_summary).set_index("file")
    df = df.join(_df)
    log.info("Updating metadata")
    df.to_json(metadata_path)


if __name__ == "__main__":
    # typer.run(go)
    fpath = "/Users/f.weber/tmp-fweber/heating/vague_chaleur/ref_nHeatWave_winter.png"
    scale_range = [
        2,
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        26,
        28,
        30,
        32,
        34,
        36,
        38,
        40,
    ]
    convert(fpath, scale_range)
