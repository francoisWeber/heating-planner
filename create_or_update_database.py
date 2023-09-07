import os
import traceback
from os import path as osp

import numpy as np
import pandas as pd
import typer
from loguru import logger as log
from matplotlib import pyplot as plt
from glob import glob
from heating_planner.file_layout import craft_metadata_from_fname

from heating_planner.colorscale import convert


# This script reads one file or an entire directory to retrieve PNG
# maps to convert into value-2D-arrays. The colorscale of the PNG is used
# to convert colors to floats representing the value of the physical variable
# exposed in the PNG map.
# The file naming is important ; it should contains the following info:
# - var: the name of the represented variable
# - season: the season displayed on the map
# - term: the term of the modelization
# - values: the numerical values of the colorscale, separated with a $ symbol
# Each key must be separated from it's value with an '=' symbol.
# The key-value pairs must be separated with a '_' symbol.
# Example of naming:
# term=medium_season=summer_var=rain_values=1$2$3$4$5.png


def craft_title(variable, term, season, **kwargs):
    return f"Map {term} term for var={variable} during {season}"


def go(
    input_location: str = typer.Option("/Users/f.weber/tmp-fweber/heating/raw_data"),
    metadata_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata2.json"
    ),
    output_dir: str = typer.Option("/Users/f.weber/tmp-fweber/heating/processed/"),
    force: bool = typer.Option(False),
    resize_factor: int = typer.Option(5),
):
    input_location = osp.expanduser(input_location)
    metadata_path = osp.expanduser(metadata_path)
    output_dir = osp.expanduser(output_dir)
    df = pd.read_json(metadata_path) if osp.isfile(metadata_path) else None

    if osp.isfile(input_location):
        input_files = [input_location]
    elif osp.isdir(input_location):
        input_files = glob(osp.join(input_location, "**", "*.png"), recursive=True)
    else:
        raise ValueError(f"Nothing at {input_location}")

    resize_factor = resize_factor if resize_factor else 1

    array_output_subdir = osp.join(output_dir, "arrays")
    os.makedirs(array_output_subdir, exist_ok=True)
    maps_output_subdir = osp.join(output_dir, "maps")
    os.makedirs(maps_output_subdir, exist_ok=True)

    col_array_output = "fpath_array"
    col_map_output = "fpath_map"
    output_summary = []
    index_to_drop = []

    for input_file in input_files:
        log.info(f"Dealing with {input_file=}")
        try:
            # check if we already have it
            if df is not None and input_file in df.index:
                if force:
                    index_to_drop.append(input_file)
                else:
                    log.info(f"Data already processed: {input_file}")
                    continue

            # else, proceed to conversion
            metadata = craft_metadata_from_fname(input_file)

            # get value map
            value_map = convert(input_file, resize_factor=resize_factor, **metadata)
            value_map = value_map[(330 // resize_factor) :, : (1200 // resize_factor)]
            # set IO
            fname, _ = osp.splitext(osp.basename(input_file))
            array_output_path = osp.join(array_output_subdir, fname + ".npy")
            maps_output_path = osp.join(maps_output_subdir, fname + ".png")

            with open(array_output_path, "wb") as f:
                np.save(f, value_map)
                log.info(f"\tserialized as array at {array_output_path}")
            with open(maps_output_path, "wb") as f:
                plt.imshow(value_map, cmap="jet")
                plt.colorbar()
                plt.title(craft_title(**metadata))
                plt.savefig(f)
                plt.close()
                log.info(f"\terialized as array at {maps_output_path}")

            output_summary.append(
                {
                    "file": input_file,
                    col_array_output: array_output_path,
                    col_map_output: maps_output_path,
                    "resize_factor": resize_factor,
                    **metadata,
                }
            )
        except Exception as e:
            log.warning(f"\tException {e}")
            traceback.print_exception(e)
    # backup metadata
    if len(output_summary) > 0:
        df_new = pd.DataFrame(data=output_summary).set_index("file")
        if df is not None:
            df = pd.concat([df, df_new])
            log.info("Updating metadata")
        else:
            df = df_new
            log.info("Creating metadata")
        df.to_json(metadata_path)
    log.info("Done :)")


if __name__ == "__main__":
    typer.run(go)
