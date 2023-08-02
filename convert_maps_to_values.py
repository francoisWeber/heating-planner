import os
from os import path as osp

import numpy as np
import pandas as pd
import typer
from loguru import logger as log
from PIL import Image

from heating_planner import colorscale


def get_value_map(final_fname, scale_mini, scale_maxi, **kwargs) -> np.ndarray:
    img = Image.open(osp.join("data/screenshots/", final_fname))
    im = np.asarray(img)
    color_scale = colorscale.extract_colorscale_from_image(img)
    value_scale = colorscale.get_value_scale_from_min_max(
        scale_mini, scale_maxi, len(color_scale)
    )
    colorhash2value = colorscale.get_colorsha2value(color_scale, value_scale)
    hashed_map = colorscale.hash1D_color(im)
    color_hashes = np.array(list(colorhash2value.keys()), dtype=np.int16)
    values = np.array(list(colorhash2value.values()))

    apply_value_map = np.vectorize(
        lambda q: colorscale.approx_retrieve_values_from_hashed_colormap(
            q, colorhash2value, color_hashes, values
        )
    )

    return apply_value_map(hashed_map)


def get_valuemap_fname(s: pd.Series) -> str:
    return s.final_fname.replace("screen", "values").replace("png", "npy")


def convert(
    metadata_file_path: str = typer.Option(...),
    output_dir: str = typer.Option(...),
):
    log.info("Starting conversion tool to cast color maps to maps of values")
    log.info(f"Reading from {metadata_file_path}")
    df = pd.read_csv(metadata_file_path)
    df["valuemap_fname"] = df.apply(get_valuemap_fname, axis=1)

    os.makedirs(output_dir, exist_ok=True)
    log.info("Converting colormaps to value-maps ...")
    for i, data in df.iterrows():
        try:
            log.info(f"{i=} : converting {data.final_fname}")
            value_map = get_value_map(**data.to_dict())

            with open(osp.join(output_dir, data.valuemap_fname), "wb") as f:
                np.save(f, value_map)
                log.info(f"Serialized as {data.valuemap_fname}")
        except Exception as e:
            log.warning(f"Exception {e}")

    log.info("Done :)")


if __name__ == "__main__":
    typer.run(convert)
