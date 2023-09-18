import os
import typer
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from os import path as osp
import pandas as pd
from heating_planner import constantes as c
from loguru import logger as log
from scipy.signal import convolve2d


def go(
    input_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/aux_maps/seaLevel cropped.png"
    ),
    output_dir: str = typer.Option("/Users/f.weber/tmp-fweber/heating/processed/aux"),
    aux_metadata_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata_aux2.json"
    ),
    resize_factor: int = typer.Option(5),
):
    previous_df = (
        pd.read_json(aux_metadata_path) if osp.isfile(aux_metadata_path) else None
    )

    # load resize convert
    im = Image.open(input_path)
    im = im.resize((1388, 1510), resample=Image.Resampling.BICUBIC)
    img = np.asanyarray(im, dtype=np.float32)
    # Extract seaflooding
    img_sealevel = img[:, :, 2] * img[:, :, 1] / (img[:, :, 0] + 1)
    sea_elevation = (img_sealevel < 110).astype(np.float32)
    psf_w = 2
    sea_elevation = convolve2d(sea_elevation, np.ones((psf_w, psf_w)), mode="same") > 0
    # reshape it back
    mask = np.load("/Users/f.weber/tmp-fweber/heating/france_mask_factor5.npy")
    array = np.asanyarray(
        Image.fromarray(sea_elevation).resize(
            (1388 // resize_factor, 1510 // resize_factor)
        )
    )
    array = array[330 // resize_factor :, : 1200 // resize_factor] * mask
    array = array.astype(np.float16)
    # save it
    output_fname = "term=near_season=anno_variable=seaLevel_values=0$1"
    plt.imshow(array, cmap="jet")
    plt.colorbar()
    plt.title("SeaLevel elevation +0.5m")
    maps_output_dir = osp.join(output_dir, "maps")
    maps_output_path = osp.join(maps_output_dir, output_fname + ".png")
    os.makedirs(maps_output_dir, exist_ok=True)
    plt.savefig(maps_output_path)
    plt.close()
    # dump it
    array_output_dir = osp.join(output_dir, "arrays")
    array_output_path = osp.join(array_output_dir, output_fname + ".npy")
    os.makedirs(array_output_dir, exist_ok=True)
    np.save(array_output_path, array)
    # update meta
    col_array_output = "fpath_array"
    col_map_output = "fpath_map"
    metadata = {
        c.KEY_VARIABLE: "seaLevelElevation",
        c.KEY_TERM: "near",
        c.KEY_SEASON: "anno",
        c.KEY_VALUES: [0, 1],
    }
    df = pd.DataFrame(
        data=[
            {
                "file": input_path,
                col_array_output: array_output_path,
                col_map_output: maps_output_path,
                "resize_factor": resize_factor,
                **metadata,
            }
        ]
    ).set_index("file")
    if previous_df is not None:
        log.info("Updating previous metdata")
        previous_df = pd.concat([previous_df, df])
    else:
        log.info("No previous data: considering current map")
        previous_df = df

    previous_df.to_json(aux_metadata_path)
    log.info("Done :)")


if __name__ == "__main__":
    typer.run(go)
