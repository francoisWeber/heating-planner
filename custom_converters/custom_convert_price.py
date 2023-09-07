import os
import typer
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from os import path as osp
import pandas as pd
from heating_planner import constantes as c
from loguru import logger as log


def convert(input_file, resize_factor, values):
    if resize_factor != 5:
        raise NotImplementedError("Only for resize_factor=5")
    im_ini = Image.open(input_file)
    im_resized = im_ini.resize((1388, 1510), resample=Image.Resampling.BILINEAR)
    im = im_resized.resize(
        (im_resized.width // resize_factor, im_resized.height // resize_factor),
        resample=Image.Resampling.NEAREST,
    )
    img = np.asarray(im)[:, :, :3]
    mask_f5 = np.load("/Users/f.weber/tmp-fweber/heating/france_mask_factor5.npy")

    refcololors_locations = [
        [(1, 3), (0, 5)],
        [(7, 8), (0, 5)],
        [(11, 13), (0, 5)],
        [(16, 18), (0, 5)],
        [(21, 23), (0, 5)],
        [(26, 28), (0, 5)],
        [(31, 33), (0, 5)],
        [(36, 38), (0, 5)],
        [(41, 43), (0, 5)],
        [(46, 48), (0, 5)],
        [(51, 53), (0, 5)],
        [(56, 58), (0, 5)],
    ]
    img_colorbar = img[228:290, 0:3, :]
    maps = []
    for ref_col_loc in refcololors_locations:
        a, b = ref_col_loc
        ref_patch = img_colorbar[a[0] : a[1], b[0] : b[1]]
        v, h, c = ref_patch.shape
        ref_col = ref_patch.reshape((v * h, c)).astype(np.float64).mean(axis=0)
        ref_col = np.expand_dims(ref_col.reshape(1, -1), 0)
        map_where_refcol = np.linalg.norm(
            img.astype(np.float64) - ref_col, axis=-1, ord=1
        )
        maps.append(map_where_refcol)
    map_distances = np.stack(maps, axis=-1)
    maps_proximity = 1.0 / np.power(map_distances, 1.1)
    maps_weihts = maps_proximity / np.expand_dims(maps_proximity.sum(axis=-1), -1)
    values_ = np.array([[values]])
    maps_values_weighted = maps_weihts * values_
    map_distances = np.stack(maps, axis=-1)
    final_map = (
        maps_values_weighted.sum(axis=-1)[
            330 // resize_factor :, : 1200 // resize_factor
        ]
        * mask_f5
    )
    return final_map


def go(
    input_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/aux_maps/carte_prix_hd_resized.png"
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
    values = [8000, 4100, 2600, 2200, 1750, 1575, 1455, 1350, 1250, 1170, 1080, 500]
    array = convert(input_file=input_path, resize_factor=resize_factor, values=values)
    array = array.astype(np.float16)
    # save it
    values_str = "$".join([str(v) for v in values])
    output_fname = "term=ref_season=anno_variable=estate_values=" + values_str
    plt.imshow(array, cmap="jet")
    plt.colorbar()
    plt.title("Real estate prices €/m2")
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
        c.KEY_VARIABLE: "realEstate",
        c.KEY_TERM: "ref",
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
