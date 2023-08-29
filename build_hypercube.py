import numpy as np
import pandas as pd
import typer
from loguru import logger as log


def go(
    metadata_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata.json"
    ),
    output_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/processed/hypercube.npy"
    ),
):
    df = pd.read_json(metadata_path)

    slices = [np.load(fpath) for fpath in df.fpath_array]
    slices = [np.expand_dims(slice_, axis=-1) for slice_ in slices]
    log.info(f"Gathered {len(slices)} maps")
    hypercube = np.concatenate(slices, axis=-1)

    np.save(output_path, hypercube)


if __name__ == "__main__":
    typer.run(go)
