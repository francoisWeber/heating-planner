import numpy as np
import pandas as pd
import typer
from loguru import logger as log


def go(
    metadata_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata2.json"
    ),
    metadata_aux_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/metadata_aux2.json"
    ),
    output_path: str = typer.Option(
        "/Users/f.weber/tmp-fweber/heating/processed/hypercube3.npz"
    ),
):
    # primary data
    df = pd.read_json(metadata_path)
    slices = [np.load(fpath) for fpath in df.fpath_array]
    log.info(f"Gathered {len(slices)} maps")
    hypercube = np.stack(slices, -1)
    # aux data
    df_aux = pd.read_json(metadata_aux_path)
    slices_aux = [np.load(fpath) for fpath in df_aux.fpath_array]
    log.info(f"Gathered {len(slices_aux)} aux maps")
    hypercube_aux = np.stack(slices_aux, -1)

    np.savez(output_path, map=hypercube, aux=hypercube_aux)


if __name__ == "__main__":
    typer.run(go)
