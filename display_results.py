import streamlit as st
import click
from heating_planner.streamlit import display2
from heating_planner.utils import load_pil_from_anywhere
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np

st.set_page_config(layout="wide")


@st.cache_data
def st_load_pil_from_anywhere(path, factor=None):
    im = load_pil_from_anywhere(path)
    if factor:
        im = im.resize(np.array(im.size) // factor)
    return im


@click.command()
@click.option("--basemap-path", type=str, default=None)
@click.option("--metadata-path", type=str, default=None)
@click.option("--metadata-aux-path", type=str, default=None)
@click.option("--viable-path", type=str, default=None)
@click.option("--mask-path", type=str, default=None)
@click.option("--hypercube-path", type=str, default=None)
def run(
    basemap_path,
    metadata_path,
    metadata_aux_path,
    viable_path,
    mask_path,
    hypercube_path,
):
    with st.expander("Select a reference location"):
        if "map_clicked_xy" not in st.session_state:
            st.session_statemap_clicked_xy = None
        base_map = st_load_pil_from_anywhere(basemap_path)
        streamlit_image_coordinates(base_map, key="map_clicked_xy")

    with st.container():
        # display.display(
        #     metadata_path,
        #     metadata_aux_path,
        #     viable_path,
        #     mask_path,
        #     hypercube_path,
        # )
        # from heating_planner.streamlit import display2

        display2.render(
            metadata_path,
            metadata_aux_path,
            viable_path,
            mask_path,
            hypercube_path,
        )


if __name__ == "__main__":
    run(standalone_mode=False, auto_envvar_prefix="HEAT")
