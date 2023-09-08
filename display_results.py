import streamlit as st
import click
from heating_planner.streamlit import display
from heating_planner.streamlit.io import load_data
from streamlit_image_coordinates import streamlit_image_coordinates


st.set_page_config(layout="wide")


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
    if "data" not in st.session_state:
        st.session_state["data"] = load_data(
            metadata_path=metadata_path,
            metadata_aux_path=metadata_aux_path,
            viable_ranges_path=viable_path,
            mask_path=mask_path,
            hypercubes_path=hypercube_path,
            base_map_path=basemap_path,
        )

    with st.expander("Select a reference location"):
        if "map_clicked_xy" not in st.session_state:
            st.session_statemap_clicked_xy = None
        streamlit_image_coordinates(basemap_path, key="map_clicked_xy")

    with st.container():
        display.display(
            metadata_path,
            metadata_aux_path,
            viable_path,
            mask_path,
            hypercube_path,
        )
        # from heating_planner.streamlit import display2

        # display2.render()


if __name__ == "__main__":
    run(standalone_mode=False, auto_envvar_prefix="HEAT")
