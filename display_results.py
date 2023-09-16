import streamlit as st
import click
from heating_planner.streamlit import display2
from heating_planner.utils import load_pil_from_anywhere
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import json
import yaml
import streamlit_authenticator as stauth

st.set_page_config(layout="wide")
st.title("Heating planner")

with open("./credentials.yaml") as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)
authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)


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
@click.option("--hypercube-path", type=str, default=None)
def run(
    basemap_path,
    metadata_path,
    metadata_aux_path,
    viable_path,
    hypercube_path,
):
    name, authentication_status, username = authenticator.login("Login", "main")
    if authentication_status:
        st.toast(f"Welcome {name} :smiley: ")
        with st.expander("Select a reference location"):
            if "map_clicked_xy" not in st.session_state:
                st.session_statemap_clicked_xy = None
            base_map = st_load_pil_from_anywhere(basemap_path)
            streamlit_image_coordinates(base_map, key="map_clicked_xy")

        with st.container():
            display2.render(
                metadata_path,
                metadata_aux_path,
                viable_path,
                hypercube_path,
            )

        with st.sidebar:
            st.download_button(
                "Download map",
                display2.save_fig(),
                file_name="score.png",
                mime="image/png",
            )
            st.download_button(
                "Download params",
                json.dumps(display2.extract_params()),
                mime="application/json",
                file_name="params.json",
            )
    elif authentication_status is False:
        st.error("Incorrect login :warning:")
    elif authentication_status is None:
        st.warning("Enter credentials :upside_down_face: ")


if __name__ == "__main__":
    run(standalone_mode=False, auto_envvar_prefix="HEAT")
