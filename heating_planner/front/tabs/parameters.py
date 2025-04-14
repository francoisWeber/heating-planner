import streamlit as st

from heating_planner.front.loading import load_drias_dataset

PARAMS_INIT = {
    "loaded": False,
    "dataset_ref": None,
    "dataset_proj": None,
}

drias_ref_path = "/Users/francois.weber/perso/tmp/heating2/data/drias/ref_indicesMAX_25041416092929830.txt"
drias_proj_path = "/Users/francois.weber/perso/tmp/heating2/data/drias/2050_indicesMAX_25041416092929830.txt"
clay_path = "/Users/francois.weber/perso/tmp/heating2/data/clay"
sea_elevation_path = "/Users/francois.weber/perso/tmp/heating2/data/overflow_1m"

def display():
    st.subheader("Parameters")
    input_file_ref = st.text_input("Reference file", drias_ref_path)
    input_file_proj = st.text_input(
        "Projection file", drias_proj_path
    )

    if input_file_proj and input_file_ref:
        st.session_state["dataset_proj"] = load_drias_dataset(input_file_proj)
        st.session_state["dataset_ref"] = load_drias_dataset(input_file_ref)
        st.session_state["loaded"] = True
        st.markdown(":heavy_check_mark: Files loaded")
