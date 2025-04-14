import streamlit as st

from heating_planner.front.loading import load_drias_dataset

PARAMS_INIT = {
    "loaded": False,
    "dataset_ref": None,
    "dataset_proj": None,
}


def display():
    st.subheader("Parameters")
    input_file_ref = st.text_input("Reference file", "/Users/francois.weber/perso/tmp/drias_ref_indicesMAX_25041011522929379.txt")
    input_file_proj = st.text_input(
        "Projection file", "/Users/francois.weber/perso/tmp/drias_data_20250_max_indicesMAX_25040910182929192.txt"
    )

    if input_file_proj and input_file_ref:
        st.session_state["dataset_proj"] = load_drias_dataset(input_file_proj)
        st.session_state["dataset_ref"] = load_drias_dataset(input_file_ref)
        st.session_state["loaded"] = True
        st.markdown(":heavy_check_mark: Files loaded")
