import streamlit as st

from heating_planner.back.data.drias import DriasDataset


@st.cache_data
def load_drias_dataset(input_file: str):
    return DriasDataset.load_from_path(input_file)
