import streamlit as st

from heating_planner.back.data.drias import DriasDataset

import streamlit as st


@st.cache_data
def load_drias_dataset(input_file: str):
    drias_dataset = DriasDataset(input_file)
    drias_dataset.load()
    return drias_dataset
