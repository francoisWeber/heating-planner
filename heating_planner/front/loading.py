import streamlit as st

from heating_planner.back.data.drias import DriasDataset
from heating_planner.back.data.base import HazardDataset


@st.cache_data
def load_and_cache_dataset(input_file: str, class_: HazardDataset):
    return class_.load_from_path(input_file)
