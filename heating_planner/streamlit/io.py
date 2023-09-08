import streamlit as st
import pandas as pd
from heating_planner.utils import load_np_from_anywhere
from heating_planner.data import Datum
from collections import namedtuple


@st.cache_data
def load_data(
    metadata_path,
    metadata_aux_path,
    viable_ranges_path,
    mask_path,
    hypercubes_path,
    base_map_path,
) -> Datum:
    return Datum(
        metadata_path,
        metadata_aux_path,
        viable_ranges_path,
        mask_path,
        hypercubes_path,
        base_map_path,
    )
