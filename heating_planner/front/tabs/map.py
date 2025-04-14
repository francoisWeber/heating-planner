from matplotlib import pyplot as plt
import streamlit as st
from heating_planner.back.map import DriasMap
from heating_planner.back.scoring import WeightedDriasScoring

PARAMS_INIT = {"a": 1}


def display():
    st.title("Map")
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        dataset = st.session_state.dataset_proj
        coefs = {
            "nortmm_seas_jja": -1,
            "nortmm_seas_djf": 1,
            "nortxm_seas_jja": -1,
            "nortx35d_yr": -1,
            "nortx30d_yr": -1,
            "nortr_yr": -1,
            "norrr_yr": 1,
            "norrr_seas_jja": 1,
            "norrr_seas_djf": 1,
            "norrrq99_yr": -1,
            "norrx1d_yr": -1,
            "norrrq99refd_yr": -1,
            "norifm40_yr": -1,
            "norswi04_yr": -1,
            "atx35d_yr": -1,
            "atx30d_yr": -1,
        }
        map = DriasMap(dataset)
        scoring = WeightedDriasScoring(dataset.df, coefs)
        map_ = map.prepare_score_map(scoring)
        fig, _ = plt.subplots()
        plt.imshow(map_)
        st.pyplot(fig)
