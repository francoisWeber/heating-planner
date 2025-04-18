import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.scoring import FactorwiseScoring, ScoringFusion
from heating_planner.front.cst import VAR_TREND_2_EMOJI

PARAMS_INIT = {}
RESTRICT_TO_KEYS = [
    "nortmm_seas_jja",
    "nortxm_seas_jja",
    "nortx35d_yr",
    "nortx30d_yr",
    "nortr_yr",
    "norrr_yr",
    "norrr_seas_jja",
    "norrr_seas_djf",
    "norrrq99_yr",
    "norrx1d_yr",
    "norrrq99refd_yr",
    "norifm40_yr",
    "norswi04_yr",
    # 'clay_hazard'
]


def display():
    st.title("Map")
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        hazard_dataset: HazardDataset = st.session_state.dataset_proj
        cols = st.columns(2)
        with cols[0]:
            st.selectbox("scoring method", FactorwiseScoring.get_available_scorings(), index=0, key="scoring_method")
        with cols[1]:
            st.selectbox("fusion method", ScoringFusion.get_available_fusions(), index=0, key="fusion_method")
        # scoring = 1
        # cols = st.columns([2, 1, 1])
        # coefs = {key: 1 for key in RESTRICT_TO_KEYS}
        # for i, key in enumerate(coefs.keys()):
        #     with cols[1 + i % 2]:
        #         var_definition = hazard_dataset.factors_definitions[key]
        #         var_trend = VAR_TREND_2_EMOJI[hazard_dataset.factors_types[key]]
        #         coefs[key] = st.slider(f"coeff {var_definition[:50]} ({var_trend})", 0, 3, step=1, value=1)
        # with cols[0]:
        #     fig, ax = plt.subplots()
        #     scoring.compute_rrf_score(coefs).plot("score", ax=ax, legend=True, cmap="RdYlGn")
        #     st.pyplot(fig)
