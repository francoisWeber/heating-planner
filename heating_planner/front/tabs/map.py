import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.scoring import FactorwiseScoring, ScoringFusion, Contrast, process_score
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
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        dataset_proj: HazardDataset = st.session_state.dataset_proj
        dataset_hist: HazardDataset = st.session_state.dataset_ref

        cols = st.columns([2, 2, 1])
        with st.container(border=True):
            with cols[0]:
                scoring_strategy = st.radio("scoring method", FactorwiseScoring.get_available_options(), index=0, key="scoring_method")
            with cols[1]:
                fusion_strategy = st.radio("fusion method", ScoringFusion.get_available_options(), index=0, key="fusion_method")
            with cols[2]:
                scaling = st.toggle("scale scores", value=True, key="scaling")
                contrast = st.radio("contrast management", options=Contrast.get_available_options(), index=1)

        with st.container(border=True):
            cols = st.columns([2, 1, 1])

            # display boolean keys
            binary_factor_infos = {}
            n_binary = 0
            for factor in dataset_proj.factors:
                if not factor.is_binary():
                    continue
                with cols[1 + n_binary % 2]:
                    binary_factor_infos[factor.name] = st.toggle("With: " + factor.description)
                n_binary += 1

            # display continuous keys
            coefs = {key: 1 for key in RESTRICT_TO_KEYS}
            for i, key in enumerate(coefs.keys()):
                factor = dataset_proj.get_factor(key)
                with cols[1 + i % 2]:
                    var_trend = VAR_TREND_2_EMOJI[factor.trend.value]
                    coefs[key] = st.slider(f"coeff {factor.description[:50]} ({var_trend})", 0, 3, step=1, value=1)

            with st.spinner("Computing score ..."):
                scores = scoring_strategy(dataset_proj, dataset_hist, st.session_state.reference_ranges, scaled=scaling)
                score = fusion_strategy(scores, coefs)
                hazard_map = process_score(dataset_proj, score, contrast=contrast, binary_factor_infos=binary_factor_infos)
            with cols[0]:
                with st.spinner("Creating map ..."):
                    fig, ax = plt.subplots()
                    hazard_map.plot("score", ax=ax, legend=True, cmap="RdYlGn")
                    st.pyplot(fig)
