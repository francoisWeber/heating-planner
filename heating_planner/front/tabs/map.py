import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset, MappableFactors
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

        factors_cols = st.columns([2, 2, 1])
        with st.container(border=True):
            with factors_cols[0]:
                scoring_strategy = st.radio("scoring method", FactorwiseScoring.get_available_options(), index=0, key="scoring_method")
            with factors_cols[1]:
                fusion_strategy = st.radio("fusion method", ScoringFusion.get_available_options(), index=0, key="fusion_method")
            with factors_cols[2]:
                scaling = st.toggle("scale scores", value=True, key="scaling")
                contrast = st.radio("contrast management", options=Contrast.get_available_options(), index=1)

        mappable_factors = MappableFactors.from_hazard_datasets(dataset_hist, dataset_proj)
        
        map_and_factors_cols = st.columns(2)
        with map_and_factors_cols[1]:
            with st.container(border=True):
                
                # display boolean keys
                factors_cols = st.columns(2)
                binary_factor_infos = {}
                for i, factor in enumerate(mappable_factors.binaries):
                    with factors_cols[i % 2]:
                        binary_factor_infos[factor.name] = st.toggle("With: " + factor.description)

                
                # display weightable factors
                factors_cols = st.columns(2)
                coefs = {}
                for i, factor in enumerate(mappable_factors.weightables):
                    with factors_cols[i % 2]:
                        coefs[factor.name] = st.slider(f"Coeff: {factor.description[:50]}", 0, 3, step=1, value=1)
                

        with map_and_factors_cols[0]:
            with st.spinner("Computing score ..."):
                scores = scoring_strategy(dataset_proj, dataset_hist, st.session_state.reference_ranges, scaled=scaling)
                score = fusion_strategy(scores, coefs)
                hazard_map = process_score(dataset_proj, score, contrast=contrast, binary_factor_infos=binary_factor_infos)
            with st.spinner("Creating map ..."):
                fig, ax = plt.subplots()
                hazard_map.plot("score", ax=ax, legend=True, cmap="RdYlGn")
                st.pyplot(fig)
