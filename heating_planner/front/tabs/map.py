import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset, MappableFactors
from heating_planner.back.scoring import FactorwiseScoring, ScoringFusion, Contrast, process_score

PARAMS_INIT = {}
MARKER_SIZE = 2.0



def display():
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        dataset_proj: HazardDataset = st.session_state.dataset_proj
        dataset_hist: HazardDataset = st.session_state.dataset_ref

        param_cols = st.columns(3)
        with st.container(border=True):
            with param_cols[0]:
                scoring_strategy = st.radio("scoring method", FactorwiseScoring.get_available_options(), index=0, key="scoring_method")
            with param_cols[1]:
                fusion_strategy = st.radio("fusion method", ScoringFusion.get_available_options(), index=0, key="fusion_method")
            with param_cols[2]:
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
                        binary_factor_infos[factor] = st.toggle("With: " + factor.description, value=True)

                # display weightable factors
                factors_cols = st.columns(2)
                coefs = {factor.name: 1 for factor in mappable_factors.weightables}
                for i, factor in enumerate(mappable_factors.weightables):
                    with factors_cols[i % 2]:
                        coefs[factor.name] = st.slider(f"Coeff: {factor.description[:50]}", 0, 3, step=1, value=coefs[factor.name])
                
                reset_coefs = st.button("Reset all coefficients", key="reset_coefs")
                if reset_coefs:
                    for k in coefs.keys():
                        coefs[k] = 0

        with map_and_factors_cols[0]:
            with st.spinner("Computing score ..."):
                scores = scoring_strategy(dataset_proj, dataset_hist, st.session_state.reference_ranges, scaled=scaling)
                score = fusion_strategy(scores, coefs)
                hazard_map = process_score(dataset_proj, score, contrast=contrast, binary_factor_infos=binary_factor_infos)
            with st.spinner("Creating map ..."):
                fig, ax = plt.subplots()
                hazard_map.plot("score", ax=ax, legend=True, cmap="RdYlGn", markersize=MARKER_SIZE)
                st.pyplot(fig)
