import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset, MappableFactors
from heating_planner.back.scoring.factorwise_scoring import FactorwiseScoringStrategy
from heating_planner.back.scoring.fusion import ScoringFusion
from heating_planner.back.scoring.processor import process_score, Contrast
from heating_planner.back.scoring.scaler import GeoPandasScalingStrategy

PARAMS_INIT = {}
MARKER_SIZE = 2.0

FACTOR_WEIGHTS_NCOLS = 3


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
                with st.container(border=True):
                    st.subheader("Individual factor scoring")
                    subparams_cols = st.columns([2, 1])
                    with subparams_cols[0]:
                        factor_scoring_strategy = st.radio(
                            "scoring method", FactorwiseScoringStrategy.get_available_options(), index=0, key="scoring_method"
                        )
                    with subparams_cols[1]:
                        factors_scaling = st.radio("factors score scaling", GeoPandasScalingStrategy.get_available_options())
            with param_cols[1]:
                with st.container(border=True):
                    st.subheader("Factor's scores fusion")
                    fusion_strategy = st.radio("fusion method", ScoringFusion.get_available_options(), index=0, key="fusion_method")
            with param_cols[2]:
                with st.container(border=True):
                    st.subheader("Display mode")
                    subparams_cols = st.columns([1, 1])
                    with subparams_cols[0]:
                        score_scaling_strategy = st.radio(
                            "scale method", GeoPandasScalingStrategy.get_available_options(), index=0, key="scaling"
                        )
                    with subparams_cols[1]:
                        contrast = st.radio("contrast management", options=Contrast.get_available_options(), index=1)

        mappable_factors = MappableFactors.from_hazard_datasets(dataset_hist, dataset_proj)

        map_and_factors_cols = st.columns(2)
        with map_and_factors_cols[1]:
            with st.container(border=True):
                # display boolean keys
                factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)
                binary_factor_infos = {}
                for i, factor in enumerate(mappable_factors.binaries):
                    with factors_cols[i % 2]:
                        binary_factor_infos[factor] = st.toggle("With: " + factor.description, value=True)

                st.markdown("Coefficients for numeric factors")

                # display weightable factors
                coefs = {factor: 1 for factor in mappable_factors.weightables}
                for i, factor in enumerate(mappable_factors.weightables):
                    if i % FACTOR_WEIGHTS_NCOLS == 0:
                        factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)

                    with factors_cols[i % FACTOR_WEIGHTS_NCOLS]:
                        coefs[factor] = st.slider(factor.description[:50], 0, 3, step=1, value=coefs[factor])

                reset_coefs = st.button("Reset all coefficients", key="reset_coefs")
                if reset_coefs:
                    for k in coefs.keys():
                        coefs[k] = 0

        with map_and_factors_cols[0]:
            with st.spinner("Computing score ..."):
                scores = factor_scoring_strategy(dataset_proj, dataset_hist, st.session_state.reference_ranges, scaling=factors_scaling)
                score = fusion_strategy(scores, coefs)
                hazard_map = process_score(
                    dataset_proj, score, contrast=contrast, scaling_strategy=score_scaling_strategy, binary_factor_infos=binary_factor_infos
                )
            with st.spinner("Creating map ..."):
                fig, ax = plt.subplots()
                hazard_map.plot("score", ax=ax, legend=True, cmap="RdYlGn", markersize=MARKER_SIZE)
                st.pyplot(fig)
