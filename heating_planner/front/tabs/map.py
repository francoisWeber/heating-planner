import streamlit as st
from matplotlib import pyplot as plt

from heating_planner.back.data.base import HazardDataset
from heating_planner.back.geo.coder import geocoding
from heating_planner.back.geo.tools import get_topn_with_surroundings
from heating_planner.back.scoring.factorwise_scoring import \
    FactorsScoringStrategy
from heating_planner.back.scoring.fusion import ScoringFusion
from heating_planner.back.scoring.processor import Contrast, process_score
from heating_planner.back.scoring.scaler import ScoreScalingStrategy
from heating_planner.crs import GPS_CRS

PARAMS_INIT = {}
MARKER_SIZE = 2.0

FACTOR_WEIGHTS_NCOLS = 3


class OptionsDisplayer:
    def __init__(self):
        self.scores_col, self.scaling_col, self.display_col = st.columns(3)
        with self.scores_col:
            st.subheader("Factorwise scoring strategy")
        with self.scaling_col:
            st.subheader("Factor's scores fusion")
        with self.display_col:
            st.subheader("Display mode")

    def display(self):
        factors_scoring_strategy = self.display_factors_scoring_strategy()
        factors_scaling = self.display_factors_scaling()
        fusion_strategy = self.display_fusion_strategy()
        contrast = self.display_contrast()
        return factors_scoring_strategy, factors_scaling, fusion_strategy, contrast

    def display_factors_scoring_strategy(self):
        with self.scores_col:
            factors_scoring_strategy = st.radio("scoring method", FactorsScoringStrategy.get_options(), index=0, key="scoring_method")
        return factors_scoring_strategy

    def display_factors_scaling(self):
        with self.scaling_col:
            factors_scaling = st.radio("factors score scaling", ScoreScalingStrategy.get_options())
        return factors_scaling

    def display_fusion_strategy(self):
        return ScoringFusion.WEIGHTED_MEAN

    def display_contrast(self):
        with self.display_col:
            contrast = st.radio("contrast management", options=Contrast.get_options(), index=1)
        return contrast


@st.fragment
def display():
    if not st.session_state.loaded:
        st.warning("Please load the files first")
        st.button("retry")
    else:
        dataset_proj: HazardDataset = st.session_state.dataset_proj
        dataset_hist: HazardDataset = st.session_state.dataset_ref

        options_displayer = OptionsDisplayer()
        factor_scoring_strategy, factors_scaling, fusion_strategy, contrast = options_displayer.display()

        ds_scored_factors = factor_scoring_strategy(dataset_proj, dataset_hist, st.session_state.reference_ranges)
        ds_scaled_scored_factors = factors_scaling(ds_scored_factors)

        map_and_factors_cols = st.columns(2)
        with map_and_factors_cols[1]:
            with st.container(border=True):
                binary_factors = [factor for factor in ds_scored_factors.factors if factor.is_binary()]
                # display boolean keys
                factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)
                binary_factor_infos = {}
                for i, factor in enumerate(binary_factors):
                    with factors_cols[i % 2]:
                        binary_factor_infos[factor] = st.toggle("With: " + factor.description, value=True)

                st.markdown("Coefficients for numeric factors")
                weightable_factors = [factor for factor in ds_scored_factors.factors if not factor.is_binary()]
                with st.form("weight-form"):
                    # display weightable factors
                    coefs = {factor: 1 for factor in weightable_factors}
                    for i, factor in enumerate(weightable_factors):
                        if i % FACTOR_WEIGHTS_NCOLS == 0:
                            factors_cols = st.columns(FACTOR_WEIGHTS_NCOLS)

                        with factors_cols[i % FACTOR_WEIGHTS_NCOLS]:
                            coefs[factor] = st.slider(factor.description[:50], 0, 3, step=1, value=coefs[factor])
                    st.form_submit_button("update")

        with map_and_factors_cols[0]:
            score = fusion_strategy(ds_scaled_scored_factors, coefs)
            # Display only rows where score is not a float
            hazard_map = process_score(score, contrast=contrast, binary_factor_infos=binary_factor_infos)
            fig, ax = plt.subplots()
            hazard_map.plot("score", ax=ax, legend=True, cmap="RdYlGn", markersize=MARKER_SIZE)
            st.pyplot(fig)
            st.download_button("Download GeoDF", hazard_map.to_json(), "heating_map_scores.json")

    with st.container(border=True):
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Top 10 points")
            top_rows = get_topn_with_surroundings(hazard_map, n=10, score_colname="score").to_crs(GPS_CRS)
            geometries = top_rows.geometry.to_list()
            coords = [tuple(coord[0] for coord in geo.coords.xy[::-1]) for geo in geometries]
            locations = [geocoding.reverse(coord) for coord in coords]
            for i, loc in enumerate(locations):
                st.markdown(f"**Top {i + 1}**\n => {loc.address}")

        with cols[1]:
            st.subheader("Score of selected cities")
            cities_str = st.text_input("cities to check")
            cities = [city.strip() for city in cities_str.split(",")]
